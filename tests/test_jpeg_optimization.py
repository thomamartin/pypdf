"""Test the lossless optimization of JPEG images."""

import random
import subprocess
import zlib
from io import BytesIO
from unittest import mock

import pytest
from PIL import Image

from pypdf import PdfWriter, _jpeg_optimization
from pypdf._jpeg_optimization import _read_frames, optimize_jpegs
from pypdf.generic import ArrayObject, EncodedStreamObject, NameObject

from . import RESOURCE_ROOT

needs_package = pytest.mark.skipif(not _jpeg_optimization.is_available(), reason="Needs the optimizer")


def _jpeg(**parameters: object) -> bytes:
    """A JPEG with default Huffman tables, which leaves room for optimizing them."""
    output = BytesIO()
    Image.effect_noise((300, 200), 25).convert("RGB").save(
        output, "JPEG", quality=90, optimize=False, progressive=False, **parameters
    )
    return output.getvalue()


def _pixels(jpeg: bytes) -> bytes:
    return Image.open(BytesIO(jpeg)).convert("RGB").tobytes()


def _damaged(jpeg: bytes, generator: random.Random) -> bytes:
    data = bytearray(jpeg)
    start = jpeg.find(b"\xff\xda") + 20
    for _ in range(40):
        data[generator.randrange(start, len(data) - 2)] = generator.randrange(256)
    return bytes(data)


@needs_package
def test_optimize_jpegs():
    jpeg = _jpeg()

    optimized = optimize_jpegs([jpeg, jpeg])

    assert len(optimized) == 2
    assert len(optimized[0]) < len(jpeg)
    assert _pixels(optimized[0]) == _pixels(jpeg)


@needs_package
def test_optimize_jpegs__keeps_metadata():
    profile = b"Pretend this is a color profile." * 10
    jpeg = _jpeg(icc_profile=profile)
    assert Image.open(BytesIO(optimize_jpegs([jpeg])[0])).info["icc_profile"] == profile


@needs_package
def test_optimize_jpegs__reports_invalid_data():
    jpeg = _jpeg()
    optimized = optimize_jpegs([b"This is not a JPEG.", jpeg])
    assert optimized[0] is None
    assert optimized[1] is not None


@needs_package
def test_optimize_jpegs__survives_files_which_end_the_process():
    generator = random.Random(1)  # noqa: S311
    jpeg = _jpeg()
    damaged = [_damaged(jpeg, generator) for _ in range(300)]

    optimized = optimize_jpegs([*damaged, jpeg])

    assert len(optimized) == 301


def test_optimize_jpegs__continues_after_a_crash():
    worker = """
import os, struct, sys
while True:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        break
    data = sys.stdin.buffer.read(struct.unpack(">I", header)[0])
    if data == b"crash":
        os._exit(1)
    sys.stdout.buffer.write(struct.pack(">I", len(data) + 1) + data + b"!")
    sys.stdout.buffer.flush()
"""
    with mock.patch("pypdf._jpeg_optimization._WORKER", worker):
        assert optimize_jpegs([b"a", b"crash", b"b", b"crash", b"c"]) == [b"a!", None, b"b!", None, b"c!"]


def test_optimize_jpegs__limits_the_number_of_workers():
    worker = "import os; os._exit(1)"
    with mock.patch("pypdf._jpeg_optimization._WORKER", worker), mock.patch(
        "pypdf._jpeg_optimization._MAXIMUM_WORKERS", 3
    ):
        assert optimize_jpegs([b"a"] * 10) == [None] * 10


def test_optimize_jpegs__nothing_to_do():
    assert optimize_jpegs([]) == []


def test_optimize_jpegs__no_interpreter():
    with mock.patch("pypdf._jpeg_optimization.sys.executable", "/does/not/exist"):
        assert optimize_jpegs([b"a", b"b"]) == [None, None]


def test_optimize_jpegs__timeout():
    error = subprocess.TimeoutExpired("python", 1, output=b"\x00\x00\x00\x01x")
    with mock.patch("pypdf._jpeg_optimization.subprocess.run", side_effect=error):
        assert optimize_jpegs([b"a", b"b", b"c"]) == [b"x", None, None]


def test_read_frames__ignores_incomplete_data():
    assert _read_frames(b"\x00\x00\x00\x02ab\x00\x00\x00\x05abc") == [b"ab"]


def _stream(data: bytes, filters: object) -> EncodedStreamObject:
    stream = EncodedStreamObject()
    stream[NameObject("/Filter")] = filters
    stream._data = data
    return stream


def _writer_with(stream: EncodedStreamObject) -> PdfWriter:
    writer = PdfWriter(clone_from=RESOURCE_ROOT / "crazyones.pdf")
    writer.root_object[NameObject("/TestStream")] = writer._add_object(stream)
    return writer


@needs_package
def test_compress__optimizes_jpegs():
    jpeg = _jpeg()
    stream = _stream(jpeg, NameObject("/DCTDecode"))

    _writer_with(stream).compress()

    assert len(stream._data) < len(jpeg)
    assert _pixels(stream._data) == _pixels(jpeg)


@needs_package
def test_compress__optimizes_jpegs_with_a_filter_array():
    jpeg = _jpeg()
    stream = _stream(jpeg, ArrayObject([NameObject("/DCTDecode")]))

    _writer_with(stream).compress()

    assert len(stream._data) < len(jpeg)
    assert _pixels(stream._data) == _pixels(jpeg)


@needs_package
def test_compress__optimizes_jpegs_below_flate():
    jpeg = _jpeg()
    data = zlib.compress(jpeg, 9)
    stream = _stream(data, ArrayObject([NameObject("/FlateDecode"), NameObject("/DCTDecode")]))

    _writer_with(stream).compress()

    assert len(stream._data) < len(data)
    assert stream["/Filter"] == ["/FlateDecode", "/DCTDecode"]
    assert _pixels(zlib.decompress(stream._data)) == _pixels(jpeg)


def test_compress__keeps_jpegs_without_the_package():
    jpeg = _jpeg()
    stream = _stream(jpeg, NameObject("/DCTDecode"))
    with mock.patch("pypdf._writer.jpeg_optimization_is_available", return_value=False):
        _writer_with(stream).compress()
    assert stream._data == jpeg


def test_compress__keeps_jpegs_which_do_not_get_smaller():
    jpeg = _jpeg()
    stream = _stream(jpeg, NameObject("/DCTDecode"))
    with mock.patch("pypdf._writer.jpeg_optimization_is_available", return_value=True), mock.patch(
        "pypdf._writer.optimize_jpegs", return_value=[jpeg + b"more"]
    ):
        _writer_with(stream).compress()
    assert stream._data == jpeg


def test_compress__keeps_jpegs_which_could_not_be_optimized():
    jpeg = _jpeg()
    stream = _stream(jpeg, NameObject("/DCTDecode"))
    with mock.patch("pypdf._writer.jpeg_optimization_is_available", return_value=True), mock.patch(
        "pypdf._writer.optimize_jpegs", return_value=[None]
    ):
        _writer_with(stream).compress()
    assert stream._data == jpeg


def test_compress__ignores_other_filters_for_jpegs():
    optimizer = mock.Mock(return_value=[])
    writer = _writer_with(_stream(b"Data", NameObject("/JPXDecode")))
    writer.root_object[NameObject("/TestChain")] = writer._add_object(
        _stream(b"Data", ArrayObject([NameObject("/ASCII85Decode"), NameObject("/DCTDecode")]))
    )
    writer.root_object[NameObject("/TestBroken")] = writer._add_object(
        _stream(b"Data", ArrayObject([NameObject("/FlateDecode"), NameObject("/DCTDecode")]))
    )
    with mock.patch("pypdf._writer.jpeg_optimization_is_available", return_value=True), mock.patch(
        "pypdf._writer.optimize_jpegs", optimizer
    ):
        writer.compress()
    assert optimizer.call_args.args[0] == []
