"""Test the storage of images with few colors."""

import sys
import zlib
from io import BytesIO
from typing import cast
from unittest import mock

import pytest
from PIL import Image

from pypdf import PdfReader, PdfWriter
from pypdf._image_palette import _read_single_strip, is_rgb, use_group4, use_palette
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    EncodedStreamObject,
    NameObject,
    NumberObject,
    StreamObject,
)

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30)]


def _pixels(width: int, height: int, color_count: int) -> bytes:
    return b"".join(
        bytes(COLORS[(x // 7 + y // 5) % color_count]) for y in range(height) for x in range(width)
    )


def _stream(data: bytes, width: int, height: int) -> EncodedStreamObject:
    stream = EncodedStreamObject()
    stream[NameObject("/Subtype")] = NameObject("/Image")
    stream[NameObject("/Width")] = NumberObject(width)
    stream[NameObject("/Height")] = NumberObject(height)
    stream[NameObject("/BitsPerComponent")] = NumberObject(8)
    stream[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    stream[NameObject("/Filter")] = NameObject("/FlateDecode")
    stream._data = zlib.compress(data, 9)
    return stream


def _expand(stream: EncodedStreamObject) -> bytes:
    """Get the RGB pixels of an indexed image."""
    lookup = cast(bytes, stream["/ColorSpace"][3].original_bytes)
    bits = stream["/BitsPerComponent"]
    width, height = stream["/Width"], stream["/Height"]
    row_size = (width * bits + 7) // 8
    data = stream.get_data()
    output = bytearray()
    for y in range(height):
        row = int.from_bytes(data[y * row_size:(y + 1) * row_size], "big")
        total_bits = row_size * 8
        for x in range(width):
            index = (row >> (total_bits - (x + 1) * bits)) & ((1 << bits) - 1)
            output += lookup[3 * index:3 * index + 3]
    return bytes(output)


@pytest.mark.parametrize(
    ("color_count", "bits"),
    [(1, 1), (2, 1), (3, 2), (4, 2)],
)
def test_use_palette(color_count, bits):
    data = _pixels(203, 100, color_count)
    stream = _stream(data, 203, 100)
    size_before = len(stream._data)

    assert use_palette(stream, 9) is True

    assert len(stream._data) < size_before
    assert stream["/BitsPerComponent"] == bits
    assert stream["/ColorSpace"][0] == "/Indexed"
    assert stream["/ColorSpace"][1] == "/DeviceRGB"
    assert stream["/ColorSpace"][2] == color_count - 1
    assert _expand(stream) == data


@pytest.mark.parametrize(("color_count", "bits"), [(10, 4), (16, 4), (17, 8), (256, 8)])
def test_use_palette__more_colors(color_count, bits):
    palette = [bytes((index, 255 - index, index // 2)) for index in range(color_count)]
    data = b"".join(palette[(x * x + 31 * y) % color_count] for y in range(100) for x in range(300))
    stream = _stream(data, 300, 100)

    assert use_palette(stream, 9) is True

    assert stream["/BitsPerComponent"] == bits
    assert _expand(stream) == data


def test_use_palette__too_many_colors():
    data = b"".join(bytes((x % 256, y % 256, 0)) for y in range(300) for x in range(300))
    stream = _stream(data, 300, 300)
    encoded = stream._data
    assert use_palette(stream, 9) is False
    assert stream._data == encoded


def test_use_palette__too_many_colors_only_late_in_the_image():
    size = 400
    first_rows = _pixels(size, size - 20, 2)
    other_rows = b"".join(bytes((x % 256, y % 256, 7)) for y in range(20) for x in range(size))
    data = first_rows + other_rows
    stream = _stream(data, size, size)
    assert use_palette(stream, 9) is False
    assert stream["/ColorSpace"] == "/DeviceRGB"


def _png_rows(data: bytes, width: int) -> bytes:
    row_size = width * 3
    return b"".join(b"\x00" + data[start:start + row_size] for start in range(0, len(data), row_size))


def _predictor_parameters(width: int, predictor: int) -> DictionaryObject:
    return DictionaryObject(
        {
            NameObject("/Predictor"): NumberObject(predictor),
            NameObject("/Colors"): NumberObject(3),
            NameObject("/Columns"): NumberObject(width),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )


@pytest.mark.parametrize("predictor", [10, 15])
def test_use_palette__below_a_png_predictor(predictor):
    data = _pixels(200, 100, 4)
    stream = _stream(_png_rows(data, 200), 200, 100)
    stream[NameObject("/DecodeParms")] = _predictor_parameters(200, predictor)

    assert use_palette(stream, 9) is True

    assert "/DecodeParms" not in stream
    assert _expand(stream) == data


def test_use_palette__rejects_photos_below_a_png_predictor_without_decoding_them_all():
    data = b"".join(bytes((x % 256, y % 256, (x + y) % 256)) for y in range(2000) for x in range(400))
    stream = _stream(_png_rows(data, 400), 400, 2000)
    stream[NameObject("/DecodeParms")] = _predictor_parameters(400, 15)
    encoded = stream._data

    assert use_palette(stream, 9) is False

    assert stream._data == encoded
    assert stream.decoded_self is None


def test_use_palette__tiff_predictor():
    data = _pixels(200, 100, 4)
    stream = _stream(data, 200, 100)
    stream[NameObject("/DecodeParms")] = _predictor_parameters(200, 1)
    assert use_palette(stream, 9) is True
    assert _expand(stream) == data


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("/Decode", ArrayObject([NumberObject(0)] * 6)),
        ("/Mask", ArrayObject([NumberObject(0)] * 6)),
        ("/Interpolate", BooleanObject(True)),
        ("/ColorSpace", NameObject("/DeviceGray")),
        ("/ColorSpace", NameObject("/DeviceCMYK")),
        ("/BitsPerComponent", NumberObject(4)),
        ("/Subtype", NameObject("/Form")),
        ("/Width", NumberObject(0)),
        ("/Filter", NameObject("/DCTDecode")),
    ],
)
def test_use_palette__keeps_other_images(key, value):
    stream = _stream(_pixels(200, 100, 4), 200, 100)
    stream[NameObject(key)] = value
    assert use_palette(stream, 9) is False


def test_use_palette__keeps_broken_streams():
    stream = _stream(_pixels(200, 100, 4), 200, 100)
    stream._data = stream._data[:-20]
    data = stream._data
    assert use_palette(stream, 9) is False
    assert stream._data == data


def test_use_palette__keeps_data_of_unexpected_size():
    assert use_palette(_stream(b"too short", 200, 100), 9) is False


def test_use_palette__keeps_images_which_do_not_get_smaller():
    stream = _stream(bytes(3), 1, 1)
    assert use_palette(stream, 9) is False


def test_is_rgb():
    profile = StreamObject()
    profile[NameObject("/N")] = NumberObject(3)
    stream = EncodedStreamObject()

    stream[NameObject("/ColorSpace")] = ArrayObject([NameObject("/ICCBased"), profile])
    assert is_rgb(stream) is True
    stream[NameObject("/ColorSpace")] = ArrayObject([NameObject("/CalRGB"), DictionaryObject()])
    assert is_rgb(stream) is True
    profile[NameObject("/N")] = NumberObject(4)
    stream[NameObject("/ColorSpace")] = ArrayObject([NameObject("/ICCBased"), profile])
    assert is_rgb(stream) is False
    stream[NameObject("/ColorSpace")] = ArrayObject([NameObject("/Lab"), DictionaryObject()])
    assert is_rgb(stream) is False
    del stream["/ColorSpace"]
    assert is_rgb(stream) is False


def test_compress__palette_image_survives_a_round_trip():
    data = _pixels(200, 100, 4)
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Image"): writer._add_object(_stream(data, 200, 100))})}
    )

    writer.compress()
    output = BytesIO()
    writer.write(output)

    image = PdfReader(output).pages[0].images[0].image.convert("RGB")
    assert image.tobytes() == data


def test_compress__removes_thumbnails():
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    thumbnail = writer._add_object(_stream(_pixels(10, 10, 2), 10, 10))
    page[NameObject("/Thumb")] = thumbnail

    writer.compress()

    assert "/Thumb" not in writer.pages[0]
    assert writer._objects[thumbnail.idnum - 1] is None


def _scan(width: int, height: int) -> bytes:
    """Dark marks like those of a scanned page, which CCITT stores well."""
    rows = bytearray()
    state = 12345
    marks = []
    for _ in range(600):
        state = (state * 1103515245 + 12345) % 2**31
        marks.append((state % width, (state >> 12) % height, 2 + state % 5, 3 + (state >> 7) % 9))
    for y in range(height):
        row = (1 << width) - 1
        for left, top, mark_width, mark_height in marks:
            if top <= y < top + mark_height:
                row &= ~(((1 << mark_width) - 1) << max(0, width - left - mark_width))
        padding = -width % 8
        rows += ((row & ((1 << width) - 1)) << padding).to_bytes((width + padding) // 8, "big")
    return bytes(rows)


def _bilevel_stream(data: bytes, width: int, height: int) -> EncodedStreamObject:
    stream = _stream(data, width, height)
    stream[NameObject("/BitsPerComponent")] = NumberObject(1)
    stream[NameObject("/ColorSpace")] = NameObject("/DeviceGray")
    return stream


def test_use_group4():
    data = _scan(603, 500)
    stream = _bilevel_stream(data, 603, 500)
    size_before = len(stream._data)

    assert use_group4(stream) is True

    assert len(stream._data) < size_before
    assert stream["/Filter"] == "/CCITTFaxDecode"
    assert stream["/DecodeParms"] == {"/K": -1, "/Columns": 603, "/Rows": 500, "/BlackIs1": True}


def test_use_group4__image_mask():
    data = _scan(600, 500)
    stream = _bilevel_stream(data, 600, 500)
    del stream["/BitsPerComponent"]
    del stream["/ColorSpace"]
    stream[NameObject("/ImageMask")] = BooleanObject(True)

    assert use_group4(stream) is True

    assert stream["/ImageMask"] is not None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("/BitsPerComponent", NumberObject(8)),
        ("/DecodeParms", DictionaryObject()),
        ("/Filter", NameObject("/DCTDecode")),
        ("/Subtype", NameObject("/Form")),
        ("/Width", NumberObject(0)),
    ],
)
def test_use_group4__keeps_other_images(key, value):
    stream = _bilevel_stream(_scan(600, 500), 600, 500)
    stream[NameObject(key)] = value
    assert use_group4(stream) is False


def test_use_group4__keeps_broken_data():
    stream = _bilevel_stream(_scan(600, 500), 600, 500)
    stream._data = stream._data[:-20]
    assert use_group4(stream) is False
    assert use_group4(_bilevel_stream(b"too short", 600, 500)) is False


def test_use_group4__keeps_images_which_flate_stores_better():
    stream = _bilevel_stream(bytes(75 * 500), 600, 500)
    assert use_group4(stream) is False
    assert stream["/Filter"] == "/FlateDecode"


def test_use_group4__needs_pillow():
    stream = _bilevel_stream(_scan(600, 500), 600, 500)
    with mock.patch.dict(sys.modules, {"PIL": None}):
        assert use_group4(stream) is False


def test_use_group4__needs_a_single_strip():
    output = BytesIO()
    Image.frombytes("1", (600, 500), _scan(600, 500)).save(
        output, "TIFF", compression="group4", tiffinfo={278: 100}
    )
    assert _read_single_strip(output) is None


def test_use_group4__needs_the_usual_bit_order():
    output = BytesIO()
    Image.frombytes("1", (600, 500), _scan(600, 500)).save(
        output, "TIFF", compression="group4", tiffinfo={278: 500, 266: 2}
    )
    assert _read_single_strip(output) is None
