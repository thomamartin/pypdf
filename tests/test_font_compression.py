"""Test the removal of unused font tables."""

import struct
import zlib
from io import BytesIO

import pytest
from fontTools.ttLib import TTFont

from pypdf import PdfWriter
from pypdf._font_compression import remove_unused_font_tables
from pypdf.generic import DictionaryObject, EncodedStreamObject, NameObject, NumberObject, StreamObject

from . import RESOURCE_ROOT

HEAD = struct.pack(">IIII", 0x00010000, 1, 0, 0x5F0F3CF5) + bytes(38)


def _font(tables: dict[bytes, bytes], version: bytes = b"\x00\x01\x00\x00") -> bytes:
    header = version + struct.pack(">HHHH", len(tables), 0, 0, 0)
    offset = 12 + 16 * len(tables)
    directory = b""
    data = b""
    for tag, table in sorted(tables.items()):
        directory += struct.pack(">4sIII", tag, 0, offset, len(table))
        data += table + bytes(-len(table) % 4)
        offset += len(table) + (-len(table) % 4)
    return header + directory + data


def _tables(font: bytes) -> dict[bytes, bytes]:
    tables = {}
    for index in range(struct.unpack(">H", font[4:6])[0]):
        tag, _, offset, length = struct.unpack_from(">4sIII", font, 12 + 16 * index)
        tables[tag] = font[offset:offset + length]
    return tables


def _sum_of_words(data: bytes) -> int:
    return sum(struct.unpack(f">{len(data) // 4}I", data)) & 0xFFFFFFFF


FONT = _font(
    {
        b"GPOS": b"layout" * 100,
        b"GSUB": b"substitution" * 100,
        b"OS/2": b"metrics",
        b"cmap": b"character map",
        b"glyf": b"glyphs" * 50,
        b"hdmx": b"device metrics" * 100,
        b"head": HEAD,
        b"name": b"names",
    }
)


def test_remove_unused_font_tables():
    reduced = remove_unused_font_tables(FONT)

    tables = _tables(reduced)
    assert sorted(tables) == [b"OS/2", b"cmap", b"glyf", b"head", b"name"]
    original_tables = _tables(FONT)
    for tag in (b"OS/2", b"cmap", b"glyf", b"name"):
        assert tables[tag] == original_tables[tag]
    assert len(reduced) < len(FONT)


def test_remove_unused_font_tables__fixes_the_header_and_the_checksum():
    reduced = remove_unused_font_tables(FONT)

    assert struct.unpack(">HHHH", reduced[4:12]) == (5, 64, 2, 16)
    assert _sum_of_words(reduced) == 0xB1B0AFBA
    head = _tables(reduced)[b"head"]
    assert head[:8] == HEAD[:8]
    assert head[12:] == HEAD[12:]


def test_remove_unused_font_tables__readable_by_font_tools():
    font = TTFont(BytesIO(remove_unused_font_tables(FONT)))
    assert sorted(font.keys()) == ["GlyphOrder", "OS/2", "cmap", "glyf", "head", "name"]


def test_remove_unused_font_tables__true_type_marker():
    assert remove_unused_font_tables(_font({b"head": HEAD, b"kern": b"kerning"}, b"true"))[:4] == b"true"


@pytest.mark.parametrize(
    "font",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\x00\x01\x00\x00", id="short"),
        pytest.param(b"OTTO" + FONT[4:], id="OpenType with CFF"),
        pytest.param(b"ttcf" + FONT[4:], id="collection"),
        pytest.param(FONT[:40], id="truncated directory"),
        pytest.param(FONT[:300], id="truncated table"),
        pytest.param(_font({b"glyf": b"glyphs", b"head": HEAD}), id="nothing to remove"),
        pytest.param(_font({b"glyf": b"glyphs", b"kern": b"kerning"}), id="no head table"),
        pytest.param(_font({b"head": b"short", b"kern": b"kerning"}), id="short head table"),
    ],
)
def test_remove_unused_font_tables__keeps_other_data(font):
    assert remove_unused_font_tables(font) is None


def _font_stream(data: bytes) -> EncodedStreamObject:
    stream = EncodedStreamObject()
    stream[NameObject("/Filter")] = NameObject("/FlateDecode")
    stream[NameObject("/Length1")] = NumberObject(len(FONT))
    stream._data = data
    return stream


def _writer_with_font(stream: EncodedStreamObject) -> tuple[PdfWriter, EncodedStreamObject]:
    writer = PdfWriter(clone_from=RESOURCE_ROOT / "crazyones.pdf")
    reference = writer._add_object(stream)
    descriptor = writer._add_object(DictionaryObject({NameObject("/FontFile2"): reference}))
    writer.root_object[NameObject("/TestDescriptor")] = descriptor
    return writer, reference.get_object()


def test_compress__removes_unused_font_tables():
    writer, stream = _writer_with_font(_font_stream(zlib.compress(FONT)))

    writer.compress()

    font = zlib.decompress(stream._data)
    assert stream["/Length1"] == len(font)
    assert b"hdmx" not in _tables(font)
    assert b"glyf" in _tables(font)


def test_compress__keeps_broken_font_files():
    data = zlib.compress(FONT)[:-20]
    writer, stream = _writer_with_font(_font_stream(data))

    writer.compress()

    assert stream._data == data
    assert stream["/Length1"] == len(FONT)


def test_compress__ignores_font_files_which_are_not_flate_streams():
    plain = StreamObject()
    plain.set_data(FONT)
    writer, _ = _writer_with_font(_font_stream(b"not used"))
    writer.root_object["/TestDescriptor"].get_object()[NameObject("/FontFile2")] = writer._add_object(plain)
    writer.root_object[NameObject("/TestDirect")] = writer._add_object(
        DictionaryObject({NameObject("/FontFile2"): NameObject("/NotAStream")})
    )

    writer.compress()

    assert writer.root_object["/TestDescriptor"]["/FontFile2"].get_data() == FONT
