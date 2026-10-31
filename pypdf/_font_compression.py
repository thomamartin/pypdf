"""Removal of TrueType tables which PDF does not use, for :meth:`PdfWriter.compress<pypdf.PdfWriter.compress>`."""

import struct

# PDF positions every glyph itself and accesses glyphs by their index, so that the
# tables for layout and for device-specific metrics have no effect. A signature would
# not be valid any more. Tables with a possible effect on rendering stay, such as gasp.
_UNUSED_TABLES = frozenset(
    (b"BASE", b"DSIG", b"GDEF", b"GPOS", b"GSUB", b"JSTF", b"LTSH", b"VDMX", b"hdmx", b"kern")
)
_HEADER_SIZE = 12
_TABLE_ENTRY_SIZE = 16
_CHECKSUM_ADJUSTMENT_OFFSET = 8
_CHECKSUM_MAGIC_NUMBER = 0xB1B0AFBA


def _sum_of_words(data: bytes) -> int:
    padded = data + bytes(-len(data) % 4)
    words: tuple[int, ...] = struct.unpack(f">{len(padded) // 4}I", padded)
    return sum(words) & 0xFFFFFFFF


def remove_unused_font_tables(font: bytes) -> bytes | None:
    """
    Remove the tables of a TrueType font which have no effect in a PDF file.

    Args:
        font: The font file.

    Returns:
        The smaller font, or ``None`` if there is nothing to remove or the file
        is not a well-formed TrueType font.

    """
    if len(font) < _HEADER_SIZE or font[:4] not in (b"\x00\x01\x00\x00", b"true"):
        return None
    table_count = struct.unpack(">H", font[4:6])[0]
    if len(font) < _HEADER_SIZE + table_count * _TABLE_ENTRY_SIZE:
        return None
    tables = []
    for index in range(table_count):
        tag, checksum, offset, length = struct.unpack_from(">4sIII", font, _HEADER_SIZE + index * _TABLE_ENTRY_SIZE)
        if offset + length > len(font):
            return None
        tables.append((tag, checksum, font[offset:offset + length]))
    kept = [table for table in tables if table[0] not in _UNUSED_TABLES]
    if len(kept) == len(tables) or not any(table[0] == b"head" for table in kept):
        return None

    search_range = 2 ** (len(kept).bit_length() - 1) * _TABLE_ENTRY_SIZE
    entry_selector = len(kept).bit_length() - 1
    output = bytearray(font[:4])
    output += struct.pack(">HHHH", len(kept), search_range, entry_selector, len(kept) * 16 - search_range)
    offset = _HEADER_SIZE + len(kept) * _TABLE_ENTRY_SIZE
    head_offset = 0
    for tag, checksum, data in kept:
        output += struct.pack(">4sIII", tag, checksum, offset, len(data))
        if tag == b"head":
            head_offset = offset
        offset += len(data) + (-len(data) % 4)
    for _, _, data in kept:
        output += data + bytes(-len(data) % 4)

    if head_offset + _CHECKSUM_ADJUSTMENT_OFFSET + 4 > len(output):
        return None
    adjustment = head_offset + _CHECKSUM_ADJUSTMENT_OFFSET
    output[adjustment:adjustment + 4] = bytes(4)
    struct.pack_into(">I", output, adjustment, (_CHECKSUM_MAGIC_NUMBER - _sum_of_words(bytes(output))) & 0xFFFFFFFF)
    return bytes(output)
