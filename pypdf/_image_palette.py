"""Lossless storage of images with few colors, for :meth:`PdfWriter.compress<pypdf.PdfWriter.compress>`."""

import struct
from io import BytesIO
from itertools import islice
from typing import cast

from .constants import ImageAttributes as IA
from .constants import StreamAttributes as SA
from .errors import DECODING_ERRORS
from .filters import FlateDecode
from .generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    DictionaryObject,
    EncodedStreamObject,
    NameObject,
    NumberObject,
)

_CHUNK_SIZE = 65536
_SAMPLE_SIZE = 256 * 1024
_GROUP4_DICTIONARY_SIZE = 60
_MAXIMUM_COLORS = 256
_TIFF_FILL_ORDER = 266
_TIFF_ROWS_PER_STRIP = 278
_TIFF_STRIP_BYTE_COUNTS = 279
_TIFF_STRIP_OFFSETS = 273
_TIFF_TYPE_SHORT = 3
_RGB_COLOR_SPACES = ("/DeviceRGB", "/CalRGB")


def is_rgb(stream: EncodedStreamObject) -> bool:
    color_space = stream.get(IA.COLOR_SPACE)
    if isinstance(color_space, NameObject):
        return color_space in _RGB_COLOR_SPACES
    if isinstance(color_space, ArrayObject) and len(color_space) == 2:
        family = color_space[0].get_object()
        if family == "/ICCBased":
            profile = color_space[1].get_object()
            return isinstance(profile, DictionaryObject) and profile.get("/N") == 3
        return family in _RGB_COLOR_SPACES
    return False


def _find_colors(data: bytes) -> list[tuple[int, int, int]] | None:
    """Get the colors of RGB pixels, and stop early if there are too many."""
    pixels = zip(data[0::3], data[1::3], data[2::3])
    colors: set[tuple[int, int, int]] = set()
    while True:
        size_before = len(colors)
        chunk = list(islice(pixels, _CHUNK_SIZE))
        colors.update(chunk)
        if len(colors) > _MAXIMUM_COLORS:
            return None
        if len(chunk) < _CHUNK_SIZE:
            return sorted(colors) if colors or size_before else None


def _has_too_many_colors_at_the_start(stream: EncodedStreamObject, decoded: bytes, width: int) -> bool:
    """
    Check the first rows, which avoids decoding the PNG predictor of photos in pure Python.

    Args:
        stream: The image.
        decoded: The data of the image without Flate, but maybe with a predictor.
        width: The width of the image.

    """
    row_size = width * 3
    rows = _SAMPLE_SIZE // row_size + 1
    parameters = stream.get(SA.DECODE_PARMS)
    if parameters is None:
        return _find_colors(decoded[:rows * row_size]) is None
    if not isinstance(parameters, DictionaryObject) or cast(int, parameters.get("/Predictor", 1)) < 10:
        return False
    sample = FlateDecode._decode_png_prediction(decoded[:rows * (row_size + 1)], width, row_size + 1)
    return _find_colors(sample) is None


def _pack_rows(indices: bytes, width: int, height: int, bits: int) -> bytes:
    """Store color indices with fewer than 8 bits per pixel."""
    if bits == 8:
        return indices
    pixels_per_byte = 8 // bits
    padding = b"0" * (-width % pixels_per_byte)
    row_size = (width + pixels_per_byte - 1) // pixels_per_byte
    digits = indices.translate(b"0123456789abcdef" + bytes(240))
    rows = []
    for start in range(0, len(digits), width):
        number = int(digits[start:start + width] + padding, 2**bits)
        rows.append(number.to_bytes(row_size, "big"))
    return b"".join(rows)


def use_palette(stream: EncodedStreamObject, level: int) -> bool:
    """
    Store an RGB image with at most 256 colors as an indexed image if that makes it smaller.

    Args:
        stream: The image.
        level: The compression level.

    Returns:
        True if the stream has a new encoding.

    """
    width, height = stream.get(IA.WIDTH), stream.get(IA.HEIGHT)
    if (
        stream.get("/Subtype") != "/Image"
        or not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0)
        or stream.get(IA.BITS_PER_COMPONENT) != 8
        or IA.DECODE in stream
        or isinstance(stream.get(IA.MASK), ArrayObject)
        or stream.get(IA.INTERPOLATE)
        or not is_rgb(stream)
        or not stream._is_only_flate_encoded()
    ):
        return False
    decoded = stream._decode_flate_exactly()
    if decoded is None:
        return False
    try:
        if _has_too_many_colors_at_the_start(stream, decoded, width):
            return False
        data = stream.get_data()
    except DECODING_ERRORS:
        return False
    if len(data) != width * height * 3:
        return False
    colors = _find_colors(data)
    if colors is None:
        return False

    numbers = {color: index for index, color in enumerate(colors)}
    indices = bytes(map(numbers.__getitem__, zip(data[0::3], data[1::3], data[2::3])))
    bits = next(bits for bits in (1, 2, 4, 8) if len(colors) <= 2**bits)
    encoded = FlateDecode.encode(_pack_rows(indices, width, height, bits), level)
    if len(encoded) + 3 * len(colors) >= len(stream._data):
        return False

    stream._data = encoded
    stream.decoded_self = None
    stream.pop(SA.DECODE_PARMS, None)
    stream[NameObject(IA.BITS_PER_COMPONENT)] = NumberObject(bits)
    stream[NameObject(IA.COLOR_SPACE)] = ArrayObject(
        [
            NameObject("/Indexed"),
            stream.raw_get(IA.COLOR_SPACE),
            NumberObject(len(colors) - 1),
            ByteStringObject(bytes(value for color in colors for value in color)),
        ]
    )
    return True


def _read_single_strip(tiff_file: BytesIO) -> bytes | None:
    """
    Get the data of a TIFF file with one strip.

    This reads the tags directly, as Pillow refuses to open very large images.
    """
    tiff = tiff_file.getvalue()
    if tiff[:4] != b"II*\x00":
        return None
    directory = struct.unpack_from("<I", tiff, 4)[0]
    tags = {}
    for index in range(struct.unpack_from("<H", tiff, directory)[0]):
        tag, value_type, count = struct.unpack_from("<HHI", tiff, directory + 2 + 12 * index)
        value_format = "<H" if value_type == _TIFF_TYPE_SHORT else "<I"
        tags[tag] = (count, struct.unpack_from(value_format, tiff, directory + 10 + 12 * index)[0])
    offset_count, offset = tags[_TIFF_STRIP_OFFSETS]
    size = tags[_TIFF_STRIP_BYTE_COUNTS][1]
    if offset_count != 1 or tags.get(_TIFF_FILL_ORDER, (1, 1))[1] != 1:
        return None
    return tiff[offset:offset + size]


def _encode_group4(data: bytes, width: int, height: int) -> bytes | None:
    """Get CCITT group 4 data for 1-bit pixels, if Pillow can create it."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None
    output = BytesIO()
    try:
        Image.frombytes("1", (width, height), data).save(
            output, "TIFF", compression="group4", tiffinfo={_TIFF_ROWS_PER_STRIP: height}
        )
    except (OSError, ValueError, KeyError):
        return None
    return _read_single_strip(output)


def use_group4(stream: EncodedStreamObject) -> bool:
    """
    Store a Flate-encoded image with 1 bit per pixel with CCITT group 4 if that makes it smaller.

    Args:
        stream: The image.

    Returns:
        True if the stream has a new encoding.

    """
    width, height = stream.get(IA.WIDTH), stream.get(IA.HEIGHT)
    bits = 1 if stream.get(IA.IMAGE_MASK) else stream.get(IA.BITS_PER_COMPONENT)
    if (
        stream.get("/Subtype") != "/Image"
        or bits != 1
        or not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0)
        or SA.DECODE_PARMS in stream
        or not stream._is_only_flate_encoded()
    ):
        return False
    data = stream._decode_flate_exactly()
    if data is None or len(data) != height * ((width + 7) // 8):
        return False
    encoded = _encode_group4(data, width, height)
    if encoded is None or len(encoded) + _GROUP4_DICTIONARY_SIZE >= len(stream._data):
        return False
    stream._data = encoded
    stream.decoded_self = None
    stream[NameObject(SA.FILTER)] = NameObject("/CCITTFaxDecode")
    stream[NameObject(SA.DECODE_PARMS)] = DictionaryObject(
        {
            NameObject("/BlackIs1"): BooleanObject(True),
            NameObject("/Columns"): NumberObject(width),
            NameObject("/K"): NumberObject(-1),
            NameObject("/Rows"): NumberObject(height),
        }
    )
    return True
