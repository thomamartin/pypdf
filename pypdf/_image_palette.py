"""Lossless storage of RGB images with few colors, for :meth:`PdfWriter.compress<pypdf.PdfWriter.compress>`."""

from itertools import islice
from typing import cast

from .constants import ImageAttributes as IA
from .constants import StreamAttributes as SA
from .filters import FlateDecode
from .generic import ArrayObject, ByteStringObject, DictionaryObject, EncodedStreamObject, NameObject, NumberObject

_CHUNK_SIZE = 65536
_SAMPLE_SIZE = 256 * 1024
_MAXIMUM_COLORS = 256
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
    if decoded is None or _has_too_many_colors_at_the_start(stream, decoded, width):
        return False
    data = stream.get_data()
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
