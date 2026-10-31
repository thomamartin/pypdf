"""Lossy compression of images for :meth:`PdfWriter.compress<pypdf.PdfWriter.compress>`."""

from io import BytesIO
from typing import cast

try:
    from PIL import Image
except ImportError as exception:
    raise ImportError(
        "Pillow is required for lossy image compression. Install it with: pip install pypdf[image]"
    ) from exception

from .constants import FilterTypeAbbreviations as FTA
from .constants import FilterTypes as FT
from .constants import ImageAttributes as IA
from .constants import StreamAttributes as SA
from .errors import DECODING_ERRORS
from .filters import FlateDecode
from .generic import ArrayObject, DictionaryObject, EncodedStreamObject, NameObject, NumberObject, StreamObject

_JPEG_QUALITY_WITHOUT_REQUEST = 85
_MAXIMUM_NEW_SIZE = 0.9
# Better entropy coding alone can save 10 % on a JPEG, without the loss of encoding it again.
_MAXIMUM_NEW_SIZE_OF_LOSSY_IMAGES = 0.8
_MAXIMUM_COLORS_OF_GRAPHICS = {"L": 32, "RGB": 256}
_PLAIN_COLOR_SPACES = {"/DeviceGray": 1, "/CalGray": 1, "/DeviceRGB": 3, "/CalRGB": 3}


def _get_components(stream: StreamObject) -> int | None:
    """Get the number of color components if they can be treated as gray or RGB."""
    color_space = stream.get(IA.COLOR_SPACE)
    if isinstance(color_space, NameObject):
        return _PLAIN_COLOR_SPACES.get(color_space)
    if isinstance(color_space, ArrayObject) and len(color_space) == 2:
        family = color_space[0].get_object()
        if family in _PLAIN_COLOR_SPACES:
            return _PLAIN_COLOR_SPACES[family]
        if family == "/ICCBased":
            profile = color_space[1].get_object()
            components = profile.get("/N") if isinstance(profile, DictionaryObject) else None
            return components if components in (1, 3) else None
    return None


def _is_kept_as_it_is(stream: StreamObject, width: object, height: object) -> bool:
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        return True
    return bool(stream.get(IA.IMAGE_MASK)) or IA.DECODE in stream or isinstance(stream.get(IA.MASK), ArrayObject)


def _get_encoded_image(stream: StreamObject, names: list[str]) -> bytes | None:
    """Get the JPEG or JPEG 2000 file which a stream stores."""
    if SA.DECODE_PARMS in stream or len(names) > 2:
        return None
    if len(names) == 1:
        return stream._data
    if names[0] in (FT.FLATE_DECODE, FTA.FL) and isinstance(stream, EncodedStreamObject):
        return stream._decode_flate_exactly()
    return None


def _open_image(stream: StreamObject) -> tuple[Image.Image, bool] | None:
    """
    Get the pixels of an image which can be stored in another way without side effects.

    Returns:
        The image and whether it uses a lossy encoding at the moment, or ``None``
        for images which have to stay as they are.

    """
    width, height = stream.get(IA.WIDTH), stream.get(IA.HEIGHT)
    components = _get_components(stream)
    if _is_kept_as_it_is(stream, width, height) or components is None:
        return None

    filters = stream.get(SA.FILTER)
    names = list(filters) if isinstance(filters, ArrayObject) else ([] if filters is None else [filters])
    mode = "L" if components == 1 else "RGB"
    if names and names[-1] in (FT.DCT_DECODE, FTA.DCT, FT.JPX_DECODE):
        encoded = _get_encoded_image(stream, names)
        if encoded is None:
            return None
        try:
            image = Image.open(BytesIO(encoded))
            image.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            return None
        return (image, True) if image.mode == mode and image.size == (width, height) else None

    if stream.get(IA.BITS_PER_COMPONENT) != 8:
        return None
    size = (cast(int, width), cast(int, height))
    try:
        data = stream.get_data()
    except DECODING_ERRORS:
        return None
    if len(data) != size[0] * size[1] * components:
        return None
    return Image.frombytes(mode, size, data), False


def _is_a_graphic(image: Image.Image) -> bool:
    """Charts and screenshots have few colors, and JPEG would damage their sharp edges."""
    return image.getcolors(_MAXIMUM_COLORS_OF_GRAPHICS[image.mode]) is not None


def compress_image(
    stream: StreamObject, level: int, quality: int | None, scale: float
) -> EncodedStreamObject | None:
    """
    Store an image at a lower quality or resolution.

    Args:
        stream: The image.
        level: The compression level for images which stay lossless.
        quality: The JPEG quality. ``None`` keeps a lossless image lossless.
        scale: The factor for the width and the height, at most 1.

    Returns:
        The new image if it is clearly smaller.

    """
    opened = _open_image(stream)
    if opened is None:
        return None
    image, is_lossy = opened
    if quality is not None and not is_lossy and _is_a_graphic(image):
        quality = None
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    is_resized = (width, height) != image.size
    if is_resized:
        image = image.resize((width, height), Image.LANCZOS)  # type: ignore[attr-defined]
    elif quality is None:
        return None

    if quality is None and not is_lossy:
        data = FlateDecode.encode(image.tobytes(), level)
        filter_name = FT.FLATE_DECODE
    else:
        output = BytesIO()
        image.save(
            output,
            "JPEG",
            quality=_JPEG_QUALITY_WITHOUT_REQUEST if quality is None else quality,
            optimize=True,
            progressive=True,
        )
        data = output.getvalue()
        filter_name = FT.DCT_DECODE
    maximum_size = _MAXIMUM_NEW_SIZE_OF_LOSSY_IMAGES if is_lossy and not is_resized else _MAXIMUM_NEW_SIZE
    if len(data) >= maximum_size * len(stream._data):
        return None

    compressed = EncodedStreamObject()
    compressed.update(stream)
    compressed.pop(SA.DECODE_PARMS, None)
    compressed[NameObject(SA.FILTER)] = NameObject(filter_name)
    compressed[NameObject(IA.WIDTH)] = NumberObject(width)
    compressed[NameObject(IA.HEIGHT)] = NumberObject(height)
    compressed[NameObject(IA.BITS_PER_COMPONENT)] = NumberObject(8)
    compressed._data = data
    compressed.indirect_reference = getattr(stream, "indirect_reference", None)
    return compressed
