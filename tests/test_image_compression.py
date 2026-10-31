"""Test the lossy image compression of PdfWriter.compress."""

import sys
import zlib
from io import BytesIO
from unittest import mock

import pytest
from PIL import Image

from pypdf import PdfReader, PdfWriter
from pypdf._image_compression import compress_image
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DecodedStreamObject,
    DictionaryObject,
    EncodedStreamObject,
    IndirectObject,
    NameObject,
    NumberObject,
    StreamObject,
)

from . import RESOURCE_ROOT


def _photo(width: int = 400, height: int = 300) -> Image.Image:
    """A smooth image with some noise, which JPEG compresses far better than Flate."""
    image = Image.effect_noise((width, height), 30).convert("L")
    gradient = Image.linear_gradient("L").resize((width, height))
    return Image.merge("RGB", (image, gradient, Image.blend(image, gradient, 0.5)))


def _stream(image: Image.Image, color_space: object = NameObject("/DeviceRGB")) -> EncodedStreamObject:
    stream = EncodedStreamObject()
    stream[NameObject("/Type")] = NameObject("/XObject")
    stream[NameObject("/Subtype")] = NameObject("/Image")
    stream[NameObject("/Width")] = NumberObject(image.width)
    stream[NameObject("/Height")] = NumberObject(image.height)
    stream[NameObject("/BitsPerComponent")] = NumberObject(8)
    stream[NameObject("/ColorSpace")] = color_space
    stream[NameObject("/Filter")] = NameObject("/FlateDecode")
    stream._data = zlib.compress(image.tobytes(), 9)
    return stream


def _encoded(image: Image.Image, image_format: str, **parameters: object) -> bytes:
    output = BytesIO()
    image.save(output, image_format, **parameters)
    return output.getvalue()


def _writer_with_image(stream: StreamObject, page_size: int = 72) -> tuple[PdfWriter, IndirectObject]:
    writer = PdfWriter()
    page = writer.add_blank_page(width=page_size, height=page_size)
    reference = writer._add_object(stream)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Image"): reference})}
    )
    return writer, reference


def test_compress_image__to_jpeg():
    image = _photo()
    stream = _stream(image)
    stream[NameObject("/DecodeParms")] = DictionaryObject()

    compressed = compress_image(stream, 9, 75, 1.0)

    assert compressed["/Filter"] == "/DCTDecode"
    assert "/DecodeParms" not in compressed
    assert compressed["/ColorSpace"] == "/DeviceRGB"
    assert (compressed["/Width"], compressed["/Height"]) == image.size
    assert len(compressed._data) < len(stream._data)
    assert Image.open(BytesIO(compressed._data)).size == image.size


def _chart(colors: int, mode: str = "RGB") -> Image.Image:
    """A large image with few colors and sharp edges."""
    image = Image.new(mode, (400, 300), 255 if mode == "L" else (255, 255, 255))
    for index in range(colors - 1):
        box = (index * 3 % 390, index * 5 % 290, index * 3 % 390 + 9, index * 5 % 290 + 9)
        value = index * 7 % 255
        image.paste(value if mode == "L" else (value, 255 - value, index), box)
    return image


@pytest.mark.parametrize(("colors", "mode"), [(5, "RGB"), (200, "RGB"), (5, "L")])
def test_compress_image__keeps_graphics_lossless(colors, mode):
    image = _chart(colors, mode)
    color_space = NameObject("/DeviceRGB" if mode == "RGB" else "/DeviceGray")
    assert compress_image(_stream(image, color_space), 9, 75, 1.0) is None


def test_compress_image__reduces_graphics_without_lossy_coding():
    speckles = Image.effect_noise((400, 300), 100).point(lambda value: 255 if value > 128 else 0)
    compressed = compress_image(_stream(speckles.convert("RGB")), 9, 75, 0.25)
    assert compressed["/Filter"] == "/FlateDecode"
    assert compressed["/Width"] == 100


def test_compress_image__keeps_the_reference():
    stream = _stream(_photo())
    stream.indirect_reference = IndirectObject(5, 0, mock.Mock())
    assert compress_image(stream, 9, 75, 1.0).indirect_reference == stream.indirect_reference


def test_compress_image__unfiltered():
    image = _photo()
    stream = DecodedStreamObject()
    stream.update(_stream(image))
    del stream["/Filter"]
    stream.set_data(image.tobytes())

    assert compress_image(stream, 9, 75, 1.0)["/Filter"] == "/DCTDecode"


@pytest.mark.parametrize(
    "color_space",
    [
        NameObject("/DeviceGray"),
        NameObject("/CalGray"),
        ArrayObject([NameObject("/CalGray"), DictionaryObject()]),
    ],
)
def test_compress_image__gray(color_space):
    image = _photo().convert("L")
    compressed = compress_image(_stream(image, color_space), 9, 75, 1.0)
    assert compressed["/Filter"] == "/DCTDecode"
    assert Image.open(BytesIO(compressed._data)).mode == "L"


@pytest.mark.parametrize(("components", "mode"), [(1, "L"), (3, "RGB")])
def test_compress_image__icc_based(components, mode):
    profile = StreamObject()
    profile[NameObject("/N")] = NumberObject(components)
    image = _photo().convert(mode)
    color_space = ArrayObject([NameObject("/ICCBased"), profile])
    assert compress_image(_stream(image, color_space), 9, 75, 1.0)["/ColorSpace"] == color_space


def test_compress_image__from_jpeg():
    image = _photo()
    stream = _stream(image)
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream._data = _encoded(image, "JPEG", quality=95)

    compressed = compress_image(stream, 9, 50, 1.0)

    assert len(compressed._data) < 0.9 * len(stream._data)


def test_compress_image__from_jpeg_below_flate():
    image = _photo()
    stream = _stream(image)
    stream[NameObject("/Filter")] = ArrayObject([NameObject("/FlateDecode"), NameObject("/DCTDecode")])
    stream._data = zlib.compress(_encoded(image, "JPEG", quality=95), 9)

    assert compress_image(stream, 9, 50, 1.0)["/Filter"] == "/DCTDecode"


def test_compress_image__from_jpeg_2000():
    image = _photo()
    stream = _stream(image)
    stream[NameObject("/Filter")] = NameObject("/JPXDecode")
    stream._data = _encoded(image, "JPEG2000", irreversible=False)

    assert compress_image(stream, 9, 50, 1.0)["/Filter"] == "/DCTDecode"


def test_compress_image__keeps_images_without_a_clear_benefit():
    image = _photo()
    stream = _stream(image)
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream._data = _encoded(image, "JPEG", quality=50)

    assert compress_image(stream, 9, 50, 1.0) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("/ImageMask", BooleanObject(True)),
        ("/Decode", ArrayObject([NumberObject(1), NumberObject(0)] * 3)),
        ("/Mask", ArrayObject([NumberObject(0)] * 6)),
        ("/ColorSpace", NameObject("/DeviceCMYK")),
        ("/ColorSpace", ArrayObject([NameObject("/Indexed"), NameObject("/DeviceRGB"), NumberObject(255), b""])),
        ("/ColorSpace", ArrayObject([NameObject("/Lab"), DictionaryObject()])),
        ("/ColorSpace", ArrayObject([NameObject("/ICCBased"), DictionaryObject({NameObject("/N"): NumberObject(4)})])),
        ("/BitsPerComponent", NumberObject(16)),
        ("/Filter", NameObject("/CCITTFaxDecode")),
        ("/Width", NumberObject(0)),
        ("/Width", NameObject("/NotANumber")),
    ],
)
def test_compress_image__keeps_images_with_side_effects(key, value):
    stream = _stream(_photo())
    stream[NameObject(key)] = value
    assert compress_image(stream, 9, 75, 0.5) is None


def test_compress_image__keeps_data_of_unexpected_size():
    stream = _stream(_photo())
    stream._data = zlib.compress(b"too short", 9)
    assert compress_image(stream, 9, 75, 1.0) is None


def test_compress_image__keeps_broken_jpeg():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream._data = b"This is not a JPEG."
    assert compress_image(stream, 9, 75, 1.0) is None


def test_compress_image__keeps_jpeg_below_another_filter():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = ArrayObject([NameObject("/ASCII85Decode"), NameObject("/DCTDecode")])
    assert compress_image(stream, 9, 50, 1.0) is None


def test_compress_image__keeps_jpeg_below_broken_flate():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = ArrayObject([NameObject("/FlateDecode"), NameObject("/DCTDecode")])
    stream._data = b"This is not Flate."
    assert compress_image(stream, 9, 50, 1.0) is None


def test_compress_image__keeps_huge_images_which_pillow_refuses():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream._data = _encoded(_photo(), "JPEG", quality=95)
    with mock.patch("pypdf._image_compression.Image.open", side_effect=Image.DecompressionBombError):
        assert compress_image(stream, 9, 50, 1.0) is None


def test_compress_image__keeps_jpeg_which_does_not_match_the_dictionary():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream._data = _encoded(_photo().convert("L"), "JPEG", quality=95)
    assert compress_image(stream, 9, 50, 1.0) is None


def test_compress_image__keeps_jpeg_with_decode_parameters():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream[NameObject("/DecodeParms")] = DictionaryObject({NameObject("/ColorTransform"): NumberObject(0)})
    stream._data = _encoded(_photo(), "JPEG", quality=95)
    assert compress_image(stream, 9, 50, 1.0) is None


def test_compress_image__reduces_the_resolution_without_lossy_coding():
    compressed = compress_image(_stream(_photo()), 9, None, 0.5)

    assert compressed["/Filter"] == "/FlateDecode"
    assert (compressed["/Width"], compressed["/Height"]) == (200, 150)
    assert compressed.get_data() != b""
    assert len(compressed.get_data()) == 200 * 150 * 3


def test_compress_image__reduces_jpeg_as_jpeg():
    stream = _stream(_photo())
    stream[NameObject("/Filter")] = NameObject("/DCTDecode")
    stream._data = _encoded(_photo(), "JPEG", quality=95)

    compressed = compress_image(stream, 9, None, 0.5)

    assert compressed["/Filter"] == "/DCTDecode"
    assert (compressed["/Width"], compressed["/Height"]) == (200, 150)


def test_compress_image__nothing_to_do():
    assert compress_image(_stream(_photo()), 9, None, 1.0) is None


def test_compress__is_lossless_without_image_parameters():
    image = _photo()
    writer, reference = _writer_with_image(_stream(image))
    writer.compress()
    assert reference.get_object()["/Filter"] == "/FlateDecode"
    assert reference.get_object().get_data() == image.tobytes()


def test_compress__image_quality():
    writer, reference = _writer_with_image(_stream(_photo()))

    writer.compress(image_quality=75)

    assert reference.get_object()["/Filter"] == "/DCTDecode"
    output = BytesIO()
    writer.write(output)
    image = PdfReader(output).pages[0].images[0].image
    assert image.size == (400, 300)


def test_compress__image_max_dpi():
    writer, reference = _writer_with_image(_stream(_photo(1000, 500)), page_size=72)

    writer.compress(image_max_dpi=150)

    assert (reference.get_object()["/Width"], reference.get_object()["/Height"]) == (150, 75)


def test_compress__image_max_dpi_uses_the_largest_page():
    image_stream = _stream(_photo(1000, 500))
    writer, reference = _writer_with_image(image_stream, page_size=72)
    page = writer.add_blank_page(width=720, height=720)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Image"): reference})}
    )

    writer.compress(image_max_dpi=50)

    assert reference.get_object()["/Width"] == 500


def test_compress__image_max_dpi_keeps_images_below_the_limit():
    writer, reference = _writer_with_image(_stream(_photo(400, 300)), page_size=720)
    writer.compress(image_max_dpi=150)
    assert reference.get_object()["/Width"] == 400


def test_compress__image_max_dpi_finds_images_in_forms():
    writer, _ = _writer_with_image(_stream(_photo(100, 100)))
    image = writer._add_object(_stream(_photo(1000, 500)))
    form = StreamObject()
    form[NameObject("/Subtype")] = NameObject("/Form")
    form[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Image"): image})}
    )
    form_reference = writer._add_object(form)
    writer.pages[0]["/Resources"]["/XObject"][NameObject("/Form")] = form_reference

    writer.compress(image_max_dpi=150)

    assert image.get_object()["/Width"] == 150


def test_compress__image_max_dpi_ignores_images_of_no_page():
    writer, _ = _writer_with_image(_stream(_photo(100, 100)))
    image = writer._add_object(_stream(_photo(1000, 500)))
    writer.root_object[NameObject("/TestImage")] = image

    writer.compress(image_max_dpi=150)

    assert image.get_object()["/Width"] == 1000


def test_compress__image_max_dpi_ignores_pages_without_a_valid_size():
    writer, reference = _writer_with_image(_stream(_photo(1000, 500)))
    writer.pages[0][NameObject("/MediaBox")] = ArrayObject([NumberObject(0)] * 2)

    writer.compress(image_max_dpi=150)

    assert reference.get_object()["/Width"] == 1000


def test_compress__keeps_masks():
    image = _stream(_photo())
    mask = _stream(_photo().convert("L"), NameObject("/DeviceGray"))
    writer, reference = _writer_with_image(image)
    mask_reference = writer._add_object(mask)
    reference.get_object()[NameObject("/SMask")] = mask_reference

    writer.compress(image_quality=50)

    assert reference.get_object()["/Filter"] == "/DCTDecode"
    assert mask_reference.get_object()["/Filter"] == "/FlateDecode"


def test_compress__reduces_masks_with_their_image():
    writer, reference = _writer_with_image(_stream(_photo(1000, 500)), page_size=72)
    mask = writer._add_object(_stream(_photo(1000, 500).convert("L"), NameObject("/DeviceGray")))
    reference.get_object()[NameObject("/SMask")] = mask

    writer.compress(image_quality=75, image_max_dpi=150)

    assert mask.get_object()["/Filter"] == "/FlateDecode"
    assert mask.get_object()["/Width"] == 150
    assert reference.get_object()["/Width"] == 150


def test_compress__keeps_masks_with_matte():
    writer, reference = _writer_with_image(_stream(_photo(1000, 500)), page_size=72)
    mask = _stream(_photo(1000, 500).convert("L"), NameObject("/DeviceGray"))
    mask[NameObject("/Matte")] = ArrayObject([NumberObject(0)] * 3)
    mask_reference = writer._add_object(mask)
    reference.get_object()[NameObject("/SMask")] = mask_reference

    writer.compress(image_max_dpi=150)

    assert mask_reference.get_object()["/Width"] == 1000


@pytest.mark.parametrize("quality", [0, 96, -1])
def test_compress__invalid_image_quality(quality):
    writer = PdfWriter(clone_from=RESOURCE_ROOT / "crazyones.pdf")
    with pytest.raises(ValueError, match=rf"^image_quality must be between 1 and 95, got {quality}\.$"):
        writer.compress(image_quality=quality)


@pytest.mark.parametrize("dpi", [0, -1])
def test_compress__invalid_image_max_dpi(dpi):
    writer = PdfWriter(clone_from=RESOURCE_ROOT / "crazyones.pdf")
    with pytest.raises(ValueError, match=rf"^image_max_dpi must be positive, got {dpi}\.$"):
        writer.compress(image_max_dpi=dpi)


def test_compress__needs_pillow_only_for_lossy_compression():
    writer = PdfWriter(clone_from=RESOURCE_ROOT / "crazyones.pdf")
    with mock.patch.dict(sys.modules, {"PIL": None, "pypdf._image_compression": None}):
        writer.compress()
        with pytest.raises(ImportError):
            writer.compress(image_quality=75)
