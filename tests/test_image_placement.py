"""Test finding out how large images appear on their pages."""

import time

import pytest

from pypdf import PdfWriter
from pypdf._image_placement import _PlacementError, _read_operators, find_image_sizes
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    EncodedStreamObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NumberObject,
)

from .test_image_compression import _photo, _stream


def _content(writer: PdfWriter, data: bytes) -> IndirectObject:
    stream = DecodedStreamObject()
    stream.set_data(data)
    return writer._add_object(stream)


def _form(
    writer: PdfWriter, data: bytes, xobjects: dict[str, IndirectObject] | None, matrix: list[float] | None = None
) -> IndirectObject:
    form = DecodedStreamObject()
    form.set_data(data)
    form[NameObject("/Subtype")] = NameObject("/Form")
    if xobjects is not None:
        form[NameObject("/Resources")] = _resources(xobjects)
    if matrix is not None:
        form[NameObject("/Matrix")] = ArrayObject([FloatObject(value) for value in matrix])
    return writer._add_object(form)


def _resources(xobjects: dict[str, IndirectObject]) -> DictionaryObject:
    return DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject(key): value for key, value in xobjects.items()})}
    )


def _image(writer: PdfWriter, width: int = 400, height: int = 300) -> IndirectObject:
    return writer._add_object(_stream(_photo(width, height)))


def _page(writer: PdfWriter, data: bytes, xobjects: dict[str, IndirectObject]) -> None:
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Contents")] = _content(writer, data)
    page[NameObject("/Resources")] = _resources(xobjects)


def _sizes(writer: PdfWriter) -> dict[int, tuple[float, float]]:
    return find_image_sizes(writer.pages, writer._objects)


def test_read_operators():
    assert _read_operators(b"q\n2 0 0 3 4 5 cm\n/Im1 Do\nQ") == [
        ("q", None),
        ("cm", (2.0, 0.0, 0.0, 3.0, 4.0, 5.0)),
        ("Do", "Im1"),
        ("Q", None),
    ]


@pytest.mark.parametrize(
    "data",
    [
        b"(q 1 0 0 1 0 0 cm /Im1 Do Q) Tj",
        b"(nested (q) strings \\) Q ) Tj",
        b"<71> Tj",
        b"% q 1 0 0 1 0 0 cm\n",
        b"BI /W 1 /H 1 ID q /Im1 Do Q EI ",
        b"/Fontq Tf /Q gs",
    ],
)
def test_read_operators__ignores_other_tokens(data):
    assert _read_operators(data) == []


def test_read_operators__unfinished_string():
    with pytest.raises(_PlacementError):
        _read_operators(b"(This string does not end q")


def test_read_operators__strings_nested_too_deeply():
    with pytest.raises(_PlacementError):
        _read_operators(b"(a(b(c(d)c)b)a)")


def test_read_operators__nested_strings():
    assert _read_operators(b"(a(b(c)b)a) Tj q") == [("q", None)]


def test_read_operators__unfinished_inline_image():
    with pytest.raises(_PlacementError):
        _read_operators(b"BI /W 1 /H 1 ID q")


def test_read_operators__numbers():
    operators = _read_operators(b"-.5 +1. 0 1e0 cm 1 2 3 4 5 6 cm")
    assert operators == [("cm", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))]


def test_find_image_sizes():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"q 200 0 0 100 10 10 cm /Image Do Q", {"/Image": image})
    assert _sizes(writer) == {image.idnum: (200.0, 100.0)}


def test_find_image_sizes__combines_matrices():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"q 2 0 0 2 0 0 cm 50 0 0 25 0 0 cm /Image Do Q", {"/Image": image})
    assert _sizes(writer) == {image.idnum: (100.0, 50.0)}


def test_find_image_sizes__restores_the_matrix():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"q 9 0 0 9 0 0 cm Q 10 0 0 20 0 0 cm /Image Do", {"/Image": image})
    assert _sizes(writer) == {image.idnum: (10.0, 20.0)}


def test_find_image_sizes__rotation():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"0 100 -50 0 0 0 cm /Image Do", {"/Image": image})
    assert _sizes(writer) == {image.idnum: (100.0, 50.0)}


def test_find_image_sizes__keeps_the_largest_use():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"q 50 0 0 50 0 0 cm /Image Do Q q 100 0 0 20 0 0 cm /Image Do Q", {"/Image": image})
    _page(writer, b"q 70 0 0 300 0 0 cm /Image Do Q", {"/Image": image})
    assert _sizes(writer) == {image.idnum: (100.0, 300.0)}


def test_find_image_sizes__form():
    writer = PdfWriter()
    image = _image(writer)
    form = _form(writer, b"10 0 0 20 0 0 cm /Image Do", {"/Image": image}, [2, 0, 0, 2, 0, 0])
    _page(writer, b"3 0 0 3 0 0 cm /Form Do", {"/Form": form})
    assert _sizes(writer) == {image.idnum: (60.0, 120.0)}


def test_find_image_sizes__form_with_the_resources_of_its_parent():
    writer = PdfWriter()
    image = _image(writer)
    form = _form(writer, b"10 0 0 20 0 0 cm /Image Do", None)
    _page(writer, b"/Form Do", {"/Form": form, "/Image": image})
    assert _sizes(writer) == {image.idnum: (10.0, 20.0)}


def test_find_image_sizes__form_used_twice():
    writer = PdfWriter()
    image = _image(writer)
    form = _form(writer, b"10 0 0 10 0 0 cm /Image Do", {"/Image": image})
    _page(writer, b"q /Form Do Q q 5 0 0 5 0 0 cm /Form Do Q", {"/Form": form})
    assert _sizes(writer) == {image.idnum: (50.0, 50.0)}


def test_find_image_sizes__contents_array():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"", {"/Image": image})
    writer.pages[0][NameObject("/Contents")] = ArrayObject(
        [_content(writer, b"q 30 0 0 40 0 0"), _content(writer, b"cm /Image Do Q")]
    )
    assert _sizes(writer) == {image.idnum: (30.0, 40.0)}


def test_find_image_sizes__unknown_name():
    writer = PdfWriter()
    _page(writer, b"/Other Do", {"/Image": _image(writer)})
    assert _sizes(writer) == {}


def test_find_image_sizes__page_without_contents():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    assert _sizes(writer) == {}


def test_find_image_sizes__unbalanced_stack():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"10 0 0 10 0 0 cm /Image Do Q", {"/Image": image})
    assert image.idnum not in _sizes(writer)


def test_find_image_sizes__endless_recursion_of_forms():
    writer = PdfWriter()
    image = _image(writer)
    form = _form(writer, b"/Image Do /Form Do", {"/Image": image})
    form.get_object()["/Resources"]["/XObject"][NameObject("/Form")] = form
    _page(writer, b"/Form Do", {"/Form": form})
    assert image.idnum not in _sizes(writer)


def test_find_image_sizes__too_many_form_calls(monkeypatch):
    monkeypatch.setattr("pypdf._image_placement._MAXIMUM_FORM_CALLS", 2)
    writer = PdfWriter()
    image = _image(writer)
    form = _form(writer, b"/Image Do", {"/Image": image})
    _page(writer, b"/Form Do /Form Do /Form Do", {"/Form": form, "/Image": image})
    assert image.idnum not in _sizes(writer)


def test_find_image_sizes__ignores_images_also_used_elsewhere():
    """A pattern or an annotation shows the image at an unknown size."""
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"10 0 0 10 0 0 cm /Image Do", {"/Image": image})
    pattern = DecodedStreamObject()
    pattern[NameObject("/PatternType")] = NumberObject(1)
    pattern[NameObject("/Resources")] = _resources({"/Image": image})
    writer.root_object[NameObject("/TestPattern")] = writer._add_object(pattern)
    assert image.idnum not in _sizes(writer)


def test_find_image_sizes__direct_resources_in_arrays():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"10 0 0 10 0 0 cm /Image Do", {"/Image": image})
    writer.root_object[NameObject("/TestArray")] = ArrayObject(
        [ArrayObject([DictionaryObject({NameObject("/Resources"): _resources({"/Image": image})})])]
    )
    assert image.idnum not in _sizes(writer)


def test_compress__image_max_dpi_uses_the_size_on_the_page():
    writer = PdfWriter()
    image = _image(writer, 1000, 500)
    _page(writer, b"q 72 0 0 36 0 0 cm /Image Do Q", {"/Image": image})

    writer.compress(image_max_dpi=150)

    assert (image.get_object()["/Width"], image.get_object()["/Height"]) == (150, 75)


def test_compress__image_max_dpi_keeps_images_at_the_requested_resolution():
    writer = PdfWriter()
    image = _image(writer, 1500, 750)
    _page(writer, b"q 720 0 0 360 0 0 cm /Image Do Q", {"/Image": image})

    writer.compress(image_max_dpi=150)

    assert image.get_object()["/Width"] == 1500


def test_compress__image_max_dpi_follows_the_finer_direction():
    writer = PdfWriter()
    image = _image(writer, 1000, 1000)
    _page(writer, b"q 720 0 0 72 0 0 cm /Image Do Q", {"/Image": image})

    writer.compress(image_max_dpi=100)

    assert image.get_object()["/Width"] == 1000


def test_compress__image_max_dpi_reduces_masks_by_the_size_of_their_image():
    writer = PdfWriter()
    image = _image(writer, 1000, 500)
    mask = writer._add_object(_stream(_photo(800, 400).convert("L"), NameObject("/DeviceGray")))
    image.get_object()[NameObject("/SMask")] = mask
    _page(writer, b"q 72 0 0 36 0 0 cm /Image Do Q", {"/Image": image})

    writer.compress(image_max_dpi=150)

    assert mask.get_object()["/Width"] == 150


def test_compress__image_max_dpi_ignores_images_shown_without_a_size():
    writer = PdfWriter()
    image = _image(writer, 1000, 500)
    _page(writer, b"0 0 0 0 0 0 cm /Image Do", {"/Image": image})

    writer.compress(image_max_dpi=150)

    assert image.get_object()["/Width"] == 1000


@pytest.mark.parametrize(
    ("pattern", "tail"),
    [
        pytest.param(b"(", b"", id="only opening parentheses"),
        pytest.param(b"(a", b"", id="unclosed strings"),
        pytest.param(b"(()", b"", id="unclosed nested strings"),
        pytest.param(b"()(", b"", id="pairs and unclosed strings"),
        pytest.param(b"\\(", b"", id="escapes"),
        pytest.param(b"(\\", b"", id="unclosed escapes"),
        pytest.param(b"<", b"", id="unclosed hexadecimal strings"),
        pytest.param(b"<0", b"", id="unclosed hexadecimal data"),
        pytest.param(b"BI ", b"", id="inline images without data"),
        pytest.param(b"BI ID ", b"", id="inline images without end"),
        pytest.param(b" ID ", b"BI ", id="many data markers"),
        pytest.param(b"1 ", b"", id="numbers without operator"),
        pytest.param(b"1 2 3 4 5 6 ", b"", id="matrices without operator"),
        pytest.param(b"1234567890123456789012345678901234567890 ", b"", id="long numbers without operator"),
        pytest.param(b"12345678901234567890.1234567890 ", b"", id="long decimals without operator"),
        pytest.param(b".", b"", id="dots"),
        pytest.param(b"/a ", b"", id="names without operator"),
        pytest.param(b"%", b"", id="comments"),
    ],
)
def test_read_operators__takes_linear_time(pattern, tail):
    """Damaged content streams must not make the regular expression slow."""

    def duration(size: int) -> float:
        data = tail + pattern * (size // len(pattern))
        start = time.perf_counter()
        try:
            _read_operators(data)
        except _PlacementError:
            pass
        return time.perf_counter() - start

    assert duration(1_500_000) < 2


def test_find_image_sizes__contents_which_do_not_decode():
    writer = PdfWriter()
    image = _image(writer)
    _page(writer, b"10 0 0 10 0 0 cm /Image Do", {"/Image": image})
    contents = writer.pages[0]["/Contents"].get_object()
    broken = EncodedStreamObject()
    broken[NameObject("/Filter")] = NameObject("/FlateD@ecode")
    broken._data = contents.get_data()
    writer._objects[writer.pages[0].raw_get("/Contents").idnum - 1] = broken

    assert image.idnum not in _sizes(writer)
