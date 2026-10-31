"""
Find out how large images appear on their pages, for :meth:`PdfWriter.compress<pypdf.PdfWriter.compress>`.

Parsing all content streams with pypdf would take minutes for large files, and only
four operators matter, so that a regular expression finds them. A result which looks
wrong is no result: Such images keep the estimate from the size of the page, which
gives them at least the requested resolution.
"""

import math
import re
from typing import Any

from .generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

Matrix = tuple[float, float, float, float, float, float]

_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_MAXIMUM_FORM_DEPTH = 8
_MAXIMUM_FORM_CALLS = 100_000
_NUMBER = rb"[-+]?(?:\d+\.?\d*|\.\d+)"
_INLINE_IMAGE = re.compile(rb"(?<![^\s])BI\s.*?\sID\s.*?\sEI(?=\s|$)", re.DOTALL)
_TOKEN = re.compile(
    rb"\((?:[^()\\]|\\.|\((?:[^()\\]|\\.|\((?:[^()\\]|\\.)*\))*\))*\)"  # literal string
    rb"|<[0-9A-Fa-f\s]*>"  # hexadecimal string
    rb"|%[^\r\n]*"  # comment
    rb"|(?<![^\s\]>)])(q|Q)(?=[\s/\[(<]|$)"
    rb"|(?<![^\s\]>)])(" + rb"\s+".join([_NUMBER] * 6) + rb")\s+cm(?=[\s/\[(<]|$)"
    rb"|/([^\s/\[\]()<>{}%]+)\s+Do(?=[\s/\[(<]|$)",
    re.DOTALL,
)


class _PlacementError(Exception):
    """The operators of a content stream do not make sense."""


def _multiply(first: Matrix, second: Matrix) -> Matrix:
    a, b, c, d, e, f = first
    g, h, i, j, k, l = second  # noqa: E741
    return (
        a * g + b * i,
        a * h + b * j,
        c * g + d * i,
        c * h + d * j,
        e * g + f * i + k,
        e * h + f * j + l,
    )


def _read_operators(data: bytes) -> list[tuple[str, Any]]:
    """Get the operators which matter for the position of images."""
    if b"ID" in data:
        data = _INLINE_IMAGE.sub(b" ", data)
    operators: list[tuple[str, Any]] = []
    for match in _TOKEN.finditer(data):
        stack_operator, matrix, name = match.groups()
        if stack_operator is not None:
            operators.append((stack_operator.decode(), None))
        elif matrix is not None:
            operators.append(("cm", tuple(float(value) for value in matrix.split())))
        elif name is not None:
            operators.append(("Do", name.decode("latin-1")))
    return operators


def _get_data(contents: Any) -> bytes:
    contents = contents.get_object()
    if isinstance(contents, ArrayObject):
        return b"\n".join(_get_data(part) for part in contents)
    return contents.get_data() if isinstance(contents, StreamObject) else b""


class _PlacementFinder:
    def __init__(self) -> None:
        self.sizes: dict[int, tuple[float, float]] = {}
        self.visited_xobjects: set[int] = set()
        self.form_calls = 0
        self.operators_of_forms: dict[int, list[tuple[str, Any]]] = {}

    def _show(self, reference: IndirectObject, resources: Any, matrix: Matrix, depth: int) -> None:
        xobject = reference.get_object()
        if not isinstance(xobject, StreamObject):
            return
        subtype = xobject.get("/Subtype")
        if subtype == "/Image":
            width = math.hypot(matrix[0], matrix[1])
            height = math.hypot(matrix[2], matrix[3])
            old_width, old_height = self.sizes.get(reference.idnum, (0.0, 0.0))
            self.sizes[reference.idnum] = (max(old_width, width), max(old_height, height))
        elif subtype == "/Form":
            self.form_calls += 1
            if depth >= _MAXIMUM_FORM_DEPTH or self.form_calls > _MAXIMUM_FORM_CALLS:
                raise _PlacementError
            operators = self.operators_of_forms.get(reference.idnum)
            if operators is None:
                operators = _read_operators(xobject.get_data())
                self.operators_of_forms[reference.idnum] = operators
            form_matrix = xobject.get("/Matrix")
            if isinstance(form_matrix, ArrayObject) and len(form_matrix) == 6:
                numbers = tuple(float(value) for value in form_matrix)
                matrix = _multiply(numbers, matrix)  # type: ignore[arg-type]
            self.walk(operators, xobject.get("/Resources", resources), matrix, depth + 1)

    def walk(self, operators: list[tuple[str, Any]], resources: Any, matrix: Matrix, depth: int) -> None:
        xobjects = resources.get("/XObject") if isinstance(resources, DictionaryObject) else None
        if isinstance(xobjects, DictionaryObject):
            self.visited_xobjects.add(id(xobjects))
        stack = []
        for operator, value in operators:
            if operator == "q":
                stack.append(matrix)
            elif operator == "Q":
                if not stack:
                    raise _PlacementError
                matrix = stack.pop()
            elif operator == "cm":
                matrix = _multiply(value, matrix)
            elif isinstance(xobjects, DictionaryObject):
                reference = xobjects.raw_get(f"/{value}") if f"/{value}" in xobjects else None
                if isinstance(reference, IndirectObject):
                    self._show(reference, resources, matrix, depth)


def _find_all_xobject_dictionaries(objects: list[Any]) -> list[DictionaryObject]:
    found = []
    todo = [obj for obj in objects if isinstance(obj, (DictionaryObject, ArrayObject))]
    while todo:
        obj = todo.pop()
        if isinstance(obj, ArrayObject):
            todo.extend(value for value in obj if isinstance(value, (DictionaryObject, ArrayObject)))
            continue
        for key, value in obj.items():
            if isinstance(value, (DictionaryObject, ArrayObject)):
                todo.append(value)
            if key == "/XObject":
                resolved = value.get_object()
                if isinstance(resolved, DictionaryObject):
                    found.append(resolved)
    return found


def find_image_sizes(pages: Any, objects: list[Any]) -> dict[int, tuple[float, float]]:
    """
    Get the largest size at which each image appears on a page.

    Args:
        pages: The pages of the document.
        objects: All objects of the document.

    Returns:
        The largest width and height in points for each object number of an image
        whose every use is known.

    """
    finder = _PlacementFinder()
    not_understood: set[int] = set()
    for page in pages:
        resources = page.get("/Resources")
        before = dict(finder.sizes)
        try:
            contents = page.get("/Contents")
            operators = _read_operators(_get_data(contents)) if contents is not None else []
            finder.walk(operators, resources, _IDENTITY, 0)
        except (_PlacementError, ValueError, OSError):
            not_understood.update(
                idnum for idnum, size in finder.sizes.items() if before.get(idnum) != size
            )
            xobjects = resources.get("/XObject") if isinstance(resources, DictionaryObject) else None
            if isinstance(xobjects, DictionaryObject):
                not_understood.update(
                    value.idnum for value in xobjects.values() if isinstance(value, IndirectObject)
                )

    # Annotations, patterns and transparency groups show images in ways which are unknown here.
    for xobjects in _find_all_xobject_dictionaries(objects):
        if id(xobjects) not in finder.visited_xobjects:
            not_understood.update(value.idnum for value in xobjects.values() if isinstance(value, IndirectObject))
    return {idnum: size for idnum, size in finder.sizes.items() if idnum not in not_understood}
