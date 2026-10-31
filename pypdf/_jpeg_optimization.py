"""
Lossless optimization of JPEG images for :meth:`PdfWriter.compress<pypdf.PdfWriter.compress>`.

This needs the optional package ``mozjpeg-lossless-optimization``. It ends the whole
process on some damaged JPEG files instead of reporting an error, so that it runs in a
process of its own. Such a file then only stays as it is.
"""

import struct
import subprocess
import sys
from importlib.util import find_spec

_FRAME_HEADER = ">I"
_FRAME_HEADER_SIZE = 4
_MAXIMUM_WORKERS = 8
_TIMEOUT_IN_SECONDS = 600
_WORKER = """
import struct
import sys

sys.path.pop(0)
import mozjpeg_lossless_optimization as mozjpeg

while True:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        break
    jpeg = sys.stdin.buffer.read(struct.unpack(">I", header)[0])
    try:
        optimized = mozjpeg.optimize(jpeg, mozjpeg.COPY_MARKERS.ALL)
    except ValueError:
        optimized = b""
    sys.stdout.buffer.write(struct.pack(">I", len(optimized)) + optimized)
    sys.stdout.buffer.flush()
"""


def is_available() -> bool:
    return find_spec("mozjpeg_lossless_optimization") is not None


def _read_frames(data: bytes) -> list[bytes]:
    frames = []
    position = 0
    while position + _FRAME_HEADER_SIZE <= len(data):
        size = struct.unpack_from(_FRAME_HEADER, data, position)[0]
        position += _FRAME_HEADER_SIZE
        if position + size > len(data):
            break
        frames.append(data[position:position + size])
        position += size
    return frames


def _run_worker(jpegs: list[bytes]) -> tuple[list[bytes], bool]:
    """
    Optimize JPEG files until the worker ends.

    Returns:
        The results that are complete, and whether another worker may continue.

    """
    frames = b"".join(struct.pack(_FRAME_HEADER, len(jpeg)) + jpeg for jpeg in jpegs)
    try:
        process = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _WORKER],
            input=frames,
            capture_output=True,
            timeout=_TIMEOUT_IN_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exception:
        return _read_frames(exception.stdout or b""), False
    except OSError:
        return [], False
    return _read_frames(process.stdout), True


def optimize_jpegs(jpegs: list[bytes]) -> list[bytes | None]:
    """
    Store JPEG files in a smaller way which decodes to the same pixels.

    Args:
        jpegs: The JPEG files.

    Returns:
        The optimized file for each input, or ``None`` if it could not be optimized.

    """
    results: list[bytes | None] = []
    for _ in range(_MAXIMUM_WORKERS):
        if len(results) >= len(jpegs):
            break
        frames, may_continue = _run_worker(jpegs[len(results):])
        results.extend(frame or None for frame in frames)
        if len(results) < len(jpegs):
            results.append(None)  # This file ended the worker.
        if not may_continue:
            break
    results.extend([None] * (len(jpegs) - len(results)))
    return results
