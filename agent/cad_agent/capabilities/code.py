"""Working with DSL code text: extract it from a model reply, read it, cut a step.

Pure functions without dependencies; both the scaffold and the harness use them.
"""

from __future__ import annotations

import math
import re
from pathlib import Path


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_code(text: str) -> str:
    text = text.split("<|im_end|>")[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = _THINK_RE.sub("", text)
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()


def extract_think(text: str) -> str:
    text = text.split("<|im_end|>")[0]
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()

    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return "" if match is None else match.group(1).strip()


def read_code(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def get_last_step_code(code: str) -> str:
    lines = code.splitlines()
    if not lines:
        raise ValueError("No written code at the current step")
    return lines[-1]


# A profile point in dialect code: `(-12.34,56.78)`. Whitespace is allowed:
# emitters do not produce it, but the model does.
_POINT = r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"
_POINT_RUN_RE = re.compile(rf"{_POINT}(?:\s*,\s*{_POINT})+")
_POINT_RE = re.compile(_POINT)

# From this many vertices a list counts as long. Eight is not a round number but
# the `_fit_circle` threshold of the image2cad emitter: it does not try to collapse
# shorter rings into `circle()`, so a short list in dialect code is meaningful as a
# whole, while a long one is tessellation.
DEFAULT_ELIDE_MIN_POINTS = 8


def elide_point_lists(code: str, min_points: int = DEFAULT_ELIDE_MIN_POINTS) -> tuple[str, int]:
    """Collapse long coordinate lists to `<N pts>`.

    Why: `det` writes the residual profile as an explicit vertex list
    (`_sketch_of` -> `.polygon([(x,y), ... ])`) and there is no cap on their count.
    A ring after decimation to 6000 faces is hundreds of vertices, i.e. thousands
    of tokens **in one line**, and that line stays in the accumulated prefix
    forever. This is how the selection prompt overflowed the assistant's context.

    Why collapsing is safe: the decision agent judges the structure of a step (which
    operation it is, where it attaches, whether it continues the construction) and
    the picture. Concrete tessellation coordinates tell it nothing that is not
    visible in the render, while taking more room than the rest of the prompt.

    Returns a pair (text, number of lists collapsed). The caller needs the second
    value: the collapse must be announced in the prompt, otherwise the agent takes
    `<N pts>` for corrupted code.
    """
    if not code:
        return code, 0

    elided = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal elided
        n_points = len(_POINT_RE.findall(match.group(0)))
        if n_points < min_points:
            return match.group(0)
        elided += 1
        return f"<{n_points} pts>"

    return _POINT_RUN_RE.sub(replace, code), elided


# Neighboring `polygon([...])` vertices closer than this (in the model frame,
# extent 200) are merged. The `det` snapshot emits contours with point pairs 1e-4
# apart, and OCC builds an invalid solid with an open mesh from such a polygon.
# The thinned line is only a fallback: it cannot always be executed, since it opens
# some closed operations.
POLYGON_MIN_EDGE = 1e-3

_POLYGON_RE = re.compile(r"polygon\(\[([^\]]*)\]")
_POLYGON_POINT_RE = re.compile(r"\(\s*(-?[0-9.eE+-]+)\s*,\s*(-?[0-9.eE+-]+)\s*\)")


def _thin_polygon_points(match: re.Match[str]) -> str:
    raw = _POLYGON_POINT_RE.findall(match.group(1))
    points = [(float(x), float(y)) for x, y in raw]
    if len(points) < 4:
        return match.group(0)

    def close(a: int, b: int) -> bool:
        return math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]) <= POLYGON_MIN_EDGE

    keep = [0]
    for i in range(1, len(points) - 1):
        if not close(i, keep[-1]):
            keep.append(i)
    # The closing point always stays; a penultimate point coinciding with it is dropped.
    last = len(points) - 1
    while len(keep) > 1 and close(keep[-1], last):
        keep.pop()
    keep.append(last)
    if len(keep) == len(points):
        return match.group(0)
    body = ",".join(f"({raw[i][0]},{raw[i][1]})" for i in keep)
    return f"polygon([{body}]"


def thin_polygons(line: str) -> str:
    """Merge nearly coincident neighboring vertices in every `polygon([...])` of the line.

    A contour without such pairs is returned byte-for-byte unchanged: the caller
    uses string equality to find out whether anything was thinned.
    """
    if "polygon([" not in line:
        return line
    return _POLYGON_RE.sub(_thin_polygon_points, line)
