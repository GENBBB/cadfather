"""Convert a detector's full-program output into an append-able CadQuery
*block* that operates on the running ``r``.

Detectors emit programs like::

    import cadquery as cq
    result = cq.Workplane("XY").moveTo(...)...close().extrude(h)
    result = result.cut(cq.Workplane("XY").center(...).circle(...).extrude(...))

We rewrite them into a block to be appended to ``prev_code``::

    _piece = cq.Workplane("XY").moveTo(...)...close().extrude(h)
    _piece = _piece.cut(cq.Workplane("XY").center(...).circle(...).extrude(...))
    r = r.union(_piece)        # for ADD residual
    # or
    r = r.cut(_piece)          # for CUT residual

The rewrite is a plain string substitution (the detectors emit a fixed
shape: every line starts with ``result = ...`` or ``result = result....``).
That's enough for the v1 - no AST parsing.
"""
from __future__ import annotations

from typing import Literal

Op = Literal["union", "cut"]


def detector_program_to_block(program: str, op: Op,
                              piece_name: str = "_piece") -> str:
    """Rewrite a detector's full program into an append-able block.

    Parameters
    ----------
    program : str
        Output of a detector script - lines like ``import cadquery as cq``,
        ``result = cq.Workplane(...)...``, ``result = result.cut(...)``.
    op : "union" or "cut"
        How to combine the piece into the running ``r``.
    piece_name : str
        Name for the local variable holding the new piece.
    """
    if op not in ("union", "cut"):
        raise ValueError(f"op must be 'union' or 'cut', got {op!r}")

    # Strip imports/comments/blank lines.
    body = [l.rstrip() for raw in program.splitlines()
            if (l := raw.rstrip())
            and not l.lstrip().startswith(("import ", "from ", "#"))]
    if not body:
        raise ValueError("detector program had no executable lines")

    # FAST/CLEAN PATH: a single `result = <expr>` assignment (the form
    # emit_union_program / the primitive emitters produce).  Emit ONE
    # statement: `r = r.<op>(<expr>)`.  This honors the one-line-per-step
    # convention -- each det step appends exactly one line, like the VLM.
    n_assign = sum(1 for l in body
                   if l.lstrip().startswith(("result =", "result=")))
    if n_assign == 1 and body[0].lstrip().startswith(("result =", "result=")):
        rhs = "\n".join(body).split("=", 1)[1].strip()
        return f"r = r.{op}({rhs})"

    # FALLBACK: multi-statement program (e.g. sweep/helix path setup,
    # or extrude with hole cuts) -> keep the `_piece` form.
    lines_out: list[str] = []
    for line in body:
        lines_out.append(line.replace("result", piece_name))
    lines_out.append(f"r = r.{op}({piece_name})")
    return "\n".join(lines_out)


def append_block_to_prev(prev_code: str, block: str) -> str:
    """Concatenate prev_code + block.  Ensures there is exactly one blank
    line in between so the resulting file is human-readable.
    """
    sep = "\n" if prev_code.endswith("\n") else "\n\n"
    return f"{prev_code.rstrip()}\n\n{block.strip()}\n"


def modifier_program_to_block(program: str) -> str:
    """Rewrite a detector's program for *modifier* operations (fillet,
    chamfer, shell, ...).

    These ops operate ON ``r`` directly (``r = r.edges(...).fillet(R)``)
    rather than producing a separate piece to union/cut.  We just:
      - drop imports & comments,
      - replace ``result`` with ``r`` in each remaining line.
    """
    lines_out: list[str] = []
    for raw in program.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        s = line.lstrip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        if s.startswith("#"):
            continue
        lines_out.append(line.replace("result", "r"))
    if not lines_out:
        raise ValueError("modifier program had no executable lines")
    return "\n".join(lines_out)
