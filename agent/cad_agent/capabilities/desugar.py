"""Translate a prediction from the **wrapped** dialect into a CadQuery method chain.

Why this exists. The numeric parameter optimizer (`capabilities/optimize.py`) reads
code with a single parser from the snapshot, `cq_parser.parse_cadquery`, and that parser
only understands a CadQuery **method chain**. Our models are trained on wrapped, i.e.
they emit function calls (`r=extrude(r, ...)`), so the parser had nothing to unroll
and every optimizer call failed with `Could not unroll CadQuery chain`.

The translation is not our code: it is taken from the snapshot
(`cadgen_desugar._desugar_impl`) and works so that point projections are **exact by
construction**: the script is executed with the real cadgen operations while an
equivalent chain is written out alongside. Only the policy around it lives here:

- **when to translate**: the wrapped form is recognized by the snapshot's own function
  (`is_cadgen_functional`), so the test cannot drift from the translator;
- **what translates**: `_desugar_impl` in our process, not `desugar_cadgen()` with a
  foreign interpreter hardcoded; why this is possible is explained in
  `dsl_runtime.import_desugar`;
- **what to strip from the result**: the snapshot preamble (:func:`_strip_snapshot_preamble`):
  the duplicate `import cadquery as cq` and the selector import via `cadquery_addons`;
- **what to return**: the active dialect prefix is glued back onto the translated
  code (:func:`restore_prefix`).

The last point is the only decision here that is visible from outside. The optimizer
gets the **bare** chain: the same parser would choke on a glued wrapped prefix
(`r = None` is not the start of a chain). But the code must go back out with the
prefix, because the prefix is part of the part's code everywhere: `Branch.code` starts
with `obs.prefix_code`, and that whole string is what gets executed. The glued prefix
also keeps the code **extendable**: a wrapped operation takes `cq.Workplane` as its
first argument, so `r=hole(r, ...)` can still be appended to a translated chain.

What is **not** here. The snapshot has a second, hybrid path: if the script contains
an exotic operation (`gear`, `sweep`, `spring`, `loft`, `shell`, `rib`, `thread`), it
freezes the script prefix into an STL and emits `r = __MESH__("path.stl")` plus a
translatable tail. The snapshot parser understands `__MESH__`, our execution does not:
`execute` does `exec(code)` and takes `r` from the namespace, with no source for
`__MESH__`. The part code would stop being self-contained (it would reference a file
in the run directory), so the hybrid path is deliberately not wired in: a script with an
exotic operation gets a clear refusal.
"""

from __future__ import annotations

import ast
import logging
import time
from typing import Any

from cad_agent import dsl_runtime

logger = logging.getLogger(__name__)

# The dialect the snapshot can translate from. The only one: chain needs no translation
# (its parser reads it directly), and we have no third dialect.
SOURCE_DIALECT = "wrapped"

# Operations the snapshot unrolls into a chain. The list mirrors `_OPS` in
# `cadgen_desugar._desugar_impl` and is used only for the refusal text: the snapshot
# decides, and here we merely explain to the user what tripped it.
SUPPORTED_OPS = ("extrude", "hole", "orto_cut", "revolve")

# Marker in the returned code. Not decoration: candidate codes in wrapped lie next to
# each other in `logs/figures/**`, and without the marker a translated code would read
# like a model prediction in another language, i.e. like a defect of ours.
MARKER = "# desugar: wrapped -> chain (parameter optimizer)"


# The preamble the snapshot prepends to the translation (`cadgen_desugar._desugar_impl`):
# a duplicate `import cadquery as cq` and, if a selector name occurred in the source,
# a selector import via `cadquery_addons`. We strip it; see
# :func:`_strip_snapshot_preamble`.
_PREAMBLE_CQ = "import cadquery as cq"
_PREAMBLE_SELECTORS = (
    "try:",
    "    from cadquery_addons.selectors import PointOnEdgeSelector, PointOnFaceSelector",
    "except ImportError:",
    "    from cadgen.selectors import PointOnEdgeSelector, PointOnFaceSelector",
)
_SELECTOR_CALLS = ("PointOnEdgeSelector(", "PointOnFaceSelector(")


class DesugarFailed(RuntimeError):
    """Translation failed: wrong code shape, wrong dialect, or an exotic operation."""


# Appended tail. The code after a previous `optimize` is a prefix, a marker and a chain;
# if a wrapped operation is appended to it, the snapshot translates such code
# incorrectly. It executes and rewrites the chain lines as is, but its "first body"
# flag is not reset by them (`cadgen_desugar._desugar_impl`, branch `not is_op_call`).
# Two outcomes follow: the first appended `extrude`/`revolve` comes out as `r=<chain>` and
# REPLACES the body instead of uniting with it, and a tail of only `hole`/`orto_cut`
# fails with `no ops desugared`. We do not patch the snapshot; we fix the output instead.
#
# Operations after which the snapshot itself resets the flag: only they are written
# as `r=` / `r=r.union(...)`; the others are written as `r=r.cut(...)`.
_BODY_OPS = ("extrude", "revolve")
# Helper operation for a tail without `extrude`/`revolve`: lets the snapshot reset the flag
# and reach the end. Its line is removed from the output and does not affect the part code.
# `False` means no projection onto a surface: the plane is taken from the point as is.
_SENTINEL = "r=extrude(r,(0,0,0),'XY',\"sketch().push([(0,0)]).rect(1,1).finalize()\",1,False)"


def _op_name(stmt: ast.stmt) -> str | None:
    """Operation name if the statement is an operation in the snapshot's sense (`r = name(...)`), else `None`.

    A method chain (`r=cq.Workplane(...)...`, `r=r.union(...)`) is not an operation.
    """
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "r"
            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)):
        return stmt.value.func.id
    return None


def _appended_tail(code: str) -> tuple[int, list[str]] | None:
    """Number of chain statements before the appended tail, and the operations in the tail.

    `None` means no tail: there is no code after `optimize` here, or nothing was appended
    to the chain. Statements are counted the way the snapshot sees them: without imports and
    `r = None`.
    """
    if MARKER not in code:
        return None
    body = [s for s in ast.parse(code).body
            if not isinstance(s, (ast.Import, ast.ImportFrom))
            and not (isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant)
                     and s.value.value is None)]
    names = [_op_name(s) for s in body]
    first_op = next((i for i, name in enumerate(names) if name is not None), None)
    if not first_op:
        return None
    return first_op, [name for name in names[first_op:] if name is not None]


def _join_tail(chain: str, n_chain: int, sentinel: bool) -> str:
    """Fix the snapshot output on mixed code: the first body of the tail goes in by union."""
    tree = ast.parse(chain)
    parts = [ast.get_source_segment(chain, s) or "" for s in tree.body]
    if sentinel:
        parts.pop()
    for i in range(n_chain, len(parts)):
        if parts[i].startswith("r=cq."):
            parts[i] = f"r=r.union({parts[i][2:]})"
            break
    return "\n".join(parts) + "\n"


def is_wrapped_functional(code: str) -> bool:
    """Whether the code looks like the functional wrapped form (`from cadgen.*` + `r=op(r, ...)`).

    The test is asked of the snapshot rather than written here as a second regular
    expression: drifting from the translator it would either refuse translatable code or
    try to translate untranslatable code, silently in both cases.
    """
    return bool(dsl_runtime.import_desugar().is_cadgen_functional(code))


def _strip_snapshot_preamble(chain: str) -> str:
    """Strip the snapshot preamble from the translated code.

    Why. The translation goes up through :func:`restore_prefix`, which glues on
    ``CODE_PREFIX_WRAPPED``, where ``import cadquery as cq`` and ``PointOnEdgeSelector``
    already exist. The snapshot preamble duplicates them, and the duplicate lands in
    ``Branch.code``, i.e. in artifacts and in the model prompt, whose training never
    contained such text.

    Worse than the duplicate is its try branch. ``from cadquery_addons.selectors import ...``
    executes ``cadquery_addons/__init__.py``, which patches ``Workplane.extrude``
    (``from . import sequential_extrude  # patches Workplane.extrude on import``).
    The run environment lacks the package, so ``except ImportError`` always fires there;
    but the generator environment has it, and there the header would silently replace the
    operation almost every part is made of.

    What is NOT here: a condition "keep the import if a selector is used".
    Predictions contain no selector calls (all occurrences of the names are import lines).
    But the second consumer of the translation, the snapshot parser, receives the chain
    **without** a prefix and defines only ``cq`` (``cq_parser``: ``_ns = {'cq': _cq}``).
    If a selector call ever appeared, the prefix ``exec`` in ``_try_selective_bevel`` would
    fail with ``NameError`` and the outer ``try`` would silently return ``[]``. So such code
    is not fixed here, but it does not pass unnoticed either.
    """
    lines = chain.split("\n")
    if lines and lines[0] == _PREAMBLE_CQ:
        lines = lines[1:]
    else:
        logger.warning(
            "Translation did not start with %r: the snapshot preamble has changed, nothing to strip",
            _PREAMBLE_CQ,
        )
    if tuple(lines[: len(_PREAMBLE_SELECTORS)]) == _PREAMBLE_SELECTORS:
        lines = lines[len(_PREAMBLE_SELECTORS) :]
    body = "\n".join(lines)
    if any(call in body for call in _SELECTOR_CALLS):
        logger.warning(
            "The translated code calls a selector but the import is stripped: the snapshot parser "
            "gets a chain without the prefix and would silently lose the selective bevel"
        )
    return body


def to_chain(code: str, dialect: str | None = None) -> dict[str, Any]:
    """Translate a wrapped script into a CadQuery method chain.

    Returns ``{"code": chain, "wall_sec": elapsed}``.
    Raises :class:`DesugarFailed` if there is nothing to translate or nothing to translate with.

    The translation costs about one execution: the script is **executed** in full with the
    real cadgen operations. So the time is measured here and passed up as a separate field:
    the cost of the `optimize` capability now has two parts, and summing them into one
    number would hide the translation.
    """
    active = dialect or dsl_runtime.active_dialect()
    if active != SOURCE_DIALECT:
        # chain needs no translation, and feeding it in would call the wrapped cadgen
        # modules where chain is on sys.path: the translation would fail on import and the
        # refusal would look like a broken snapshot.
        raise DesugarFailed(
            f"wrapped->chain translation with the active dialect {active!r} is neither needed nor working"
        )

    module = dsl_runtime.import_desugar()
    if not module.is_cadgen_functional(code):
        raise DesugarFailed(
            "the code does not look like the functional wrapped form: there is no call "
            "of the form `r=op(r, ...)` with `from cadgen.*` imports; nothing to translate"
        )

    tail = _appended_tail(code)
    sentinel = tail is not None and not any(op in _BODY_OPS for op in tail[1])
    source = code.rstrip("\n") + "\n" + _SENTINEL + "\n" if sentinel else code

    started = time.monotonic()
    try:
        # Specifically `_desugar_impl`, not `desugar_cadgen`: the latter starts a subprocess
        # of a foreign interpreter at a path hardcoded in the snapshot. See
        # dsl_runtime.import_desugar.
        chain = module._desugar_impl(source)
    except DesugarFailed:
        raise
    except Exception as exc:
        raise DesugarFailed(
            f"wrapped->chain translation failed ({type(exc).__name__}: {exc}); "
            f"the snapshot expands only {', '.join(SUPPORTED_OPS)}"
        ) from exc
    wall = time.monotonic() - started

    if not (chain or "").strip():
        raise DesugarFailed("wrapped->chain translation returned empty code")

    chain = _strip_snapshot_preamble(chain)
    if not chain.strip():
        raise DesugarFailed("nothing of the translation remained after stripping the snapshot preamble")
    if tail is not None:
        n_out = len(ast.parse(chain).body)
        n_in = tail[0] + len(tail[1]) + int(sentinel)
        if n_out != n_in:
            raise DesugarFailed(
                f"translating code with an appended tail: {n_out} statements out "
                f"vs {n_in} in; lines cannot be matched"
            )
        chain = _join_tail(chain, tail[0], sentinel)

    return {"code": chain, "wall_sec": wall}


def restore_prefix(chain_code: str, dialect: str | None = None) -> str:
    """Glue the active dialect prefix and the marker onto the translated code.

    Without the prefix the code would run (it is plain CadQuery) but would no longer be
    the same kind of object as other `Branch.code`: our part code **includes** the
    preamble, it is written to artifacts and steps are counted from it. Also, with the
    prefix wrapped operations can again be appended to the chain, since they take
    `cq.Workplane` as the first argument.
    """
    prefix = dsl_runtime.code_prefix(dialect)
    return f"{prefix}\n{MARKER}\n{chain_code.strip()}\n"
