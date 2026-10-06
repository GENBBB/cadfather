#!/usr/bin/env python3
"""Check of the wrapped -> CadQuery method chain translation and the optimizer seam.

Run: ``python agent/tests/desugar_check.py`` from the ``agent`` directory.

What is checked on the live snapshot (no CAD, `cadquery` is not needed):

1. `dsl_runtime.import_desugar()` really brings up `cadgen_desugar` from the snapshot
   and with **the same** package as the optimizer: two registrations of one directory
   under different names would mean two copies of `cq_parser` in memory.
2. The wrapped-form predicate is asked of the snapshot and answers as needed: "yes"
   for a model prediction, "no" for a chain.
3. Code we **ourselves** return from the optimizer (prefix + chain) is no longer
   considered wrapped. This is not cosmetic: otherwise a repeated `optimize` call
   would translate what is already translated.

What is checked on the `_desugar_impl` stub (the translation itself executes cadgen
operations, which are absent locally):

4. The optimizer receives the **translated** code, not the original.
5. The code goes back out with the dialect prefix attached, and the call carries the
   fields `desugared` and `desugar_sec`.
5a. The snapshot preamble (a duplicate `import cadquery as cq` plus a selector import
   via `cadquery_addons`) is stripped from the translation: the dialect prefix gives
   both, and the try-branch of the second patches `Workplane.extrude` where the
   package exists.
6. A chain on input bypasses translation (`desugared: False`), and the switch
   `desugar=False` also disables it.
7. A translation failure is a capability failure: the original code goes out,
   `success: False` and a reason **readable in one line**, not a traceback header.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from cad_agent import dsl_runtime  # noqa: E402

dsl_runtime.configure("wrapped")

from cad_agent.capabilities import desugar as desugar_mod  # noqa: E402
from cad_agent.capabilities import optimize as optimize_mod  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# A prediction in the form the model returns it: the dialect prefix plus operation lines.
# Sketches are shortened: the form predicate looks at the calls, not at the sketch.
SKETCH = "sketch().rect(20,10).finalize()"
WRAPPED_CODE = (
    dsl_runtime.code_prefix("wrapped")
    + f"\nr=extrude(r,(0,0,0),'XY',\"{SKETCH}\",5)"
    + f"\nr=hole(r,(1,2,3),'XY',\"{SKETCH}\",-3)"
)

# What the snapshot translates it into. Here it is a stub: the real translation executes
# cadgen operations to get exact projections.
#
# The stub repeats the snapshot VERBATIM, preamble included (`_desugar_impl` starts its
# output with `import cadquery as cq` and adds an import via `cadquery_addons` when a
# selector is mentioned). Otherwise stripping the preamble would be checked on code that
# never had one, i.e. not checked at all.
CHAIN_BODY = (
    f"r=cq.Workplane('XY').workplane(offset=0.0).{SKETCH}.extrude(5)\n"
    f"r=r.cut(cq.Workplane('XY').workplane(offset=2.0).{SKETCH}.extrude(13.0))\n"
)
SNAPSHOT_PREAMBLE = (
    "import cadquery as cq\n"
    "try:\n"
    "    from cadquery_addons.selectors import PointOnEdgeSelector, PointOnFaceSelector\n"
    "except ImportError:\n"
    "    from cadgen.selectors import PointOnEdgeSelector, PointOnFaceSelector\n"
)
CHAIN_CODE = SNAPSHOT_PREAMBLE + CHAIN_BODY


class FakeResult:
    """Optimizer answer of the same shape as the snapshot's `OptResult`."""

    def __init__(self, best_code: str) -> None:
        self.best_code = best_code
        self.final_loss = 0.125
        self.params = [1.0]
        self.param_names = ["p0"]
        self.iou_before = 0.5
        self.iou_after = 0.7
        self.guard_reverted = False


class FakeOptimizer:
    """An optimizer that returns exactly the code it received.

    This makes the question "what was it given" checkable in the parent: the call itself
    runs in a fork, and arguments recorded there do not travel out.
    """

    @staticmethod
    def optimize(cadquery_code: str, **_kwargs: object) -> FakeResult:
        return FakeResult(cadquery_code)


def stub_desugar(impl) -> None:
    """Replace `_desugar_impl` in the snapshot module (the fork inherits the replacement)."""
    dsl_runtime.import_desugar()._desugar_impl = impl


print("== snapshot and form predicate ==")

module = dsl_runtime.import_desugar()
check(
    "import_desugar loads cadgen_desugar from the snapshot",
    Path(module.__file__).resolve()
    == (dsl_runtime.VENDOR_ROOT / "cad_optimizer" / "python" / "cadgen_desugar.py").resolve(),
    str(getattr(module, "__file__", None)),
)
check(
    "module is registered as a submodule of the snapshot package",
    module.__name__ == f"{dsl_runtime.OPTIMIZER_PACKAGE}.cadgen_desugar",
    module.__name__,
)
check(
    "the snapshot has both the translation and the form predicate",
    callable(getattr(module, "_desugar_impl", None))
    and callable(getattr(module, "is_cadgen_functional", None)),
)

print("== absolute imports inside the snapshot ==")

# Why this section: the snapshot calls itself in two places not relatively but by the
# literal directory name (`from python.cq_parser import ...`), and this used to fail
# EVERY optimizer call with normalization. It went uncaught because the only optimizer
# check replaced `import_optimizer` with a stub and never touched the real import. Here
# it is touched for real.
#
# The snapshot does not import without `_cad_grad`, and the checks environment usually
# lacks it. The section checks the package structure, not the C++, so when the module is
# not built an empty one takes its place.
try:
    import _cad_grad  # noqa: F401
except ImportError:
    import types

    sys.modules["_cad_grad"] = types.ModuleType("_cad_grad")

optimizer = dsl_runtime.import_optimizer()
check(
    "import_optimizer loads optimizer_numerical from the snapshot",
    Path(optimizer.__file__).resolve()
    == (dsl_runtime.VENDOR_ROOT / "cad_optimizer" / "python" / "optimizer_numerical.py").resolve(),
    str(getattr(optimizer, "__file__", None)),
)

# Exactly the line that used to fail. Not a retelling about `sys.modules` but the import
# itself: it either passes or it does not.
try:
    exec("from python.cq_parser import _SyntheticNode, _make_slot", {})  # noqa: S102
    absolute_import_works, why = True, ""
except Exception as exc:  # pragma: no cover — the message matters more than the branch
    absolute_import_works, why = False, f"{type(exc).__name__}: {exc}"
check("absolute snapshot import by directory name works", absolute_import_works, why)

check(
    "cq_parser is in memory ONCE, not as two copies",
    sys.modules.get(f"{dsl_runtime.OPTIMIZER_ALIAS}.cq_parser")
    is sys.modules.get(f"{dsl_runtime.OPTIMIZER_PACKAGE}.cq_parser"),
)

# The alias is listed by name, and the list must cover the whole snapshot: a snapshot
# update that adds another absolute import should fail the check here, not silently fail
# every optimizer call in a run. Parsed via ast, not by text search: the same spelling
# appears in module headers as usage examples, and it is not an import.
snapshot_dir = dsl_runtime.VENDOR_ROOT / "cad_optimizer" / "python"
absolute_uses: set[str] = set()
for source_path in sorted(snapshot_dir.glob("*.py")):
    for node in ast.walk(ast.parse(source_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            head, _, tail = node.module.partition(".")
            if head == dsl_runtime.OPTIMIZER_ALIAS and tail:
                absolute_uses.add(tail)
check(
    "all absolute snapshot imports are covered by aliases",
    absolute_uses <= set(dsl_runtime.OPTIMIZER_ALIASED_SUBMODULES),
    f"in the snapshot {sorted(absolute_uses)}, in the list {list(dsl_runtime.OPTIMIZER_ALIASED_SUBMODULES)}",
)
check(
    "the alias list has no extras - otherwise it describes a different snapshot",
    set(dsl_runtime.OPTIMIZER_ALIASED_SUBMODULES) <= absolute_uses,
    f"extras: {sorted(set(dsl_runtime.OPTIMIZER_ALIASED_SUBMODULES) - absolute_uses)}",
)

print("== snapshot preamble is described verbatim ==")

# The `_PREAMBLE_*` constants repeat the text the snapshot prepends to a translation. If
# they diverged from the snapshot they would stop matching, and the preamble would leak
# out silently. So the text is compared with the snapshot's own literals, not with
# memory of them.
desugar_src = (dsl_runtime.VENDOR_ROOT / "cad_optimizer" / "python" / "cadgen_desugar.py")
literals = {
    node.value
    for node in ast.walk(ast.parse(desugar_src.read_text(encoding="utf-8")))
    if isinstance(node, ast.Constant) and isinstance(node.value, str)
}
check(
    "duplicate `import cadquery as cq` is described as in the snapshot",
    desugar_mod._PREAMBLE_CQ in literals,
    desugar_mod._PREAMBLE_CQ,
)
check(
    "selector import is described as in the snapshot",
    "\n".join(desugar_mod._PREAMBLE_SELECTORS) in literals,
)

print("== form predicate ==")
check(
    "a wrapped prediction is recognized as the functional form",
    desugar_mod.is_wrapped_functional(WRAPPED_CODE),
)
check(
    "a chain is not taken for the functional form",
    not desugar_mod.is_wrapped_functional(CHAIN_BODY),
)
# The snapshot preamble brings a line `from cadgen.selectors import ...`. If the predicate
# looked only at imports, the translation output would pass for a prediction.
check(
    "the snapshot preamble does not make a chain a functional form",
    not desugar_mod.is_wrapped_functional(CHAIN_CODE),
)

restored = desugar_mod.restore_prefix(CHAIN_BODY)
check(
    "restore_prefix puts the dialect prefix first",
    restored.startswith(dsl_runtime.code_prefix("wrapped")),
)
check("restore_prefix keeps the translation marker", desugar_mod.MARKER in restored)
try:
    ast.parse(restored)
    parses = True
except SyntaxError as exc:  # pragma: no cover — the message matters more than the branch
    parses = False
    print(f"    {exc}")
check("the returned code parses as Python", parses)
check(
    "the returned code assigns r after the prefix",
    restored.rstrip().splitlines()[-1].startswith("r="),
)
# The re-translation trap: the prefix brings `from cadgen.*` with it, and if the predicate
# looked only at imports, our own output would pass for a model prediction and a second
# optimize call would translate what is translated.
check(
    "translated code is not translated again",
    not desugar_mod.is_wrapped_functional(restored),
)

print("== translation refusals ==")

try:
    desugar_mod.to_chain(WRAPPED_CODE, dialect="chain")
    refused_chain = ""
except desugar_mod.DesugarFailed as exc:
    refused_chain = str(exc)
check("with the chain dialect the translation refuses", bool(refused_chain), refused_chain)

try:
    desugar_mod.to_chain(CHAIN_CODE)
    refused_shape = ""
except desugar_mod.DesugarFailed as exc:
    refused_shape = str(exc)
check("on a chain the translation refuses clearly", "nothing to translate" in refused_shape,
      refused_shape)

print("== tail appended to a chain ==")

# The stub repeats the snapshot's OUTPUT by its rules, without executing cadgen: chain
# lines are copied as they are and do not touch the "first body" flag; `extrude` is written
# as `r=<body>` with the flag raised and `r=r.union(...)` with it lowered; `hole` as
# `r=r.cut(...)`; a flag not lowered by the end gives `no ops desugared`. These very rules
# broke mixed code.
SEEN: list[str] = []


def emulate_snapshot(code: str, frozen_stl: object = None) -> str:
    SEEN.append(code)
    out, first = ["import cadquery as cq"], True
    for stmt in ast.parse(code).body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
            continue
        src = ast.get_source_segment(code, stmt) or ""
        call = stmt.value if isinstance(stmt, ast.Assign) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            out.append(src)
            continue
        body = f"cq.Workplane('XY').workplane(offset=0.0).{SKETCH}"
        if call.func.id == "extrude":
            out.append(f"r={body}.extrude(5)" if first else f"r=r.union({body}.extrude(5))")
            first = False
        else:
            out.append(f"r=r.cut({body}.extrude(3))")
    if first:
        raise ValueError("no ops desugared")
    return "\n".join(out) + "\n"


stub_desugar(emulate_snapshot)
OPTIMIZED = desugar_mod.restore_prefix(CHAIN_BODY)

mixed_extrude = OPTIMIZED + f"r=extrude(r,(0,0,9),'XY',\"{SKETCH}\",4)\n"
SEEN.clear()
out = desugar_mod.to_chain(mixed_extrude)["code"]
lines = out.strip().splitlines()
check("tail with extrude: the chain is kept whole", out.startswith(CHAIN_BODY), out[:160])
check("tail with extrude: the appended body is unioned, not replacing",
      lines[-1].startswith("r=r.union(cq.Workplane(") and len(lines) == 3, lines[-1])
check("tail with extrude: the helper operation was not needed", "rect(1,1)" not in SEEN[-1])
check("tail with extrude: the result parses as Python", bool(ast.parse(out)))

mixed_hole = OPTIMIZED + f"r=hole(r,(1,2,3),'XY',\"{SKETCH}\",-2)\n"
SEEN.clear()
try:
    out, why = desugar_mod.to_chain(mixed_hole)["code"], ""
except desugar_mod.DesugarFailed as exc:
    out, why = "", str(exc)
lines = out.strip().splitlines()
check("a tail of holes only is translated, not failing on no ops desugared", bool(out), why)
check("tail of holes only: a helper operation is appended to the snapshot input", "rect(1,1)" in SEEN[-1])
check("tail of holes only: its line is absent from the output",
      len(lines) == 3 and lines[-1].startswith("r=r.cut("), str(lines))

mixed_both = OPTIMIZED + (f"r=hole(r,(1,2,3),'XY',\"{SKETCH}\",-2)\n"
                          f"r=extrude(r,(0,0,9),'XY',\"{SKETCH}\",4)\n"
                          f"r=extrude(r,(0,0,19),'XY',\"{SKETCH}\",4)\n")
out = desugar_mod.to_chain(mixed_both)["code"]
lines = out.strip().splitlines()
check("hole, then two extrudes: the tail has no body replacement",
      [line[:12] for line in lines[2:]] == ["r=r.cut(cq.W", "r=r.union(cq", "r=r.union(cq"], str(lines[2:]))

plain = desugar_mod.to_chain(WRAPPED_CODE)["code"]
check("plain wrapped is translated as before: the first body is `r=`",
      plain.strip().splitlines()[0].startswith("r=cq.Workplane("), plain[:80])

print("== seam with the optimizer ==")

optimize_mod.dsl_runtime.import_optimizer = lambda: FakeOptimizer  # type: ignore[assignment]

stub_desugar(lambda code, frozen_stl=None: CHAIN_CODE)
answer = optimize_mod.optimize_params(WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1)
check("call succeeded", bool(answer.get("success")), str(answer.get("error")))
check("translation is flagged in the answer", answer.get("desugared") is True)
check("translation time is a separate item", isinstance(answer.get("desugar_sec"), float),
      repr(answer.get("desugar_sec")))
check(
    "the optimizer got the translated code, not the original",
    CHAIN_BODY.strip() in (answer.get("code") or "") and "r=extrude(r," not in (answer.get("code") or ""),
)

# The snapshot preamble is stripped. A duplicate `import cadquery as cq` and an import via
# `cadquery_addons` are not needed (the dialect prefix gives both `cq` and the selector),
# and the try-branch of the second silently patches `Workplane.extrude` in an environment
# with that package.
returned = answer.get("code") or ""
check("snapshot preamble stripped: no cadquery_addons", "cadquery_addons" not in returned)
check(
    "snapshot preamble stripped: exactly one `import cadquery as cq`, from the prefix",
    returned.count("import cadquery as cq") == 1,
    f"found {returned.count('import cadquery as cq')} times",
)
check(
    "the translated body follows the marker directly",
    returned.split(desugar_mod.MARKER, 1)[-1].lstrip().startswith("r=cq.Workplane"),
)
check(
    "the code came back with the dialect prefix",
    (answer.get("code") or "").startswith(dsl_runtime.code_prefix("wrapped")),
)

# A chain on input: there is nothing to translate, and this is not a call failure.
answer = optimize_mod.optimize_params(CHAIN_BODY, gt_mesh_path="/no/such.stl", steps=1)
check("a chain bypasses the translation", answer.get("success") and answer.get("desugared") is False,
      str(answer.get("error")))
check("chain code does not grow a prefix", answer.get("code") == CHAIN_BODY)

# The switch: the same wrapped code, but translation is forbidden.
answer = optimize_mod.optimize_params(
    WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1, desugar=False
)
check("desugar=False disables the translation", answer.get("desugared") is False,
      str(answer.get("error")))
check("without translation the optimizer got the original wrapped code", "r=extrude(r," in (answer.get("code") or ""))


def raise_exotic(code: str, frozen_stl: object = None) -> str:
    raise ValueError("unsupported cadgen op: gear")


stub_desugar(raise_exotic)
answer = optimize_mod.optimize_params(WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1)
check("an exotic operation - the capability refuses", answer.get("success") is False)
check("on refusal the original code is returned", answer.get("code") == WRAPPED_CODE)
reason = str(answer.get("error"))
check("the refusal reason reads as one line", "gear" in reason and "\n" not in reason, reason)
check("the full traceback is kept in a separate field",
      "Traceback" in str(answer.get("traceback")))

print("== snapshot guard did not finish the comparison ==")

# The snapshot swallows the failure of its own render silently and leaves iou = nan, while
# returning the unfitted code as the best. On a probe such code did not build at all
# (`StdFail_NotDone` on `revolve`), so `nan` with a CHANGED code we count as a capability
# failure.


class NanGuardOptimizer:
    """The guard fired: the code was fitted and changed, and the IoUs stayed `nan`."""

    @staticmethod
    def optimize(cadquery_code: str, **_kwargs: object) -> FakeResult:
        result = FakeResult(cadquery_code + "\nr=r.faces('>Z').shell(1.0)")
        result.iou_before = float("nan")
        result.iou_after = float("nan")
        return result


class NanUnchangedOptimizer:
    """A legitimate `nan`: the code did not change, the guard had nothing to compare."""

    @staticmethod
    def optimize(cadquery_code: str, **_kwargs: object) -> FakeResult:
        result = FakeResult(cadquery_code)
        result.iou_before = float("nan")
        result.iou_after = float("nan")
        return result


stub_desugar(lambda code, frozen_stl=None: CHAIN_CODE)

# The guard is removed: `nan` is no longer a failure for any code, and the fitness of the
# fitted solution is decided by our pipeline. Previously a changed code with `nan` was
# declared a capability failure, and that discarded most results, none of which we had
# ever measured.
optimize_mod.dsl_runtime.import_optimizer = lambda: NanGuardOptimizer  # type: ignore[assignment]
answer = optimize_mod.optimize_params(WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1)
check("nan with changed code is no longer a refusal",
      answer.get("success") is True, str(answer.get("error")))
check("the tuned code is returned, not replaced by the original",
      answer.get("code") != WRAPPED_CODE, str(answer.get("code"))[:120])
check("and is flagged as changed", answer.get("changed") is True, str(answer.get("changed")))

# The knob's default is part of the decision, not an implementation detail: an enabled
# guard costs two renders and an IoU on tens of thousands of points per call and does not
# complete the comparison. It is checked by signature, so that a return of the default does
# not pass silently.
import inspect  # noqa: E402
check("the guard is off by default",
      inspect.signature(optimize_mod.optimize_params).parameters["safety_guard"].default is False)

class LrEchoOptimizer:
    """Returns the `lr` it received inside the code: the call runs in a fork, otherwise it is not visible."""

    @staticmethod
    def optimize(cadquery_code: str, **kwargs: object) -> FakeResult:
        return FakeResult(cadquery_code + f"\n# lr={kwargs.get('lr')!r}")


optimize_mod.dsl_runtime.import_optimizer = lambda: LrEchoOptimizer  # type: ignore[assignment]
answer = optimize_mod.optimize_params(WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1)
check("the optimizer gets the default lr 1e-2, not the snapshot default",
      "# lr=0.01" in str(answer.get("code")), str(answer.get("code"))[-20:])
answer = optimize_mod.optimize_params(WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1, lr=1e-3)
check("an explicit lr reaches the optimizer", "# lr=0.001" in str(answer.get("code")),
      str(answer.get("code"))[-20:])

optimize_mod.dsl_runtime.import_optimizer = lambda: NanUnchangedOptimizer  # type: ignore[assignment]
answer = optimize_mod.optimize_params(WRAPPED_CODE, gt_mesh_path="/no/such.stl", steps=1)
check("unchanged code is a legitimate outcome", answer.get("success") is True,
      str(answer.get("error")))
check("and is flagged as unchanged", answer.get("changed") is False, str(answer.get("changed")))

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: " + ", ".join(FAILURES))
    sys.exit(1)
print("All wrapped->chain translation checks passed")
