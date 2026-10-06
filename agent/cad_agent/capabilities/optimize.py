"""Optimizer of numeric code parameters against a target.

The `optimize` capability of the contract: the code is already written, the
topology is fixed, and only numbers are fitted. The implementation is third-party:
`vendor/cad_optimizer`, the numerical variant (central differences, C++ backend
`_cad_grad`).

Between the harness and the optimizer sits a **dialect translation**: the
snapshot's parser reads a CadQuery method chain while our models speak wrapped, so
the code is first translated (`capabilities/desugar.py`) and returned from the
optimizer with the dialect prefix glued back. Without translation a call under
wrapped was a guaranteed `Could not unroll CadQuery chain` failure.

Pitfalls:

- `_cad_grad` is built per environment (`agent/tools/build_cad_grad.sh`); the
  snapshot ships no prebuilt binaries, and without it the snapshot cannot be imported.
- the snapshot modules are written as parts of a package, so they are imported
  through `dsl_runtime.import_optimizer()` and not directly by file name.
- it runs in an isolated process: the optimizer runs CadQuery for hundreds of
  iterations, and a hang inside the part's process would stop its rollout.
- the dialect translation costs about one execution (it **executes** the script
  with cadgen operations), so it runs in the same fork under the same timeout and
  is reported separately as `desugar_sec` inside `wall_sec`;
- **the snapshot's safety guard is disabled** (`safety_guard=False`). It rendered
  the original and the fitted code through CadQuery, measured the IoU of each
  against the target and reverted a regression. It never completed a single
  comparison: rendering happens in an empty namespace (`ns = {}`), while we pass a
  chain without `import cadquery as cq` (our translation strips it), and the
  resulting `NameError` was swallowed by a third-party `except Exception` at
  `verbose=0`. It was not fixed: the fitted code is measured by our own pipeline,
  against GT and with our metric, and the best candidate is selected by it. The
  foreign verdict adds nothing here, and its scale (`m.apply_scale(extent / 200.0)`)
  carries its own assumption about units.

A call is expensive and is therefore counted by a separate counter (`n_opt`):
whether to call it and with what step budget is up to the scaffold.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cad_agent import dsl_runtime
from cad_agent.capabilities import desugar as desugar_mod
from cad_agent.capabilities.execute import OUTCOME_TIMEOUT, WORKER_DEATHS, run_in_fork

logger = logging.getLogger(__name__)

DEFAULT_STEPS = 200
# Adam step size. Our own default rather than the snapshot's (1e-3): an Adam step is
# about lr in magnitude, so over 200 steps a literal in a +-100 frame moves no more
# than ~0.15, less than the generator's grid step (0.5). 1e-2 brings thin parts
# further and does not harm the control.
DEFAULT_LR = 1e-2
# A last-resort cap, not the run's cap. The run's cap lives in the harness
# (`execution.opt_timeout_sec`) and is always passed here as an argument:
# `resources.optimize` additionally trims it by the part's remaining wall time, and
# this module knows nothing about the part.
DEFAULT_TIMEOUT = 600.0
# Translate wrapped into a chain before the call. On by default: without
# translation the snapshot's parser does not read our dialect at all. The switch
# remains for diagnostics: it separates "the optimizer failed" from "the
# translation failed".
DEFAULT_DESUGAR = True


# There is one error for the whole optimizer bring-up path and it lives where the
# import happens, in `dsl_runtime`. It is re-exported here so callers need not know
# which module to catch it from.
OptimizerUnavailable = dsl_runtime.OptimizerUnavailable


def _reason(payload: Any) -> str:
    """The last line of the trace, i.e. the error text itself.

    The fork returns `traceback.format_exc()`, while the log writes the first 200
    characters of the event (`resources.optimize`). In those 200 characters a trace
    holds the header and the first stack frame, so the failure reason never reached
    the log. The full trace is returned in a separate `traceback` field.

    With the wrapped->chain translation this matters: "the script contains an exotic
    operation `gear`" is a normal expected outcome, not a breakage, and it can only
    be told from a real optimizer failure by the text.
    """
    lines = [line.strip() for line in str(payload).strip().splitlines() if line.strip()]
    return lines[-1] if lines else str(payload)


def _optimize_impl(
    code: str,
    stl_path: str,
    steps: int,
    optimize_only_last_line: bool,
    safety_guard: bool,
    desugar_enabled: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    # Import as a package: the snapshot modules use relative imports
    # (`from .cq_parser import ...`) and `optimizer_numerical` does not load at all as
    # a top-level module; see dsl_runtime.import_optimizer.
    optimizer = dsl_runtime.import_optimizer()
    _optimize = optimizer.optimize

    # Timing starts before the translation: translation executes the script with real
    # cadgen operations, i.e. costs about one execution, and subtracting it from the
    # call price would be untrue. It goes up as a separate item, `desugar_sec`.
    started = time.monotonic()

    # Translate wrapped -> method chain. The snapshot's parser reads only a chain, so
    # without translation a call under wrapped was a guaranteed failure.
    prepared, desugared, desugar_sec = code, False, 0.0
    if desugar_enabled and desugar_mod.is_wrapped_functional(code):
        # A translation failure is raised as a capability failure: running the
        # optimizer on untranslated wrapped means paying for
        # `Could not unroll CadQuery chain`.
        translated = desugar_mod.to_chain(code)
        prepared, desugared = translated["code"], True
        desugar_sec = float(translated["wall_sec"])

    result = _optimize(
        cadquery_code=prepared,
        stl_path=stl_path,
        steps=steps,
        # Translation preserves "one operation, one line", so "only the last line"
        # means the same after it as before.
        optimize_only_last_line=optimize_only_last_line,
        safety_guard=safety_guard,
        **extra,
    )
    best_code = result.best_code

    # The fitted code goes out as is, whatever it is: validity is decided by our
    # pipeline: the candidate is built, measured against GT and enters selection by our
    # metric. A candidate that does not build is a normal outcome with a diagnosis here,
    # not a tool failure, and costs one execution.
    #
    # The "numbers moved" flag is computed HERE, before the prefix is glued back: after
    # `restore_prefix` the text always differs from the parent (at least in dialect), and
    # the question "did the optimizer do anything" would be indistinguishable from "the
    # translation changed the notation".
    changed = best_code != prepared
    if desugared:
        # The code goes out with the active dialect's prefix: our part code includes the
        # preamble, and without it it would no longer be the same kind of object as the
        # other `Branch.code`.
        best_code = desugar_mod.restore_prefix(best_code)
    return {
        # Time is measured here, inside the fork: in the parent it would include the
        # process start wait, while the capability's cost is work, not queueing. The
        # execution fork does the same.
        "wall_sec": time.monotonic() - started,
        "desugared": desugared,
        "desugar_sec": desugar_sec,
        "code": best_code,
        "changed": changed,
        "final_loss": float(result.final_loss),
        "params": list(result.params),
        "param_names": list(result.param_names),
        "iou_before": float(result.iou_before),
        "iou_after": float(result.iou_after),
        "guard_reverted": bool(result.guard_reverted),
    }


def optimize_params(
    code: str,
    gt_mesh_path: str | Path,
    work_dir: Path | None = None,
    steps: int = DEFAULT_STEPS,
    optimize_only_last_line: bool = False,
    safety_guard: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    desugar: bool = DEFAULT_DESUGAR,
    lr: float = DEFAULT_LR,
    **extra: Any,
) -> dict[str, Any]:
    """Fit the numbers in the code to the target. Returns new code or the failure reason.

    `safety_guard=False` by default: the third-party comparison guard never completed
    a comparison and cost two renders and an IoU over 20k points per call; see the
    module docstring. It would catch a geometry regression correctly, but that is not
    needed here: `optimize` produces a CANDIDATE rather than editing the best prefix in
    place, and a candidate worse than its parent simply does not win selection. The
    knob remains to enable the third-party check when one wants to compare it with ours.

    `desugar=True` by default: wrapped code is translated into a CadQuery method chain
    because the snapshot's parser reads only that (`capabilities/desugar.py`). The
    translation runs in the same fork: it executes cadgen operations and therefore must
    be under the same timeout as the optimizer itself.
    """
    started = time.monotonic()
    ok, payload, outcome = run_in_fork(
        _optimize_impl,
        (code, str(gt_mesh_path), steps, optimize_only_last_line, safety_guard,
         bool(desugar), {**extra, "lr": float(lr)}),
        timeout,
    )
    wall = time.monotonic() - started
    if ok:
        # The fork's measurement is more precise than the parent's, but it is absent on
        # failure, in which case the parent's is used. Same branching as in `execute.py`.
        return {"success": True, **payload, "wall_sec": float(payload.get("wall_sec", wall))}

    timed_out = outcome == OUTCOME_TIMEOUT
    if timed_out:
        pass
    elif outcome in WORKER_DEATHS:
        logger.warning("Parameter optimizer died (%s): %s", outcome, payload)
    return {
        "success": False, "code": code, "error": _reason(payload),
        "traceback": str(payload),
        "timed_out": timed_out, "outcome": outcome, "wall_sec": wall,
    }
