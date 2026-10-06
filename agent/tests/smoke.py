#!/usr/bin/env python3
"""Local package check without CAD, GPU or servers.

Run: ``python agent/tests/smoke.py`` from the repository root.

What is checked: importing the whole package on stubs, isolation and caps of the
deterministic branch, optimizer caps, separation of executor results, strict config
validation, the semantics of the selection objective and its fallback, GT derivatives,
the transport to the model, coordinate collapsing and the prompt cap.

What is NOT here: the search loop and policies; they are checked in `search_check.py`,
which runs them on the real `SearchLoop`. The split is by subject, not by depth:
duplicating the loop on second stubs would give two descriptions of one mechanism that
drift apart.

This does not replace a run on a server: there is no OCC and no models here.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import types
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

# Stubs for what does not exist outside the server environment.
if "openai" not in sys.modules:
    stub = types.ModuleType("openai")
    stub.OpenAI = object
    sys.modules["openai"] = stub

from cad_agent import dsl_runtime  # noqa: E402

dsl_runtime.configure("wrapped")

# Importing the whole package is a separate check: modules must not break from a
# reordering of dependencies. A list of names rather than `import ... as`, so that module
# names do not collide with the stub names below.
import importlib  # noqa: E402

PACKAGE_MODULES = [
    "cad_agent.capabilities.code",
    "cad_agent.capabilities.desugar", "cad_agent.capabilities.det",
    "cad_agent.capabilities.difficulty",
    "cad_agent.capabilities.execute",
    "cad_agent.capabilities.llm", "cad_agent.capabilities.metrics",
    "cad_agent.capabilities.optimize", "cad_agent.capabilities.propose",
    "cad_agent.capabilities.render", "cad_agent.capabilities.resources",
    "cad_agent.harness.artifacts", "cad_agent.harness.budget",
    "cad_agent.harness.dataset", "cad_agent.harness.figure_run",
    "cad_agent.harness.journal", "cad_agent.harness.pool",
    "cad_agent.harness.run_eval", "cad_agent.harness.subsample",
]
for _name in PACKAGE_MODULES:
    importlib.import_module(_name)

from cad_agent.capabilities import objective as objective_mod  # noqa: E402
from cad_agent.capabilities import resources  # noqa: E402
from cad_agent.harness import budget, dataset  # noqa: E402
from cad_agent.scaffold import base  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


class FakeProposal:
    def __init__(self, code_text: str):
        self.error = None
        self.step_code = code_text
        self.full_code = code_text
        self.point = "(0,0,0)"


class FakeEval:
    def __init__(self, cd, mesh_path="/tmp/fake.stl", success=True, error=None, **extra_metrics):
        self.success = success
        self.error = error
        self.mesh_path = mesh_path if success else None
        self.metrics = {"cd_runtime": cd} if cd is not None else {}
        # Objective metrics are stored the same way as the real `measure_pair` does it:
        # the stub must repeat the interface, otherwise the check passes where a run
        # fails (this has already happened three times).
        self.metrics.update({k: v for k, v in extra_metrics.items() if v is not None})


class FakeResources:
    """Fake capabilities: return a predefined sequence of CD values."""

    def __init__(self, cd_sequence, agent_answer="B", budget_obj=None,
                 quality_sequence=None, gt_watertight=True):
        self.cd_sequence = list(cd_sequence)
        # Quality metric values (iou/gms_norm) per candidate, for objective-mode checks.
        # Empty means only CD is computed, as before.
        self.quality_sequence = list(quality_sequence or [])
        self.gt_watertight = gt_watertight
        self.needs_seen: list[tuple] = []
        self.agent_answer = agent_answer
        self.calls = {"propose": 0, "evaluate": 0, "agent": 0, "det": 0, "optimize": 0, "notes": []}
        self.notes_full: list[tuple] = []
        self.mesh_names: list[str] = []
        self.step_tags: list[tuple] = []
        # How many variants were requested at each step; input for re-sampling checks.
        self.k_requested: list[tuple] = []
        # Which draw variant was requested; the policy's only seeding knob.
        self.variants_requested: list[tuple] = []
        self.det_k_requested: list[int] = []
        self.budget = budget_obj or budget.Budget()
        # The harness's draw base. In the real `Resources` the harness sets it; the stub
        # must have it too, otherwise a beam with `seed: null` fails here and not in a run
        # (a stub is part of the interface).
        self.seed = 12345
        self.work_dir = Path("/tmp")
        self.render = types.SimpleNamespace(selection_image=lambda **kwargs: "IMAGE")

    def note(self, decision, **fields):
        self.calls.setdefault("notes", [])
        self.calls["notes"].append(decision) if isinstance(self.calls.get("notes"), list) else None
        # The decision fields are needed separately: the selection threshold is visible
        # only in them, and a mismatch of the threshold with the objective was a defect.
        self.notes_full.append((decision, fields))

    def propose_steps(self, pred_mesh_path, prev_code, k, step=None, variant=0, tag=""):
        # The signature repeats the real one: `seed` is no longer passed to the harness,
        # the seed is computed by the harness and the policy keeps only `variant`.
        self.calls["propose"] += 1
        self.step_tags.append((step, tag))
        self.variants_requested.append((step, int(variant)))
        self.k_requested.append((step, k))
        self.budget.spend("vlm", k)
        return [FakeProposal(f"{prev_code}\nstep{self.calls['propose']}_{i}") for i in range(k)]

    def difficulty(self):
        # None means the features were not computed (unreadable mesh). Return exactly what
        # the real `shape_features` returns in this case: a dict with an error and WITHOUT
        # the `gt_watertight` key.
        if self.gt_watertight is None:
            return {"error": "mesh is not readable"}
        return {"gt_watertight": self.gt_watertight}

    def evaluate_codes(self, codes, gt_mesh_path, name_prefix, measure=True, step=None, needs=()):
        self.calls["evaluate"] += 1
        self.needs_seen.append(tuple(needs))
        # Mesh paths are built from name_prefix and the index, so a name collision means
        # candidates overwrite each other's geometry.
        self.mesh_names.extend(f"{name_prefix}_{idx}" for idx in range(len(codes)))
        # The stub MUST honor `needs`: the real capability computes only what was
        # requested, and "no metric" there means "not requested". While the stub returned
        # everything, the `resolve` guard checks went green even on broken code: the stub
        # lagged behind the interface.
        field_of = {objective_mod.NEED_IOU: "iou", objective_mod.NEED_GMS: "gms_norm"}
        allowed = {field for need, field in field_of.items() if need in needs}
        out = []
        for _ in codes:
            cd = self.cd_sequence.pop(0) if self.cd_sequence else None
            quality = self.quality_sequence.pop(0) if self.quality_sequence else None
            if cd is None and quality is None:
                out.append(FakeEval(None, success=False, error="did not build"))
                continue
            filtered = {
                key: value for key, value in (quality or {}).items()
                if key not in field_of.values() or key in allowed
            }
            out.append(FakeEval(cd, **filtered))
        return out

    def ask_agent(self, prompt, image=None, max_tokens=16, temperature=0.0):
        self.calls["agent"] += 1
        self.last_prompt = prompt
        return self.agent_answer

    def algo_rebuild(self, pred_mesh_path, k):
        self.calls["det"] += 1
        self.det_k_requested.append(k)
        return [f"det_op_{i}" for i in range(k)]

    def optimize(self, code_text, **kwargs):
        self.calls["optimize"] += 1
        return {"success": True, "code": code_text + "\noptimized"}


def observation() -> base.Observation:
    return base.Observation(
        figure_id="test/fig",
        gt_mesh_path=Path("/tmp/gt.stl"),
        work_dir=Path("/tmp"),
        prefix_code="PREFIX",
    )


print("1. Package import")
check("all modules imported", len(PACKAGE_MODULES) == 19, f"modules: {len(PACKAGE_MODULES)}")

print("7. The set signature is reproducible")
figures = [
    dataset.FigureSpec(figure_id="g/a", gt_mesh_path=Path("/tmp/a.stl"), group="g"),
    dataset.FigureSpec(figure_id="g/b", gt_mesh_path=Path("/tmp/b.stl"), group="g"),
]
check("the signature is stable", dataset.dataset_signature(figures) == dataset.dataset_signature(figures))
check("order affects the signature", dataset.dataset_signature(figures) != dataset.dataset_signature(figures[::-1]))

print("8. The deterministic branch is isolated and does not hang the part process")
import time as _time  # noqa: E402

from cad_agent.capabilities import det as det_mod  # noqa: E402

# The GT skeleton warm-up runs in a fork (otherwise its failure takes down the whole
# part), so an in-memory counter in the parent no longer sees it: the increment happens in
# the child and dies with it. It is counted by what survives the fork: files.
_warm_dir = Path(tempfile.mkdtemp(prefix="smoke_warm_"))


def _fake_warm(gt, cache):
    (_warm_dir / f"{len(list(_warm_dir.iterdir()))}").write_text("1")


def warm_count() -> int:
    return len(list(_warm_dir.iterdir()))


det_mod.warm_gt_frame = _fake_warm


def build_det_resources(config=None, figure_budget=None):
    return resources.build_resources(
        figure_id="test/fig",
        gt_mesh_path=Path("/tmp/gt.stl"),
        work_dir=Path("/tmp"),
        executor=None,
        renderer=None,
        proposer=None,
        budget=figure_budget or budget.Budget(),
        config=config or {},
    )


det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {
    "success": True, "ops": [f"op{i}" for i in range(5)],
}
res = build_det_resources()
ops = res.algo_rebuild(None, 3)
check("operations crossed the process boundary", ops == ["op0", "op1", "op2"], str(ops))

# The pool is computed once and handed out by a cursor. This checks exactly what the
# slice was taken for: a repeat from the same parent gives the NEXT operations, not a copy
# of the top of the list. While `kept[:k]` cut inside, re-sampling in the deterministic
# branch bought a copy for a full call.
again = res.algo_rebuild(None, 2)
check("a repeat returns the next slice, not a copy", again == ["op3", "op4"], str(again))
check("the GT cache was warmed exactly once", warm_count() == 1, f"warm-ups: {warm_count()}")
check("the repeat did not charge a det call", res.budget.counts["det"] == 1, str(res.budget.counts["det"]))
check("the pool remainder is visible from outside", res.det_remaining(None) == 0, str(res.det_remaining(None)))
check("an exhausted pool answers with an empty list", res.algo_rebuild(None, 1) == [])
check("an exhausted pool did not charge a call", res.budget.counts["det"] == 1)

# Another parent has its own pool and its own real call: the saving applies to a repeat,
# not to the deterministic branch in general.
check("the remainder is unknown before the first call", res.det_remaining("/tmp/other.stl") is None)
other = res.algo_rebuild("/tmp/other.stl", 2)
check("another parent is computed afresh", other == ["op0", "op1"], str(other))
check("and it charges its call", res.budget.counts["det"] == 2, str(res.budget.counts["det"]))

# A failed call does NOT create a pool: an empty list from a crashed fork means "we do
# not know", not "nothing to propose". Remembering it as a pool would close the
# deterministic branch for this parent forever over one failure.
det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {
    "success": False, "error": "detectors failed",
}
res_fail = build_det_resources()
check("the failure returned nothing", res_fail.algo_rebuild(None, 2) == [])
check("the failure did not start a pool", res_fail.det_remaining(None) is None)
det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {
    "success": True, "ops": ["op0"],
}
check("after a failure the call goes again", res_fail.algo_rebuild(None, 1) == ["op0"])
check("and it is charged", res_fail.budget.counts["det"] == 2, str(res_fail.budget.counts["det"]))

# Warm-up is the only place of the deterministic branch that used to run in the part's
# process. It once took down a rollout and the whole run with it, so what is checked is not
# "warm-up is called" but "the part survives its failure and its hang".
det_mod.warm_gt_frame = lambda gt, cache: (_ for _ in ()).throw(RuntimeError("the GT skeleton was not built"))
det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {"success": True, "ops": ["op0"]}
check("a warm-up crash does not fail the part", build_det_resources().algo_rebuild(None, 1) == ["op0"])

det_mod.warm_gt_frame = lambda gt, cache: _time.sleep(30)
started = _time.monotonic()
ops = build_det_resources({"execution": {"det_timeout_sec": 1.0}}).algo_rebuild(None, 1)
check("a hung warm-up was interrupted by timeout", _time.monotonic() - started < 5,
      f"{_time.monotonic() - started:.1f} s")
det_mod.warm_gt_frame = _fake_warm

det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: _time.sleep(30)
started = _time.monotonic()
ops = build_det_resources({"execution": {"det_timeout_sec": 1.0}}).algo_rebuild(None, 2)
elapsed = _time.monotonic() - started
check("a hung det was interrupted by timeout", ops == [] and elapsed < 5, f"{elapsed:.1f} s, returned {ops}")

det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {"success": False, "error": "detectors failed"}
check("a det failure does not fail the part", build_det_resources().algo_rebuild(None, 2) == [])

# A parent that ate the timeout is not computed a second time. The algorithm is
# deterministic: the same GT and the same mesh give the same work, and a repeat would buy
# the same empty answer for the same seconds. What is checked is not "it returned empty"
# (it was empty the first time too) but that the second call did NOT go to a fork: by wall
# time and by the untouched seconds of the part's budget.
det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: _time.sleep(30)
res = build_det_resources({"execution": {"det_timeout_sec": 1.0, "det_budget_sec": 100.0}})
check("the first call hits the timeout", res.algo_rebuild("/tmp/t1.stl", 1) == [])
spent_calls = res.budget.counts["det"]
started = _time.monotonic()
repeat = res.algo_rebuild("/tmp/t1.stl", 1)
elapsed = _time.monotonic() - started
check("a repeat after a timeout does not go to the fork", repeat == [] and elapsed < 0.5, f"{elapsed:.2f} s")
check("a repeat after a timeout did not charge a call", res.budget.counts["det"] == spent_calls,
      f"{res.budget.counts['det']} vs {spent_calls}")
# Zero, not None: the search loop removes the action from the legal ones exactly by zero
# (`SearchState.det_remaining`), while None means "the pool was not computed, the call
# will be real", i.e. permission to repeat.
check("a timeout is visible from outside as an empty remainder", res.det_remaining("/tmp/t1.stl") == 0,
      str(res.det_remaining("/tmp/t1.stl")))
# The cache is per parent: a parent is closed, not the deterministic branch for the part.
started = _time.monotonic()
res.algo_rebuild("/tmp/t2.stl", 1)
check("a timeout closes the parent, not the branch for the part", _time.monotonic() - started >= 0.9,
      f"{_time.monotonic() - started:.2f} s")

# A fork failure does NOT create a cache entry: a process death is a property of the
# machine and the moment, not of the input. Closing the branch forever on it would lose the
# parent over one failure, exactly what the pool not being created empty is for.
det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {"success": False, "error": "detectors failed"}
res = build_det_resources({"execution": {"det_timeout_sec": 10.0}})
check("a det failure is not cached", res.algo_rebuild("/tmp/f1.stl", 1) == []
      and res.det_remaining("/tmp/f1.stl") is None, str(res.det_remaining("/tmp/f1.stl")))
check("a repeat after a failure is charged again", res.algo_rebuild("/tmp/f1.stl", 1) == []
      and res.budget.counts["det"] == 2, str(res.budget.counts["det"]))

# A cap on the PART, not the call. A call timeout does not save from a slow but honestly
# working det: the harness calls it at every step, and three parts once sat in it for an
# hour each without ever violating the timeout. This checks that an exhausted budget
# answers with a refusal and costs neither a call nor a counter.
def _slow_det(gt, pred, cache, timeout=None, max_faces=None):
    _time.sleep(0.6)
    return {"success": True, "ops": ["op0"]}


det_mod.det_rebuild_isolated = _slow_det
res = build_det_resources({"execution": {"det_budget_sec": 1.0, "det_timeout_sec": 10.0}})
# The parents are DIFFERENT, and this matters: a repeat from one parent is served from the
# pool, does not go to a fork and does not touch the part cap. Checking the cap on repeats
# would check the cache.
check("the first call goes through", res.algo_rebuild("/tmp/p1.stl", 1) == ["op0"])
check("the second call goes through", res.algo_rebuild("/tmp/p2.stl", 1) == ["op0"])
spent_calls = res.budget.counts["det"]
check("the third call is refused by the part budget", res.algo_rebuild("/tmp/p3.stl", 1) == [])
check("the failure did not charge a call", res.budget.counts["det"] == spent_calls,
      f"{res.budget.counts['det']} vs {spent_calls}")
check("the budget is per part, not per run",
      build_det_resources({"execution": {"det_budget_sec": 1.0}}).algo_rebuild(None, 1) == ["op0"])

# The part's WALL cap is the third, outermost one, and it is asked here too. It must be
# checked on `algo_rebuild`, not only on `propose_steps`: a step with a `det` source does
# not call the generator at all, and both parts that failed to finish in a past run stood
# exactly on such a step.
det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {"success": True, "ops": ["op0"]}
expired = budget.Budget(wall_sec=1.0, started=_time.monotonic() - 5.0)
try:
    build_det_resources(figure_budget=expired).algo_rebuild(None, 1)
    check("an exhausted part wall time stops det", False, "the call went through")
except budget.DeadlineExceeded as exc:
    check("an exhausted part wall time stops det", True)
    check("and the stop point is named", "algo_rebuild" in str(exc), str(exc))
check("a wall-time stop did not charge a det call", expired.counts["det"] == 0,
      str(expired.counts["det"]))
# The difference from an exhausted `det_budget_sec`: that one answers with an empty list
# (the part goes on without det), while the part's wall stops the rollout entirely.
# Merging them into one outcome would give a part that keeps spinning without the only item
# for which it was stopped.
alive = budget.Budget(wall_sec=1000.0)
check("a wall time that has not expired does not hinder det",
      build_det_resources(figure_budget=alive).algo_rebuild(None, 1) == ["op0"])
det_mod.det_rebuild_isolated = _slow_det

# Call arguments are read from files rather than from an in-memory list: det lives in a
# fork, and whatever the stub stored in a variable dies with it. The warm-up counter above
# failed on exactly this.
_probe_dir = Path(tempfile.mkdtemp(prefix="smoke_det_probe_"))


def _probe(gt, pred, cache, timeout=None, max_faces=None):
    (_probe_dir / f"{len(list(_probe_dir.iterdir()))}").write_text(
        f"{timeout}\n{max_faces}", encoding="utf-8"
    )
    _time.sleep(0.6)
    return {"success": True, "ops": ["op0"]}


def _probed(field: int) -> list:
    rows = sorted(_probe_dir.iterdir(), key=lambda path: int(path.name))
    return [row.read_text(encoding="utf-8").split("\n")[field] for row in rows]


def _probe_reset() -> None:
    for row in _probe_dir.iterdir():
        row.unlink()


# The remaining budget trims the call timeout: otherwise the last call would go for the
# full timeout on top of the cap, and "10 minutes per part" would mean more.
det_mod.det_rebuild_isolated = _probe
_probe_reset()
# The budget here is noticeably more than a second: trimming has a 1 s floor (passing a
# non-positive timeout to a fork is pointless), and with a one-second budget the floor
# would be checked, not the trimming.
res = build_det_resources({"execution": {"det_budget_sec": 5.0, "det_timeout_sec": 100.0}})
res.algo_rebuild("/tmp/p1.stl", 1)
res.algo_rebuild("/tmp/p2.stl", 1)
timeouts = [float(value) for value in _probed(0)]
check("the call timeout is cut to the budget remainder",
      len(timeouts) == 2 and timeouts[0] <= 5.0 and timeouts[1] < timeouts[0], str(timeouts))

# The decimation threshold is a run knob: without plumbing it would stay a constant in
# `det.py` and the config would silently change nothing.
_probe_reset()
build_det_resources({"execution": {"det_residual_faces": 700}}).algo_rebuild(None, 1)
check("the residual face threshold came from the config", _probed(1) == ["700"], str(_probed(1)))
_probe_reset()
build_det_resources().algo_rebuild(None, 1)
check("without a config the default det.RESIDUAL_MAX_FACES is used",
      _probed(1) == [str(det_mod.RESIDUAL_MAX_FACES)], str(_probed(1)))

print("8a. Optimizer call cap and part wall time")

from cad_agent.capabilities import optimize as optimize_mod  # noqa: E402

# The stub lives in the part's process, not in a fork: the fork here is inside
# `optimize_params`, and we replace it whole. So the arguments can be read from an
# in-memory list, unlike det, where they have to be stored in files.
_opt_calls: list[dict] = []


def _fake_optimize_params(code, gt_mesh_path, work_dir=None, **kwargs):
    _opt_calls.append(dict(kwargs))
    return {"success": True, "code": (code or "") + "\n# tuned", "wall_sec": 0.0}


optimize_mod.optimize_params = _fake_optimize_params


def build_opt_resources(execution=None, figure_budget=None):
    # `optimize` is a tool on explicit request (`opt_in`), and without a list in the
    # config the capability answers with a refusal without looking at anything.
    config = {"tools": ["stepwise", "optimize"]}
    if execution is not None:
        config["execution"] = execution
    return build_det_resources(config, figure_budget)


_opt_calls.clear()
build_opt_resources().optimize("box(1)")
check("the optimizer call uses the harness cap, not the module cap",
      len(_opt_calls) == 1 and _opt_calls[0].get("timeout") == resources.DEFAULT_OPT_TIMEOUT,
      str(_opt_calls))

_opt_calls.clear()
build_opt_resources({"opt_timeout_sec": 45}).optimize("box(1)")
check("the call cap came from the config", _opt_calls[0].get("timeout") == 45.0, str(_opt_calls))

# An explicit `null` means "there is no run cap" and must differ from an absent key: the
# default protects, and the protection can be removed only out loud.
_opt_calls.clear()
build_opt_resources({"opt_timeout_sec": None}).optimize("box(1)")
check("an explicit null removes the run cap, leaving the module timeout",
      _opt_calls[0].get("timeout") == optimize_mod.DEFAULT_TIMEOUT, str(_opt_calls))

# The part's wall is SOFT: it decides whether to start a new unit of work and does not
# shorten one already started. A part with seconds left gets a full call and may exceed the
# cap on it: trimming would give it a stub that surely cannot finish, and the call would be
# wasted. This is exactly what is checked: the remaining wall does not affect the timeout,
# and the size of the overshoot is limited by the call cap itself.
_opt_calls.clear()
almost_out = budget.Budget(wall_sec=100.0, started=_time.monotonic() - 95.0)
build_opt_resources(figure_budget=almost_out).optimize("box(1)")
check("a nearly exhausted wall time does not shorten a started call",
      _opt_calls[0].get("timeout") == resources.DEFAULT_OPT_TIMEOUT, str(_opt_calls))

_opt_calls.clear()
roomy = budget.Budget(wall_sec=10_000.0)
build_opt_resources(figure_budget=roomy).optimize("box(1)")
check("and a roomy wall time does not change the call cap either",
      _opt_calls[0].get("timeout") == resources.DEFAULT_OPT_TIMEOUT, str(_opt_calls))

# An exhausted wall stops the rollout rather than answering with a capability refusal, as
# with det. And it does not charge the call: an uncalled capability must not cost, or
# `n_opt` in the report counts what did not happen.
_opt_calls.clear()
expired_opt = budget.Budget(wall_sec=1.0, started=_time.monotonic() - 5.0)
try:
    build_opt_resources(figure_budget=expired_opt).optimize("box(1)")
    check("an exhausted part wall time stops the optimizer", False, "the call went through")
except budget.DeadlineExceeded as exc:
    check("an exhausted part wall time stops the optimizer", True)
    check("and the stop point is named", "optimize" in str(exc), str(exc))
check("a wall-time stop did not reach the optimizer", _opt_calls == [], str(_opt_calls))
check("a wall-time stop did not charge an opt call", expired_opt.counts["opt"] == 0,
      str(expired_opt.counts["opt"]))

print("12. The executor does not mix up results for identical task names")


execute_mod = importlib.import_module("cad_agent.capabilities.execute")
execute_mod._preload_cad = lambda: None
execute_mod._evaluate = lambda task: {
    "mesh_path": task.mesh_path,
    "metrics": {"cd_runtime": task.extra["cd"]},
    "wall_sec": 0.0,
}
duplicate_tasks = [
    execute_mod.EvalTask(task_id="same name", code="", mesh_path=f"/tmp/d{i}.stl", extra={"cd": 0.1 * i})
    for i in range(4)
]
for backend, kwargs in (("serial_fork", {}), ("ephemeral_pool", {"pool_size": 2}), ("proxy_pool", {"pool_size": 2})):
    executor = execute_mod.build_executor(backend=backend, timeout=5.0, **kwargs)
    try:
        results = executor.evaluate(duplicate_tasks)
    finally:
        executor.close()
    got = [r.metrics["cd_runtime"] for r in results]
    check(f"{backend}: results are not mixed up", got == [0.0, 0.1, 0.2, 0.30000000000000004], str(got))

print("13. Strict config validation")
config_mod = importlib.import_module("cad_agent.harness.config")


# A server on which `dialogue_lean` is legal: the assistant is up, function calls are
# parsed by the parser, reasoning is explicitly off (`agent.thinking: false` below).
# Parser keys are derived from `launch` (`run_experiment._server_section`); here they are
# set by hand, which is how the validator sees them.
LEAN_SERVER = {"generation_base_url": "http://x", "generation_served_model_name": "gen",
               "assistant_base_url": "http://a", "assistant_served_model_name": "asst",
               "assistant_enabled": True, "assistant_tool_call_parser": "qwen3_coder",
               "assistant_reasoning_parser": None}


def good_config(**overrides):
    base = {
        "details": [{"g": "/tmp"}],
        "n_workers": 2,
        "server": dict(LEAN_SERVER),
        "agent": {"thinking": False},
        "scaffold": {"kind": "policy", "policy": "dialogue_lean"},
        "execution": {"backend": "serial_fork", "pool_size": 1, "timeout_sec": 30},
        "logging": {"level": "metrics", "profile": False},
        "budget": {"vlm": 100, "total": None},
    }
    base.update(overrides)
    return base


def rejects(name, **overrides):
    try:
        config_mod.validate_run_config(good_config(**overrides))
        check(name, False, "config accepted although it should not be")
    except config_mod.ConfigError:
        check(name, True)


config_mod.validate_run_config(good_config())
check("a valid config is accepted", True)
rejects("typo in a section key", execution={"backend": "serial_fork", "poolsize": 4})
rejects("unknown execution backend", execution={"backend": "threads"})
rejects("unknown logging level", logging={"level": "verbose"})
# Policy knobs do not belong in the config: they belong to the policy, and an extra key
# next to the name is a second source of truth, not a trifle.
rejects("a policy knob in the config is rejected", scaffold={"kind": "policy", "policy": "dialogue_lean", "k_variants": 3})
rejects("the policy name is required", scaffold={"kind": "policy"})
rejects("a nonexistent policy is rejected", scaffold={"kind": "policy", "policy": "no-such-policy"})
rejects("a removed harness kind is rejected", scaffold={"kind": "baseline"})
rejects("no set given at all", details=None)
rejects("set given twice", subsample={"manifest": "/tmp/m.json"})
rejects("zero workers", n_workers=0)
rejects("boolean instead of a number", n_workers=True)
rejects("negative budget", budget={"vlm": -1})
rejects("unknown call kind in the budget", budget={"vlm_visual": 10})
rejects("assistant configured only halfway",
        server={"generation_base_url": "http://x", "generation_served_model_name": "gen",
                "assistant_base_url": "http://y"})

# Whether there will be an assistant is decided not by the address but by whether the
# server is started, and a policy that calls it must fail at start when the assistant is
# disabled, not return an assistant refusal on every part.
ASSISTANT_OFF = {**LEAN_SERVER, "assistant_enabled": False}
rejects("dialogue_lean with the assistant disabled", server=ASSISTANT_OFF)
rejects("agent limits with the assistant disabled",
        agent={"thinking": False, "context_limit": 49152}, server=ASSISTANT_OFF)

# The same protection for the GENERATOR. Added later than the assistant's and for the
# opposite reason: the assistant had it, but the harness knew nothing about a disabled
# generator, so the run went to the end answering with a refusal on EVERY part. What is
# asked is not the tool name but its counter (`vlm`): a list of names here would be a
# second copy of the registry.
GENERATION_OFF = {**LEAN_SERVER, "generation_enabled": False}
rejects("a set with the generator while the generator is disabled", server=GENERATION_OFF)
try:
    config_mod.validate_run_config(good_config(server=GENERATION_OFF))
except config_mod.ConfigError as exc:
    check("and the error names the tool, not the server", "stepwise" in str(exc), str(exc))
# The deterministic branch does not call the generator, so such a run is legal.
config_mod.validate_run_config(good_config(server=GENERATION_OFF, tools=["det_cold", "det_warm"]))
check("a det set with the generator disabled is accepted", True)

# The run no longer has a sampling knob: the generation mode belongs to the policy. The
# old keys must make the config FAIL rather than silently do nothing: `greedy: true` looks
# like a working setting but would mean a run where the policy's `n` and temperature are
# silently overridden.
for dead in ("greedy", "temperature", "top_p", "top_k"):
    rejects(f"removed key generation.{dead}", generation={dead: 1})

# Every registry name must pass config validation. Cheap, and catches exactly the class
# of mistake where a temperature pin was set for a whole family while one policy asks for
# two variants: the name started being rejected by its own guard, and there was no way to
# notice because the name has no config.
from cad_agent.scaffold.policies import POLICY_NAMES as _NAMES

_broken = []
for _name in _NAMES:
    try:
        config_mod.validate_run_config(good_config(
            scaffold={"kind": "policy", "policy": _name}))
    except config_mod.ConfigError as _exc:
        _broken.append(f"{_name}: {_exc}")
check("every registry name passes config validation", not _broken, "; ".join(_broken)[:160])

# A policy with function calling and a server without a parser: vLLM would reject every
# question. Parser keys are derived from `launch` (`run_experiment._server_section`); here
# they are set by hand, which is how the validator sees them.
_tool_server = {k: v for k, v in LEAN_SERVER.items()
                if k not in ("assistant_tool_call_parser", "assistant_reasoning_parser")}
_tools_scaffold = {"kind": "policy", "policy": "dialogue_lean"}
rejects("function calling with a server without a call parser",
        scaffold=_tools_scaffold, agent={"thinking": False},
        server={**_tool_server, "assistant_tool_call_parser": None,
                "assistant_reasoning_parser": None})
rejects("function calling with possible reasoning and no reasoning-parser",
        scaffold=_tools_scaffold, agent={},
        server={**_tool_server, "assistant_tool_call_parser": "qwen3_coder",
                "assistant_reasoning_parser": None})
config_mod.validate_run_config(good_config(
    scaffold=_tools_scaffold, agent={"thinking": False},
    server={**_tool_server, "assistant_tool_call_parser": "qwen3_coder",
            "assistant_reasoning_parser": None}))
check("function calling with a parser on a non-thinking line is accepted", True)

# Prefix-cache warm-up with a server without a cache is an extra request on every turn.
rejects("prewarm with a server without a prefix cache",
        agent={"thinking": False, "prewarm": True},
        server={**LEAN_SERVER, "assistant_prefix_caching": None})
config_mod.validate_run_config(good_config(
    agent={"thinking": False, "prewarm": True},
    server={**LEAN_SERVER, "assistant_prefix_caching": True}))
check("warm-up with the prefix cache enabled is accepted", True)
from cad_agent.launch_plan import server_cache_setup as _cache_setup

_setup = _cache_setup("assistant", {"launch": {"servers": {"assistant": {"args": {
    "data-parallel-size": 3, "enable-prefix-caching": True}}}}})
check("assistant replicas and cache are derived from launch",
      _setup == {"data_parallel_size": 3, "prefix_caching": True}, str(_setup))
_setup = _cache_setup("assistant", {"launch": {"servers": {"assistant": {"args": {
    "no-enable-prefix-caching": True}}}}})
check("cache disabled and one replica", _setup == {"data_parallel_size": None,
                                                    "prefix_caching": False}, str(_setup))

rejects("candidates with the log disabled", logging={"level": "off", "candidates": True})

# The progress bar. What is checked is the OUTCOME of its decision, not the presence of
# a key: it must stay silent without a terminal and be able to not exist at all, otherwise
# a run environment without tqdm would crash the run for the sake of decoration.
from cad_agent.harness import progress as progress_mod  # noqa: E402

config_mod.validate_run_config(good_config(logging={"level": "metrics", "progress": True}))
check("the logging.progress key is accepted", True)
rejects("a non-boolean in logging.progress", logging={"level": "metrics", "progress": "yes"})
check("an explicit false overrides the terminal", progress_mod.enabled({"progress": False}) is False)
check("an explicit true overrides the lack of a terminal", progress_mod.enabled({"progress": True}) is True)
check("null without a terminal — no bar", progress_mod.enabled({}) is False)
# The dummy must accept the same calls as the real bar: the calling code always calls
# `advance` and never asks whether there is a bar.
_silent = progress_mod.build(0, {"progress": True})
_silent.advance({"score": 0.5, "metrics": {}})
_silent.advance(None)
_silent.close()
_silent.close()
check("the dummy survives advance and a double close", True)
with progress_mod.build(3, {"progress": False}) as _off:
    _off.advance({"score": 0.1, "metrics": {}})
check("a disabled bar is a context manager", True)


class _FakeBar:
    """A bar without a terminal: records what it was told to show."""

    def __init__(self):
        self.posts: list[str] = []
        self.updates = 0

    def set_postfix_str(self, text, refresh=False):
        self.posts.append(text)

    def update(self, n):
        self.updates += n

    def close(self):
        pass


def _postfix(records):
    bar = _FakeBar()
    shown = progress_mod._TqdmProgress(bar)
    for record in records:
        shown.advance(record)
    return bar.posts[-1], bar.updates


# The mean is computed over those that gave a score, NOT by subtracting failures from
# the total: a part with no record at all is neither a failure nor a valid one, and with
# subtraction it would silently lower the mean (caught by hand before this check).
_good = [{"score": 0.8, "metrics": {}}, {"score": 0.6, "metrics": {}},
         {"score": None, "metrics": None}, {"score": 0.9, "metrics": {}}]
check("the bar averages over valid parts", _postfix(_good)[0] == "score 0.767, failures 1",
      _postfix(_good)[0])
check("a part without a record does not lower the mean",
      _postfix([None, {"score": 0.5, "metrics": {}}])[0] == "score 0.500, failures 0",
      _postfix([None, {"score": 0.5, "metrics": {}}])[0])
check("all-failure input does not divide by zero",
      _postfix([{"score": None, "metrics": None}] * 3)[0] == "score 0.000, failures 3",
      _postfix([{"score": None, "metrics": None}] * 3)[0])
check("each part advances the bar by exactly one", _postfix(_good)[1] == len(_good),
      str(_postfix(_good)[1]))


class _AngryBar(_FakeBar):
    """A bar that breaks on its very first show."""

    def update(self, n):
        raise RuntimeError("stream closed")


_angry = progress_mod._TqdmProgress(_AngryBar())
_angry.advance({"score": 0.5, "metrics": {}})
_angry.advance({"score": 0.5, "metrics": {}})
_angry.close()
check("a broken bar does not fail the run", True)
rejects("zero agent context cap", agent={"context_limit": 0},
        server={"generation_base_url": "http://x", "generation_served_model_name": "gen",
                "assistant_base_url": "http://y", "assistant_served_model_name": "a"})
rejects("agent limits without the agent itself", agent={"context_limit": 49152})

# The tool set. Both a typo and the fact that a removed section fails the config are
# checked: `capabilities: {optimize: true}` looks like a working setting but after the move
# to `tools` means nothing.
rejects("a tool not in the registry", tools=["stepwise", "det"])
rejects("empty tool set", tools=[])
config_mod.validate_run_config(good_config(tools=["stepwise"]))
check("a one-tool set is accepted", True)
config_mod.validate_run_config(good_config())
check("the default set (no key) is accepted", True)

check("the default set does not include optimize",
      "optimize" not in config_mod.run_tools({}), str(config_mod.run_tools({})))
check("the config set resolves in registry order",
      config_mod.run_tools({"tools": ["optimize", "stepwise"]}) == ("stepwise", "optimize"),
      str(config_mod.run_tools({"tools": ["optimize", "stepwise"]})))

# A removed section must fail the config BEFORE assembly: `build_run_config` builds from an
# enumeration of known keys, and it would never have reached the validator.
try:
    run_experiment_capabilities = importlib.import_module("run_experiment")
    run_experiment_capabilities.build_run_config(
        {"dsl": "wrapped",
         "server": {"generation_base_url": "http://x", "generation_served_model_name": "gen"},
         "experiment": {"capabilities": {"optimize": True}, "details": ["a"]}},
        None, None)
except Exception as exc:
    check("a removed capabilities section fails config assembly",
          "tools" in str(exc), str(exc)[:120])
else:
    check("a removed capabilities section fails config assembly", False, "passed silently")

# Config sections must reach the run. What is checked is the OUTCOME (the value is visible
# in the run config), not that the key is legal: strict validation of `experiment.metrics`
# passed before too, but it did not reach the runtime.
run_experiment_mod = importlib.import_module("run_experiment")
raw_config = {
    "dsl": "wrapped",
    "server": {"generation_base_url": "http://x", "generation_served_model_name": "gen"},
    "experiment": {
        "metrics": {"cd": True},
        "tools": ["stepwise", "optimize"],
        "agent": {"context_limit": 49152},
        "logging": {"level": "full", "candidates": True},
        "scaffold": {"kind": "policy", "policy": "dialogue_lean"},
    },
}
carried = run_experiment_mod.build_run_config(raw_config, None, None)
check("experiment.metrics reaches the run config", carried.get("metrics") == {"cd": True},
      str(carried.get("metrics")))
check("experiment.tools reaches it", carried.get("tools") == ["stepwise", "optimize"],
      str(carried.get("tools")))
check("experiment.agent reaches it", carried.get("agent") == {"context_limit": 49152},
      str(carried.get("agent")))
check("experiment.logging.candidates reaches it", carried.get("logging", {}).get("candidates") is True,
      str(carried.get("logging")))
# The part wall cap reaches the run and is not empty by default: a run without a cap is a
# run that may never finish.
check("the part wall-time cap reaches the run",
      run_experiment_mod.build_run_config(
          {**raw_config, "experiment": {**raw_config["experiment"], "figure_wall_sec": 42}},
          None, None,
      ).get("figure_wall_sec") == 42)
check("and it is set by default", (carried.get("figure_wall_sec") or 0) > 0,
      str(carried.get("figure_wall_sec")))
# A removed knob must fail the config, not dissolve. Checked on the RAW config:
# `build_run_config` builds the run from an enumeration of known keys, so the validator
# no longer sees a removed key, and silence would look like working protection.
try:
    run_experiment_mod.build_run_config(
        {**raw_config, "experiment": {**raw_config["experiment"], "figure_stall_sec": 900}},
        None, None,
    )
    check("a removed pool guard is rejected by the config", False, "accepted silently")
except config_mod.ConfigError as exc:
    check("a removed pool guard is rejected by the config", True)
    check("and the hint names the replacement", "figure_wall_sec" in str(exc), str(exc))

# `model` is needed by preflight to compare the `root` of a running endpoint with the
# config. Without being carried over, the comparison would silently never fire.
check("model reaches the run config",
      run_experiment_mod.build_run_config(
          {**raw_config, "model": {"assistant_model_path": "Qwen/X"}}, None, None,
      ).get("model") == {"assistant_model_path": "Qwen/X"})

# Whether the assistant server is started is known only to the `launch` section, which is
# read by `run_system.sh`. The run needs this fact so as not to create a client for a port
# where nobody listens. What is checked is the OUTCOME: the flag reached the run config and
# the client is indeed not created because of it.
launch_off = {**raw_config, "model": {"assistant_model_path": "Qwen/X"},
              "server": {"generation_base_url": "http://x", "generation_served_model_name": "gen",
                         "assistant_base_url": "http://y", "assistant_served_model_name": "a"},
              "launch": {"servers": {"assistant": {"enabled": False}}}}
launch_on = {**launch_off, "launch": {"servers": {"assistant": {"enabled": True}}}}
check("a disabled assistant reaches the run config",
      run_experiment_mod.build_run_config(launch_off, None, None)["server"]["assistant_enabled"] is False)
check("an enabled assistant reaches the run config",
      run_experiment_mod.build_run_config(launch_on, None, None)["server"]["assistant_enabled"] is True)
# Without a `launch` section the behavior must stay as before: the client is created.
check("a config without a launch section treats the assistant as up",
      run_experiment_mod.build_run_config(
          {k: v for k, v in launch_on.items() if k != "launch"}, None, None,
      )["server"]["assistant_enabled"] is True)

# The same for the GENERATOR. The protection was added later than the assistant's and for
# the opposite reason: the assistant had it, but a disabled generator went unnoticed by the
# harness, and the run went to the end answering with a refusal on EVERY part. The outcome
# is checked: the flag arrived, and a tool set that calls the generator is rejected when
# the server is disabled.
gen_off = {**launch_on, "model": {**launch_on["model"], "generation_model_path": "/w/gen"},
           "launch": {"servers": {"generation": {"enabled": False}}}}
check("a disabled generator reaches the run config",
      run_experiment_mod.build_run_config(
          {**gen_off, "experiment": {**gen_off["experiment"], "tools": ["det_cold"]}},
          None, None,
      )["server"]["generation_enabled"] is False)


figure_run_mod = importlib.import_module("cad_agent.harness.figure_run")


class CountingClient:
    """A client that counts how many times it was created and what it was asked.

    No network is needed here: what is checked is not the server's answer but the fact of
    the call. `/v1/models` returns an empty list, which is also how the answer of a server
    that does not report `max_model_len` looks.
    """

    created = 0
    listed = 0

    def __init__(self, **kwargs):
        CountingClient.created += 1

        class _Models:
            def list(self_inner):
                CountingClient.listed += 1
                return types.SimpleNamespace(data=[])

        self.models = _Models()


def clients_for(server_section):
    """Build clients on a fake OpenAI and return what came out."""
    figure_run_mod._CLIENTS.clear()
    CountingClient.created = CountingClient.listed = 0
    saved = sys.modules["openai"].OpenAI
    sys.modules["openai"].OpenAI = CountingClient
    try:
        return dict(figure_run_mod._get_clients(server_section))
    finally:
        sys.modules["openai"].OpenAI = saved
        figure_run_mod._CLIENTS.clear()


server_off = {"generation_base_url": "http://x", "generation_served_model_name": "gen",
              "assistant_base_url": "http://y", "assistant_served_model_name": "a",
              "assistant_enabled": False}
clients = clients_for(server_off)
check("no assistant client is created when the server is disabled", "assistant" not in clients,
      str(sorted(clients)))
check("a disabled server is not asked about the context", CountingClient.listed == 0)
check("the generator client is always created", CountingClient.created == 1, f"created {CountingClient.created}")

# The other side of the same check: where the server is up the client must appear,
# otherwise the agent branch would silently be left without an agent.
clients = clients_for({**server_off, "assistant_enabled": True})
check("with the assistant up, the client is created", "assistant" in clients, str(sorted(clients)))
check("an assistant that is up is asked about its context once per worker",
      CountingClient.listed == 1, f"queries {CountingClient.listed}")

# Guard: a new schema section forgotten in the transfer must fail at start, not work with
# a default value.
_saved_keys = dict(config_mod.EXPERIMENT_KEYS)
config_mod.EXPERIMENT_KEYS["unported_section"] = dict
try:
    run_experiment_mod.build_run_config(
        {**raw_config, "experiment": {**raw_config["experiment"], "unported_section": {"x": 1}}},
        None, None,
    )
    check("a forgotten config section is caught by the guard", False, "the section was lost silently")
except ValueError:
    check("a forgotten config section is caught by the guard", True)
finally:
    config_mod.EXPERIMENT_KEYS.clear()
    config_mod.EXPERIMENT_KEYS.update(_saved_keys)

print("15. The optimizer is loaded as a package, not as a top-level module")
# The snapshot modules are written with relative imports (`from .cq_parser import`), so
# `import optimizer_numerical` with the directory on sys.path failed with "attempted
# relative import with no known parent package", i.e. the `res.optimize` capability did not
# work at all. We check that the bring-up path is different: locally the import will fail
# anyway (no cadquery), but it must fail on the missing dependency, not on the package
# layout.
dsl_runtime_mod = importlib.import_module("cad_agent.dsl_runtime")
dsl_runtime_mod.configure("wrapped")
try:
    dsl_runtime_mod.import_optimizer()
    check("the optimizer imports", True, "the environment is complete")
except dsl_runtime_mod.OptimizerUnavailable as exc:
    message = str(exc)
    check("an import miss is its own error, not a bare ImportError", True)
    check("it does not fail on the package layout", "relative import" not in message, message[:200])
except Exception as exc:
    check("an import miss is its own error, not a bare ImportError", False, f"{type(exc).__name__}: {exc}")

managed = [str(path) for path in dsl_runtime_mod._managed_paths("wrapped")]
check("we place the built _cad_grad directory ourselves",
      str(dsl_runtime_mod.NATIVE_BUILD_ROOT / "cad_grad") in managed, str(managed))

print("16. Selection modes: cd, iou, gms, hmean")

# --- scale and direction ---
cd_obj = objective_mod.get_objective("cd")
iou_obj = objective_mod.get_objective("iou")
hmean_obj = objective_mod.get_objective("hmean")

check("cd: lower is better", cd_obj.better(0.1, 0.2) and not cd_obj.better(0.2, 0.1))
check("iou: higher is better", iou_obj.better(0.8, 0.7) and not iou_obj.better(0.7, 0.8))
check("the success threshold reads in both directions",
      cd_obj.reached(1e-5, 1e-4) and iou_obj.reached(0.99, 0.98)
      and not cd_obj.reached(1e-3, 1e-4) and not iou_obj.reached(0.9, 0.98))
check("indistinguishability on both scales is (0, 1]",
      abs(cd_obj.ambiguity(0.98, 1.0) - 0.98) < 1e-9
      and abs(iou_obj.ambiguity(1.0, 0.98) - 0.98) < 1e-9)

# --- harmonic mean ---
check("hmean of two metrics", abs(hmean_obj.value({"iou": 0.5, "gms_norm": 1.0}) - 2 / 3) < 1e-9,
      str(hmean_obj.value({"iou": 0.5, "gms_norm": 1.0})))
check("hmean of a single available metric is that metric", hmean_obj.value({"gms_norm": 0.7}) == 0.7)
check("hmean is zeroed by a failure on one axis", hmean_obj.value({"iou": 0.0, "gms_norm": 0.9}) == 0.0)
check("no metrics — no value", hmean_obj.value({"cd_runtime": 0.1}) is None)

# --- the metrics order reaches the measurement itself ---
# Numbers are not checked here: locally there is neither a boolean engine (IoU) nor
# pykdtree (GMS). What is
# checked is what breaks silently: that `needs` reaches `measure_pair` at all and that
# the cd mode does not pay for extras.
import trimesh as _trimesh  # noqa: E402
import tempfile as _tempfile  # noqa: E402

metrics_mod = importlib.import_module("cad_agent.capabilities.metrics")
_tmp = Path(_tempfile.mkdtemp(prefix="objective_needs_"))
_trimesh.creation.box((20, 15, 10)).export(_tmp / "gt.stl")
_trimesh.creation.box((18, 15, 10)).export(_tmp / "pred.stl")

plain = metrics_mod.measure_pair(str(_tmp / "gt.stl"), str(_tmp / "pred.stl"))
check("without an order nothing is computed except prediction watertightness",
      plain.get("cd_runtime") is None and "gms_norm" not in plain
      and "gms_error" not in plain and "iou" not in plain
      and plain.get("pred_watertight") is not None, str(plain))

only_cd = metrics_mod.measure_pair(str(_tmp / "gt.stl"), str(_tmp / "pred.stl"), needs=("cd",))
check("requesting cd gives only CD",
      only_cd.get("cd_runtime") is not None and "gms_norm" not in only_cd
      and "iou" not in only_cd, str(only_cd))

with_gms = metrics_mod.measure_pair(str(_tmp / "gt.stl"), str(_tmp / "pred.stl"), needs=("gms",))
check("requesting gms reaches the measurement",
      "gms_norm" in with_gms or "gms_error" in with_gms, str(with_gms))
check("requesting gms does not pull in iou", "iou" not in with_gms, str(with_gms))

with_iou = metrics_mod.measure_pair(str(_tmp / "gt.stl"), str(_tmp / "pred.stl"), needs=("iou",))
check("ordering iou gives the GT watertight status", "gt_watertight" in with_iou, str(with_iou))
check("requesting iou does not pull in gms",
      "gms_norm" not in with_iou and "gms_error" not in with_iou, str(with_iou))

# An open prediction is a refusal without numbers: the full order gives it no metric.
_open = _trimesh.creation.box((18, 15, 10))
_open.update_faces(list(range(1, len(_open.faces))))
_open.export(_tmp / "open.stl")
open_pred = metrics_mod.measure_pair(str(_tmp / "gt.stl"), str(_tmp / "open.stl"), extended=True)
check("an unclosed prediction has no metrics", open_pred == {"pred_watertight": False}, str(open_pred))

# The prediction's watertight must be measured in the contract frame, the same one as IoU
# and the final recomputation. A live frame mismatch rests on rounding noise for a
# zero-thickness body and cannot be reproduced synthetically, so the frame itself is
# checked: which mesh reached `is_watertight`.
_seen_extents = []
_real_is_watertight = metrics_mod.is_watertight
metrics_mod.is_watertight = lambda mesh: (_seen_extents.append(mesh.extents.copy()),
                                          _real_is_watertight(mesh))[1]
try:
    metrics_mod.measure_pair(str(_tmp / "gt.stl"), str(_tmp / "pred.stl"))
finally:
    metrics_mod.is_watertight = _real_is_watertight
check("watertight of the prediction is measured in the contract frame (prediction / 200)",
      len(_seen_extents) == 1
      and all(abs(a - b) < 1e-9 for a, b in zip(_seen_extents[0], (18 / 200, 15 / 200, 10 / 200))),
      str(_seen_extents))

# --- config: the selection objective is a policy knob, not a run key ---
# The objective lives in the policy, so "another objective" is another policy NAME, not a
# key here. That every registry name passes validation is checked above; here it is that an
# attempt to set the objective in the config is rejected with a CLEAR refusal rather than
# silently accepted as a second source of truth.
try:
    config_mod.validate_run_config(good_config(
        scaffold={"kind": "policy", "policy": "dialogue_lean", "objective": "hmean"}))
    check("an objective in the config is rejected", False, "accepted silently")
except config_mod.ConfigError as exc:
    check("an objective in the config is rejected", True)
    check("the rejection names where the policy knobs live",
          "live in the policy itself" in str(exc), str(exc)[:160])

print("19. Objective fallback: chain, scale and threshold")
# The subject here is the semantics of the objective itself, and it is checked by direct
# calls: how these rules behave INSIDE the search loop is answered by `search_check`, and
# running another harness for them would measure the same thing twice.

# While no one has been selected on the part, the WHOLE fallback chain is ordered:
# otherwise the guard would judge the next objective by data it did not ask for.
check("the fallback chain starts at the objective and ends at cd",
      objective_mod.chain_needs(iou_obj) == ("iou", "gms", "cd"),
      str(objective_mod.chain_needs(iou_obj)))
check("an objective without a fallback has a chain of just itself",
      objective_mod.chain_needs(cd_obj) == ("cd",), str(objective_mod.chain_needs(cd_obj)))

# The scale falls to the one that IS computed and goes no further.
memory: dict = {}
picked = objective_mod.scale_for(iou_obj, memory, [{"gms_norm": 0.7}])
check("the guard stops at gms if gms is computed", picked.name == "gms", picked.name)
memory_cd: dict = {}
picked_cd = objective_mod.scale_for(iou_obj, memory_cd, [{"cd_runtime": 0.5}])
check("without iou and gms the scale falls back to cd", picked_cd.name == "cd", picked_cd.name)

# The threshold belongs to the SCALE, not the config: after a fallback to cd the threshold
# must become cd's. With the old code it stayed iou's (0.98), and `reached(0.5, 0.98)` on a
# "lower is better" scale was true, so a part stopped at the first candidate that built.
check("the threshold travels with the scale",
      picked_cd.success_threshold(None) == cd_obj.default_success
      and picked_cd.success_threshold(None) != iou_obj.default_success,
      f"{picked_cd.success_threshold(None)} vs {iou_obj.default_success}")
check("after the fallback to cd the threshold is not reached at cd 0.5",
      not picked_cd.reached(0.5, picked_cd.success_threshold(None)),
      str(picked_cd.success_threshold(None)))
# An explicitly set threshold stays explicit, and that is also its trap: it is written on
# the scale of the CONFIGURED objective and is not converted on fallback (a known caveat).
# In all our configs it is null, so it does not affect measurements.
check("an explicit threshold is not replaced by the scale default",
      picked_cd.success_threshold(0.85) == 0.85, str(picked_cd.success_threshold(0.85)))

print("20. GT derivatives are computed once per part")
# `split()` with a filter and the GMS handler depend only on GT, yet were paid for on every
# candidate. The outcome is checked, not the presence of a cache: how many times `split()`
# was called and whether the numbers match the former ones.
import trimesh as _trimesh  # noqa: E402
from cad_agent.capabilities import metrics as _metrics  # noqa: E402

_tmp = Path(tempfile.mkdtemp(prefix="smoke_gtcache_"))
_gt_path = _tmp / "gt.stl"
_trimesh.creation.box((30.0, 20.0, 10.0)).export(_gt_path)
_pred_path = _tmp / "pred.stl"
_trimesh.creation.box((28.0, 21.0, 11.0)).export(_pred_path)

_splits = {"count": 0}
_original_split = _trimesh.Trimesh.split


def _counting_split(self, *args, **kwargs):
    _splits["count"] += 1
    return _original_split(self, *args, **kwargs)


_cache: dict = {}
_trimesh.Trimesh.split = _counting_split
try:
    for _ in range(3):
        _metrics.measure_pair(gt_mesh_path=str(_gt_path), pred_mesh_path=str(_pred_path),
                             gt_cache=_cache, needs=("iou",))
    # Three candidates: GT is parsed once, the prediction every time, since it is new.
    check("GT is parsed once for three candidates", _splits["count"] == 4,
          f"split calls: {_splits['count']}")
finally:
    _trimesh.Trimesh.split = _original_split

check("the parsed GT is in the part cache", "iou_components" in _cache[str(_gt_path)])
# Shim warm-up must parse GT itself: the grandchild's cache disappears with it, and
# merging the bodies of a multi-body GT costs more than the execution timeout.
from cad_agent.capabilities import execute as _execute  # noqa: E402

_execute._GT_CACHE.pop(str(_gt_path), None)
_execute._warm_gt(str(_gt_path))
check("proxy warm-up parses GT for IoU",
      "iou_components" in _execute._GT_CACHE.get(str(_gt_path), {}))
check("iou_components gives the same as the filter inside compute_iou",
      [len(m.faces) for m in _metrics.iou_components(_cache[str(_gt_path)]["mesh"])]
      == [len(m.faces) for m in _cache[str(_gt_path)]["iou_components"]])

# The numbers must match those of the computation without a cache. Locally there is
# neither a boolean engine nor pykdtree, so the comparison runs where they exist.
_gt_mesh = _trimesh.load_mesh(_gt_path)
_pred_mesh = _trimesh.load_mesh(_pred_path)
_metrics.normalize_for_metrics(_gt_mesh, _pred_mesh)
_plain = _metrics.compute_iou(_gt_mesh, _pred_mesh)
if _plain[0] is None:
    # compute_iou swallows the absence of a boolean engine and returns a triple of None:
    # there is nothing to compare in that case, and "matched" would be a false success.
    print("  SKIPPED  IoU comparison: boolean engine unavailable, the metric was not computed")
else:
    _cached = _metrics.compute_iou(_gt_mesh, _pred_mesh,
                                   gt_components=_metrics.iou_components(_gt_mesh))
    check("IoU with the component cache matches bit for bit", _plain == _cached, f"{_plain} vs {_cached}")
try:
    _plain_gms = _metrics.compute_gms(_gt_mesh, _pred_mesh)
    _cached_gms = _metrics.compute_gms(_gt_mesh, _pred_mesh,
                                       gt_handler=_metrics.gms_handler(_gt_mesh))
    check("GMS with the handler cache matches bit for bit", _plain_gms == _cached_gms,
          f"{_plain_gms} vs {_cached_gms}")
except Exception as exc:  # noqa: BLE001
    print(f"  SKIPPED  GMS comparison: no pykdtree ({exc})")

# GT with inverted normals: IoU must match that of the same body written correctly. A
# cavity inside the body is not turned over.
_box = _trimesh.creation.box((0.4, 0.3, 0.2))
_box.apply_translation((0.5, 0.5, 0.5))
_pred_box = _trimesh.creation.box((0.35, 0.3, 0.25))
_pred_box.apply_translation((0.52, 0.5, 0.5))
_inverted = _box.copy()
_inverted.invert()
check("an inside-out GT gives the same part as a normal one",
      [round(float(m.volume), 12) for m in _metrics.iou_components(_inverted)]
      == [round(float(m.volume), 12) for m in _metrics.iou_components(_box)])
# A closed GT with some inverted faces: `is_valid_gt` lets it in, `is_volume` does not;
# after the fix it is the same part as for a whole box.
_mixed = _box.copy()
_mixed.faces[:2] = _mixed.faces[:2, ::-1]
check("a GT with inconsistent winding gives the same part as a normal one",
      not _mixed.is_winding_consistent and _metrics.is_valid_gt(_mixed)
      and [round(float(m.volume), 12) for m in _metrics.iou_components(_mixed)]
      == [round(float(m.volume), 12) for m in _metrics.iou_components(_box)])
# A hollow box: an outer wall and an inner, inverted one. The volume of the whole mesh is
# positive, and the fix does not touch it: the list of parts stays as before.
_cavity = _trimesh.creation.box((0.2, 0.15, 0.1))
_cavity.apply_translation((0.5, 0.5, 0.5))
_cavity.invert()
_hollow = _trimesh.util.concatenate([_box, _cavity])
check("a cavity inside a body is not flipped",
      [round(float(m.volume), 12) for m in _metrics.iou_components(_hollow)]
      == [round(float(m.volume), 12) for m in _hollow.split() if m.is_watertight and m.is_volume])
_ref = _metrics.compute_iou(_box, _pred_box)
if _ref[0] is None:
    print("  SKIPPED  IoU on an inside-out GT: boolean engine unavailable")
else:
    check("IoU on an inside-out GT equals IoU on a normal one",
          _metrics.compute_iou(_inverted, _pred_box) == _ref)
    # Overlapping bodies: IoU must match the IoU on their union, for both GT and the
    # prediction. The tolerance is boolean-operation noise; without merging IoU drops by
    # tenths.
    _left = _trimesh.creation.box((0.3, 0.3, 0.2))
    _left.apply_translation((0.45, 0.5, 0.5))
    _right = _trimesh.creation.box((0.3, 0.3, 0.2))
    _right.apply_translation((0.55, 0.5, 0.5))
    _overlap = _trimesh.util.concatenate([_left, _right])
    check("overlapping GT bodies merge into one", len(_metrics.iou_components(_overlap)) == 1)
    _gt_side = _metrics.compute_iou(_overlap, _pred_box)
    check("IoU on overlapping GT bodies equals IoU on their union",
          _gt_side[0] is not None and abs(_gt_side[0] - _ref[0]) < 1e-6, f"{_gt_side} vs {_ref}")
    _pred_side = _metrics.compute_iou(_pred_box, _overlap)
    check("the same for prediction bodies",
          _pred_side[0] is not None and abs(_pred_side[0] - _ref[0]) < 1e-6, f"{_pred_side} vs {_ref}")
    _far = _trimesh.creation.box((0.1, 0.1, 0.1))
    _far.apply_translation((0.9, 0.9, 0.9))
    _apart = _trimesh.util.concatenate([_box, _far])
    check("separate bodies merge without losing volume",
          abs(sum(float(m.volume) for m in _metrics.iou_components(_apart))
              - float(_box.volume) - float(_far.volume)) < 1e-8)
    # Guard against an impossible intersection (C2): a boolean operation that returned an
    # intersection larger than an operand gives "IoU unavailable". The volume of the fake
    # intersection is 2.5% larger than GT, its IoU is 0.96: the 1.05 threshold does not catch it.
    _orig_intersection = _trimesh.Trimesh.intersection
    _trimesh.Trimesh.intersection = lambda self, other, **kw: _trimesh.creation.box((0.4, 0.3, 0.205))
    try:
        check("the intersection is larger than an operand — IoU unavailable",
              _metrics.compute_iou(_box, _pred_box) == (None, None, None))
    finally:
        _trimesh.Trimesh.intersection = _orig_intersection

print("20b. det: nearly coincident polygon vertices")
from cad_agent.capabilities.code import thin_polygons as code_utils_thin  # noqa: E402

_clean = "r=extrude(None,(0,0,-10),'XY',\"sketch().push([(0.0,0.0)]).polygon([(0.0000,0.0000),(10.0000,0.0000),(10.0000,5.0000),(0.0000,0.0000)]).assemble()\",20)"
check("a contour without close vertices does not change by a single byte", code_utils_thin(_clean) == _clean)
_dirty = _clean.replace("(10.0000,0.0000),", "(10.0000,0.0000),(10.0001,0.0001),(10.0001,0.0001),")
check("close vertices merge, the closure stays", code_utils_thin(_dirty) == _clean,
      code_utils_thin(_dirty))
_tail = _clean.replace(",(0.0000,0.0000)])", ",(0.0001,0.0000),(0.0000,0.0000)])")
check("the second-to-last vertex next to the closing one is dropped", code_utils_thin(_tail) == _clean, code_utils_thin(_tail))
check("a line without polygon does not change", code_utils_thin("r=revolve(r,(0,0,0),'XY','x',360,'Y')")
      == "r=revolve(r,(0,0,0),'XY','x',360,'Y')")

print("21. Transport: a dropped connection is retried, a content refusal is not")
from cad_agent.capabilities import code as code_mod  # noqa: E402
from cad_agent.capabilities import llm as llm_mod  # noqa: E402


class FakeStatusError(Exception):
    """An endpoint error with a code, in the form the OpenAI client delivers it."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


CONTEXT_MESSAGE = (
    "Error code: 400 - {'error': {'message': \"This model's maximum context length is 49152 "
    "tokens. However, you requested 16 output tokens and your prompt contains at least 49137 "
    "input tokens\", 'param': 'input_tokens'}}"
)

check("a 400 on length is recognized as a length refusal",
      llm_mod.is_context_length_error(FakeStatusError(400, CONTEXT_MESSAGE)))
check("a 400 on length is recognized as non-retryable",
      llm_mod.is_permanent_error(FakeStatusError(400, CONTEXT_MESSAGE)))
check("429 is retried", not llm_mod.is_permanent_error(FakeStatusError(429, "rate limited")))
check("503 is retried", not llm_mod.is_permanent_error(FakeStatusError(503, "unavailable")))
check("a dropped connection without a code is retried", not llm_mod.is_permanent_error(OSError("connection reset")))
check("401 is non-retryable but not about length",
      llm_mod.is_permanent_error(FakeStatusError(401, "unauthorized"))
      and not llm_mod.is_context_length_error(FakeStatusError(401, "unauthorized")))


class RaisingClient:
    """A client that always fails with a given error. Counts attempts."""

    def __init__(self, exc):
        self.exc = exc
        self.attempts = 0
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.attempts += 1
                raise outer.exc

        class Chat:
            completions = Completions()

        self.chat = Chat()


_saved_backoff = llm_mod.RETRY_BACKOFF_SEC
llm_mod.RETRY_BACKOFF_SEC = 0.0
try:
    client = RaisingClient(FakeStatusError(400, CONTEXT_MESSAGE))
    call = llm_mod.call_vision(client=client, model_name="assistant", text="a long prompt")
    # The OUTCOME is checked, not that the classification function was called: such a
    # refusal used to cost three requests and three seconds of sleep on every step.
    check("a length refusal is not retried", client.attempts == 1, f"attempts: {client.attempts}")
    check("a length refusal is marked in the measurement", call.permanent and call.context_overflow, str(call.to_dict()))

    client = RaisingClient(FakeStatusError(503, "unavailable"))
    call = llm_mod.call_vision(client=client, model_name="assistant", text="prompt")
    check("a dropped connection is retried to the end", client.attempts == llm_mod.DEFAULT_MAX_ATTEMPTS,
          f"attempts: {client.attempts}")
    check("a dropped connection is not marked non-retryable", not call.permanent and not call.context_overflow)
finally:
    llm_mod.RETRY_BACKOFF_SEC = _saved_backoff


class ChatClient:
    """A chat reply in the vLLM 0.18 form: `reasoning` instead of `reasoning_content`, function calls."""

    def __init__(self):
        self.requests: list[dict] = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                function = types.SimpleNamespace(name="act", arguments='{"n": 2}')
                message = types.SimpleNamespace(
                    content="", reasoning="thinking",
                    tool_calls=[types.SimpleNamespace(id="call_9", function=function)])
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message, finish_reason="tool_calls")],
                    usage=None)

        class Chat:
            completions = Completions()

        self.chat = Chat()


client = ChatClient()
history = [{"role": "system", "content": "intro"},
           {"role": "assistant", "content": "", "tool_calls": []}]
call = llm_mod.call_vision(client=client, model_name="assistant", text="question",
                           generation_kwargs={"tools": [{"type": "function"}]}, history=history)
sent = client.requests[-1]
check("history goes before the question, the question last",
      sent["messages"][:2] == history and sent["messages"][-1] == {"role": "user", "content": "question"},
      str(sent["messages"]))
check("functions go out as a request field", sent.get("tools") == [{"type": "function"}], str(sent))
check("function calls are parsed from the response",
      call.tool_calls == [{"id": "call_9", "name": "act", "arguments": '{"n": 2}'}],
      str(call.tool_calls))
check("reasoning is also read from the `reasoning` field (vLLM 0.18)", call.reasoning == "thinking",
      repr(call.reasoning))
call = llm_mod.call_vision(client=client, model_name="assistant", text="question")
check("without history there is one message, as before",
      client.requests[-1]["messages"] == [{"role": "user", "content": "question"}],
      str(client.requests[-1]["messages"]))


class ModelsClient:
    """A `/v1/models` reply in the form vLLM returns."""

    def __init__(self, entries):
        outer = self

        class Models:
            def list(self):
                return types.SimpleNamespace(data=outer.entries)

        self.entries = entries
        self.models = Models()


check("the context is read from /v1/models",
      llm_mod.fetch_context_limit(
          ModelsClient([types.SimpleNamespace(id="assistant", max_model_len=49152)]), "assistant",
      ) == 49152)
check("the context is taken from its own model, not from the first one found",
      llm_mod.fetch_context_limit(
          ModelsClient([
              types.SimpleNamespace(id="generation", max_model_len=8192),
              types.SimpleNamespace(id="assistant", max_model_len=49152),
          ]),
          "assistant",
      ) == 49152)
check("a server without max_model_len gives None, not zero",
      llm_mod.fetch_context_limit(ModelsClient([types.SimpleNamespace(id="assistant")]), "assistant") is None)


class DeadModelsClient:
    class Models:
        def list(self):
            raise OSError("server does not answer")

    models = Models()


check("an unavailable /v1/models does not fail the run",
      llm_mod.fetch_context_limit(DeadModelsClient(), "assistant") is None)

print("22. Selection set: coordinate folding, labels, common parent")
# A ring of 60 vertices is not an invented size but the lower bound of what `_sketch_of`
# gives on a residual after decimation: there are hundreds of them. On nine vertices the
# collapse would also work but would not show what it is for.
POLY = ",".join(f"({idx}.0,{idx * 2}.0)" for idx in range(60))
LONG_STEP = f'r=extrude(r,(0.0,0.0,0.0),\'XY\',"sketch().polygon([{POLY}]).finalize()",5.0,False)'

shortened, n_elided = code_mod.elide_point_lists(LONG_STEP)
check("a long coordinate list is folded", n_elided == 1 and "<60 pts>" in shortened, shortened)
check("folding shrinks the string several times over", len(shortened) * 2 < len(LONG_STEP),
      f"{len(LONG_STEP)} -> {len(shortened)}")
check("a short list is left alone", code_mod.elide_point_lists("r=box(r,(1.0,2.0),(3.0,4.0))")[1] == 0)
check("the vertex count is kept in the fold", "<60 pts>" in shortened)

# The set shown to the assistant is prepared by the harness (`state_render`, which also
# collapses coordinates). The check goes through this seam, not through a run.
from cad_agent.harness import state_render as render_mod  # noqa: E402
from cad_agent.harness.search_types import Candidate, Origin  # noqa: E402


def _cand(cid, code, parent_id="root", tool="stepwise"):
    return Candidate(
        id=cid, parent_id=parent_id, depth=1, origin=Origin(tool=tool, attempt=0),
        code=code, mesh_path=f"/tmp/{cid}.stl", metrics={"iou": 0.5}, built=True,
    )


_pool_cands = {
    "root": _cand("root", "PREFIX\n" + LONG_STEP, parent_id=None),
    "a": _cand("a", "PREFIX\n" + LONG_STEP + "\n" + LONG_STEP, tool="det_warm"),
    "b": _cand("b", "PREFIX\n" + LONG_STEP + "\nr=box(r,1)"),
}
_render = render_mod.build_renderer(
    res=types.SimpleNamespace(render=None), obs=types.SimpleNamespace(gt_mesh_path="/tmp/gt.stl"),
    lookup=_pool_cands.get,
)
choice = _render(["a", "b"], with_image=False)
check("the harness counted how many lists it folded", choice.elided > 0, str(choice.elided))
plain_choice = _render(["a", "b"], with_image=False, elide=False)
check("disabled elision computes nothing", plain_choice.elided == 0, str(plain_choice.elided))

# Own labels (the dialogue labels panels by id) are tied to the id, not the position: a
# candidate dropped from the middle does not shift the neighbors' labels.
labelled = _render(["a", "gone", "b"], labels=["a", "gone", "b"], with_image=False)
check("labels stay on their own candidates when filtering",
      labelled.ids == ["a", "b"] and labelled.labels == ["a", "b"],
      f"{labelled.ids} {labelled.labels}")

# A common parent is shown only when it is TRULY common: for candidates of different
# parents the "previous code" is different code, and one of them in the prompt would be an
# outright untruth.
_pool_cands["c"] = _cand("c", "PREFIX\nr=box(r,2)", parent_id="a")
mixed = _render(["a", "c"], with_image=False)
check("candidates of different parents do not show a common prefix",
      mixed.parent_code == "", repr(mixed.parent_code[:60]))

print("23. Prompt cap: checked before sending and before charging the budget")


class CountingAgentClient:
    """An assistant that counts how many requests reached it."""

    def __init__(self, answer="A"):
        self.requests = 0
        self.last_kwargs: dict = {}
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.requests += 1
                # The request is stored whole: a key accepted by the config and not
                # delivered to the server looks like a working one, so the outcome must
                # be checked, not legality.
                outer.last_kwargs = kwargs
                message = types.SimpleNamespace(content=answer)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)], usage=None,
                )

        class Chat:
            completions = Completions()

        self.chat = Chat()


def build_agent_resources(context_limit, journal_obj=None, thinking=None):
    client = CountingAgentClient()
    res_obj = resources.build_resources(
        figure_id="test/fig",
        gt_mesh_path=Path("/tmp/gt.stl"),
        work_dir=Path("/tmp"),
        executor=None,
        renderer=None,
        proposer=None,
        agent_client=client,
        agent_model="assistant",
        budget=budget.Budget(),
        config={},
        journal=journal_obj,
        agent_context_limit=context_limit,
        agent_thinking=thinking,
    )
    return res_obj, client


res_obj, client = build_agent_resources(context_limit=1000)
answer = res_obj.ask_agent("a short question", max_tokens=16)
check("a prompt within size reaches the server", client.requests == 1 and answer == "A", answer)
check("a call that completed is charged to the budget", res_obj.budget.counts["agent_text"] == 1)

res_obj, client = build_agent_resources(context_limit=1000)
huge = "x" * 100000
try:
    res_obj.ask_agent(huge, max_tokens=16)
    refused = False
except llm_mod.PromptTooLarge:
    refused = True
check("a too long prompt is rejected before sending", refused and client.requests == 0,
      f"requests to the server: {client.requests}")
# The cost counter counts calls, not intentions: a request that did not happen cost the
# server nothing, and recording it as a call would overstate the price of the agent branch
# exactly where it already degraded.
check("a call that did not happen is not charged to the budget", res_obj.budget.counts["agent_text"] == 0,
      str(dict(res_obj.budget.counts)))

res_obj, client = build_agent_resources(context_limit=None)
res_obj.ask_agent("x" * 100000, max_tokens=16)
check("without a known cap the check stays silent", client.requests == 1)

print("24. The assistant reasoning mode reaches the request")

res_obj, client = build_agent_resources(context_limit=1000, thinking=True)
res_obj.ask_agent("question", max_tokens=16)
extra = client.last_kwargs.get("extra_body") or {}
check("thinking reached the server as a chat template key",
      (extra.get("chat_template_kwargs") or {}).get("enable_thinking") is True,
      str(client.last_kwargs))

# A disabled mode MUST arrive as a key. `enable_thinking: false` and the absence of the
# key are different things, exactly the reverse of how it was recorded before: the `Qwen3`
# template reads absence as ENABLED (`enable_thinking is undefined or ... is true`), so
# silence about a disabled mode turns reasoning on. While the check demanded silence, a whole
# run went with reasoning while labeling every call as disabled.
res_obj, client = build_agent_resources(context_limit=1000, thinking=False)
res_obj.ask_agent("question", max_tokens=16)
extra = client.last_kwargs.get("extra_body") or {}
check("the disabled mode reached the server as a chat template key",
      (extra.get("chat_template_kwargs") or {}).get("enable_thinking") is False,
      str(client.last_kwargs))

# Silence stays for exactly one state: no key in the config at all. Earlier measurements
# were taken on such configs, and substituting a value of our own would change their
# conditions retroactively.
res_obj, client = build_agent_resources(context_limit=1000, thinking=None)
res_obj.ask_agent("question", max_tokens=16)
check("a mode not set by the run sends no key",
      "extra_body" not in client.last_kwargs, str(client.last_kwargs))

# A per-call override is stronger than the run mode in both directions.
res_obj, client = build_agent_resources(context_limit=1000, thinking=True)
res_obj.ask_agent("question", max_tokens=16, thinking=False)
extra = client.last_kwargs.get("extra_body") or {}
check("a call may turn reasoning off on a thinking run",
      (extra.get("chat_template_kwargs") or {}).get("enable_thinking") is False,
      str(client.last_kwargs))

# The run mode is visible to the policy through the seam: without it the policy cannot
# tell "cut off in the middle of reasoning" from "cut off in the answer".
for asked, expected in ((True, True), (False, False), (None, None)):
    res_obj, _ = build_agent_resources(context_limit=1000, thinking=asked)
    check(f"run mode {asked} is visible in the part resources",
          res_obj.agent_thinking is expected, str(res_obj.agent_thinking))

check("the estimate grows with prompt length",
      llm_mod.estimate_prompt_tokens("x" * 3000) > llm_mod.estimate_prompt_tokens("x" * 300))
check("an image counts toward the estimate",
      llm_mod.estimate_prompt_tokens("x", images=1) - llm_mod.estimate_prompt_tokens("x")
      == llm_mod.AGENT_IMAGE_TOKENS)
# The length guard counts the IMAGES, not their presence: a question may carry a list, and
# an estimate of "one image" would let through a prompt four times longer than counted.
check("an image list is counted by number",
      llm_mod.estimate_prompt_tokens("x", images=4) - llm_mod.estimate_prompt_tokens("x")
      == 4 * llm_mod.AGENT_IMAGE_TOKENS)

print("25. An image list goes out as separate blocks")


class _FakeImage:
    def save(self, buffer, format="PNG"):
        buffer.write(b"png")


res_obj, client = build_agent_resources(context_limit=100000)
res_obj.ask_agent("question", image=[_FakeImage(), _FakeImage(), _FakeImage()], max_tokens=16)
content = client.last_kwargs["messages"][-1]["content"]
blocks = [part for part in content if part.get("type") == "image_url"]
check("the request has as many images as were given", len(blocks) == 3, str(len(content)))
check("text comes after the images", content[-1]["type"] == "text", str(content[-1]["type"]))
# The counter counts QUESTIONS with an image, not images: a list goes as one request, and
# recording it as three calls would overstate the price of the channel.
check("a list is a single visual call", res_obj.budget.counts["agent_visual"] == 1,
      str(dict(res_obj.budget.counts)))

res_obj, client = build_agent_resources(context_limit=1000)
try:
    res_obj.ask_agent("x" * 100, image=[_FakeImage()] * 10, max_tokens=16)
    refused = False
except llm_mod.PromptTooLarge:
    refused = True
check("a long list is rejected before sending by its own estimate",
      refused and client.requests == 0, f"requests: {client.requests}")

print("25a. Prefix cache prewarm before the question (`experiment.agent.prewarm`)")


class _PrewarmClient:
    """An assistant that remembers all requests and returns `cached_tokens` in `usage`."""

    def __init__(self, fail_prewarm=False):
        self.sent: list[dict] = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.sent.append(kwargs)
                if fail_prewarm and kwargs.get("max_tokens") == 1:
                    raise RuntimeError("warm-up failed")
                usage = types.SimpleNamespace(
                    prompt_tokens=100, completion_tokens=1,
                    prompt_tokens_details=types.SimpleNamespace(cached_tokens=64))
                message = types.SimpleNamespace(content="A")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                             usage=usage)

        self.chat = types.SimpleNamespace(completions=Completions())


def _prewarm_resources(prewarm=True, dp=3, fail_prewarm=False):
    client = _PrewarmClient(fail_prewarm=fail_prewarm)
    journal_obj = types.SimpleNamespace(
        events=[], calls={}, stage=lambda name: contextlib.nullcontext(),
        event=lambda kind, **fields: journal_obj.events.append((kind, fields)),
        record_call=lambda kind, latency_sec=0.0: journal_obj.calls.__setitem__(
            kind, journal_obj.calls.get(kind, 0) + 1),
        record_llm=lambda kind, call: None, save_agent_call=lambda **kwargs: None)
    res_obj = resources.build_resources(
        figure_id="test/fig", gt_mesh_path=Path("/tmp/gt.stl"), work_dir=Path("/tmp"),
        executor=None, renderer=None, proposer=None, agent_client=client,
        agent_model="assistant", budget=budget.Budget(), config={}, journal=journal_obj,
        agent_prewarm=prewarm, agent_dp_size=dp)
    return res_obj, client, journal_obj


_slot = llm_mod.IMAGE_SLOT
_head = f"rules{_slot}\n\nTurn 1 executed.\n\n"
_prompt = _head + f"{_slot}note\n\ntable"
_pics = [_FakeImage(), _FakeImage()]
res_obj, client, journal_obj = _prewarm_resources()
res_obj.ask_agent(_prompt, image=_pics, max_tokens=16, prefix=_head)
check("prewarm and question are two requests, prewarm first",
      len(client.sent) == 2 and client.sent[0]["max_tokens"] == 1
      and client.sent[1]["max_tokens"] == 16, str([k.get("max_tokens") for k in client.sent]))
warm, main = client.sent
check("warm-up is an open message without the assistant prompt",
      warm["extra_body"].get("continue_final_message") is True
      and warm["extra_body"].get("add_generation_prompt") is False, str(warm["extra_body"]))
_ranks = {k["extra_headers"]["X-data-parallel-rank"] for k in client.sent}
check("both requests go to one replica", len(_ranks) == 1
      and 0 <= int(next(iter(_ranks))) < 3, str(_ranks))
_warm_parts = warm["messages"][-1]["content"]
check("the prewarm has the question start up to and including the transcript, one image",
      [p["type"] for p in _warm_parts] == ["text", "image_url", "text"]
      and _warm_parts[-1]["text"] == "\n\nTurn 1 executed.\n\n", str(_warm_parts))
check("warm-up is not charged to the budget", res_obj.budget.counts["agent_visual"] == 1
      and journal_obj.calls.get("agent_prewarm") == 1, str(dict(res_obj.budget.counts)))
_asked = [f for kind, f in journal_obj.events if kind == "ask_agent"]
check("the question cache hit is in the event", _asked and _asked[0]["cached_tokens"] == 64,
      str(_asked))
res_obj.ask_agent(_prompt, image=_pics, max_tokens=16, prefix=_head)
check("a re-ask with the same start does not warm up again", len(client.sent) == 3, str(len(client.sent)))

res_obj, client, journal_obj = _prewarm_resources()
res_obj.ask_agent(_prompt, image=_pics, max_tokens=16, prefix="other\n\n")
check("a start that is not a prefix is not warmed up",
      len(client.sent) == 1 and ("agent_prewarm", {"skipped": "not_a_prefix", "prefix_chars": 7})
      in journal_obj.events, str(journal_obj.events))

res_obj, client, journal_obj = _prewarm_resources(fail_prewarm=True)
answer = res_obj.ask_agent(_prompt, image=_pics, max_tokens=16, prefix=_head)
check("a prewarm failure does not break the question", answer == "A" and len(client.sent) == 2, answer)

res_obj, client, _ = _prewarm_resources(prewarm=False)
res_obj.ask_agent(_prompt, image=_pics, max_tokens=16, prefix=_head)
check("without a warm-up key — one request without a replica header",
      len(client.sent) == 1 and "extra_headers" not in client.sent[0], str(client.sent[0].keys()))
res_obj, client, _ = _prewarm_resources(dp=1)
res_obj.ask_agent(_prompt, image=_pics, max_tokens=16, prefix=_head)
check("one replica — warm-up without a header", len(client.sent) == 2
      and all("extra_headers" not in k for k in client.sent), str(len(client.sent)))

print("26. Mesh validity and the iou_unavailable refusal")
# Rules: GT goes into IoU only if the whole mesh is closed; a prediction is valid only if
# the whole mesh is closed; a candidate without IoU with a valid GT and prediction is a
# refusal with its own IR share. The old rule ("at least one closed body") on the same
# meshes would say "yes", and that is what is checked.
import numpy as np  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402
from cad_agent.harness import search as _search_mod  # noqa: E402

_solid = _trimesh.creation.box((1.0, 1.0, 1.0))
_open = _trimesh.creation.box((0.5, 0.5, 0.5))
_open.update_faces(np.arange(len(_open.faces)) > 0)
_open.remove_unreferenced_vertices()
_open.apply_translation((3.0, 0.0, 0.0))
_mixed = _trimesh.util.concatenate([_solid, _open])
check("the old rule would have let a body with an open part through",
      any(m.is_watertight and m.is_volume for m in _mixed.split()))
check("a prediction with an open part is invalid", not _metrics.is_watertight(_mixed))
check("a GT with an open part is excluded from IoU", not _metrics.is_valid_gt(_mixed))
check("a whole body is valid on both sides",
      _metrics.is_watertight(_solid) and _metrics.is_valid_gt(_solid))

_real_engine, _real_iou = _metrics.iou_engine_available, _metrics.compute_iou
try:
    _metrics.compute_iou = lambda *args, **kwargs: (None, None, None)
    _metrics.iou_engine_available = lambda: True
    _no_iou = _metrics.evaluate_pair("f_noiou", _solid.copy(), _solid.copy())
    _metrics.iou_engine_available = lambda: False
    _no_engine = _metrics.evaluate_pair("f_noengine", _solid.copy(), _solid.copy())
finally:
    _metrics.iou_engine_available, _metrics.compute_iou = _real_engine, _real_iou
check("an empty IoU for a valid pair is an iou_unavailable refusal",
      _no_iou.failure == _metrics.FAILURE_IOU_UNAVAILABLE and _no_iou.score() == 0.0,
      str(_no_iou.failure))
check("an empty IoU without a boolean engine is not a refusal", _no_engine.failure is None, str(_no_engine.failure))

_agg = _metrics.aggregate([_no_iou, _no_engine])
check("its own IR share, and the decomposition adds up",
      abs(_agg.ir_iou_unavailable - 0.5) < 1e-12
      and abs(_agg.ir - (_agg.ir_execution + _agg.ir_not_watertight
                         + _agg.ir_no_result + _agg.ir_iou_unavailable)) < 1e-12,
      str(_agg.to_dict()))

_flagged = _NS(success=True, metrics={"pred_watertight": True, "gms_norm": 0.9,
                                      objective_mod.FIELD_IOU_UNAVAILABLE: True})
check("the harness classifies the measurement flag as a refusal",
      _search_mod._classify(_flagged) == "iou_unavailable", str(_search_mod._classify(_flagged)))

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
    sys.exit(1)
print("All checks passed.")
