"""Execute DSL code in an isolated process.

Why isolation is mandatory: on bad geometry OCC (via CadQuery/OCP) does not just
raise an exception -- it may hang, crash the whole process and leave corrupted
global state behind. So candidate code runs in a separate process that **dies**
afterwards. Reusing one process for several executions saves forks but
accumulates OCC state in the live process; that is deliberately avoided.

Three backends sit behind one interface, because it is not known in advance
which is faster with N parallel part processes; the choice is made by
measurement, not opinion:

``serial_fork``
    The part process forks one child per candidate and waits for it.
    Concurrency 1.
``ephemeral_pool``
    The part process keeps up to k concurrent children; each lives for one task.
    Forks here come from a multithreaded parent (threads watch the children),
    which is a classic trap: the child inherits only the forking thread, and if
    another thread held an allocator lock at fork time the child may hang on its
    first allocation. One more argument for ``proxy_pool``, where a
    single-threaded shim does the forking.
``proxy_pool``
    k shims that do not execute code themselves but fork a grandchild per task.
    The fork comes from a thin shim rather than a bloated part process, and the
    grandchild inherits for free the GT cache, the already-imported execution
    language (OCP, cadquery, cadgen) and the metrics stack it measures with:
    numpy, scipy, trimesh, pykdtree (see `_preload_cad` and
    `_preload_metrics`). Shims live for the **whole worker process**, not one
    part: they know nothing about the part except the warmed GT, which is
    evicted on every new one (see `_warm_gt`).

All three share one primitive, :func:`run_in_fork`: fork, wait for the result
through a pipe with a timeout, ``SIGKILL`` for a hung child, a proper reap.
The child exits through ``os._exit`` so as not to run ``atexit`` handlers or
flush the parent's buffers a second time.
"""

from __future__ import annotations

import importlib
import logging
import math
import multiprocessing as mp
import os
import pickle
import queue as queue_mod
import select
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_POOL_SIZE = 4

# How often to poll the result queues, how often to check shim liveness, and how
# much slack to give beyond a task's timeout before treating it as lost.
#
# Result polling and liveness checks are **different** intervals; merging them
# into one number cost a run a third of its time. With a single 2 s constant,
# every `evaluate` went to sleep for a full tick although a candidate executes
# in a fraction of a second. Waiting for a result must be short (its price is
# the latency of every candidate); the liveness check must be rare (its price is
# `is_alive()` per shim, and there is no hurry there).
_PROXY_POLL_SEC = 0.02
_PROXY_REAP_SEC = 1.0
_PROXY_GRACE_SEC = 10.0

# OCP thread pool limit: there is nothing to parallelise inside a child process,
# and by default OCC starts a pool sized by core count and disturbs neighbours.
_OCP_THREADS = 1


@dataclass
class EvalTask:
    """One unit of work: execute code and, if requested, measure it right away."""

    task_id: str
    code: str
    mesh_path: str | None = None
    export_format: str = "stl"
    gt_mesh_path: str | None = None
    measure: bool = False
    # Extended observables at every step (IoU/GMS/iog/iop) are optional: they
    # cost more than CD, and are worth paying for in analysis, not in bulk runs.
    # Enabled by `experiment.extended_metrics`.
    extended: bool = False
    # What must be computed so that the harness can decide at all: the set
    # `objective.Objective.needs` (`"iou"`, `"gms"`). It comes from the harness,
    # not the config: the selection objective is part of the policy, and whoever
    # chose it pays for it.
    needs: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


# How the execution ended. The distinction is mandatory: "the code raised an
# exception" is about geometry and goes into the IR as such, while "the process
# was killed by a signal" is about infrastructure; silently mixing them would
# blame a cluster failure on poor model quality.
OUTCOME_OK = "ok"                  # the function returned a value
OUTCOME_ERROR = "error"            # the code inside raised an exception
OUTCOME_TIMEOUT = "timeout"        # exceeded the timeout, killed
OUTCOME_DIED = "died"              # died without a result (signal, OOM, OCC)
OUTCOME_UNREADABLE = "unreadable"  # returned something that could not be parsed
OUTCOME_LOST = "lost"              # the executor lost the task (a shim died)

# Outcomes that mean the executor died, as opposed to the code failing.
WORKER_DEATHS = (OUTCOME_TIMEOUT, OUTCOME_DIED, OUTCOME_UNREADABLE, OUTCOME_LOST)

# STL export tolerance ladder. The first rung is the working tolerance, and for
# the vast majority of bodies it is all that is needed. The rest are tried ONLY
# when the mesh came out open: OCC can tessellate a valid closed solid with
# holes (a bevel gear with a tooth slant does this for about a third of samples).
# An open prediction cannot be measured -- IoU is undefined on it -- so the
# candidate would be lost entirely although a valid body was built.
#
# The rungs are not "more precise" but DIFFERENT: refining the tolerance is not
# monotone in holes -- 0.0005 gives more of them than 0.001. So the rung is
# chosen by measurement (did the mesh close), not by reasoning about precision.
EXPORT_LADDER: tuple[tuple[float, float], ...] = (
    (0.001, 0.1),
    (0.0001, 0.1),
    (0.00005, 0.05),
)

# Fallback rung: the first tolerance, but absolute. `Shape.export` meshes with
# `relative=True`, and on long edges the meshes of adjacent faces diverge,
# leaving cracks on every rung of the ladder. It is called only if the ladder
# failed to close the mesh, so outputs closed on the ladder do not change. Its
# rung number in `export_rung` is `len(EXPORT_LADDER)`.
EXPORT_FALLBACK: tuple[float, float] = (0.001, 0.1)

# Index a shim uses to mark a warm-up acknowledgement: it is not a task result
# and must not be confused with one -- real indices are non-negative.
_WARM_ACK_INDEX = -1


@dataclass
class EvalResult:
    task_id: str
    success: bool
    error: str | None = None
    mesh_path: str | None = None
    metrics: dict[str, Any] | None = None
    wall_sec: float = 0.0
    timed_out: bool = False
    # How it ended. `error` is a code failure; the rest of WORKER_DEATHS is the
    # death of the executor, which must not be lost: it distorts both the IR and
    # the cost.
    outcome: str = OUTCOME_OK
    # Number of threads in the fork that computed this geometry. Measured by the
    # fork itself: a background sampler almost always misses such a short process.
    n_threads: int | None = None
    # GT cache hits and misses **for this task**, as seen by the fork. Non-zero
    # hits occur only with `proxy_pool`: other backends have nobody to inherit
    # the warmed state from, which is exactly why it was chosen.
    gt_cache: dict[str, int] | None = None
    # The rung of the tolerance ladder (`EXPORT_LADDER`) at which the body came
    # out. 0 is the working tolerance; later rungs are tried only when the mesh
    # came out open. This is a **condition of measurement** for the metrics, not
    # a detail: a body that came out at the second rung is tessellated
    # differently from its neighbour. `None` for non-STL and tasks without export.
    export_rung: int | None = None
    # Breakdown of `wall_sec` into fork phases: `build_sec`, `export_sec`,
    # `metrics_sec`, `overhead_sec` and `n`. A dict rather than fields: the next
    # phase should not require edits in four files. Filled only for tasks that
    # ran to the end -- a failed candidate's fork returns nothing but the error
    # text, so there is nothing to split.
    phases_sec: dict[str, float] | None = None

    @property
    def worker_died(self) -> bool:
        return self.outcome in WORKER_DEATHS


def run_in_fork(
    func: Callable[..., Any],
    args: tuple,
    timeout: float,
) -> tuple[bool, Any, str]:
    """Run ``func(*args)`` in a forked process and collect the result.

    Returns ``(ok, payload, outcome)``, where outcome is one of the
    ``OUTCOME_*`` constants. The parent never waits longer than ``timeout``: a
    hung child gets ``SIGKILL`` and is always reaped with ``waitpid``, otherwise
    zombies would accumulate.

    The outcome is returned separately from the error text on purpose. Formerly
    "the code raised ValueError" and "the process was killed by signal 11"
    arrived upstream as the same ``success=False`` with a string, so an
    executor's death could not be told from bad geometry and ended up in
    ``ir_execution``, blaming a cluster failure on model quality.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()

    if pid == 0:  # child
        os.close(read_fd)
        try:
            payload = (True, func(*args))
        except BaseException:
            payload = (False, traceback.format_exc())
        try:
            with os.fdopen(write_fd, "wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            pass
        os._exit(0)

    # parent
    os.close(write_fd)
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    timed_out = False
    # The status is remembered where the child is actually reaped: it can be
    # reaped exactly once, and a second `waitpid` raises ChildProcessError.
    status: int | None = None
    reaped = False

    with os.fdopen(read_fd, "rb") as stream:
        os.set_blocking(stream.fileno(), False)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([stream], [], [], min(remaining, 0.5))
            if not ready:
                done, status = _reap(pid, block=False)
                if done:
                    reaped = True
                    break
                continue
            chunk = stream.read()
            if chunk is None:
                # Non-blocking read with no data: select said "ready" but the
                # bytes have not arrived yet. This is not EOF -- do not exit,
                # or the result would be cut off mid-way.
                continue
            if chunk == b"":
                break
            chunks.append(chunk)

    if timed_out:
        _kill(pid)
        return False, f"Execution aborted on timeout after {timeout:.0f} s.", OUTCOME_TIMEOUT

    if not reaped:
        _done, status = _reap(pid, block=True)

    if not chunks:
        return False, f"The child process died without returning a result ({_exit_reason(status)}).", OUTCOME_DIED

    try:
        ok, payload = pickle.loads(b"".join(chunks))
    except Exception:
        return False, "Could not parse the child process result.", OUTCOME_UNREADABLE
    return ok, payload, OUTCOME_OK if ok else OUTCOME_ERROR


def _exit_reason(status: int | None) -> str:
    """Decode a `waitpid` status -- a signal is worth more than the word "died".

    SIGSEGV means OCC crashed the process on bad geometry; SIGKILL without our
    own timeout is almost always the OOM killer, which is a question for
    n_workers, not for the model.
    """
    if status is None:
        return "status unknown"
    if os.WIFSIGNALED(status):
        number = os.WTERMSIG(status)
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = "?"
        hint = " (probably the OOM killer)" if name == "SIGKILL" else ""
        return f"killed by signal {number} ({name}){hint}"
    if os.WIFEXITED(status):
        return f"exited with code {os.WEXITSTATUS(status)}"
    return f"status {status}"


def _reap(pid: int, block: bool) -> tuple[bool, int | None]:
    """`(reaped, status)`. The status is needed to name the cause of death."""
    try:
        done, status = os.waitpid(pid, 0 if block else os.WNOHANG)
    except ChildProcessError:
        return True, None  # already reaped by someone else
    if done == 0:
        return False, None
    return True, status


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _reap(pid, block=True)


# --- what runs inside the child process --------------------------------------

_GT_CACHE: dict[str, Any] = {}


def _limit_ocp_threads() -> None:
    """Limit the OCC thread pool. Cheap: OCP is already imported by now.

    Called both in the parent (as part of warm-up) and in the fork. In the fork
    it is "just in case": the limit is inherited, but it costs one `sys.modules`
    lookup and a call, while a mistake costs a 192-thread pool in each of four
    concurrent grandchildren.

    The call itself lives in ``dsl_runtime``: the same limit is needed by the
    build fork of the gear benchmark, and while each path had its own line, one
    of them had none at all.
    """
    from cad_agent import dsl_runtime  # noqa: PLC0415

    dsl_runtime.limit_ocp_threads(_OCP_THREADS)


# The metrics stack: what the execution fork imports itself, **not** through the
# dialect preamble, when a task asks for measurement (`measure=True`). The
# preamble knows nothing about these modules, so preamble-based warm-up did not
# load them, and the cost landed in `overhead_sec` -- for every candidate, in
# every grandchild.
#
# One list serves both warm-up and the guard: warm-up imports exactly these
# names, and the guard requires exactly these. A second list would silently
# drift from the first -- the same trap that makes `preamble_modules()` derive
# from the preamble itself.
#
# `cad_agent.capabilities.metrics` pulls in numpy, trimesh and scipy on its own;
# they are named separately so that the guard's message shows what exactly was
# not inherited, not only the name of our module.
METRICS_MODULES = (
    "numpy",
    "scipy.spatial",
    "trimesh",
    "cad_agent.capabilities.metrics",
)

# GMS machinery from `vendor/cicada_metrics` and its KD-tree. Separate from
# `METRICS_MODULES` because it is not always needed: `measure_pair` builds GMS
# only with `extended` or `"gms" in needs`, and `pykdtree` may be absent from the
# environment altogether (then GMS is not computed in any mode).
GMS_MODULES = ("pykdtree.kdtree", "gms")

# Warn about unavailable GMS once per process. `_preload_cad()` is called not
# only at start: the `serial_fork` and `ephemeral_pool` constructors call it for
# every part, and without this flag a run without `pykdtree` would print the
# same warning a thousand times.
_gms_warned = False


def _preload_metrics() -> bool:
    """Warm up the metrics stack. Returns whether the GMS machinery came up.

    Metrics are measured **inside the fork**, where the mesh is still at hand
    (see `_evaluate`), so numpy, scipy, trimesh and our metrics module are
    needed by the grandchild. Warm-up used to know nothing about them. Some of
    them reached the grandchild **by accident**: the worker imports
    `harness/figure_run.py`, which pulls `capabilities/metrics` and its whole
    stack at top level. That accident rests on someone else's import and would
    vanish with it, so here it is replaced with an explicit statement.

    The GMS machinery **never** arrived: `import_gms()` is called lazily from
    `gms_handler`, i.e. for the first time in the grandchild, and again for
    every candidate (`gms` plus `pykdtree`).

    It is brought up separately and softly: `pykdtree` is not in every
    environment, and with `objective: cd|iou` the GMS machinery is not needed at
    all. Failing here would forbid a run that would have worked anyway.
    """
    global _gms_warned

    for name in METRICS_MODULES:
        importlib.import_module(name)

    from cad_agent import dsl_runtime  # noqa: PLC0415

    try:
        dsl_runtime.import_gms()
    except Exception as exc:
        if not _gms_warned:
            _gms_warned = True
            logger.warning(
                "GMS machinery is not warmed up (%s: %s). If a measurement needs it "
                "(objective: gms or extended_metrics), the guard will fail every such "
                "candidate, and rightly so: pykdtree is required, an environment without it "
                "is not ready for a run. The cd and iou objectives do not need it.",
                type(exc).__name__, exc,
            )
        return False
    return True


def _preload_cad() -> None:
    """Warm up the parent: OCP, the **execution language** and the metrics stack -- free for children.

    Called only in processes that do not execute code themselves but fork: in a
    ``proxy_pool`` shim and in the constructors of the other two backends.

    OCP alone was not enough here, and that cost three quarters of the execution
    price. Candidate code starts with the dialect preamble (``import cadquery``,
    eleven ``from cadgen.*``), and nobody in the whole process tree imported it:
    ``dsl_runtime.configure()`` only edits ``sys.path``. The first real import
    happened inside ``exec(task.code)``, i.e. in the grandchild itself and again
    for every candidate.

    We warm with the preamble from ``dsl_runtime`` rather than our own list of
    names: a second source of truth would let warm-up drift from what actually
    runs.

    The **metrics stack** (`_preload_metrics`) had the same gap as the preamble:
    measurement happens inside the fork, and nobody warmed numpy, scipy, trimesh
    and our metrics module there. The order inside warm-up does not matter; the
    order outside does: numpy reads ``OMP_NUM_THREADS`` and its neighbours **at
    import**, so ``dsl_runtime.apply_thread_limits()`` must run before this call
    (in ``run_experiment.py`` it is the line above).
    """
    _limit_ocp_threads()
    from cad_agent import dsl_runtime  # noqa: PLC0415

    exec(compile(dsl_runtime.code_prefix(), "<preload>", "exec"), {})
    gms_ready = _preload_metrics()
    _assert_preloaded("parent warm-up", metrics=True, gms=gms_ready)


def preload_cad() -> None:
    """Warm up the **parent** of the process tree: the entry point for a run and for measurement.

    Same as `_preload_cad()`, but public and with a different caller. The private
    one is called by processes that fork executors themselves (a shim, the
    constructors of the other two backends); this one is called by the topmost
    process, before it brings up the part pool.

    The point is inheritance: `harness/pool.py` picks `fork` precisely so as not
    to import OCP again in every process, but nobody imported it in the parent --
    `dsl_runtime.configure()` only edits `sys.path`. Without this call each of N
    workers warmed itself, all at once and on its first part.

    The call goes through the module-level name, not by reference: tests replace
    `_preload_cad` with a stub, and the replacement must take effect here too.
    """
    _preload_cad()
    _preload_optimizer()


def _preload_optimizer() -> None:
    """Import the parameter optimizer (`_cad_grad`) in the top process.

    The `optimize` fork spawns a part process, and that process got everything
    from the top process except the optimizer: `import_optimizer()` was first
    called in the fork itself and died with it. The import is cheap, but there is
    no reason to repeat it on every call.

    Only here, not in `_preload_cad`: the execution shims fork candidate code and
    do not need the optimizer. Unavailability is not a warm-up error: the
    `optimize` call itself reports `OptimizerUnavailable` as its own failure, and
    a run without `_cad_grad` is legitimate (`harness/config.py`, tool set).
    """
    from cad_agent import dsl_runtime  # noqa: PLC0415

    try:
        dsl_runtime.import_optimizer()
        dsl_runtime.import_desugar()
    except Exception as exc:  # noqa: BLE001 — the optimize call itself reports the failure
        logger.info("Optimizer not warmed up (%s: %s)", type(exc).__name__, exc)


def required_modules(*, metrics: bool = False, gms: bool = False) -> tuple[str, ...]:
    """What must be in the fork's ``sys.modules`` for a given task composition.

    A separate function because two parties ask: the guard (fail if something is
    missing) and the inheritance check (`tests/exec_inherit_check.py`, which asks
    the fork itself). They must share one list.

    The dialect preamble is always needed -- every candidate's code starts with
    it. The metrics stack only when the fork will measure; the GMS machinery only
    when GMS is part of the measurement.
    """
    from cad_agent import dsl_runtime  # noqa: PLC0415

    names = list(dsl_runtime.preamble_modules())
    if metrics:
        names += list(METRICS_MODULES)
    if gms:
        names += list(GMS_MODULES)
    return tuple(names)


def _assert_preloaded(where: str, *, metrics: bool = False, gms: bool = False) -> None:
    """Check that inherited modules are already in ``sys.modules`` and fail clearly if not.

    An explicit ``raise`` rather than an ``assert`` statement: under ``python -O``
    the latter is stripped, and the check would vanish exactly where a run is
    launched with optimisation. It costs a dict comparison per call.

    The check is needed because a miss here breaks nothing -- it only makes the
    run more expensive. Silent degradation has lived in this code before; it is
    caught not by a test of "how it is now" but by an assertion about how it must
    be.

    ``metrics``/``gms`` are not signature decoration but the task composition:
    requiring the metrics stack from a fork that has nothing to measure would
    fail on a legitimate case (`measure=False`, used by backend benchmarks and
    some checks).
    """
    from cad_agent import dsl_runtime  # noqa: PLC0415

    required = required_modules(metrics=metrics, gms=gms)
    missing = [name for name in required if name not in sys.modules]
    if missing:
        raise AssertionError(
            f"{where}: {len(missing)} of {len(required)} modules not inherited "
            f"(dialect {dsl_runtime.active_dialect()!r}): {', '.join(missing)}. "
            "So the execution fork imports them itself, again for every candidate. "
            "The dialect preamble then costs ~1.9 s per candidate, the metrics stack "
            "(numpy, scipy, trimesh, pykdtree) adds its share, and all of this cost "
            "lands in overhead_sec, where it looks like a floor under the work rather than work. "
            "They were expected to be loaded by _preload_cad() in the parent before the fork."
        )


def _evaluate(task: EvalTask) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {"mesh_path": None, "metrics": None}
    # Fork phases are measured separately: "execution" in the summary is really
    # three different cost items (build the body, export the mesh, measure it
    # against GT), and they must be optimised separately. The total is kept as
    # `wall_sec`, and the sum of phases plus `overhead` equals it identically.
    phases: dict[str, float] = {"build_sec": 0.0, "export_sec": 0.0, "metrics_sec": 0.0}

    # What the fork will do is known before it starts: measurement happens only
    # if there is something to measure and something to compare with, and GMS
    # only if requested. The condition is computed once here and reused below,
    # so that the guard and the measurement cannot disagree.
    will_measure = bool(task.measure and task.gt_mesh_path and task.mesh_path)
    # Mirror of `metrics.measure_pair`: there `want_gms = extended or "gms" in needs`.
    will_need_gms = will_measure and (task.extended or "gms" in (task.needs or ()))

    # The guard goes first, before the thread limiter. The order is not
    # cosmetic: `_limit_ocp_threads()` itself imports `OCP.OSD`, so with a cold
    # parent it would bring up part of the CAD stack in the fork, and the guard
    # would look at state created by its neighbour one line above. The guard must
    # describe what the fork **inherited**, not what it has since been brought to.
    #
    # Warming up in the fork would be pointless: everything it imports is already
    # inherited, and what is not inherited it cannot catch up on -- the price
    # would be paid here anyway.
    _assert_preloaded("execution fork", metrics=will_measure, gms=will_need_gms)
    _limit_ocp_threads()

    phase_started = time.monotonic()
    namespace: dict[str, Any] = {}
    exec(task.code, namespace)
    shape = namespace["r"].val()
    if len(shape.Faces()) <= 2:
        raise ValueError("The built solid is degenerate: at most two faces.")
    phases["build_sec"] = time.monotonic() - phase_started

    is_stl = task.export_format == "stl"
    if task.mesh_path is not None:
        phase_started = time.monotonic()
        Path(task.mesh_path).parent.mkdir(parents=True, exist_ok=True)
        if is_stl:
            tolerance, angular = EXPORT_LADDER[0]
            shape.export(task.mesh_path, tolerance=tolerance, angularTolerance=angular)
            result["export_rung"] = 0
        else:
            shape.export(task.mesh_path)
        result["mesh_path"] = task.mesh_path
        phases["export_sec"] = time.monotonic() - phase_started

    if will_measure and result["mesh_path"] is not None:
        phase_started = time.monotonic()
        # The mesh is not re-read from disk by another process: measure here,
        # while it is at hand. On NFS that is a network round trip per candidate.
        from cad_agent.capabilities import metrics as metrics_mod

        def _measure() -> dict[str, Any]:
            return metrics_mod.measure_pair(
                gt_mesh_path=task.gt_mesh_path,
                pred_mesh_path=result["mesh_path"],
                gt_cache=_GT_CACHE,
                extended=task.extended,
                needs=task.needs,
            )

        # The GT cache is inherited by the fork, so what goes out is the
        # **increment** for this task, not the accumulated counter: otherwise a
        # miss warmed by the shim would arrive again with every grandchild. There
        # may be several measurements (the ladder below), and the increment
        # covers all of them.
        before = metrics_mod.gt_cache_stats(_GT_CACHE)
        result["metrics"] = _measure()

        # Tolerance ladder. An open mesh is a failure of the WHOLE candidate, not
        # a slightly worse metric: IoU is undefined on it. Before the ladder
        # existed, a valid body was lost to OCC tessellation. The ladder is paid
        # for ONLY on failure: a body closed at the first rung incurs no extra
        # export and no extra measurement.
        if is_stl and not result["metrics"].get("pred_watertight", True):

            def _export(rung: int) -> None:
                """One ladder rung. Export is export, not measurement: its time
                goes to `export_sec`, and the start of the measurement phase is
                shifted by the same amount, otherwise `metrics_sec` would absorb
                the writer's work. `rung == len(EXPORT_LADDER)` is the fallback
                rung `EXPORT_FALLBACK`.
                """
                nonlocal phase_started
                export_started = time.monotonic()
                if rung == len(EXPORT_LADDER):
                    # A copy without triangulation: the shape itself has its
                    # triangulation cached by the ladder, and an absolute tolerance
                    # on top of it would not remesh everything.
                    tolerance, angular = EXPORT_FALLBACK
                    shape.copy().exportStl(task.mesh_path, tolerance=tolerance,
                                           angularTolerance=angular, relative=False)
                else:
                    tolerance, angular = EXPORT_LADDER[rung]
                    shape.export(task.mesh_path, tolerance=tolerance, angularTolerance=angular)
                spent = time.monotonic() - export_started
                phases["export_sec"] = phases.get("export_sec", 0.0) + spent
                phase_started += spent

            for rung in range(1, len(EXPORT_LADDER) + 1):
                _export(rung)
                retry = _measure()
                if retry.get("pred_watertight"):
                    result["metrics"] = retry
                    result["export_rung"] = rung
                    break
            else:
                # No rung helped. The mesh on disk must be the one the returned
                # metrics refer to, otherwise the run artifact would disagree with
                # the number computed from it.
                _export(0)

        after = metrics_mod.gt_cache_stats(_GT_CACHE)
        result["gt_cache"] = {
            field: after.get(field, 0) - before.get(field, 0) for field in ("hits", "misses")
        }
        phases["metrics_sec"] = time.monotonic() - phase_started

    wall = time.monotonic() - started
    result["wall_sec"] = wall
    # The remainder is not "other" but concrete work: the inheritance guard and
    # the thread limiter. There is no import here any more -- neither the
    # preamble nor the metrics stack: both are inherited from the parent, so the
    # remainder shows exactly what was not an import. It is computed by
    # subtraction, so the breakdown adds up to `wall_sec` with no gap.
    phases["overhead_sec"] = max(wall - sum(phases.values()), 0.0)
    phases["n"] = 1
    result["phases"] = phases
    # The fork counts its own threads rather than an external sampler. The
    # execution fork lives a second or so, while a background sampler runs once
    # every few seconds -- it catches such a process in about a quarter of cases,
    # and the execution's contribution to the peak stays invisible. Here the
    # figure is exact and costs one read.
    result["n_threads"] = own_threads()
    return result


def own_threads() -> int | None:
    """Number of threads in the current process. `None` if /proc is unavailable.

    This is the value **at the end of the work**, not the peak: pools are created
    once and live until the process ends, so one reading is enough. Catching the
    real peak would mean starting another thread in the fork just to count --
    a price the measurement is not worth.

    Public because both forks call it: execution and det. Any short-lived process
    must count itself -- a background sampler runs once every few seconds and
    simply misses such processes.
    """
    try:
        raw = Path("/proc/self/stat").read_text()
        return int(raw[raw.rindex(")") + 2:].split()[17])
    except (OSError, ValueError, IndexError):
        return None


def _warm_gt(gt_mesh_path: str) -> bool:
    """Warm the GT of a part, evicting the previous one.

    Eviction is mandatory since shims live for the whole worker process rather
    than one part: a shim needs exactly one GT -- of the part it is currently
    processing. Without eviction the dict would accumulate a mesh and an
    8192-point cloud for every part of the run, in every shim.

    The hit counters (`GT_CACHE_STATS_KEY`) survive eviction: they describe the
    cache's work over its whole life, not its contents.
    """
    from cad_agent.capabilities import metrics as metrics_mod

    for key in [key for key in _GT_CACHE if key != metrics_mod.GT_CACHE_STATS_KEY]:
        if key != gt_mesh_path:
            del _GT_CACHE[key]
    metrics_mod.warm_gt_iou(metrics_mod.load_gt_cached(gt_mesh_path, _GT_CACHE))
    return True


# --- backends ----------------------------------------------------------------


class Executor:
    """Common interface: take a list of tasks, return results in the same order."""

    def evaluate(self, tasks: list[EvalTask]) -> list[EvalResult]:
        raise NotImplementedError

    def warm_gt(self, gt_mesh_path: str) -> None:
        """Preload GT where it will outlive a task (proxy_pool only)."""

    def stats(self) -> dict[str, int]:
        """What happened to the executor itself over its lifetime.

        Separate from task results: a shim restart is an event of the executor,
        not of any one task, and cannot be reconstructed from the results.
        """
        return {}

    def close(self) -> None:
        pass

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _to_result(task: EvalTask, ok: bool, payload: Any, outcome: str, wall: float) -> EvalResult:
    if ok:
        return EvalResult(
            task_id=task.task_id,
            success=True,
            mesh_path=payload.get("mesh_path"),
            metrics=payload.get("metrics"),
            wall_sec=payload.get("wall_sec", wall),
            outcome=OUTCOME_OK,
            n_threads=payload.get("n_threads"),
            gt_cache=payload.get("gt_cache"),
            export_rung=payload.get("export_rung"),
            phases_sec=payload.get("phases"),
        )
    result = EvalResult(
        task_id=task.task_id,
        success=False,
        error=str(payload),
        wall_sec=wall,
        timed_out=outcome == OUTCOME_TIMEOUT,
        outcome=outcome,
    )
    if result.worker_died:
        # An executor's death must be visible in the process log too, not only in
        # the counters: on a run of a thousand parts it is the first sign that the
        # machine is in trouble, and must not be learned after the fact from a report.
        #logger.warning("Executor died on task %s (%s): %s", task.task_id, outcome, payload)
        pass
    return result


class SerialForkExecutor(Executor):
    """A fork per candidate, one at a time, straight from the part process."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SEC, preload: bool = True):
        self.timeout = timeout
        if preload:
            _preload_cad()

    def evaluate(self, tasks: list[EvalTask]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for task in tasks:
            started = time.monotonic()
            ok, payload, outcome = run_in_fork(_evaluate, (task,), self.timeout)
            results.append(_to_result(task, ok, payload, outcome, time.monotonic() - started))
        return results


class EphemeralPoolExecutor(Executor):
    """Up to k concurrent children; each lives exactly one task."""

    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        preload: bool = True,
    ):
        self.pool_size = max(1, pool_size)
        self.timeout = timeout
        if preload:
            _preload_cad()

    def evaluate(self, tasks: list[EvalTask]) -> list[EvalResult]:
        from concurrent.futures import ThreadPoolExecutor as _Threads

        if not tasks:
            return []

        # Threads here only watch the forks: the work itself runs in separate
        # processes, so the GIL is no obstacle -- a thread sits in select().
        with _Threads(max_workers=min(self.pool_size, len(tasks))) as threads:
            return list(threads.map(self._run_one, tasks))

    def _run_one(self, task: EvalTask) -> EvalResult:
        started = time.monotonic()
        ok, payload, outcome = run_in_fork(_evaluate, (task,), self.timeout)
        return _to_result(task, ok, payload, outcome, time.monotonic() - started)


class ProxyPoolExecutor(Executor):
    """k long-lived shims; the code is executed by a grandchild forked by them.

    A shim is created once per worker process (`build_executor` hands out a
    `SharedProxyView` over a shared pool) and holds OCP together with the dialect
    preamble. Every grandchild then inherits both the import and the GT cache.
    The import is expensive and saves most of the cost of **every** candidate.

    The pool is warmed **before** the shims are forked, in the constructor --
    like the two neighbouring backends. Otherwise cold-start cost grows with k:
    every shim imports OCP itself, and with 16 workers and 8 shims that is 128
    simultaneous imports instead of 16.

    The pool used to be per-part by oversight, not by design: a shim knows
    nothing about the part -- it holds libraries and forks. While the pool was
    built per part, a run of 1000 parts with `pool_size: 4` paid 4000 cold
    imports and created 8000 queues. Measurement confirmed it: `proxy_pool`
    degraded as k grew, while `ephemeral_pool` improved.

    **Each shim has its own pair of queues, not one shared by the pool.** This
    is not a matter of taste: `mp.Queue` is protected by an inter-process lock,
    and a shim killed by a signal inside `get()` takes that lock with it for
    good. A shared queue is dead for everyone after such a death -- a new shim
    got stuck on it, so a restart cured the symptom and left the pool unusable.
    Private queues die with their shim and are created anew.

    Tasks are dispatched by the parent, one per free shim. That way it is known
    **which** task the dead shim had: it alone is counted as lost, the rest
    arrive.
    """

    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        timeout: float = DEFAULT_TIMEOUT_SEC,
    ):
        self.pool_size = max(1, pool_size)
        self.timeout = timeout
        self.restarts = 0
        self.lost = 0
        # Tokens run across the whole pool and are not reset between parts:
        # that is their whole point (see `_dispatch`).
        self._seq = 0
        self._ctx = mp.get_context("fork")
        self._procs: list[Any] = [None] * self.pool_size
        self._task_queues: list[Any] = [None] * self.pool_size
        self._result_queues: list[Any] = [None] * self.pool_size
        self._gt_mesh_path: str | None = None
        # Warm up BEFORE forking the shims, not in each of them. A shim calls
        # `_preload_cad()` itself (`_proxy_loop`), but after this line it resolves
        # from `sys.modules` and costs microseconds. Without warm-up here, the
        # pool's cold start cost k independent imports of OCP and the preamble per
        # worker process -- against one for the two neighbouring backends, which
        # warm up right in the constructor. With `n_workers: 16` and `pool_size: 8`
        # that is 128 simultaneous imports, and `warm_gt` stopped waiting for its
        # acknowledgement within `timeout + _PROXY_GRACE_SEC`.
        #
        # The objection "shims must be forked while the process is thin" is
        # answered by copy-on-write: the shim brings up OCP itself anyway, and the
        # inherited pages are shared, so the grandchild gets the same ones. Only
        # the worker gets fatter -- exactly as in `serial_fork` and `ephemeral_pool`.
        _preload_cad()
        for position in range(self.pool_size):
            self._spawn(position)

    def _spawn(self, position: int) -> None:
        """Bring up a shim with a fresh pair of queues."""
        task_queue = self._ctx.Queue()
        result_queue = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_proxy_loop,
            args=(task_queue, result_queue, self.timeout),
            daemon=True,
        )
        proc.start()
        self._procs[position] = proc
        self._task_queues[position] = task_queue
        self._result_queues[position] = result_queue

    def _restart(self, position: int) -> None:
        proc = self._procs[position]
        logger.error(
            "Execution pool proxy died (pid=%s, exitcode=%s), restarting it",
            getattr(proc, "pid", None), getattr(proc, "exitcode", None),
        )
        self.restarts += 1
        self._spawn(position)
        # The GT cache lived in the dead shim; the fresh one must warm it,
        # otherwise its first grandchildren would each pay for loading GT.
        if self._gt_mesh_path is not None:
            self._warm_one(position, self._gt_mesh_path)

    def _warm_one(self, position: int, gt_mesh_path: str) -> None:
        self._task_queues[position].put(("warm", gt_mesh_path))
        try:
            self._result_queues[position].get(timeout=self.timeout + _PROXY_GRACE_SEC)
        except queue_mod.Empty:
            logger.warning("Proxy %d did not confirm the GT cache warm-up", position)

    def warm_gt(self, gt_mesh_path: str) -> None:
        """Warm the GT cache in all shims before a rollout.

        With private queues warm-up is exact: each shim gets exactly one job. The
        former shared-queue scheme gave no such guarantee -- a fast shim could take
        two.
        """
        self._gt_mesh_path = str(gt_mesh_path)
        for position in range(self.pool_size):
            self._task_queues[position].put(("warm", self._gt_mesh_path))
        for position in range(self.pool_size):
            try:
                self._result_queues[position].get(timeout=self.timeout + _PROXY_GRACE_SEC)
            except queue_mod.Empty:
                logger.warning("Proxy %d did not confirm the GT cache warm-up", position)

    def evaluate(self, tasks: list[EvalTask]) -> list[EvalResult]:
        if not tasks:
            return []

        collected: dict[int, EvalResult] = {}
        pending = deque(range(len(tasks)))
        in_flight: dict[int, int] = {}  # shim -> token of the task issued to it
        tokens: dict[int, int] = {}  # token -> task index in this call
        deadline = time.monotonic() + self._patience(len(tasks))

        next_reap = 0.0
        while len(collected) < len(tasks) and time.monotonic() < deadline:
            # Order matters: first bring back the dead -- otherwise a dead shim
            # simply receives no tasks, nobody learns of its death, and the pool
            # quietly loses a third of its capacity until the part ends. But there
            # is no need to check liveness on every turn: turns are now frequent,
            # while a shim's death is rare and not urgent.
            now = time.monotonic()
            if now >= next_reap:
                self._reap_dead(in_flight, tasks, collected, tokens)
                next_reap = now + _PROXY_REAP_SEC
            self._dispatch(pending, in_flight, tasks, tokens)
            if self._collect(in_flight, tasks, collected, tokens):
                continue
            if not in_flight and not pending:
                break
            time.sleep(_PROXY_POLL_SEC)

        # Everything that did not arrive: a dead shim's task, or one never started.
        for index, task in enumerate(tasks):
            if index in collected:
                continue
            self.lost += 1
            logger.error("Task %s lost: the pool proxy returned no result", task.task_id)
            collected[index] = EvalResult(
                task_id=task.task_id,
                success=False,
                error="Execution pool proxy died, task lost.",
                outcome=OUTCOME_LOST,
            )
        return [collected[index] for index in range(len(tasks))]

    def _reap_dead(
        self,
        in_flight: dict[int, int],
        tasks: list[EvalTask],
        collected: dict,
        tokens: dict[int, int],
    ) -> None:
        """Restart dead shims; declare their tasks lost at once.

        **All** shims are checked, not only busy ones: a shim can die while idle
        (the OOM killer does not choose by load), and an unnoticed dead shim never
        receives tasks again.
        """
        for position in range(self.pool_size):
            if self._procs[position].is_alive():
                continue
            token = in_flight.pop(position, None)
            if token is not None:
                index = tokens[token]
                self.lost += 1
                task = tasks[index]
                logger.error(
                    "Task %s lost: the pool shim died without returning a result", task.task_id
                )
                collected[index] = EvalResult(
                    task_id=task.task_id,
                    success=False,
                    error="Execution pool proxy died, task lost.",
                    outcome=OUTCOME_LOST,
                )
            self._restart(position)

    def _dispatch(
        self,
        pending: deque,
        in_flight: dict[int, int],
        tasks: list[EvalTask],
        tokens: dict[int, int],
    ) -> None:
        """Hand tasks to free live shims, one each.

        A task goes out with a **token** -- a number unique over the whole life of
        the pool -- rather than its index in the list. Indices are positional:
        every part starts again at 0, 1, 2. While the pool died with the part,
        there was nothing to confuse them with; now shims live for the whole
        worker process, and the result of a task abandoned as lost on the previous
        part arrives during the next one. With an index it would be
        indistinguishable from that part's own result -- foreign geometry and
        foreign metrics would silently take the place of the real ones.
        """
        for position in range(self.pool_size):
            if not pending:
                return
            if position in in_flight or not self._procs[position].is_alive():
                continue
            index = pending.popleft()
            self._seq += 1
            token = self._seq
            tokens[token] = index
            in_flight[position] = token
            self._task_queues[position].put(("eval", (token, tasks[index])))

    def _collect(
        self,
        in_flight: dict[int, int],
        tasks: list[EvalTask],
        collected: dict,
        tokens: dict[int, int],
    ) -> bool:
        """Collect what is ready; restart dead shims. True means something arrived.

        Only a reply **to the task issued to this shim** is accepted: the token is
        checked. Everything else -- a late warm-up acknowledgement or the result of
        a task already declared lost -- is discarded. Formerly the task index was
        taken straight from the message, and any stray message in the queue
        replaced the real result.
        """
        progress = False
        for position in list(in_flight):
            try:
                token, ok, payload, outcome, wall = self._result_queues[position].get_nowait()
            except queue_mod.Empty:
                continue  # a shim's death is handled by `_reap_dead`
            if token == _WARM_ACK_INDEX:
                # A late warm-up acknowledgement. `warm_gt` waits for it for a
                # bounded time and on a cold start (the shim is still bringing up
                # OCP) does not get it -- but the reply arrives later and stays in
                # the queue. Without this branch it was read as a task result:
                # `collected[-1]` inflated the done counter, and
                # `del in_flight[position]` dropped the **real** task from
                # accounting, so it was declared lost.
                continue
            if in_flight.get(position) != token:
                # A reply not to this shim's current task: it finished an
                # abandoned task from a previous part. Tokens are global, so the
                # foreign reply is recognised at once; with a positional index it
                # would silently take the place of the real result.
                logger.warning(
                    "Proxy %d answered a foreign ticket %s, discarding", position, token
                )
                continue
            index = tokens[token]
            collected[index] = _to_result(tasks[index], ok, payload, outcome, wall)
            del in_flight[position]
            progress = True
        return progress

    def _patience(self, n_tasks: int) -> float:
        """How long to wait for all results before treating them as lost.

        Each task is already bounded by its own timeout inside the shim, so all
        that is needed on top is slack for the queue: there may be more tasks than
        shims, and they go in waves.
        """
        waves = math.ceil(n_tasks / self.pool_size)
        return waves * (self.timeout + _PROXY_GRACE_SEC) + _PROXY_GRACE_SEC

    def stats(self) -> dict[str, int]:
        return {"proxy_restarts": self.restarts, "lost_tasks": self.lost}

    def close(self) -> None:
        for position, proc in enumerate(self._procs):
            if proc is None:
                continue
            try:
                self._task_queues[position].put(("stop", None))
            except Exception:
                pass
        for proc in self._procs:
            if proc is None:
                continue
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()


def _proxy_loop(task_queue: Any, result_queue: Any, timeout: float) -> None:
    """Shim body: executes nothing itself, only forks a grandchild."""
    try:
        _preload_cad()
    except Exception:
        logger.exception("Proxy failed to warm up: OCP or the dialect preamble")

    while True:
        kind, payload = task_queue.get()
        if kind == "stop":
            return
        if kind == "warm":
            try:
                _warm_gt(payload)
            except Exception:
                logger.exception("Could not warm the GT cache for %s", payload)
            result_queue.put((_WARM_ACK_INDEX, True, {"mesh_path": None}, False, 0.0))
            continue

        token, task = payload
        started = time.monotonic()
        ok, result, outcome = run_in_fork(_evaluate, (task,), timeout)
        result_queue.put((token, ok, result, outcome, time.monotonic() - started))


BACKENDS = {
    "serial_fork": SerialForkExecutor,
    "ephemeral_pool": EphemeralPoolExecutor,
    "proxy_pool": ProxyPoolExecutor,
}


# Process-wide shared pool of shims. A shim knows nothing about the part: it
# holds an imported OCP with the dialect preamble and forks a grandchild -- and
# neither depends on the part. When the pool was built per part, every part
# brought up k processes with a cold import again: on a 1000-part run with
# `pool_size: 4` that is 4000 cold imports and 8000 queues for nothing. The
# worker process lives for the whole run (`max_tasks_per_child` is not set), so
# the pool outlives the part together with it.
_SHARED_PROXY_POOL: ProxyPoolExecutor | None = None
_SHARED_PROXY_KEY: tuple[int, float] | None = None


class SharedProxyView(Executor):
    """A per-part view of the shared pool of shims.

    Needed for two things that must stay per-part now that the pool itself is not:

    - ``close()`` no longer stops the shims. The part is over, the worker process
      lives on, and the next part gets an already-warm pool.
    - ``stats()`` returns the **increment** for this part, not the counter for
      the worker's whole life. Otherwise restarts and losses on the first part
      would arrive in ``tech.json`` of every following one and be summed in the
      run summary once per part that followed.
    """

    def __init__(self, pool: "ProxyPoolExecutor"):
        self._pool = pool
        self._baseline = dict(pool.stats())

    def evaluate(self, tasks: list[EvalTask]) -> list[EvalResult]:
        return self._pool.evaluate(tasks)

    def warm_gt(self, gt_mesh_path: str) -> None:
        self._pool.warm_gt(gt_mesh_path)

    def stats(self) -> dict[str, int]:
        current = self._pool.stats()
        return {key: value - self._baseline.get(key, 0) for key, value in current.items()}

    def close(self) -> None:
        """Nothing: the shims belong to the process, not to the part."""


def shutdown_shared_executors() -> None:
    """Stop the process's shared pool. A run does not need this -- shims are
    daemonic and die with the worker; tests do, where the process lives on."""
    global _SHARED_PROXY_POOL, _SHARED_PROXY_KEY

    if _SHARED_PROXY_POOL is not None:
        _SHARED_PROXY_POOL.close()
    _SHARED_PROXY_POOL = None
    _SHARED_PROXY_KEY = None


def build_executor(
    backend: str = "proxy_pool",
    pool_size: int = DEFAULT_POOL_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Executor:
    global _SHARED_PROXY_POOL, _SHARED_PROXY_KEY

    if backend not in BACKENDS:
        raise ValueError(f"Unknown execution backend {backend!r}. Expected one of {sorted(BACKENDS)}")
    if backend == "serial_fork":
        return SerialForkExecutor(timeout=timeout)
    if backend != "proxy_pool":
        return BACKENDS[backend](pool_size=pool_size, timeout=timeout)

    # Pool size and timeout come from the config and do not change during a run;
    # the key check guards against reusing a pool with foreign parameters (as
    # happens in tests that run different configurations in one process).
    key = (int(pool_size), float(timeout))
    if _SHARED_PROXY_POOL is not None and _SHARED_PROXY_KEY != key:
        shutdown_shared_executors()
    if _SHARED_PROXY_POOL is None:
        _SHARED_PROXY_POOL = ProxyPoolExecutor(pool_size=pool_size, timeout=timeout)
        _SHARED_PROXY_KEY = key
    return SharedProxyView(_SHARED_PROXY_POOL)
