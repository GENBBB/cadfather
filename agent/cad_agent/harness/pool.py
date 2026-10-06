"""Parallel run over parts: N processes and a job queue.

One part is one process, with a sequential rollout inside it. The N processes
themselves fill the batch of requests to vLLM: the server batches them on its
side, so there is no artificial batching in the client.

The start method is `fork` only, and this is not an implementation detail: with
`fork` the scaffold object loaded in the parent is inherited by workers for
free. With `spawn` it would have to be passed by value and OCP re-imported in
every process.

The pool is built on `ProcessPoolExecutor`, not `multiprocessing.Pool`, for a
hard reason: `Pool` workers are **daemonic**, and a daemonic process may not
have children. The `proxy_pool` execution backend spawns shims inside the part
process and would fail with `daemonic processes are not allowed to have
children`, so the default backend would not work at all. `ProcessPoolExecutor`
workers are non-daemonic and may have children.

The pool survives a worker death: OCC can kill the whole process, and over a
thousand parts that is a matter of time. `BrokenProcessPool` does not abort the
run -- the pool is rebuilt, the part that was in progress is counted as a
failure, the rest complete.

There is no longer a watchdog that kills workers for lack of progress, and it
must not be brought back. It did not and could not work: a part worker forks its
execution shims, `proc.kill()` hits only the worker itself, and the shims live
on as orphans -- holding open the executor's **sentinels**, which is exactly the
signal `ProcessPoolExecutor` uses to learn a worker died. The result was a run
that never finished: zombie workers, live orphans, the manager thread stuck in
`mp.connection.wait` forever.

Instead of a watchdog, the part limits itself: `experiment.figure_wall_sec` is
checked inside the part process before candidates are spawned
(`Budget.check_wall`). Nobody needs to be killed; the rollout stops normally
and returns what it found.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from cad_agent.harness import progress as progress_mod
from cad_agent.harness.dataset import FigureSpec
from cad_agent.harness.figure_run import run_figure

logger = logging.getLogger(__name__)

# Inherited by workers through fork; not passed as an argument.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(scaffold: Any, config: dict[str, Any], run_dir: str) -> None:
    _WORKER_STATE["scaffold"] = scaffold
    _WORKER_STATE["config"] = config
    _WORKER_STATE["run_dir"] = run_dir


def _worker_run(spec: FigureSpec) -> dict[str, Any]:
    try:
        return run_figure(
            spec=spec,
            config=_WORKER_STATE["config"],
            scaffold=_WORKER_STATE["scaffold"],
            run_dir=_WORKER_STATE["run_dir"],
        )
    except Exception as exc:  # a worker must not die silently
        # The exception is caught in place and the process is alive: this is a
        # part failure, not a worker death, and the two must not be mixed.
        logger.exception("Worker crashed on part %s", spec.figure_id)
        return _failed_record(spec, exc, worker_died=False)


# Worker deaths per run. Filled by `_run_pooled`, read by the harness after the
# run: `BrokenProcessPool` used to go only to the log, and a part lost with its
# worker reached the table as an ordinary failure, indistinguishable from
# "the model failed".
POOL_DEATHS: dict[str, int] = {}


def pool_deaths() -> dict[str, int]:
    """What happened to part workers during the run."""
    return dict(POOL_DEATHS)


def _count_death(kind: str, count: int = 1) -> None:
    POOL_DEATHS[kind] = POOL_DEATHS.get(kind, 0) + count


def run_figures(
    figures: list[FigureSpec],
    config: dict[str, Any],
    scaffold: Any,
    run_dir: str | Path,
    n_workers: int = 8,
) -> list[dict[str, Any]]:
    """Run all parts and return records **in the order of `figures`**.

    The order is mandatory: the contract `run_eval` returns per-part scores in
    the order of the input list, and keeping that order is part of the contract.
    """
    if not figures:
        return []

    POOL_DEATHS.clear()
    n_workers = max(1, min(n_workers, len(figures)))
    logger.info(
        "Run start: parts=%d, workers=%d, cores=%d, execution backend=%s",
        len(figures),
        n_workers,
        os.cpu_count() or 0,
        config.get("execution", {}).get("backend", "proxy_pool"),
    )
    started = time.monotonic()

    if n_workers == 1:
        # Debug path: no pool, exceptions are visible as is. Rendering then
        # opens a VTK window right here in the parent -- see `_check_no_plotter`.
        _worker_init(scaffold, config, str(run_dir))
        records = {}
        with progress_mod.build(len(figures), config.get("logging")) as bar:
            for index, spec in enumerate(figures):
                records[index] = _worker_run(spec)
                bar.advance(records[index])
    else:
        _check_no_plotter()
        records = _run_pooled(figures, config, scaffold, str(run_dir), n_workers)

    wall = time.monotonic() - started
    logger.info(
        "Run finished: parts=%d, time=%.1f s, parts per hour=%.1f",
        len(figures),
        wall,
        3600.0 * len(figures) / max(wall, 1e-9),
    )
    return [records[index] for index in range(len(figures))]


def _check_no_plotter() -> None:
    """A VTK window opened in this process kills every forked worker.

    Measured, not assumed: the parent creates a `pv.Plotter`, forks a child, the
    child creates its own and crashes with `malloc(): unaligned tcache chunk
    detected` -- heap corruption, not an exception. Upstream this shows up as
    "worker returned no result" for ALL parts in a row, and the cause cannot be
    found from that symptom: none of our messages appear in the log.

    A real run does not hit this: the parent warms up OCP and metrics but does
    not render. The trap awaits two future edits -- warming up rendering in the
    parent (tempting: rendering is a third of the wall time) and a run where
    `n_workers: 1` ran before a pooled one in the same process. Both are silent,
    so the check lives here and not in a comment.

    We ask via `sys.modules`, not by import: if rendering was not imported there
    is certainly no window, and pulling in pyvista just to answer is pointless.
    """
    render = sys.modules.get("cad_agent.capabilities.render")
    if render is not None and render.plotter_open_here():
        raise RuntimeError(
            "A VTK window (pyvista.Plotter) is already open in this process, and the part pool "
            "starts via fork: every worker will die on its plotter from heap corruption, "
            "silently. Rendering in the parent and forking workers do not mix: either do not open "
            "the window before the pool, or move rendering to a separate process."
        )


def _run_pooled(
    figures: list[FigureSpec],
    config: dict[str, Any],
    scaffold: Any,
    run_dir: str,
    n_workers: int,
) -> dict[int, dict[str, Any]]:
    """Run in a pool with recovery after a worker death."""
    ctx = mp.get_context("fork")
    records: dict[int, dict[str, Any]] = {}
    pending = list(enumerate(figures))
    # The progress bar is created outside the `while`: a pool collapse and a
    # repeated round continue the same run, not a new one. A bar inside the loop
    # would count a part twice and show progress going back where it does not.
    bar = progress_mod.build(len(figures), config.get("logging"))
    while pending:
        done_before = len(records)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(scaffold, config, run_dir),
        ) as pool:
            futures = {pool.submit(_worker_run, spec): (index, spec) for index, spec in pending}
            crashed: list[tuple[int, FigureSpec]] = []
            broken = False
            try:
                for future in as_completed(futures):
                    index, spec = futures[future]
                    try:
                        records[index] = future.result()
                        bar.advance(records[index])
                    except BrokenProcessPool:
                        # A worker death aborts ALL unfinished tasks, not only
                        # the one that killed it: the executor sets this error
                        # on every unfinished future. Recording them as failures
                        # would declare many innocent parts failed -- they go to
                        # the next round on a fresh pool.
                        broken = True
                    except Exception as exc:
                        logger.exception("Part %s returned no result", spec.figure_id)
                        _count_death("figure_no_result")
                        records[index] = _failed_record(spec, exc)
                        bar.advance(records[index])
            except BrokenProcessPool as exc:
                # Same thing, but the error came from `as_completed` itself.
                logger.error("The worker pool broke down, restarting it: %s", exc)
                broken = True

            if broken:
                _count_death("pool_broken")
                crashed = [
                    (index, spec)
                    for future, (index, spec) in futures.items()
                    if index not in records
                ]
                logger.error(
                    "Worker pool broke down, %d parts go to the next round", len(crashed)
                )

        if not crashed:
            break

        # One of the parts that did not return killed the worker, but which one
        # is unknown, so the round is repeated on a fresh pool. The exit
        # condition is no progress: if a round produced no new record, repeating
        # it is pointless and the loop would become endless.
        if len(records) == done_before:
            logger.error(
                "Pool restart made no progress, %d parts counted as failures", len(crashed)
            )
            _count_death("figure_lost_with_worker", len(crashed))
            for index, spec in crashed:
                records[index] = _failed_record(spec, RuntimeError("worker died, part did not finish"))
                bar.advance(records[index])
            break

        pending = crashed

    bar.close()
    for index, spec in enumerate(figures):
        if index not in records:
            _count_death("figure_unprocessed")
            records[index] = _failed_record(spec, RuntimeError("part was not processed"))
    return records


def _failed_record(spec: FigureSpec, exc: BaseException, worker_died: bool = True) -> dict[str, Any]:
    """A part that produced no result.

    `worker_died` distinguishes a loss together with the process from an
    exception caught in place: in the table both give zero but explain different
    things -- the first is about the machine, the second about the code. Default
    True because all pool paths arrive here after a process death.
    """
    return {
        "worker_died": worker_died,
        "figure_id": spec.figure_id,
        "group": spec.group,
        "gt_mesh_path": str(spec.gt_mesh_path),
        "mesh_path": None,
        "n_steps": 0,
        "stop_reason": "worker returned no result",
        "error": repr(exc),
        "runtime_metrics": {},
        "metrics": None,
        "score": 0.0,
        "cost": {},
        "wall_sec": 0.0,
    }
