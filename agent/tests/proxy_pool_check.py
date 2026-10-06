#!/usr/bin/env python3
"""Shims live with the process, not with the part, and this confuses nothing.

Why. A shim knows nothing about a part: it holds an imported OCP with the dialect
preamble and forks a grandchild. While the pool was built per part, every part started
k processes with a cold import again: in a run of 1000 parts with `pool_size: 4` that
is 4000 cold imports and 8000 queues for nothing. The worker process lives the whole
run, so the pool can too.

But a pool that outlives a part brings three things that could not exist before, and
each of them is silent:

1. counters (restarts, losses) become cumulative, and a part's `tech.json` must get
   **its own** increment, otherwise one death on the first part would travel into every
   following part and be summed up in the summary;
2. the GT cache in a shim would accumulate a mesh and an 8192-point cloud for every part
   of the run, in every shim;
3. the result of a task abandoned on the previous part as lost arrives during the next
   one. Task ids are positional (every part starts again at 0, 1, 2), so a foreign result
   would silently take the place of the real one: foreign geometry and foreign metrics
   on someone else's part.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from cad_agent.capabilities import execute  # noqa: E402
from cad_agent.capabilities.execute import (  # noqa: E402
    OUTCOME_OK,
    EvalTask,
    SharedProxyView,
    build_executor,
    shutdown_shared_executors,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _fine(task=None) -> dict:
    return {"mesh_path": "/tmp/missing.stl", "metrics": {}, "wall_sec": 0.0, "n_threads": 1}


def main() -> None:
    execute._PROXY_GRACE_SEC = 1.0
    execute._preload_cad = lambda: None
    execute._evaluate = _fine

    print("1. One pool per process; a part does not shut it down")
    first = build_executor(backend="proxy_pool", pool_size=2, timeout=5.0)
    check("a view of the shared pool is returned, not the pool itself", isinstance(first, SharedProxyView))
    pool = first._pool
    pids = [proc.pid for proc in pool._procs]

    results = first.evaluate([EvalTask(task_id="part1", code="r = r.box(1)")])
    check("the first part is evaluated", results[0].success, results[0].outcome)

    # The part is over: the harness calls close() in a finally block.
    first.close()
    check("closing a part does not stop the proxies", all(proc.is_alive() for proc in pool._procs))

    second = build_executor(backend="proxy_pool", pool_size=2, timeout=5.0)
    check("the second part got the same pool", second._pool is pool)
    check("same proxies, not restarted",
          [proc.pid for proc in second._pool._procs] == pids, str(pids))
    results = second.evaluate([EvalTask(task_id="part2", code="r = r.box(2)")])
    check("the second part is evaluated on warmed-up proxies", results[0].success, results[0].outcome)

    print("\n2. Counters are per part, not cumulative")
    # The pool counter is bumped artificially: what matters is that the view does not
    # show it, not why it grew.
    pool.restarts += 3
    pool.lost += 1
    third = build_executor(backend="proxy_pool", pool_size=2, timeout=5.0)
    check("a part that started after the events does not inherit them",
          third.stats() == {"proxy_restarts": 0, "lost_tasks": 0}, str(third.stats()))
    pool.restarts += 2
    check("the part sees its own increment",
          third.stats().get("proxy_restarts") == 2, str(third.stats()))
    check("the pool counter stayed cumulative",
          pool.stats().get("proxy_restarts") == 5, str(pool.stats()))

    print("\n3. A stale ticket does not replace the real result")
    # Exactly the case a long-lived pool creates: a shim finished a task abandoned on
    # the previous part and puts the answer into the queue. Its task id is positional
    # and coincides with the current one.
    fourth = build_executor(backend="proxy_pool", pool_size=1, timeout=5.0)
    stale = {"mesh_path": "/tmp/FOREIGN.stl", "metrics": {"iou": 0.999}, "wall_sec": 9.9}
    fourth._pool._result_queues[0].put((10**9, True, stale, OUTCOME_OK, 9.9))
    time.sleep(0.5)
    results = fourth.evaluate([EvalTask(task_id="own", code="r = r.box(3)")])
    check("exactly one result returned", len(results) == 1, str(len(results)))
    check("this is the result of its own task", results[0].task_id == "own", results[0].task_id)
    check("foreign geometry was not substituted",
          results[0].mesh_path != "/tmp/FOREIGN.stl", str(results[0].mesh_path))
    check("foreign metrics were not substituted",
          (results[0].metrics or {}).get("iou") != 0.999, str(results[0].metrics))
    check("no losses were recorded",
          fourth.stats().get("lost_tasks", 0) == 0, str(fourth.stats()))

    print("\n4. A proxy keeps the GT of one part, not of all of them")
    from cad_agent.capabilities import metrics as metrics_mod

    loaded: list[str] = []

    def fake_load(path, cache, n_points=None):
        loaded.append(path)
        cache[path] = {"mesh": None, "points": None}
        return cache[path]

    real_load = metrics_mod.load_gt_cached
    metrics_mod.load_gt_cached = fake_load
    try:
        execute._GT_CACHE.clear()
        execute._GT_CACHE[metrics_mod.GT_CACHE_STATS_KEY] = {"hits": 7, "misses": 3}
        for path in ("/sets/00000001.stl", "/sets/00000002.stl", "/sets/00000003.stl"):
            execute._warm_gt(path)
        meshes = [key for key in execute._GT_CACHE if key != metrics_mod.GT_CACHE_STATS_KEY]
        check("one part is left in the cache", meshes == ["/sets/00000003.stl"], str(meshes))
        check("hit counters survived eviction",
              execute._GT_CACHE.get(metrics_mod.GT_CACHE_STATS_KEY) == {"hits": 7, "misses": 3},
              str(execute._GT_CACHE.get(metrics_mod.GT_CACHE_STATS_KEY)))
        check("each part was warmed up exactly once", len(loaded) == 3, str(loaded))
    finally:
        metrics_mod.load_gt_cached = real_load
        execute._GT_CACHE.clear()

    print("\n5. A pool with different parameters is not reused")
    other = build_executor(backend="proxy_pool", pool_size=3, timeout=5.0)
    check("a different pool size started its own pool", other._pool is not pool)
    check("it has as many proxies as requested", len(other._pool._procs) == 3)
    check("the previous pool is shut down", all(not proc.is_alive() for proc in pool._procs))

    shutdown_shared_executors()
    check("shutdown stops the shared pool", all(not proc.is_alive() for proc in other._pool._procs))

    print("\n6. The other backends stay per part")
    a = build_executor(backend="ephemeral_pool", pool_size=2, timeout=5.0)
    b = build_executor(backend="ephemeral_pool", pool_size=2, timeout=5.0)
    check("ephemeral_pool is not shared", a is not b)
    a.close()
    b.close()

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("The shared proxy pool behaves as expected.")


if __name__ == "__main__":
    main()
