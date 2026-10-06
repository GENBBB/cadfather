#!/usr/bin/env python3
"""Executor deaths reach the statistics instead of dissolving into failures.

Why. The process that executes DSL code dies not only from bad code: OCC crashes
it with a signal, the OOM killer kills it under load, a pool shim can die whole
together with the task it took. All of this used to arrive upstairs as the same
`success=False` with a string in `error`, land in `ir_execution` alongside bad
geometry and was counted nowhere. So a cluster failure was indistinguishable from
a model failure, in exactly the number the run was made for.

What is checked:

1. the fork outcome distinguishes a code failure from a process death, and the
   cause of death is named by the signal, not by the word "died";
2. `proxy_pool` survives a shim death: tasks are not lost forever, the shim comes
   up again, losses are counted;
3. deaths reach the part's `tech.json`;
4. and the run summary, the per-figure table and the report, through the real
   harness whose executor kills itself with a signal;
5. a long part stops ITSELF at the soft wall cap (`experiment.figure_wall_sec`)
   without losing what it found, and if the scaffold did not catch the deadline,
   the harness catches it;
6. the pool refuses to fork workers from a process where a VTK window is already
   up: otherwise every worker dies of heap corruption, silently.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "tests"))

from cad_agent.capabilities import execute  # noqa: E402
from cad_agent.capabilities.execute import (  # noqa: E402
    OUTCOME_DIED,
    OUTCOME_ERROR,
    OUTCOME_LOST,
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    EvalTask,
    ProxyPoolExecutor,
    run_in_fork,
)

FAILURES: list[str] = []


def wall_scenario(folder: str, run_dir: str) -> None:
    """Scenario of section 6 as a whole, run in a SEPARATE process.

    Why separate, and `spawn` at that. Sections 4-5 run with `n_workers: 1`, which
    is the pool's debug path: the part is computed in this very process, and the
    render brings up a VTK window here. Section 6 uses the pool, i.e. a fork, and a
    child forked from a process with a window up dies on its `pv.Plotter` of heap
    corruption (`malloc(): unaligned tcache chunk detected`), without a single
    message of ours. Measured on the server: a parent without a plotter gives a
    live child, a parent with a plotter gives SIGABRT. Locally this does not
    reproduce (no EGL/GL), so the check would be green here and fail there.

    A real run does not do this, since the parent does not render, so what is fixed
    is the test, not the pipeline: the scenario is moved to a clean process. The
    pipeline itself now also refuses to fork the pool from a process with a window
    up (`pool._check_no_plotter`), but the test cannot rely on that: the point of
    the section is the wall cap, not this error.

    Results go up as run files (`per_figure.json`, `summary.json`), not as a return
    value: there is nothing to return from another process.
    """
    import harness_e2e

    from cad_agent.capabilities.propose import StepProposer
    from cad_agent.harness import run_eval

    harness_e2e.install_fakes()

    real_propose = StepProposer.propose

    def slow_propose(self, *, gt_mesh_path, **kwargs):
        # ONE part is made slow: the neighbors must run to completion.
        if "figure_01" in str(gt_mesh_path):
            time.sleep(2.0)
        return real_propose(self, gt_mesh_path=gt_mesh_path, **kwargs)

    StepProposer.propose = slow_propose
    config = harness_e2e.base_config(Path(folder), backend="serial_fork", n_workers=2)
    # The cap is twice the duration of one slow step: it is HEADROOM for the
    # neighboring parts, not an indulgence for the slow one. The assertions below
    # are unchanged: the slow part must hit the cap, the neighbors must finish. With
    # three seconds the section measured the machine speed: neighbors run in a
    # fraction of a second solo and hit the cap when the preceding set loaded the
    # machine. With a 6 s cap and 2 s per step the slow part still spends three
    # steps of eight.
    config["figure_wall_sec"] = 6
    config["limits"] = {"iterations": 8, "depth": 8}
    run_eval.run_experiment(config=config, run_dir=Path(run_dir))


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --- what is executed in the forked child ------------------------------------
def _fine() -> dict:
    return {"mesh_path": "/tmp/ok.stl", "metrics": {"cd_runtime": 0.5}}


def _raises() -> dict:
    raise ValueError("code failed to build")


def _segfault() -> dict:
    os.kill(os.getpid(), signal.SIGSEGV)
    return {}


def _oom() -> dict:
    os.kill(os.getpid(), signal.SIGKILL)
    return {}


def _hangs() -> dict:
    time.sleep(30)
    return {}


def _slow(task=None) -> dict:
    """A task that lives long enough to be orphaned."""
    time.sleep(4)
    return _fine()


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="worker_death_"))

    print("1. The fork outcome distinguishes a code failure from a process death")
    ok, _payload, outcome = run_in_fork(_fine, (), 10.0)
    check("success is labelled ok", ok and outcome == OUTCOME_OK, outcome)

    ok, payload, outcome = run_in_fork(_raises, (), 10.0)
    check("an exception in the code is an error, not a death", not ok and outcome == OUTCOME_ERROR, outcome)
    check("the code traceback arrived", "ValueError" in str(payload))

    ok, payload, outcome = run_in_fork(_segfault, (), 10.0)
    check("a signal is an executor death", not ok and outcome == OUTCOME_DIED, outcome)
    check("the cause is named by the signal", "SIGSEGV" in str(payload), str(payload)[:90])

    ok, payload, outcome = run_in_fork(_oom, (), 10.0)
    check("SIGKILL recognized", outcome == OUTCOME_DIED and "SIGKILL" in str(payload), str(payload)[:90])
    check("OOM is stated directly", "OOM" in str(payload), str(payload)[:90])

    ok, payload, outcome = run_in_fork(_hangs, (), 1.0)
    check("a hung one is a timeout", not ok and outcome == OUTCOME_TIMEOUT, outcome)

    print("\n2. proxy_pool survives the death of the shim")
    # Patience is counted from the task timeout; without shortening the margin the
    # test would wait tens of seconds where one behavior is being checked.
    # The real poll period is needed by section 4: it checks exactly that, and a
    # shortened override would mask the defect the section was written for.
    real_poll_sec = execute._PROXY_POLL_SEC
    execute._PROXY_GRACE_SEC = 1.0
    execute._PROXY_POLL_SEC = 0.3
    execute._evaluate = lambda task: _fine()
    execute._preload_cad = lambda: None

    pool = ProxyPoolExecutor(pool_size=1, timeout=1.0)
    try:
        results = pool.evaluate([EvalTask(task_id="alive", code="r = r.box(1)")])
        check("a live shim returns a result", results[0].success, results[0].outcome)

        # The death of an idle shim must cost nothing: tasks are handed only to live
        # ones, so there is nothing to lose here.
        pool._procs[0].kill()
        pool._procs[0].join()
        results = pool.evaluate([
            EvalTask(task_id="after_death_1", code="r = r.box(1)"),
            EvalTask(task_id="after_death_2", code="r = r.box(2)"),
        ])
        check("evaluate does not hang forever", True)
        check("the death of an idle proxy loses nothing",
              all(r.success for r in results), str([r.outcome for r in results]))
        check("restart is counted", pool.stats().get("proxy_restarts", 0) > 0, str(pool.stats()))
        check("no losses meanwhile", pool.stats().get("lost_tasks", 0) == 0, str(pool.stats()))

        # After the restart the pool is usable again: otherwise one death would mean
        # the end of the part, which is no better than a hang.
        results = pool.evaluate([EvalTask(task_id="after", code="r = r.box(3)")])
        check("the pool works after a restart", results[0].success, results[0].outcome)
    finally:
        pool.close()

    print("\n3. A late warm-up acknowledgement does not eat a task")
    # `warm_gt` waits for confirmation for a limited time and on a cold start (the
    # shim is still bringing up OCP) does not get it, but the reply arrives later
    # and stays in the queue. It was then read as a task result: number `-1` went
    # into `collected`, inflating the ready counter, while the real task was dropped
    # from the books and declared lost. That cost one task per part.
    pool = ProxyPoolExecutor(pool_size=1, timeout=1.0)
    try:
        pool._result_queues[0].put((execute._WARM_ACK_INDEX, True, {"mesh_path": None}, False, 0.0))
        # The queue is inter-process: without a pause the confirmation may not reach
        # `get_nowait`, and the check would be green having checked nothing.
        time.sleep(0.5)
        results = pool.evaluate([EvalTask(task_id="after_warmup", code="r = r.box(1)")])
        check("the task survived the late acknowledgement", results[0].success, results[0].outcome)
        check("no loss recorded", pool.stats().get("lost_tasks", 0) == 0, str(pool.stats()))
    finally:
        pool.close()

    print("\n4. A candidate does not pay the queue polling tick")
    # The dispatch loop waited for a result by sleeping for `_PROXY_POLL_SEC`, and the
    # same constant set the period of shim liveness checks. While it was 2 s, every
    # `evaluate` went to sleep for a full tick with execution taking a fraction of a
    # second: that added seconds per candidate over the time inside the fork and
    # explained the `execute` stage almost entirely. Checked with the real poll
    # period, not the shortened one above.
    execute._PROXY_POLL_SEC = real_poll_sec
    pool = ProxyPoolExecutor(pool_size=2, timeout=5.0)
    try:
        tasks = [EvalTask(task_id=f"fast_{i}", code="r = r.box(1)") for i in range(8)]
        started = time.monotonic()
        results = pool.evaluate(tasks)
        elapsed = time.monotonic() - started
        check("all candidates returned", all(r.success for r in results),
              str([r.outcome for r in results]))
        # There is no work here at all (`_evaluate` is a stub), so the cost is pure
        # pool latency. The threshold has a large margin: what is caught is not
        # slowness but a tick of a second or more per candidate.
        check("eight candidates do not cost a tick each", elapsed < 1.0, f"{elapsed:.2f} s")
    finally:
        pool.close()
        execute._PROXY_POLL_SEC = 0.3

    # A shim death **with a task in hand** is a real loss, and it is the one that
    # must be counted: the result will never come back.
    #
    # A separate pool is needed for this: a shim is forked once, and substituting
    # `_evaluate` in the parent does not change a shim that already lives. The first
    # version of the check was fooled exactly on this: the task managed to finish
    # before it was about to be orphaned.
    execute._evaluate = _slow
    slow_pool = ProxyPoolExecutor(pool_size=1, timeout=10.0)
    try:
        thread = threading.Thread(
            target=lambda: (time.sleep(1.0), slow_pool._procs[0].kill()), daemon=True
        )
        thread.start()
        results = slow_pool.evaluate([EvalTask(task_id="in_flight", code="r = r.box(9)")])
        thread.join()

        check("the task of a dead shim is labelled lost",
              results[0].outcome == OUTCOME_LOST, results[0].outcome)
        check("a loss is an executor death", results[0].worker_died)
        check("losses are counted", slow_pool.stats().get("lost_tasks", 0) > 0, str(slow_pool.stats()))
        check("restart after a loss is counted",
              slow_pool.stats().get("proxy_restarts", 0) > 0, str(slow_pool.stats()))
    finally:
        slow_pool.close()
    execute._evaluate = lambda task: _fine()

    print("\n3. Deaths reach the part tech.json")
    import harness_e2e  # noqa: E402  — the same stubs as in the end-to-end test
    from cad_agent.capabilities.execute import EvalResult
    from cad_agent.capabilities.resources import build_resources
    from cad_agent.harness.budget import Budget
    from cad_agent.harness.journal import FigureJournal

    class DyingExecutor(execute.Executor):
        def evaluate(self, tasks):
            return [
                EvalResult(task_id=task.task_id, success=False, error="killed",
                           outcome=OUTCOME_DIED, wall_sec=0.1)
                for task in tasks
            ]

        def stats(self):
            return {"proxy_restarts": 2}

    journal = FigureJournal(figure_dir=tmp / "figure", level="metrics", profile=False)
    resources = build_resources(
        figure_id="t/f", gt_mesh_path=Path("/tmp/gt.stl"), work_dir=tmp / "figure",
        executor=DyingExecutor(), renderer=None, proposer=None,
        budget=Budget(), config={}, journal=journal,
    )
    resources.evaluate([EvalTask(task_id="a", code="x"), EvalTask(task_id="b", code="y")])
    tech = journal.tech_metrics()
    check("deaths are counted in tech.json", tech["n_worker_deaths"] == 2, str(tech["worker_deaths"]))
    check("the cause of death is named", tech["worker_deaths"].get(OUTCOME_DIED) == 2)
    check("executor events are saved", tech["executor_stats"].get("proxy_restarts") == 2)

    print("\n4. Deaths reach the summary, the table and the report")
    from cad_agent.harness import run_eval
    from cad_agent.harness.report import build_report

    harness_e2e.install_fakes()
    folder = harness_e2e.make_dataset(tmp, n=3)

    real_evaluate = execute._evaluate

    def killing_evaluate(task):
        """Candidates of ONE part kill their own process with a signal.

        This is a real death in a forked child, not a stub result: the whole road
        from `waitpid` to a line in the report is checked.

        The victim selection rule is **deterministic and based on the task itself**:
        a counter in the closure does not work here, because `_evaluate` already runs
        in the child and its increment dies with it.

        The victim is chosen by the PART, not by the text of `task_id`. The earlier
        rule (`sum(ord(task_id)) % 3`) depended on the candidate labeling, and when
        that changed, all tasks fell under it at once: the check "not all died in a
        row" turned red although the death mechanism was intact. The first version
        of the check was fooled exactly this way too: "the first four" meant "all".
        Binding to the part keeps both classes non-empty under any labeling.
        """
        if "figure_00" in str(task.gt_mesh_path):
            os.kill(os.getpid(), signal.SIGKILL)
        return real_evaluate(task)

    execute._evaluate = killing_evaluate
    try:
        config = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1)
        result = run_eval.run_experiment(config=config, run_dir=tmp / "run")
    finally:
        execute._evaluate = real_evaluate

    tech = result["summary"]["tech"]
    check("deaths got into the run summary", tech.get("n_worker_deaths", 0) > 0,
          f"{tech.get('n_worker_deaths')} — {tech.get('worker_deaths')}")
    check("the parts with deaths are named", tech.get("figures_with_worker_deaths", 0) > 0)

    csv_text = (Path(result["run_dir"]) / "per_figure.csv").read_text(encoding="utf-8")
    check("the table has a deaths column", "n_worker_deaths" in csv_text.splitlines()[0])

    report = build_report(result["run_dir"])
    check("the report has a deaths section", "Executor deaths" in report)
    check("the report names the cause", "killed (signal, OOM, OCC)" in report, )
    check("the report warns about IR", "IR" in report.split("Executor deaths")[1][:900])

    check("execution fork threads are measured",
          bool(tech.get("fork_threads", {}).get("exec", {}).get("n")),
          str(tech.get("fork_threads")))
    check("not all candidates died in a row",
          0 < tech.get("n_worker_deaths", 0) < int(result["summary"]["cost_calls_total"].get("exec", 0)),
          f"deaths {tech.get('n_worker_deaths')} of {result['summary']['cost_calls_total'].get('exec')} executions")

    # The section is printed even when all is well: "zero" is a measured fact,
    # while a missing section reads as "not counted".
    clean = run_eval.run_experiment(
        config=harness_e2e.base_config(folder, backend="serial_fork", n_workers=1),
        run_dir=tmp / "run_clean",
    )
    clean_report = build_report(clean["run_dir"])
    check("the section is present without deaths too", "the executor never died" in clean_report)

    print("\n5. A part lost with its worker stays in the denominator")
    # Above, the executor died: an inner fork, after which the part still arrives and
    # brings a measurement. Here is a different event: the part's own process dies,
    # and the record arrives with no metrics at all (`pool._failed_record`). While
    # such a record simply did not enter the aggregate, a hard part silently dropped
    # out of the denominator and **improved** both scores.
    fabricated = [
        {"figure_id": "test/ok", "score": 0.8, "wall_sec": 1.0, "cost": {"calls": {"vlm": 1}},
         "metrics": {"figure_id": "test/ok", "gt_watertight": True, "iou": 0.8, "gms_norm": 0.8}},
        {"figure_id": "test/lost", "score": 0.0, "wall_sec": 0.0, "cost": {}, "metrics": None,
         "worker_died": True, "error": "worker died, part did not finish"},
    ]
    lost_summary = run_eval.summarize(fabricated, signature="test", compute_metrics=True)
    quality = lost_summary["quality"]
    check("the lost part stays in the denominator", quality["n_figures"] == 2, str(quality["n_figures"]))
    check("it is counted as a failure", quality["n_failures"] == 1, str(quality["n_failures"]))
    check("a separate cause, not execution", quality["ir_no_result"] == 0.5
          and quality["ir_execution"] == 0.0, str(quality))
    check("the IR decomposition adds up",
          abs(quality["ir"] - (quality["ir_execution"] + quality["ir_not_watertight"]
                               + quality["ir_no_result"])) < 1e-9, str(quality["ir"]))
    check("zero entered the mean", abs(quality["score_with_zeros"] - 0.4) < 1e-9,
          str(quality["score_with_zeros"]))
    check("the contract identity holds", quality["identity_holds"] is True)
    # And where metrics were not computed at all (`compute_metrics: false` in smoke
    # runs), records without metrics are normal, not failures: otherwise a switched-off
    # measurement would read as "everything failed".
    off = run_eval.summarize(
        [{"figure_id": "test/a", "score": 0.0, "wall_sec": 0.0, "cost": {}, "metrics": None}],
        signature="test",
        compute_metrics=False,
    )
    check("with metrics off there is no quality", not off["quality"], str(off["quality"])[:60])
    # Without a hint from outside, the composition of the records decides, not a silent default.
    guessed = run_eval.summarize(
        [{"figure_id": "test/a", "score": 0.0, "wall_sec": 0.0, "cost": {}, "metrics": None}],
        signature="test",
    )
    check("without a hint an empty run is not a failure either", not guessed["quality"],
          str(guessed["quality"])[:60])

    print("\n6. A long part stops ITSELF and returns what it found")
    # A real scenario: a part is computed for an unbounded time. The watchdog that
    # killed workers is gone, and what must be checked here is not "the run reached
    # the summary" (that is too little) but that the part stopped by itself and did
    # **not lose** the best prefix found. The difference between these two outcomes
    # is the point of the soft cap.
    from cad_agent.harness.budget import WALL_STOP_REASON, Budget, DeadlineExceeded

    # Why the scenario is moved to a separate process is checked, not just told in a
    # comment. Sections 4-5 ran with `n_workers: 1`, i.e. computed parts right here,
    # and the render brought up a VTK window in THIS process. After that the pool
    # cannot be forked, and the pipeline now knows this itself instead of silently
    # handing out heap corruption in every worker.
    from cad_agent.capabilities import render as render_mod
    from cad_agent.harness import pool as pool_mod

    check("the debug path opened a VTK window in this process", render_mod.plotter_open_here())
    try:
        pool_mod._check_no_plotter()
        check("the pool refuses to fork from a process with a window", False, "did not refuse")
    except RuntimeError as exc:
        check("the pool refuses to fork from a process with a window", True)
        check("the problem is explained", "VTK window" in str(exc), str(exc)[:60])

    # The scenario runs in its own process; why, is written in `wall_scenario`.
    # `spawn`, not `fork`: a fork would inherit exactly the state that made the
    # isolation necessary.
    wall_dir = tmp / "run_wall"
    scenario = mp.get_context("spawn").Process(
        target=wall_scenario, args=(str(folder), str(wall_dir)),
    )
    started = time.monotonic()
    scenario.start()
    # A cap with margin: the scenario is three parts on a stub generator, one of
    # which sleeps two seconds per step. It exists so that the check for "the run
    # does not hang" does not itself hang.
    scenario.join(600)
    elapsed = time.monotonic() - started
    if scenario.is_alive():
        scenario.kill()
        scenario.join()
    check("the cap scenario ran", scenario.exitcode == 0, f"exit code {scenario.exitcode}")
    if scenario.exitcode != 0 or not (wall_dir / "per_figure.json").is_file():
        print(f"  (no run artifacts, nothing to check: {wall_dir})")
        sys.exit(1)
    walled = {
        "run_dir": str(wall_dir),
        "per_figure": json.loads((wall_dir / "per_figure.json").read_text(encoding="utf-8")),
        "summary": json.loads((wall_dir / "summary.json").read_text(encoding="utf-8")),
    }

    slow = [r for r in walled["per_figure"] if str(r["figure_id"]).endswith("figure_01")][0]
    others = [r for r in walled["per_figure"] if not str(r["figure_id"]).endswith("figure_01")]
    check("the run finished and did not hang", elapsed < 300, f"{elapsed:.0f} s")
    check("summary written", (Path(walled["run_dir"]) / "summary.json").is_file())
    check("all parts are in the table", len(walled["per_figure"]) == 3, str(len(walled["per_figure"])))
    # The part hit the PRICE, and this must be visible in the reason. A planning
    # policy (`affordable_n`) stops one step before the harness cap, i.e.
    # `DeadlineExceeded` does not occur at all here, but the outcome is still not
    # `done`: `done` is a judgment of quality ("good enough"), while here the policy
    # simply has no affordable actions left. They must not be merged, otherwise cost
    # silently reads as quality. The path where the cap reaches a harness exception
    # is covered by sections 7 and 8 below.
    # `dialogue_lean` has no `done:exhausted`: with nothing affordable left it returns an
    # empty plan, so for the long part `plan_empty` is the exhaustion too. The neighbours
    # may end with an empty plan for other reasons, so for them only the cap is excluded.
    check("the long part is stopped by exhaustion, not by a policy decision",
          slow["stop_reason"] in (WALL_STOP_REASON, "done:exhausted", "plan_empty"),
          repr(slow["stop_reason"]))
    check("neighbouring parts stopped otherwise: their budget did not run out",
          all(r["stop_reason"] not in (WALL_STOP_REASON, "done:exhausted") for r in others),
          str([(r["figure_id"], r["stop_reason"]) for r in others]))
    # The main difference from the old watchdog: this is NOT a failure.
    check("the part is not counted as a worker loss", not slow.get("worker_died"),
          str(slow.get("worker_died")))
    # What exactly "found" means: the steps taken, the best prefix and its mesh. The
    # score is deliberately not checked here: locally there is no boolean engine, and
    # `metrics` is empty for ALL parts of the run, not only the stopped one. Checking
    # a property of the environment instead of a property of the code is an
    # evergreen check in reverse: it would always fail and say nothing.
    check("what was found is kept: steps", slow["n_steps"] > 0, str(slow["n_steps"]))
    check("and the mesh is on disk", bool(slow["mesh_path"]) and Path(slow["mesh_path"]).exists(),
          str(slow["mesh_path"]))
    slow_dir = Path(walled["run_dir"]) / "figures" / slow["figure_id"]
    check("and the best prefix is written",
          (slow_dir / "best.py").is_file() and (slow_dir / "best.py").stat().st_size > 0)
    # The cap is soft: it prevents STARTING the next step, and a started one is
    # carried to the end. So the part must overshoot the cap, but by no more than one
    # step, and must stop before its own `max_steps`.
    check("the cap fired before max_steps", slow["n_steps"] < 8, str(slow["n_steps"]))
    check("overshoot is at most one step",
          slow["wall_sec"] < 6 + 2 * 2.0, f"{slow['wall_sec']:.1f} s with a 6 s cap")
    check("neighbouring parts finished", all(not r.get("worker_died") for r in others),
          str([(r["figure_id"], r.get("worker_died")) for r in others]))
    check("neighbouring parts were not stopped by the cap",
          all(r["stop_reason"] != WALL_STOP_REASON for r in others),
          str([(r["figure_id"], r["stop_reason"], f"{r.get('wall_sec') or 0:.1f} s")
               for r in others]))
    # Nobody was killed, so there must be no death counters either.
    check("nobody killed the workers", not walled["summary"]["tech"]["pool_deaths"],
          str(walled["summary"]["tech"]["pool_deaths"]))

    print("\n7. The wall cap and the call cap are different outcomes")
    # One field for two events gives a plausible report with a wrong diagnosis, so
    # what is checked is DISTINGUISHABILITY, not that it fires.
    spent = Budget(wall_sec=1.0, started=time.monotonic() - 5.0)
    try:
        spent.check_wall("propose_steps")
        check("an exhausted wall cap raises", False, "did not raise")
    except DeadlineExceeded as exc:
        check("an exhausted wall cap raises", True)
        check("the text shows what it stopped on", "propose_steps" in str(exc), str(exc))
    fresh = Budget(wall_sec=1000.0)
    fresh.check_wall("propose_steps")
    check("an unexpired cap stays silent", True)
    # Without a cap the mechanism must switch off entirely, not count from zero.
    Budget(wall_sec=None, started=time.monotonic() - 10_000).check_wall("propose_steps")
    check("null means no cap", True)

    print("\n8. A harness that misses the deadline loses the rollout, not the run")
    # The policy evolves, and one cannot rely on it catching the deadline. The harness
    # must do its work itself: the part becomes a failure, the run lives.
    class DeafScaffold:
        """Calls capabilities and catches nothing, like a mutant that lost its catch."""

        def run(self, obs, res):
            for step in range(1, 50):
                res.propose_steps(None, obs.prefix_code, 1, step=step)
            raise AssertionError("unreachable: the cap had to fire")

    deaf_config = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1)
    # The cap is already overdue at the very first call: on a stub generator fifty
    # steps pass in a fraction of a second, and a whole number of seconds would make
    # the check a race. Zero here would mean "no cap", hence not zero.
    deaf_config["figure_wall_sec"] = 0.001
    deaf = run_eval.run_experiment(
        config=deaf_config, run_dir=tmp / "run_deaf", scaffold=DeafScaffold(),
    )
    stopped = [r for r in deaf["per_figure"] if r["stop_reason"] and WALL_STOP_REASON in r["stop_reason"]]
    check("the run reached the summary", (Path(deaf["run_dir"]) / "summary.json").is_file())
    check("parts were stopped by the cap", len(stopped) == len(deaf["per_figure"]),
          f"{len(stopped)} of {len(deaf['per_figure'])}")
    check("and it is distinguishable from a regular stop",
          all("the scaffold did not stop by itself" in r["stop_reason"] for r in stopped),
          str([r["stop_reason"] for r in stopped][:1]))

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Executor deaths are accounted for.")


if __name__ == "__main__":
    main()
