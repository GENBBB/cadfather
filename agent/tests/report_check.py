#!/usr/bin/env python3
"""Check of the run-analysis tool on real run directories.

The directories are produced by the same harness and the same stubs as in
`tests/harness_e2e.py` (only the model request and CadQuery execution are
replaced), so what is parsed is not invented JSON but what a run really writes.

It checks that:

1. the run report builds and contains all sections;
2. the numbers in the report match `summary.json` (the report recomputes nothing);
3. the single-part breakdown finds steps and events;
4. comparing two runs computes the difference and **catches incomparability**:
   different logging level, a truncated dataset, different backends;
5. the report does not fail on a run without metrics or with level `off`;
6. **a run writes the report itself** to `logs/report.txt`, and the written text
   equals what the builder returns, otherwise the file and the screen diverge.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "tests"))
sys.path.insert(0, str(AGENT_ROOT / "tools"))

import harness_e2e  # noqa: E402  — reuse the stubs of the end-to-end test
from cad_agent.harness import report as report_run  # noqa: E402
from cad_agent.harness import run_eval  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def render(run: report_run.Run, **kwargs) -> str:
    """Build the full report as a string, exactly as a run does."""
    return report_run.build_report(run.run_dir, top=kwargs.get("top", 10))


def main() -> None:
    harness_e2e.install_fakes()
    tmp = Path(tempfile.mkdtemp(prefix="report_check_"))
    folder = harness_e2e.make_dataset(tmp, n=4)

    print("1. Run directories are produced by the real harness")
    config_a = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1,
                                       log_level="full", profile=True)
    result_a = run_eval.run_experiment(config=config_a, run_dir=tmp / "run_baseline")
    run_a = report_run.Run(result_a["run_dir"])
    check("baseline run read", len(run_a.per_figure) == 4, f"parts: {len(run_a.per_figure)}")
    check("events read", bool(run_a.events), f"events: {len(run_a.events)}")

    print("\n2. The report builds and has all sections")
    text = render(run_a)
    for title in ("Quality", "How rollouts ended", "Cost", "Time",
                  "Log volume", "Caches", "Scaffold decisions"):
        check(f"section '{title}'", title in text)
    check("scaffold is named", "baseline" in text)
    check("log volume forecast present", "per 1000 parts" in text)
    check("profiled stages broken down", "stages (summed" in text)
    # A hit rate without explanation is misread: 0% for `render_pred` is normal
    # (every prediction is new), while 0% for `exec_gt` means lost warm-up.
    check("caches have a hit rate", "% hits" in text)
    check("what a miss means is stated", "miss =" in text)

    print("\n3. Numbers come from summary.json, not recomputed")
    summary = json.loads((Path(result_a["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    score = summary["quality"]["score_with_zeros"]
    check("score from summary is in the report", report_run._fmt(score) in text,
          f"score_with_zeros={report_run._fmt(score)}")
    exec_calls = summary["cost_calls_total"]["exec"]
    check("exec call count matches", report_run._fmt(exec_calls) in text, f"exec={exec_calls}")

    print("\n4. A failed part is visible in the breakdown")
    broken = [r for r in run_a.per_figure if r.get("error")]
    check("part with an execution crash found", len(broken) == 1, f"such: {len(broken)}")
    check("errors are grouped", "rollouts with an error" in text)

    print("\n5. Single-part breakdown")
    figure_id = run_a.per_figure[0]["figure_id"]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report_run.report_figure(run_a, figure_id)
    figure_text = buffer.getvalue()
    check("part found", figure_id in figure_text)
    check("scaffold steps shown", "steps (scaffold journal)" in figure_text)
    check("events shown", "events:" in figure_text)
    check("metrics shown", "metrics:" in figure_text)

    print("\n6. Run comparison and comparability check")
    # The second run is deliberately different: another logging level, another
    # backend, a truncated dataset - all three triggers of the incomparability
    # warning at once. The policy is the same (there is no other here).
    # Exactly one worker matters: step 5 above ran with `n_workers: 1`, i.e.
    # computed parts in THIS process, and rendering opened a VTK window here. The
    # pool starts by fork, and a child forked from a process with an open window
    # dies on its own `pv.Plotter` with heap corruption. That happened on a
    # server: the run consisted of dead parts only, and the check did not notice
    # because it looks at the report, not at part outcomes. Parallelism does not
    # hinder comparison: incomparability here comes from log level, backend and
    # `limit`.
    config_b = harness_e2e.base_config(folder, backend="ephemeral_pool", n_workers=1,
                                       log_level="metrics", profile=False)
    config_b["limit"] = 3
    result_b = run_eval.run_experiment(config=config_b, run_dir=tmp / "run_agentic")
    run_b = report_run.Run(result_b["run_dir"], read_events=False)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report_run.report_compare(run_b, run_a)
    compare_text = buffer.getvalue()

    check("incomparability noticed", "NOT COMPARABLE" in compare_text)
    for problem in ("logging level", "truncated", "Execution backends"):
        check(f"reason '{problem.lower()}' is named", problem.lower() in compare_text.lower())
    check("call difference computed", "vlm per part" in compare_text)
    check("both harnesses in the table", "agentic" in compare_text and "baseline" in compare_text)

    print("\n7. Comparable runs raise no warnings")
    config_c = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1,
                                       log_level="full", profile=True)
    result_c = run_eval.run_experiment(config=config_c, run_dir=tmp / "run_baseline2")
    run_c = report_run.Run(result_c["run_dir"], read_events=False)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report_run.report_compare(run_c, run_a)
    same_text = buffer.getvalue()
    check("comparable runs raise no warning", "NOT COMPARABLE" not in same_text)
    check("match is stated explicitly", "profiling and backend match" in same_text)

    print("\n8. A run without metrics and without a log is parsed")
    config_d = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1, log_level="off")
    config_d["compute_metrics"] = False
    result_d = run_eval.run_experiment(config=config_d, run_dir=tmp / "run_off")
    run_d = report_run.Run(result_d["run_dir"])
    text_d = render(run_d)
    check("report builds without metrics", "metrics were not computed" in text_d)
    check("disabled profiling is named", "stage profiling is off" in text_d)

    print("\n9. Running as a CLI")
    completed = subprocess.run(
        [sys.executable, str(AGENT_ROOT / "tools" / "report_run.py"), str(result_a["run_dir"]), "--no-events"],
        capture_output=True, text=True,
    )
    check("CLI finished", completed.returncode == 0, completed.stderr.strip()[:200])
    check("CLI printed the report", "Quality" in completed.stdout)

    missing = subprocess.run(
        [sys.executable, str(AGENT_ROOT / "tools" / "report_run.py"), str(tmp / "no-such-dir")],
        capture_output=True, text=True,
    )
    check("missing directory gives a clear error", missing.returncode != 0 and "No such run directory" in missing.stderr)

    print("\n10. The run writes the report itself")
    written = Path(result_a["run_dir"]) / "logs" / "report.txt"
    check("logs/report.txt appeared after the run", written.is_file())
    if written.is_file():
        saved = written.read_text(encoding="utf-8")
        # Verbatim equality: the file and the screen are built by the same code,
        # so they have nowhere to diverge; if they do, the report is no longer a report.
        check("saved report matches the builder", saved == text)
        check("file has the sections", "Quality" in saved and "Cost" in saved)

    config_off = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1,
                                         log_level="metrics", profile=False)
    config_off.setdefault("logging", {})["report"] = False
    result_off = run_eval.run_experiment(config=config_off, run_dir=tmp / "run_no_report")
    check("logging.report: false disables the write",
          not (Path(result_off["run_dir"]) / "logs" / "report.txt").exists())

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Run analysis is fine.")


if __name__ == "__main__":
    main()
