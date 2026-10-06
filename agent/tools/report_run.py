#!/usr/bin/env python3
"""Manual analysis of a run directory: quality, failures, cost, timing, log.

A run now writes the same report for itself to `logs/report.txt`
(`harness/report.py`). This CLI is for a directory that is foreign, old, or that
you want to look at from another angle:

    python agent/tools/report_run.py work_dirs/<run>
    python agent/tools/report_run.py work_dirs/<run> --figure mcb/00001234
    python agent/tools/report_run.py work_dirs/<run> --compare work_dirs/<baseline>

`--compare` checks comparability separately: whether the dataset, logging level
and profiling match. Logging overhead that differs between two runs would go
straight into the measured difference. It is not part of the automatic report,
since during a run there is nothing to compare with yet.

The tool only reads. It recomputes nothing and writes nothing to the run
directory: numbers are taken from what the run itself recorded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runs as a script from the repository root, so the package path must be added explicitly.
AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from cad_agent.harness.report import (  # noqa: E402
    Run,
    build_report,
    report_compare,
    report_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", help="Run directory (the one printed by run_experiment.py).")
    parser.add_argument("--figure", default=None, help="Break down one part: steps, events, metrics.")
    parser.add_argument("--compare", default=None, help="Compare with another run (usually the baseline).")
    parser.add_argument("--top", type=int, default=10, help="How many rows to show in the top lists.")
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Do not read events.jsonl. On large runs these are thousands of files on NFS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.figure:
        report_figure(Run(args.run_dir, read_events=not args.no_events), args.figure)
        return 0

    print(build_report(args.run_dir, top=args.top, read_events=not args.no_events), end="")

    if args.compare:
        report_compare(
            Run(args.run_dir, read_events=False), Run(args.compare, read_events=False)
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
