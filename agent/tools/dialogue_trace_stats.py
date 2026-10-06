#!/usr/bin/env python3
"""Dialogue-mode turn losses for a run directory, and a comparison of two runs.

Counts what `summary.json` does not show: how turns ended, how many fell back to
the harness, how many named actions were dropped and why, and how many calls
returned only repeats.

Sources are `per_figure.json`, `events.jsonl` and the LAST
`agent/call_*_prompt.txt` of a part, which holds the whole conversation
transcript. Standard library only, so it can run directly on the node, in the run
directory.

    ./agent/tools/dialogue_trace_stats.py stats work_dirs/<run> [--json out.json]
    ./agent/tools/dialogue_trace_stats.py compare work_dirs/<old> work_dirs/<new>

Transcript counting relies on line beginnings that the policy keeps unchanged for
this purpose ("not on the list, ignored", "result: no new candidates",
"— best so far"); outcomes rely on the `Action.reason` prefixes in the journal. If
they drift apart the collector shows zeros rather than an error, so check it on a
directory with a known answer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter

# `Action.reason` prefixes (`DialogueLeanPolicy.OUTCOMES`). A prefix rather than the
# whole string: the tail of the "taken ..." reason has changed, but older runs
# must be counted with the same key.
REASONS = (
    ("chosen", "assistant's choice"),
    ("looked_only", "assistant asked for images"),
    ("rejected", "no named action is on the list"),
    ("reasked", "named action not on the list"),
    ("reasked_unparsed", "answer without an action, decided"),
    ("retried", "answer cut off, decided"),
    ("truncated", "answer cut off before the named"),
    ("unparsed", "answer not recognized"),
    ("silent", "assistant did not answer"),
    ("unaffordable", "assistant's choice is not affordable"),
    ("done_too_early", "assistant finished the part before"),
)

_TURN_ANSWER = re.compile(r"^Turn (\d+), your answer:\n", re.M)
_DROPPED = re.compile(r"^Turn (\d+): not on the list, ignored — (.*)$", re.M)
_PAIR = re.compile(r"tool=(\S+) candidate=([A-Za-z0-9_]+)")
_EXECUTED = re.compile(r"^Turn (\d+) executed:(.*)\n((?:  tool=.*\n?)+)", re.M)


def _reason_kind(reason: str | None) -> str:
    # Order matters: the "assistant choice not affordable" reason starts with the "assistant
    # choice" reason, so the longer prefix is checked before the shorter one.
    text = reason or ""
    for kind, prefix in sorted(REASONS, key=lambda item: -len(item[1])):
        if text.startswith(prefix):
            return kind
    return "other"


def _last_prompt(figure_dir: str) -> str | None:
    agent = os.path.join(figure_dir, "agent")
    if not os.path.isdir(agent):
        return None
    prompts = sorted(name for name in os.listdir(agent) if name.endswith("_prompt.txt"))
    if not prompts:
        return None
    with open(os.path.join(agent, prompts[-1]), encoding="utf-8") as handle:
        return handle.read()


def _answer_actions(transcript: str, answers: dict[int, list[re.Match]], turn: int) -> set:
    """(tool, candidate) pairs of the turn's LAST reply, which is the one the turn went out with."""
    matches = answers.get(turn)
    if not matches:
        return set()
    start = matches[-1].end()
    end = transcript.find("\nTurn ", start)
    body = transcript[start:end if end >= 0 else len(transcript)]
    return set(re.findall(r"^ACTION:\s*tool=(\S+) candidate=([A-Za-z0-9_]+)", body, re.M))


def _transcript_counts(transcript: str, counts: Counter) -> dict:
    body = transcript.split("\nCandidates you can act on")[0]
    answers: dict[int, list[re.Match]] = {}
    for match in _TURN_ANSWER.finditer(body):
        answers.setdefault(int(match.group(1)), []).append(match)

    counts["results"] += len(re.findall(r"^Turn \d+ result:", body, re.M))
    counts["no_new_bare"] += len(re.findall(r"^Turn \d+ result: no new candidates\.", body, re.M))
    counts["no_new_with_reason"] += len(
        re.findall(r"^Turn \d+ result: no new candidates \(", body, re.M))
    counts["no_new_named_duplicates"] += len(re.findall(r"already on the list as", body))
    counts["reask_lines"] += len(re.findall(r"^Turn \d+: asked again — (?:name actions|call act)", body, re.M))

    dropped_turns = 0
    for match in _DROPPED.finditer(body):
        dropped_turns += 1
        turn = int(match.group(1))
        previous = set()
        for earlier in range(turn - 1, 0, -1):
            if earlier in answers:
                previous = _answer_actions(body, answers, earlier)
                break
        for tool, candidate in _PAIR.findall(match.group(2)):
            counts["dropped_pairs"] += 1
            if (tool, candidate) in previous:
                counts["dropped_same_as_previous_answer"] += 1
    counts["dropped_turns"] += dropped_turns

    # Hidden fallback in old runs: everything named was dropped, `executed`
    # carries no mark, and something other than what was asked got executed.
    # New runs mark such a turn, so it does not land here; that is the fix check.
    for match in _EXECUTED.finditer(body):
        turn = int(match.group(1))
        if match.group(2).strip():
            continue
        if not re.search(rf"^Turn {turn}: not on the list, ignored", body, re.M):
            continue
        asked = _answer_actions(body, answers, turn)
        executed = set(_PAIR.findall(match.group(3)))
        if executed and not any((tool, cand[:8]) in {(t, c[:8]) for t, c in asked}
                                for tool, cand in executed):
            counts["masked_fallback_turns"] += 1

    improved = [int(turn) for turn in re.findall(
        r"^Turn (\d+) result:(?:(?!\nTurn ).)*?— best so far", body, re.M | re.S)]
    return {"last_improvement_turn": max(improved) if improved else 0,
            "dropped_turns": dropped_turns}


def collect(root: str) -> dict:
    with open(os.path.join(root, "per_figure.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    counts: Counter = Counter()
    reasons: Counter = Counter()
    rejects: Counter = Counter()
    calls: Counter = Counter()
    stops = Counter(row.get("stop_reason") for row in rows)
    # Finish door: runs from before the field have none at all, which means
    # "not recorded", not "no finishes".
    doors: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("stop_reason") or "").startswith("done"):
            door = row["done_by"] if "done_by" in row else "(no field)"
            doors.setdefault(str(door), []).append(row.get("score") or 0.0)
    figures: dict[str, dict] = {}
    looked_figs = looked3_figs = dropped_figs = 0
    tails = []

    for row in rows:
        fid = row["figure_id"]
        figure_dir = os.path.join(root, "figures", fid)
        score = row.get("score") or 0.0
        metrics = row.get("runtime_metrics") or {}
        figures[fid] = {"score": score, "stop": row.get("stop_reason"),
                        "group": row.get("group"), "iterations": row.get("iterations")}
        for kind, value in ((row.get("tech") or {}).get("calls") or {}).items():
            calls[kind] += value or 0
        counts["wall_sec"] += row.get("wall_sec") or 0.0
        counts["iterations"] += row.get("iterations") or 0
        if metrics.get("gt_watertight") and metrics.get("iou") is not None:
            counts["iou_n"] += 1
            counts["iou_sum"] += metrics["iou"]

        figure_looked = 0
        events_path = os.path.join(figure_dir, "events.jsonl")
        if os.path.exists(events_path):
            with open(events_path, encoding="utf-8") as handle:
                for line in handle:
                    event = json.loads(line)
                    kind = event.get("kind")
                    if kind == "tool_call":
                        counts["tool_calls"] += 1
                        reason = _reason_kind(event.get("reason"))
                        reasons[reason] += 1
                        figure_looked += reason == "looked_only"
                    elif kind == "plan_rejected":
                        rejects[(event.get("reason") or "")[:45]] += 1
                    elif kind == "candidate_dedupe":
                        counts["dedupe_events"] += 1
        looked_figs += figure_looked > 0
        looked3_figs += figure_looked >= 3

        transcript = _last_prompt(figure_dir)
        if transcript is not None:
            info = _transcript_counts(transcript, counts)
            dropped_figs += info["dropped_turns"] > 0
            if row.get("stop_reason") == "limit:iterations" and row.get("iterations"):
                tails.append(row["iterations"] - info["last_improvement_turn"])

    scores = [figure["score"] for figure in figures.values()]
    n = len(rows)
    return {
        "run": os.path.basename(os.path.normpath(root)),
        "figures": n,
        "score_mean": st.mean(scores) if scores else None,
        "score_median": st.median(scores) if scores else None,
        "iou_mean_watertight": counts["iou_sum"] / counts["iou_n"] if counts["iou_n"] else None,
        "iterations_mean": counts["iterations"] / n if n else None,
        "wall_sec_mean": counts["wall_sec"] / n if n else None,
        "stop_reason": dict(stops.most_common()),
        "done_by": {door: len(values) for door, values in sorted(doors.items())},
        "done_by_score_mean": {door: round(st.mean(values), 4)
                               for door, values in sorted(doors.items())},
        "calls": dict(calls),
        "tool_calls": counts["tool_calls"],
        "tool_call_reason": dict(reasons.most_common()),
        "plan_rejected": dict(rejects.most_common()),
        "figures_with_looked_only": looked_figs,
        "figures_with_3plus_looked_only": looked3_figs,
        "figures_with_dropped": dropped_figs,
        "dropped_turns": counts["dropped_turns"],
        "dropped_pairs": counts["dropped_pairs"],
        "dropped_same_as_previous_answer": counts["dropped_same_as_previous_answer"],
        "masked_fallback_turns": counts["masked_fallback_turns"],
        "reask_lines": counts["reask_lines"],
        "turn_results": counts["results"],
        "no_new_bare": counts["no_new_bare"],
        "no_new_with_reason": counts["no_new_with_reason"],
        "no_new_named_duplicates": counts["no_new_named_duplicates"],
        "dedupe_events": counts["dedupe_events"],
        "limit_tail_mean": st.mean(tails) if tails else None,
        "limit_tail_median": st.median(tails) if tails else None,
        "_figures": figures,
    }


def _bootstrap(deltas: list[float], rounds: int, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(
        st.mean(rng.choices(deltas, k=len(deltas))) for _ in range(rounds)
    )
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def compare(old: dict, new: dict, rounds: int) -> dict:
    """Paired comparison over common parts: one part, one difference.

    Pairing is not decoration: the score spread between parts is an order of
    magnitude larger than the expected effect, and unpaired means would drown it.
    The interval is a bootstrap over parts; there is NO baseline noise floor for
    these splits (a rerun of the old code was not taken), so the interval only
    reflects the spread over parts, not run-to-run spread.
    """
    common = sorted(set(old["_figures"]) & set(new["_figures"]))
    deltas = [new["_figures"][fid]["score"] - old["_figures"][fid]["score"] for fid in common]
    transitions = Counter(
        (old["_figures"][fid]["stop"], new["_figures"][fid]["stop"]) for fid in common)
    low, high = _bootstrap(deltas, rounds) if deltas else (None, None)
    return {
        "common_figures": len(common),
        "delta_mean": st.mean(deltas) if deltas else None,
        "delta_median": st.median(deltas) if deltas else None,
        "delta_ci95": [low, high],
        "better": sum(1 for d in deltas if d > 1e-4),
        "worse": sum(1 for d in deltas if d < -1e-4),
        "same": sum(1 for d in deltas if abs(d) <= 1e-4),
        "stop_transitions": {f"{a} -> {b}": count
                             for (a, b), count in transitions.most_common(12)},
    }


def _print_side_by_side(old: dict, new: dict) -> None:
    keys = [key for key in old if not key.startswith("_") and key != "run"]
    print(f"{'':40}{old['run']:>28}{new['run']:>28}")
    for key in keys:
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) or isinstance(b, dict):
            print(f"{key}:")
            for sub in sorted(set(a or {}) | set(b or {}), key=str):
                print(f"  {str(sub)[:38]:38}{str((a or {}).get(sub, 0)):>28}"
                      f"{str((b or {}).get(sub, 0)):>28}")
            continue
        fmt = (lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{key:40}{fmt(a):>28}{fmt(b):>28}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("stats", help="counters of one run")
    one.add_argument("run")
    one.add_argument("--json")
    two = sub.add_parser("compare", help="two runs side by side and the paired score difference")
    two.add_argument("old")
    two.add_argument("new")
    two.add_argument("--bootstrap", type=int, default=2000)
    two.add_argument("--json")
    args = parser.parse_args()

    if args.command == "stats":
        result = collect(args.run)
        printable = {key: value for key, value in result.items() if not key.startswith("_")}
        print(json.dumps(printable, ensure_ascii=False, indent=1))
        output = printable
    else:
        old, new = collect(args.old), collect(args.new)
        _print_side_by_side(old, new)
        paired = compare(old, new, args.bootstrap)
        print("\npaired over common parts:")
        print(json.dumps(paired, ensure_ascii=False, indent=1))
        output = {"old": {k: v for k, v in old.items() if not k.startswith("_")},
                  "new": {k: v for k, v in new.items() if not k.startswith("_")},
                  "paired": paired}
    if getattr(args, "json", None):
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
