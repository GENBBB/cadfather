"""Report on a run directory: quality, failures, cost, time, log volume.

A run writes a lot to disk (`summary.json`, `per_figure.json`, `per_figure.csv`,
and per part `events.jsonl`, `tech.json`, `journal.json`), but it is all raw
material. The questions asked after a run are always the same: what came out,
where it failed, what it cost, where the time went, and how much space the same
run would take on the whole subsample. This module answers them in one text.

The report lives in the harness rather than in `tools/` because reporting is a
harness layer: the run itself writes `logs/report.txt` (`build_report` ->
`write_report`), and `tools/report_run.py` is a thin CLI over the same code for
inspecting a foreign or old directory by hand.

Sections print to stdout and `build_report` captures their output into a string.
This is deliberate: the sections stay exactly as they were in the CLI, and the
same text reaches both the screen and the file, so the two cannot diverge.

Reads only what the run itself recorded and recomputes nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from cad_agent.capabilities import objective as objective_mod

logger = logging.getLogger(__name__)

# Report file name inside the run directory. It sits next to the run log rather
# than in the directory root, which holds machine-readable artifacts.
REPORT_PATH = Path("logs") / "report.txt"

CALL_KINDS = ("vlm", "agent_text", "agent_visual", "agent_repair", "opt", "det", "exec")


# --- reading the run directory --------------------------------------------------
class Run:
    """A run directory read in full (except per-step artifacts)."""

    def __init__(self, run_dir: str | Path, read_events: bool = True):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise SystemExit(f"No such run directory: {self.run_dir}")

        self.summary = _load_json(self.run_dir / "summary.json", {})
        self.per_figure = _load_json(self.run_dir / "per_figure.json", [])
        self.config = _load_json(self.run_dir / "config.json", {})
        self.dataset = _load_json(self.run_dir / "dataset.json", {})
        self.provenance = _load_json(self.run_dir / "provenance.json", {})
        self.events = self._load_events() if read_events else []

        if not self.summary and not self.per_figure:
            raise SystemExit(
                f"{self.run_dir} does not look like a run directory: no summary.json and no per_figure.json"
            )

    def _load_events(self) -> list[dict[str, Any]]:
        """Collect every part's `events.jsonl` into one list.

        A corrupt line is skipped silently: the journal is written during the
        run, and a truncated last line is normal for an interrupted run, not a
        reason to refuse to parse.
        """
        events: list[dict[str, Any]] = []
        for path in sorted((self.run_dir / "figures").rglob("events.jsonl")):
            figure_id = str(path.parent.relative_to(self.run_dir / "figures"))
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["figure_id"] = figure_id
                events.append(record)
        return events

    # --- derived values ---------------------------------------------
    @property
    def scaffold_kind(self) -> str:
        return str((self.config.get("scaffold") or {}).get("kind", "?"))

    @property
    def objective(self) -> str:
        """Selection objective declared by the config. Empty means there is none.

        Policies have no such config key: the selection scale is owned by the
        policy, and the config knows nothing about it. Returning a default
        here would print an objective the run never used.
        """
        return str((self.config.get("scaffold") or {}).get("objective", ""))

    @property
    def policy_name(self) -> str:
        return str((self.config.get("scaffold") or {}).get("policy", ""))

    @property
    def fitness_scales(self) -> dict[str, int]:
        """Scale in which each part's best candidate was chosen, counted per part.

        `contract` is the mean of IoU and GMS, `cd_runtime` is the fallback.
        This is the HARNESS scale (what the part is reported in), not the
        policy's own selection scale inside the rollout, which the harness
        does not know.
        """
        used: dict[str, int] = {}
        for record in self.per_figure:
            name = (record.get("runtime_metrics") or {}).get("fitness_scale")
            if name:
                used[str(name)] = used.get(str(name), 0) + 1
        return used

    @property
    def figures_without_iou(self) -> int:
        """Parts whose best candidate has no IoU computed.

        With a non-watertight GT (or prediction) there is no volumetric IoU,
        and the part was selected on a different scale than the rest. Counted
        from the measurement, not from what the scaffold declares, so it also
        holds for a policy that declares its own objective.
        """
        return sum(
            1 for record in self.per_figure
            if (record.get("metrics") or {}).get("iou") is None
        )

    @property
    def objectives_used(self) -> dict[str, int]:
        used: dict[str, int] = {}
        for record in self.per_figure:
            name = (record.get("runtime_metrics") or {}).get("objective")
            if name:
                used[str(name)] = used.get(str(name), 0) + 1
        return used

    @property
    def logging_mode(self) -> tuple[str, bool]:
        logging_config = self.config.get("logging") or {}
        return str(logging_config.get("level", "metrics")), bool(logging_config.get("profile", False))

    @property
    def walls(self) -> list[float]:
        return sorted(float(record.get("wall_sec") or 0.0) for record in self.per_figure)

    def figure(self, figure_id: str) -> dict[str, Any] | None:
        for record in self.per_figure:
            if record.get("figure_id") == figure_id:
                return record
        return None


# --- report sections -----------------------------------------------------------
def section_header(run: Run) -> None:
    _title(f"Run: {run.run_dir}")
    level, profile = run.logging_mode
    dataset_source = "subsample from manifest" if (run.config.get("subsample") or {}).get("manifest") else "directory of .stl"

    if run.objective:
        objective_line = f"  objective        {run.objective}"
        used = run.objectives_used
        if set(used) - {run.objective}:
            # A per-part fallback matters: part of the set was selected on a
            # different scale, so the run aggregate mixes two scales.
            objective_line += "   (actual: " + ", ".join(
                f"{name}: {count}" for name, count in sorted(used.items())
            ) + ")"
    else:
        # A policy has no selection scale in the config. Print what the harness
        # really knows: the scale of the chosen best and how many parts have no IoU.
        scales = ", ".join(f"{name}: {count}" for name, count in sorted(run.fitness_scales.items()))
        objective_line = f"  best-of scale    {scales or 'not recorded'}   (the policy owns the selection scale)"

    kind = run.scaffold_kind + (f" ({run.policy_name})" if run.policy_name else "")
    print(f"  scaffold         {kind}")
    print(objective_line)
    no_iou = run.figures_without_iou
    if no_iou:
        print(f"  no IoU           {no_iou} parts   (GT or prediction not watertight; selected on another scale)")
    print(f"  parts            {run.summary.get('num_figures', len(run.per_figure))}"
          f"   (set: {dataset_source}, signature {str(run.summary.get('dataset_signature', ''))[:12]})")
    print(f"  logging          level={level}, profile={profile}")
    # The seed is printed in the header: two runs are comparable only if it
    # matches, while a repeat for the noise floor needs it to differ.
    print(f"  run seed         {run.config.get('seed', 'not recorded (run predates seeding)')}")
    _print_provenance(run.provenance)
    print(f"  backend          {(run.config.get('execution') or {}).get('backend', '?')}"
          f", pool_size={(run.config.get('execution') or {}).get('pool_size', '?')}"
          f", n_workers={run.config.get('n_workers', '?')}")
    if run.summary.get("limited_to"):
        print(f"  WARNING          set truncated to {run.summary['limited_to']} parts; this is not a measurement")


def section_quality(run: Run) -> None:
    quality = run.summary.get("quality") or {}
    if not quality:
        _title("Quality")
        print("  metrics were not computed (compute_metrics: false); no quality breakdown")
        return

    _title("Quality")
    # score_with_zeros is primary: only it cannot be gamed through failures,
    # since score_without_invalid improves by failing hard parts (they drop
    # out of its denominator).
    print(f"  {'parts':<22}{_fmt(quality.get('n_figures')):>12}")
    print(f"  {'failures':<22}{_fmt(quality.get('n_failures')):>12}")
    print(f"  {'score (with zeros)':<22}{_fmt(quality.get('score_with_zeros')):>12}   <- primary")
    print(f"  {'score (no failures)':<22}{_fmt(quality.get('score_without_invalid')):>12}")

    print("\n  share of invalid (IR) and its breakdown:")
    print(f"    {'ir':<22}{_fmt(quality.get('ir')):>12}")
    print(f"    {'ir_execution':<22}{_fmt(quality.get('ir_execution')):>12}   did not build")
    print(f"    {'ir_not_watertight':<22}{_fmt(quality.get('ir_not_watertight')):>12}   built, but not watertight")
    # Printed only when nonzero: it means parts were lost together with the
    # machine, which is different from "the model failed" and must not hide
    # among always-zero rows.
    if quality.get("ir_no_result"):
        print(f"    {'ir_no_result':<22}{_fmt(quality.get('ir_no_result')):>12}   "
              "the part returned no measurement (worker died, run aborted)")
    # Printed only when nonzero, like `ir_no_result`; older runs lack the field.
    if quality.get("ir_iou_unavailable"):
        print(f"    {'ir_iou_unavailable':<22}{_fmt(quality.get('ir_iou_unavailable')):>12}   "
              "GT and prediction are valid, but IoU was not computed")

    # The identity score_with_zeros = (1-IR) * score_without checks failure
    # accounting. A violation is a code bug, not numeric noise, and must be
    # reported: all measurements of such a run are invalid.
    if quality.get("identity_holds") is False:
        print(f"\n  CONTRACT IDENTITY VIOLATED: gap {_fmt(quality.get('identity_gap'))}")
        print("  score_with_zeros != (1-IR)*score_without; failure accounting is broken, the measurement is invalid")

    by_stratum = quality.get("by_stratum") or {}
    if by_stratum:
        print("\n  by stratum:")
        print(f"    {'stratum':<22}{'parts':>9}{'score':>12}{'IR':>9}")
        for stratum, aggregate in sorted(by_stratum.items()):
            print(f"    {stratum:<22}{_fmt(aggregate.get('n_figures')):>9}"
                  f"{_fmt(aggregate.get('score_with_zeros')):>12}{_fmt(aggregate.get('ir')):>9}")


def section_failures(run: Run, top: int) -> None:
    _title("How rollouts ended")
    stop_reasons = Counter(record.get("stop_reason") or "(empty)" for record in run.per_figure)
    for reason, count in stop_reasons.most_common():
        print(f"  {count:>5}  {reason}")

    failed = [record for record in run.per_figure if record.get("error")]
    if not failed:
        print("  no rollout errors")
    else:
        print(f"\n  rollouts with an error: {len(failed)} of {len(run.per_figure)}")
        errors = Counter(_error_signature(record.get("error", "")) for record in failed)
        for signature, count in errors.most_common(top):
            print(f"    {count:>4}  {signature}")

    # Metric-level failures differ from rollout errors: a part may have been
    # built but failed the watertight filter.
    failures = Counter(
        (record.get("metrics") or {}).get("failure")
        for record in run.per_figure
        if (record.get("metrics") or {}).get("failure")
    )
    if failures:
        print("\n  failures by metric:")
        for failure, count in failures.most_common():
            print(f"    {count:>4}  {failure}")


def section_stop_scale(run: Run) -> None:
    """Stop threshold versus the scale in which the contract score is computed.

    The scaffold stops on the value of the **objective**, but reports the
    contract `score_i`, the mean of normalized IoU and GMS. These differ, and
    when the objective is narrower than the score, a rollout stops with
    unclosed slack on a metric the objective does not see. This section fixes
    nothing; it makes the mismatch visible.
    """
    stopped = [
        record for record in run.per_figure
        if "success threshold" in (record.get("stop_reason") or "")
    ]
    if not stopped:
        return

    # The part's objective lives in `runtime_metrics` (written by the scaffold),
    # not in the contract `metrics`, which hold measurement results only.
    objectives = {(record.get("runtime_metrics") or {}).get("objective") for record in stopped}
    objectives.discard(None)

    # Which contract-score metrics the objective actually sees. Derived from
    # `needs`, not from the name: guessing from a name substring would add
    # another place that diverges from `objective.py` on the next new objective.
    seen_fields: set[str] = set()
    for name in objectives:
        try:
            seen_fields |= set(objective_mod.get_objective(name).needs)
        except Exception:
            continue

    _title("Stop threshold vs report scale")
    print(f"  stopped by threshold           {len(stopped)} of {len(run.per_figure)}")
    print("  objective on these parts      "
          + (", ".join(sorted(objectives)) if objectives else "owned by the policy, not in the part record"))

    scores = [float(record["score"]) for record in stopped if record.get("score") is not None]
    if scores:
        print(f"  their contract score           median {_fmt(_percentile(scores, 50))},"
              f" min {_fmt(min(scores))}")

    # Both score metrics are always shown: a lagging one answers "why did it
    # stop early".
    for field, need in (("iou", objective_mod.NEED_IOU), ("gms_norm", objective_mod.NEED_GMS)):
        values = [
            float((record.get("metrics") or {})[field])
            for record in stopped
            if (record.get("metrics") or {}).get(field) is not None
        ]
        if not values:
            continue
        low = sum(1 for v in values if v < 0.9)
        note = ""
        # A metric the objective does not measure may lag, but if it lags on a
        # noticeable share of parts, the threshold stopped in the wrong place.
        if seen_fields and need not in seen_fields and low:
            note = f"   <- the objective does not see it, below 0.9 for {low} of {len(values)}"
        print(f"  {field:<30} median {_fmt(_percentile(values, 50))},"
              f" min {_fmt(min(values))}{note}")

    if objectives and objectives != {"hmean"}:
        print()
        print("  The objective is narrower than the contract score. A run on objective: hmean")
        print("  stops on the same scale it reports on.")


def section_worker_deaths(run: Run) -> None:
    """Executor deaths and lost parts.

    Always printed, even with no deaths: "zero" is a fact the reader needs,
    while a missing section reads as "not counted". A failure caused by a
    process death enters IR like bad geometry, so their number must be known
    before interpreting quality.
    """
    tech = run.summary.get("tech") or {}
    deaths = tech.get("worker_deaths") or {}
    total = int(tech.get("n_worker_deaths") or 0)
    lost_figures = int(tech.get("figures_lost_with_worker") or 0)
    executor_stats = tech.get("executor_stats") or {}
    pool = tech.get("pool_deaths") or {}

    _title("Executor deaths")
    n_figures = len(run.per_figure) or 1

    if not total and not lost_figures and not pool and not executor_stats:
        print("  the executor never died")
        return

    print(f"  {'executor deaths':<30}{_fmt(total):>10}"
          f"   on {tech.get('figures_with_worker_deaths', 0)} parts")
    for kind, count in sorted(deaths.items(), key=lambda item: -item[1]):
        print(f"    {_death_name(kind):<28}{_fmt(count):>10}")

    if lost_figures:
        print(f"\n  {'parts lost with the process':<30}{_fmt(lost_figures):>10}"
              f"   of {len(run.per_figure)}")
        for kind, count in sorted(pool.items(), key=lambda item: -item[1]):
            print(f"    {_death_name(kind):<28}{_fmt(count):>10}")

    if executor_stats:
        print("\n  execution pool events:")
        for kind, count in sorted(executor_stats.items()):
            print(f"    {_death_name(kind):<28}{_fmt(count):>10}")

    # The denominator is the number of executions, not parts: a part has dozens
    # of candidates, so "180% of parts" would mean nothing.
    n_exec = int((run.summary.get("cost_calls_total") or {}).get("exec") or 0)
    if n_exec:
        print(f"\n  share of executions that died: {total / n_exec:.1%} ({total} of {n_exec})")
    if lost_figures:
        print(f"  share of parts lost with the process: {lost_figures / n_figures:.1%}")

    alarming = (n_exec and total / n_exec > 0.05) or (lost_figures / n_figures > 0.01)
    if alarming:
        print("\n  WARNING: enough deaths to spoil the measurement. These failures went into IR,")
        print("  but say nothing about geometry: sort out the machine before the quality.")
    else:
        print("\n  These failures were counted in IR alongside bad geometry; when analysing quality")
        print("  subtract them from the failure count by hand.")


def _death_name(kind: str) -> str:
    """Human-readable name of a death cause. Unknown kinds print as is."""
    return {
        "timeout": "exceeded the timeout",
        "died": "killed (signal, OOM, OCC)",
        "unreadable": "result could not be parsed",
        "lost": "task lost by the proxy",
        "pool_broken": "worker pool broke down",
        "figure_no_result": "part returned no result",
        "figure_lost_with_worker": "part lost with the worker",
        "figure_unprocessed": "part was not processed",
        "proxy_restarts": "proxy restarts",
        "lost_tasks": "tasks lost",
    }.get(kind, kind)


def section_cost(run: Run) -> None:
    _title("Cost")
    total = run.summary.get("cost_calls_total") or {}
    per_figure = run.summary.get("cost_calls_per_figure") or {}
    if total:
        print(f"  {'call kind':<14}{'total':>10}{'per part':>12}")
        for kind in CALL_KINDS:
            if kind in total:
                print(f"  {kind:<14}{_fmt(total[kind]):>10}{_fmt(per_figure.get(kind)):>12}")
        other = sorted(set(total) - set(CALL_KINDS))
        for kind in other:
            print(f"  {kind:<14}{_fmt(total[kind]):>10}{_fmt(per_figure.get(kind)):>12}")

    tech = run.summary.get("tech") or {}
    tokens = tech.get("tokens") or {}
    if tokens:
        print("\n  tokens:")
        for key in ("prompt", "completion", "prompt_visual", "completion_visual"):
            if tokens.get(key):
                print(f"    {key:<20}{_fmt(tokens[key]):>14}"
                      f"{_fmt(tokens[key] / max(len(run.per_figure), 1)):>12} per part")

    attempts = tech.get("attempts") or {}
    calls = total
    retries = {
        kind: attempts[kind] - calls.get(kind, 0)
        for kind in attempts
        if attempts[kind] - calls.get(kind, 0) > 0
    }
    if retries:
        print("\n  retries after connection drops (attempts beyond calls):")
        for kind, extra in sorted(retries.items()):
            print(f"    {kind:<20}{extra:>10}")
    truncated = {kind: n for kind, n in (tech.get("truncated") or {}).items() if n}
    if truncated:
        # A separate line, not among errors: the call succeeded but the reply is
        # only its beginning. Fixed by the reply cap; without this line the cap
        # shows up only in the decision journal as an unclear assistant reply.
        print("\n  replies cut off by the cap (`max_tokens`):")
        for kind, n in sorted(truncated.items()):
            share = n / calls[kind] if calls.get(kind) else None
            tail = f"   of {_fmt(calls[kind])}, {share:.0%}" if share is not None else ""
            print(f"    {kind:<20}{_fmt(n):>10}{tail}")

    by_call = tech.get("tokens_by_call") or {}
    if len(by_call) > 1:
        # Reply length PER CHANNEL. The prompt/completion split above does not
        # answer this: scaffold decisions have short replies while repair
        # returns the whole candidate code, and the totals cannot tell them apart.
        print("\n  tokens per call by channel (prompt / completion):")
        for kind, row in sorted(by_call.items()):
            n = calls.get(kind) or 0
            if not n:
                continue
            print(f"    {kind:<14}{row.get('prompt', 0) / n:>10.0f}"
                  f"{row.get('completion', 0) / n:>10.0f}")

    if tech.get("n_endpoint_errors"):
        print(f"\n  endpoint errors: {tech['n_endpoint_errors']}")

    by_model = tech.get("calls_by_model") or {}
    if by_model:
        print("\n  calls by model:")
        for model, count in sorted(by_model.items(), key=lambda item: -item[1]):
            print(f"    {_short_model(model):<40}{_fmt(count):>10}")


def section_time(run: Run, top: int) -> None:
    _title("Time")
    walls = run.walls
    # Real wall time goes first: the sum of per-part times answers "how much
    # work was done", not "how long the run took".
    rollout = run.summary.get("wall_sec_rollout")
    if rollout:
        rollout = float(rollout)
        workers = run.summary.get("n_workers")
        effective = run.summary.get("effective_workers")
        hours = f" ({rollout / 3600:.2f} h)" if rollout >= 600 else ""
        print(f"  rollout wall     {_fmt(rollout)} s{hours}")
        if effective:
            share = f", utilisation {float(effective) / int(workers) * 100:.0f}%" if workers else ""
            print(f"  parallelism      {float(effective):.1f} effective workers"
                  f"{f' of {int(workers)}' if workers else ''}{share}")
    if walls:
        print(f"  wall per part    median {_fmt(_percentile(walls, 50))} s,"
              f" p90 {_fmt(_percentile(walls, 90))} s,"
              f" max {_fmt(walls[-1])} s")
        print(f"  wall total       {_fmt(sum(walls))} s"
              f"  (the run wall is shorter: parts ran in parallel)")
    if run.summary.get("wall_sec_run"):
        print(f"  whole run        {_fmt(run.summary.get('wall_sec_run'))} s"
              f"  (dataset, rollouts and summary; the report itself is not included)")

    slowest = sorted(run.per_figure, key=lambda record: -float(record.get("wall_sec") or 0.0))[:top]
    if slowest and float(slowest[0].get("wall_sec") or 0) > 0:
        print("\n  slowest parts:")
        for record in slowest:
            print(f"    {_fmt(record.get('wall_sec')):>9} s  {record.get('figure_id')}"
                  f"   steps: {record.get('n_steps')}, {record.get('stop_reason')}")

    stages = (run.summary.get("tech") or {}).get("stages_sec") or {}
    if stages:
        total_stage = sum(stages.values()) or 1.0
        print("\n  stages (summed over all parts):")
        for stage, seconds in sorted(stages.items(), key=lambda item: -item[1]):
            print(f"    {stage:<16}{_fmt(seconds):>12} s   {seconds / total_stage * 100:5.1f}%")
    else:
        print("\n  stage profiling is off (logging.profile: false); no per-stage breakdown")

    # Fork phases print regardless of profiling: the fork measures them itself,
    # at the cost of a few clock calls per candidate.
    section_exec_phases(run)


def section_exec_phases(run: Run) -> None:
    """What "execution" consists of: build, export, measure.

    The `execute` stage is the part's wall time over the whole call, while this
    is the sum over candidates inside forks, so the numbers need not match: the
    difference is executor overhead (queues, fork, serialization). Only
    candidates that ran to the end are counted; failed ones show up as the gap
    between `n` and the `exec` calls.
    """
    phases = (run.summary.get("tech") or {}).get("exec_phases") or {}
    n = int(phases.get("n") or 0)
    seconds = {key: float(value) for key, value in phases.items() if key != "n" and value}
    if not n or not seconds:
        return

    total = sum(seconds.values()) or 1.0
    n_exec = int((run.summary.get("cost_calls_total") or {}).get("exec") or 0)
    print(f"\n  of which execution, by fork phase ({n} candidates of {n_exec or n}):")
    titles = {
        "build_sec": "build",
        "export_sec": "mesh export",
        "metrics_sec": "metrics",
        "overhead_sec": "fork overhead",
    }
    for key, value in sorted(seconds.items(), key=lambda item: -item[1]):
        print(f"    {titles.get(key, key):<16}{_fmt(value):>12} s   {value / total * 100:5.1f}%"
              f"   {value / n * 1000:8.1f} ms/candidate")

    stage_exec = float(((run.summary.get("tech") or {}).get("stages_sec") or {}).get("execute") or 0.0)
    if stage_exec > total:
        print(f"    {'outside fork':<16}{_fmt(stage_exec - total):>12} s"
              f"          - pool queues, fork, result transfer")


def section_log_volume(run: Run) -> None:
    _title("Log volume")
    tech = run.summary.get("tech") or {}
    total = float(tech.get("log_bytes_total") or 0)
    mean = float(tech.get("log_bytes_per_figure_mean") or 0)
    level, _ = run.logging_mode
    print(f"  total            {_bytes(total)}")
    print(f"  per part         {_bytes(mean)}   (level {level})")
    if mean:
        # 3396 is the size of the held-out `cadenabench` set.
        print(f"  forecast         {_bytes(mean * 1000)} per 1000 parts,"
              f" {_bytes(mean * 3396)} for the dataset of 3396")


# What a miss means for each cache. Without this the hit rate is misread: a low
# rate is normal for `render_pred` (every prediction is new), but for `exec_gt`
# it means a lost warm-up and extra work on every candidate.
CACHE_HINTS = {
    "gt_image": "miss = part changed",
    "render_pred": "miss = new prediction (normal)",
    "gt_points": "miss = part changed",
    "exec_gt": "miss = GT reloaded in the fork",
    "det_gt": "miss = GT wireframe reloaded in the det fork",
    "det_gt_repair": "miss = leaky GT repaired by voxelization again",
}


def section_caches(run: Run) -> None:
    """Caches: how often they helped and how often work was redone.

    The question is whether the run pays again for what is already computed.
    Next to the hit rate the section prints **what a miss means** for that
    cache: for half of them a miss is routine (new prediction, new part), but
    for `exec_gt` it means the GT warm-up did not reach the fork and every
    candidate reloads and resamples the target.
    """
    caches = (run.summary.get("tech") or {}).get("caches") or {}
    if not caches:
        return

    _title("Caches")
    for name in sorted(caches):
        stats = caches[name] or {}
        hits = int(stats.get("hits") or 0)
        misses = int(stats.get("misses") or 0)
        if not (hits + misses):
            continue
        rate = stats.get("hit_rate")
        extra = []
        if stats.get("writes"):
            extra.append(f"writes {stats['writes']}")
        if stats.get("evictions"):
            extra.append(f"evicted {stats['evictions']}")
        tail = f"   ({', '.join(extra)})" if extra else ""
        share = "  —  " if rate is None else f"{rate * 100:5.1f}%"
        print(f"  {name:<14}{share} hits   {hits} of {hits + misses}{tail}"
              f"   {CACHE_HINTS.get(name, '')}")

    exec_gt = caches.get("exec_gt") or {}
    lookups = int(exec_gt.get("hits") or 0) + int(exec_gt.get("misses") or 0)
    backend = (run.config.get("execution") or {}).get("backend")
    if lookups and not exec_gt.get("hits") and backend == "proxy_pool":
        print("\n  WARNING: with backend=proxy_pool the GT cache in the fork never hit.")
        print("  The proxy warm-up does not reach the grandchild, so every candidate")
        print("  reloads and resamples the target. This directly wastes what")
        print("  this backend was chosen for.")

    evicted = sum(int((stats or {}).get("evictions") or 0) for stats in caches.values())
    if evicted:
        print(f"\n  WARNING: {evicted} entries evicted; the cache is too small for the run")
        print("  (experiment.cache.render_images). A part pays with a render for what")
        print("  it had already computed.")


def section_load(run: Run) -> None:
    """Machine load: processes, threads, core oversubscription."""
    load = run.summary.get("load") or {}
    if not load.get("n_samples"):
        return

    _title("Load")
    config = run.config
    n_workers = config.get("n_workers")
    execution = config.get("execution") or {}
    print(f"  configured       n_workers={n_workers},"
          f" backend={execution.get('backend', '?')}, pool_size={execution.get('pool_size', '?')}")
    quota = load.get("n_cpu_quota")
    machine = load.get("n_cpu_machine")
    if quota is not None and machine and quota < machine:
        # Always printed when they differ: oversubscription below is computed
        # against available cores, and without this line the reader would take
        # it to be against machine cores.
        print(f"  cores            {load.get('n_cpu')} available"
              f"   <- cgroup quota, {machine} on the machine")
    else:
        print(f"  cores            {load.get('n_cpu')}")
    print("  counted over the run process subtree: workers, execution pool, forks")
    print(f"  processes        peak {_fmt(load.get('n_processes_max'))},"
          f" mean {_fmt(load.get('n_processes_mean'))}")
    print(f"  threads          peak {_fmt(load.get('n_threads_max'))},"
          f" mean {_fmt(load.get('n_threads_mean'))}")

    per_cpu = load.get("threads_per_cpu_max")
    if per_cpu is not None:
        # The key figure of the section. The threshold of 2 threads per core
        # means "definitely contending", not "optimum".
        verdict = "cores oversubscribed" if per_cpu > 2 else "headroom left"
        print(f"  threads per core  peak {_fmt(per_cpu)}   <- {verdict}")

    print(f"  RSS              peak {_fmt(load.get('rss_mb_max'))} MB,"
          f" mean {_fmt(load.get('rss_mb_mean'))} MB")
    share = load.get("cpu_throttled_share")
    if share is None:
        pass
    elif quota is None:
        # Without a quota there is nothing to throttle, so CFS lines would be
        # noise; but say there was no quota, since silence reads as "not measured".
        print(f"  pod CPU          {_fmt(load.get('cpu_usage_sec'))} s of work,"
              " no cgroup quota")
    else:
        # The one cost absent from every stage: time when work was ready to run
        # but the quota held it back. Stages show it only as "everything got slower".
        verdict = "wall time lost to throttling" if share > 0.05 else "quota did not interfere"
        print(f"  pod CPU          {_fmt(load.get('cpu_usage_sec'))} s of work,"
              f" {_fmt(load.get('cpu_throttled_sec'))} s taken by the quota"
              f" ({100.0 * share:.0f}%)   <- {verdict}")
        print(f"  CFS periods      {_fmt(load.get('cpu_periods'))},"
              f" throttled {_fmt(load.get('cpu_periods_throttled'))}")
        print("  cgroup counters are pod-wide: they include the vLLM servers")

    if load.get("loadavg_1_max") is not None:
        # loadavg is machine-wide: it includes vLLM servers and neighbors.
        # Reading it as our own load is a common mistake, hence the note in the line.
        print(f"  loadavg(1m)      peak {_fmt(load.get('loadavg_1_max'))}   <- whole machine,"
              f" not only the run")

    # An execution fork lives only a second or so and the background sampler
    # almost always misses it, so the fork measures its own threads and the
    # figure is shown separately.
    tech = run.summary.get("tech") or {}
    fork_threads = dict(tech.get("fork_threads") or {})
    # Older runs recorded only the execution fork and named the field differently.
    if not fork_threads and (tech.get("exec_threads") or {}).get("n"):
        fork_threads = {"exec": tech["exec_threads"]}
    FORK_LABELS = {"exec": "exec", "det": "det"}
    for kind, stats in sorted(fork_threads.items()):
        if not stats.get("n"):
            continue
        print(f"  fork threads    {FORK_LABELS.get(kind, kind):11} max {_fmt(stats.get('max'))},"
              f" mean {_fmt(stats.get('mean'))}"
              f"   (measured by the fork itself, {stats.get('n')} calls)")

    env = load.get("thread_env") or {}
    if env:
        print("\n  thread limits of native libraries:")
        for name, value in sorted(env.items()):
            print(f"    {name:<28}{value}")
    else:
        print("\n  thread limits of native libraries are not set:"
              "\n  each library decides its thread count itself")

    graphics = load.get("graphics_env") or {}
    if graphics:
        print("\n  render graphics stack:")
        for name, value in sorted(graphics.items()):
            # An empty value is a working state, not a missing setting: this is
            # how GPUs are hidden from the driver.
            shown = "'' (GPUs hidden)" if name == "CUDA_VISIBLE_DEVICES" and value == "" else value
            print(f"    {name:<28}{shown}")

    print(f"\n  samples          {load.get('n_samples')}"
          f" (every {_fmt(load.get('interval_sec'))} s), details in load.jsonl")


def section_servers(run: Run) -> None:
    """vLLM engine counters over the rollouts: the difference of two `/metrics` snapshots."""
    servers = run.summary.get("servers") or {}
    if not servers:
        return
    _title("Model servers (/metrics, server-wide counters)")
    for role, d in sorted(servers.items()):
        if d.get("restarted"):
            print(f"  {role:<12}restarted mid-run, counters are not comparable")
            continue
        print(f"  {role:<12}prefix cache {_fmt(d.get('prefix_cache_hit_share'))},"
              f" multimodal {_fmt(d.get('mm_cache_hit_share'))},"
              f" prompt {_fmt(d.get('prompt_tok_per_sec'))} tok/s,"
              f" completion {_fmt(d.get('generation_tok_per_sec'))} tok/s"
              f" over {_fmt(d.get('window_sec'))} s")


def section_decisions(run: Run, top: int) -> None:
    if not run.events:
        return
    _title("Scaffold decisions and capability calls")
    kinds = Counter(event.get("kind") for event in run.events)
    for kind, count in kinds.most_common():
        print(f"  {count:>6}  {kind}")

    decisions = Counter(
        event.get("decision") for event in run.events if event.get("kind") == "decision"
    )
    if decisions:
        print("\n  decisions:")
        for decision, count in decisions.most_common():
            print(f"    {count:>5}  {decision}")

    modes = Counter(
        event.get("mode") for event in run.events
        if event.get("kind") == "decision" and event.get("decision") == "selection"
    )
    if modes:
        print("\n  selection mode:")
        for mode, count in modes.most_common():
            print(f"    {count:>5}  {mode}")

    reasons = Counter(
        _short(str(event.get("reason") or event.get("reason") or ""))
        for event in run.events
        if event.get("kind") == "decision" and (event.get("reason") or event.get("reason"))
    )
    if reasons:
        print("\n  most frequent reasons for a decision:")
        for reason, count in reasons.most_common(top):
            print(f"    {count:>5}  {reason}")

    # Empty generator replies are visible only here: the scaffold filters them
    # out before journaling, so they never reach the per-part table.
    proposals = [event for event in run.events if event.get("kind") == "propose_steps"]
    if proposals:
        requested = sum(int(event.get("requested") or 0) for event in proposals)
        returned = sum(int(event.get("returned") or 0) for event in proposals)
        failed = sum(int(event.get("failed") or 0) for event in proposals)
        print(f"\n  step generator: requested {requested}, returned {returned}, failed {failed}")

    evaluations = [event for event in run.events if event.get("kind") == "evaluate"]
    if evaluations:
        total = sum(int(event.get("n") or 0) for event in evaluations)
        ok = sum(int(event.get("ok") or 0) for event in evaluations)
        timeouts = sum(int(event.get("timeouts") or 0) for event in evaluations)
        share = ok / total * 100 if total else 0.0
        print(f"  execution: candidates {total}, built {ok} ({share:.1f}%), timeouts {timeouts}")


# --- single part ------------------------------------------------------
def report_figure(run: Run, figure_id: str) -> None:
    record = run.figure(figure_id)
    if record is None:
        available = ", ".join(sorted(r.get("figure_id", "") for r in run.per_figure)[:10])
        raise SystemExit(f"No such part in the run: {figure_id}. For example: {available}")

    _title(f"Part {figure_id}")
    for key in ("group", "n_steps", "stop_reason", "score", "wall_sec", "mesh_path"):
        print(f"  {key:<14} {_fmt(record.get(key))}")
    if record.get("error"):
        print(f"  error:\n    {_short(record['error'], 300)}")

    metrics = record.get("metrics") or {}
    if metrics:
        print("\n  metrics:")
        for key, value in sorted(metrics.items()):
            print(f"    {key:<16} {_fmt(value)}")

    journal = _load_json(run.run_dir / "figures" / figure_id / "journal.json", [])
    if journal:
        print("\n  steps (scaffold journal):")
        for entry in journal:
            step = entry.get("step", "-")
            bits = [f"{key}={_fmt(value)}" for key, value in entry.items() if key != "step"]
            print(f"    step {str(step):>3}  " + ", ".join(bits)[:200])

    events = [event for event in run.events if event.get("figure_id") == figure_id]
    if events:
        print("\n  events:")
        for event in events:
            fields = {
                key: value for key, value in event.items()
                if key not in ("t", "kind", "figure_id")
            }
            rendered = ", ".join(f"{key}={_short(str(value), 80)}" for key, value in fields.items())
            print(f"    {_fmt(event.get('t')):>9}  {event.get('kind'):<16} {rendered[:200]}")


# --- comparing two runs --------------------------------------------------
def report_compare(run: Run, other: Run) -> None:
    _title(f"Comparison: {run.run_dir.name} vs {other.run_dir.name}")

    # Comparability is checked before the numbers: runs with different sets or
    # logging levels cannot be compared, and this must not pass silently.
    problems: list[str] = []
    # Notes are not violations: they are printed, but the runs stay comparable
    # (repeating the same scaffold with another seed is a routine measurement).
    notes: list[str] = []
    if run.summary.get("dataset_signature") != other.summary.get("dataset_signature"):
        problems.append("different sets (dataset signature differs)")
    if run.logging_mode != other.logging_mode:
        problems.append(
            f"logging level and profiling differ: {_mode(run.logging_mode)} vs "
            f"{_mode(other.logging_mode)} - logging overhead will go straight into the difference"
        )
    if run.summary.get("limited_to") or other.summary.get("limited_to"):
        problems.append("one of the runs is truncated by limit; this is not a measurement")
    if (run.config.get("execution") or {}).get("backend") != (other.config.get("execution") or {}).get("backend"):
        problems.append("execution backends differ; the time difference comes from this too")
    # Different seeds mean different sampled points, i.e. different model input.
    # For two DIFFERENT scaffolds this breaks the comparison: the scaffold effect
    # adds to sampling noise. For ONE scaffold it is a routine repeat to estimate
    # the noise floor, and complaining would be wrong.
    seeds = (run.config.get("seed"), other.config.get("seed"))
    if seeds[0] != seeds[1]:
        if run.scaffold_kind != other.scaffold_kind:
            problems.append(
                f"seeds differ ({seeds[0]} vs {seeds[1]}): the model got different points, "
                "and the scaffold effect is mixed with sampling noise"
            )
        else:
            notes.append(
                f"same scaffold with different seeds ({seeds[0]} vs {seeds[1]}): "
                "the difference is the noise floor, not a scaffold effect"
            )

    if problems:
        print("  NOT COMPARABLE:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  comparability: set, logging level, profiling and backend match")
    for note in notes:
        print(f"  REPEAT           {note}")

    rows: list[tuple[str, Any, Any]] = [
        ("scaffold", run.scaffold_kind, other.scaffold_kind),
        ("parts", run.summary.get("num_figures"), other.summary.get("num_figures")),
        ("score (with zeros)", (run.summary.get("quality") or {}).get("score_with_zeros"),
         (other.summary.get("quality") or {}).get("score_with_zeros")),
        ("score (no failures)", (run.summary.get("quality") or {}).get("score_without_invalid"),
         (other.summary.get("quality") or {}).get("score_without_invalid")),
        ("IR", (run.summary.get("quality") or {}).get("ir"),
         (other.summary.get("quality") or {}).get("ir")),
        ("wall per part", run.summary.get("wall_sec_per_figure_mean"),
         other.summary.get("wall_sec_per_figure_mean")),
    ]
    for kind in CALL_KINDS:
        left = (run.summary.get("cost_calls_per_figure") or {}).get(kind)
        right = (other.summary.get("cost_calls_per_figure") or {}).get(kind)
        if left or right:
            rows.append((f"{kind} per part", left, right))

    print(f"\n  {'':<22}{run.run_dir.name:>16}{other.run_dir.name:>16}{'difference':>14}")
    for label, left, right in rows:
        print(f"  {label:<22}{_fmt(left):>16}{_fmt(right):>16}{_delta(left, right):>14}")

    print("\n  Cost weights are derived from this table: normalised to the cheapest")
    print("  call kind, then how many mutants fit into the experiment budget.")


# --- helpers -------------------------------------------------------------------
def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[!] {path} is not valid JSON: {exc}")
        return default


def _print_provenance(prov: dict[str, Any]) -> None:
    """Code and servers of the run go into the header, so pairs are not compared from memory."""
    if not prov:
        print("  code             not recorded (run predates provenance.json)")
        return
    code = prov.get("code") or {}
    git = prov.get("git") or {}
    line = f"  code             digest {code.get('digest') or code.get('error') or '?'}"
    if git:
        line += f", commit {git.get('commit', '')[:10]}" + (f" + {len(git['dirty'])} dirty" if git.get("dirty") else "")
    print(line)
    changed = (prov.get("code_end") or {}).get("changed") or []
    if changed:
        print(f"  WARNING          code on disk changed during the run: {len(changed)} files ({', '.join(changed[:3])}…)")
    for role, ident in sorted((prov.get("servers") or {}).items()):
        roots = ", ".join(str(m.get("root") or m.get("id")) for m in ident.get("models") or []) or "?"
        end = (prov.get("servers_end") or {}).get(role)
        moved = bool(end) and bool(ident) and end != ident
        print(f"  server {role:<10}{roots}" + ("   CHANGED during the run" if moved else ""))


def _error_signature(error: str) -> str:
    """Reduce a traceback to a single line that errors can be grouped by."""
    lines = [line.strip() for line in (error or "").strip().splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    for line in reversed(lines):
        if ":" in line and not line.startswith(("File ", "  ")):
            return _short(line, 120)
    return _short(lines[-1], 120)


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    index = min(int(round(percent / 100 * (len(values) - 1))), len(values) - 1)
    return values[index]


def _delta(left: Any, right: Any) -> str:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return ""
    if isinstance(left, bool) or isinstance(right, bool):
        return ""
    if right == 0:
        return f"{left - right:+.4g}"
    return f"{left - right:+.4g} ({(left - right) / abs(right) * 100:+.0f}%)"


def _mode(mode: tuple[str, bool]) -> str:
    level, profile = mode
    return f"level={level}, profile={'yes' if profile else 'no'}"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def _short(text: str, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short_model(model: str) -> str:
    """A checkpoint path in a model name is long and uninformative in full."""
    return model if len(model) <= 40 else "…" + model[-39:]


def _title(text: str) -> None:
    print(f"\n{text}\n" + "-" * max(len(text), 40))


# --- assembly and writing ----------------------------------------------------------
def build_report(run_dir: str | Path, top: int = 10, read_events: bool = True) -> str:
    """Build the full report for a run directory as one string."""
    run = Run(run_dir, read_events=read_events)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        section_header(run)
        section_quality(run)
        section_failures(run, top)
        section_stop_scale(run)
        section_worker_deaths(run)
        section_cost(run)
        section_time(run, top)
        section_log_volume(run)
        section_caches(run)
        section_load(run)
        section_servers(run)
        section_decisions(run, top)
        print()
    return buffer.getvalue()


def write_report(run_dir: str | Path, top: int = 10) -> Path | None:
    """Write the report to `<run dir>/logs/report.txt`.

    Called at the very end of a run, when all artifacts are on disk. The report
    is a convenience, not a result: any failure while building it is logged and
    swallowed so it cannot fail an already computed run.
    """
    path = Path(run_dir) / REPORT_PATH
    try:
        text = build_report(run_dir, top=top)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception:
        logger.exception("Report could not be built; this does not invalidate the run, the data on disk is intact")
        return None
    logger.info("Run report: %s", path)
    return path
