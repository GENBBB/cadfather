"""Contract entry point: run the pipeline with the given scaffold.

    def run_eval(scaffold, shapes) -> list[float]

The experiment validator calls only this function, and both branches (fast and
agent) go through it. `scaffold=None` means the fast branch: the baseline against
which everything else is measured.

The order and length of the result are stable between calls: scores are returned
in the order of the input part list, a failure gives zero, not a gap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cad_agent.capabilities import metrics as metrics_mod
from cad_agent.harness import scratch
from cad_agent.harness.artifacts import RunLayout
from cad_agent.harness.config import DEFAULT_SCAFFOLD_KIND, validate_run_config
from cad_agent.harness.dataset import (
    FigureSpec,
    dataset_signature,
    load_figures,
    load_figures_from_manifest,
)
from cad_agent.harness.load import LoadSampler
from cad_agent.harness.pool import pool_deaths, run_figures
from cad_agent.harness import provenance, server_metrics
from cad_agent.harness.report import write_report
from cad_agent.harness.subsample import SPLIT_EVO

logger = logging.getLogger(__name__)


def run_eval(
    scaffold: Any,
    shapes: list[FigureSpec],
    config: dict[str, Any],
    run_dir: str | Path,
) -> list[float]:
    """Per-figure scores in the order of `shapes`. `scaffold=None` is the fast branch."""
    records = run_experiment(scaffold=scaffold, figures=shapes, config=config, run_dir=run_dir)["per_figure"]
    return [float(record.get("score") or 0.0) for record in records]


def build_scaffold(config: dict[str, Any], scaffold: Any = None) -> Any:
    """Complete the scaffold: one passed explicitly or the named policy from the config."""
    if scaffold is not None:
        return scaffold

    scaffold_config = dict(config.get("scaffold", {}) or {})
    kind = scaffold_config.pop("kind", DEFAULT_SCAFFOLD_KIND)
    if kind != DEFAULT_SCAFFOLD_KIND:
        raise ValueError(
            f"Unknown scaffold scaffold.kind={kind!r}. Expected {DEFAULT_SCAFFOLD_KIND!r}"
        )

    # The search loop lives in the harness, the policy arrives as a named policy.
    # Outwardly it is an object with `run(obs, res)`: the harness defines the
    # contract seam, and the policy is responsible only for `plan` and `select`.
    from cad_agent.harness.search import SearchLoop
    from cad_agent.scaffold.policies import build as build_policy

    return SearchLoop(build_policy(scaffold_config.pop("policy")))


def run_experiment(
    config: dict[str, Any],
    run_dir: str | Path,
    scaffold: Any = None,
    figures: list[FigureSpec] | None = None,
    source_config: str | Path | None = None,
) -> dict[str, Any]:
    """Full run: set -> rollouts -> per-figure table and aggregates.

    `source_config` is the YAML the run started from; its text goes into
    `provenance.json`, because the file in the repository is edited after the run too.
    """
    run_started = time.monotonic()
    # The config is validated before processes and models come up: a typo in a key
    # should cost a second at startup, not hours of a run.
    validate_run_config(config)

    figures = figures if figures is not None else load_dataset(config)
    signature = dataset_signature(figures)

    layout = RunLayout(run_dir)
    layout.save_config(config)
    layout.save_dataset(figures, signature)
    # Provenance is written at the start, not at the end: an interrupted run must
    # also know which code it ran on. At the end the file is supplemented with what
    # could have changed during the run (code on disk, server processes).
    started = provenance.collect_start(config, source_config)
    layout.save_provenance(started)

    scaffold = build_scaffold(config, scaffold)
    logger.info("Scaffold: %s, set: %d parts, signature %s", type(scaffold).__name__, len(figures), signature[:12])

    # Load sampling runs for the whole run and covers the whole subtree: workers,
    # execution-pool shims and their forks. Without it it is unknown whether the
    # run is bound by cores or idle.
    logging_config = config.get("logging") or {}
    sampler = LoadSampler(
        run_dir=layout.run_dir,
        interval_sec=float(logging_config.get("load_interval_sec", 5.0)),
        enabled=bool(logging_config.get("load", True)),
    )
    n_workers = int(config.get("n_workers", 8))
    # The real wall time of the rollouts. The sum of per-figure `wall_sec` answers
    # "how much work was done", not "how long it took": parts run in parallel, and
    # with 16 workers these numbers differ by more than an order of magnitude.
    # Measured outside the pool, because inside nobody sees the whole: each worker
    # knows only its own part.
    # Engine counters are a snapshot before and after the rollouts: the cache hit
    # share at DP > 1 exists only in `/metrics`, the engine log does not write it.
    servers_before = server_metrics.snapshot(config)
    rollout_started = time.monotonic()
    with sampler:
        try:
            records = run_figures(
                figures=figures,
                config=config,
                scaffold=scaffold,
                run_dir=layout.run_dir,
                n_workers=n_workers,
            )
        finally:
            # Scratch files (candidate meshes in tmpfs) are cleaned up by the part, but
            # a dead worker never reaches its cleanup. Here is a general sweep for the
            # whole run: tmpfs is memory, and leaving garbage in it until the node
            # reboots is not acceptable.
            scratch.cleanup(
                scratch.run_dir(
                    layout.run_dir.name, (config.get("execution") or {}).get("scratch_dir")
                )
            )

    rollout_wall = time.monotonic() - rollout_started
    servers = server_metrics.summarize(servers_before, server_metrics.snapshot(config), rollout_wall)

    summary = summarize(
        records,
        signature=signature,
        wall_sec_rollout=rollout_wall,
        n_workers=n_workers,
        compute_metrics=bool(config.get("compute_metrics", True)),
    )
    summary["tech"]["pool_deaths"] = pool_deaths()
    summary["tech"]["figures_lost_with_worker"] = sum(
        1 for record in records if record.get("worker_died")
    )
    summary["load"] = sampler.summary()
    _log_load(summary["load"])
    summary["servers"] = servers
    _log_servers(servers)
    _log_worker_deaths(summary["tech"], len(records), summary.get("cost_calls_total"))
    if config.get("limit"):
        summary["limited_to"] = int(config["limit"])
    layout.save_per_figure(records)
    layout.save_per_figure_csv(records)
    # The run wall time is the last thing that goes into the summary: after it only
    # the report, which reads the already written `summary.json` and so is not included.
    summary["wall_sec_run"] = round(time.monotonic() - run_started, 3)
    layout.save_summary(summary)
    finished = provenance.collect_end(started, config)
    layout.save_provenance(finished)
    _log_provenance(finished)

    # The report goes last: it reads what is written above. Switched off with
    # `logging.report: false`: on big runs it walks `events.jsonl` of all parts,
    # which is thousands of small NFS reads.
    report_started = time.monotonic()
    if (config.get("logging") or {}).get("report", True):
        write_report(layout.run_dir)
        # The report time goes to the log, not to the summary: the summary is already
        # written, and appending to it afterwards would make the report
        # irreproducible from it (`tests/report_check.py` catches this). On big runs
        # the number is needed: the report walks `events.jsonl` of all parts, which
        # is thousands of small NFS reads.
        logger.info("Report built in %.1f s", time.monotonic() - report_started)

    return {"per_figure": records, "summary": summary, "run_dir": str(layout.run_dir)}


def _log_worker_deaths(tech: dict[str, Any], n_figures: int, cost: dict[str, Any] | None = None) -> None:
    """Executor deaths must be announced, not left lying in JSON.

    A failure caused by a process death enters IR alongside bad geometry, that is,
    it spoils exactly the number the run was made for. It must not be passed over
    in silence even when deaths are few.
    """
    total = int(tech.get("n_worker_deaths") or 0)
    restarts = int((tech.get("executor_stats") or {}).get("proxy_restarts") or 0)
    lost_figures = int(tech.get("figures_lost_with_worker") or 0)
    if not total and not restarts and not lost_figures:
        return

    if lost_figures:
        logger.error(
            "Parts lost together with the process: %d of %d (%s). This is not a model failure: "
            "they score zero in the table, but the machine explains it, not the geometry",
            lost_figures, n_figures,
            ", ".join(f"{kind}: {count}" for kind, count in sorted((tech.get("pool_deaths") or {}).items())) or "—",
        )

    logger.warning(
        "The executor died %d times on %d parts (%s); proxy restarts: %d. "
        "These failures went into IR but say nothing about geometry",
        total, tech.get("figures_with_worker_deaths", 0),
        ", ".join(f"{kind}: {count}" for kind, count in sorted((tech.get("worker_deaths") or {}).items())) or "—",
        restarts,
    )
    n_exec = int((cost or {}).get("exec") or 0)
    if n_exec and total / n_exec > 0.05:
        logger.error(
            "Executor deaths are %.1f%% of executions: the measurement is unreliable; "
            "sort out the machine before computing quality", 100.0 * total / n_exec,
        )


def _log_load(load: dict[str, Any]) -> None:
    """Report load to the log immediately: the number is needed before someone finds it."""
    if not load.get("n_samples"):
        return
    quota = load.get("n_cpu_quota")
    machine = load.get("n_cpu_machine")
    # The machine count is mentioned only when it lies: under a quota the run goes
    # at the quota's speed, and "192 cores" in this line was read as headroom for years.
    cores = (f"{load.get('n_cpu')} available cores (cgroup quota, {machine} on the machine)"
             if quota is not None and machine and quota < machine
             else f"{load.get('n_cpu')} cores")
    logger.info(
        "Load: peak %s processes / %s threads (%s per core on %s), RSS up to %s MB",
        load.get("n_processes_max"), load.get("n_threads_max"),
        load.get("threads_per_cpu_max"), cores, load.get("rss_mb_max"),
    )
    if (load.get("threads_per_cpu_max") or 0) > 2:
        logger.warning(
            "Peak threads per core %s: cores are oversubscribed, threads fight instead of working. "
            "Check n_workers, execution.pool_size and thread limits (%s)",
            load.get("threads_per_cpu_max"), load.get("thread_env") or "not set",
        )
    # CPU time taken away by the quota is not visible in any stage: it is smeared
    # across all of them and looks like "got slower".
    share = load.get("cpu_throttled_share")
    if share is not None and quota is not None:
        logger.info(
            "Pod CPU over the run: %s s of work, %s s taken by the quota (%.0f%%), "
            "%.0f%% of periods throttled",
            load.get("cpu_usage_sec"), load.get("cpu_throttled_sec"), 100.0 * share,
            100.0 * (load.get("cpu_periods_throttled_share") or 0.0),
        )
    if (share or 0) > 0.05:
        logger.warning(
            "The quota took %.0f%% of the pod CPU time: the run wall was measured under "
            "throttling and is comparable only with a run under the same. Check n_workers "
            "and execution.pool_size against cpu.max, not against os.cpu_count()",
            100.0 * share,
        )


def _log_provenance(prov: dict[str, Any]) -> None:
    """A change of code or servers in the middle of a run must be announced."""
    changed = (prov.get("code_end") or {}).get("changed") or []
    if changed:
        logger.warning(
            "Code on disk changed during the run (%d files: %s): some parts may have run "
            "on the new code; workers forked before the edit keep the old one",
            len(changed), ", ".join(changed[:5]) + (" …" if len(changed) > 5 else ""),
        )
    for role, before in sorted((prov.get("servers") or {}).items()):
        after = (prov.get("servers_end") or {}).get(role) or {}
        if before and after and before != after:
            logger.warning("Server %s changed during the run: %s -> %s", role, before, after)


def _log_servers(servers: dict[str, dict[str, Any]]) -> None:
    """Engine cache hit share goes to the log immediately, next to the load."""
    for role, d in sorted(servers.items()):
        if d.get("restarted"):
            logger.warning("Server %s restarted mid-run: /metrics counters are not comparable", role)
            continue
        logger.info(
            "Server %s over the rollouts: prefix cache %s, multimodal %s, prompt %s tok/s, "
            "completion %s tok/s (server-wide counters)",
            role, d.get("prefix_cache_hit_share"), d.get("mm_cache_hit_share"),
            d.get("prompt_tok_per_sec"), d.get("generation_tok_per_sec"),
        )


def load_dataset(config: dict[str, Any]) -> list[FigureSpec]:
    """The run set: a fixed manifest subsample or a directory with `.stl` files.

    The manifest takes priority: if it is given, the directory from `details` is
    not looked at at all, otherwise it is easy to get a run on "almost the subsample".
    """
    subsample = config.get("subsample") or {}
    manifest_path = subsample.get("manifest")
    if manifest_path:
        figures = load_figures_from_manifest(
            manifest_path=manifest_path,
            split=str(subsample.get("split", SPLIT_EVO)),
            verify=bool(subsample.get("verify", True)),
        )
    else:
        figures = load_figures(config["details"])

    limit = config.get("limit")
    if limit:
        # Truncating the set is for smoke runs only. It is written loudly to the log
        # and goes into the summary: a silently truncated run looks full and spoils
        # any comparison.
        limit = int(limit)
        if limit < len(figures):
            logger.warning(
                "SET TRUNCATED: %d parts of %d (experiment.limit). Not valid for measurements.",
                limit, len(figures),
            )
            figures = figures[:limit]
    return figures


def summarize(
    records: list[dict[str, Any]],
    signature: str = "",
    wall_sec_rollout: float | None = None,
    n_workers: int | None = None,
    compute_metrics: bool | None = None,
) -> dict[str, Any]:
    """Run and per-stratum aggregates plus cost in calls.

    Quality is computed via `metrics.aggregate`, which itself checks the identity
    `score_with_zeros = (1 - IR) * score_without` and splits IR into execution
    errors and non-watertight.

    `wall_sec_rollout` is the **real** rollout wall time, measured outside the
    pool. Without it only the sum of per-figure times is known, and that is not the
    run time: parts went in parallel.

    A part that returned no measurement at all (the worker died with it, the run
    was interrupted) goes into the aggregate as a failure, not a gap: by contract
    it has zero in the score vector, and a gap would drop it from the denominator
    and **improve** both scores exactly where the part turned out hard.
    `compute_metrics` separates this case from a run where nobody has metrics
    because they were not computed: there is no quality aggregate at all.
    """
    if compute_metrics is None:
        compute_metrics = any(record.get("metrics") for record in records)

    figure_metrics = []
    for record in records:
        if record.get("metrics"):
            figure_metrics.append(metrics_mod.FigureMetrics.from_dict(record["metrics"]))
        elif compute_metrics:
            figure_metrics.append(
                metrics_mod.FigureMetrics(
                    figure_id=str(record.get("figure_id") or ""),
                    failure=metrics_mod.FAILURE_NO_RESULT,
                    error=record.get("error"),
                )
            )

    quality: dict[str, Any] = {}
    if figure_metrics:
        quality = metrics_mod.aggregate(figure_metrics).to_dict()
        quality["by_stratum"] = {
            stratum: aggregate.to_dict()
            for stratum, aggregate in metrics_mod.aggregate_by_stratum(figure_metrics).items()
        }

    cost: dict[str, int] = {}
    for record in records:
        for kind, value in (record.get("cost", {}).get("calls", {}) or {}).items():
            cost[kind] = cost.get(kind, 0) + int(value)

    tech = _aggregate_tech(records)
    wall = [float(record.get("wall_sec") or 0.0) for record in records]
    summary = {
        "dataset_signature": signature,
        "num_figures": len(records),
        "num_failed": sum(1 for record in records if record.get("error")),
        "quality": quality,
        "cost_calls_total": cost,
        "cost_calls_per_figure": {kind: value / max(len(records), 1) for kind, value in cost.items()},
        "wall_sec_total": sum(wall),
        "wall_sec_per_figure_mean": sum(wall) / max(len(records), 1),
        "tech": tech,
    }
    if n_workers is not None:
        summary["n_workers"] = int(n_workers)
    if wall_sec_rollout is not None:
        summary["wall_sec_rollout"] = round(float(wall_sec_rollout), 3)
        # How many workers were actually busy on average. Less than declared means
        # the machine idled: a tail of long parts, waiting for the endpoint, too
        # fine slicing. Read only together with `n_workers`: 15.0 of 16 and 15.0 of
        # 32 are different news.
        summary["effective_workers"] = round(sum(wall) / wall_sec_rollout, 2) if wall_sec_rollout > 0 else None
    return summary


def _aggregate_tech(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Technical metrics for the run: tokens, latency, retries, stages, log volume.

    Always computed, regardless of the logging level: without them the cost of
    branches cannot be compared, and they cost a few additions.
    """
    totals: dict[str, dict[str, float]] = {
        "tokens": {}, "latency_sec": {}, "attempts": {}, "stages_sec": {}, "calls_by_model": {},
        # Answers cut off by the cap, per channel: summed like everything else
        # here, by channel key.
        "truncated": {},
        # Executor deaths by reason and events of the execution pool itself.
        "worker_deaths": {}, "executor_stats": {},
        # Execution broken down by fork phases: summed by the same loop, including
        # the counter `n`: candidates per part are summed exactly so.
        "exec_phases": {},
    }
    endpoint_errors = 0
    log_bytes = 0
    figures_with_deaths = 0
    # Fork threads by kind (`exec`, `det`): each has its own maximum and its own
    # mean weighted by the number of measurements, otherwise a rare but fat fork
    # would dissolve into a frequent thin one.
    fork_threads: dict[str, dict[str, float]] = {}
    # Caches: hits and misses are summed over all parts, and the share is
    # recomputed from the sum. Averaging per-figure shares would be wrong: a part
    # with two accesses would weigh as much as a part with two hundred.
    caches: dict[str, dict[str, int]] = {}
    # Tokens per channel. With its own accumulator rather than the common loop
    # below: the values here are nested, and that loop sums flat numbers.
    tokens_by_call: dict[str, dict[str, int]] = {}

    for record in records:
        tech = record.get("tech") or {}
        for kind, row in (tech.get("tokens_by_call") or {}).items():
            acc = tokens_by_call.setdefault(kind, {"prompt": 0, "completion": 0})
            for field in acc:
                acc[field] += int(row.get(field) or 0)
        for name, stats in (tech.get("caches") or {}).items():
            acc = caches.setdefault(name, {"hits": 0, "misses": 0, "writes": 0, "evictions": 0})
            for field in acc:
                acc[field] += int(stats.get(field) or 0)
        for kind, stats in (tech.get("fork_threads") or {}).items():
            if not stats.get("n"):
                continue
            acc = fork_threads.setdefault(kind, {"max": 0, "sum": 0.0, "n": 0})
            acc["max"] = max(acc["max"], int(stats.get("max") or 0))
            acc["sum"] += float(stats.get("mean") or 0) * int(stats["n"])
            acc["n"] += int(stats["n"])
        log_bytes += int(record.get("log_bytes") or 0)
        endpoint_errors += int(tech.get("n_endpoint_errors") or 0)
        if tech.get("n_worker_deaths"):
            figures_with_deaths += 1
        for section in totals:
            for key, value in (tech.get(section) or {}).items():
                totals[section][key] = totals[section].get(key, 0) + value

    n = max(len(records), 1)
    return {
        **{section: values for section, values in totals.items()},
        "n_endpoint_errors": endpoint_errors,
        "tokens_by_call": {kind: dict(acc) for kind, acc in sorted(tokens_by_call.items())},
        "fork_threads": {
            kind: {
                "max": int(acc["max"]),
                "mean": round(acc["sum"] / acc["n"], 2),
                "n": int(acc["n"]),
            }
            for kind, acc in sorted(fork_threads.items())
            if acc["n"]
        },
        "caches": {
            name: {
                "hits": acc["hits"],
                "misses": acc["misses"],
                "hit_rate": (
                    round(acc["hits"] / (acc["hits"] + acc["misses"]), 4)
                    if acc["hits"] + acc["misses"] else None
                ),
                **({"writes": acc["writes"]} if acc["writes"] else {}),
                **({"evictions": acc["evictions"]} if acc["evictions"] else {}),
            }
            for name, acc in sorted(caches.items())
        },
        "n_worker_deaths": int(sum(totals["worker_deaths"].values())),
        "figures_with_worker_deaths": figures_with_deaths,
        "log_bytes_total": log_bytes,
        "log_bytes_per_figure_mean": log_bytes / n,
    }
