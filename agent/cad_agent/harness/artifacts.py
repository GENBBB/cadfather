"""Run directory: layout, per-figure records, summary.

The log is files on disk, not a service: runs go on a cluster without access to
external trackers, and are analyzed with scripts and by eye. Hence simple formats
and predictable paths.

Layout:

    run_<id>/
      config.yaml           the full resolved config; the run is reproducible
      provenance.json       which code, which YAML, which servers
      dataset.json          dataset composition and its signature
      per_figure.json       per-figure table: quality and cost
      summary.json          aggregates over the run and over strata
      figures/<figure_id>/  part working directory: codes, meshes, step log
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=str)


class RunLayout:
    """Run paths. The single place that decides what lives where."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.figures_dir = self.run_dir / "figures"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def figure_dir(self, figure_id: str) -> Path:
        # figure_id looks like "mcb/00001234": the slash becomes a subdirectory
        path = self.figures_dir / figure_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_dataset(self, figures: list[Any], signature: str) -> None:
        save_json(
            {"signature": signature, "num_figures": len(figures), "figures": [f.to_dict() for f in figures]},
            self.run_dir / "dataset.json",
        )

    def save_config(self, config: dict[str, Any]) -> None:
        save_json(config, self.run_dir / "config.json")

    def save_provenance(self, provenance: dict[str, Any]) -> None:
        save_json(provenance, self.run_dir / "provenance.json")

    def save_per_figure(self, records: list[dict[str, Any]]) -> None:
        save_json(records, self.run_dir / "per_figure.json")

    def save_summary(self, summary: dict[str, Any]) -> None:
        save_json(summary, self.run_dir / "summary.json")

    def save_per_figure_csv(self, records: list[dict[str, Any]]) -> None:
        """The same table as CSV, easier to inspect by eye and load into a report.

        Only flat fields are expanded: nested dicts (metrics, cost, technical
        counters) stay in JSON, and what lands here is what is usually sorted by.
        """
        columns = [
            "figure_id", "group", "score", "n_steps", "stop_reason", "error",
            "objective", "objective_value",
            "cd_runtime", "iou", "gms_norm", "stratum", "failure",
            "worker_died", "n_worker_deaths",
            "n_vlm", "n_agent_text", "n_agent_visual", "n_det", "n_opt", "n_exec",
            "wall_sec", "log_bytes",
        ]
        rows = []
        for record in records:
            metrics = record.get("metrics") or {}
            calls = (record.get("cost") or {}).get("calls") or {}
            rows.append({
                "figure_id": record.get("figure_id"),
                "group": record.get("group"),
                "score": record.get("score"),
                "n_steps": record.get("n_steps"),
                "stop_reason": record.get("stop_reason"),
                "error": (record.get("error") or "").splitlines()[0][:200] if record.get("error") else "",
                # What the runtime selected by and where it stopped, next to the
                # final score: without this pair the table cannot answer "in which
                # scale was the decision made", and runs with different objectives
                # look the same.
                "objective": (record.get("runtime_metrics") or {}).get("objective"),
                "objective_value": (record.get("runtime_metrics") or {}).get("objective_value"),
                "cd_runtime": (record.get("runtime_metrics") or {}).get("cd_runtime"),
                "iou": metrics.get("iou"),
                "gms_norm": metrics.get("gms_norm"),
                "stratum": metrics.get("stratum"),
                "failure": metrics.get("failure"),
                # A zero `score` has different explanations: the model or the machine.
                # Without these two columns the table cannot tell one from the other,
                # and the difference decides whether to trust the measurement.
                "worker_died": record.get("worker_died", False),
                "n_worker_deaths": (record.get("tech") or {}).get("n_worker_deaths", 0),
                "n_vlm": calls.get("vlm"),
                "n_agent_text": calls.get("agent_text"),
                "n_agent_visual": calls.get("agent_visual"),
                "n_det": calls.get("det"),
                "n_opt": calls.get("opt"),
                "n_exec": calls.get("exec"),
                "wall_sec": record.get("wall_sec"),
                "log_bytes": record.get("log_bytes"),
            })

        path = self.run_dir / "per_figure.csv"
        with open(path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
