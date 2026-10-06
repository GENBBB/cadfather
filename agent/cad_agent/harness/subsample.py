"""Fixed subsample: dataset catalog, strata, manifest.

Why a separate layer. A reference measurement is meaningful only if the set it
was taken on is reproducible byte for byte: the same composition, the same
order, the same stratum proportions. Otherwise a harness gain cannot be told
apart from a difference in the set.

Strata are by the **watertight status of the GT**, and this is not a formality:
IoU is undefined on a non-watertight GT, so the second stratum rests on GMS
alone. Measuring them together and averaging mixes quantities of different
nature, so aggregates are computed per stratum as well.

Workflow: `build_catalog` walks the whole set once (expensive -- every mesh is
loaded and checked for watertightness, hence in parallel), `cross_split` then
cuts the samples, and `write_manifest` stores the result in the repository. A
run starts from the manifest and checks the signature: if the set on disk
changed, it shows immediately and not in the numbers.

**Set roles.** A large pool set is cut into two disjoint samples: `evo` (the
working sample) and `control` (protection against overfitting to particular
parts). A separate held-out `test` set has its own manifest; it is measured once
and never enters selection from the pool.

**Why a control sample rather than rotating the set between generations.** A
paired gate compares the length of the per-part score vector, not the set
signature: with a constant-size rotation it would silently pair different
parts. A control sample does not touch the gate and shows a gap at the moment
it appears.

**Selection axes form a cross**: watertight stratum x scouting-outcome bin x
cost bin. The watertight stratum alone is not enough: it says nothing about
whether a part has headroom for improvement or how much it costs, and a
subsample that drifted along either axis skews the estimates.

What the selection deliberately does **not** do: "take the hardest" (selection
by noise: a part with an unlucky draw regresses to the mean on the next run)
and "drop the solved ones" (breaks both the IR and representativeness).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Version 2: samples are named by role (`evo`/`control`/`test`) and the strata
# became a cross. There is no version 1 in circulation.
MANIFEST_VERSION = 2
STRATUM_WATERTIGHT = "watertight_gt"
STRATUM_NON_WATERTIGHT = "non_watertight_gt"

# Sample names follow the role, not ML custom. `val` was dropped: it fit both
# the working and the control sample equally, while their roles differ, and
# confusing a name with a role is costly.
SPLIT_EVO = "evo"
SPLIT_CONTROL = "control"
SPLIT_TEST = "test"
SPLITS = (SPLIT_EVO, SPLIT_CONTROL, SPLIT_TEST)

# Cell label when the axis value is absent. A label of its own rather than
# dumping into a neighbouring cell: "scouting never saw this part" and "scouting
# saw it and got zero" are different events, and one field for two events gives
# a plausible report with a wrong diagnosis.
BIN_MISSING = "no_data"
BIN_REFUSED = "refused"


@dataclass
class CatalogEntry:
    """One part of the set: path, stratum and proxy difficulty features."""

    figure_id: str
    gt_mesh_path: str
    gt_watertight: bool
    features: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def stratum(self) -> str:
        return STRATUM_WATERTIGHT if self.gt_watertight else STRATUM_NON_WATERTIGHT

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stratum"] = self.stratum
        return data


def scan_entry(mesh_path: str) -> dict[str, Any]:
    """Compute the status and features of one part. Called in a separate process."""
    import trimesh

    from cad_agent.capabilities import difficulty as difficulty_mod
    from cad_agent.capabilities import metrics as metrics_mod

    path = Path(mesh_path)
    try:
        mesh = trimesh.load_mesh(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        watertight = metrics_mod.is_watertight(mesh)
        features = difficulty_mod.shape_features(path)
        return CatalogEntry(
            figure_id=path.stem,
            gt_mesh_path=str(path),
            gt_watertight=bool(watertight),
            features=features,
        ).to_dict()
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return CatalogEntry(
            figure_id=path.stem,
            gt_mesh_path=str(path),
            gt_watertight=False,
            error=repr(exc),
        ).to_dict()


def build_catalog(
    source_dir: str | Path,
    n_workers: int = 16,
    limit: int | None = None,
) -> list[CatalogEntry]:
    """Walk the set and compute the stratum and features of each part.

    The result order is lexicographic by file name, regardless of the order in
    which processes finished: the set signature must be reproducible.
    """
    # `resolve()` rather than `Path(...)`: the key for matching against scouting
    # outcomes is a path, and a run writes it absolute. A relative `--source`
    # would give a catalog with relative keys, zero matches and a guard stop
    # only AFTER walking the whole set. It also makes the set signature
    # independent of the directory the build was launched from.
    source_dir = Path(source_dir).resolve()
    mesh_paths = sorted(source_dir.rglob("*.stl"))
    if limit is not None:
        mesh_paths = mesh_paths[:limit]
    if not mesh_paths:
        raise ValueError(f"No .stl files in the directory: {source_dir}")

    logger.info("Cataloguing the set: %d parts, %d workers", len(mesh_paths), n_workers)
    n_workers = max(1, min(n_workers, len(mesh_paths)))

    if n_workers == 1:
        raw = [scan_entry(str(path)) for path in mesh_paths]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            raw = list(pool.map(scan_entry, [str(path) for path in mesh_paths], chunksize=8))

    entries = [
        CatalogEntry(
            figure_id=item["figure_id"],
            gt_mesh_path=item["gt_mesh_path"],
            gt_watertight=item["gt_watertight"],
            features=item.get("features") or {},
            error=item.get("error"),
        )
        for item in raw
    ]
    entries.sort(key=lambda entry: entry.gt_mesh_path)
    return entries


def stratum_proportions(entries: Iterable[CatalogEntry]) -> dict[str, float]:
    entries = list(entries)
    total = max(len(entries), 1)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
    return {stratum: count / total for stratum, count in sorted(counts.items())}


@dataclass
class Outcome:
    """Outcome of one part in a scouting run.

    Scouting is a run of a fast branch over the whole pool; exactly two things
    are taken from it: how the part ended (`score`) and what it cost (`calls`).
    The rest (`wall_sec`, `n_steps`) does not enter selection and serves as a
    representativeness check: a feature used for selection is representative by
    construction and checks nothing.
    """

    figure_id: str
    gt_mesh_path: str
    score: float | None = None
    calls: int | None = None
    wall_sec: float | None = None
    n_steps: int | None = None


def _match_key(path: str | Path) -> str:
    """Key for matching catalog and outcomes: an absolute normalized path.

    Used on both sides. `resolve()` does not fail on a nonexistent path, so it
    can be called on paths from a foreign `per_figure.json` that may not exist
    on this machine.
    """
    return str(Path(path).resolve())


def load_outcomes(per_figure_path: str | Path) -> dict[str, Outcome]:
    """Scouting outcomes from a run's `per_figure.json`. The key is the GT path.

    The key is the path, not `figure_id`: in the set catalog the id is the file
    name (`00000022`), while in a run it is `group/name` (`mcb/00000022`).
    Matching by id would silently fail on the whole set, every part would fall
    into the "no data" cell, and the cross would degenerate into the watertight
    stratum alone, without a single error in the log. So matching is by path,
    and the number of records that found no part is counted and printed
    (`match_stats`).
    """
    with open(Path(per_figure_path), "r", encoding="utf-8") as stream:
        records = json.load(stream)
    if isinstance(records, dict):
        records = records.get("per_figure") or []

    outcomes: dict[str, Outcome] = {}
    for record in records:
        path = _match_key(record["gt_mesh_path"]) if record.get("gt_mesh_path") else ""
        cost = record.get("cost") or {}
        outcomes[path] = Outcome(
            figure_id=str(record.get("figure_id") or ""),
            gt_mesh_path=path,
            score=None if record.get("score") is None else float(record["score"]),
            calls=None if cost.get("total") is None else int(cost["total"]),
            wall_sec=None if record.get("wall_sec") is None else float(record["wall_sec"]),
            n_steps=None if record.get("n_steps") is None else int(record["n_steps"]),
        )
    return outcomes


def match_outcomes(
    entries: list[CatalogEntry],
    outcomes: dict[str, Outcome],
) -> tuple[dict[str, Outcome], dict[str, Any]]:
    """Match outcomes to catalog parts. Returns (by path, statistics).

    The statistics are not decoration: they are the only thing separating
    "scouting covered the pool" from "the keys did not match". The caller checks
    the trust threshold: continuing quietly on a tiny match share is not allowed.
    """
    by_path = {_match_key(entry.gt_mesh_path): entry for entry in entries}
    matched = {path: outcome for path, outcome in outcomes.items() if path in by_path}
    stats = {
        "n_catalog": len(entries),
        "n_outcomes": len(outcomes),
        "n_matched": len(matched),
        "n_catalog_without_outcome": len(entries) - len(matched),
        "n_outcomes_without_figure": len(outcomes) - len(matched),
        "share_matched": (len(matched) / len(entries)) if entries else 0.0,
        # Example keys from both sides. Without them a mismatch message says
        # WHAT broke but not HOW, and it has to be debugged by hand after
        # walking the whole set.
        "example_catalog_key": next(iter(sorted(by_path)), None),
        "example_outcome_key": next(iter(sorted(outcomes)), None),
    }
    return matched, stats


def quantile_edges(values: list[float], n_bins: int) -> list[float]:
    """Bin edges by quantiles -- `n_bins - 1` of them.

    Quantiles rather than fixed thresholds: the outcome scale depends on the
    run objective (`iou`, `gms`, `hmean`), and any hard-coded threshold would
    have to be moved when the objective changes -- while a threshold living in
    the objective's scale lies when moved. Quantiles move by themselves.
    """
    if n_bins < 2 or not values:
        return []
    ordered = sorted(values)
    edges: list[float] = []
    for index in range(1, n_bins):
        position = index * len(ordered) / n_bins
        low = ordered[min(int(position), len(ordered) - 1)]
        edges.append(float(low))
    return edges


def _bin_of(value: float | None, edges: list[float], names: list[str]) -> str:
    if value is None:
        return BIN_MISSING
    index = 0
    while index < len(edges) and value >= edges[index]:
        index += 1
    return names[min(index, len(names) - 1)]


def cell_axes(
    entries: list[CatalogEntry],
    outcomes: dict[str, Outcome] | None = None,
    n_bins: int = 3,
) -> tuple[dict[str, str], dict[str, Any]]:
    """The cross cell of each part: stratum x outcome bin x cost bin.

    Without scouting outcomes the cross shrinks to one axis, the stratum. This
    is a legitimate mode (the catalog can be built before any runs), but it is
    **recorded** in the axes description: a subsample selected without outcomes
    does not know whether parts have headroom for improvement, and the manifest
    reader must be able to see that.
    """
    outcomes = outcomes or {}
    bin_names = ["low", "mid", "high"][:n_bins] if n_bins <= 3 else [f"q{i + 1}" for i in range(n_bins)]

    scored = [
        outcome.score for outcome in outcomes.values()
        if outcome.score is not None and outcome.score > 0.0
    ]
    costed = [outcome.calls for outcome in outcomes.values() if outcome.calls is not None]
    score_edges = quantile_edges(scored, n_bins)
    cost_edges = quantile_edges([float(value) for value in costed], n_bins)

    cells: dict[str, str] = {}
    for entry in entries:
        path = _match_key(entry.gt_mesh_path)
        parts = [entry.stratum]
        if outcomes:
            outcome = outcomes.get(path)
            if outcome is None:
                parts.append(f"outcome:{BIN_MISSING}")
                parts.append(f"cost:{BIN_MISSING}")
            else:
                # A refusal is a cell of its own, not the lowest bin: zero means
                # "the part failed entirely", and mixing it with "came out
                # badly" would lose from the subsample the share of refusals
                # that defines the future invalidity threshold.
                if outcome.score is None:
                    parts.append(f"outcome:{BIN_MISSING}")
                elif outcome.score <= 0.0:
                    parts.append(f"outcome:{BIN_REFUSED}")
                else:
                    parts.append(f"outcome:{_bin_of(outcome.score, score_edges, bin_names)}")
                calls = None if outcome.calls is None else float(outcome.calls)
                parts.append(f"cost:{_bin_of(calls, cost_edges, bin_names)}")
        cells[path] = " | ".join(parts)

    axes = {
        "axes": ["watertight stratum"] + (["scouting outcome bin", "cost bin (calls)"] if outcomes else []),
        "n_bins": n_bins,
        "outcome_edges": score_edges,
        "cost_edges": cost_edges,
        "outcomes_given": bool(outcomes),
        "warning": None if outcomes else (
            "scouting outcomes not given: the cross degenerated to one axis (stratum). "
            "The subsample is balanced neither by difficulty nor by cost"
        ),
    }
    return cells, axes


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Distribute `total` over cells proportionally to weights (largest remainders).

    Rounding each cell separately on a cross of a dozen or more cells moves the
    sum away from the requested total by whole units: many round-downs mean
    many missing parts. The largest-remainder method gives a sum of exactly
    `total` and does so deterministically, so the sample signature is
    reproducible.
    """
    if total <= 0 or not weights:
        return {cell: 0 for cell in weights}
    scale = sum(weights.values()) or 1.0
    exact = {cell: total * weight / scale for cell, weight in weights.items()}
    quotas = {cell: int(value) for cell, value in exact.items()}
    remainder = total - sum(quotas.values())
    # Sort by (remainder, name): the name is a tie-break, otherwise dict order
    # would decide who gets the extra part and the signature would drift.
    order = sorted(exact, key=lambda cell: (-(exact[cell] - quotas[cell]), cell))
    for cell in order[:remainder]:
        quotas[cell] += 1
    return quotas


def cross_split(
    entries: list[CatalogEntry],
    sizes: dict[str, int],
    outcomes: dict[str, Outcome] | None = None,
    n_bins: int = 3,
    seed: int = 42,
    skip_broken: bool = True,
) -> tuple[dict[str, list[CatalogEntry]], dict[str, Any]]:
    """Cut disjoint samples along the strata cross.

    `sizes` is the number of parts per sample, e.g. ``{"evo": 100, "control":
    200}``. Allocation order is dict order: the first sample takes its quota
    from a cell, the second from the remainder, so they cannot overlap by
    construction rather than by a later check.

    Returns the samples and a **plan**: per cell, how many are in the pool, how
    many were requested and how many are missing. A shortfall is not swallowed:
    a three-part cell with a quota of five means the subsample is skewed, and
    that must be visible in the manifest rather than discovered while reading
    results.
    """
    usable = [entry for entry in entries if not (skip_broken and entry.error)]
    if not usable:
        raise ValueError("The set is empty after dropping unreadable parts")
    unknown = set(sizes) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown split names: {sorted(unknown)}; expected from {list(SPLITS)}")

    cells, axes = cell_axes(usable, outcomes=outcomes, n_bins=n_bins)
    by_cell: dict[str, list[CatalogEntry]] = {}
    for entry in usable:
        by_cell.setdefault(cells[_match_key(entry.gt_mesh_path)], []).append(entry)

    weights = {cell: float(len(items)) for cell, items in by_cell.items()}
    quotas = {name: allocate(size, weights) for name, size in sizes.items()}

    rng = random.Random(seed)
    splits: dict[str, list[CatalogEntry]] = {name: [] for name in sizes}
    plan: list[dict[str, Any]] = []

    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell], key=lambda entry: entry.gt_mesh_path)
        rng.shuffle(pool)
        cursor = 0
        row: dict[str, Any] = {"cell": cell, "n_pool": len(pool), "share_pool": len(pool) / len(usable)}
        for name in sizes:
            want = quotas[name][cell]
            take = pool[cursor:cursor + want]
            cursor += len(take)
            splits[name].extend(take)
            row[f"n_{name}"] = len(take)
            row[f"shortfall_{name}"] = want - len(take)
        plan.append(row)

    for name in splits:
        splits[name].sort(key=lambda entry: entry.gt_mesh_path)

    shortfalls = {
        name: sum(row[f"shortfall_{name}"] for row in plan) for name in sizes
    }
    for name, missing in shortfalls.items():
        if missing:
            logger.warning(
                "Split %s is short by %d parts: the cross cells ran out of pool",
                name, missing,
            )
    report = {
        "axes": axes,
        "n_usable": len(usable),
        "n_broken": len(entries) - len(usable),
        "n_cells": len(by_cell),
        "requested": dict(sizes),
        "selected": {name: len(items) for name, items in splits.items()},
        "shortfalls": shortfalls,
        "cells": plan,
    }
    return splits, report


# Features used to check representativeness. They do not enter selection --
# that is the whole point: an axis used for selection matches by construction
# and checks nothing. `wall_sec` and `n_steps` come from scouting outcomes.
REPRESENTATIVENESS_FEATURES = (
    "n_faces",
    "n_vertices",
    "extent_longest",
    "aspect_ratio",
    "fill_ratio",
    "surface_area",
    "radial_std",
    "radial_mean",
)
OUTCOME_FEATURES = ("wall_sec", "n_steps")

# Threshold from which a discrepancy counts as noticeable. 0.2 is the usual
# boundary of a "small" standardized difference; it is not a law, but it lets
# the report say "drifted" instead of dumping a table and staying silent.
STD_DIFF_ALARM = 0.2


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    variance = sum((value - mean) ** 2 for value in ordered) / n
    def percentile(share: float) -> float:
        return ordered[min(int(share * n), n - 1)]
    return {
        "n": n,
        "mean": mean,
        "sd": variance ** 0.5,
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
    }


def representativeness(
    pool: list[CatalogEntry],
    sample: list[CatalogEntry],
    outcomes: dict[str, Outcome] | None = None,
) -> list[dict[str, Any]]:
    """Compare the sample with the pool on features that did NOT enter selection.

    A standardized difference rather than a significance test: with thousands
    against a hundred any test finds significance in a trifle, and the question
    is not "is there a difference" but "is it large relative to the spread".
    """
    outcomes = outcomes or {}

    def column(entries: list[CatalogEntry], feature: str) -> list[float]:
        values: list[float] = []
        for entry in entries:
            if feature in OUTCOME_FEATURES:
                outcome = outcomes.get(str(Path(entry.gt_mesh_path)))
                raw = None if outcome is None else getattr(outcome, feature, None)
            else:
                raw = entry.features.get(feature)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                values.append(float(raw))
        return values

    rows: list[dict[str, Any]] = []
    features = list(REPRESENTATIVENESS_FEATURES) + ([f for f in OUTCOME_FEATURES] if outcomes else [])
    for feature in features:
        pool_stats = _stats(column(pool, feature))
        sample_stats = _stats(column(sample, feature))
        std_diff = None
        if pool_stats.get("n") and sample_stats.get("n") and pool_stats.get("sd"):
            std_diff = (sample_stats["mean"] - pool_stats["mean"]) / pool_stats["sd"]
        rows.append({
            "feature": feature,
            "pool": pool_stats,
            "sample": sample_stats,
            "std_diff": std_diff,
            "alarm": bool(std_diff is not None and abs(std_diff) > STD_DIFF_ALARM),
        })
    return rows


def signature(entries: Iterable[CatalogEntry]) -> str:
    """Sample signature: composition and order. The same one `dataset` computes."""
    payload = "\n".join(f"{entry.figure_id}\t{entry.gt_mesh_path}" for entry in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    name: str,
    source_dir: str | Path,
    catalog: list[CatalogEntry],
    splits: dict[str, list[CatalogEntry]],
    seed: int,
    created: str,
    plan: dict[str, Any] | None = None,
    outcomes: dict[str, Outcome] | None = None,
    outcomes_source: str | None = None,
    match_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Subsample manifest -- what is stored in the repository next to the configs.

    Besides composition and signatures it records **how** the sample was
    obtained: cross axes and their edges, the per-cell plan with shortfalls, the
    scouting outcome source and the share of matched records, and the
    representativeness report on features that did not enter selection. Without
    this the manifest answers "which parts" but not "can they be trusted", and
    the latter is what will be asked later.
    """
    usable = [entry for entry in catalog if not entry.error]
    return {
        "version": MANIFEST_VERSION,
        "name": name,
        "created": created,
        "source_dir": str(source_dir),
        "seed": seed,
        "source": {
            "n_total": len(catalog),
            "n_broken": sum(1 for entry in catalog if entry.error),
            "stratum_proportions": stratum_proportions(catalog),
        },
        "selection": plan or {},
        "outcomes": {
            "source": outcomes_source,
            "match": match_stats or {},
        },
        "splits": {
            split_name: {
                "n": len(entries),
                "signature": signature(entries),
                "stratum_proportions": stratum_proportions(entries),
                # Representativeness is computed against the pool, not another
                # sample: the question is whether the sample resembles the set it
                # was drawn from; comparing two samples does not answer it.
                "representativeness": representativeness(usable, entries, outcomes=outcomes),
                "figures": [
                    {
                        "figure_id": entry.figure_id,
                        "gt_mesh_path": entry.gt_mesh_path,
                        "stratum": entry.stratum,
                    }
                    for entry in entries
                ],
            }
            for split_name, entries in splits.items()
        },
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Manifest version {manifest.get('version')}, expected {MANIFEST_VERSION}: {path}"
        )
    return manifest


def write_catalog_csv(catalog: list[CatalogEntry], path: str | Path) -> Path:
    """Dataset catalog as CSV, for choosing stratification thresholds by eye."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_keys = sorted({key for entry in catalog for key in entry.features})
    columns = ["figure_id", "gt_mesh_path", "stratum", "gt_watertight", "error", *feature_keys]

    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in catalog:
            row = {
                "figure_id": entry.figure_id,
                "gt_mesh_path": entry.gt_mesh_path,
                "stratum": entry.stratum,
                "gt_watertight": entry.gt_watertight,
                "error": entry.error or "",
            }
            row.update({key: entry.features.get(key) for key in feature_keys})
            writer.writerow(row)
    return path
