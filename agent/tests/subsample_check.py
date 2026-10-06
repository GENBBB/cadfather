#!/usr/bin/env python3
"""Check of the subsample builder on a synthetic dataset.

Run: ``python agent/tests/subsample_check.py`` from the ``agent`` directory.

The dataset is built here rather than read from disk: some parts are made
deliberately non-watertight (faces removed from the mesh) so both strata are
non-empty, and the "scouting" outcomes are faked so there are both refusals and
varying quality.

It checks what the subsample exists for:

- scouting outcomes are matched to parts **by path**, and a key mismatch shows
  up in the statistics instead of silently collapsing the cross to one axis;
- the strata cross (watertight x outcome x cost) really yields cells, and a
  refusal lives in its own cell rather than in the lowest quality bin;
- quotas sum to the requested size, the `evo` and `control` splits do not
  overlap, and a shortfall in a cell is counted and reported;
- selection is reproducible: same seed, same signature;
- representativeness is computed over features NOT used for selection;
- a dataset swapped on disk is caught by the signature check.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import trimesh

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from cad_agent.harness import subsample  # noqa: E402
from cad_agent.harness.dataset import dataset_signature, load_figures_from_manifest  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_source(folder: Path, n_watertight: int, n_open: int, n_hole: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for idx in range(n_watertight):
        trimesh.creation.box((10 + idx, 20, 30)).export(folder / f"solid_{idx:03d}.stl")

    for idx in range(n_open):
        mesh = trimesh.creation.box((15, 15 + idx, 25))
        # Remove some faces. `mesh.split()` would fill a single hole by itself
        # (split repairs components) and the part would land in the watertight
        # stratum again; the second stratum needs a mesh that cannot be repaired.
        open_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[6:], process=False)
        open_mesh.export(folder / f"open_{idx:03d}.stl")

    for idx in range(n_hole):
        mesh = trimesh.creation.box((12 + idx, 18, 22))
        # A single hole: `split()` fills it, so the part stays in the watertight stratum.
        holed = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[2:], process=False)
        holed.export(folder / f"hole_{idx:03d}.stl")


def _fake_score(index: int) -> float:
    """Synthetic scouting outcome: some parts are refusals, the rest vary in quality.

    Refusals are required: they have their own cross cell, and without them the
    "refusal did not mix into the lowest bin" check would pass vacuously.
    """
    if index % 11 == 0:
        return 0.0
    return round(0.30 + (index % 7) * 0.10, 2)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="subsample_check_"))
    source = tmp / "source"
    make_source(source, n_watertight=40, n_open=20)

    print("1. Cataloguing the set and strata")
    catalog = subsample.build_catalog(source, n_workers=4)
    check("all parts are read", len(catalog) == 63, f"parts: {len(catalog)}")
    check("catalog order is deterministic",
          [entry.figure_id for entry in catalog] == [e.figure_id for e in subsample.build_catalog(source, n_workers=2)])

    shares = subsample.stratum_proportions(catalog)
    watertight = [entry for entry in catalog if entry.gt_watertight]
    check("both strata are non-empty", 0 < len(watertight) < len(catalog), str(shares))
    check("closed meshes are recognized as watertight",
          all(entry.gt_watertight for entry in catalog if entry.figure_id.startswith("solid")))
    check("holey meshes are recognized as non-watertight",
          all(not entry.gt_watertight for entry in catalog if entry.figure_id.startswith("open")))
    check("complexity features are computed",
          all(entry.features.get("n_faces") for entry in catalog if not entry.error))
    check("the stratum agrees with the difficulty feature",
          all(entry.gt_watertight == entry.features.get("gt_watertight") for entry in catalog if not entry.error))
    # A part with a single hole is not closed "as is". `mesh.split()` repairs it,
    # and the metrics criterion used to put it in the watertight stratum, along
    # with GTs whose open bodies silently dropped out of IoU. The criterion is now
    # "the whole mesh is closed" (`metrics.is_valid_gt`), which the split repair
    # does not fool.
    patched = [entry for entry in catalog if entry.figure_id.startswith("hole")]
    check("a part with one hole did not land in the watertight stratum",
          bool(patched) and not any(entry.gt_watertight for entry in patched),
          str([e.figure_id for e in patched if e.gt_watertight]))
    check("but as is, it is not closed",
          all(entry.features.get("gt_watertight_raw") is False for entry in patched))

    print("2. Scout outcomes are matched by path, not by id")
    # per_figure.json of a real run: the id there is `group/name`, while in the
    # catalog it is just the name. Matching by id would silently fail on the
    # whole dataset.
    per_figure = [
        {
            "figure_id": f"mcb/{entry.figure_id}",
            "gt_mesh_path": entry.gt_mesh_path,
            "score": _fake_score(index),
            "cost": {"calls": {"vlm": index}, "total": 10 + (index % 7) * 5},
            "wall_sec": 20.0 + index,
            "n_steps": 1 + index % 9,
        }
        for index, entry in enumerate(catalog)
        if not entry.error
    ]
    outcomes_path = tmp / "per_figure.json"
    outcomes_path.write_text(json.dumps(per_figure, ensure_ascii=False), encoding="utf-8")

    raw = subsample.load_outcomes(outcomes_path)
    outcomes, match = subsample.match_outcomes(catalog, raw)
    check("outcomes matched by path", match["n_matched"] == len(per_figure),
          f"{match['n_matched']} of {len(per_figure)}")
    check("run id differs from catalog id, and that did not get in the way",
          all(outcome.figure_id.startswith("mcb/") for outcome in outcomes.values()))

    # The same file but with paths from another dataset: the match share must
    # drop to zero AND be visible in the statistics. A silent zero would mean a
    # one-axis cross and a plausible report with the wrong composition.
    alien = {
        str(Path("/other/set") / Path(path).name): outcome
        for path, outcome in raw.items()
    }
    _, alien_match = subsample.match_outcomes(catalog, alien)
    check("foreign paths give zero matches and this is visible",
          alien_match["n_matched"] == 0 and alien_match["n_outcomes_without_figure"] == len(alien),
          str(alien_match))

    # A relative `--source`: the catalog must match outcomes that record absolute
    # paths. This is not hypothetical: the first real selection run failed on it,
    # AFTER walking the whole dataset.
    cwd = os.getcwd()
    try:
        os.chdir(source.parent.parent)
        relative = Path(source.parent.name) / source.name
        check("the relative path really is relative", not relative.is_absolute())
        relative_catalog = subsample.build_catalog(relative, n_workers=2)
        _, relative_match = subsample.match_outcomes(relative_catalog, raw)
    finally:
        os.chdir(cwd)
    check("a catalog built from a relative path matches absolute outcomes",
          relative_match["n_matched"] == len(per_figure),
          f"{relative_match['n_matched']} of {len(per_figure)}")
    check("the catalog stores absolute paths regardless of where it was launched from",
          all(Path(entry.gt_mesh_path).is_absolute() for entry in relative_catalog))
    check("the set signature does not depend on how paths are written",
          subsample.signature(relative_catalog) == subsample.signature(catalog))

    # The guard's message must show the keys on both sides: without them it says
    # WHAT broke but not HOW.
    check("mismatch statistics show the keys of both sides",
          alien_match.get("example_catalog_key") and alien_match.get("example_outcome_key")
          and alien_match["example_catalog_key"] != alien_match["example_outcome_key"],
          f"{alien_match.get('example_catalog_key')} / {alien_match.get('example_outcome_key')}")

    print("3. Stratum cross: stratum x outcome x cost")
    cells, axes = subsample.cell_axes(
        [entry for entry in catalog if not entry.error], outcomes=outcomes, n_bins=3
    )
    check("more cells than strata", len(set(cells.values())) > 2, f"cells: {len(set(cells.values()))}")
    check("axes recorded", len(axes["axes"]) == 3, str(axes["axes"]))
    check("refusals got their own cell",
          any(subsample.BIN_REFUSED in cell for cell in cells.values()),
          str(sorted({cell for cell in cells.values()}))[:120])
    _, axes_bare = subsample.cell_axes([e for e in catalog if not e.error], outcomes=None)
    check("without outcomes the cross degenerates and says so",
          len(axes_bare["axes"]) == 1 and axes_bare["warning"],
          str(axes_bare["warning"]))

    print("4. Quota allocation and slicing of samples")
    check("quotas add up to the requested total", sum(subsample.allocate(100, {c: 1.0 for c in "abcdefghijklmnopqr"}.copy()).values()) == 100)
    splits, plan = subsample.cross_split(
        catalog, sizes={"evo": 12, "control": 18}, outcomes=outcomes, seed=42
    )
    evo, control = splits["evo"], splits["control"]
    check("sizes are respected", len(evo) == 12 and len(control) == 18,
          f"evo={len(evo)}, control={len(control)}, shortfalls {plan['shortfalls']}")
    evo_ids = {entry.figure_id for entry in evo}
    control_ids = {entry.figure_id for entry in control}
    check("samples do not overlap", not (evo_ids & control_ids), str(sorted(evo_ids & control_ids))[:80])
    check("per-cell plan is recorded", len(plan["cells"]) == plan["n_cells"] > 1, str(plan["n_cells"]))
    check("no shortfalls", not any(plan["shortfalls"].values()), str(plan["shortfalls"]))

    evo_shares = subsample.stratum_proportions(evo)
    check("stratum proportions are preserved in evo",
          all(abs(evo_shares.get(name, 0) - share) <= 0.15 for name, share in shares.items()),
          f"{evo_shares} vs {shares}")

    # A shortfall must be counted and reported, not smoothed over: the order exceeds the pool.
    _, tight_plan = subsample.cross_split(
        catalog, sizes={"evo": 500}, outcomes=outcomes, seed=42
    )
    check("shortfall is counted, not hidden", tight_plan["shortfalls"]["evo"] > 0,
          str(tight_plan["shortfalls"]))

    print("5. Reproducibility of selection")
    splits_again, _ = subsample.cross_split(
        catalog, sizes={"evo": 12, "control": 18}, outcomes=outcomes, seed=42
    )
    check("same seed — same evo", subsample.signature(evo) == subsample.signature(splits_again["evo"]))
    check("same seed — same control",
          subsample.signature(control) == subsample.signature(splits_again["control"]))
    splits_other, _ = subsample.cross_split(
        catalog, sizes={"evo": 12, "control": 18}, outcomes=outcomes, seed=7
    )
    check("different seed — different sample",
          subsample.signature(evo) != subsample.signature(splits_other["evo"]))

    print("6. Representativeness is computed on features that were not used in selection")
    rows = subsample.representativeness(
        [entry for entry in catalog if not entry.error], evo, outcomes=outcomes
    )
    features = {row["feature"] for row in rows}
    check("catalog features are in the report", {"n_faces", "fill_ratio", "radial_std"} <= features, str(sorted(features)))
    check("wall_sec and n_steps came from the outcomes", {"wall_sec", "n_steps"} <= features)
    check("the selection axis did not get into the report", "score" not in features and "calls" not in features)
    check("features with data have a standardized difference computed",
          all(row["std_diff"] is not None for row in rows if row["pool"].get("sd")),
          str([row["feature"] for row in rows if row["pool"].get("sd") and row["std_diff"] is None]))

    print("7. Manifest and set loading")
    manifest = subsample.build_manifest(
        name="synthetic", source_dir=source, catalog=catalog, splits=splits,
        seed=42, created="2026-08-13T00:00:00+00:00", plan=plan,
        outcomes=outcomes, outcomes_source=str(outcomes_path), match_stats=match,
    )
    manifest_path = subsample.write_manifest(manifest, tmp / "synthetic.json")
    check("manifest written", manifest_path.exists())
    check("the manifest has splits by role", set(manifest["splits"]) == {"evo", "control"})
    check("manifest version bumped", manifest["version"] == subsample.MANIFEST_VERSION == 2)
    check("the manifest has the stratum shares of the source set", bool(manifest["source"]["stratum_proportions"]))
    check("the manifest records where the outcomes came from",
          manifest["outcomes"]["source"] == str(outcomes_path)
          and manifest["outcomes"]["match"]["n_matched"] == match["n_matched"])
    check("the manifest records the selection plan", manifest["selection"]["n_cells"] == plan["n_cells"])
    check("representativeness is stored next to the sample",
          bool(manifest["splits"]["evo"]["representativeness"]))

    figures = load_figures_from_manifest(manifest_path, split="evo")
    check("the set is loaded from the manifest", len(figures) == len(evo))
    check("order matches the manifest",
          [figure.figure_id for figure in figures] == [entry.figure_id for entry in evo])
    check("the set signature matches the manifest signature",
          dataset_signature(figures) == manifest["splits"]["evo"]["signature"])
    check("the control sample is available separately",
          len(load_figures_from_manifest(manifest_path, split="control")) == len(control))
    try:
        load_figures_from_manifest(manifest_path, split="test")
        missing_test_caught = False
    except ValueError as exc:
        missing_test_caught = "test" in str(exc)
    check("held-out test is absent from the pool manifest, and that is an error, not an empty set",
          missing_test_caught)

    print("8. Held-out test — the whole set as one sample")
    whole = {"test": sorted([e for e in catalog if not e.error], key=lambda e: e.gt_mesh_path)}
    whole_manifest = subsample.build_manifest(
        name="cadenabench_like", source_dir=source, catalog=catalog, splits=whole,
        seed=42, created="2026-08-13T00:00:00+00:00",
    )
    whole_path = subsample.write_manifest(whole_manifest, tmp / "whole.json")
    check("the whole set became one sample",
          whole_manifest["splits"]["test"]["n"] == len(whole["test"]))
    check("signature matches",
          dataset_signature(load_figures_from_manifest(whole_path, split="test"))
          == whole_manifest["splits"]["test"]["signature"])

    print("9. A swapped set on disk is detected")
    victim = Path(evo[0].gt_mesh_path)
    victim.rename(victim.with_name("renamed_" + victim.name))
    try:
        load_figures_from_manifest(manifest_path, split="evo")
        caught = False
    except ValueError as exc:
        caught = "Missing" in str(exc) or "signature" in str(exc).lower()
    check("a missing mesh is noticed before the run", caught)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("The subsample is fine.")


if __name__ == "__main__":
    main()
