#!/usr/bin/env python3
"""Single-step det-only evaluation on the cadeval category dataset.

For every STL in ``<dataset>/<category>/meshes/*.stl``:
  1. Run ``make_candidates(prev_code="", gt_stl=stl, num_candidates=K,
     detectors=ALL)``.
  2. Render each candidate, sample-based volumetric IoU vs the GT.
  3. Pick the BEST candidate by IoU.
  4. Record per-mesh: best_iou, winning_detector, mean of IoU across
     candidates, n_candidates_emitted, n_candidates_rendered.

Aggregates per category (mean IoU, win rate where IoU > 0.5,
detector breakdown), writes a JSON summary.

Modifier detectors (fillet_chamfer) are skipped at iter 1 since there
is no ``r`` to modify -- this is a known limitation; we still emit the
residual-side detectors and report fillet/chamfer categories as
"baseline" IoU (the residual reconstruction without the modifier).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates import make_candidates
from cadfit.candidates.residual import render_cadquery_to_stl


def _normalize_unit(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = float(np.max(mesh.extents))
    if extent > 1e-9:
        mesh.apply_scale(2.0 / extent)
    return mesh


def _iou(a: trimesh.Trimesh, b: trimesh.Trimesh, n_pts: int = 100_000,
         seed: int = 42) -> float:
    if a is None or b is None or len(a.faces) == 0 or len(b.faces) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, size=(n_pts, 3))
    try:
        in_a = a.contains(pts)
        in_b = b.contains(pts)
    except Exception:
        return 0.0
    inter = int((in_a & in_b).sum())
    union = int((in_a | in_b).sum())
    return inter / union if union > 0 else 0.0


def _evaluate_one(arg):
    stl_path, num_candidates, detectors, render_timeout = arg
    t_total = time.time()
    try:
        cands = make_candidates(
            prev_code="",
            gt_stl_path=str(stl_path),
            num_candidates=num_candidates,
            detectors=detectors,
            render_timeout=render_timeout,
        )
    except Exception as e:
        return dict(stl=str(stl_path), error=f"make_candidates: {e}",
                    best_iou=0.0, n_emitted=0, n_rendered=0,
                    elapsed_s=time.time() - t_total)

    if not cands:
        return dict(stl=str(stl_path), error="no candidates",
                    best_iou=0.0, n_emitted=0, n_rendered=0,
                    elapsed_s=time.time() - t_total)

    gt = trimesh.load(stl_path, process=True)
    if isinstance(gt, trimesh.Scene):
        gt = trimesh.util.concatenate(list(gt.geometry.values()))
    _normalize_unit(gt)

    best_iou = 0.0
    best_detector = ""
    best_idx = -1
    per_det_iou: dict[str, float] = {}
    n_rendered = 0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(cands):
            stl_out = Path(td) / f"c{i}.stl"
            if not render_cadquery_to_stl(c.code, stl_out,
                                          timeout=render_timeout):
                continue
            n_rendered += 1
            pred = trimesh.load(stl_out, process=True)
            if isinstance(pred, trimesh.Scene):
                pred = trimesh.util.concatenate(list(pred.geometry.values()))
            _normalize_unit(pred)
            iou = _iou(pred, gt, n_pts=50_000)
            cur = per_det_iou.get(c.detector, -1.0)
            if iou > cur:
                per_det_iou[c.detector] = iou
            if iou > best_iou:
                best_iou = iou
                best_detector = c.detector
                best_idx = i

    return dict(
        stl=str(stl_path),
        best_iou=float(best_iou),
        best_detector=best_detector,
        best_idx=best_idx,
        per_detector_best_iou={k: float(v) for k, v in per_det_iou.items()},
        n_emitted=len(cands),
        n_rendered=n_rendered,
        elapsed_s=time.time() - t_total,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="path containing <category>/meshes/*.stl subdirs")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--num_candidates", type=int, default=20)
    ap.add_argument("--detectors", default=("extrude,revolve,loft,sweep,"
                                            "slice_fit,planar_cluster"))
    ap.add_argument("--render_timeout", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit_per_category", type=int, default=0,
                    help="cap N meshes per category (0=all)")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    work = []
    cat_to_stl: dict[str, list[str]] = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        m_dir = cat_dir / "meshes"
        if not m_dir.exists():
            continue
        stls = sorted(m_dir.glob("*.stl"))
        if args.limit_per_category:
            stls = stls[:args.limit_per_category]
        cat_to_stl[cat_dir.name] = [str(s) for s in stls]
        for s in stls:
            work.append((str(s), args.num_candidates, detectors,
                         args.render_timeout))

    print(f"evaluating {len(work)} meshes across {len(cat_to_stl)} categories",
          flush=True)
    t0 = time.time()
    if args.workers <= 1:
        results = [_evaluate_one(a) for a in work]
    else:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_evaluate_one, work,
                                               chunksize=2))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)

    by_stl = {r["stl"]: r for r in results}

    per_cat = {}
    for cat, stl_list in cat_to_stl.items():
        rs = [by_stl.get(s) for s in stl_list if by_stl.get(s)]
        if not rs:
            continue
        ious = [r["best_iou"] for r in rs]
        det_wins: dict[str, int] = defaultdict(int)
        for r in rs:
            det_wins[r.get("best_detector", "")] += 1
        per_cat[cat] = dict(
            n=len(rs),
            mean_iou=float(np.mean(ious)),
            median_iou=float(np.median(ious)),
            iou_ge_0_5=float(np.mean([x >= 0.5 for x in ious])),
            iou_ge_0_8=float(np.mean([x >= 0.8 for x in ious])),
            detector_wins=dict(det_wins),
            mean_n_emitted=float(np.mean([r["n_emitted"] for r in rs])),
            mean_n_rendered=float(np.mean([r["n_rendered"] for r in rs])),
            mean_elapsed_s=float(np.mean([r["elapsed_s"] for r in rs])),
        )

    out = dict(
        config=dict(
            num_candidates=args.num_candidates,
            detectors=detectors,
            render_timeout=args.render_timeout,
        ),
        per_category=per_cat,
        total_elapsed_min=(time.time() - t0) / 60,
    )
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print("\n=== per-category summary ===")
    print(f"{'category':22s} {'n':>4s} {'mean_iou':>10s} {'>=0.5':>8s} "
          f"{'>=0.8':>8s} {'top_detector':>18s}")
    for cat, r in per_cat.items():
        top = max(r["detector_wins"].items(), key=lambda kv: kv[1])[0]
        print(f"{cat:22s} {r['n']:>4d} {r['mean_iou']:>10.3f} "
              f"{r['iou_ge_0_5']:>8.2f} {r['iou_ge_0_8']:>8.2f} {top:>18s}")
    print(f"\nwritten -> {args.out_json}")


if __name__ == "__main__":
    main()
