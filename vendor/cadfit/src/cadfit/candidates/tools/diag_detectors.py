#!/usr/bin/env python3
"""Per-detector diagnostic: run each detector INDIVIDUALLY on a small
sample per category and report:
   - n_candidates emitted
   - n_candidates rendered
   - best IoU achieved by THAT detector alone

Lets us answer "why did X win over Y" by seeing each detector's
ceiling on each category.

Usage:
    python eval_det_categories.py is the broad rollup.
    This script is for forensics.
"""
import argparse
import json
import sys
import tempfile
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


def _normalize_unit(mesh):
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = float(np.max(mesh.extents))
    if extent > 1e-9:
        mesh.apply_scale(2.0 / extent)
    return mesh


def _iou(a, b, n_pts=30_000, seed=42):
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


def _eval_one_pair(arg):
    """Worker: (stl_path, detector, num_candidates) -> (cat, det, stl,
    best_iou, n_emitted, n_rendered)."""
    stl_path, det, num_cands = arg
    gt = trimesh.load(stl_path, process=True)
    if isinstance(gt, trimesh.Scene):
        gt = trimesh.util.concatenate(list(gt.geometry.values()))
    _normalize_unit(gt)
    try:
        cands = make_candidates(
            prev_code="",
            gt_stl_path=str(stl_path),
            num_candidates=num_cands,
            detectors=[det],
            render_timeout=30.0,
        )
    except Exception:
        return (stl_path, det, 0.0, 0, 0)
    if not cands:
        return (stl_path, det, 0.0, 0, 0)
    best_iou = 0.0
    n_rendered = 0
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(cands):
            stl_out = Path(td) / f"c{i}.stl"
            if not render_cadquery_to_stl(c.code, stl_out, timeout=30.0):
                continue
            n_rendered += 1
            pred = trimesh.load(stl_out, process=True)
            if isinstance(pred, trimesh.Scene):
                pred = trimesh.util.concatenate(list(pred.geometry.values()))
            _normalize_unit(pred)
            iou = _iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
    return (stl_path, det, float(best_iou), len(cands), n_rendered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--n_per_cat", type=int, default=3)
    ap.add_argument("--num_candidates", type=int, default=20)
    ap.add_argument("--detectors", default="extrude,revolve,loft,sweep,slice_fit,planar_cluster")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    root = Path(args.dataset_root)

    # Build work list: every (mesh, detector) pair.
    work: list = []
    stl_to_cat: dict[str, str] = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        m_dir = cat_dir / "meshes"
        if not m_dir.exists():
            continue
        stls = sorted(m_dir.glob("*.stl"))[: args.n_per_cat]
        for s in stls:
            stl_to_cat[str(s)] = cat_dir.name
            for d in detectors:
                work.append((str(s), d, args.num_candidates))

    print(f"work: {len(work)} (mesh, detector) pairs, workers={args.workers}",
          flush=True)
    t0 = time.time()
    if args.workers <= 1:
        results = [_eval_one_pair(a) for a in work]
    else:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_eval_one_pair, work, chunksize=2))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Per (category, detector) -> list of best-iou floats
    rows: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "ious": [], "n_emitted": [], "n_rendered": []
    }))
    for stl, det, iou, n_emit, n_rnd in results:
        cat = stl_to_cat.get(stl, "?")
        rows[cat][det]["ious"].append(iou)
        rows[cat][det]["n_emitted"].append(n_emit)
        rows[cat][det]["n_rendered"].append(n_rnd)

    # ---- summary table -----------------------------------------------------
    print("\n\n=== per-category x detector  (best IoU mean across n samples) ===")
    hdr = f"{'category':18s}  " + "  ".join(f"{d:>12s}" for d in detectors)
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for cat in sorted(rows.keys()):
        cells = []
        out[cat] = {}
        for det in detectors:
            rec = rows[cat][det]
            mean_iou = float(np.mean(rec["ious"])) if rec["ious"] else 0.0
            mean_emit = float(np.mean(rec["n_emitted"])) if rec["n_emitted"] else 0
            mean_rnd = float(np.mean(rec["n_rendered"])) if rec["n_rendered"] else 0
            cells.append(f"{mean_iou:>7.3f}({int(mean_emit):>2d}/{int(mean_rnd):>2d})")
            out[cat][det] = {
                "mean_iou": mean_iou,
                "mean_emitted": mean_emit,
                "mean_rendered": mean_rnd,
            }
        print(f"{cat:18s}  " + "  ".join(cells))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\nwritten -> {args.out_json}")


if __name__ == "__main__":
    main()
