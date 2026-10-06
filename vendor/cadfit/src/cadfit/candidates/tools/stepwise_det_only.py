#!/usr/bin/env python3
"""Stepwise det-only inference.

For each STL in the dataset:
  prev_code = ""
  for iter in 1..max_iters:
      cands = make_candidates(prev_code, gt_stl, num_candidates=K, detectors=ALL,
                              prev_stl_path=prev_stl)
      render each candidate; score by Chamfer Distance against gt mesh.
      winner = lowest-CD candidate
      prev_code = winner.code
      prev_stl  = winner_stl_path

Report per (category, iter) the mean IoU at each step + final IoU.

Mimics the kabisov stepwise pipeline but with NO VLM -- pure detectors.
This is the experiment that says "can detectors alone reconstruct the
multi-step categories of cadeval?"
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
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates import make_candidates
from cadfit.candidates.residual import render_cadquery_to_stl


def _normalize_unit(m):
    c = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-c)
    e = float(np.max(m.extents))
    if e > 1e-9:
        m.apply_scale(2.0 / e)
    return m


def _chamfer(pred, gt, n=4096, seed=42):
    """Symmetric mean L2 CD between two meshes (matches CADFit eq)."""
    if pred is None or gt is None or len(pred.faces) == 0 or len(gt.faces) == 0:
        return float("inf")
    try:
        rng = np.random.default_rng(seed)
        pp, _ = trimesh.sample.sample_surface(pred, n)
        gp, _ = trimesh.sample.sample_surface(gt,   n)
        d_p2g, _ = cKDTree(gp).query(pp, k=1)
        d_g2p, _ = cKDTree(pp).query(gp, k=1)
        return 0.5 * (float(np.mean(d_p2g)) + float(np.mean(d_g2p)))
    except Exception:
        return float("inf")


def _iou(pred, gt, n=40_000, seed=42):
    if pred is None or gt is None or len(pred.faces) == 0 or len(gt.faces) == 0:
        return 0.0
    try:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(-1, 1, size=(n, 3))
        return int(np.sum(pred.contains(pts) & gt.contains(pts))) / max(
            int(np.sum(pred.contains(pts) | gt.contains(pts))), 1)
    except Exception:
        return 0.0


def _evaluate_one(arg):
    stl_path, max_iters, num_candidates, detectors = arg
    work_dir = Path(tempfile.mkdtemp(prefix="sw_"))
    gt = trimesh.load(stl_path, process=True)
    if isinstance(gt, trimesh.Scene):
        gt = trimesh.util.concatenate(list(gt.geometry.values()))
    _normalize_unit(gt)

    iter_iou = []
    iter_cd = []
    prev_code = ""
    prev_stl = None

    for it in range(1, max_iters + 1):
        try:
            cands = make_candidates(
                prev_code=prev_code,
                gt_stl_path=str(stl_path),
                num_candidates=num_candidates,
                detectors=detectors,
                render_timeout=30.0,
                prev_stl_path=prev_stl,
            )
        except Exception:
            cands = []
        if not cands:
            iter_iou.append(iter_iou[-1] if iter_iou else 0.0)
            iter_cd.append(iter_cd[-1] if iter_cd else float("inf"))
            continue
        # Render + score every candidate; pick lowest CD.
        best = None
        for i, c in enumerate(cands):
            out_stl = work_dir / f"iter{it}_c{i}.stl"
            if not render_cadquery_to_stl(c.code, out_stl, timeout=30.0):
                continue
            pred = trimesh.load(out_stl, process=True)
            if isinstance(pred, trimesh.Scene):
                pred = trimesh.util.concatenate(list(pred.geometry.values()))
            _normalize_unit(pred)
            cd = _chamfer(pred, gt)
            if best is None or cd < best["cd"]:
                best = dict(cd=cd, pred=pred, stl=str(out_stl), code=c.code,
                            detector=c.detector, op=c.op)
        if best is None:
            iter_iou.append(iter_iou[-1] if iter_iou else 0.0)
            iter_cd.append(iter_cd[-1] if iter_cd else float("inf"))
            continue
        # Update history.
        iter_cd.append(best["cd"])
        iter_iou.append(_iou(best["pred"], gt))
        prev_code = best["code"]
        prev_stl = best["stl"]

    # Clean up but keep last iter file paths in case caller wants them.
    return dict(
        stl=str(stl_path),
        iter_iou=iter_iou,
        iter_cd=iter_cd,
        final_iou=iter_iou[-1] if iter_iou else 0.0,
        final_cd=iter_cd[-1] if iter_cd else float("inf"),
        n_iters=len(iter_iou),
        work_dir=str(work_dir),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--n_per_cat", type=int, default=3)
    ap.add_argument("--max_iters", type=int, default=4)
    ap.add_argument("--num_candidates", type=int, default=10)
    ap.add_argument("--detectors",
                    default="extrude,revolve,loft,sweep,slice_fit,planar_cluster,fillet_chamfer")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    root = Path(args.dataset_root)

    work = []
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
            work.append((str(s), args.max_iters, args.num_candidates, detectors))

    print(f"work: {len(work)} meshes, workers={args.workers}", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        results = [_evaluate_one(a) for a in work]
    else:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_evaluate_one, work, chunksize=1))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)

    per_cat: dict[str, dict] = {}
    for cat in sorted(set(stl_to_cat.values())):
        rs = [r for r in results if stl_to_cat.get(r["stl"]) == cat]
        if not rs:
            continue
        # Per-iter mean IoU (pad shorter histories with their last value).
        max_n = max(len(r["iter_iou"]) for r in rs)
        iter_mean = []
        for it in range(max_n):
            vals = []
            for r in rs:
                if it < len(r["iter_iou"]):
                    vals.append(r["iter_iou"][it])
                else:
                    vals.append(r["iter_iou"][-1] if r["iter_iou"] else 0.0)
            iter_mean.append(float(np.mean(vals)))
        per_cat[cat] = dict(
            n=len(rs),
            iter_mean_iou=iter_mean,
            final_mean_iou=float(np.mean([r["final_iou"] for r in rs])),
            final_median_iou=float(np.median([r["final_iou"] for r in rs])),
            final_mean_cd=float(np.mean([r["final_cd"] for r in rs])),
        )

    out = dict(
        config=dict(
            n_per_cat=args.n_per_cat,
            max_iters=args.max_iters,
            num_candidates=args.num_candidates,
            detectors=detectors,
        ),
        per_category=per_cat,
        total_min=(time.time() - t0) / 60,
    )
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    print(f"\n=== stepwise det-only summary ===")
    print(f"{'category':18s} {'n':>3s} " +
          " ".join(f"iter{i+1:>2d}" for i in range(args.max_iters)) +
          f"  {'final':>7s}")
    for cat, r in per_cat.items():
        iters = r["iter_mean_iou"][:args.max_iters]
        while len(iters) < args.max_iters:
            iters.append(iters[-1] if iters else 0.0)
        s = " ".join(f"{v:>6.3f}" for v in iters)
        print(f"{cat:18s} {r['n']:>3d} {s}  {r['final_mean_iou']:>7.3f}")
    print(f"\nwritten -> {args.out_json}")


if __name__ == "__main__":
    main()
