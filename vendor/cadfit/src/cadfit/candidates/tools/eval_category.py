#!/usr/bin/env python3
"""Per-category eval: run the proposer on every STL in a dataset category,
take the BEST det-candidate, render, compute IoU vs the ground-truth mesh.

Reports aggregate IoU stats so we can track ">0.995 on every sample".

Usage
-----
    python eval_category.py --dir DATASET/d-revolve/meshes
    python eval_category.py --dir DATASET/d-revolve/meshes --detectors revolve
    python eval_category.py --dir DATASET/d-revolve/meshes --json out.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# In-process rendering is ~5-10x faster than subprocess and avoids
# OOM from concurrent cadquery imports.  Default it on for the eval.
os.environ.setdefault("DET_RENDER_INPROCESS", "1")

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates.proposer import make_candidates
from cadfit.candidates.residual import render_cadquery_to_stl


def normalize_to_cube(m: trimesh.Trimesh) -> trimesh.Trimesh:
    c = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-c)
    s = float(np.max(m.extents))
    if s > 0:
        m.apply_scale(2.0 / s)
    return m


def iou(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 100_000,
        seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, 3))
    ia = a.contains(pts); ib = b.contains(pts)
    inter = int((ia & ib).sum()); union = int((ia | ib).sum())
    return inter / max(union, 1)


def _render_and_iou(args: tuple) -> tuple:
    """Worker: render one candidate, compute IoU vs gt_path.  Returns
    (idx, detector, iou, error_msg).  Each call loads gt itself --
    cheap, and avoids pickling the trimesh into the worker."""
    idx, code, detector, gt_stl_bytes, render_timeout, iou_n = args
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            gt_path = Path(f.name)
        gt_path.write_bytes(gt_stl_bytes)
        gt = trimesh.load(gt_path, process=True)
        if isinstance(gt, trimesh.Scene):
            gt = trimesh.util.concatenate(list(gt.geometry.values()))
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            p = Path(f.name)
        try:
            ok = render_cadquery_to_stl(code, p, timeout=render_timeout)
            if not ok or p.stat().st_size == 0:
                return (idx, detector, 0.0, "empty render")
            pred = trimesh.load(p, process=True)
            if isinstance(pred, trimesh.Scene):
                pred = trimesh.util.concatenate(list(pred.geometry.values()))
            if pred is None or len(pred.faces) == 0:
                return (idx, detector, 0.0, "no faces")
            normalize_to_cube(pred)
            score = iou(gt, pred, n=iou_n)
            return (idx, detector, float(score), "")
        finally:
            try: p.unlink()
            except OSError: pass
            try: gt_path.unlink()
            except OSError: pass
    except Exception as e:
        return (idx, detector, 0.0, str(e)[:80])


def eval_one(stl_path: Path, detectors: list[str], k: int = 8,
             pool: Optional["ProcessPoolExecutor"] = None,
             render_timeout: float = 15.0,
             iou_n: int = 30_000) -> dict:
    gt = trimesh.load(stl_path, process=True)
    if isinstance(gt, trimesh.Scene):
        gt = trimesh.util.concatenate(list(gt.geometry.values()))
    gt = normalize_to_cube(gt)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        gt_path = Path(f.name)
    try:
        gt.export(gt_path)
        candidates = make_candidates(
            prev_code="",
            gt_stl_path=str(gt_path),
            num_candidates=k,
            detectors=detectors,
            render_timeout=render_timeout,
        )
        gt_bytes = gt_path.read_bytes()
    finally:
        try: gt_path.unlink()
        except OSError: pass

    if not candidates:
        return {"name": stl_path.name, "n_candidates": 0, "best_iou": 0.0,
                "best_detector": None}

    # Dedupe candidates by exact code (multiple depth_frac variants etc.
    # produce many duplicates after rounding).
    seen = set()
    deduped = []
    for c in candidates:
        if c.code in seen: continue
        seen.add(c.code); deduped.append(c)

    tasks = [(i, c.code, c.detector, gt_bytes, render_timeout, iou_n)
             for i, c in enumerate(deduped)]

    best_iou = 0.0
    best_det = None
    best_idx = -1
    if pool is None:
        for t in tasks:
            i, det, score, _ = _render_and_iou(t)
            if score > best_iou:
                best_iou = score; best_det = det; best_idx = i
    else:
        futures = [pool.submit(_render_and_iou, t) for t in tasks]
        for f in as_completed(futures):
            i, det, score, _ = f.result()
            if score > best_iou:
                best_iou = score; best_det = det; best_idx = i

    return {"name": stl_path.name, "n_candidates": len(deduped),
            "best_iou": float(best_iou), "best_detector": best_det,
            "best_idx": best_idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel render workers per sample.")
    ap.add_argument("--render-timeout", type=float, default=15.0)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    if args.limit:
        stl_paths = stl_paths[:args.limit]
    if not stl_paths:
        print(f"no .stl files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(stl_paths)} samples in {args.dir}", flush=True)
    print(f"detectors: {args.detectors or 'ALL'}, k={args.k} workers={args.workers}", flush=True)
    results = []
    t0 = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for i, p in enumerate(stl_paths):
            r = eval_one(p, args.detectors, args.k, pool=pool,
                         render_timeout=args.render_timeout)
            results.append(r)
            det_str = (r['best_detector'] or 'NONE')[:14]
            print(f"[{i+1}/{len(stl_paths)}] {p.name:36s} IoU={r['best_iou']:.4f} "
                  f"det={det_str:14s} k={r['n_candidates']:2d}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    ious = np.array([r["best_iou"] for r in results])
    print(f"\n=== summary {Path(args.dir).parent.name} ({time.time()-t0:.1f}s) ===")
    print(f"  mean    IoU: {ious.mean():.4f}")
    print(f"  median  IoU: {np.median(ious):.4f}")
    print(f"  min     IoU: {ious.min():.4f}")
    print(f"  max     IoU: {ious.max():.4f}")
    print(f"  > 0.995 :    {int((ious > 0.995).sum())}/{len(ious)}")
    print(f"  > 0.95  :    {int((ious > 0.95).sum())}/{len(ious)}")
    print(f"  > 0.90  :    {int((ious > 0.90).sum())}/{len(ious)}")
    by_det = {}
    for r in results:
        d = r["best_detector"] or "NONE"
        by_det.setdefault(d, []).append(r["best_iou"])
    print("  by detector:")
    for d, lst in sorted(by_det.items()):
        print(f"    {d:18s} n={len(lst):3d} mean={np.mean(lst):.4f} min={np.min(lst):.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
