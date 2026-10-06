#!/usr/bin/env python3
"""Memory-bounded per-category eval — runs in-process, GC between
samples, 20k IoU points (instead of 100k) to keep RSS low.

Usage:
    python eval_quick.py CATEGORY_DIR [--limit N] [--detectors d1 d2 ...]
"""
import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DET_RENDER_INPROCESS", "1")

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates.proposer import make_candidates
from cadfit.candidates.residual import render_cadquery_to_stl


def normalize(m):
    c = (m.bounds[0] + m.bounds[1]) / 2
    m.apply_translation(-c)
    m.apply_scale(2.0 / float(np.max(m.extents)))
    return m


def iou(a, b, pts):
    ia = a.contains(pts); ib = b.contains(pts)
    inter = int((ia & ib).sum()); union = int((ia | ib).sum())
    return inter / max(union, 1)


def eval_one(stl_path, detectors, k, pts, render_timeout):
    gt = trimesh.load(stl_path, process=True)
    if isinstance(gt, trimesh.Scene):
        gt = trimesh.util.concatenate(list(gt.geometry.values()))
    gt = normalize(gt)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        gt_path = Path(f.name)
    gt.export(gt_path)
    try:
        cands = make_candidates(prev_code="", gt_stl_path=str(gt_path),
                                num_candidates=k, detectors=detectors,
                                render_timeout=render_timeout)
    finally:
        try: gt_path.unlink()
        except OSError: pass

    if not cands:
        return 0.0, None, 0

    seen = set(); deduped = []
    for c in cands:
        if c.code in seen: continue
        seen.add(c.code); deduped.append(c)

    best = 0.0; best_det = None
    for c in deduped:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            pp = Path(f.name)
        try:
            ok = render_cadquery_to_stl(c.code, pp, timeout=render_timeout)
            if not ok or pp.stat().st_size == 0:
                continue
            pred = trimesh.load(pp, process=True)
            if isinstance(pred, trimesh.Scene):
                pred = trimesh.util.concatenate(list(pred.geometry.values()))
            if pred is None or len(pred.faces) == 0:
                continue
            normalize(pred)
            s = iou(gt, pred, pts)
            if s > best:
                best = s; best_det = c.detector
            del pred
        except Exception:
            pass
        finally:
            try: pp.unlink()
            except OSError: pass
    del gt
    gc.collect()
    return best, best_det, len(deduped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render-timeout", type=float, default=8.0)
    ap.add_argument("--iou-n", type=int, default=20_000)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    if args.limit:
        stl_paths = stl_paths[:args.limit]

    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(args.iou_n, 3)).astype(np.float64)

    print(f"eval dir={args.dir} n={len(stl_paths)} k={args.k} dets={args.detectors or 'ALL'}", flush=True)
    results = []
    for i, p in enumerate(stl_paths):
        try:
            best, det, n = eval_one(p, args.detectors, args.k, pts, args.render_timeout)
        except Exception as e:
            best, det, n = 0.0, f"ERR:{type(e).__name__}", 0
        results.append((p.name, best, det, n))
        print(f"[{i+1:2d}/{len(stl_paths)}] {p.name:42s} IoU={best:.4f} "
              f"det={(det or 'NONE')[:14]:14s} n_cand={n}", flush=True)
        gc.collect()

    ious = np.array([r[1] for r in results])
    print(f"\n=== summary {Path(args.dir).name} ===", flush=True)
    print(f"  mean    {ious.mean():.4f}", flush=True)
    print(f"  median  {float(np.median(ious)):.4f}", flush=True)
    print(f"  min     {ious.min():.4f}", flush=True)
    print(f"  >0.995  {int((ious > 0.995).sum())}/{len(ious)}", flush=True)
    print(f"  >0.95   {int((ious > 0.95).sum())}/{len(ious)}", flush=True)
    print(f"  >0.90   {int((ious > 0.90).sum())}/{len(ious)}", flush=True)
    by_det = {}
    for _, s, d, _ in results:
        d = d or "NONE"
        by_det.setdefault(d, []).append(s)
    for d, lst in sorted(by_det.items()):
        print(f"  {d:18s} n={len(lst):3d} mean={np.mean(lst):.4f} min={np.min(lst):.4f}", flush=True)


if __name__ == "__main__":
    main()
