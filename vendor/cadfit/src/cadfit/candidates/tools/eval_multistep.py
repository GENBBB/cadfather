#!/usr/bin/env python3
"""Multi-step deterministic eval.

For each sample:
  iter 1: make_candidates(prev_code='', gt_stl), render each, keep BEST.
  iter k: make_candidates(prev_code=best_so_far, gt_stl), render each,
          keep best (only commit if IoU strictly increased).
  Stop when no candidate improves IoU or MAX_ITERS reached.

Reports final IoU per sample + aggregate stats.
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


def render_one(code, timeout):
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        p = Path(f.name)
    try:
        ok = render_cadquery_to_stl(code, p, timeout=timeout)
        if not ok or p.stat().st_size == 0:
            return None, None
        pred = trimesh.load(p, process=True)
        if isinstance(pred, trimesh.Scene):
            pred = trimesh.util.concatenate(list(pred.geometry.values()))
        if pred is None or len(pred.faces) == 0:
            return None, None
        return pred, p
    except Exception:
        return None, None


def eval_one(stl_path, detectors, k, pts, render_timeout, max_iters):
    gt = trimesh.load(stl_path, process=True)
    if isinstance(gt, trimesh.Scene):
        gt = trimesh.util.concatenate(list(gt.geometry.values()))
    gt = normalize(gt)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        gt_path = Path(f.name)
    gt.export(gt_path)

    prev_code = ""
    best_iou = 0.0
    last_detectors = []
    try:
        for it in range(max_iters):
            cands = make_candidates(
                prev_code=prev_code, gt_stl_path=str(gt_path),
                num_candidates=k, detectors=detectors,
                render_timeout=render_timeout,
            )
            if not cands:
                break

            seen = set(); deduped = []
            for c in cands:
                if c.code in seen: continue
                seen.add(c.code); deduped.append(c)

            best_local = best_iou
            best_local_code = None
            best_local_det = None
            for c in deduped:
                pred, p = render_one(c.code, render_timeout)
                try:
                    if pred is None:
                        continue
                    pred = normalize(pred)
                    s = iou(gt, pred, pts)
                    if s > best_local:
                        best_local = s; best_local_code = c.code
                        best_local_det = c.detector
                    del pred
                finally:
                    if p is not None:
                        try: p.unlink()
                        except OSError: pass
                gc.collect()

            if best_local <= best_iou + 1e-4:
                break
            best_iou = best_local
            prev_code = best_local_code
            last_detectors.append(best_local_det)
    finally:
        try: gt_path.unlink()
        except OSError: pass
        del gt
        gc.collect()
    return best_iou, last_detectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render-timeout", type=float, default=10.0)
    ap.add_argument("--iou-n", type=int, default=20_000)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    if args.limit:
        stl_paths = stl_paths[:args.limit]

    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(args.iou_n, 3)).astype(np.float64)

    print(f"multi-step eval dir={args.dir} n={len(stl_paths)} k={args.k} max_iters={args.max_iters}", flush=True)
    results = []
    for i, p in enumerate(stl_paths):
        try:
            best, dets = eval_one(p, args.detectors, args.k, pts,
                                  args.render_timeout, args.max_iters)
        except Exception as e:
            best, dets = 0.0, [f"ERR:{type(e).__name__}"]
        results.append((p.name, best, dets))
        dchain = ">".join(d or "x" for d in dets)[:30]
        print(f"[{i+1:2d}/{len(stl_paths)}] {p.name:42s} IoU={best:.4f} steps={len(dets):d} chain={dchain}", flush=True)
        gc.collect()

    ious = np.array([r[1] for r in results])
    print(f"\n=== summary multi-step {Path(args.dir).name} ===", flush=True)
    print(f"  mean    {ious.mean():.4f}", flush=True)
    print(f"  median  {float(np.median(ious)):.4f}", flush=True)
    print(f"  min     {ious.min():.4f}", flush=True)
    print(f"  >0.995  {int((ious > 0.995).sum())}/{len(ious)}", flush=True)
    print(f"  >0.95   {int((ious > 0.95).sum())}/{len(ious)}", flush=True)
    print(f"  >0.90   {int((ious > 0.90).sum())}/{len(ious)}", flush=True)


if __name__ == "__main__":
    main()
