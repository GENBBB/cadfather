#!/usr/bin/env python3
"""For a single category, run the multi-step eval, sort by IoU, save
the K BEST and K WORST samples as paired (gt, pred) .stl files for
visual inspection.
"""
import argparse
import gc
import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("DET_RENDER_INPROCESS", "1")

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_HERE.parent))

from eval_ms import (
    _DETECTORS, candidates_for_iter, combine, voxel_iou, normalize_pair,
    _alarm, SampleTimeout,
)


def eval_with_pred(stl_path, pts, k, max_iters, sample_timeout, eps):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(sample_timeout))
    best_iou = 0.0
    chain = []
    running = None
    try:
        gt = trimesh.load(stl_path, process=True)
        if isinstance(gt, trimesh.Scene):
            gt = trimesh.util.concatenate(list(gt.geometry.values()))
        c = (gt.bounds[0] + gt.bounds[1]) / 2.0
        gt.apply_translation(-c)
        gt.apply_scale(2.0 / float(np.max(gt.extents)))

        for it in range(max_iters):
            if best_iou >= 0.99:
                break
            cands = candidates_for_iter(gt, running, _DETECTORS, k)
            if not cands:
                break
            best_local = best_iou
            best_local_pred = None
            best_local_name = None
            for name, score, m, op in cands:
                cand_pred = m.copy()
                if op == "init" and running is None:
                    cand_pred = normalize_pair(cand_pred, gt)
                merged = combine(running, cand_pred, op)
                if merged is None or len(merged.faces) == 0:
                    continue
                try:
                    s = voxel_iou(gt, merged, pts)
                except Exception:
                    continue
                if s > best_local + eps:
                    best_local = s
                    best_local_pred = merged
                    best_local_name = (name, op)
            if best_local_pred is None or best_local <= best_iou + eps:
                break
            best_iou = best_local
            running = best_local_pred
            chain.append(best_local_name)
    except SampleTimeout:
        if not chain:
            chain = [("TIMEOUT", "init")]
    finally:
        signal.alarm(0)
    return best_iou, chain, running, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--n-worst", type=int, default=5)
    ap.add_argument("--sample-timeout", type=int, default=60)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(15_000, 3)).astype(np.float64)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"running {len(stl_paths)} samples...", flush=True)
    results = []
    for i, p in enumerate(stl_paths):
        try:
            iou, chain, pred, gt = eval_with_pred(
                p, pts, args.k, args.max_iters, args.sample_timeout, 0.001)
        except Exception as e:
            print(f"[{i+1}] ERR: {e}", flush=True)
            continue
        results.append((p, iou, chain, pred, gt))
        print(f"[{i+1:2d}/{len(stl_paths)}] {p.name[:40]:40} IoU={iou:.4f} steps={len(chain)}",
              flush=True)
        gc.collect()

    results.sort(key=lambda r: r[1])  # ascending IoU
    worst = results[:args.n_worst]
    best = results[-args.n_best:][::-1]

    print(f"\n=== BEST {args.n_best} ===")
    for r in best:
        p, iou, chain, pred, gt = r
        gt_out = out_dir / f"BEST_iou{iou:.3f}_{p.stem}_gt.stl"
        pred_out = out_dir / f"BEST_iou{iou:.3f}_{p.stem}_pred.stl"
        gt.export(gt_out)
        if pred is not None: pred.export(pred_out)
        ch = " > ".join(f"{n}/{op[:3]}" for n, op in chain)
        print(f"  IoU={iou:.4f} {p.name:36}  chain: {ch}")
    print(f"\n=== WORST {args.n_worst} ===")
    for r in worst:
        p, iou, chain, pred, gt = r
        gt_out = out_dir / f"WORST_iou{iou:.3f}_{p.stem}_gt.stl"
        pred_out = out_dir / f"WORST_iou{iou:.3f}_{p.stem}_pred.stl"
        gt.export(gt_out)
        if pred is not None: pred.export(pred_out)
        ch = " > ".join(f"{n}/{op[:3]}" for n, op in chain)
        print(f"  IoU={iou:.4f} {p.name:36}  chain: {ch}")
    print(f"\nWrote pairs to: {out_dir}")


if __name__ == "__main__":
    main()
