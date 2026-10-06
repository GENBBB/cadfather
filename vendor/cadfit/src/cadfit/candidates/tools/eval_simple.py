#!/usr/bin/env python3
"""Single-process eval — no subprocess.run capture (which deadlocks
when the child writes too much to stdout/stderr).  Just runs each
sample in-process with explicit gc and writes per-sample IoU to a
JSONL file as we go.  Easy to resume if interrupted.

Usage:
    python eval_simple.py CATEGORY_DIR [--limit N] [--detectors d1 d2]
"""
import argparse
import gc
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

# In-process render is much faster than subprocess; cadquery sometimes
# hangs in OCCT but we use a per-render SIGALRM timeout.
os.environ.setdefault("DET_RENDER_INPROCESS", "1")

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates.proposer import make_candidates
from cadfit.candidates.residual import render_cadquery_to_stl


class SampleTimeout(Exception):
    pass


def _alarm(*_):
    raise SampleTimeout()


def normalize(m):
    c = (m.bounds[0] + m.bounds[1]) / 2
    m.apply_translation(-c)
    m.apply_scale(2.0 / float(np.max(m.extents)))
    return m


def iou(a, b, pts):
    ia = a.contains(pts); ib = b.contains(pts)
    inter = int((ia & ib).sum()); union = int((ia | ib).sum())
    return inter / max(union, 1)


def eval_sample(stl_path, detectors, k, pts, render_timeout, sample_timeout):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(sample_timeout))
    best = 0.0; best_det = None; n_cand = 0
    try:
        gt = trimesh.load(stl_path, process=True)
        if isinstance(gt, trimesh.Scene):
            gt = trimesh.util.concatenate(list(gt.geometry.values()))
        gt = normalize(gt)
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            gt_path = Path(f.name)
        try:
            gt.export(gt_path)
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
        n_cand = len(deduped)

        for c in deduped:
            pred = None
            tmp_path = None
            try:
                # FAST PATH: detector emitted a trimesh approximation
                # directly (no cadquery render needed).  ~1000x faster.
                fast = getattr(c, "mesh", None)
                if fast is not None and getattr(fast, "faces", None) is not None \
                        and len(fast.faces) > 0:
                    pred = fast.copy()
                else:
                    # Fall back to rendering via cadquery (slow).
                    with tempfile.NamedTemporaryFile(suffix=".stl",
                                                    delete=False) as f:
                        tmp_path = Path(f.name)
                    ok = render_cadquery_to_stl(c.code, tmp_path,
                                                timeout=render_timeout)
                    if not ok or tmp_path.stat().st_size == 0:
                        continue
                    pred = trimesh.load(tmp_path, process=True)
                    if isinstance(pred, trimesh.Scene):
                        pred = trimesh.util.concatenate(
                            list(pred.geometry.values()))
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
                if tmp_path is not None:
                    try: tmp_path.unlink()
                    except OSError: pass
        del gt
    except SampleTimeout:
        if best_det is None:
            best_det = "TIMEOUT"
    finally:
        signal.alarm(0)
        gc.collect()
    return best, best_det, n_cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render-timeout", type=float, default=8.0)
    ap.add_argument("--sample-timeout", type=int, default=90)
    ap.add_argument("--iou-n", type=int, default=15_000)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    if args.limit:
        stl_paths = stl_paths[:args.limit]

    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(args.iou_n, 3)).astype(np.float64)

    cat = Path(args.dir).parent.name if Path(args.dir).name == "meshes" else Path(args.dir).name
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    if jsonl_path:
        jsonl_path.write_text("")  # truncate

    print(f"eval_simple cat={cat} n={len(stl_paths)} k={args.k}", flush=True)
    results = []
    t0 = time.time()
    for i, p in enumerate(stl_paths):
        try:
            best, det, nc = eval_sample(p, args.detectors, args.k, pts,
                                        args.render_timeout, args.sample_timeout)
        except Exception as e:
            best, det, nc = 0.0, f"ERR:{type(e).__name__}", 0
        rec = {"name": p.name, "iou": float(best), "detector": det, "n_cand": nc}
        results.append(rec)
        if jsonl_path:
            with jsonl_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        d = (det or "NONE")[:14]
        print(f"[{i+1:2d}/{len(stl_paths)}] {p.name:42s} IoU={best:.4f} det={d:14s} n_cand={nc}",
              flush=True)

    ious = np.array([r["iou"] for r in results])
    elapsed = time.time() - t0
    print(f"\n=== summary {cat} ({elapsed:.0f}s) ===", flush=True)
    print(f"  mean    {ious.mean():.4f}", flush=True)
    print(f"  median  {float(np.median(ious)):.4f}", flush=True)
    print(f"  min     {ious.min():.4f}", flush=True)
    print(f"  >0.995  {int((ious > 0.995).sum())}/{len(ious)}", flush=True)
    print(f"  >0.95   {int((ious > 0.95).sum())}/{len(ious)}", flush=True)
    print(f"  >0.90   {int((ious > 0.90).sum())}/{len(ious)}", flush=True)
    by_det = {}
    for r in results:
        d = r["detector"] or "NONE"
        by_det.setdefault(d, []).append(r["iou"])
    for d, lst in sorted(by_det.items()):
        print(f"  {d:18s} n={len(lst):3d} mean={np.mean(lst):.4f} min={np.min(lst):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
