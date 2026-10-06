#!/usr/bin/env python3
"""Per-sample-subprocess eval — memory bulletproof.

The main process spawns ONE subprocess per sample.  Each subprocess
loads its sample, runs all detectors, renders candidates in-process,
computes IoU, prints a single result line, exits.  When it exits, the
OS reclaims ALL its memory (trimesh + cadquery + shapely + numpy state).

This sidesteps the slow accumulation of native-extension memory that
in-process eval suffers from after ~5-10 samples.

Usage:
    python eval_iso.py CATEGORY_DIR [--detectors d1 d2 ...] [--limit N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


_PER_SAMPLE_SCRIPT = r"""
import sys, os, json, gc, tempfile
# Subprocess render: each candidate render runs in a fresh child
# process.  Slower per-render (~3s startup) but bulletproof against
# OCCT C++ deadlocks that ignore Python SIGALRM.  In-process render
# hangs forever on some borderline-profile revolves; subprocess timeout
# kills cleanly via SIGKILL.
os.environ['DET_RENDER_INPROCESS'] = '0'
from pathlib import Path
import numpy as np, trimesh

stl_path = sys.argv[1]
detectors = json.loads(sys.argv[2]) or None
k = int(sys.argv[3])
iou_n = int(sys.argv[4])
render_timeout = float(sys.argv[5])
pkg_root = sys.argv[6]
sys.path.insert(0, pkg_root)

from cadfit.candidates.proposer import make_candidates
from cadfit.candidates.residual import render_cadquery_to_stl

m = trimesh.load(stl_path, process=True)
if isinstance(m, trimesh.Scene):
    m = trimesh.util.concatenate(list(m.geometry.values()))
c = (m.bounds[0] + m.bounds[1]) / 2
m.apply_translation(-c)
m.apply_scale(2 / float(np.max(m.extents)))
with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
    gt_path = Path(f.name)
m.export(gt_path)
try:
    cands = make_candidates(prev_code='', gt_stl_path=str(gt_path),
                            num_candidates=k, detectors=detectors,
                            render_timeout=render_timeout)
finally:
    try: gt_path.unlink()
    except OSError: pass

if not cands:
    print(json.dumps({'iou': 0.0, 'detector': None, 'n_cand': 0}))
    sys.exit(0)

seen = set(); deduped = []
for c in cands:
    if c.code in seen: continue
    seen.add(c.code); deduped.append(c)

rng = np.random.default_rng(42)
pts = rng.uniform(-1, 1, size=(iou_n, 3)).astype(np.float64)

best = 0.0; best_det = None
for c in deduped:
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        p = Path(f.name)
    try:
        ok = render_cadquery_to_stl(c.code, p, timeout=render_timeout)
        if not ok or p.stat().st_size == 0:
            continue
        pred = trimesh.load(p, process=True)
        if isinstance(pred, trimesh.Scene):
            pred = trimesh.util.concatenate(list(pred.geometry.values()))
        if pred is None or len(pred.faces) == 0:
            continue
        cc = (pred.bounds[0] + pred.bounds[1]) / 2
        pred.apply_translation(-cc)
        pred.apply_scale(2 / float(np.max(pred.extents)))
        ia = m.contains(pts); ib = pred.contains(pts)
        s = int((ia & ib).sum()) / max(int((ia | ib).sum()), 1)
        if s > best:
            best = s; best_det = c.detector
        del pred
    finally:
        try: p.unlink()
        except OSError: pass

print(json.dumps({'iou': float(best), 'detector': best_det,
                  'n_cand': len(deduped)}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render-timeout", type=float, default=10.0)
    ap.add_argument("--iou-n", type=int, default=20_000)
    ap.add_argument("--per-sample-timeout", type=int, default=120,
                    help="Subprocess timeout per sample (seconds).")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    if args.limit:
        stl_paths = stl_paths[:args.limit]

    pkg_root = str(Path(__file__).resolve().parent.parent.parent)
    detectors_json = json.dumps(args.detectors)
    python_exe = sys.executable

    print(f"eval_iso dir={args.dir} n={len(stl_paths)} k={args.k}", flush=True)
    results = []
    t0 = time.time()
    for i, p in enumerate(stl_paths):
        cmd = [python_exe, "-c", _PER_SAMPLE_SCRIPT,
               str(p), detectors_json, str(args.k), str(args.iou_n),
               str(args.render_timeout), pkg_root]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.per_sample_timeout)
            if r.returncode != 0:
                rec = {"iou": 0.0, "detector": f"EXIT{r.returncode}",
                       "n_cand": 0}
            else:
                # Last non-empty stdout line is the JSON.
                lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
                rec = json.loads(lines[-1]) if lines else {"iou": 0.0,
                                                            "detector": "NOOUT",
                                                            "n_cand": 0}
        except subprocess.TimeoutExpired:
            rec = {"iou": 0.0, "detector": "TIMEOUT", "n_cand": 0}
        except Exception as e:
            rec = {"iou": 0.0, "detector": f"ERR:{type(e).__name__}", "n_cand": 0}

        rec["name"] = p.name
        results.append(rec)
        d = (rec["detector"] or "NONE")[:14]
        print(f"[{i+1:2d}/{len(stl_paths)}] {p.name:42s} IoU={rec['iou']:.4f} "
              f"det={d:14s} n_cand={rec['n_cand']}", flush=True)

    ious = np.array([r["iou"] for r in results])
    elapsed = time.time() - t0
    print(f"\n=== summary {Path(args.dir).name} ({elapsed:.0f}s) ===", flush=True)
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
        print(f"  {d:18s} n={len(lst):3d} mean={np.mean(lst):.4f} min={np.min(lst):.4f}", flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
