#!/usr/bin/env python3
"""Trace a few revolve samples through detect_revolve to find quality bugs."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates.detectors.revolve import (
    _axisymmetry_ratio,
    _extract_rz_profile,
    detect_revolve,
)
from cadfit.candidates.residual import render_cadquery_to_stl


def iou(a, b, n=30000):
    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(n, 3))
    ia = a.contains(pts); ib = b.contains(pts)
    inter = int((ia & ib).sum()); union = int((ia | ib).sum())
    return inter / max(union, 1)


def main(stl_paths):
    for sp in stl_paths:
        sp = Path(sp)
        m = trimesh.load(sp, process=True)
        c = (m.bounds[0] + m.bounds[1]) / 2
        m.apply_translation(-c)
        m.apply_scale(2 / float(np.max(m.extents)))
        center = np.zeros(3)
        print(f"\n=== {sp.name} extents={m.extents.round(3).tolist()} ===")
        for ai in (0, 1, 2):
            r = _axisymmetry_ratio(m, ai, center)
            print(f"  axis={ai} sym_ratio={r:.3f}")
        outs = detect_revolve(m)
        print(f"  detect_revolve: {len(outs)} candidates")
        for o in outs:
            print(f"    axis={o.debug['axis']} sym={o.debug['symmetry_ratio']:.3f} n_pts={o.debug['n_points']}")
            with tempfile.TemporaryDirectory() as td:
                out_stl = Path(td) / "r.stl"
                ok = render_cadquery_to_stl(o.program, out_stl, timeout=30.0)
                if not ok:
                    print(f"      RENDER FAILED.  Program:\n{o.program[:400]}")
                    continue
                pred = trimesh.load(out_stl, process=True)
                if isinstance(pred, trimesh.Scene):
                    pred = trimesh.util.concatenate(list(pred.geometry.values()))
                cc = (pred.bounds[0] + pred.bounds[1]) / 2
                pred.apply_translation(-cc)
                pred.apply_scale(2 / float(np.max(pred.extents)))
                print(f"      IoU={iou(pred, m):.3f}  pred extents={pred.extents.round(3).tolist()}")
                # Dump first 6 (r, z) points to see the profile shape:
                axis_idx = {"X": 0, "Y": 1, "Z": 2}[o.debug["axis"]]
                rz = _extract_rz_profile(m, axis_idx, center, max_points=40, simplify_frac=0.005)
                if rz is not None:
                    print(f"      profile head: {rz[:6].round(4).tolist()}")
                    print(f"      profile tail: {rz[-3:].round(4).tolist()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--dir"):
        # --dir=PATH --n=K iterates the first K STLs in a directory.
        d = sys.argv[1].split("=", 1)[1]
        n = int(sys.argv[2].split("=", 1)[1]) if len(sys.argv) > 2 else 3
        paths = sorted(Path(d).glob("*.stl"))[:n]
        main([str(p) for p in paths])
    else:
        main(sys.argv[1:])
