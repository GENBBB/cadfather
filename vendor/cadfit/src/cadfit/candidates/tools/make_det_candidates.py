#!/usr/bin/env python3
"""CLI driver — writes N deterministic step candidates as .py files.

Example
-------
    python -m cadfit.candidates.tools.make_det_candidates \\
        --prev tests/student_repro/deepcad_samples/00000093+0.py \\
        --gt   tests/student_repro/deepcad_samples/00000093.stl \\
        --out  /tmp/det_cands \\
        --n    3
"""
import argparse
import json
import sys
from pathlib import Path

# Make `det_candidates` importable when this file is run directly.
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent   # .../python/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates import make_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True,
                    help="path to prev CadQuery .py (may be empty file for iter 1)")
    ap.add_argument("--gt",   required=True,
                    help="target STL path")
    ap.add_argument("--out",  required=True, help="output directory")
    ap.add_argument("--n",    type=int, default=3,
                    help="number of candidates to emit")
    ap.add_argument("--detectors", default="extrude",
                    help="comma-separated detector names (default: extrude)")
    ap.add_argument("--render_timeout", type=float, default=30.0)
    args = ap.parse_args()

    prev_code = Path(args.prev).read_text() if Path(args.prev).exists() else ""
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    cands = make_candidates(
        prev_code=prev_code,
        gt_stl_path=args.gt,
        num_candidates=args.n,
        detectors=[d.strip() for d in args.detectors.split(",") if d.strip()],
        render_timeout=args.render_timeout,
    )

    print(f"emitted {len(cands)} candidates -> {out_dir}", file=sys.stderr)
    summary = []
    for i, c in enumerate(cands):
        py_path = out_dir / f"cand_{i:02d}_{c.detector}_{c.op}.py"
        py_path.write_text(c.code)
        summary.append({
            "path": str(py_path),
            "op": c.op,
            "detector": c.detector,
            "score": c.score,
            "debug": c.debug,
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
