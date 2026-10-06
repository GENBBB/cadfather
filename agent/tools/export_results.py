#!/usr/bin/env python3
"""Copy the final reconstruction of every part out of a run directory.

For each part of the run, writes into the output directory:

    <out>/<part>.py     the best CadQuery code (`figures/<figure_id>/best.py`)
    <out>/<part>.stl    its mesh, when the run saved meshes (`logging.save_meshes`)
    <out>/results.csv   one row per part: figure id, score, IoU, GMS, failure, error

`<part>` is the STL file name the part came from. Parts from several splits of one
run are kept apart as `<out>/<split>/<part>.py`. A part without `best.py` (no valid
candidate was found) gets a row in `results.csv` and no files.

    python agent/tools/export_results.py work_dirs/<run> <out>

Standard library only; reads the run directory and never writes into it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy the final reconstruction of every part out of a run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory (work_dirs/<run>).")
    parser.add_argument("out_dir", type=Path, help="Where to write the results.")
    args = parser.parse_args()

    per_figure = args.run_dir / "per_figure.json"
    if not per_figure.is_file():
        print(f"Not a finished run directory (no per_figure.json): {args.run_dir}", file=sys.stderr)
        return 1
    records = json.loads(per_figure.read_text(encoding="utf-8"))

    groups = {str(r.get("group")) for r in records}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, n_code, n_mesh = [], 0, 0
    for record in records:
        figure_id = str(record["figure_id"])
        group, _, part = figure_id.partition("/")
        target = args.out_dir / group if len(groups) > 1 else args.out_dir
        source = args.run_dir / "figures" / figure_id
        code = source / "best.py"
        if code.is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(code, target / f"{part}.py")
            n_code += 1
            mesh = source / "best.stl"
            if mesh.is_file():
                shutil.copyfile(mesh, target / f"{part}.stl")
                n_mesh += 1
        metrics = record.get("metrics") or {}
        error = record.get("error") or ""
        rows.append({
            "figure_id": figure_id,
            "has_code": code.is_file(),
            "score": record.get("score"),
            "iou": metrics.get("iou"),
            "gms_norm": metrics.get("gms_norm"),
            "failure": metrics.get("failure"),
            "error": error.splitlines()[0][:200] if error else "",
        })

    with open(args.out_dir / "results.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["figure_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(records)} parts: {n_code} with code, {n_mesh} with a mesh -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
