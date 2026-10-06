#!/usr/bin/env python3
"""Check that the repair of a holed GT survives a fork.

Background: `det_rebuild` is called in a fork, and so is the warm-up itself.
Nobody inherits the fork's memory, so the voxel repair of a holed GT (256^3,
seconds) would be paid again in EVERY det call, on the part of the set where GT
is not watertight. The GT frame already survived this because it is stored as a
file; the repair now does too.

A fork is modelled here by a **new** `DetProposer` object: from our state's point
of view that is exactly what `fork` does: nothing warm in memory, files in the
part's scratch directory still there.

Checks:

1. a holed GT is repaired and the result is stored as a file next to the frame;
2. the next "fork" takes the ready result without repeating voxelization;
3. a round trip through STL keeps the mesh watertight (otherwise
   `compute_residuals` would silently fall from the exact boolean to the voxel
   fallback);
4. a whole GT creates no repair file: nothing to repair;
5. an unusable file (empty, corrupt) is not used;
6. if a write still loses watertightness, the file is removed, not returned;
7. repair counters reach the fork's payload as a per-call increment.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

import trimesh  # noqa: E402

from cad_agent import dsl_runtime  # noqa: E402
from cad_agent.capabilities import det as det_mod  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def fork() -> det_mod.DetProposer:
    """A new `DetProposer` is the same as a new fork: memory empty, files present."""
    return det_mod.DetProposer()


def main() -> None:
    # The snapshot dependencies are not needed here: this checks GT handling, not
    # operation selection. The stub is installed before the constructor that needs them.
    dsl_runtime.import_det_deps = lambda: object()

    tmp = Path(tempfile.mkdtemp(prefix="det_gt_repair_"))
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    holed = trimesh.Trimesh(box.vertices.copy(), box.faces[:-2].copy(), process=True)
    holed_path = tmp / "holed.stl"
    holed.export(holed_path)
    whole_path = tmp / "whole.stl"
    box.export(whole_path)

    print("1. A leaky GT is repaired and stored as a file")
    first = fork()
    frame = first._gt_frame_path(holed_path, tmp)
    check("the original GT is really leaky", not first._load_mesh(frame).is_watertight)
    mesh = first._load_gt_repaired(frame)
    repaired_path = first._repaired_path(frame)
    check("the repair closed the geometry", mesh.is_watertight)
    check("repair file is created", repaired_path.exists(), repaired_path.name)
    check("repair is recorded as a miss and a write",
          (first._repair_stats.misses, first._repair_stats.writes) == (1, 1),
          first._repair_stats.to_dict())

    print("\n2. The next fork takes the ready result")
    second = fork()
    reused = second._load_gt_repaired(frame)
    check("hit in the repair cache",
          (second._repair_stats.hits, second._repair_stats.misses) == (1, 0),
          second._repair_stats.to_dict())
    check("voxelization was not repeated", second._repair_stats.writes == 0)

    print("\n3. A round trip through STL keeps the mesh closed")
    check("the one loaded from disk is closed", reused.is_watertight)
    check("volume matches the one repaired in memory",
          abs(reused.volume - mesh.volume) < 1e-9 * abs(mesh.volume),
          f"{reused.volume} vs {mesh.volume}")

    print("\n4. An intact GT needs no repair")
    clean = fork()
    clean_frame = clean._gt_frame_path(whole_path, tmp)
    clean_mesh = clean._load_gt_repaired(clean_frame)
    check("the intact GT stayed closed", clean_mesh.is_watertight)
    check("there is no repair file", not clean._repaired_path(clean_frame).exists())
    check("repair counter is untouched",
          clean._repair_stats.lookups == 0 and clean._repair_stats.writes == 0,
          clean._repair_stats.to_dict())

    print("\n5. A bad file is not used")
    # An empty file is an unfinished write, not a repair. The threshold is exactly
    # "non-empty": a closed box is 12 faces, and a threshold of 50 (copied from the
    # acceptance condition of the voxel repair) would reject a valid result.
    repaired_path.write_text("", encoding="utf-8")
    broken = fork()
    again = broken._load_gt_repaired(frame)
    check("empty file rejected, GT repaired anew",
          again.is_watertight and broken._repair_stats.hits == 0,
          broken._repair_stats.to_dict())
    # There is one lookup, so one miss: a bad file and the repair that follows are
    # not two different misses, otherwise the hit rate lies.
    check("miss counted once", broken._repair_stats.misses == 1,
          broken._repair_stats.to_dict())
    check("a valid repair is written instead", repaired_path.exists() and repaired_path.stat().st_size > 0)

    print("\n6. Loss of closedness on write is caught, not passed on")
    repaired_path.unlink()
    liar = fork()
    real_load = liar._load_mesh
    calls: list[Path] = []

    def load_breaking_watertight(path):
        # Break exactly the verification read, the one that checks the file after
        # writing. The first read (of the frame itself) must stay genuine.
        mesh = real_load(path)
        calls.append(Path(path))
        if ".tmp." in Path(path).name:
            return trimesh.Trimesh(box.vertices.copy(), box.faces[:-2].copy(), process=True)
        return mesh

    liar._load_mesh = load_breaking_watertight
    kept = liar._load_gt_repaired(frame)
    check("the repair is still returned to the caller", kept.is_watertight)
    check("the bad file is not left behind", not repaired_path.exists())
    check("the write is not counted", liar._repair_stats.writes == 0, liar._repair_stats.to_dict())
    check("temporary file is removed", not any(p.name.endswith(".stl") and ".tmp." in p.name
                                          for p in tmp.iterdir()))

    print("\n7. Counters reach the fork payload")
    det_mod._DET_PROPOSER = None
    probe = det_mod._get_det_proposer()
    frame_for_probe = probe._gt_frame_path(holed_path, tmp)
    real_rebuild = det_mod.det_rebuild

    def rebuild_touching_gt(gt, pred, cache_dir, deadline=None, max_faces=None):
        det_mod._get_det_proposer()._load_gt_repaired(Path(frame_for_probe))
        return ["op0"]

    det_mod.det_rebuild = rebuild_touching_gt
    try:
        payload = det_mod.det_rebuild_isolated(str(holed_path), None, str(tmp), 10.0)
    finally:
        det_mod.det_rebuild = real_rebuild
    repair = payload.get("gt_repair_cache")
    check("payload carries the repair counter", isinstance(repair, dict), str(repair))
    check("it holds the per-call increment, not the accumulated value",
          bool(repair) and (repair.get("hits", 0) + repair.get("misses", 0)) == 1, str(repair))
    check("the skeleton is counted separately from the repair",
          payload.get("gt_cache", {}).get("misses") == 1, str(payload.get("gt_cache")))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("GT repair survives the fork.")


if __name__ == "__main__":
    main()
