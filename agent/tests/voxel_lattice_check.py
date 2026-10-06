#!/usr/bin/env python3
"""Check of lattice voxelization: the same result as trimesh, bit for bit.

Background. Inside det, `voxelized` (`subdivide`) is replaced by a per-face lattice
(`dsl_runtime._use_voxel_lattice`, `capabilities/voxel_lattice.py`): trimesh splits long
thin tessellation faces into 4^k children, sorting edges at every level, which was a
hot spot of det. The replacement is justified only by the output being the same, so
what is checked is exact equality of the grid (`matrix`, `transform`), not
"similarity".

Checked:

1. equality with `trimesh.voxel.creation.voxelize` on boxes, cylinders, spheres and
   rings, with rotation, with integer shifts (points on cell boundaries) and fractional
   ones, at different pitches, and on a long thin face (hundreds of splitting levels in
   total);
2. `max_iter` exceeded gives `ValueError`, as in trimesh (det catches it and lives
   without the lattice);
3. the substitution is installed once and lets other methods (`method='ray'`) pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from trimesh.voxel import creation  # noqa: E402

from cad_agent import dsl_runtime  # noqa: E402
from cad_agent.capabilities.voxel_lattice import voxelize_lattice  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def same(a, b) -> bool:
    return a.shape == b.shape and np.array_equal(a.transform, b.transform) and np.array_equal(a.matrix, b.matrix)


def meshes(rng: np.random.Generator):
    for i in range(40):
        kind = i % 4
        if kind == 0:
            m = trimesh.creation.box(extents=rng.uniform(1, 30, 3))
        elif kind == 1:
            m = trimesh.creation.cylinder(radius=rng.uniform(1, 10), height=rng.uniform(1, 30),
                                          sections=int(rng.integers(8, 48)))
        elif kind == 2:
            m = trimesh.creation.icosphere(subdivisions=int(rng.integers(1, 3)), radius=rng.uniform(2, 12))
        else:
            m = trimesh.creation.annulus(r_min=rng.uniform(1, 4), r_max=rng.uniform(5, 15),
                                         height=rng.uniform(0.5, 4), sections=int(rng.integers(16, 64)))
        if i % 3:
            m.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)))
        shift = rng.integers(-20, 20, 3).astype(float) if i % 5 == 0 else rng.uniform(-20, 20, 3)
        m.apply_translation(shift)
        yield f"{('box', 'cyl', 'sphere', 'annulus')[kind]}#{i}", m, [2.0, 1.0, 0.5, 0.8][i % 4]


def main() -> None:
    rng = np.random.default_rng(0)
    bad = [name for name, m, pitch in meshes(rng)
           if not same(creation.voxelize(m, pitch).fill(), voxelize_lattice(m, pitch).fill())]
    check("matches trimesh on 40 synthetic meshes", not bad, ", ".join(bad[:5]))

    # A long thin sliver: 120 long, 0.3 wide, i.e. k = 8 levels on the face.
    sliver = trimesh.creation.box(extents=(120.0, 0.3, 4.0))
    sliver.apply_transform(trimesh.transformations.rotation_matrix(0.3, [0, 0, 1]))
    check("matches on a long thin face",
          same(creation.voxelize(sliver, 2.0), voxelize_lattice(sliver, 2.0)))

    try:
        voxelize_lattice(sliver, 2.0, max_iter=2)
        check("max_iter exceeded: ValueError", False, "no exception")
    except ValueError:
        check("max_iter exceeded: ValueError", True)

    original = trimesh.Trimesh.voxelized
    dsl_runtime._use_voxel_lattice()
    patched = trimesh.Trimesh.voxelized
    dsl_runtime._use_voxel_lattice()
    check("the patch is installed once", trimesh.Trimesh.voxelized is patched and patched is not original)
    box = trimesh.creation.box(extents=(9.0, 5.0, 3.0))
    check("mesh method after the patch gives the same result",
          same(creation.voxelize(box, 1.0), box.voxelized(1.0)) and same(creation.voxelize(box, 1.0),
                                                                          box.voxelized(pitch=1.0)))
    check("method='ray' goes through the trimesh path",
          same(creation.voxelize(box, 1.0, method="ray"), box.voxelized(1.0, method="ray")))
    trimesh.Trimesh.voxelized = original

    print("SUMMARY:", "all passed" if not FAILURES else f"failures: {len(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
