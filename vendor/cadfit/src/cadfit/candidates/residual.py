"""Render prev_code, compute ADD/CUT residual meshes against the GT.

The CadQuery render is done in a child process (matches kabisov's
`generate_meshes` style) so a hanging/crashing script can be killed
cleanly without polluting the caller.

Boolean differences go through `manifold3d` (already in the venv) - the
default trimesh boolean engine.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


# ---------- CadQuery render (subprocess + timeout) -------------------------

_RENDER_TEMPLATE = textwrap.dedent("""\
    import sys, traceback
    try:
        ns = {}
        with open(sys.argv[1]) as f:
            exec(f.read(), ns)
        r = ns.get("r") or ns.get("result")
        if r is None:
            sys.exit("no `r` or `result` in namespace")
        # CadQuery Workplane.val() -> first solid; .export writes STL.
        compound = r.val() if hasattr(r, "val") else r
        compound.export(sys.argv[2], tolerance=0.001, angularTolerance=0.1)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
""")


def _render_cadquery_inprocess(py_code: str, out_stl: Path,
                               timeout: float = 30.0) -> bool:
    """Render `py_code` to `out_stl` in the CURRENT interpreter.

    No subprocess startup cost (~2s per call saved) but provides no
    crash isolation: a CadQuery / OCCT segfault will take the caller
    down.  Use only when the cost of subprocess startup dominates and
    inputs are trusted (e.g. test fixtures).

    Timeout enforced via SIGALRM (POSIX only).
    """
    import signal
    out_stl = Path(out_stl)
    old_handler = None
    try:
        def _abort(*_):
            raise TimeoutError("render timeout")
        old_handler = signal.signal(signal.SIGALRM, _abort)
        signal.alarm(int(max(1, timeout)))
        try:
            ns: dict = {}
            exec(py_code, ns)
            r = ns.get("r") or ns.get("result")
            if r is None:
                return False
            compound = r.val() if hasattr(r, "val") else r
            compound.export(str(out_stl), tolerance=0.001,
                            angularTolerance=0.1)
        except Exception:
            return False
        return out_stl.exists() and out_stl.stat().st_size > 0
    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def render_cadquery_to_stl(py_code: str, out_stl: Path,
                           timeout: float = 30.0,
                           python: str = sys.executable,
                           inprocess: Optional[bool] = None) -> bool:
    """Render `py_code` to `out_stl`.

    Two backends:
      * subprocess (default): isolated; survives CadQuery segfaults;
        ~2-5 s per call due to Python/cadquery startup.
      * in-process: ~5x faster but no crash isolation.

    Selection (in order):
      1. The ``inprocess`` argument if explicitly True/False.
      2. The env var ``DET_RENDER_INPROCESS=1`` toggles the in-process
         path globally (useful in pytest sessions).
      3. Default: subprocess.

    Returns True iff the .stl was written and non-empty.
    """
    if inprocess is None:
        inprocess = os.environ.get("DET_RENDER_INPROCESS", "0") == "1"
    if inprocess:
        return _render_cadquery_inprocess(py_code, out_stl, timeout)

    out_stl = Path(out_stl)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(py_code)
        py_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RENDER_TEMPLATE)
        runner_path = f.name
    try:
        proc = subprocess.run(
            [python, runner_path, py_path, str(out_stl)],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0 or not out_stl.exists():
            return False
        return out_stl.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        for p in (py_path, runner_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------- Boolean residual computation ----------------------------------

@dataclass
class Residuals:
    """Result of computing ADD = gt \\ pred and CUT = pred \\ gt."""
    add_mesh: Optional[trimesh.Trimesh]
    cut_mesh: Optional[trimesh.Trimesh]
    gt_volume: float
    pred_volume: float
    add_volume: float
    cut_volume: float
    # Iou before any candidate is added.  Useful to early-exit when
    # current build already matches.
    iou: float


def _safe_volume(m: trimesh.Trimesh) -> float:
    try:
        return float(abs(m.volume))
    except Exception:
        return 0.0


def _manifold_difference64(a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh:
    """`trimesh.boolean.boolean_manifold`, but through `Mesh64`.

    trimesh hands manifold3d float32, and manifold3d 3.0.0 is nondeterministic on
    such input: the same meshes give many different triangulations. The input is
    still rounded to float32, as in trimesh: full float64 is deterministic but leaves
    gaps in the remainder, and "parent + ops" more often comes out open (not
    watertight).
    """
    from manifold3d import Manifold, Mesh64

    if not (a.is_volume and b.is_volume):
        raise ValueError("Not all meshes are volumes!")
    ma, mb = (Manifold(mesh=Mesh64(vert_properties=np.array(m.vertices, dtype=np.float32).astype(np.float64),
                                   tri_verts=np.array(m.faces, dtype=np.uint64)))
              for m in (a, b))
    res = (ma - mb).to_mesh64()
    # The output also goes through float32, as trimesh's `to_mesh()` does. The copies
    # also detach the arrays from the `Mesh64` memory (otherwise nanobind prints "leaked" on exit).
    return trimesh.Trimesh(vertices=np.array(res.vert_properties, dtype=np.float32).astype(np.float64),
                           faces=np.array(res.tri_verts), process=False)


def _boolean_trimesh(a: trimesh.Trimesh, b: trimesh.Trimesh,
                     engine: str) -> Optional[trimesh.Trimesh]:
    """trimesh.boolean.difference(a-b) with full error suppression."""
    try:
        if engine == "manifold":
            out = _manifold_difference64(a, b)
        else:
            out = trimesh.boolean.difference([a, b], engine=engine)
        if isinstance(out, list):
            out = trimesh.util.concatenate(out) if out else None
        if out is None or len(out.faces) == 0:
            return None
        return out
    except Exception:
        return None


def _voxel_diff(a: trimesh.Trimesh, b: trimesh.Trimesh,
                pitch: float) -> Optional[trimesh.Trimesh]:
    """Robust voxel-grid difference (a \\ b), works on non-watertight
    meshes too.  Voxelize both at a shared grid, XOR the occupancy,
    marching-cubes the result back to a triangle mesh.
    """
    # Shared bbox so the two voxel grids align.
    lo = np.minimum(a.bounds[0], b.bounds[0]) - pitch
    hi = np.maximum(a.bounds[1], b.bounds[1]) + pitch
    extent = hi - lo
    nx, ny, nz = (np.ceil(extent / pitch)).astype(int) + 1
    if min(nx, ny, nz) < 4:
        return None
    # Build a regular grid of cell centres, test point-in-mesh for each.
    xs = lo[0] + (np.arange(nx) + 0.5) * pitch
    ys = lo[1] + (np.arange(ny) + 0.5) * pitch
    zs = lo[2] + (np.arange(nz) + 0.5) * pitch
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    try:
        in_a = a.contains(pts).reshape(nx, ny, nz)
        in_b = b.contains(pts).reshape(nx, ny, nz)
    except Exception:
        return None
    occ = in_a & (~in_b)
    if not occ.any():
        return None
    # marching cubes via trimesh.voxel.ops
    try:
        from trimesh.voxel.ops import matrix_to_marching_cubes
        out = matrix_to_marching_cubes(occ, pitch=pitch)
        # marching_cubes returns a mesh in voxel-grid coords; shift to lo.
        out.apply_translation(lo)
        if out is None or len(out.faces) == 0:
            return None
        return out
    except Exception:
        return None


def compute_residuals(pred_mesh: trimesh.Trimesh,
                      gt_mesh: trimesh.Trimesh,
                      engine: str = "manifold",
                      voxel_pitch_frac: float = 0.01) -> Residuals:
    """ADD = gt \\ pred ; CUT = pred \\ gt.

    Tries trimesh.boolean first (fast, exact, needs watertight); on
    failure falls back to a voxel-grid XOR + marching cubes (robust to
    non-watertight, accuracy controlled by ``voxel_pitch_frac`` of the
    union bbox extent).
    """
    gv = _safe_volume(gt_mesh)
    pv = _safe_volume(pred_mesh)

    # Try the exact Boolean path first.
    add_mesh = _boolean_trimesh(gt_mesh, pred_mesh, engine)
    cut_mesh = _boolean_trimesh(pred_mesh, gt_mesh, engine)

    # If either side failed, fall back to voxel diff at a shared pitch.
    if add_mesh is None or cut_mesh is None:
        lo = np.minimum(gt_mesh.bounds[0], pred_mesh.bounds[0])
        hi = np.maximum(gt_mesh.bounds[1], pred_mesh.bounds[1])
        pitch = float(max(hi - lo)) * voxel_pitch_frac
        if pitch > 0:
            if add_mesh is None:
                add_mesh = _voxel_diff(gt_mesh, pred_mesh, pitch)
            if cut_mesh is None:
                cut_mesh = _voxel_diff(pred_mesh, gt_mesh, pitch)

    av = _safe_volume(add_mesh) if add_mesh is not None else 0.0
    cv = _safe_volume(cut_mesh) if cut_mesh is not None else 0.0

    inter = max(gv - av, 0.0)
    union = max(gv + cv, pv + av)
    iou = inter / union if union > 1e-9 else 0.0

    return Residuals(add_mesh, cut_mesh, gv, pv, av, cv, iou)
