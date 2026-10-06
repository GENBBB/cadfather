"""Sweep detector — constant-profile / curved-trajectory case.

Algorithm
---------
For each candidate axis:
  1. Slice the mesh into K sections perpendicular to the axis.
  2. Fit a primitive (CIRCLE / RECT / POLYGON) per section via the
     C++ section fitter.
  3. Require all sections to share the SAME primitive kind AND
     near-equal SHAPE params (radius / w / h) -- i.e. a CONSTANT
     profile.  Allowed to vary: centre position (the centroid is what
     defines the trajectory).
  4. Build the centroid trajectory: 3D polyline [(cx_i, cy_i, z_i)].
  5. Reject straight trajectories (those are extrudes!): require RMS
     deviation from the linear fit to exceed ``curve_thresh`` of the
     trajectory length.
  6. Emit a CadQuery sweep:
        _path = cq.Workplane("XY").add(<3D-wire from centroids>)
        result = cq.Workplane(WP).<profile sketch>.sweep(_path)

If sections are too few, too inconsistent, or the path is too
straight, return [].

Limitations of v1
-----------------
- Only detects sweeps whose trajectory is roughly aligned with one of
  the principal axes.  A coil sweep whose path winds around an axis
  works (the per-axis sections still capture the circular profile).
- Profile-rotation along the path is not handled; CadQuery's sweep
  uses Frenet frames by default which is fine for moderate paths.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import (
    Section,
    _HAS_CPP_FITS,
    extract_sections,
)
from .extrude import DetectorOutput


_AXIS_TO_WP = {0: "YZ", 1: "XZ", 2: "XY"}
_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _shape_params_close(p_ref: list[float], p_cur: list[float], kind: str,
                        rtol: float) -> bool:
    """Compare the SHAPE-defining params (NOT the centroid position).
    For CIRCLE we compare only the radius; for RECT only (w, h); for
    POLYGON we compare the number of vertices and relative spread.
    """
    if kind == "CIRCLE":
        return abs(p_ref[2] - p_cur[2]) / max(abs(p_ref[2]), 1e-9) < rtol
    if kind == "RECT":
        return (abs(p_ref[2] - p_cur[2]) / max(abs(p_ref[2]), 1e-9) < rtol
                and abs(p_ref[3] - p_cur[3]) / max(abs(p_ref[3]), 1e-9) < rtol)
    if kind == "POLYGON":
        # Crude: same vertex count + similar bbox extent.
        if len(p_ref) != len(p_cur):
            return False
        a = np.asarray(p_ref).reshape(-1, 2)
        b = np.asarray(p_cur).reshape(-1, 2)
        ra = (a.max(axis=0) - a.min(axis=0)).max()
        rb = (b.max(axis=0) - b.min(axis=0)).max()
        return abs(ra - rb) / max(ra, 1e-9) < rtol
    return False


def _centroid_of_section(sec: Section) -> tuple[float, float]:
    """Best-fit centroid for the section's primitive."""
    p = sec.best_params
    kind = sec.best_kind
    if kind in ("CIRCLE", "RECT"):
        return float(p[0]), float(p[1])
    # POLYGON: mean of vertices
    arr = np.asarray(p).reshape(-1, 2)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _trajectory_3d(sections: list[Section], axis_idx: int
                   ) -> np.ndarray:
    """Build (cx, cy, z_world) trajectory.  cx, cy are in the section's
    2D plane coords; z_world is the world-axis coord at that section.
    """
    pts = []
    for sec in sections:
        cx, cy = _centroid_of_section(sec)
        pts.append((cx, cy, sec.z))
    return np.asarray(pts, dtype=np.float64)


def _curvature_score(traj_2d_xy: np.ndarray, traj_z: np.ndarray) -> float:
    """RMS deviation of the (x, y) centroid path from a straight line,
    normalised by the trajectory's axial length.  Returns 0 for a
    straight path; values > ~0.05 indicate a curved sweep.
    """
    if len(traj_2d_xy) < 3:
        return 0.0
    # Linear regression x = a*z + b ; y = c*z + d.  Residuals -> RMS.
    z = traj_z
    z_var = (z - z.mean())
    denom = float((z_var ** 2).sum())
    if denom < 1e-12:
        return 0.0
    ax = float((z_var * (traj_2d_xy[:, 0] - traj_2d_xy[:, 0].mean())).sum() / denom)
    bx = float(traj_2d_xy[:, 0].mean() - ax * z.mean())
    ay = float((z_var * (traj_2d_xy[:, 1] - traj_2d_xy[:, 1].mean())).sum() / denom)
    by = float(traj_2d_xy[:, 1].mean() - ay * z.mean())
    pred_x = ax * z + bx
    pred_y = ay * z + by
    rms = float(np.sqrt(((traj_2d_xy[:, 0] - pred_x) ** 2
                        + (traj_2d_xy[:, 1] - pred_y) ** 2).mean()))
    axial_len = float(z.max() - z.min())
    if axial_len < 1e-9:
        return 0.0
    return rms / axial_len


def _emit_sweep_program(traj_world: np.ndarray, axis_idx: int,
                        kind: str, params: list[float]) -> str:
    """Emit a CadQuery sweep program.  `traj_world` is shape (N, 3) in
    world-frame coordinates (the centroid path in world space).
    """
    if len(traj_world) < 2:
        return ""
    wp = _AXIS_TO_WP[axis_idx]
    # Center the profile at the origin (the sweep will start at the path's
    # first point and orient the profile along the tangent).
    if kind == "CIRCLE":
        # params = [cx, cy, r]
        sketch = f'cq.Workplane("{wp}").circle({params[2]:.4f})'
    elif kind == "RECT":
        # params = [cx, cy, w, h, theta]
        w, h = params[2], params[3]
        sketch = f'cq.Workplane("{wp}").rect({w:.4f}, {h:.4f})'
    elif kind == "POLYGON":
        pts = list(zip(params[0::2], params[1::2]))
        if len(pts) < 3:
            return ""
        # Re-centre to origin so the polyline doesn't translate the
        # profile relative to the path tangent.
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        sketch = f'cq.Workplane("{wp}").moveTo({pts[0][0] - cx:.4f}, {pts[0][1] - cy:.4f})'
        for x, y in pts[1:]:
            sketch += f'.lineTo({x - cx:.4f}, {y - cy:.4f})'
        sketch += '.close()'
    else:
        return ""

    # Build the path as a 3D wire from world-space points.
    pts_str = ", ".join(
        f"cq.Vector({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})" for p in traj_world)
    program = (
        "import cadquery as cq\n"
        f"_pts = [{pts_str}]\n"
        f"_edges = [cq.Edge.makeLine(_pts[i], _pts[i + 1]) "
        f"for i in range(len(_pts) - 1)]\n"
        f"_path = cq.Workplane(\"XY\").newObject([cq.Wire.assembleEdges(_edges)])\n"
        f"result = {sketch}.sweep(_path)\n"
    )
    return program


def _build_sweep_mesh(traj_world: np.ndarray, kind: str,
                      params: list[float]) -> Optional[trimesh.Trimesh]:
    """Fast trimesh approximation of the swept solid, so a sweep candidate can
    be CD-ranked and IoU-evaluated at iter-1 (without it, sweep candidates have
    no ``.mesh`` and the proposer's CD-rerank buries them).

    CIRCLE profile only: a circle is rotation-invariant, so trimesh's
    parallel-transported path frame cannot mis-orient it.  RECT/POLYGON in-plane
    orientation is NOT guaranteed to match the transported frame, so we return
    None there rather than risk emitting a wrongly-rotated mesh.  A wrong mesh
    would only ever rank LOW (bad CD) and be ignored, but None is cleaner.
    """
    try:
        if kind != "CIRCLE" or traj_world is None or len(traj_world) < 2:
            return None
        r = float(params[2])
        if r <= 1e-6:
            return None
        from shapely.geometry import Point
        from trimesh.creation import sweep_polygon
        path = np.asarray(traj_world, dtype=np.float64)
        # sweep_polygon rejects zero-length segments -> drop dup consecutive pts
        keep = [0]
        for i in range(1, len(path)):
            if np.linalg.norm(path[i] - path[keep[-1]]) > 1e-6:
                keep.append(i)
        path = path[keep]
        if len(path) < 2:
            return None
        poly = Point(0.0, 0.0).buffer(r, resolution=32)
        m = sweep_polygon(poly, path, cap=True)
        if m is None or len(m.faces) == 0:
            return None
        return m
    except Exception:
        return None


# ----------------------------- top-level ----------------------------------

def detect_sweep(mesh: trimesh.Trimesh,
                 n_sections: int = 6,
                 shape_rtol: float = 0.40,
                 curve_thresh: float = 0.02,
                 ) -> list[DetectorOutput]:
    """Detect sweeps along each principal axis.  Returns one candidate
    per (axis, qualifying trajectory).
    """
    if not _HAS_CPP_FITS:
        return []
    outs: list[DetectorOutput] = []
    for axis_idx in (0, 1, 2):
        secs = extract_sections(mesh, axis_idx=axis_idx, n_sections=n_sections)
        if len(secs) < 3:
            continue
        kinds = [s.best_kind for s in secs]
        if len(set(kinds)) != 1:
            continue
        kind = kinds[0]
        if kind not in ("CIRCLE", "RECT", "POLYGON"):
            continue
        p0 = secs[0].best_params
        if not all(_shape_params_close(p0, s.best_params, kind, shape_rtol)
                   for s in secs[1:]):
            continue

        traj = _trajectory_3d(secs, axis_idx)
        # Convert 2D centroids from section-local frame back to world.
        # extract_sections stores (cx, cy) in the workplane-2D frame;
        # need to map these to world coords.  For axis_idx perpendicular
        # cuts, the section's 2D coords ARE the workplane (u, v) coords,
        # which map to world coords directly:
        #   axis 0 (YZ wp): (u, v) -> (Y, Z); world = (sec.z, u, v) wait...
        # Use a consistent recipe: per axis, place (cx, cy, z) in world.
        # For section perpendicular to axis_idx, world coords are:
        #   axis 0: world = (z, cx, cy)  (Y, Z swap)? Actually the
        # to_planar transform we used in section_analyzer hands us 2D
        # coords whose mapping back to world depends on the basis.
        # Simpler: re-compute world centroids directly from the sliced
        # 3D vertices of each section.
        world_traj = []
        for s in secs:
            # Use the section.contour: shape (N, 2) in plane local
            # coords; the corresponding 3D coords can be reconstructed
            # from the slice plane (axis_idx = constant z).  We pick the
            # CENTROID of the 3D slice vertices as a robust trajectory
            # point.
            cont = s.contour
            if cont.size == 0:
                world_traj = None
                break
            cx2 = float(cont[:, 0].mean())
            cy2 = float(cont[:, 1].mean())
            # Map (cx2, cy2) on the slice plane back to world.  For the
            # axis-perpendicular slice we cut with, the 2D->3D inverse
            # is non-trivial; use the section's z (along axis) and
            # PROJECT the 2D centroid to the two non-axis world coords
            # by treating (cx2, cy2) directly as those world coords
            # (this is exact when to_planar uses the identity basis on
            # axis-aligned cuts, which it generally does).
            world = [0.0, 0.0, 0.0]
            world[axis_idx] = float(s.z)
            other = [i for i in range(3) if i != axis_idx]
            world[other[0]] = cx2
            world[other[1]] = cy2
            world_traj.append(world)
        if world_traj is None or len(world_traj) < 3:
            continue
        world_traj = np.asarray(world_traj, dtype=np.float64)

        # Curvature score: deviation of the OTHER two coords from a
        # straight line as a function of the axial coord.
        z = world_traj[:, axis_idx]
        xy = np.column_stack(
            [world_traj[:, i] for i in range(3) if i != axis_idx])
        curv = _curvature_score(xy, z)
        if curv < curve_thresh:
            continue  # straight -> this is an extrude, not a sweep.

        program = _emit_sweep_program(world_traj, axis_idx, kind, p0)
        if not program:
            continue
        # Higher = better; prefer curvier (= clearly a sweep, not extrude)
        # and simpler kinds (CIRCLE > RECT > POLYGON).
        kind_bonus = {"CIRCLE": 0.10, "RECT": 0.05, "POLYGON": 0.0}.get(kind, 0.0)
        score = float(min(curv * 5.0, 1.0)) + kind_bonus
        fast_mesh = _build_sweep_mesh(world_traj, kind, p0)
        outs.append(DetectorOutput(
            program=program,
            score=score,
            debug={
                "axis": _AXIS_NAMES[axis_idx],
                "section_kind": kind,
                "n_sections": len(secs),
                "curvature": float(curv),
                "shape_param_ref": [round(float(x), 4) for x in p0],
            },
            mesh=fast_mesh,
        ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
