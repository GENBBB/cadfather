"""Revolve detector — axisymmetric solids → ``.revolve()``.

Algorithm
---------
For each candidate axis (X, Y, Z):
  1. Cut two perpendicular cross-sections that both contain the axis.
     If the shape is axisymmetric, the two cross-sections have nearly
     equal area.  We require ratio = min_area / max_area > ``sym_thresh``.
  2. Take one of those cross-sections, project to 2D (r, z) where
     r = distance from axis, z = position along axis.  Take the r >= 0
     half of the polygon.
  3. Simplify the (r, z) profile (Ramer-Douglas-Peucker via C++) and
     emit ``cq.Workplane(<plane>).polyline([(r0,z0),...]).close().revolve()``.

Score = ratio (higher = more symmetric).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from .extrude import DetectorOutput
from ..section_analyzer import _cg, _HAS_CPP_FITS
from .. import cadrecode_emit as cre


_AXIS_NAME = {0: "X", 1: "Y", 2: "Z"}

# CadQuery's .revolve() rotates around the workplane's LOCAL Y axis by
# default.  Pick a standard workplane string whose local-Y axis aligns
# with the chosen world revolve axis:
#   axis X  -> "YX"  (local Y = world X)
#   axis Y  -> "XY"  (local Y = world Y)
#   axis Z  -> "XZ"  (local Y = world Z)
# In each of these planes the profile (r, z) maps directly to local
# (x, y) via moveTo(r, z).
_AXIS_TO_WP_REVOLVE = {
    0: "YX",
    1: "XY",
    2: "XZ",
}


# ----------------------------- Axisymmetry --------------------------------

def _section_area(mesh: trimesh.Trimesh, origin: np.ndarray,
                  normal: np.ndarray) -> float:
    """Sum-of-area of the 2D polygons cut by a plane.  0 on failure."""
    try:
        sec3d = mesh.section(plane_origin=origin.tolist(),
                             plane_normal=normal.tolist())
        if sec3d is None:
            return 0.0
        if hasattr(sec3d, "to_planar"):
            p2d, _ = sec3d.to_planar()
        elif hasattr(sec3d, "to_2D"):
            p2d, _ = sec3d.to_2D()
        else:
            return 0.0
        if p2d is None or not p2d.polygons_full:
            return 0.0
        return sum(p.area for p in p2d.polygons_full)
    except Exception:
        return 0.0


def _axisymmetry_ratio(mesh: trimesh.Trimesh, axis_idx: int,
                       center: np.ndarray,
                       n_angles: int = 4) -> float:
    """Sample cross-section areas at ``n_angles`` evenly-spaced angles
    around the candidate axis.  Return min/max area ratio.

    For a true revolve this is ~1.0 at any number of angles.  For a
    prismatic slab the two perpendicular cuts give equal areas (false
    positive), but a 45-degree cut gives a different area -> the ratio
    drops <1.  Hence n_angles >= 4 (default).
    """
    # Build two orthonormal vectors u, v perpendicular to the axis.
    e_axis = np.zeros(3); e_axis[axis_idx] = 1.0
    others = [i for i in range(3) if i != axis_idx]
    u = np.zeros(3); u[others[0]] = 1.0
    v = np.zeros(3); v[others[1]] = 1.0
    areas = []
    for k in range(n_angles):
        theta = math.pi * k / n_angles   # 0 .. pi (planes contain axis)
        # plane normal = perpendicular to axis, rotated by theta in (u, v).
        normal = u * math.cos(theta) + v * math.sin(theta)
        areas.append(_section_area(mesh, center, normal))
    areas = [a for a in areas if a > 1e-9]
    if len(areas) < n_angles:
        return 0.0
    return min(areas) / max(areas)


# ---------------------------- (r, z) profile ------------------------------

def _extract_rz_profile(mesh: trimesh.Trimesh, axis_idx: int,
                        center: np.ndarray, max_points: int,
                        simplify_frac: float) -> Optional[np.ndarray]:
    """Build the EXACT (r, z) revolve profile from a half-section through
    the revolve axis (preserving cavities and concavities).

    Algorithm
    ---------
    1. Cut a slice perpendicular to one of the non-axis directions.
       The slice plane contains the revolve axis.
    2. ``sec3d.discrete`` returns one or more closed 3D rings.
    3. Project each ring to 2D (r_signed, z):
            z        = world[axis_idx]
            r_signed = world[other_axis]
    4. Build a shapely MultiPolygon from these rings and intersect
       with the half-plane r_signed >= 0.  The result is the
       cross-section of the upper half of the body -- exactly the
       revolve profile.
    5. Pick the largest resulting polygon (the others, if any, are
       disjoint features that need a separate revolve).

    Handles both cases:
        - Annular bodies whose section has one ring per half-plane
          (e.g. tube with an inner cavity).
        - Solid bodies whose section is a single ring crossing the
          axis -- shapely's half-plane intersection clips it correctly.
    """
    from shapely.geometry import Polygon, MultiPolygon, box
    from shapely.ops import unary_union

    others = [i for i in range(3) if i != axis_idx]
    cut_idx, r_idx = others[0], others[1]

    try:
        sec3d = mesh.section(plane_origin=center.tolist(),
                             plane_normal=[1.0 if i == cut_idx else 0.0
                                           for i in range(3)])
    except Exception:
        return None
    if sec3d is None:
        return None
    try:
        discrete = sec3d.discrete
    except Exception:
        discrete = []
    if not discrete:
        return None

    # Build shapely polygons in (r_signed, z).
    polys: list[Polygon] = []
    for ring in discrete:
        ring = np.asarray(ring, dtype=np.float64)
        if len(ring) < 4:
            continue
        rs = ring[:, r_idx] - center[r_idx]
        zs = ring[:, axis_idx] - center[axis_idx]
        # Drop duplicate closing vertex.
        if np.allclose(ring[0], ring[-1]):
            rs = rs[:-1]; zs = zs[:-1]
        try:
            p = Polygon(np.column_stack([rs, zs]))
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and p.area > 1e-9:
                polys.append(p)
        except Exception:
            pass
    if not polys:
        return None

    # Half-plane clip: keep r_signed >= 0.
    full = unary_union(polys)
    if full.is_empty:
        return None
    half_bounds = full.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = half_bounds
    eps = max((maxx - minx) + (maxy - miny), 1.0) * 0.01
    right_half = box(0.0, miny - eps, maxx + eps, maxy + eps)
    clipped = full.intersection(right_half)
    if clipped.is_empty:
        return None

    # Normalize to list of polygons; pick the largest by area.
    if isinstance(clipped, Polygon):
        polys_out = [clipped]
    elif isinstance(clipped, MultiPolygon):
        polys_out = list(clipped.geoms)
    else:
        # GeometryCollection — pick its polygon members.
        polys_out = [g for g in getattr(clipped, "geoms", [])
                     if isinstance(g, Polygon)]
        if not polys_out:
            return None
    polys_out.sort(key=lambda p: p.area, reverse=True)
    profile_poly = polys_out[0]

    coords = list(profile_poly.exterior.coords)[:-1]
    rz = np.asarray(coords, dtype=np.float64)
    if len(rz) < 3:
        return None

    # Lift any on-axis vertices by eps so OCCT doesn't reject the edge.
    z_extent = float(rz[:, 1].max() - rz[:, 1].min())
    if z_extent < 1e-9:
        return None
    eps_r = 1e-4 * z_extent
    rz = rz.copy()
    rz[:, 0] = np.maximum(rz[:, 0], eps_r)

    # Simplify via C++ RDP if available.
    if _HAS_CPP_FITS:
        scale = float(np.max(rz.max(axis=0) - rz.min(axis=0)))
        tol = max(simplify_frac * scale, 1e-6)
        f = _cg.section.fit_polygon(rz.astype(np.float64), tol)
        params = list(f.params)
        if len(params) >= 6:
            rz = np.asarray(params).reshape(-1, 2)

    if len(rz) > max_points:
        idx2 = np.linspace(0, len(rz) - 1, max_points).astype(int)
        rz = rz[idx2]
    return rz


# ----------------------------- emit ---------------------------------------

def _emit_revolve_program(rz: np.ndarray, axis_idx: int,
                          center: np.ndarray) -> str:
    """Emit a CadQuery revolve snippet using STANDARD named workplanes.

    Custom ``cq.Plane(..., xDir=..., normal=...)`` workplanes silently
    produce 1-face degenerate solids when combined with ``.revolve()``
    (default axis through workplane origin).  Standard plane strings
    ("XY", "XZ", "YX") map cleanly: their LOCAL Y axis is the chosen
    world axis, so the default revolve rotates around the right axis.

    Profile (r, z) maps to ``moveTo(r, z)``:
        local_x = r (radial)
        local_y = z (axial = revolve axis)

    The revolve produces a body centered on the workplane's origin
    (world origin).  We post-translate by ``center`` so the body sits
    at the mesh's bbox centroid.
    """
    wp = _AXIS_TO_WP_REVOLVE[axis_idx]
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    # Center offset folded into the workplane ORIGIN (no trailing
    # .translate): the profile is sketched and revolved about the axis
    # through this origin directly.
    # CADRecode-style emission (native cadquery Sketch API). The default
    # .revolve() axis is the workplane LOCAL Y axis through the origin, i.e.
    # the explicit axis ((0,0),(0,1)); the profile (r, z) maps to local (x, y).
    pts = [(float(r), float(z)) for r, z in rz]
    chain = cre.poly_revolve(wp, (cx, cy, cz), pts, angle=360,
                             axis=((0.0, 0.0), (0.0, 1.0)))
    if not chain:
        return ""
    return "import cadquery as cq\nresult = " + chain


# ----------------------------- top-level ----------------------------------

def _build_revolve_mesh(rz: np.ndarray, axis_idx: int, center: np.ndarray,
                        n_sections: int = 64) -> Optional[trimesh.Trimesh]:
    """Build a Trimesh by revolving (r, z) profile around the chosen
    world axis -- bypasses cadquery entirely.

    trimesh.creation.revolve revolves around the local-Z axis by default;
    we revolve the (r, z) profile points then permute axes to align
    with the chosen world axis, finally translate to ``center``.
    """
    try:
        from trimesh.creation import revolve
        # Build linestring sampled at the profile vertices (r, z).
        # trimesh.creation.revolve expects (r, z) pairs.
        m3 = revolve(rz, sections=n_sections)
        if m3 is None or len(m3.faces) == 0:
            return None
        # revolve produces a body around the Z axis: (x=r·cos, y=r·sin, z=z).
        # Permutations: [2,0,1] (axis 0) is EVEN (det=+1, no flip);
        # [0,2,1] (axis 1) is ODD (det=-1, flip).
        if axis_idx == 0:
            m3.vertices = m3.vertices[:, [2, 0, 1]]
        elif axis_idx == 1:
            m3.vertices = m3.vertices[:, [0, 2, 1]]
            m3.invert()
        # axis_idx == 2: trimesh-z already maps to world-z; no swap.
        m3.apply_translation([float(center[0]), float(center[1]), float(center[2])])
        # Safety: detect inverted winding via negative volume.
        try:
            if float(m3.volume) < 0:
                m3.invert()
        except Exception:
            pass
        return m3
    except Exception:
        return None


def detect_revolve(mesh: trimesh.Trimesh,
                   sym_thresh: float = 0.65,
                   max_points: int = 40,
                   simplify_frac: float = 0.005,
                   ) -> list[DetectorOutput]:
    """Run the axisymmetry test on every cardinal axis, emit a revolve
    DetectorOutput for each passing axis (best first, by ratio)."""
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    candidates = []
    for axis_idx in (0, 1, 2):
        ratio = _axisymmetry_ratio(mesh, axis_idx, center)
        if ratio < sym_thresh:
            continue
        rz = _extract_rz_profile(mesh, axis_idx, center, max_points, simplify_frac)
        if rz is None or len(rz) < 3:
            continue
        program = _emit_revolve_program(rz, axis_idx, center)
        fast_mesh = _build_revolve_mesh(rz, axis_idx, center)
        # Score: ratio itself; we use NEGATIVE so lower-score = better
        # (matches the rest of the codebase convention from extrude.py).
        # ratio in [sym_thresh, 1] -> score in [-1, -sym_thresh].
        candidates.append(DetectorOutput(
            program=program,
            # Higher = better.  Use the symmetry ratio (already in
            # [sym_thresh, 1.0]) directly + a small kind-of-shape bonus
            # for short profiles (simpler revolves usually fit better).
            score=float(ratio),
            debug={
                "axis": _AXIS_NAME[axis_idx],
                "symmetry_ratio": ratio,
                "n_points": int(len(rz)),
            },
            mesh=fast_mesh,
        ))
    # Best-symmetry first (descending).
    candidates.sort(key=lambda o: o.score, reverse=True)
    return candidates
