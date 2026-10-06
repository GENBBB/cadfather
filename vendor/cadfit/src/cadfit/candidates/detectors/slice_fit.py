"""Slice-fit detector — primitive-aware extrude.

Cuts ONE cross-section through the residual's midpoint along each
candidate axis, fits primitives via the C++ ``fit_all``, and emits a
CadQuery snippet that uses the *winning* primitive as the sketch then
extrudes the full axial span.

Why it complements the silhouette-extrude path:
  - silhouette extrude is robust but always emits a polyline (no
    semantic recovery of circle / rect / arc primitives).
  - slice_fit emits ``.circle(r)`` for cylindrical residuals,
    ``.rect(w, h)`` for boxy ones, etc.  Cleaner code, fewer params for
    the optimizer to tune, often higher CAD-validity.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, _cg
from .extrude import DetectorOutput
from .. import cadrecode_emit as cre


_AXIS_TO_WP = {0: "YZ", 1: "XZ", 2: "XY"}
_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _slice_polygon(mesh: trimesh.Trimesh, axis_idx: int,
                   z: float) -> Optional[np.ndarray]:
    normal = [0.0, 0.0, 0.0]; normal[axis_idx] = 1.0
    origin = [0.0, 0.0, 0.0]; origin[axis_idx] = z
    try:
        sec3d = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec3d is None:
            return None
        if hasattr(sec3d, "to_planar"):
            p2d, _ = sec3d.to_planar()
        elif hasattr(sec3d, "to_2D"):
            p2d, _ = sec3d.to_2D()
        else:
            return None
        if p2d is None or not p2d.polygons_full:
            return None
        poly = max(p2d.polygons_full, key=lambda g: g.area)
        coords = np.asarray(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return None
        return coords.astype(np.float64)
    except Exception:
        return None


def _emit_primitive_extrude(kind: str, params: list, axis_idx: int,
                            lo: np.ndarray, hi: np.ndarray,
                            min_thickness_frac: float = 0.02) -> str:
    wp = _AXIS_TO_WP[axis_idx]
    raw_thickness = float(hi[axis_idx] - lo[axis_idx])
    max_extent = float(max(hi - lo))
    thickness = max(raw_thickness, min_thickness_frac * max_extent)
    delta = (thickness - raw_thickness) / 2.0
    # Shift along the extrude axis is folded into the workplane ORIGIN
    # (no trailing .translate) -- cleaner and matches the CADRecode
    # convention cq.Workplane(plane, origin=(...)).
    o = [0.0, 0.0, 0.0]
    o[axis_idx] = float(lo[axis_idx]) - delta

    # CADRecode-style emission (native cadquery Sketch API) so det blocks read
    # like VLM steps: cq.Workplane(...).sketch().<prim>.finalize().extrude().
    if kind == "CIRCLE":
        cx, cy, r = params
        chain = cre.circle_extrude(wp, o, cx, cy, r, thickness)
    elif kind == "RECT":
        cx, cy, w, h, th = params
        chain = cre.rect_extrude(wp, o, cx, cy, w, h, thickness, theta=th)
    elif kind == "POLYGON":
        pts = list(zip(params[0::2], params[1::2]))
        chain = cre.poly_extrude(wp, o, pts, thickness)
    else:
        return ""

    if not chain:
        return ""
    return "import cadquery as cq\nresult = " + chain


def _build_primitive_mesh(kind: str, params: list, axis_idx: int,
                          lo: np.ndarray, hi: np.ndarray
                          ) -> Optional[trimesh.Trimesh]:
    """Build a trimesh approximation of the primitive-extrude output.

    Bypasses cadquery so the IoU eval doesn't need to render.
    """
    try:
        from shapely.geometry import Polygon
        from trimesh.creation import extrude_polygon
        thickness = float(hi[axis_idx] - lo[axis_idx])
        if thickness <= 0:
            return None
        if kind == "CIRCLE":
            cx, cy, r = params
            n = 64
            shell = [(cx + r * math.cos(2 * math.pi * i / n),
                      cy + r * math.sin(2 * math.pi * i / n))
                     for i in range(n)]
        elif kind == "RECT":
            cx, cy, w, h, th = params
            ct = math.cos(th); st = math.sin(th)
            hw, hh = w * 0.5, h * 0.5
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            shell = [(cx + ct * cx_ - st * cy_,
                      cy + st * cx_ + ct * cy_) for cx_, cy_ in corners]
        elif kind == "POLYGON":
            pts = list(zip(params[0::2], params[1::2]))
            if len(pts) < 3:
                return None
            shell = [(float(x), float(y)) for x, y in pts]
        else:
            return None
        poly = Polygon(shell)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        m3 = extrude_polygon(poly, thickness)
        if m3 is None or len(m3.faces) == 0:
            return None
        # Permute axes to align with chosen world axis (extrude_polygon
        # extrudes along Z).  Permutation [2,0,1] (axis 0) is even
        # (det=+1, no flip); [0,2,1] (axis 1) is odd (det=-1, flip).
        if axis_idx == 0:
            m3.vertices = m3.vertices[:, [2, 0, 1]]
        elif axis_idx == 1:
            m3.vertices = m3.vertices[:, [0, 2, 1]]
            m3.invert()
        m3.apply_translation([
            0.0 if axis_idx != 0 else float(lo[0]),
            0.0 if axis_idx != 1 else float(lo[1]),
            0.0 if axis_idx != 2 else float(lo[2]),
        ])
        try:
            if float(m3.volume) < 0:
                m3.invert()
        except Exception:
            pass
        return m3
    except Exception:
        return None


def detect_slice_fit(mesh: trimesh.Trimesh,
                     depth_fracs: list[float] | None = None,
                     ) -> list[DetectorOutput]:
    """For each axis, fit primitives to the mid-section.  Emit one
    DetectorOutput per (axis, winning_primitive, depth_frac).

    ``depth_fracs`` defaults to [0.25, 0.5, 1.0]: each scales the raw
    mesh axial extent so the optimizer-free scorer can pick the right
    thickness directly from CD.  Each value is floored to
    ``0.02 * max_extent`` and capped at the raw extent so the prism
    never overshoots the residual along the axis.
    """
    if not _HAS_CPP_FITS:
        return []
    if depth_fracs is None:
        depth_fracs = [0.25, 0.5, 1.0]
    lo, hi = mesh.bounds
    max_extent = float(max(hi - lo))
    # Reference bbox area perpendicular to each axis — used to penalize
    # samples where the largest mid-slice polygon is tiny (e.g. a body
    # of disjoint thin columns gives a slice with many fragments; the
    # largest fragment fits a perfect circle but emits a tiny solid that
    # ignores most of the body).
    bbox_areas = {
        0: float((hi[1] - lo[1]) * (hi[2] - lo[2])),
        1: float((hi[0] - lo[0]) * (hi[2] - lo[2])),
        2: float((hi[0] - lo[0]) * (hi[1] - lo[1])),
    }

    outs: list[DetectorOutput] = []
    for axis_idx in (0, 1, 2):
        z_mid = 0.5 * (lo[axis_idx] + hi[axis_idx])
        coords = _slice_polygon(mesh, axis_idx, float(z_mid))
        if coords is None:
            continue
        # Compute polygon area via shoelace.
        x = coords[:, 0]; y = coords[:, 1]
        poly_area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                                    np.dot(y, np.roll(x, -1))))
        # Slice fill ratio: largest polygon area / bbox slice area.
        # Near 1.0 = the slice fills the bbox well; near 0 = the
        # detector would emit a vastly smaller solid than the body.
        bb_a = bbox_areas[axis_idx]
        fill = poly_area / bb_a if bb_a > 1e-9 else 0.0
        fits = _cg.section.fit_all(coords, -1.0)
        if not fits:
            continue
        top = fits[0]
        kind = str(top.kind).split(".")[-1]
        if kind not in ("CIRCLE", "RECT", "POLYGON"):
            continue
        # Wrap the existing program emitter, then string-replace the
        # extrude depth for each frac.  Cheaper than rebuilding from
        # scratch: the chain is identical except the extrude argument.
        raw_thickness = float(hi[axis_idx] - lo[axis_idx])
        for frac in depth_fracs:
            # Cap at the raw extent so the constant-section prism never
            # overshoots the residual along the slice axis (frac>1.0 used to
            # extrude 25% past each face -> oversized slabs that gouge cuts).
            target_depth = max(min(raw_thickness * float(frac), raw_thickness),
                               0.02 * max_extent)
            # Reuse _emit_primitive_extrude with the SAME bbox lo/hi but
            # patch the extrude depth in the output string.  Easier:
            # synthesize fake lo/hi that produce the target depth.
            scale = target_depth / max(raw_thickness, 1e-9)
            lo_fake = lo.copy().astype(float)
            hi_fake = hi.copy().astype(float)
            mid_axis = 0.5 * (lo_fake[axis_idx] + hi_fake[axis_idx])
            half = 0.5 * raw_thickness * scale
            lo_fake[axis_idx] = mid_axis - half
            hi_fake[axis_idx] = mid_axis + half
            program = _emit_primitive_extrude(kind, list(top.params),
                                              axis_idx, lo_fake, hi_fake)
            if not program:
                continue
            fast_mesh = _build_primitive_mesh(kind, list(top.params),
                                              axis_idx, lo_fake, hi_fake)
            # Normalize the C++ score (LOWER better) -> [0, 1] HIGHER.
            # Multiply by `fill` so a tiny slice (1 of N fragments) is
            # appropriately penalized vs a slice that fills the bbox.
            base = math.exp(-float(top.score)) * fill
            depth_penalty = 0.05 * abs(float(frac) - 1.0)
            outs.append(DetectorOutput(
                program=program,
                score=float(base - depth_penalty),
                debug={
                    "axis": _AXIS_NAMES[axis_idx],
                    "section_kind": kind,
                    "primitive_score": float(top.score),
                    "primitive_residual": float(top.residual),
                    "depth_frac": float(frac),
                    "depth": target_depth,
                    "slice_fill": float(fill),
                },
                mesh=fast_mesh,
            ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
