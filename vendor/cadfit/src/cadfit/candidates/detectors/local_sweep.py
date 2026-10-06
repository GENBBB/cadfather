"""Local-sweep detector — CADFit-style local swept bands.

Detects a (roughly) constant cross-section profile that travels along a
curved centroid trajectory, over a LOCAL axial band.  Good for pipes,
tubes, bent bars, and the simpler g-sweep parts.

Algorithm
---------
For each candidate axis:
  1. Slice into N sections; record each section's primitive kind,
     shape params, and 3D centroid.
  2. Group consecutive sections with the SAME kind + similar shape
     (radius / w,h) into bands -- the profile is ~constant within a band.
  3. For a band, build the centroid polyline (the sweep path) and
     sweep the band's mean profile along it via trimesh sweep_polygon.
  4. Emit one candidate per band whose path is non-trivially long.

Fast-mesh only; no cadquery.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, extract_sections
from .extrude import DetectorOutput


_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _section_centroid_world(sec, axis_idx: int) -> Optional[np.ndarray]:
    """World-space 3D centroid of a section's contour."""
    cont = getattr(sec, "contour", None)
    if cont is None or cont.size == 0:
        return None
    c2 = cont.mean(axis=0)  # (u, v) in the slice 2D frame
    # extract_sections builds the 2D frame via to_planar; for axis-perp
    # slices the 2D coords map back to the two non-axis world coords.
    other = [i for i in range(3) if i != axis_idx]
    w = np.zeros(3)
    w[axis_idx] = float(sec.z)
    w[other[0]] = float(c2[0])
    w[other[1]] = float(c2[1])
    return w


def _profile_close(p_ref, p_cur, kind: str, rtol: float) -> bool:
    if kind == "CIRCLE":
        return abs(p_ref[2] - p_cur[2]) / max(abs(p_ref[2]), 1e-9) < rtol
    if kind == "RECT":
        return (abs(p_ref[2] - p_cur[2]) / max(abs(p_ref[2]), 1e-9) < rtol
                and abs(p_ref[3] - p_cur[3]) / max(abs(p_ref[3]), 1e-9) < rtol)
    return False


def _emit_sweep_program(kind: str, params, path_pts: np.ndarray) -> str:
    """Emit a clean multi-statement CadQuery sweep program: build the 3D
    path wire from the centroid polyline, sketch a circle/rect profile,
    and sweep it.  Returns "" for unsupported kinds."""
    pts = np.asarray(path_pts, dtype=np.float64)
    if len(pts) < 3:
        return ""
    if kind == "CIRCLE":
        r = float(params[2]); prof = f".circle({r:.5f})"
    elif kind == "RECT":
        w, h = float(params[2]), float(params[3]); prof = f".rect({w:.5f}, {h:.5f})"
    else:
        return ""
    pstr = ", ".join(f"cq.Vector({p[0]:.5f},{p[1]:.5f},{p[2]:.5f})" for p in pts)
    t0 = pts[1] - pts[0]
    n = t0 / max(np.linalg.norm(t0), 1e-9)
    o = pts[0]
    return (
        "import cadquery as cq\n"
        f"_pts = [{pstr}]\n"
        "_edges = [cq.Edge.makeLine(_pts[i], _pts[i+1]) for i in range(len(_pts)-1)]\n"
        "_path = cq.Workplane('XY').newObject([cq.Wire.assembleEdges(_edges)])\n"
        f"_wp = cq.Workplane(cq.Plane(origin=cq.Vector({o[0]:.5f},{o[1]:.5f},{o[2]:.5f}), "
        f"normal=cq.Vector({n[0]:.5f},{n[1]:.5f},{n[2]:.5f})))\n"
        f"result = _wp{prof}.sweep(_path)\n"
    )


def _build_sweep_mesh(kind: str, params, path_pts: np.ndarray
                      ) -> Optional[trimesh.Trimesh]:
    try:
        from shapely.geometry import Polygon
        from trimesh.creation import sweep_polygon
        if kind == "CIRCLE":
            r = float(params[2])
            n = 32
            prof = [(r * math.cos(2 * math.pi * i / n),
                     r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        elif kind == "RECT":
            w, h = float(params[2]), float(params[3])
            hw, hh = w / 2, h / 2
            prof = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        else:
            return None
        poly = Polygon(prof)
        if not poly.is_valid or poly.area <= 0:
            return None
        if len(path_pts) < 3:
            return None
        m3 = sweep_polygon(poly, path_pts)
        if m3 is None or len(m3.faces) == 0:
            return None
        if float(m3.volume) < 0:
            m3.invert()
        return m3
    except Exception:
        return None


def detect_local_sweeps(mesh: trimesh.Trimesh,
                        n_sections: int = 20,
                        shape_rtol: float = 0.25,
                        min_curve_frac: float = 0.02,
                        ) -> list[DetectorOutput]:
    if not _HAS_CPP_FITS:
        return []
    try:
        gt_vol = float(abs(mesh.volume))
    except Exception:
        gt_vol = 0.0
    max_extent = float(np.max(mesh.extents))
    outs: list[DetectorOutput] = []

    for axis_idx in (0, 1, 2):
        secs = extract_sections(mesh, axis_idx=axis_idx,
                                n_sections=n_sections)
        if len(secs) < 4:
            continue
        # Group into constant-profile bands.
        bands = []
        cur = []
        ref_kind = None; ref_params = None
        for sec in secs:
            k = sec.best_kind
            if k not in ("CIRCLE", "RECT"):
                if len(cur) >= 4:
                    bands.append((ref_kind, cur))
                cur = []; ref_kind = None; ref_params = None
                continue
            if ref_kind is None:
                ref_kind = k; ref_params = sec.best_params
                cur = [sec]
            elif k == ref_kind and _profile_close(ref_params,
                                                  sec.best_params, k, shape_rtol):
                cur.append(sec)
            else:
                if len(cur) >= 4:
                    bands.append((ref_kind, cur))
                ref_kind = k; ref_params = sec.best_params; cur = [sec]
        if len(cur) >= 4:
            bands.append((ref_kind, cur))

        for bi, (kind, band) in enumerate(bands):
            path = []
            for sec in band:
                w = _section_centroid_world(sec, axis_idx)
                if w is not None:
                    path.append(w)
            if len(path) < 4:
                continue
            path = np.asarray(path, dtype=np.float64)
            # Curvature: RMS deviation of centroid path from a straight
            # line, normalized by path length.  Straight => extrude, skip.
            d = path[-1] - path[0]
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            d /= L
            proj = (path - path[0]) @ d
            closest = path[0][None, :] + proj[:, None] * d[None, :]
            rms = float(np.sqrt(np.mean(np.sum((path - closest) ** 2, axis=1))))
            curve_frac = rms / max(L, 1e-9)
            if curve_frac < min_curve_frac:
                continue  # straight -> extrude handles it
            # Mean profile params across the band.
            mean_params = list(np.mean([s.best_params for s in band], axis=0))
            m = _build_sweep_mesh(kind, mean_params, path)
            if m is None or len(m.faces) == 0:
                continue
            try:
                pv = float(abs(m.volume))
            except Exception:
                pv = 0.0
            cov = pv / gt_vol if gt_vol > 1e-9 else 0.0
            outs.append(DetectorOutput(
                program=_emit_sweep_program(kind, mean_params, path),
                score=float(min(cov, 1.0)) * 0.6 + 0.1 * min(curve_frac, 1.0),
                debug={
                    "axis": _AXIS_NAMES[axis_idx],
                    "band": bi,
                    "kind": kind,
                    "curve_frac": round(curve_frac, 4),
                    "n_sections_in_band": len(band),
                },
                mesh=m,
            ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
