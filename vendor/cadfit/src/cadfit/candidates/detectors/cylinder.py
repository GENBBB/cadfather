"""Cylinder detector — circular bore / boss recovery.

Mechanical parts are full of cylindrical features: through-holes, counter-
bores, bosses, pins.  The silhouette / planar_cluster / slice_fit detectors
approximate a round hole as a rectangular slot (planar_cluster) or as the
single largest mid-section (slice_fit), so a real bore is never carved cleanly.

This detector finds genuinely cylindrical CONNECTED COMPONENTS of a residual
and emits a circle-extrude for each:
  - on the CUT residual (prev \\ gt) a cylindrical component is the solid plug
    filling a hole the prediction over-filled -> cut it (the bore appears);
  - on the ADD residual (gt \\ prev) it is a missing boss / pin -> union it.
The op (union/cut) is chosen by the proposer from which residual it ran on, so
this detector only has to emit a clean circle-extrude + its fast mesh.

Precision-oriented: a component must slice to CONSISTENT circles along one axis
AND its volume must match a solid cylinder (pi r^2 h), else it is skipped -- so
we never carve a spurious round hole into a boxy residual.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, _cg
from .extrude import DetectorOutput
from .slice_fit import _slice_polygon, _emit_primitive_extrude, _build_primitive_mesh

_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _fit_axis_cylinder(comp: trimesh.Trimesh, axis_idx: int,
                       n_sections: int, r_tol: float, c_tol: float
                       ) -> Optional[tuple]:
    """Try to read `comp` as a cylinder along `axis_idx`.  Returns
    (cx, cy, r, z_lo, z_hi, support_frac) or None."""
    lo, hi = comp.bounds
    span = float(hi[axis_idx] - lo[axis_idx])
    if span <= 1e-6:
        return None
    zs = np.linspace(lo[axis_idx], hi[axis_idx], n_sections + 2)[1:-1]
    rec = []  # (z, cx, cy, r)
    for z in zs:
        coords = _slice_polygon(comp, axis_idx, float(z))
        if coords is None or len(coords) < 6:
            continue
        fits = _cg.section.fit_all(coords, -1.0)
        if not fits:
            continue
        top = fits[0]
        if str(top.kind).split(".")[-1] != "CIRCLE":
            continue
        cx, cy, r = (float(top.params[0]), float(top.params[1]),
                     float(top.params[2]))
        if r > 1e-6:
            rec.append((float(z), cx, cy, r))
    if len(rec) < max(3, int(0.6 * n_sections)):
        return None
    rec = np.asarray(rec)
    rs = rec[:, 3]; cxs = rec[:, 1]; cys = rec[:, 2]
    rmed = float(np.median(rs))
    if rmed <= 1e-6:
        return None
    cxm = float(np.median(cxs)); cym = float(np.median(cys))
    consistent = ((np.abs(rs - rmed) <= r_tol * rmed) &
                  (np.abs(cxs - cxm) <= c_tol) & (np.abs(cys - cym) <= c_tol))
    if consistent.sum() < max(3, int(0.6 * n_sections)):
        return None
    zk = rec[consistent, 0]
    return (cxm, cym, rmed, float(zk.min()), float(zk.max()),
            float(consistent.sum()) / n_sections)


def detect_cylinders(mesh: trimesh.Trimesh,
                     n_sections: int = 7,
                     r_tol: float = 0.14,
                     c_tol_frac: float = 0.08,
                     vol_ratio_lo: float = 0.6,
                     vol_ratio_hi: float = 1.5,
                     max_candidates: int = 6,
                     ) -> list[DetectorOutput]:
    if not _HAS_CPP_FITS or mesh is None or len(mesh.faces) == 0:
        return []
    full_lo, full_hi = mesh.bounds
    max_extent = float(np.max(full_hi - full_lo))
    c_tol = c_tol_frac * max_extent
    min_r = 0.015 * max_extent

    # Connected components: a clean bore/boss is its own component even when the
    # rest of the residual is messy.  only_watertight=False keeps open shells.
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        comps = [mesh]
    if not comps:
        comps = [mesh]

    outs: list[DetectorOutput] = []
    for comp in comps:
        if comp is None or len(comp.faces) < 24:
            continue
        try:
            comp_vol = float(abs(comp.volume))
        except Exception:
            comp_vol = 0.0
        best = None
        for axis_idx in (0, 1, 2):
            fit = _fit_axis_cylinder(comp, axis_idx, n_sections, r_tol, c_tol)
            if fit is None:
                continue
            cx, cy, r, z_lo, z_hi, support = fit
            if r < min_r or (z_hi - z_lo) < 0.02 * max_extent:
                continue
            cyl_vol = math.pi * r * r * (z_hi - z_lo)
            if comp_vol > 1e-9:
                ratio = comp_vol / max(cyl_vol, 1e-9)
                if not (vol_ratio_lo <= ratio <= vol_ratio_hi):
                    continue
            # score: well-supported + larger cylinders first
            sc = support * (r * r) * (z_hi - z_lo)
            if best is None or sc > best[0]:
                best = (sc, axis_idx, cx, cy, r, z_lo, z_hi, support)
        if best is None:
            continue
        sc, axis_idx, cx, cy, r, z_lo, z_hi, support = best
        lo_f = full_lo.copy().astype(float); hi_f = full_hi.copy().astype(float)
        lo_f[axis_idx] = z_lo; hi_f[axis_idx] = z_hi
        program = _emit_primitive_extrude("CIRCLE", [cx, cy, r], axis_idx, lo_f, hi_f)
        if not program:
            continue
        fast = _build_primitive_mesh("CIRCLE", [cx, cy, r], axis_idx, lo_f, hi_f)
        outs.append(DetectorOutput(
            program=program,
            score=float(sc / (max_extent ** 3 + 1e-9)),
            debug={"axis": _AXIS_NAMES[axis_idx], "radius": r,
                   "span": z_hi - z_lo, "support_frac": support,
                   "center": [round(cx, 2), round(cy, 2)]},
            mesh=fast,
        ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs[:max_candidates]
