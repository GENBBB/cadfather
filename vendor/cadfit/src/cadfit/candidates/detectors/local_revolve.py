"""Local-revolve detector — CADFit-style local axisymmetric bands.

Unlike the global ``revolve`` detector (which requires the WHOLE body
to pass a 4-angle axisymmetry test), this finds the axial BANDS that
are locally round and revolves only those.  A "cylinder + square base"
part fails global revolve but yields a clean local_revolve over the
cylindrical band; the base is left to the residual.

Algorithm
---------
For each candidate axis:
  1. Slice into N sections perpendicular to the axis.
  2. Mark each section "round" iff its best primitive fit is CIRCLE.
  3. Group consecutive round sections into bands.
  4. For each band, build the (r, z) profile from the per-section
     circle radii and revolve it (partial axial span).  This handles
     cylinders, cones, and stepped/tapered round bands.
  5. Emit one candidate per band + a UNION of all bands.

Fast-mesh only (reuses ``revolve._build_revolve_mesh``); no cadquery.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, extract_sections
from .extrude import DetectorOutput, chain_of, emit_union_program
from .revolve import _build_revolve_mesh, _emit_revolve_program


_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _round_bands(sections, axis_idx: int, center: np.ndarray,
                 max_radius_jump: float):
    """Group consecutive CIRCLE sections into bands.

    Returns a list of bands, each a list of (r, z) tuples where
    z is measured relative to ``center[axis_idx]``.
    """
    bands = []
    cur = []
    prev_r = None
    cz = float(center[axis_idx])
    for sec in sections:
        is_round = (sec.best_kind == "CIRCLE" and len(sec.best_params) >= 3)
        if is_round:
            r = float(sec.best_params[2])
            z = float(sec.z) - cz
            # Break the band if the radius jumps too sharply (a real
            # step is fine within one revolve profile, but a huge jump
            # usually means a different feature).
            if prev_r is not None and abs(r - prev_r) > max_radius_jump:
                if len(cur) >= 2:
                    bands.append(cur)
                cur = []
            cur.append((r, z))
            prev_r = r
        else:
            if len(cur) >= 2:
                bands.append(cur)
            cur = []
            prev_r = None
    if len(cur) >= 2:
        bands.append(cur)
    return bands


def _band_profile(band) -> np.ndarray:
    """Build a closed (r, z) revolve profile for one band.

    Walk up the outer radii, then close back down to the axis at the
    two ends (a small eps so OCCT/trimesh don't choke on r=0 edges).
    """
    rs = [b[0] for b in band]
    zs = [b[1] for b in band]
    z_lo, z_hi = zs[0], zs[-1]
    eps = 1e-4 * max(abs(z_hi - z_lo), 1.0)
    profile = [(eps, z_lo)]
    for r, z in band:
        profile.append((max(r, eps), z))
    profile.append((eps, z_hi))
    return np.asarray(profile, dtype=np.float64)


def detect_local_revolves(mesh: trimesh.Trimesh,
                          n_sections: int = 20,
                          max_radius_jump_frac: float = 0.35,
                          ) -> list[DetectorOutput]:
    if not _HAS_CPP_FITS:
        return []
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    max_extent = float(np.max(mesh.extents))
    max_radius_jump = max_radius_jump_frac * max_extent
    try:
        gt_vol = float(abs(mesh.volume))
    except Exception:
        gt_vol = 0.0

    outs: list[DetectorOutput] = []
    for axis_idx in (0, 1, 2):
        secs = extract_sections(mesh, axis_idx=axis_idx,
                                n_sections=n_sections)
        if len(secs) < 3:
            continue
        bands = _round_bands(secs, axis_idx, center, max_radius_jump)
        if not bands:
            continue
        pieces = []
        chains = []
        for bi, band in enumerate(bands):
            rz = _band_profile(band)
            m = _build_revolve_mesh(rz, axis_idx, center)
            if m is None or len(m.faces) == 0:
                continue
            pieces.append(m)
            ch = chain_of(_emit_revolve_program(rz, axis_idx, center))
            if ch:
                chains.append(ch)
            # Score by how much volume this band covers.
            try:
                pv = float(abs(m.volume))
            except Exception:
                pv = 0.0
            cov = pv / gt_vol if gt_vol > 1e-9 else 0.0
            outs.append(DetectorOutput(
                program=emit_union_program([ch]) if ch else "",
                score=float(min(cov, 1.0)) * 0.6,
                debug={
                    "axis": _AXIS_NAMES[axis_idx],
                    "band": bi,
                    "n_bands": len(bands),
                    "n_sections_in_band": len(band),
                    "z_lo": round(band[0][1], 3),
                    "z_hi": round(band[-1][1], 3),
                },
                mesh=m,
            ))
        # Union of all bands on this axis.
        if len(pieces) >= 2:
            try:
                um = trimesh.boolean.union(pieces)
            except Exception:
                try:
                    um = trimesh.util.concatenate(pieces)
                except Exception:
                    um = None
            if um is not None and len(um.faces) > 0:
                outs.append(DetectorOutput(
                    program=emit_union_program(chains),
                    score=0.55,
                    debug={"axis": _AXIS_NAMES[axis_idx],
                           "kind": "union", "n_bands": len(pieces)},
                    mesh=um,
                ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
