"""Slab-stack detector — CADFit-style local approximation.

Instead of approximating the whole body with ONE extrude that spans the
full bbox along an axis, this detector:

  1. Samples K cross-sections perpendicular to each cardinal axis.
  2. Fits a primitive (CIRCLE / RECT / POLYGON) to each section.
  3. Groups consecutive sections that share the same primitive shape
     within ``shape_rtol`` into **slabs**.
  4. For each slab, emits a local extrude (sketch + small thickness).
  5. The fast-mesh is the UNION of all slabs.

This handles "stepped cylinder" and similar shapes where a single
extrude over-projects: each slab captures one section's local
contribution.
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
from .extrude import DetectorOutput, chain_of, emit_union_program
from .slice_fit import _build_primitive_mesh, _emit_primitive_extrude


_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _params_close(p_a, p_b, kind: str, shape_rtol: float,
                  ctr_atol: float) -> bool:
    """Compare two primitive params: same kind, similar shape, similar
    centroid (allowing some drift since centroids may have noise).
    """
    if kind == "CIRCLE":
        cx_a, cy_a, r_a = p_a
        cx_b, cy_b, r_b = p_b
        if abs(r_a - r_b) / max(abs(r_a), 1e-9) > shape_rtol:
            return False
        if math.hypot(cx_a - cx_b, cy_a - cy_b) > ctr_atol:
            return False
        return True
    if kind == "RECT":
        cx_a, cy_a, w_a, h_a, t_a = p_a
        cx_b, cy_b, w_b, h_b, t_b = p_b
        if abs(w_a - w_b) / max(abs(w_a), 1e-9) > shape_rtol:
            return False
        if abs(h_a - h_b) / max(abs(h_a), 1e-9) > shape_rtol:
            return False
        if math.hypot(cx_a - cx_b, cy_a - cy_b) > ctr_atol:
            return False
        return True
    if kind == "POLYGON":
        if len(p_a) != len(p_b):
            return False
        a = np.asarray(p_a).reshape(-1, 2)
        b = np.asarray(p_b).reshape(-1, 2)
        ra = a.max(axis=0) - a.min(axis=0)
        rb = b.max(axis=0) - b.min(axis=0)
        if any(abs(ra[i] - rb[i]) / max(abs(ra[i]), 1e-9) > shape_rtol
               for i in range(2)):
            return False
        if np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) > ctr_atol:
            return False
        return True
    return False


def _group_into_slabs(sections: list[Section], shape_rtol: float,
                      ctr_atol: float) -> list[dict]:
    """Run-length group consecutive sections with similar best primitive.

    Returns a list of slab dicts:
        {kind, params (avg), z_lo, z_hi, n_sections}
    """
    slabs: list[dict] = []
    cur_start = 0
    cur_kind = sections[0].best_kind
    cur_params = sections[0].best_params
    if cur_kind not in ("CIRCLE", "RECT", "POLYGON"):
        return []
    for i in range(1, len(sections)):
        kind_i = sections[i].best_kind
        params_i = sections[i].best_params
        same = (kind_i == cur_kind and
                _params_close(cur_params, params_i, cur_kind,
                              shape_rtol, ctr_atol))
        if not same:
            slabs.append({
                "kind": cur_kind,
                "params": cur_params,
                "z_lo": sections[cur_start].z,
                "z_hi": sections[i - 1].z,
                "n_sections": i - cur_start,
            })
            cur_start = i
            cur_kind = kind_i
            cur_params = params_i
    # Tail slab.
    slabs.append({
        "kind": cur_kind,
        "params": cur_params,
        "z_lo": sections[cur_start].z,
        "z_hi": sections[-1].z,
        "n_sections": len(sections) - cur_start,
    })
    # Drop slabs whose params don't fit (kind not in primitives).
    return [s for s in slabs
            if s["kind"] in ("CIRCLE", "RECT", "POLYGON")]


def _slab_lohi(s: dict, axis_idx: int, lo, hi):
    lo_fake = lo.copy().astype(float)
    hi_fake = hi.copy().astype(float)
    lo_fake[axis_idx] = float(s["z_lo"])
    hi_fake[axis_idx] = float(s["z_hi"])
    return lo_fake, hi_fake


def _build_slab_stack_mesh(slabs: list[dict], axis_idx: int,
                           lo, hi) -> Optional[trimesh.Trimesh]:
    """Union all slab meshes via trimesh boolean union (manifold3d)."""
    pieces = []
    for s in slabs:
        lo_fake, hi_fake = _slab_lohi(s, axis_idx, lo, hi)
        if hi_fake[axis_idx] - lo_fake[axis_idx] < 1e-6:
            continue
        m = _build_primitive_mesh(s["kind"], list(s["params"]),
                                  axis_idx, lo_fake, hi_fake)
        if m is None or len(m.faces) == 0:
            continue
        pieces.append(m)
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    try:
        return trimesh.boolean.union(pieces)
    except Exception:
        # Fallback: trimesh.util.concatenate (won't be exact but is fast).
        try:
            return trimesh.util.concatenate(pieces)
        except Exception:
            return None


def _emit_slab_stack_program(slabs: list[dict], axis_idx: int, lo, hi) -> str:
    """Emit a RENDERABLE CadQuery program: union of per-slab primitive
    extrudes (reuses slice_fit's verified axis-aligned emit)."""
    chains = []
    for s in slabs:
        lo_fake, hi_fake = _slab_lohi(s, axis_idx, lo, hi)
        if hi_fake[axis_idx] - lo_fake[axis_idx] < 1e-6:
            continue
        prog = _emit_primitive_extrude(s["kind"], list(s["params"]),
                                       axis_idx, lo_fake, hi_fake)
        ch = chain_of(prog)
        if ch:
            chains.append(ch)
    return emit_union_program(chains)


def detect_slab_stack(mesh: trimesh.Trimesh,
                      n_sections: int = 20,
                      shape_rtol: float = 0.15,
                      ctr_atol_frac: float = 0.08,
                      ) -> list[DetectorOutput]:
    """For each axis, slice K sections, group into slabs, emit a UNION
    of per-slab primitives as a single candidate.
    """
    if not _HAS_CPP_FITS:
        return []
    lo, hi = mesh.bounds
    max_extent = float(np.max(hi - lo))
    ctr_atol = ctr_atol_frac * max_extent
    outs: list[DetectorOutput] = []
    try:
        vol = float(abs(mesh.volume))
    except Exception:
        vol = 0.0

    for axis_idx in (0, 1, 2):
        secs = extract_sections(mesh, axis_idx=axis_idx,
                                n_sections=n_sections)
        if len(secs) < 3:
            continue
        slabs = _group_into_slabs(secs, shape_rtol, ctr_atol)
        if len(slabs) < 1:
            continue
        # Only fire when we actually decomposed into >= 2 slabs OR
        # one slab with a non-trivial primitive (CIRCLE/RECT) — otherwise
        # we duplicate the slice_fit candidate.
        if len(slabs) == 1 and slabs[0]["kind"] == "POLYGON":
            continue
        fast_mesh = _build_slab_stack_mesh(slabs, axis_idx, lo, hi)
        if fast_mesh is None or len(fast_mesh.faces) == 0:
            continue
        program = _emit_slab_stack_program(slabs, axis_idx, lo, hi)
        # Score by volume match.
        try:
            pred_vol = float(abs(fast_mesh.volume))
        except Exception:
            pred_vol = 0.0
        if vol > 1e-9 and pred_vol > 1e-9:
            ratio = pred_vol / vol
            score = math.exp(-abs(math.log(max(ratio, 1e-9))))
        else:
            score = 0.0
        # Bonus for capturing more slabs (richer decomposition).
        score += 0.02 * min(len(slabs), 5)
        outs.append(DetectorOutput(
            program=program,
            score=score,
            debug={
                "axis": _AXIS_NAMES[axis_idx],
                "n_slabs": len(slabs),
                "n_sections": len(secs),
                "kinds": [s["kind"] for s in slabs],
                "z_ranges": [(round(float(s["z_lo"]), 3),
                              round(float(s["z_hi"]), 3))
                             for s in slabs],
            },
            mesh=fast_mesh,
        ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
