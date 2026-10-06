"""Local-loft detector — CADFit-style local tapered bands.

Groups consecutive cross-sections that share a primitive KIND but whose
params drift smoothly into bands, then lofts WITHIN each band (local),
rather than lofting the whole body's endpoints.  Good for h-loft shapes
whose section sequence is non-monotonic (a global endpoint-loft misses
the middle bulge).

Fast-mesh only (reuses ``loft._build_loft_mesh``); no cadquery.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, extract_sections
from .extrude import DetectorOutput, chain_of, emit_union_program
from .loft import _build_loft_mesh, _emit_multi_loft


_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _kind_bands(sections, min_len: int = 2):
    """Group consecutive sections sharing the same best_kind (and, for
    POLYGON, the same vertex count) into bands of length >= min_len.
    """
    bands = []
    cur = []
    cur_kind = None
    cur_nparam = None
    for sec in sections:
        k = sec.best_kind
        if k not in ("CIRCLE", "RECT", "POLYGON"):
            if len(cur) >= min_len:
                bands.append((cur_kind, cur))
            cur = []; cur_kind = None; cur_nparam = None
            continue
        np_ = len(sec.best_params)
        same = (k == cur_kind and
                (k != "POLYGON" or np_ == cur_nparam))
        if not same and cur:
            if len(cur) >= min_len:
                bands.append((cur_kind, cur))
            cur = []
        cur.append(sec)
        cur_kind = k
        cur_nparam = np_
    if len(cur) >= min_len:
        bands.append((cur_kind, cur))
    return bands


def detect_local_lofts(mesh: trimesh.Trimesh,
                       n_sections: int = 16,
                       ) -> list[DetectorOutput]:
    if not _HAS_CPP_FITS:
        return []
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
        bands = _kind_bands(secs, min_len=2)
        if not bands:
            continue
        pieces = []
        chains = []
        for bi, (kind, band) in enumerate(bands):
            m = _build_loft_mesh(band, kind, axis_idx)
            if m is None or len(m.faces) == 0:
                continue
            pieces.append(m)
            ch = chain_of(_emit_multi_loft(band, kind, axis_idx))
            if ch:
                chains.append(ch)
            try:
                pv = float(abs(m.volume))
            except Exception:
                pv = 0.0
            cov = pv / gt_vol if gt_vol > 1e-9 else 0.0
            outs.append(DetectorOutput(
                program=emit_union_program([ch]) if ch else "",
                score=float(min(cov, 1.0)) * 0.55,
                debug={
                    "axis": _AXIS_NAMES[axis_idx],
                    "band": bi,
                    "n_bands": len(bands),
                    "kind": kind,
                    "n_sections_in_band": len(band),
                },
                mesh=m,
            ))
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
                    score=0.5,
                    debug={"axis": _AXIS_NAMES[axis_idx],
                           "kind": "union", "n_bands": len(pieces)},
                    mesh=um,
                ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
