"""Loft detector — non-constant section trajectory → ``.loft()``.

Algorithm
---------
Per candidate axis:
  1. Use ``section_analyzer.extract_sections`` to get K cross-sections.
  2. Call ``classify_trajectory``.  Only emit if the hint is ``loft``
     (same primitive kind across sections, params drift).
  3. Take the FIRST and LAST sections (endpoint profiles); emit a
     CadQuery loft between them.

Emission patterns by kind:
  - CIRCLE endpoints: emit two circles on stacked workplanes, .loft().
  - RECT endpoints:   emit two centered rectangles, .loft().
  - POLYGON endpoints: emit the two simplified polylines, .loft().

Score = -confidence  (so lower = better, matching the rest of the pkg).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import (
    Section,
    _HAS_CPP_FITS,
    classify_trajectory,
    extract_sections,
)
from .extrude import DetectorOutput


_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}

# Workplane to use for sketching cross-sections perpendicular to each axis.
_AXIS_TO_WP = {0: "YZ", 1: "XZ", 2: "XY"}


def _section_chain(sec: Section, kind: str, ref_cx: float, ref_cy: float
                   ) -> str:
    """Build the .center(...).circle/rect/moveTo chain for one section,
    given a reference (cx, cy) offset (so the workplane's local center
    advances cumulatively under cq's .workplane(offset=...) semantics).

    Returns the chain as a string starting with .center / .moveTo and
    ending with the closed wire (for POLYGON) or the primitive call.
    """
    p = sec.best_params
    if kind == "CIRCLE":
        cx, cy, r = p
        return f'.center({cx - ref_cx:.4f}, {cy - ref_cy:.4f}).circle({r:.4f})'
    if kind == "RECT":
        cx, cy, w, h, _th = p
        return f'.center({cx - ref_cx:.4f}, {cy - ref_cy:.4f}).rect({w:.4f}, {h:.4f})'
    if kind == "POLYGON":
        pts = list(zip(p[0::2], p[1::2]))
        if len(pts) < 3:
            return ""
        c = f'.moveTo({pts[0][0]:.4f}, {pts[0][1]:.4f})'
        for x, y in pts[1:]:
            c += f'.lineTo({x:.4f}, {y:.4f})'
        c += '.close()'
        return c
    return ""


def _section_center(sec: Section, kind: str) -> tuple[float, float]:
    p = sec.best_params
    if kind in ("CIRCLE", "RECT"):
        return float(p[0]), float(p[1])
    if kind == "POLYGON":
        xs = p[0::2]; ys = p[1::2]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
    return 0.0, 0.0


def _emit_multi_loft(sections: list[Section], kind: str, axis_idx: int) -> str:
    """Emit a loft chain across ALL passed sections (>= 2).  Each
    section becomes its own workplane on a stacked offset.

    For POLYGON sections we DO NOT reuse .center() offsets (each section
    has its own moveTo coordinates).  For CIRCLE/RECT we accumulate the
    .center(dx, dy) offsets so the per-workplane center matches the
    section's centroid.
    """
    wp = _AXIS_TO_WP[axis_idx]
    if len(sections) < 2:
        return ""
    # CADRecode loft format (matches cadgen.loft.Loft.to_string): a single
    # chain  cq.Workplane('PLANE', origin=(...)).center(..).circle(r)/.rect(w,h)
    #   [.workplane(offset=dz).center(..).circle/rect ...] .loft()
    # Section 0 sits on the base plane (its axial coord folded into the
    # workplane origin); later sections step along the normal via
    # .workplane(offset=).  One statement, single-quoted plane, plain .loft().
    o = [0.0, 0.0, 0.0]
    o[axis_idx] = float(sections[0].z)
    chain = f"cq.Workplane('{wp}', origin=({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}))"
    ref_cx = ref_cy = 0.0
    prev_z = float(sections[0].z)
    for i, sec in enumerate(sections):
        if i > 0:
            dz = float(sec.z) - prev_z
            prev_z = float(sec.z)
            chain += f".workplane(offset={dz:.4f})"
        if kind == "POLYGON":
            sc = _section_chain(sec, kind, 0.0, 0.0)
        else:
            sc = _section_chain(sec, kind, ref_cx, ref_cy)
            cx, cy = _section_center(sec, kind)
            ref_cx, ref_cy = cx, cy
        if not sc:
            return ""
        chain += sc
    chain += ".loft()"
    return "import cadquery as cq\nresult = " + chain + "\n"


def detect_loft(mesh: trimesh.Trimesh,
                n_sections: int = 8,
                ) -> list[DetectorOutput]:
    """For each axis, classify trajectory; emit a multi-section loft
    block when the classifier returns ``loft``.

    Emits two variants per axis:
      - 2-section loft (endpoints only) — fewest params, cleanest CAD
      - N-section loft (all valid sections) — captures intermediate
        bulges that endpoint-loft misses
    The proposer then picks the actual best by IoU.
    """
    if not _HAS_CPP_FITS:
        return []
    outs: list[DetectorOutput] = []
    for axis_idx in (0, 1, 2):
        secs = extract_sections(mesh, axis_idx=axis_idx, n_sections=n_sections)
        if len(secs) < 2:
            continue
        hint = classify_trajectory(secs)
        if hint.op != "loft":
            continue
        kind = secs[0].best_kind
        if kind not in ("CIRCLE", "RECT", "POLYGON"):
            continue
        # Filter sections that match the dominant kind (POLYGON loft is
        # only valid when all sections have the SAME vertex count — drop
        # mismatched ones).
        if kind == "POLYGON":
            n0 = len(secs[0].best_params)
            kept = [s for s in secs if s.best_kind == kind
                    and len(s.best_params) == n0]
        else:
            kept = [s for s in secs if s.best_kind == kind]
        if len(kept) < 2:
            continue

        variants: list[tuple[list[Section], str]] = []
        # 2-section loft (endpoints)
        variants.append(([kept[0], kept[-1]], "2sec"))
        # N-section loft (all sections)
        if len(kept) > 2:
            variants.append((kept, "Nsec"))

        for sec_list, label in variants:
            program = _emit_multi_loft(sec_list, kind, axis_idx)
            if not program:
                continue
            fast_mesh = _build_loft_mesh(sec_list, kind, axis_idx)
            # N-section lofts get a small confidence bonus because they
            # capture intermediate sections; endpoint lofts get a small
            # parsimony bonus.
            base = float(hint.confidence)
            bonus = 0.05 if label == "Nsec" else 0.0
            outs.append(DetectorOutput(
                program=program,
                score=base + bonus,
                debug={
                    "axis": _AXIS_NAMES[axis_idx],
                    "section_kind": kind,
                    "n_sections": len(sec_list),
                    "variant": label,
                    "confidence": hint.confidence,
                },
                mesh=fast_mesh,
            ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs


def _section_polygon_2d(sec: "Section", kind: str, n_circle: int = 48
                        ) -> Optional[list]:
    """Return a list of (x, y) tuples representing the section's 2D
    perimeter, sampled at uniform parameter for CIRCLE/RECT (so all
    sections share the same vertex count -- required by triangulate-
    between-rings loft).
    """
    import math as _math
    p = sec.best_params
    if kind == "CIRCLE":
        cx, cy, r = p
        return [(cx + r * _math.cos(2 * _math.pi * i / n_circle),
                 cy + r * _math.sin(2 * _math.pi * i / n_circle))
                for i in range(n_circle)]
    if kind == "RECT":
        cx, cy, w, h, th = p
        ct = _math.cos(th); st = _math.sin(th)
        hw, hh = w * 0.5, h * 0.5
        # Sample each side at n_circle/4 points so the count matches CIRCLE
        # and so multi-section RECT lofts have consistent vertex topology.
        per_side = max(2, n_circle // 4)
        pts = []
        sides = [(-hw, -hh, hw, -hh),
                 (hw, -hh, hw, hh),
                 (hw, hh, -hw, hh),
                 (-hw, hh, -hw, -hh)]
        for x0, y0, x1, y1 in sides:
            for i in range(per_side):
                t = i / per_side
                lx = x0 + (x1 - x0) * t
                ly = y0 + (y1 - y0) * t
                pts.append((cx + ct * lx - st * ly,
                            cy + st * lx + ct * ly))
        return pts
    if kind == "POLYGON":
        pts2 = list(zip(p[0::2], p[1::2]))
        return [(float(x), float(y)) for x, y in pts2]
    return None


def _build_loft_mesh(sections: "list[Section]", kind: str, axis_idx: int
                     ) -> Optional[trimesh.Trimesh]:
    """Build a Trimesh by triangulating between section rings.

    For each pair (i, i+1) of sections, connect their rings with a
    strip of quads.  Cap the two endpoint sections with a fan
    triangulation.  Each ring is sampled at the SAME vertex count
    (per `_section_polygon_2d`) so the side strip triangulates cleanly.

    Bypasses cadquery's loft entirely -- 1000x faster.
    """
    try:
        if len(sections) < 2:
            return None
        rings_2d = [_section_polygon_2d(s, kind) for s in sections]
        if any(r is None for r in rings_2d):
            return None
        n_v = len(rings_2d[0])
        if any(len(r) != n_v for r in rings_2d):
            return None
        # Assemble 3D vertices ring-by-ring; insert section.z as the
        # axial coordinate.
        verts = []
        for ring, sec in zip(rings_2d, sections):
            z = float(sec.z)
            for (x, y) in ring:
                if axis_idx == 0:
                    verts.append((z, float(x), float(y)))
                elif axis_idx == 1:
                    verts.append((float(x), z, float(y)))
                else:
                    verts.append((float(x), float(y), z))
        verts = np.asarray(verts, dtype=np.float64)

        faces = []
        # Side strips between consecutive rings.
        n_rings = len(rings_2d)
        for i in range(n_rings - 1):
            base_a = i * n_v
            base_b = (i + 1) * n_v
            for j in range(n_v):
                j2 = (j + 1) % n_v
                a0 = base_a + j;  a1 = base_a + j2
                b0 = base_b + j;  b1 = base_b + j2
                # Two triangles per quad, oriented with outward normal
                # assuming sections wind CCW in local 2D.
                faces.append((a0, b0, a1))
                faces.append((a1, b0, b1))
        # End caps: fan triangulation at first and last ring.
        for i in range(1, n_v - 1):
            # First ring: reverse winding so cap faces outward (away).
            faces.append((0, i + 1, i))
            # Last ring.
            base = (n_rings - 1) * n_v
            faces.append((base, base + i, base + i + 1))
        m3 = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces),
                             process=True)
        if m3 is None or len(m3.faces) == 0:
            return None
        # Detect inverted winding via volume sign; flip if negative.
        try:
            if float(m3.volume) < 0:
                m3.invert()
        except Exception:
            pass
        return m3
    except Exception:
        return None
