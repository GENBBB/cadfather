"""Emit CadQuery in the CADRecode / native-Sketch-API style the VLM uses.

The VLM (CADRecode) writes every solid as

    cq.Workplane('PLANE', origin=(ox,oy,oz)).sketch()
        <primitives>            # .circle(r) / .rect(w,h) / .segment(..)/.arc(..)
        .finalize().extrude(t)  #   ... or .revolve(angle, (x1,y1), (x2,y2))

with holes expressed as inner wires (`mode='s'`).  This module produces that
exact grammar (validated to render in plain cadquery), so det blocks are
stylistically identical to VLM steps in the committed program — uniform code,
same parser, and a single placement convention.

All of these are *native cadquery* Sketch-API calls (only `attach_at` is a
cadgen monkeypatch, which we do not use), so the emitted code runs in the same
plain-cadquery render env the det path already uses.
"""
from __future__ import annotations

import math
from typing import Sequence


def _f(x) -> str:
    return f"{float(x):.4f}"


def _pt(x, y) -> str:
    return f"({_f(x)},{_f(y)})"


def base(plane: str, origin: Sequence[float]) -> str:
    """`cq.Workplane('PLANE', origin=(...))` — single-quoted plane, matches VLM."""
    o = origin
    return f"cq.Workplane('{plane}', origin=({_f(o[0])},{_f(o[1])},{_f(o[2])}))"


def _push(cx, cy) -> str:
    return "" if (abs(cx) < 1e-6 and abs(cy) < 1e-6) else f".push([{_pt(cx, cy)}])"


def circle_frag(cx, cy, r, sub: bool = False) -> str:
    m = ",mode='s'" if sub else ""
    return f"{_push(cx, cy)}.circle({_f(r)}{m})"


def rect_frag(cx, cy, w, h, sub: bool = False) -> str:
    m = ",mode='s'" if sub else ""
    return f"{_push(cx, cy)}.rect({_f(w)},{_f(h)}{m})"


def poly_frag(pts) -> str:
    """A closed polyline outer wire as a `.segment(..)…close().assemble()` chain.

    Grammar (matches cadgen): the FIRST segment carries two points (start, next),
    each subsequent segment one point, the final edge is `.close()`, then
    `.assemble()` turns the wire into a face.
    """
    pts = list(pts)
    if len(pts) < 3:
        return ""
    s = f".segment({_pt(*pts[0])},{_pt(*pts[1])})"
    for p in pts[2:]:
        s += f".segment({_pt(*p)})"
    s += ".close().assemble()"
    return s


def _holes_frag(holes) -> str:
    """Inner wires subtracted in the same sketch (mode='s')."""
    frag = ""
    for h in holes or []:
        if h[0] == "circle":
            _, cx, cy, r = h
            frag += circle_frag(cx, cy, r, sub=True)
        elif h[0] == "rect":
            _, cx, cy, w, hh = h
            frag += rect_frag(cx, cy, w, hh, sub=True)
    return frag


# ---- high-level solid emitters (return the `cq.Workplane(...)...` expression) --

def _ext_t(plane, thickness):
    """Signed extrude distance.  cadquery named-plane normals point XY->+Z,
    YZ->+X, but **XZ->-Y**, so `.extrude(t)` on an XZ workplane goes -Y.  The
    detector fast meshes (and the true geometry) always extrude along the +axis,
    so negate t for XZ -> the emitted solid spans the SAME side as the fast mesh.
    (Without this, an XZ extrude lands adjacent to the target -> IoU ~0.)"""
    return -thickness if plane == "XZ" else thickness


def circle_extrude(plane, origin, cx, cy, r, thickness) -> str:
    return f"{base(plane, origin)}.sketch(){circle_frag(cx, cy, r)}.finalize().extrude({_f(_ext_t(plane, thickness))})"


def rect_extrude(plane, origin, cx, cy, w, h, thickness, theta: float = 0.0) -> str:
    if abs(theta) < 1e-9:
        return f"{base(plane, origin)}.sketch(){rect_frag(cx, cy, w, h)}.finalize().extrude({_f(_ext_t(plane, thickness))})"
    # rotated rect -> polygon of its 4 corners
    ct, st = math.cos(theta), math.sin(theta)
    hw, hh = w * 0.5, h * 0.5
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = [(cx + ct * x - st * y, cy + st * x + ct * y) for x, y in corners]
    return poly_extrude(plane, origin, pts, thickness)


def poly_extrude(plane, origin, pts, thickness, holes=None) -> str:
    frag = poly_frag(pts)
    if not frag:
        return ""
    frag += _holes_frag(holes)
    return f"{base(plane, origin)}.sketch(){frag}.finalize().extrude({_f(_ext_t(plane, thickness))})"


def poly_revolve(plane, origin, pts, angle: float = 360,
                 axis=((0.0, 0.0), (0.0, 1.0)), holes=None) -> str:
    frag = poly_frag(pts)
    if not frag:
        return ""
    frag += _holes_frag(holes)
    a0, a1 = axis
    ang = int(angle) if float(angle).is_integer() else round(float(angle), 2)
    return f"{base(plane, origin)}.sketch(){frag}.finalize().revolve({ang}, {_pt(*a0)}, {_pt(*a1)})"
