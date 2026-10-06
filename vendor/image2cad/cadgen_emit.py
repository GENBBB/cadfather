"""Emit det (CADFit) reconstructions in the cadgen 12-op format.

Maps det's (side, Candidate) ops onto the cadgen entry points the 12op model uses:
    union extrude  -> r = extrude(r, (px,py,pz), 'XY|YZ|ZX', "sketch()...finalize()", h, False)
                      (point_on_surface=False -> absolute placement, exact for det ops)
    cut            -> r = hole(r, (px,py,pz), PLANE, sketch, depth)   (surface-snapped entry)
    revolve prim   -> r = revolve(r, ..., angle, axis) when the primitive program is convertible
Profile interiors are emitted inside the same sketch with mode='s' (native cadquery Sketch API,
builds in the same cadgen pipeline).

Not every det op is expressible (sweep/loft primitives, oriented extrudes): emit_cadgen returns
(code, coverage) where coverage < 1.0 flags partial conversion — callers validate the built mesh
against the det reference STL and fall back to the VLM-format program if IoU drops.
"""
from __future__ import annotations

import numpy as np

PLANE_OF_AXIS = {2: "XY", 0: "YZ", 1: "ZX"}
INPLANE = {"XY": (0, 1), "YZ": (1, 2), "ZX": (2, 0)}   # cadquery local (x,y) -> world components


def _f(v):
    return f"{float(v):.4f}"


def _ring_world(profile, ring, t):
    """3D world coords of a profile ring at parameter t (Nx3)."""
    co = np.asarray(ring.coords, float)
    if len(co) and np.allclose(co[0], co[-1]):
        co = co[:-1]
    P = np.column_stack([co, np.full(len(co), t), np.ones(len(co))])
    return (profile.to_3d @ P.T).T[:, :3]


def _axis_of(profile) -> int | None:
    """World axis the profile extrudes along (None if the frame is rotated)."""
    n = np.asarray(profile.to_3d[:3, 2], float)
    n = n / (np.linalg.norm(n) + 1e-12)
    ax = int(np.argmax(np.abs(n)))
    return ax if abs(abs(n[ax]) - 1.0) < 1e-3 else None


def _sketch_of(profile, axis, t):
    """cadgen sketch string for the profile's rings, in PLANE_OF_AXIS[axis] local coords."""
    plane = PLANE_OF_AXIS[axis]
    i, j = INPLANE[plane]
    parts = []
    for ridx, ring in enumerate([profile.polygon.exterior] + list(profile.polygon.interiors)):
        W = _ring_world(profile, ring, t)
        pts = [(w[i], w[j]) for w in W]
        if len(pts) < 3:
            continue
        mode = "" if ridx == 0 else ", mode='s'"
        circ = _fit_circle(pts)
        if circ is not None:
            cx, cy, r = circ
            parts.append(f".push([({_f(cx)},{_f(cy)})]).circle({_f(r)}{mode})")
        else:
            # native Sketch polygon: auto-closed ring, robust for exterior + mode='s' interiors.
            # push([(0,0)]) first: a preceding circle's push([center]) PERSISTS and would
            # translate this ring's absolute coords.
            pts_s = ",".join(f"({_f(x)},{_f(y)})" for x, y in pts)
            parts.append(f".push([(0.0,0.0)]).polygon([{pts_s},({_f(pts[0][0])},{_f(pts[0][1])})]{mode})")
    if not parts:
        return None
    return "sketch()" + "".join(parts) + ".finalize()"


def _fit_circle(pts, tol=0.6):
    P = np.asarray(pts, float)
    if len(P) < 8:
        return None
    c = P.mean(axis=0)
    r = np.linalg.norm(P - c, axis=1)
    if r.mean() < 1e-6 or (r.max() - r.min()) > tol:
        return None
    return float(c[0]), float(c[1]), float(r.mean())


def _extrude_op(c, first):
    axis = _axis_of(c.profile)
    if axis is None:
        return None
    t0, t1 = (c.t0, c.t1) if c.t1 >= c.t0 else (c.t1, c.t0)
    n = np.asarray(c.profile.to_3d[:3, 2], float)
    sgn = 1.0 if n[axis] > 0 else -1.0
    lo_t, hi_t = (t0, t1) if sgn > 0 else (t1, t0)   # lo_t maps to the smaller world coord
    sk = _sketch_of(c.profile, axis, lo_t)
    if sk is None:
        return None
    base_w = (c.profile.to_3d @ np.array([0.0, 0.0, lo_t, 1.0]))[:3]
    h = abs(t1 - t0)
    p = [0.0, 0.0, 0.0]
    p[axis] = float(base_w[axis])
    plane = PLANE_OF_AXIS[axis]
    r_arg = "None" if first else "r"
    return (f"r=extrude({r_arg},({_f(p[0])},{_f(p[1])},{_f(p[2])}),'{plane}',"
            f"\"{sk}\",{_f(h)},False)")


def _cut_op(c):
    """cut -> hole(): enter at the cut's world-max end of its slab (over-pierce handles exit)."""
    axis = _axis_of(c.profile) if getattr(c, "profile", None) is not None else None
    if axis is None:
        return None
    t0, t1 = (c.t0, c.t1) if c.t1 >= c.t0 else (c.t1, c.t0)
    n = np.asarray(c.profile.to_3d[:3, 2], float)
    sgn = 1.0 if n[axis] > 0 else -1.0
    hi_t = t1 if sgn > 0 else t0                      # world-max end of the slab
    sk = _sketch_of(c.profile, axis, hi_t)
    if sk is None:
        return None
    top_w = (c.profile.to_3d @ np.array([0.0, 0.0, hi_t, 1.0]))[:3]
    # entry point: centroid of the profile at the world-max plane
    W = _ring_world(c.profile, c.profile.polygon.exterior, hi_t)
    cen = W.mean(axis=0)
    cen[axis] = float(top_w[axis])
    depth = -abs(c.t1 - c.t0) - 1.0                   # cut downward through the slab (+1 pierce)
    plane = PLANE_OF_AXIS[axis]
    return (f"r=hole(r,({_f(cen[0])},{_f(cen[1])},{_f(cen[2])}),'{plane}',"
            f"\"{sk}\",{_f(depth)})")


import re as _re

# det wp local-Y = revolve axis: 'YX'->X, 'XY'->Y, 'XZ'->Z.
# cadgen plane for that axis + the world indices of (radial, axial) components.
_REV_MAP = {"YX": ("ZX", "X", 2, 0), "XY": ("XY", "Y", 0, 1), "XZ": ("YZ", "Z", 0, 2)}


def _revolve_op(c, first):
    """Convert a det revolve primitive (CADRecode program) to a cadgen revolve call.
    Only for the BASE op: cadgen revolve() surface-snaps `point` when r is not None."""
    if not first:
        return None
    prog = getattr(c, "program", "") or ""
    m = _re.search(r"cq\.Workplane\('(\w+)',\s*origin=\(([-\d.,\s]+)\)\)", prog)
    if not m or m.group(1) not in _REV_MAP:
        return None
    plane, axischar, ri, ai = _REV_MAP[m.group(1)]
    c3 = [float(v) for v in m.group(2).split(",")]
    pts = _re.findall(r"\.segment\(\(([-\d.]+),([-\d.]+)\)(?:,\(([-\d.]+),([-\d.]+)\))?\)", prog)
    prof = []
    for a, b, x, y in pts:
        prof.append((float(a), float(b)))
        if x:
            prof.append((float(x), float(y)))
    # first .segment(p0,p1): p0 comes first, later ones append their single point
    if len(pts) and pts[0][2]:
        prof = [(float(pts[0][0]), float(pts[0][1])), (float(pts[0][2]), float(pts[0][3]))] + prof[2:]
    if len(prof) < 3 or ".revolve(360" not in prog:
        return None
    cr, ca = c3[ri], c3[ai]
    pts_s = ",".join(f"({_f(cr + r)},{_f(ca + z)})" for r, z in prof)
    first_pt = f"({_f(cr + prof[0][0])},{_f(ca + prof[0][1])})"
    sk = f"sketch().push([(0.0,0.0)]).polygon([{pts_s},{first_pt}]).finalize()"
    return (f"r=revolve(None,({_f(c3[0])},{_f(c3[1])},{_f(c3[2])}),'{plane}',"
            f"\"{sk}\",360,'{axischar}')")


HEADER = ("import cadquery as cq\n"
          "from cadgen.selectors import PointOnEdgeSelector\n"
          "from cadgen.extrude import extrude\n"
          "from cadgen.shell import shell\n"
          "from cadgen.hole import hole\n"
          "from cadgen.revolve import revolve\n"
          "from cadgen.orto_cut import orto_cut\n"
          "from cadgen.sweep import sweep\n"
          "from cadgen.gear import gear\n"
          "r = None\n")


def emit_cadgen(ops):
    """det ops -> (cadgen_program | None, coverage 0..1).

    coverage = emitted ops / total ops. Callers should render-validate and keep the
    VLM-format emission when coverage < 1 or the built mesh diverges.
    """
    lines, total, done = [], 0, 0
    first = True
    for side, c in ops:
        total += 1
        kind = getattr(c, "kind", None)
        op = None
        if kind == "revolve":
            op = _revolve_op(c, first)
        elif kind in ("sweep", "loft", "planar_cut"):
            op = None                                  # primitive: not yet convertible
        elif getattr(c, "profile", None) is not None:
            if side == "union":
                op = _extrude_op(c, first)
            elif not first:
                op = _cut_op(c)
        if op is None:
            continue
        lines.append(op)
        done += 1
        first = False
    if not lines or lines[0].startswith("r=hole"):
        return None, 0.0
    return HEADER + "\n".join(lines) + "\n", done / max(total, 1)
