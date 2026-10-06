"""
CadQuery script parser: extracts numerical parameters and builds a
tree description for the C++ optimizer backend.

Supports CadQuery API operations:
  .rect(w, h)           -> sketch rect (centered)
  .circle(r)            -> sketch circle (centered at workplane origin)
  .sketch().slot(l, w, angle=0).finalize() -> sketch slot/stadium (Sketch API)
  .polyline(pts).close()-> sketch polygon
  .moveTo(x,y).lineTo(x,y)...close() -> sketch polygon
  .spline(pts)          -> spline through points (polyline approx)
  .extrude(d)           -> extrude sketch along Z
  .revolve()            -> revolve sketch around Y axis
  .loft()               -> loft between two cross-section profiles
  .sweep(path)          -> sweep profile along a path
  .cut(child)           -> boolean subtract
  .union(child)         -> boolean union
  .intersect(child)     -> boolean intersect
  .box(l, w, h)         -> 3D box centered at origin
  .cylinder(h, r)       -> 3D cylinder along Z
  .sphere(r)            -> 3D sphere at origin
  .hole(d) / .hole(d,depth)  -> subtract cylinder (through-hole)
  .cboreHole(d, cbd, cbdepth) -> counterbore hole
  .shell(t)             -> shell operation
  .offset2D(amount)     -> offset 2D sketch boundary
  .fillet(r)            -> fillet (smooth edges)
  .chamfer(s)           -> chamfer (bevel edges)
  .translate((x,y,z))   -> translate 3D
  .rotate((ax,ay,az),(bx,by,bz), angle) -> simplified rotation
  .mirror("XY"|"XZ"|"YZ") -> mirror
  .edges(sel) / .faces(sel) -> selectors for subsequent ops
"""

import ast
import copy
import re
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple


@dataclass
class ParamSlot:
    """A numerical parameter extracted from the CadQuery code."""
    value: float
    line: int
    col: int
    end_col: int


@dataclass
class ParseResult:
    tree_desc: dict                # Tree description for C++ backend
    params: List[float]            # Flat parameter vector (DFS order matching tree_desc)
    param_slots: List[ParamSlot]   # Source locations for write-back
    param_names: List[str]         # Human-readable names
    tied_slots: List[Tuple[int, int]] = field(default_factory=list)
    # Closed-loop polygon pairs: (master_index, follower_index) — the
    # follower's param value should be copied from master after Adam
    # optimisation (otherwise Adam can move the first and last vertices
    # of a closed polygon outline apart, opening the loop and producing
    # an invalid wire that cadquery's extrude rejects).
    # Per-arc write-back metadata (arcpoly profiles): recompute each arc's
    # through-point from the optimized endpoints + frozen r_s/side.
    arc_writeback: List[dict] = field(default_factory=list)
    # Revolve profiles: {"kind": "poly" (idx = radial coordinates of the
    # polygon/arcpoly vertices, then their axial ones) | "rect" (idx =
    # [centre, half-size] across the
    # axis), "axis": axis_offset, "side": +1/-1}.  A profile that crosses the
    # axis makes OCC fail with `BRep_API: command not done`, so
    # write-back projects these params back to the profile's side.
    revolve_bounds: List[dict] = field(default_factory=list)
    # Sketch polygons: per polygon, [vx0, vy0, vx1, vy1, ...] param indices.
    # Vertices moved one by one fold a contour into self-intersection and OCC
    # builds an invalid body; write-back pulls such a polygon back.
    polygon_bounds: List[List[int]] = field(default_factory=list)
    # Source literals with no param of their own that repeat param idx: the
    # closing copy of a polygon() first vertex, dropped by the parser.  Written
    # back with that param's value, or the contour opens in the code.
    alias_slots: List[Tuple[int, "ParamSlot"]] = field(default_factory=list)


def _extract_num(node: ast.AST) -> Optional[float]:
    """Extract a numeric literal from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _extract_num(node.operand)
        if v is not None:
            return -v
    return None


def _arc_to_points(sx, sy, mx, my, ex, ey, n_seg=8):
    """Approximate a three-point arc with n_seg line segments.

    Given start (sx,sy), midpoint (mx,my), endpoint (ex,ey) on a circular arc,
    computes the circle center and samples n_seg intermediate points along the arc.
    Returns list of (x, y) tuples (excluding start, including end).
    """
    import math
    # Compute circumscribed circle center from 3 points
    ax, ay = sx, sy
    bx, by = mx, my
    cx, cy = ex, ey
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        # Degenerate: points are collinear, return straight line
        return [(ex, ey)]
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) +
          (cx * cx + cy * cy) * (ay - by)) / D
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) +
          (cx * cx + cy * cy) * (bx - ax)) / D
    r = math.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)

    # Compute angles
    a_start = math.atan2(sy - uy, sx - ux)
    a_mid = math.atan2(my - uy, mx - ux)
    a_end = math.atan2(ey - uy, ex - ux)

    # Determine arc direction: start → mid → end
    def _normalize_angle(a, ref):
        while a - ref > math.pi:
            a -= 2 * math.pi
        while a - ref < -math.pi:
            a += 2 * math.pi
        return a

    a_mid = _normalize_angle(a_mid, a_start)
    a_end = _normalize_angle(a_end, a_start)

    # Ensure mid is between start and end
    if (a_end - a_start) > 0:
        if a_mid < a_start or a_mid > a_end:
            a_end -= 2 * math.pi
    else:
        if a_mid > a_start or a_mid < a_end:
            a_end += 2 * math.pi

    points = []
    for i in range(1, n_seg + 1):
        t = i / n_seg
        a = a_start + t * (a_end - a_start)
        px = ux + r * math.cos(a)
        py = uy + r * math.sin(a)
        points.append((px, py))
    return points


def _arc_rs_side(ax, ay, mx, my, bx, by):
    """For a 3-point arc start=(ax,ay) through=(mx,my) end=(bx,by), return
    (r_s, side) matching the C++ arc SDF parameterization (radius = 0.5*chord + r_s,
    centre on `side`*perp(chord)).  Returns None if the points are ~collinear (caller
    should fall back to a line segment)."""
    import math
    D = 2.0 * (ax * (by - my) + bx * (my - ay) + mx * (ay - by))
    chord = math.hypot(bx - ax, by - ay)
    if abs(D) < 1e-9 or chord < 1e-9:
        return None
    ux = ((ax*ax + ay*ay) * (by - my) + (bx*bx + by*by) * (my - ay)
          + (mx*mx + my*my) * (ay - by)) / D
    uy = ((ax*ax + ay*ay) * (mx - bx) + (bx*bx + by*by) * (ax - mx)
          + (mx*mx + my*my) * (bx - ax)) / D
    r = math.hypot(ax - ux, ay - uy)
    r_s = r - 0.5 * chord
    nx, ny = -(by - ay), (bx - ax)
    nl = math.hypot(nx, ny)
    nx, ny = nx / nl, ny / nl
    mcx, mcy = 0.5 * (ax + bx), 0.5 * (ay + by)
    through_dot = (mx - mcx) * nx + (my - mcy) * ny
    center_dot = (ux - mcx) * nx + (uy - mcy) * ny
    # MAJOR arc (>180 deg): the through-point lies on the SAME side of the chord
    # as the circumcentre (its apex is at r+h, the far point of the circle).  The
    # C++ arc SDF handles this via the `major` flag (complementary A,B order +
    # far apex); we just report it here.
    major = (through_dot * center_dot) > 1e-9
    # `side` places the circumcentre: c = mid + sign(side)*h*n.  For a MINOR arc
    # the centre is OPPOSITE the through-point across the chord; for a MAJOR arc
    # it's on the SAME side.  Drive it off the through-point (stable, always well
    # separated from the chord — unlike the circumcentre for ~semicircles) and
    # flip for minor.
    s = 1.0 if through_dot >= 0 else -1.0
    side = s if major else -s
    return (r_s, side, major)


def _arc_through_point(ax, ay, bx, by, r_s, side, major=False):
    """Inverse of _arc_rs_side: given optimized endpoints + frozen (r_s, side,
    major), return a valid through-point (apex) for re-emitting .arc(p1, thru, p3).
    Minor arc apex = mid - inv*(r-h)*n (opposite the centre); major arc apex =
    mid + inv*(r+h)*n (the far point of the circle, same side as the centre)."""
    import math
    chord = math.hypot(bx - ax, by - ay)
    if chord < 1e-9:
        return (mx_default := (0.5*(ax+bx), 0.5*(ay+by)))[0], mx_default[1]
    r = 0.5 * chord + r_s
    h2 = r * r - 0.25 * chord * chord
    h = math.sqrt(h2) if h2 > 0 else 0.0
    nx, ny = -(by - ay), (bx - ax)
    nl = math.hypot(nx, ny)
    nx, ny = nx / nl, ny / nl
    mcx, mcy = 0.5 * (ax + bx), 0.5 * (ay + by)
    inv = 1.0 if side >= 0 else -1.0
    if major:
        return (mcx + inv * (r + h) * nx, mcy + inv * (r + h) * ny)
    return (mcx - inv * (r - h) * nx, mcy - inv * (r - h) * ny)


def _spline_to_points(control_points, n_seg_per_span=8):
    """Approximate a spline through control points with line segments.

    Uses Catmull-Rom interpolation for smooth curves through given points.
    Returns list of (x, y) tuples (including all intermediate + end points,
    excluding the first control point which is assumed to already be in the vertex list).
    """
    import numpy as np
    pts = control_points  # list of (x, y)
    n = len(pts)
    if n < 2:
        return list(pts)

    result = []

    for i in range(n - 1):
        # Catmull-Rom: use 4 control points: p0, p1, p2, p3
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[min(i + 1, n - 1)]
        p3 = pts[min(i + 2, n - 1)]

        for j in range(1, n_seg_per_span + 1):
            t = j / n_seg_per_span
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom matrix
            x = 0.5 * ((2 * p1[0]) +
                        (-p0[0] + p2[0]) * t +
                        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) +
                        (-p0[1] + p2[1]) * t +
                        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            result.append((x, y))

    return result


def _parse_selector(args):
    """Parse a CadQuery selector string from method args.

    Returns a dict with selector info, or None if no selector / unsupported.
    Supported selectors:
      ">X", "<X", ">Y", "<Y", ">Z", "<Z"  — direction max/min
      "|X", "|Y", "|Z"                     — parallel to axis
      "#X", "#Y", "#Z"                     — perpendicular to axis
    """
    if not args:
        return None
    arg = args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        s = arg.value.strip()
        if len(s) == 2 and s[0] in ('>', '<', '|', '#') and s[1] in ('X', 'Y', 'Z'):
            return {"op": s[0], "axis": s[1]}
    return None


class _SyntheticNode:
    """Stand-in AST node for params that don't correspond to any source
    literal (workplane orientation, origin offsets, etc.)."""
    lineno = 0
    col_offset = 0
    end_col_offset = 0


def _make_slot(node: ast.AST, value: float) -> ParamSlot:
    return ParamSlot(
        value=value,
        line=getattr(node, 'lineno', 0),
        col=getattr(node, 'col_offset', 0),
        end_col=getattr(node, 'end_col_offset', 0),
    )


# --- Workplane orientation -------------------------------------------------
# CadQuery's Workplane(plane_str) lays the sketch on a specific world plane
# and extrudes along the plane normal.  Mapping (probed via cadquery):
#   XY -> sketch-X=+X, sketch-Y=+Y, normal=+Z   (standard, no rotation)
#   YZ -> sketch-X=+Y, sketch-Y=+Z, normal=+X
#   ZX -> sketch-X=+Z, sketch-Y=+X, normal=+Y
#   XZ -> sketch-X=+X, sketch-Y=+Z, normal=-Y
#   YX -> sketch-X=+Y, sketch-Y=+X, normal=-Z
#   ZY -> sketch-X=+Z, sketch-Y=+Y, normal=-X
# These are the precomputed Euler ZYX angles for rotate3d to land each shape
# in the correct world orientation (matches a Z-up extruded shape rotated to
# have sketch-X / sketch-Y / normal pointing along the listed world axes).
_WORKPLANE_EULER = {
    'XY': (0.0,        0.0,        0.0),
    'YZ': (-1.5707963, -1.5707963, 0.0),
    'ZX': ( 1.5707963,  0.0,        1.5707963),
    'XZ': (-1.5707963,  0.0,        0.0),
    'YX': ( 3.1415927,  0.0,        1.5707963),
    'ZY': ( 0.0,        1.5707963,  0.0),
}

# World-coord unit vectors for (sketch-X, sketch-Y, normal) of each named
# workplane — same mapping as the table above.  Used to map a revolve's
# sketch-frame axis into world space (for circular_pattern).
_PLANE_BASIS = {
    'XY': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    'YZ': ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    'ZX': ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    'XZ': ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    'YX': ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    'ZY': ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
}

# cadgen FUNCTIONAL ops emitted as `r = FUNC(r, ...)` instead of a method chain.
# _unroll_chain() rewrites them to the equivalent method call (receiver = first arg)
# so the existing op handlers fire.  Only forms whose remaining args line up with
# the method's positional args belong here.
_CADGEN_FUNC_AS_METHOD = {
    '_wp_attach_at': 'attach_at',   # _wp_attach_at(r, anchors, chain, combine=) -> r.attach_at(anchors, chain, combine=)
}


def _spherical_dir(call_node: ast.Call):
    """Parse cadgen.spherical_coords.SphericalAnglesDirection(theta, phi)
    where both are in degrees, polar from +Z and azimuth from +X.
    Returns the 3D unit vector as (x, y, z) or None."""
    import math
    if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, (ast.Attribute, ast.Name))):
        return None
    name = call_node.func.attr if isinstance(call_node.func, ast.Attribute) else call_node.func.id
    if name != 'SphericalAnglesDirection':
        return None
    if len(call_node.args) < 2:
        return None
    th = _extract_num(call_node.args[0])
    ph = _extract_num(call_node.args[1])
    if th is None or ph is None:
        return None
    t = math.radians(th); p = math.radians(ph)
    return (math.sin(t)*math.cos(p), math.sin(t)*math.sin(p), math.cos(t))


def _rotmat_to_euler_zyx(R):
    """Convert a 3x3 rotation matrix R (list of lists or numpy) into Euler
    ZYX angles (alpha, beta, gamma) such that R == Rz(g) * Ry(b) * Rx(a)."""
    import math
    sy = math.sqrt(R[0][0]**2 + R[1][0]**2)
    if sy > 1e-6:
        gx = math.atan2(R[2][1], R[2][2])
        gy = math.atan2(-R[2][0], sy)
        gz = math.atan2(R[1][0], R[0][0])
    else:
        gx = math.atan2(-R[1][2], R[1][1])
        gy = math.atan2(-R[2][0], sy)
        gz = 0.0
    return gx, gy, gz


def _plane_args(call_node: ast.Call):
    """Parse cadgen.spherical_coords.Plane(origin=(x,y,z),
    xDir=SphericalAnglesDirection(...), normal=SphericalAnglesDirection(...))
    into (euler_zyx_angles, origin_tuple, origin_arg_nodes).  Returns
    (None, None, None) if the call isn't a Plane factory."""
    if not isinstance(call_node, ast.Call):
        return None, None, None
    name = (call_node.func.attr if isinstance(call_node.func, ast.Attribute)
            else getattr(call_node.func, 'id', None))
    if name != 'Plane':
        return None, None, None
    origin = None
    origin_nodes = None
    xDir = None
    normal = None
    for kw in call_node.keywords:
        if kw.arg == 'origin' and isinstance(kw.value, ast.Tuple) and len(kw.value.elts) == 3:
            ox = _extract_num(kw.value.elts[0])
            oy = _extract_num(kw.value.elts[1])
            oz = _extract_num(kw.value.elts[2])
            if ox is not None:
                origin = (ox, oy, oz)
                origin_nodes = tuple(kw.value.elts)
        elif kw.arg == 'xDir':
            xDir = _spherical_dir(kw.value)
        elif kw.arg == 'normal':
            normal = _spherical_dir(kw.value)
    if xDir is None or normal is None:
        return None, None, None
    # Compute yDir = normal × xDir (right-handed).
    yDir = (normal[1]*xDir[2] - normal[2]*xDir[1],
            normal[2]*xDir[0] - normal[0]*xDir[2],
            normal[0]*xDir[1] - normal[1]*xDir[0])
    # World-shape rotation: R columns are world basis vectors for sketch-X,
    # sketch-Y, normal.  Our rotate3d node applies R to the query point, so
    # we need R_node = R_world^T to put the shape in the world frame.
    R_world = [[xDir[0], yDir[0], normal[0]],
               [xDir[1], yDir[1], normal[1]],
               [xDir[2], yDir[2], normal[2]]]
    # Transpose
    R_T = [[R_world[c][r] for c in range(3)] for r in range(3)]
    gx, gy, gz = _rotmat_to_euler_zyx(R_T)
    return (gx, gy, gz), origin, origin_nodes


def _vec_args(node: ast.AST):
    """Parse cq.Vector(x, y, z) / Vector(x, y, z) -> ((x,y,z), (xn,yn,zn)).
    Returns (None, None) if the node is not a numeric-literal Vector call."""
    if not isinstance(node, ast.Call):
        return None, None
    name = (node.func.attr if isinstance(node.func, ast.Attribute)
            else getattr(node.func, 'id', None))
    if name != 'Vector' or len(node.args) < 3:
        return None, None
    xs = _extract_num(node.args[0])
    ys = _extract_num(node.args[1])
    zs = _extract_num(node.args[2])
    if None in (xs, ys, zs):
        return None, None
    return (xs, ys, zs), (node.args[0], node.args[1], node.args[2])


def _cq_vector_plane_args(call_node: ast.AST):
    """Parse cq.Plane(origin=cq.Vector(...), xDir=cq.Vector(...),
    normal=cq.Vector(...)) -- the frame form emitted by the VLM and the det
    pipeline -- into (euler_zyx, origin_tuple, origin_arg_nodes).  Returns
    (None, None, None) when the call isn't a Vector-based Plane factory.

    This is the cq.Vector sibling of _plane_args (which handles the
    SphericalAnglesDirection form).  WITHOUT this, Workplane(cq.Plane(...))
    parsed as plane='XY' at the origin -- every off-origin / rotated block
    collapsed onto the XY origin in the SDF tree, so the optimizer fit a
    geometrically wrong (stacked) model and wrote the tuned numbers back into
    correctly-framed blocks -> det programs DEGRADED under optimization."""
    import math
    if not isinstance(call_node, ast.Call):
        return None, None, None
    name = (call_node.func.attr if isinstance(call_node.func, ast.Attribute)
            else getattr(call_node.func, 'id', None))
    if name != 'Plane':
        return None, None, None
    origin = origin_nodes = xDir = normal = None
    for kw in call_node.keywords:
        if kw.arg == 'origin':
            v, vn = _vec_args(kw.value)
            if v is not None:
                origin, origin_nodes = v, vn
        elif kw.arg == 'xDir':
            v, _ = _vec_args(kw.value)
            if v is not None:
                xDir = v
        elif kw.arg == 'normal':
            v, _ = _vec_args(kw.value)
            if v is not None:
                normal = v
    if xDir is None or normal is None:
        return None, None, None

    def _unit(u):
        m = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) or 1.0
        return (u[0]/m, u[1]/m, u[2]/m)
    xDir = _unit(xDir)
    normal = _unit(normal)
    yDir = (normal[1]*xDir[2] - normal[2]*xDir[1],
            normal[2]*xDir[0] - normal[0]*xDir[2],
            normal[0]*xDir[1] - normal[1]*xDir[0])
    # R columns are world basis for sketch-X / sketch-Y / normal; the
    # rotate3d node applies R to the query point, so use R^T (same as
    # _plane_args).
    R_world = [[xDir[0], yDir[0], normal[0]],
               [xDir[1], yDir[1], normal[1]],
               [xDir[2], yDir[2], normal[2]]]
    R_T = [[R_world[c][r] for c in range(3)] for r in range(3)]
    gx, gy, gz = _rotmat_to_euler_zyx(R_T)
    return (gx, gy, gz), origin, origin_nodes


def _workplane_args(call_node: ast.Call):
    """Extract (plane_str, origin_tuple_or_None, origin_arg_nodes) from a
    Workplane(...) call node.  Returns (None, None, None) if the call is
    not a Workplane factory."""
    if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Attribute)
            and call_node.func.attr == 'Workplane'):
        return None, None, None
    plane = 'XY'
    if call_node.args:
        a0 = call_node.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            plane = a0.value
    origin = None
    origin_nodes = None
    for kw in call_node.keywords:
        if kw.arg == 'origin' and isinstance(kw.value, ast.Tuple) and len(kw.value.elts) == 3:
            ox = _extract_num(kw.value.elts[0])
            oy = _extract_num(kw.value.elts[1])
            oz = _extract_num(kw.value.elts[2])
            if ox is not None and oy is not None and oz is not None:
                origin = (ox, oy, oz)
                origin_nodes = tuple(kw.value.elts)
    return plane, origin, origin_nodes


class CadQueryParser:
    """Walks a CadQuery method chain and builds a tree description.

    Params are emitted in DFS preorder to match C++ tree traversal:
    parent self-params first, then left child, then right child.
    """

    def __init__(self):
        self.params: List[float] = []
        self.slots: List[ParamSlot] = []
        self.names: List[str] = []
        self.tied_slots: List[Tuple[int, int]] = []
        # Per-arc write-back: dicts {a_idx,b_idx,r_s,side,mx_slot,my_slot}.  After
        # optimization the arc's through-point is recomputed from the optimized
        # endpoints (params[a_idx], params[b_idx]) + frozen r_s/side.
        self.arc_writeback: List[dict] = []
        self.revolve_bounds: List[dict] = []
        self.polygon_bounds: List[List[int]] = []
        self.alias_slots: List[Tuple[int, ParamSlot]] = []

    def _add_param(self, name: str, value: float, node: ast.AST) -> int:
        idx = len(self.params)
        self.params.append(value)
        self.slots.append(_make_slot(node, value))
        self.names.append(name)
        return idx

    def _insert_param(self, idx: int, name: str, value: float, node: ast.AST):
        """Insert a param at a specific index (for DFS ordering)."""
        self.params.insert(idx, value)
        self.slots.insert(idx, _make_slot(node, value))
        self.names.insert(idx, name)
        # Shift any tied_slots indices that come at or after the insertion
        # point.  Without this, recorded tied pairs become stale and point
        # to wrong params after later workplane/extrude wrappers insert
        # their own params at the front of the polygon's region.
        if self.tied_slots:
            self.tied_slots = [
                (m + (1 if m >= idx else 0), f + (1 if f >= idx else 0))
                for m, f in self.tied_slots
            ]
        # Same shift for arc endpoint param indices.
        for aw in self.arc_writeback:
            if aw["a_idx"] >= idx: aw["a_idx"] += 1
            if aw["b_idx"] >= idx: aw["b_idx"] += 1
        for rb in self.revolve_bounds:
            rb["idx"] = [i + (1 if i >= idx else 0) for i in rb["idx"]]
        self.polygon_bounds = [[i + (1 if i >= idx else 0) for i in pb]
                               for pb in self.polygon_bounds]
        self.alias_slots = [(i + (1 if i >= idx else 0), s)
                            for i, s in self.alias_slots]

    def _take_revolve_bounds(self, sub):
        """Adopt a sub-parser's revolve bounds; call before its params are
        appended to ours."""
        off = len(self.params)
        for rb in sub.revolve_bounds:
            self.revolve_bounds.append(dict(rb, idx=[i + off for i in rb["idx"]]))

    def _take_sub_ties(self, sub):
        """Adopt a sub-parser's tied slots and arc write-back (cut/union/
        intersect: params with source slots); call before its params are
        appended to ours.  Without it a closed sketch outline of any operation
        but the first loses its closing tie and opens in the written code."""
        off = len(self.params)
        self.tied_slots.extend((m + off, f + off) for m, f in sub.tied_slots)
        self.polygon_bounds.extend([i + off for i in pb] for pb in sub.polygon_bounds)
        self.alias_slots.extend((i + off, s) for i, s in sub.alias_slots)
        for aw in sub.arc_writeback:
            self.arc_writeback.append(dict(aw, a_idx=aw["a_idx"] + off,
                                           b_idx=aw["b_idx"] + off))

    def _build_additive_sketch(self, primitives) -> Tuple[dict, int]:
        """Build a 2D sketch_desc that's the union of pushed additive
        primitives (used when a sketch consists only of push-circle /
        push-rect, no outline polygon)."""
        sketch_start = len(self.params)

        def _emit(prim, pos):
            cx, cy, cx_node, cy_node = pos
            if prim["kind"] == "circle":
                self._add_param("circle.cx", cx, cx_node)
                self._add_param("circle.cy", cy, cy_node)
                self._add_param("circle.r", prim["r"], prim["r_node"])
                return {"type": "circle"}
            else:  # rect
                self._add_param("rect.cx", cx, cx_node)
                self._add_param("rect.cy", cy, cy_node)
                self._add_param("rect.half_w", prim["w"] / 2.0, prim["w_node"])
                self._add_param("rect.half_h", prim["h"] / 2.0, prim["h_node"])
                return {"type": "rect"}

        all_units = []
        for prim in primitives:
            for pos in prim["positions"]:
                all_units.append((prim, pos))
        if not all_units:
            raise ValueError("no additive primitives to assemble")
        # Reduce as a left-leaning union tree: ((a U b) U c) U d ...
        desc = _emit(*all_units[0])
        for prim, pos in all_units[1:]:
            right = _emit(prim, pos)
            desc = {"type": "union", "left": desc, "right": right}
        return desc, sketch_start

    def _apply_sub_primitives(self, solid_desc: dict, solid_start: int,
                              sub_prims, depth: float) -> dict:
        """For each subtractive sketch primitive (push().circle/rect with
        mode='s'), build a 3D extrude (oversized & frozen so the hole
        always punches through the parent) at the pushed position and CUT
        it from solid_desc.

        BUG FIX: cutter_depth was previously parent_depth as a separate
        tunable param.  Adam could drift it below parent_depth, leaving
        the hole short.  We oversize the cutter (3× parent depth) and
        FREEZE it; the cutter is centred by shifting -parent_depth so it
        spans [-d, +2d] in local Z, always covering parent's [0, d].
        """
        OVERSIZE = 3.0
        SHIFT = 1.0
        for prim in sub_prims:
            for pos in prim["positions"]:
                cx, cy, cx_node, cy_node = pos
                cutter_sketch_start = len(self.params)
                self._add_param("frozen.sub_extrude_depth",
                                depth * OVERSIZE, _SyntheticNode())
                if prim["kind"] == "circle":
                    self._add_param("circle.cx", cx, cx_node)
                    self._add_param("circle.cy", cy, cy_node)
                    self._add_param("circle.r", prim["r"], prim["r_node"])
                    cutter_sketch = {"type": "circle"}
                else:
                    self._add_param("rect.cx", cx, cx_node)
                    self._add_param("rect.cy", cy, cy_node)
                    self._add_param("rect.half_w", prim["w"] / 2.0, prim["w_node"])
                    self._add_param("rect.half_h", prim["h"] / 2.0, prim["h_node"])
                    cutter_sketch = {"type": "rect"}
                cutter = {"type": "extrude", "child": cutter_sketch}
                self._insert_param(cutter_sketch_start, "frozen.sub_extrude_tz",
                                   -depth * SHIFT, _SyntheticNode())
                self._insert_param(cutter_sketch_start, "frozen.sub_extrude_ty",
                                   0.0, _SyntheticNode())
                self._insert_param(cutter_sketch_start, "frozen.sub_extrude_tx",
                                   0.0, _SyntheticNode())
                cutter = {"type": "translate3d", "child": cutter}
                solid_desc = {"type": "subtract", "left": solid_desc, "right": cutter}
        return solid_desc

    def parse_chain(self, code: str) -> ParseResult:
        """Parse a CadQuery script (one or many statements) and return a
        ParseResult.  Supports both single-line chain style and multi-line
        scripts with intermediate variable bindings."""
        tree = ast.parse(code)
        # Keep the raw source: attach_at frame resolution execs the prefix
        # program to measure the true face tangent frame at each anchor.
        self._src_code = code

        # Collect all top-level assignments **in source order** so that
        # chains starting at a Name can be resolved transitively, including
        # the common DeepCAD pattern of repeated assignment to `r`:
        #     r = w0.sketch()...extrude(...)
        #     r = r.workplane(...).sketch()...cutThruAll()
        # When the second statement's RHS references `r`, we must inline
        # the *previous* assignment of `r`, not the current one.
        assigns_seq = []          # list of (name, value)
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                assigns_seq.append((stmt.targets[0].id, stmt.value))
        self._assigns_seq = assigns_seq

        # Pick the chain to process: prefer `result`, then `r`, then last assigned name.
        target_value = None
        target_index = None
        for idx, (name, val) in reversed(list(enumerate(assigns_seq))):
            if name in ('result', 'r'):
                target_value = val
                target_index = idx
                break
        if target_value is None and assigns_seq:
            target_index = len(assigns_seq) - 1
            target_value = assigns_seq[target_index][1]
        if target_value is None:
            for stmt in tree.body:
                if isinstance(stmt, ast.Expr):
                    target_value = stmt.value
                    target_index = -1
                    break

        if target_value is None:
            raise ValueError("Could not find CadQuery chain in code")

        # Resolution context: when unrolling resolves a Name `x`, look up
        # the most recent prior assignment of `x` strictly before the
        # statement currently being inlined.  This handles `r = ...; r = r.xxx()`.
        self._resolution_index = target_index
        chain = self._unroll_chain(target_value)
        if not chain:
            raise ValueError("Could not unroll CadQuery chain")

        desc = self._process_chain(chain)
        if desc is None:
            raise ValueError("Parse produced empty tree (unsupported script)")

        return ParseResult(
            tree_desc=desc,
            params=self.params,
            param_slots=self.slots,
            param_names=self.names,
            tied_slots=list(self.tied_slots),
            arc_writeback=list(self.arc_writeback),
            revolve_bounds=list(self.revolve_bounds),
            polygon_bounds=list(self.polygon_bounds),
            alias_slots=list(self.alias_slots),
        )

    def _unroll_chain(self, node: ast.AST) -> Optional[List[Tuple[str, list, ast.AST]]]:
        """Unroll a.b().c().d() into [(b, args, node), (c, args, node), ...].
        If the chain bottoms out at a Name that's bound to another chain
        in an earlier assignment, recurse through that assignment so that
        scripts split across multiple statements produce a single linear
        chain.  Self-references (`r = r.foo()`) resolve to the most recent
        *prior* binding of the name."""
        seq = getattr(self, '_assigns_seq', [])
        # Map: name -> list of indices in seq where it was assigned.
        idx_map: dict = {}
        for i, (n, _) in enumerate(seq):
            idx_map.setdefault(n, []).append(i)

        chain = []
        # Track current "resolution index" — when we inline a Name, we look
        # at the latest assignment STRICTLY BEFORE this index.
        cur_idx = getattr(self, '_resolution_index', None)
        max_inlinings = 64
        inlinings = 0
        while True:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                chain.append((node.func.attr, node.args, node))
                node = node.func.value
                continue
            # cadgen FUNCTIONAL wrappers: `r = FUNC(r, ...)` is a chain step whose
            # receiver is the first arg.  Rewrite into the equivalent method call so
            # the existing op handlers fire (e.g. `_wp_attach_at(r, A, C, combine=)`
            # == `r.attach_at(A, C, combine=)`).
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in _CADGEN_FUNC_AS_METHOD and node.args):
                chain.append((_CADGEN_FUNC_AS_METHOD[node.func.id],
                              list(node.args[1:]), node))
                node = node.args[0]
                continue
            # Frozen-mesh base: `r = __MESH__("path")` (hybrid desugar of scripts
            # with an unsupported exotic op).  It is the CHAIN ROOT (a leaf), so
            # record it and stop.
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == '__MESH__' and node.args):
                chain.append(('__MESH__', list(node.args), node))
                break
            if isinstance(node, ast.Name) and node.id in idx_map and inlinings < max_inlinings:
                indices = idx_map[node.id]
                # Pick the most recent assignment that's strictly before cur_idx.
                prev = [i for i in indices if cur_idx is None or i < cur_idx]
                if not prev:
                    break
                next_idx = prev[-1]
                node = seq[next_idx][1]
                cur_idx = next_idx
                inlinings += 1
                continue
            break
        chain.reverse()
        return chain if chain else None

    def _try_selective_bevel(self, node):
        """For r.edges(SEL).chamfer(d1[,d2]) on STRAIGHT edges: the removed
        material is exactly a triangular PRISM along each selected edge.
        Measure it instead of predicting OCC's d1/d2 face assignment: exec the
        prefix program, take the solid before and after this chamfer, diff the
        planar faces -> the NEW bevel face(s); each bevel quad's vertices
        projected along its edge give the prism cross-section exactly.
        Returns a list of synthetic cutter-chain strings (one per edge) to be
        parsed and subtracted; [] on any failure (caller keeps the old drop).
        Gated by CAD_NO_CHAMFER_BEVEL."""
        import os as _os
        if _os.environ.get('CAD_NO_CHAMFER_BEVEL'):
            return []
        try:
            import cadquery as _cq
            _src = getattr(self, '_src_code', None)
            _ln = getattr(node, 'lineno', None)
            if not _src or not _ln:
                return []
            _ns = {'cq': _cq}
            try:
                from cadquery_addons.selectors import (
                    PointOnFaceSelector as _POFS, PointOnEdgeSelector as _POES)
                _ns['PointOnFaceSelector'] = _POFS
                _ns['PointOnEdgeSelector'] = _POES
                import cadquery_addons  # noqa: registers attach_at & friends
            except Exception:
                pass
            exec('\n'.join(_src.splitlines()[:_ln - 1]), _ns)
            _wp_edges = eval(ast.get_source_segment(_src, node.func.value), _ns)
            _edges = [e for e in _wp_edges.vals() if hasattr(e, 'startPoint')]
            if not (1 <= len(_edges) <= 4):
                return []
            if any(e.geomType() != 'LINE' for e in _edges):
                return []
            # findSolid(): the context solid (.solids() after .edges() is empty
            # and .val() then silently yields the plane-origin Vector)
            _solid0 = _wp_edges.findSolid()
            _solid1 = eval(ast.get_source_segment(_src, node), _ns).findSolid()

            def _canon(f):
                # canonical plane signature (normal flipped to a fixed
                # orientation, offset along it) for duplicate detection
                if f.geomType() != 'PLANE':
                    return None
                n = f.normalAt(f.Center())
                for comp in (n.x, n.y, n.z):
                    if abs(comp) > 1e-6:
                        if comp < 0:
                            n = n.multiply(-1.0)
                        break
                return (n.x, n.y, n.z, n.dot(f.Center()))

            sig0 = [t for t in (_canon(f) for f in _solid0.Faces()) if t]

            def _seen(t):
                return any(abs(t[0] - u[0]) < 1e-4 and abs(t[1] - u[1]) < 1e-4
                           and abs(t[2] - u[2]) < 1e-4
                           and abs(t[3] - u[3]) < 1e-3 * max(1.0, abs(u[3]))
                           for u in sig0)

            new_pl = [f for f in _solid1.Faces()
                      if _canon(f) is not None and not _seen(_canon(f))]
            if len(new_pl) != len(_edges):
                return []
            outs = []
            for e in _edges:
                p0 = e.startPoint(); p1 = e.endPoint()
                ev = p1 - p0; L = ev.Length
                if L < 1e-6:
                    return []
                ed = ev.multiply(1.0 / L)
                mid = (p0 + p1).multiply(0.5)
                bf = min(new_pl, key=lambda f: (f.Center() - mid).Length)
                ax = _cq.Vector(1, 0, 0) if abs(ed.x) < 0.9 else _cq.Vector(0, 1, 0)
                xd = ed.cross(ax).normalized()
                yd = ed.cross(xd)
                pts2 = []
                for v in bf.Vertices():
                    r = _cq.Vector(v.X, v.Y, v.Z) - p0
                    pts2.append((r.dot(xd), r.dot(yd)))
                # collapse quad verts along the edge -> the two bevel-plane ends.
                # A neighbouring chamfer sharing a corner CLIPS this bevel quad
                # (extra verts), but the clip points project BETWEEN the two
                # cross-section corners on the bevel line -- so take the most
                # distant pair, which is exactly (A, B) clipped or not.
                uniq = []
                for q in pts2:
                    if not any(abs(q[0]-u[0]) < 0.5 and abs(q[1]-u[1]) < 0.5
                               for u in uniq):
                        uniq.append(q)
                if len(uniq) < 2:
                    return []
                best_pair, best_d = None, -1.0
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        d = ((uniq[i][0]-uniq[j][0])**2
                             + (uniq[i][1]-uniq[j][1])**2)
                        if d > best_d:
                            best_d = d; best_pair = (uniq[i], uniq[j])
                (a1, a2), (b1, b2) = best_pair
                # non-degenerate triangle (corner at the edge origin)
                if abs(a1 * b2 - a2 * b1) < 1e-6:
                    return []
                # corner vertex at the edge (0,0), nudged 2% past the corner so
                # the cutter's side faces clear the solid faces; the bevel
                # hypotenuse (a-b) stays EXACT.
                v0 = (-(a1 + b1) * 0.02, -(a2 + b2) * 0.02)
                o = p0 - ed.multiply(0.5)
                depth = L + 1.0
                outs.append(
                    f"cq.Workplane(cq.Plane(origin=cq.Vector({o.x!r},{o.y!r},{o.z!r}),"
                    f" xDir=cq.Vector({xd.x!r},{xd.y!r},{xd.z!r}),"
                    f" normal=cq.Vector({ed.x!r},{ed.y!r},{ed.z!r})))"
                    f".sketch().segment(({v0[0]!r},{v0[1]!r}),({a1!r},{a2!r}))"
                    f".segment(({b1!r},{b2!r})).close().assemble().finalize()"
                    f".extrude({depth!r})")
            return outs
        except Exception:
            return []

    def _wrap_with_current_wp(self, solid_desc: dict, params_start: int) -> dict:
        """Wrap solid_desc with the current workplane's translate3d(rotate3d(...)).
        Inserts the wrapper params at params_start so DFS preorder is preserved.
        Returns the new desc (unchanged if current workplane is XY at origin)."""
        plane = getattr(self, '_current_wp_plane', 'XY')
        origin = getattr(self, '_current_wp_origin', None)
        origin_nodes = getattr(self, '_current_wp_origin_nodes', None)
        euler = getattr(self, '_current_wp_euler', None)

        if plane != 'XY':
            if plane == 'CUSTOM' and euler is not None:
                gx, gy, gz = euler
            else:
                gx, gy, gz = _WORKPLANE_EULER.get(plane, (0.0, 0.0, 0.0))
            self._insert_param(params_start, 'frozen.rot.z', gz, _SyntheticNode())
            self._insert_param(params_start, 'frozen.rot.y', gy, _SyntheticNode())
            self._insert_param(params_start, 'frozen.rot.x', gx, _SyntheticNode())
            solid_desc = {"type": "rotate3d", "child": solid_desc}

        if origin is not None and any(abs(c) > 1e-12 for c in origin):
            ox, oy, oz = origin
            ox_n, oy_n, oz_n = (origin_nodes if origin_nodes else
                                (_SyntheticNode(),) * 3)
            self._insert_param(params_start, 'frozen.wp_origin.z', oz, oz_n)
            self._insert_param(params_start, 'frozen.wp_origin.y', oy, oy_n)
            self._insert_param(params_start, 'frozen.wp_origin.x', ox, ox_n)
            solid_desc = {"type": "translate3d", "child": solid_desc}

        return solid_desc

    # -- Face-relative workplanes (copyWorkplane(face_w), faces(sel).workplane) --
    def _face_seed_from_selector(self, arg):
        """faces(PointOnFaceSelector([x,y,z])) -> ('point',(x,y,z));
        faces('<Z') -> ('str','<Z'); else None."""
        amap = {n: v for n, v in getattr(self, '_assigns_seq', [])}
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return ('str', arg.value)
        if isinstance(arg, ast.Call):
            nm = (arg.func.attr if isinstance(arg.func, ast.Attribute)
                  else getattr(arg.func, 'id', ''))
            if 'PointOnFaceSelector' in nm and arg.args:
                pt = arg.args[0]
                if isinstance(pt, ast.Name) and pt.id in amap:
                    pt = amap[pt.id]
                if isinstance(pt, (ast.List, ast.Tuple)) and len(pt.elts) == 3:
                    xs = _extract_num(pt.elts[0]); ys = _extract_num(pt.elts[1]); zs = _extract_num(pt.elts[2])
                    if None not in (xs, ys, zs):
                        return ('point', (xs, ys, zs))
        return None

    def _resolve_face_seed(self, arg):
        """Trace a face-workplane variable (face_w0 = faces(sel).workplane(...))
        back to its face seed."""
        amap = {n: v for n, v in getattr(self, '_assigns_seq', [])}
        cur = arg
        for _ in range(10):
            if isinstance(cur, ast.Name):
                if cur.id in amap:
                    cur = amap[cur.id]; continue
                return None
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                if cur.func.attr == 'faces' and cur.args:
                    return self._face_seed_from_selector(cur.args[0])
                cur = cur.func.value; continue   # .workplane(), .val().Center(), ...
            return None
        return None

    def _set_face_wp(self, origin, normal):
        """Set the current workplane to a CUSTOM frame with the given origin and
        +Z = normal (so a following sketch/extrude builds on that face)."""
        import math
        nl = math.sqrt(sum(c*c for c in normal)) or 1.0
        n = tuple(c/nl for c in normal)
        a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        xd = (a[1]*n[2]-a[2]*n[1], a[2]*n[0]-a[0]*n[2], a[0]*n[1]-a[1]*n[0])
        xl = math.sqrt(sum(c*c for c in xd)) or 1.0
        xd = tuple(c/xl for c in xd)
        yd = (n[1]*xd[2]-n[2]*xd[1], n[2]*xd[0]-n[0]*xd[2], n[0]*xd[1]-n[1]*xd[0])
        Rw = [[xd[0], yd[0], n[0]], [xd[1], yd[1], n[1]], [xd[2], yd[2], n[2]]]
        Rt = [[Rw[c][r] for c in range(3)] for r in range(3)]
        self._current_wp_plane = 'CUSTOM'
        self._current_wp_euler = _rotmat_to_euler_zyx(Rt)
        self._current_wp_origin = tuple(origin)
        self._current_wp_origin_nodes = None

    def _apply_face_wp(self, seed, solid_desc, current_bbox):
        """Set the workplane to the selected face: SDF-gradient normal at the
        seed point, or the bbox face for a string selector."""
        import math
        if seed[0] == 'point' and solid_desc is not None:
            pt = seed[1]
            try:
                import _cad_grad as _cg
                bt = _cg.create_tree(solid_desc); bt.set_params(self.params)
                e = 1e-2
                g = (bt.eval_sdf(pt[0]+e, pt[1], pt[2]) - bt.eval_sdf(pt[0]-e, pt[1], pt[2]),
                     bt.eval_sdf(pt[0], pt[1]+e, pt[2]) - bt.eval_sdf(pt[0], pt[1]-e, pt[2]),
                     bt.eval_sdf(pt[0], pt[1], pt[2]+e) - bt.eval_sdf(pt[0], pt[1], pt[2]-e))
                if math.sqrt(sum(c*c for c in g)) > 1e-6:
                    self._set_face_wp(pt, g)
            except Exception:
                pass
        # NOTE: the string-selector ('<Z' etc.) form used the bbox face as an
        # approximation, but it mis-fired (regressed 01_sketch / 11_combined),
        # so it's disabled — only the precise SDF-gradient 'point' form applies.

    # -- Workplane / origin wrapping -----------------------------------------
    def _wrap_workplane(self, desc: dict) -> dict:
        """Wrap the produced solid with rotate3d (for non-XY workplane) and
        translate3d (for origin offset).  These wrapper params are inserted
        at the FRONT of the param vector and named with a 'frozen.' prefix
        so the optimizer freezes them.  The chain's first Workplane(...)
        call sets the orientation; later .copyWorkplane() within sketch
        contexts is handled inline by the sketch builder."""
        plane = getattr(self, '_root_plane', 'XY')
        origin = getattr(self, '_root_origin', None)
        origin_nodes = getattr(self, '_root_origin_nodes', None)

        # Translate first (innermost), then rotate, so the world-space shape
        # is rotate(translate_local(child)) — but cadquery's behaviour is
        # rotate around world origin THEN translate.  Equivalent layering:
        #   world_shape(p) = local_shape(R^T * (p - origin))
        # which corresponds to translate3d(rotate3d(child)) with translate
        # outermost.

        if plane != 'XY':
            gx, gy, gz = _WORKPLANE_EULER.get(plane, (0.0, 0.0, 0.0))
            self._insert_param(0, 'frozen.rot.x', gx, _SyntheticNode())
            self._insert_param(1, 'frozen.rot.y', gy, _SyntheticNode())
            self._insert_param(2, 'frozen.rot.z', gz, _SyntheticNode())
            desc = {"type": "rotate3d", "child": desc}

        if origin is not None and any(abs(c) > 1e-12 for c in origin):
            ox, oy, oz = origin
            ox_n, oy_n, oz_n = (origin_nodes if origin_nodes else
                                (_SyntheticNode(),) * 3)
            self._insert_param(0, 'frozen.wp_origin.z', oz, oz_n)
            self._insert_param(0, 'frozen.wp_origin.y', oy, oy_n)
            self._insert_param(0, 'frozen.wp_origin.x', ox, ox_n)
            desc = {"type": "translate3d", "child": desc}

        return desc

    def _process_chain(self, chain: list) -> dict:
        """Convert method chain to tree description.

        Tracks subtree_start: the param index where the current subtree begins.
        When wrapping ops (extrude, shell, translate, fillet) add self-params,
        they INSERT at subtree_start to maintain DFS preorder.
        """
        sketch_desc = None
        solid_desc = None
        # Index where the current sketch/solid subtree params begin
        sketch_start = 0
        solid_start = 0
        pending_vertices = []
        # Selector state for .edges(sel) / .faces(sel)
        pending_selector = None  # {"kind": "edges"/"faces", "op": ">/<|/#", "axis": "X/Y/Z"}
        # Track current solid's world AABB so we can compute face centers
        # for `.faces(sel).workplane(origin=r.val().Center())` chains.
        # None = unknown bbox; tuple = (xmin, xmax, ymin, ymax, zmin, zmax).
        current_bbox = None
        # Track last-used (cx, cy) sketch position for rect/circle sketches
        # so we can compute the world bbox after extrude.
        last_sketch_bbox_2d = None  # (xmin, xmax, ymin, ymax) in sketch frame
        # Loft state: previous sketches for loft accumulation
        loft_sketches = []  # [(sketch_desc, sketch_start, sketch_params_count)]
        loft_height = 0.0   # accumulated workplane offset
        # Pending workplane-local centre offset set by .center(dx,dy); applied to
        # the next DIRECT profile (circle/rect) and reset on workplane()/loft.
        pending_center = (0.0, 0.0)
        # PERSISTENT accumulated in-plane origin.  In CadQuery .center(dx,dy) is a
        # RELATIVE shift of the workplane origin, and a parallel .workplane(offset=)
        # INHERITS it -> multi-section lofts place each profile at the carried origin.
        # (Was: pending_center reset to 0 on every .workplane() -> top loft profile
        # snapped back to the axis, leaning the loft.  loft_17 X-shift bug.)
        loft_base_center = (0.0, 0.0)
        # Pending face seed from r.faces(PointOnFaceSelector([x,y,z])); the next
        # .workplane() turns it into a face-relative workplane.
        pending_face_seed = None

        # --- Sketch API state ---
        # Tracks segments/arcs accumulated inside a `.sketch()...finalize()`
        # context, plus pushed positions and additive/subtractive primitives.
        in_sketch = False
        sk_outline = []          # list of (x, y, xnode, ynode) — outline polygon
        sk_closing = None        # dropped closing copy of the outline's first vertex
        sk_current = None        # last endpoint, for .segment(p) / .arc(p1, p2)
        sk_pushed = []           # list of (cx, cy, cx_node, cy_node) — push positions
        sk_subtractive = []      # list of dicts: {kind, ...}
        sk_additive = []         # list of dicts: {kind, ...}

        # Track the *current* workplane's orientation + origin.  copyWorkplane
        # mid-chain pushes a new value onto this state; subsequent sketches
        # are interpreted in that new frame and wrapped accordingly.
        if not hasattr(self, '_current_wp_plane'):
            self._current_wp_plane = 'XY'
            self._current_wp_origin = None
            self._current_wp_origin_nodes = None

        for method, args, node in chain:
            if method == '__MESH__':
                # Frozen exotic sub-solid baked to STL.  0-param leaf; later
                # .union/.cut ops compose the optimizable suffix onto it.
                path = None
                if args and isinstance(args[0], ast.Constant):
                    path = args[0].value
                solid_desc = {"type": "mesh", "path": path}
                solid_start = len(self.params)
                continue
            if method == 'Workplane':
                # Capture the outermost workplane orientation + origin only
                # (later .copyWorkplane calls are handled separately).
                plane, origin, origin_nodes = _workplane_args(node)
                euler = None
                # Workplane(cq.Plane(origin=cq.Vector, xDir=cq.Vector,
                # normal=cq.Vector)) -- the VLM/det frame form.  Resolve it to
                # a CUSTOM euler+origin so the block lands in its true world
                # frame instead of collapsing onto the XY origin.
                if getattr(node, 'args', None):
                    e2, o2, on2 = _cq_vector_plane_args(node.args[0])
                    if e2 is not None:
                        plane = 'CUSTOM'
                        euler = e2
                        origin = o2
                        origin_nodes = on2
                if not hasattr(self, '_root_plane'):
                    self._root_plane = plane or 'XY'
                    self._root_origin = origin
                    self._root_origin_nodes = origin_nodes
                # The first Workplane in the chain also sets current_wp.
                self._current_wp_plane = plane or 'XY'
                self._current_wp_euler = euler
                self._current_wp_origin = origin
                self._current_wp_origin_nodes = origin_nodes
                continue

            elif method == 'copyWorkplane':
                # Switch to a different workplane.  Accepts:
                #   * cq.Workplane(plane_str, origin=...)                       — standard
                #   * cadgen.spherical_coords.Plane(origin, xDir, normal)
                #   * cq.Workplane(cq.Plane(origin=cq.Vector, xDir=, normal=)) — the
                #     RENDERABLE form a resolved PointOnFaceSelector face workplane is
                #     rewritten to (02_cut etc.).  cadquery's copyWorkplane needs a
                #     Workplane (not a bare Plane), so the rewrite wraps the Plane; we
                #     unwrap it here.  Without this the face cut collapses to the root.
                if args:
                    a0 = args[0]
                    nested = None
                    if (isinstance(a0, ast.Call)
                            and isinstance(getattr(a0, 'func', None), ast.Attribute)
                            and a0.func.attr == 'Workplane' and a0.args
                            and isinstance(a0.args[0], ast.Call)
                            and isinstance(getattr(a0.args[0], 'func', None), ast.Attribute)
                            and a0.args[0].func.attr == 'Plane'):
                        nested = a0.args[0]
                    e2, o2, on2 = (_cq_vector_plane_args(nested)
                                   if nested is not None else (None, None, None))
                    if e2 is not None:
                        self._current_wp_plane = 'CUSTOM'
                        self._current_wp_euler = e2
                        self._current_wp_origin = o2
                        self._current_wp_origin_nodes = on2
                    elif nested is None:
                        plane, origin, origin_nodes = _workplane_args(a0)
                        if plane is not None:
                            self._current_wp_plane = plane
                            self._current_wp_origin = origin
                            self._current_wp_origin_nodes = origin_nodes
                            self._current_wp_euler = None   # reset prior Euler
                        else:
                            euler, origin, origin_nodes = _plane_args(a0)
                            if euler is None:
                                euler, origin, origin_nodes = _cq_vector_plane_args(a0)
                            if euler is not None:
                                self._current_wp_plane = 'CUSTOM'
                                self._current_wp_euler = euler
                                self._current_wp_origin = origin
                                self._current_wp_origin_nodes = origin_nodes
                continue

            elif method == 'moveTo':
                x = _extract_num(args[0])
                y = _extract_num(args[1])
                if x is None or y is None:
                    raise ValueError("moveTo() args must be numeric literals")
                pending_vertices = [(x, y, args[0], args[1])]

            elif method == 'lineTo':
                x = _extract_num(args[0])
                y = _extract_num(args[1])
                if x is None or y is None:
                    raise ValueError("lineTo() args must be numeric literals")
                pending_vertices.append((x, y, args[0], args[1]))

            elif method == 'line':
                # Relative line: line(dx, dy) from current position
                dx = _extract_num(args[0])
                dy = _extract_num(args[1])
                if dx is None or dy is None:
                    raise ValueError("line() args must be numeric literals")
                if pending_vertices:
                    lx, ly = pending_vertices[-1][0], pending_vertices[-1][1]
                else:
                    lx, ly = 0.0, 0.0
                    pending_vertices = []
                pending_vertices.append((lx + dx, ly + dy, args[0], args[1]))

            elif method == 'hLine':
                dx = _extract_num(args[0])
                if dx is None:
                    raise ValueError("hLine() arg must be numeric literal")
                if pending_vertices:
                    lx, ly = pending_vertices[-1][0], pending_vertices[-1][1]
                else:
                    lx, ly = 0.0, 0.0
                    pending_vertices = []
                pending_vertices.append((lx + dx, ly, args[0], node))

            elif method == 'vLine':
                dy = _extract_num(args[0])
                if dy is None:
                    raise ValueError("vLine() arg must be numeric literal")
                if pending_vertices:
                    lx, ly = pending_vertices[-1][0], pending_vertices[-1][1]
                else:
                    lx, ly = 0.0, 0.0
                    pending_vertices = []
                pending_vertices.append((lx, ly + dy, args[0], node))

            elif method == 'hLineTo':
                x = _extract_num(args[0])
                if x is None:
                    raise ValueError("hLineTo() arg must be numeric literal")
                ly = pending_vertices[-1][1] if pending_vertices else 0.0
                pending_vertices.append((x, ly, args[0], node))

            elif method == 'vLineTo':
                y = _extract_num(args[0])
                if y is None:
                    raise ValueError("vLineTo() arg must be numeric literal")
                lx = pending_vertices[-1][0] if pending_vertices else 0.0
                pending_vertices.append((lx, y, args[0], node))

            elif method == 'mirrorX':
                # Mirror pending vertices about X axis (negate y), creating closed shape
                if pending_vertices:
                    mirrored = [(vx, -vy, xn, yn) for vx, vy, xn, yn in reversed(pending_vertices[1:])]
                    pending_vertices.extend(mirrored)

            elif method == 'mirrorY':
                # Mirror pending vertices about Y axis (negate x), creating closed shape
                if pending_vertices:
                    mirrored = [(-vx, vy, xn, yn) for vx, vy, xn, yn in reversed(pending_vertices[1:])]
                    pending_vertices.extend(mirrored)

            elif method == 'threePointArc':
                # threePointArc((mx,my), (ex,ey)) — arc through midpoint to endpoint
                if not pending_vertices:
                    raise ValueError("threePointArc() needs a start point (use moveTo first)")
                mid_tup = args[0]
                end_tup = args[1]
                if not (isinstance(mid_tup, ast.Tuple) and isinstance(end_tup, ast.Tuple)):
                    raise ValueError("threePointArc args must be tuples")
                mx = _extract_num(mid_tup.elts[0])
                my = _extract_num(mid_tup.elts[1])
                ex = _extract_num(end_tup.elts[0])
                ey = _extract_num(end_tup.elts[1])
                if None in (mx, my, ex, ey):
                    raise ValueError("threePointArc coords must be numeric")
                sx, sy = pending_vertices[-1][0], pending_vertices[-1][1]
                arc_pts = _arc_to_points(sx, sy, mx, my, ex, ey)
                for px, py in arc_pts:
                    pending_vertices.append((px, py, end_tup.elts[0], end_tup.elts[1]))

            elif method == 'sagittaArc':
                # sagittaArc((ex,ey), sag) — arc to endpoint with sagitta
                if not pending_vertices:
                    raise ValueError("sagittaArc() needs a start point")
                end_tup = args[0]
                sag_node = args[1]
                if not isinstance(end_tup, ast.Tuple):
                    raise ValueError("sagittaArc first arg must be tuple")
                ex = _extract_num(end_tup.elts[0])
                ey = _extract_num(end_tup.elts[1])
                sag = _extract_num(sag_node)
                if None in (ex, ey, sag):
                    raise ValueError("sagittaArc args must be numeric")
                import math
                sx, sy = pending_vertices[-1][0], pending_vertices[-1][1]
                # Midpoint of chord + perpendicular offset by sagitta
                cx_m, cy_m = (sx + ex) / 2, (sy + ey) / 2
                dx, dy = ex - sx, ey - sy
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1e-12:
                    nx, ny = -dy / d, dx / d
                else:
                    nx, ny = 0, 1
                mx_a = cx_m + sag * nx
                my_a = cy_m + sag * ny
                arc_pts = _arc_to_points(sx, sy, mx_a, my_a, ex, ey)
                for px, py in arc_pts:
                    pending_vertices.append((px, py, end_tup.elts[0], end_tup.elts[1]))

            elif method == 'polyline':
                if not args:
                    raise ValueError("polyline() needs a list argument")
                lst = args[0]
                if isinstance(lst, ast.List):
                    for elt in lst.elts:
                        if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                            x = _extract_num(elt.elts[0])
                            y = _extract_num(elt.elts[1])
                            if x is None or y is None:
                                raise ValueError("polyline coords must be numeric")
                            pending_vertices.append((x, y, elt.elts[0], elt.elts[1]))

            elif method == 'close':
                if pending_vertices and len(pending_vertices) >= 3:
                    sketch_start = len(self.params)
                    n = len(pending_vertices)
                    for vx, vy, xn, yn in pending_vertices:
                        self._add_param("poly.vx", vx, xn)
                        self._add_param("poly.vy", vy, yn)
                    sketch_desc = {"type": "polygon", "n_vertices": n}
                    self.polygon_bounds.append(list(range(sketch_start, sketch_start + 2 * n)))
                    pending_vertices = []

            elif method == 'rect' and not in_sketch:
                w = _extract_num(args[0])
                h = _extract_num(args[1])
                if w is None or h is None:
                    raise ValueError("rect() args must be numeric literals")
                sketch_start = len(self.params)
                self._add_param("rect.cx", pending_center[0], node)
                self._add_param("rect.cy", pending_center[1], node)
                self._add_param("rect.half_w", w / 2.0, args[0])
                self._add_param("rect.half_h", h / 2.0, args[1])
                sketch_desc = {"type": "rect"}
                pending_center = (0.0, 0.0)

            elif method == 'circle' and not in_sketch:
                r = _extract_num(args[0])
                if r is None:
                    raise ValueError("circle() arg must be numeric literal")
                sketch_start = len(self.params)
                self._add_param("circle.cx", pending_center[0], node)
                self._add_param("circle.cy", pending_center[1], node)
                self._add_param("circle.r", r, args[0])
                sketch_desc = {"type": "circle"}
                pending_center = (0.0, 0.0)

            elif method == 'slot':
                # slot(length, width, angle=0) — stadium/discorectangle
                # CadQuery Sketch API: total length, total width (=diameter of caps)
                length = _extract_num(args[0])
                width = _extract_num(args[1])
                if length is None or width is None:
                    raise ValueError("slot() args must be numeric literals")
                # Extract optional angle keyword
                import math
                angle_deg = 0.0
                angle_node = node  # default: synthetic (no source location)
                for kw in getattr(node, 'keywords', []):
                    if getattr(kw, 'arg', None) == 'angle':
                        v = _extract_num(kw.value)
                        if v is not None:
                            angle_deg = v
                            angle_node = kw.value
                angle_rad = angle_deg * math.pi / 180.0
                sketch_start = len(self.params)
                self._add_param("slot.cx", 0.0, node)
                self._add_param("slot.cy", 0.0, node)
                # Store original length/width — C++ converts to half_len/radius
                self._add_param("slot.length", length, args[0])
                self._add_param("slot.width", width, args[1])
                self._add_param("slot.angle", angle_rad, angle_node)
                sketch_desc = {"type": "slot"}

            elif method == 'sketch':
                # Enter Sketch API mode.
                in_sketch = True
                sk_outline = []
                sk_closing = None
                sk_current = None
                sk_pushed = []
                sk_subtractive = []
                sk_additive = []
                sk_arc_at = {}   # outline-index-of-arc-END -> (r_s, side, mx_node, my_node, mx, my)
                continue

            elif method == 'segment' and in_sketch:
                # Sketch-API segment: .segment(p1, p2) (new edge from p1->p2)
                # or .segment(p) (continue from previous endpoint to p).
                def _coord(arg):
                    if not isinstance(arg, ast.Tuple) or len(arg.elts) < 2:
                        raise ValueError("segment() arg must be a 2-tuple")
                    x = _extract_num(arg.elts[0])
                    y = _extract_num(arg.elts[1])
                    if x is None or y is None:
                        raise ValueError("segment() coords must be numeric literals")
                    return (x, y, arg.elts[0], arg.elts[1])
                if len(args) == 2:
                    a = _coord(args[0]); b = _coord(args[1])
                    if not sk_outline or (sk_outline[-1][0], sk_outline[-1][1]) != (a[0], a[1]):
                        sk_outline.append(a)
                    sk_outline.append(b)
                    sk_current = b
                elif len(args) == 1:
                    b = _coord(args[0])
                    if sk_current is None:
                        # Implicit start at (0,0)
                        sk_outline.append((0.0, 0.0, _SyntheticNode(), _SyntheticNode()))
                    sk_outline.append(b)
                    sk_current = b
                else:
                    raise ValueError(f"segment() expects 1 or 2 args, got {len(args)}")
                continue

            elif method == 'arc' and in_sketch:
                # Sketch-API arc:
                #   .arc(mid, end)        — 3-point arc using current point
                #   .arc(p1, p2, p3)      — 3-point arc through three explicit points
                def _coord(arg):
                    if not isinstance(arg, ast.Tuple) or len(arg.elts) < 2:
                        raise ValueError("arc() arg must be a 2-tuple")
                    x = _extract_num(arg.elts[0]); y = _extract_num(arg.elts[1])
                    if x is None or y is None:
                        raise ValueError("arc() coords must be numeric literals")
                    return (x, y, arg.elts[0], arg.elts[1])
                if len(args) == 3:
                    p1 = _coord(args[0]); p2 = _coord(args[1]); p3 = _coord(args[2])
                    start = p1; mid = p2; end = p3
                    if not sk_outline or (sk_outline[-1][0], sk_outline[-1][1]) != (p1[0], p1[1]):
                        sk_outline.append(p1)
                elif len(args) == 2:
                    if sk_current is None:
                        raise ValueError("arc(mid,end) without current point")
                    start = sk_current
                    mid = _coord(args[0]); end = _coord(args[1])
                else:
                    raise ValueError(f"arc() expects 2 or 3 args, got {len(args)}")
                # Native arc edge (arcpoly): keep ONLY the endpoint as a real vertex
                # and record (r_s, side) + the through-point's slots for the arc edge,
                # so the optimizer tunes the shared endpoints (not a baked 32-gon).
                # Fallback to legacy sampling if disabled or the arc is degenerate.
                import os as _os
                _rs_side = (None if _os.environ.get('CADOPT_ARC_SAMPLE')
                            else _arc_rs_side(start[0], start[1], mid[0], mid[1],
                                              end[0], end[1]))
                if _rs_side is None:
                    pts = _arc_to_points(start[0], start[1],
                                         mid[0], mid[1], end[0], end[1], n_seg=32)
                    for i, (px, py) in enumerate(pts):
                        if i == len(pts) - 1:
                            sk_outline.append((px, py, end[2], end[3]))
                        else:
                            sk_outline.append((px, py, _SyntheticNode(), _SyntheticNode()))
                else:
                    r_s, side, major = _rs_side
                    end_idx = len(sk_outline)
                    sk_outline.append((end[0], end[1], end[2], end[3]))
                    sk_arc_at[end_idx] = (r_s, side, mid[2], mid[3], mid[0], mid[1], major)
                sk_current = (end[0], end[1], end[2], end[3])
                continue

            elif method == 'close' and in_sketch:
                # Marks the outline as a closed loop. Polygon SDF auto-closes.
                continue

            elif method == 'assemble':
                # Sketch-API assemble: outline is now finalized as a wire/face.
                continue

            elif method == 'reset' and in_sketch:
                # Sketch-API reset() clears the current point — no-op in our model.
                continue

            elif method == 'face' and in_sketch:
                # .face(sub_sketch, mode='s'): subtract a sub-sketch face from
                # the current sketch.  Extract polygon vertices from the
                # sub-sketch chain (which is a sequence of .segment() calls
                # ending with .close().assemble()).  Add as a subtractive
                # polygon primitive.
                if not args:
                    continue
                mode = 'a'
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                        mode = 's' if kw.value.value == 's' else 'a'
                if mode != 's':
                    continue  # only subtractive faces supported
                # Walk the sub-chain and collect outline segments/arcs.
                # Each operation contributes vertices (and arc-interior points
                # for arcs).  Walk outside→in then reverse to get script order.
                sub_ops = []  # list of (op_name, parsed_args_list_of_(x,y,xn,yn))
                cur_sub = args[0]
                while isinstance(cur_sub, ast.Call) and isinstance(cur_sub.func, ast.Attribute):
                    attr = cur_sub.func.attr
                    sub_args = cur_sub.args
                    if attr in ('segment', 'arc'):
                        op_pts = []
                        for a in sub_args:
                            if isinstance(a, ast.Tuple) and len(a.elts) >= 2:
                                x = _extract_num(a.elts[0]); y = _extract_num(a.elts[1])
                                if x is not None and y is not None:
                                    op_pts.append((x, y, a.elts[0], a.elts[1]))
                        sub_ops.append((attr, op_pts))
                    cur_sub = cur_sub.func.value
                sub_ops.reverse()
                # Convert ops into a flat outline.  For arcs, expand to 32
                # polyline segments using _arc_to_points (matches main parser).
                outline_pts = []
                def _dedup_append(p):
                    if outline_pts and abs(outline_pts[-1][0]-p[0]) < 1e-9 \
                            and abs(outline_pts[-1][1]-p[1]) < 1e-9:
                        return
                    outline_pts.append(p)
                cur_pt = None
                for op_name, op_pts in sub_ops:
                    if op_name == 'segment':
                        for pt in op_pts:
                            _dedup_append(pt); cur_pt = pt
                    else:  # arc
                        # Determine start: 3-arg arc takes args explicit; 2-arg arc starts from cur_pt
                        if len(op_pts) == 3:
                            start, mid, end = op_pts
                        elif len(op_pts) == 2 and cur_pt is not None:
                            start = cur_pt; mid = op_pts[0]; end = op_pts[1]
                        else:
                            continue
                        pts = _arc_to_points(start[0], start[1],
                                             mid[0], mid[1], end[0], end[1], n_seg=32)
                        # Append start if not already, then 31 interior synthetic + end real
                        _dedup_append(start)
                        for i, (px, py) in enumerate(pts):
                            if i == len(pts) - 1:
                                _dedup_append((px, py, end[2], end[3]))
                            else:
                                _dedup_append((px, py, _SyntheticNode(), _SyntheticNode()))
                        cur_pt = end
                if len(outline_pts) < 3:
                    continue
                # Add as a subtractive polygon primitive.
                sk_subtractive.append({
                    "kind": "polygon",
                    "vertices": outline_pts,
                    "positions": [(0.0, 0.0, _SyntheticNode(), _SyntheticNode())],
                })
                continue

            elif method == 'polygon' and in_sketch:
                # Sketch.polygon([(x,y), ...], mode='a'|'s') — one
                # self-contained face op (det emitter v6+ uses this instead
                # of segment chains; see cadfit_pass._sketch_frag).
                if not args or not isinstance(args[0], ast.List):
                    raise ValueError("polygon() needs a list literal")
                ppts = []
                for elt in args[0].elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                        x = _extract_num(elt.elts[0])
                        y = _extract_num(elt.elts[1])
                        if x is None or y is None:
                            raise ValueError("polygon() coords must be numeric")
                        ppts.append((x, y, elt.elts[0], elt.elts[1]))
                # drop explicit closing duplicate of the first vertex
                closing = None
                if len(ppts) >= 2 and abs(ppts[0][0] - ppts[-1][0]) < 1e-9 \
                        and abs(ppts[0][1] - ppts[-1][1]) < 1e-9:
                    closing = ppts[-1]
                    ppts = ppts[:-1]
                if len(ppts) < 3:
                    raise ValueError("polygon() needs >= 3 vertices")
                pmode = 'a'
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                        pmode = 's' if kw.value.value == 's' else 'a'
                _identity_push = (not sk_pushed) or (
                    len(sk_pushed) == 1 and abs(sk_pushed[0][0]) < 1e-9
                    and abs(sk_pushed[0][1]) < 1e-9)
                if pmode == 'a' and not sk_outline and not sk_additive \
                        and _identity_push:
                    sk_outline.extend(ppts)
                    sk_closing = closing
                elif pmode == 's':
                    sk_subtractive.append({
                        "kind": "polygon",
                        "vertices": ppts,
                        "positions": list(sk_pushed) if sk_pushed else
                        [(0.0, 0.0, _SyntheticNode(), _SyntheticNode())],
                    })
                else:
                    raise ValueError(
                        "additive polygon() over existing outline unsupported")
                continue

            elif method == 'push' and in_sketch:
                # .push([(x0,y0), (x1,y1), ...]) — set positions for the
                # following circle/rect (additive or subtractive).
                if not args or not isinstance(args[0], ast.List):
                    raise ValueError("push() needs a list literal")
                sk_pushed = []
                for elt in args[0].elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                        x = _extract_num(elt.elts[0])
                        y = _extract_num(elt.elts[1])
                        if x is None or y is None:
                            raise ValueError("push() coords must be numeric")
                        sk_pushed.append((x, y, elt.elts[0], elt.elts[1]))
                continue

            elif method == 'rarray' and in_sketch:
                # .rarray(xs, ys, nx, ny): replicate the current pushed positions on
                # an nx-by-ny grid (centred), spacings xs/ys.  cadquery Sketch.rarray.
                # Was UNHANDLED -> silently a no-op, so a following .rect(mode='s')
                # subtracted ONE slot instead of the whole array (bb_03_644).
                xs = _extract_num(args[0]); ys = _extract_num(args[1])
                nx = _extract_num(args[2]) if len(args) > 2 else 1
                ny = _extract_num(args[3]) if len(args) > 3 else 1
                if None in (xs, ys, nx, ny):
                    raise ValueError("rarray() args must be numeric")
                nx = max(1, int(round(nx))); ny = max(1, int(round(ny)))
                base = sk_pushed if sk_pushed else [(0.0, 0.0, _SyntheticNode(), _SyntheticNode())]
                expanded = []
                for (px, py, _pxn, _pyn) in base:
                    for i in range(nx):
                        for j in range(ny):
                            ox = (i - (nx - 1) / 2.0) * xs
                            oy = (j - (ny - 1) / 2.0) * ys
                            # derived positions -> synthetic slots (regenerated on
                            # re-parse from the base push + rarray literals)
                            expanded.append((px + ox, py + oy, _SyntheticNode(), _SyntheticNode()))
                sk_pushed = expanded
                continue

            elif (method == 'circle' or method == 'rect') and in_sketch:
                # Inside a sketch:
                #   .push([(x,y),...]).circle(r) | .rect(w,h)        - additive at positions
                #   .push([(x,y),...]).circle(r, mode='s') | rect(.., mode='s') - subtractive
                # If no preceding .push(), the primitive sits at (0, 0)
                # and (in the absence of an outline) becomes the sketch.
                mode = 'a'
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                        mode = 's' if kw.value.value == 's' else 'a'
                positions = list(sk_pushed) if sk_pushed else \
                    [(0.0, 0.0, _SyntheticNode(), _SyntheticNode())]
                if method == 'circle':
                    r = _extract_num(args[0])
                    if r is None:
                        raise ValueError("circle() radius must be numeric")
                    primitive = {"kind": "circle", "r": r, "r_node": args[0]}
                else:
                    w = _extract_num(args[0]); h = _extract_num(args[1])
                    if w is None or h is None:
                        raise ValueError("rect() args must be numeric")
                    primitive = {"kind": "rect", "w": w, "h": h,
                                 "w_node": args[0], "h_node": args[1]}
                primitive["positions"] = positions
                (sk_subtractive if mode == 's' else sk_additive).append(primitive)
                # NOTE: don't clear sk_pushed — cadquery's push() positions
                # persist across subsequent operations until a new push().
                # E.g. push([P]).circle(R1).circle(R2, mode='s') means an
                # annulus at P, not a circle at P with a hole at origin.
                continue

            elif method == 'finalize':
                # Exit sketch context. Produce a sketch_desc representing
                # the assembled outline (if any) plus any push-additive
                # primitives, with subtractive primitives applied as 2D
                # subtracts INSIDE the sketch tree.  The 2D subtract gives
                # the correct annulus/pocket SDF — both inside the hole
                # (distance to inner wall) and outside.  Previously we did
                # 3D cuts with oversized extruded cutters, which made the
                # subtract SDF wrong in the hole interior (cutter z-boundary
                # would mask the true xy distance to the inner wall).
                in_sketch = False
                # Track sketch 2D bbox for later face-center calculations.
                last_sketch_bbox_2d = None
                if sk_outline:
                    xs = [v[0] for v in sk_outline]
                    ys = [v[1] for v in sk_outline]
                    last_sketch_bbox_2d = (min(xs), max(xs), min(ys), max(ys))
                elif sk_additive:
                    xs_min = []; xs_max = []; ys_min = []; ys_max = []
                    for prim in sk_additive:
                        for pos in prim["positions"]:
                            cx, cy = pos[0], pos[1]
                            if prim["kind"] == "circle":
                                r = prim["r"]
                                xs_min.append(cx - r); xs_max.append(cx + r)
                                ys_min.append(cy - r); ys_max.append(cy + r)
                            else:  # rect
                                hw = prim["w"] / 2.0; hh = prim["h"] / 2.0
                                xs_min.append(cx - hw); xs_max.append(cx + hw)
                                ys_min.append(cy - hh); ys_max.append(cy + hh)
                    if xs_min:
                        last_sketch_bbox_2d = (min(xs_min), max(xs_max),
                                               min(ys_min), max(ys_max))
                if sk_outline:
                    sketch_start = len(self.params)
                    n = len(sk_outline)
                    # Detect ALL duplicate-vertex pairs.  If two segments in the
                    # outline share a vertex (T-junction, closed loop, or any
                    # repeated coord like polygon `.segment((0,10),(0,90))`
                    # later followed by `.segment((0,90),(0,10))`), each
                    # written occurrence is a SEPARATE AST node → separate
                    # param slot.  Adam moves them independently and the
                    # polygon outline deforms inconsistently — cadquery's
                    # render becomes unstable or invalid.  Tie all duplicates
                    # to the FIRST occurrence's slot.
                    # Only tie REAL (non-synthetic) slots — synthetic arc
                    # interior points are derived and have no source location.
                    seen = {}  # (round_vx, round_vy) -> first_index_in_outline
                    dup_pairs = []  # (master_outline_idx, follower_outline_idx)
                    # AXIS-SHARE tying: count how many vertices share each
                    # x/y coord.  Only TIE when a coord is shared by ≥3
                    # vertices (strong evidence of structural alignment,
                    # e.g., a long edge with multiple kink-points along it).
                    # 2-vertex share happens too commonly in genuine non-rect
                    # polygons where tying would over-constrain.
                    x_count = {}; y_count = {}
                    for i, (vx, vy, xn, yn) in enumerate(sk_outline):
                        is_syn = (getattr(xn, 'lineno', 0) == 0
                                  and getattr(yn, 'lineno', 0) == 0)
                        if is_syn: continue
                        x_count[round(vx, 9)] = x_count.get(round(vx, 9), 0) + 1
                        y_count[round(vy, 9)] = y_count.get(round(vy, 9), 0) + 1
                    x_first = {}; y_first = {}
                    x_share = []; y_share = []
                    for i, (vx, vy, xn, yn) in enumerate(sk_outline):
                        is_syn = (getattr(xn, 'lineno', 0) == 0
                                  and getattr(yn, 'lineno', 0) == 0)
                        if is_syn:
                            continue
                        key = (round(vx, 9), round(vy, 9))
                        if key in seen:
                            dup_pairs.append((seen[key], i))
                        else:
                            seen[key] = i
                        rx = round(vx, 9); ry = round(vy, 9)
                        # Only tie if ≥3 vertices share this coordinate.
                        if x_count[rx] >= 3:
                            if rx in x_first:
                                x_share.append((x_first[rx], i))
                            else:
                                x_first[rx] = i
                        if y_count[ry] >= 3:
                            if ry in y_first:
                                y_share.append((y_first[ry], i))
                            else:
                                y_first[ry] = i
                    for vx, vy, xn, yn in sk_outline:
                        self._add_param("poly.vx", vx, xn)
                        self._add_param("poly.vy", vy, yn)
                    if sk_closing is not None:  # the closing copy follows vertex 0
                        vx, vy, xn, yn = sk_closing
                        self.alias_slots.append((sketch_start, _make_slot(xn, vx)))
                        self.alias_slots.append((sketch_start + 1, _make_slot(yn, vy)))
                    # Record tied pairs at the param-index level (vx and vy).
                    for master_i, follower_i in dup_pairs:
                        m_vx = sketch_start + 2 * master_i
                        f_vx = sketch_start + 2 * follower_i
                        self.tied_slots.append((m_vx, f_vx))
                        self.tied_slots.append((m_vx + 1, f_vx + 1))
                    # Tie shared x (vx of slot at +0) and shared y (vy at +1).
                    for master_i, follower_i in x_share:
                        self.tied_slots.append((sketch_start + 2*master_i,
                                                sketch_start + 2*follower_i))
                    for master_i, follower_i in y_share:
                        self.tied_slots.append((sketch_start + 2*master_i + 1,
                                                sketch_start + 2*follower_i + 1))
                    if sk_arc_at:
                        # edge i (vertices[i]->vertices[(i+1)%n]) is an arc iff vertex
                        # (i+1) was recorded as an arc endpoint.  r_s/side are frozen
                        # constants; only the endpoint vertex params optimize.  Record
                        # write-back metadata (param indices of the endpoints + the
                        # through-point's source slots) so optimized endpoints re-emit
                        # a valid .arc(p1, through, p3).
                        edges = []
                        for ei in range(n):
                            meta = sk_arc_at.get((ei + 1) % n)
                            if meta is not None:
                                r_s, side, mxn, myn, mxv, myv, major = meta
                                edges.append({"is_arc": 1, "r_s": r_s, "side": side,
                                              "major": 1 if major else 0})
                                self.arc_writeback.append({
                                    "a_idx": sketch_start + 2 * ei,
                                    "b_idx": sketch_start + 2 * ((ei + 1) % n),
                                    "r_s": r_s, "side": side, "major": bool(major),
                                    "mx_slot": _make_slot(mxn, mxv),
                                    "my_slot": _make_slot(myn, myv)})
                            else:
                                edges.append({"is_arc": 0})
                        sketch_desc = {"type": "arcpoly", "n_vertices": n, "edges": edges}
                    else:
                        sketch_desc = {"type": "polygon", "n_vertices": n}
                        self.polygon_bounds.append(
                            list(range(sketch_start, sketch_start + 2 * n)))
                elif sk_additive:
                    # Union additive primitives at their pushed positions.
                    sketch_desc, sketch_start = self._build_additive_sketch(sk_additive)
                else:
                    raise ValueError("finalize() with empty sketch")
                # Apply each subtractive primitive at the 2D sketch level.
                # BUG FIX: if sub-primitive's (cx, cy) AST node is the SAME
                # OBJECT as an additive primitive's (i.e., they share the
                # same push() position from `.push([P]).circle(R1).circle(R2,mode='s')`),
                # both slots would write back to the same source position
                # → corrupt code.  Use SYNTHETIC nodes for sub.cx/cy in that
                # case so write-back is skipped (parent's cx/cy is the one
                # written).  We additionally TIE sub's cx/cy params to the
                # parent's so Adam keeps them in lock-step.
                # Build a map of AST node id → param index for additive cx/cy.
                add_pos_to_idx = {}  # (id(x_node), id(y_node)) -> (cx_param_idx, cy_param_idx)
                # Note: sketch_start is the index BEFORE additive params were added.
                idx = sketch_start
                for prim in sk_additive:
                    for pos in prim["positions"]:
                        add_pos_to_idx[(id(pos[2]), id(pos[3]))] = (idx, idx + 1)
                        if prim["kind"] == "circle":
                            idx += 3  # cx, cy, r
                        else:
                            idx += 4  # cx, cy, half_w, half_h
                for sub_prim in sk_subtractive:
                    # POLYGON sub-primitives (from .face(sub_sketch, mode='s')):
                    # add poly.vx/vy params for each vertex, build polygon desc.
                    # FREEZE these via 'frozen.' name prefix — the .face()
                    # sub-sketch is given in cadquery's exact form; if Adam
                    # tunes the vertices our SDF can drift from cadquery's
                    # render (Adam minimises loss, cadquery renders the
                    # tuned values, but the resulting wire may be different
                    # from what our polygon SDF predicts — esp. for non-
                    # convex outlines).  Freezing keeps cadquery's render
                    # consistent with our SDF model.
                    if sub_prim["kind"] == "polygon":
                        sub_start = len(self.params)
                        verts = sub_prim["vertices"]
                        for vx, vy, xn, yn in verts:
                            self._add_param("frozen.face_poly.vx", vx, _SyntheticNode())
                            self._add_param("frozen.face_poly.vy", vy, _SyntheticNode())
                        sub_desc = {"type": "polygon", "n_vertices": len(verts)}
                        sketch_desc = {"type": "subtract",
                                       "left": sketch_desc, "right": sub_desc}
                        continue
                    for sub_pos in sub_prim["positions"]:
                        cx, cy, cx_node, cy_node = sub_pos
                        shared = add_pos_to_idx.get((id(cx_node), id(cy_node)))
                        # If position shared with an additive primitive,
                        # use synthetic nodes (no source write-back) and
                        # tie to the additive's cx/cy.
                        sub_cx_idx = len(self.params)
                        if shared:
                            self._add_param("circle.cx", cx, _SyntheticNode())
                            self._add_param("circle.cy", cy, _SyntheticNode())
                            self.tied_slots.append((shared[0], sub_cx_idx))
                            self.tied_slots.append((shared[1], sub_cx_idx + 1))
                        else:
                            if sub_prim["kind"] == "circle":
                                self._add_param("circle.cx", cx, cx_node)
                                self._add_param("circle.cy", cy, cy_node)
                            else:
                                self._add_param("rect.cx", cx, cx_node)
                                self._add_param("rect.cy", cy, cy_node)
                        if sub_prim["kind"] == "circle":
                            self._add_param("circle.r", sub_prim["r"], sub_prim["r_node"])
                            sub_desc = {"type": "circle"}
                        else:
                            self._add_param("rect.half_w", sub_prim["w"] / 2.0,
                                            sub_prim["w_node"])
                            self._add_param("rect.half_h", sub_prim["h"] / 2.0,
                                            sub_prim["h_node"])
                            sub_desc = {"type": "rect"}
                        sketch_desc = {"type": "subtract",
                                       "left": sketch_desc, "right": sub_desc}
                # No more pending 3D sub-primitives — they're consumed here.
                self._pending_sub_primitives = []
                sk_outline = []; sk_pushed = []; sk_subtractive = []; sk_additive = []
                sk_closing = None
                continue

            elif method == 'extrude':
                d = _extract_num(args[0])
                if d is None:
                    raise ValueError("extrude() arg must be numeric literal")
                if sketch_desc is None:
                    raise ValueError("extrude() without a sketch")
                # cadgen extrude kwargs:
                #   combine='s'  → subtract from current solid (like cutThruAll)
                #   combine='a'  → union (default)
                #   both=True    → extrude in BOTH +Z and -Z, total length 2*|d|
                combine = 'a'
                both = False
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'combine' and isinstance(kw.value, ast.Constant):
                        combine = 's' if kw.value.value == 's' else 'a'
                    elif kw.arg == 'both' and isinstance(kw.value, ast.Constant):
                        both = bool(kw.value.value)

                # Pass the signed depth directly — ExtrudeNode SDF now supports
                # negative depth (extends to z ∈ [min(0,d), max(0,d)]).  This
                # means when Adam tunes the depth param, the shape's extent
                # changes correctly anchored at the workplane (z=0 local),
                # matching cadquery's behaviour for negative extrudes.
                # For both=True: 2*|d| total length, centred — still needs
                # a translate to recentre.
                if both:
                    use_d = 2 * abs(d)
                else:
                    use_d = d  # signed!

                # When both=True, FREEZE the depth.  The translate by -use_d/2
                # that centres the cutter is also frozen at initial value;
                # if depth were tunable, Adam could grow it and the cutter
                # would slide off-centre (cut asymmetric).  Freezing depth
                # keeps the cutter geometry consistent.
                if both:
                    self._insert_param(sketch_start, "frozen.extrude.depth",
                                       use_d, args[0])
                else:
                    self._insert_param(sketch_start, "extrude.depth", use_d, args[0])
                new_solid_start = sketch_start
                new_solid = {"type": "extrude", "child": sketch_desc}
                sketch_desc = None

                # Apply any pending subtractive primitives from the sketch.
                pending_sub = getattr(self, '_pending_sub_primitives', [])
                if pending_sub:
                    new_solid = self._apply_sub_primitives(new_solid, new_solid_start,
                                                           pending_sub, abs(use_d))
                    self._pending_sub_primitives = []

                # Centring shift only for both=True now (negative-d is handled
                # natively by ExtrudeNode).
                if both:
                    self._insert_param(new_solid_start, "frozen.ext_z",
                                       -use_d / 2.0, _SyntheticNode())
                    self._insert_param(new_solid_start, "frozen.ext_y", 0.0, _SyntheticNode())
                    self._insert_param(new_solid_start, "frozen.ext_x", 0.0, _SyntheticNode())
                    new_solid = {"type": "translate3d", "child": new_solid}

                # Apply the current workplane transform.
                new_solid = self._wrap_with_current_wp(new_solid, new_solid_start)

                # combine='s' → subtract; else union (default) or first extrude.
                if combine == 's' and solid_desc is not None:
                    solid_desc = {"type": "subtract", "left": solid_desc, "right": new_solid}
                elif solid_desc is None:
                    solid_desc = new_solid
                    solid_start = new_solid_start
                else:
                    solid_desc = {"type": "union", "left": solid_desc, "right": new_solid}

                # Update current_bbox in world frame, for .faces().workplane()
                # support.  Uses sketch 2D bbox + extrude depth + workplane.
                if last_sketch_bbox_2d is not None:
                    sxmn, sxmx, symn, symx = last_sketch_bbox_2d
                    # Extrude direction range (signed)
                    if both:
                        emin, emax = -abs(d), abs(d)
                    elif use_d >= 0:
                        emin, emax = 0.0, use_d
                    else:
                        emin, emax = use_d, 0.0
                    plane = getattr(self, '_current_wp_plane', 'XY') or 'XY'
                    ox, oy, oz = getattr(self, '_current_wp_origin', None) or (0, 0, 0)
                    # Map (sketch_x, sketch_y, extrude_z) → (world_x, world_y, world_z)
                    # per workplane axis mapping (see _WORKPLANE_EULER comment).
                    plane_axes = {
                      'XY': ('X','Y','+Z'), 'YZ': ('Y','Z','+X'), 'ZX': ('Z','X','+Y'),
                      'XZ': ('X','Z','-Y'), 'YX': ('Y','X','-Z'), 'ZY': ('Z','Y','-X'),
                    }
                    if plane in plane_axes:
                        sxax, syax, nax_sign = plane_axes[plane]
                        wbb = {'X': None, 'Y': None, 'Z': None}
                        wbb[sxax] = (sxmn, sxmx)
                        wbb[syax] = (symn, symx)
                        nax = nax_sign[1]
                        sign = 1.0 if nax_sign[0] == '+' else -1.0
                        e_lo = sign * emin
                        e_hi = sign * emax
                        wbb[nax] = (min(e_lo, e_hi), max(e_lo, e_hi))
                        wxr = wbb['X']; wyr = wbb['Y']; wzr = wbb['Z']
                        bbox_this = (wxr[0]+ox, wxr[1]+ox,
                                     wyr[0]+oy, wyr[1]+oy,
                                     wzr[0]+oz, wzr[1]+oz)
                        if current_bbox is None:
                            current_bbox = bbox_this
                        else:
                            current_bbox = (
                              min(current_bbox[0], bbox_this[0]),
                              max(current_bbox[1], bbox_this[1]),
                              min(current_bbox[2], bbox_this[2]),
                              max(current_bbox[3], bbox_this[3]),
                              min(current_bbox[4], bbox_this[4]),
                              max(current_bbox[5], bbox_this[5]),
                            )

            elif method == 'revolve':
                if sketch_desc is None:
                    raise ValueError("revolve() without a sketch")
                # CadQuery: revolve(angleDegrees=360, axisStart=None, axisEnd=None).
                # The default axis is sketch-Y; if explicit axisStart/axisEnd
                # are given and their delta is dominantly along sketch-X, we
                # tell the C++ node to revolve around sketch-X instead.
                axis_along_x = 0
                # axis_offset: perpendicular position of the revolve axis in the
                # sketch plane.  For a sketch-Y axis (vertical) it's the sketch-u
                # (x) of the axis line; for a sketch-X axis it's the sketch-v (y).
                # The C++ RevolveNode measures the radial coordinate from here, so
                # off-axis revolves (e.g. axis at u=67) come out faithful.
                axis_offset = 0.0
                axis_dir_sign = 1.0   # sign of (axisEnd-axisStart) along the axis
                # angle (degrees) of the sweep; <360 => partial revolve (wedge).
                angle_deg = 360.0
                if len(args) >= 1:
                    _a = _extract_num(args[0])
                    if _a is not None:
                        angle_deg = _a
                if len(args) >= 3:
                    p1 = args[1]
                    p2 = args[2]
                    if isinstance(p1, ast.Tuple) and isinstance(p2, ast.Tuple):
                        if len(p1.elts) >= 2 and len(p2.elts) >= 2:
                            ax0 = _extract_num(p1.elts[0]); ay0 = _extract_num(p1.elts[1])
                            ax1 = _extract_num(p2.elts[0]); ay1 = _extract_num(p2.elts[1])
                            dx_x = (ax1 - ax0) if (ax1 is not None and ax0 is not None) else 0.0
                            dx_y = (ay1 - ay0) if (ay1 is not None and ay0 is not None) else 0.0
                            if abs(dx_x) > abs(dx_y):
                                axis_along_x = 1
                                if ay0 is not None: axis_offset = ay0
                                axis_dir_sign = 1.0 if dx_x >= 0 else -1.0
                            else:
                                if ax0 is not None: axis_offset = ax0
                                axis_dir_sign = 1.0 if dx_y >= 0 else -1.0
                solid_start = sketch_start
                # profile_side: which side of the axis the profile lies on (+1 if
                # the profile coordinate is mostly > axis_offset, else -1).  The
                # partial-revolve wedge starts at the profile's plane, which is at
                # angle 0 for the +side and angle pi for the -side; this tells the
                # C++ node which.
                profile_side = 1.0
                if angle_deg < 359.999:
                    def _find_poly(d):
                        if isinstance(d, dict):
                            if d.get("type") in ("polygon", "arcpoly"):
                                return d
                            for v in d.values():
                                r = _find_poly(v)
                                if r is not None:
                                    return r
                        return None
                    coff = 0 if axis_along_x == 0 else 1   # vx for Y-axis, vy for X-axis
                    _ctr = None
                    _poly = _find_poly(sketch_desc)
                    if _poly is not None and sketch_start is not None:
                        nvt = _poly.get("n_vertices", 0)
                        cs = [self.params[sketch_start + 2 * i + coff]
                              for i in range(nvt)
                              if sketch_start + 2 * i + coff < len(self.params)]
                        if cs:
                            _ctr = sum(cs) / len(cs)
                    elif (isinstance(sketch_desc, dict)
                          and sketch_desc.get("type") in ("circle", "rect", "slot")
                          and sketch_start is not None
                          and sketch_start + coff < len(self.params)):
                        # circle/rect/slot store (cx, cy) first -> profile centre.
                        # Previously only polygon/arcpoly got a profile_side, so a
                        # partial revolve of a circle/rect on the -side of the axis
                        # oriented the wedge wrong -> missing half the solid (FN).
                        _ctr = self.params[sketch_start + coff]
                    if _ctr is not None and _ctr < axis_offset:
                        profile_side = -1.0
                if (isinstance(sketch_desc, dict)
                        and sketch_desc.get("type") in ("polygon", "arcpoly")
                        and sketch_start is not None):
                    coff = 0 if axis_along_x == 0 else 1
                    idx = [sketch_start + 2 * i + coff
                           for i in range(sketch_desc.get("n_vertices", 0))]
                    idx = [i for i in idx if i < len(self.params)]
                    rs = [self.params[i] - axis_offset for i in idx]
                    # a profile already straddling the axis has no side to keep
                    if rs and (min(rs) >= 0 or max(rs) <= 0) and any(rs):
                        # idx: radial coordinate of each vertex, then the
                        # axial one (same vertex order) — one list, so the
                        # index shifts in _insert_param cover both
                        self.revolve_bounds.append(
                            {"kind": "poly",
                             "idx": idx + [i + 1 - 2 * coff for i in idx],
                             "axis": axis_offset,
                             "side": 1.0 if max(rs) > 0 else -1.0})
                elif (isinstance(sketch_desc, dict)
                        and sketch_desc.get("type") == "rect"
                        and sketch_start is not None
                        and sketch_start + 3 < len(self.params)):
                    # rect params: cx, cy, half_w, half_h
                    coff = 0 if axis_along_x == 0 else 1
                    c = self.params[sketch_start + coff] - axis_offset
                    half = self.params[sketch_start + 2 + coff]
                    if abs(c) >= half - 1e-9 and c != 0:
                        self.revolve_bounds.append(
                            {"kind": "rect",
                             "idx": [sketch_start + coff, sketch_start + 2 + coff],
                             "axis": axis_offset, "side": 1.0 if c > 0 else -1.0})
                # Stash this revolve's world-space axis (point on axis + unit dir)
                # so a following circular_pattern can pattern features around it.
                try:
                    _pl = getattr(self, '_current_wp_plane', 'XY') or 'XY'
                    _ori = getattr(self, '_current_wp_origin', None) or (0.0, 0.0, 0.0)
                    _sx, _sy, _nrm = _PLANE_BASIS.get(_pl, _PLANE_BASIS['XY'])
                    if axis_along_x == 0:        # axis runs along sketch-Y
                        _ndir, _perp = _sy, _sx
                    else:                         # axis runs along sketch-X
                        _ndir, _perp = _sx, _sy
                    _c = tuple(_ori[i] + axis_offset * _perp[i] for i in range(3))
                    self._last_revolve_axis = (_c, _ndir, angle_deg)
                except Exception:
                    pass
                solid_desc = {"type": "revolve", "child": sketch_desc,
                              "axis_along_x": axis_along_x,
                              "axis_offset": axis_offset,
                              "angle_deg": angle_deg,
                              "profile_side": profile_side,
                              # wedge_b_sign: rotational direction of the partial-
                              # revolve sweep.  Empirically calibrated against
                              # cadquery renders over plane x axis x side x dir
                              # (all 3 planes agree): -profile_side*axis_dir for a
                              # sketch-Y axis, +profile_side*axis_dir for a sketch-X
                              # axis.  (The earlier -profile_side*axis_dir rule
                              # missed the axis_along_x flip and regressed X-axis
                              # partial revolves like 06_revolve_01.)
                              "wedge_b_sign": (-profile_side * axis_dir_sign
                                               * (1.0 if axis_along_x == 0 else -1.0))}
                solid_desc = self._wrap_with_current_wp(solid_desc, solid_start)
                sketch_desc = None

            elif method == 'cutThruAll':
                # Cut the current sketch through the entire existing solid.
                # We model it as extruding the sketch by a generously large
                # amount centred on the workplane (so it punches through in
                # both directions) and subtracting from the current solid.
                if sketch_desc is None or solid_desc is None:
                    raise ValueError("cutThruAll() without sketch or base solid")
                BIG = 1000.0
                self._insert_param(sketch_start, "frozen.cta_depth", BIG, _SyntheticNode())
                cutter = {"type": "extrude", "child": sketch_desc}
                # Centre the cutter on z=0 by translating down BIG/2.
                self._insert_param(sketch_start, "frozen.cta_tz", -BIG / 2.0, _SyntheticNode())
                self._insert_param(sketch_start, "frozen.cta_ty", 0.0, _SyntheticNode())
                self._insert_param(sketch_start, "frozen.cta_tx", 0.0, _SyntheticNode())
                cutter = {"type": "translate3d", "child": cutter}

                pending_sub = getattr(self, '_pending_sub_primitives', [])
                if pending_sub:
                    cutter = self._apply_sub_primitives(cutter, sketch_start,
                                                       pending_sub, BIG)
                    self._pending_sub_primitives = []

                # Apply current workplane transform so the cutter is in the
                # correct world frame (handles copyWorkplane).
                cutter = self._wrap_with_current_wp(cutter, sketch_start)

                solid_desc = {"type": "subtract", "left": solid_desc, "right": cutter}
                sketch_desc = None

            elif method == 'cutBlind':
                # Blind cut: extrude the current sketch by the (signed) distance
                # and subtract from the existing solid.  Unlike cutThruAll the
                # depth is FINITE and TUNABLE, so the optimizer can fit the cut
                # depth — the colleague's "cut depth never moves" was this op
                # being silently dropped (no handler).  Mirrors the extrude
                # combine='s' path exactly: ExtrudeNode supports signed depth
                # (z ∈ [min(0,d), max(0,d)] in the workplane-local frame), so a
                # negative distance cuts downward into the solid below the face.
                if sketch_desc is None or solid_desc is None:
                    raise ValueError("cutBlind() without sketch or base solid")
                d = _extract_num(args[0])
                if d is None:
                    raise ValueError("cutBlind() distance must be numeric literal")
                use_d = d  # signed!
                self._insert_param(sketch_start, "cutBlind.depth", use_d, args[0])
                new_solid_start = sketch_start
                cutter = {"type": "extrude", "child": sketch_desc}
                sketch_desc = None
                pending_sub = getattr(self, '_pending_sub_primitives', [])
                if pending_sub:
                    cutter = self._apply_sub_primitives(cutter, new_solid_start,
                                                        pending_sub, abs(use_d))
                    self._pending_sub_primitives = []
                cutter = self._wrap_with_current_wp(cutter, new_solid_start)
                solid_desc = {"type": "subtract", "left": solid_desc, "right": cutter}

            elif method == 'box':
                solid_start = len(self.params)
                l = _extract_num(args[0])
                w = _extract_num(args[1])
                h = _extract_num(args[2])
                if None in (l, w, h):
                    raise ValueError("box() args must be numeric literals")
                self._add_param("box.cx", 0.0, node)
                self._add_param("box.cy", 0.0, node)
                self._add_param("box.cz", 0.0, node)
                self._add_param("box.hx", l / 2.0, args[0])
                self._add_param("box.hy", w / 2.0, args[1])
                self._add_param("box.hz", h / 2.0, args[2])
                solid_desc = {"type": "box"}
                # World bbox (origin-centred) so a following .faces(">Z").workplane()
                # can place the sketch plane on the real face, not the root z=0.
                current_bbox = (-l/2.0, l/2.0, -w/2.0, w/2.0, -h/2.0, h/2.0)

            elif method == 'cylinder':
                solid_start = len(self.params)
                h = _extract_num(args[0])
                r = _extract_num(args[1])
                if None in (h, r):
                    raise ValueError("cylinder() args must be numeric literals")
                self._add_param("cylinder.cx", 0.0, node)
                self._add_param("cylinder.cy", 0.0, node)
                self._add_param("cylinder.cz", 0.0, node)
                self._add_param("cylinder.r", r, args[1])
                self._add_param("cylinder.hh", h / 2.0, args[0])
                solid_desc = {"type": "cylinder"}
                current_bbox = (-r, r, -r, r, -h/2.0, h/2.0)

            elif method == 'sphere':
                solid_start = len(self.params)
                r = _extract_num(args[0])
                if r is None:
                    raise ValueError("sphere() arg must be numeric literal")
                self._add_param("sphere.cx", 0.0, node)
                self._add_param("sphere.cy", 0.0, node)
                self._add_param("sphere.cz", 0.0, node)
                self._add_param("sphere.r", r, args[0])
                solid_desc = {"type": "sphere"}
                current_bbox = (-r, r, -r, r, -r, r)

            elif method == 'hole':
                if solid_desc is None:
                    raise ValueError("hole() without a solid")
                d = _extract_num(args[0])
                if d is None:
                    raise ValueError("hole() diameter must be numeric literal")
                # DEDUPE TIE: scripts (esp. AI-generated cadgen) sometimes
                # emit identical .hole(d, depth) calls multiple times
                # (e.g., 4 copies of .hole(16,30) at same workplane).
                # Each repeat is a no-op in cadquery.  Tie all to the FIRST
                # occurrence's params so Adam tunes one set and write-back
                # syncs the source literals.
                depth_lit = _extract_num(args[1]) if len(args) > 1 else None
                this_key = (d, depth_lit,
                            getattr(self, '_current_wp_plane', 'XY'),
                            getattr(self, '_current_wp_origin', None),
                            id(pending_selector) if pending_selector else None)
                hole_map = getattr(self, '_hole_key_to_idx', None)
                if hole_map is None:
                    hole_map = {}
                    self._hole_key_to_idx = hole_map
                dedupe_tie_to = hole_map.get(this_key)
                if dedupe_tie_to is None:
                    hole_map[this_key] = len(self.params)
                # hole = subtract(solid, cylinder)
                # Determine hole axis from face selector
                hole_axis = 'Z'  # default
                if pending_selector and pending_selector.get("kind") == "faces":
                    hole_axis = pending_selector.get("axis", "Z")
                pending_selector = None

                depth = _extract_num(args[1]) if len(args) > 1 else 100.0
                this_hole_start = len(self.params)
                face_frame = (getattr(self, '_current_wp_plane', None) == 'CUSTOM'
                              and getattr(self, '_current_wp_euler', None) is not None)
                if face_frame:
                    # Hole on a RESOLVED face workplane (02_cut etc.): place at the
                    # in-plane offset (pushPoints -> pending_center) and bore into the
                    # solid along the face normal (local -Z), then apply the face frame.
                    # Previously the cylinder sat at the ROOT frame -> the hole was
                    # misplaced and the optimiser distorted the cut.  Gated on CUSTOM
                    # so .hole() on the root frame (07_hole) is unchanged.
                    self._add_param("hole.cx", pending_center[0], node)
                    self._add_param("hole.cy", pending_center[1], node)
                    self._add_param("hole.cz", -depth / 2.0, node)
                    self._add_param("hole.r", d / 2.0, args[0])
                    self._add_param("hole.hh", depth / 2.0, args[1] if len(args) > 1 else node)
                    if dedupe_tie_to is not None:
                        for k in range(5):
                            self.tied_slots.append((dedupe_tie_to + k, this_hole_start + k))
                    hole_desc = {"type": "cylinder"}
                    hole_desc = self._wrap_with_current_wp(hole_desc, this_hole_start)
                    pending_center = (0.0, 0.0)
                else:
                    self._add_param("hole.cx", 0.0, node)
                    self._add_param("hole.cy", 0.0, node)
                    self._add_param("hole.cz", 0.0, node)
                    self._add_param("hole.r", d / 2.0, args[0])
                    self._add_param("hole.hh", depth / 2.0, args[1] if len(args) > 1 else node)
                    # Tie this hole's params to the previous identical hole's so
                    # Adam only tunes one set; write-back syncs the source literals.
                    if dedupe_tie_to is not None:
                        for k in range(5):
                            self.tied_slots.append((dedupe_tie_to + k, this_hole_start + k))
                    hole_desc = {"type": "cylinder"}
                    # For non-Z holes, wrap cylinder in rotate3d to orient along the correct axis
                    if hole_axis == 'X':
                        import math
                        hole_start = len(self.params) - 5  # start of cylinder params
                        self._insert_param(hole_start, "hole_rot.z", 0.0, node)
                        self._insert_param(hole_start, "hole_rot.y", math.pi / 2.0, node)
                        self._insert_param(hole_start, "hole_rot.x", 0.0, node)
                        hole_desc = {"type": "rotate3d", "child": hole_desc}
                    elif hole_axis == 'Y':
                        import math
                        hole_start = len(self.params) - 5
                        self._insert_param(hole_start, "hole_rot.z", 0.0, node)
                        self._insert_param(hole_start, "hole_rot.y", 0.0, node)
                        self._insert_param(hole_start, "hole_rot.x", -math.pi / 2.0, node)
                        hole_desc = {"type": "rotate3d", "child": hole_desc}

                solid_desc = {"type": "subtract", "left": solid_desc, "right": hole_desc}
                # solid_start unchanged (subtract has 0 self-params)

            elif method == 'cboreHole':
                if solid_desc is None:
                    raise ValueError("cboreHole() without a solid")
                # cboreHole(holeDiam, cboreDiam, cboreDepth [, depth])
                hd = _extract_num(args[0])
                cbd = _extract_num(args[1])
                cbdepth = _extract_num(args[2])
                if None in (hd, cbd, cbdepth):
                    raise ValueError("cboreHole() args must be numeric literals")
                depth = _extract_num(args[3]) if len(args) > 3 else 100.0
                pending_selector = None

                # Frame-aware placement (like .hole()): a counterbore on a RESOLVED
                # face workplane must bore INTO the face along its normal at the
                # pushPoints offset.  Was: both cylinders at the ROOT frame, Z axis,
                # centred at origin -> a vertical shaft through the whole part
                # (11_combined_29).  Gated on CUSTOM so root-frame Z cbores are
                # unchanged.
                cbore_start = len(self.params)
                face_frame = (getattr(self, '_current_wp_plane', None) == 'CUSTOM'
                              and getattr(self, '_current_wp_euler', None) is not None)
                if face_frame:
                    cx, cy = pending_center[0], pending_center[1]
                    cz_main = -depth / 2.0      # through-hole below the face
                    cz_pocket = -cbdepth / 2.0  # shallow pocket at the face top
                else:
                    cx = cy = cz_main = cz_pocket = 0.0

                # Main through-hole cylinder
                self._add_param("cbore_hole.cx", cx, node)
                self._add_param("cbore_hole.cy", cy, node)
                self._add_param("cbore_hole.cz", cz_main, node)
                self._add_param("cbore_hole.r", hd / 2.0, args[0])
                self._add_param("cbore_hole.hh", depth / 2.0, args[3] if len(args) > 3 else node)
                main_hole = {"type": "cylinder"}

                # Counterbore cylinder (wider, shallow, at the face top)
                self._add_param("cbore_pocket.cx", cx, node)
                self._add_param("cbore_pocket.cy", cy, node)
                self._add_param("cbore_pocket.cz", cz_pocket, node)
                self._add_param("cbore_pocket.r", cbd / 2.0, args[1])
                self._add_param("cbore_pocket.hh", cbdepth / 2.0, args[2])
                cbore_cyl = {"type": "cylinder"}

                # Union of both holes, then subtract from solid
                holes_union = {"type": "union", "left": main_hole, "right": cbore_cyl}
                if face_frame:
                    holes_union = self._wrap_with_current_wp(holes_union, cbore_start)
                    pending_center = (0.0, 0.0)
                solid_desc = {"type": "subtract", "left": solid_desc, "right": holes_union}

            elif method == 'shell':
                if solid_desc is None:
                    raise ValueError("shell() without a solid")
                t = _extract_num(args[0])
                if t is None:
                    raise ValueError("shell() arg must be numeric literal")
                # INSERT thickness BEFORE child params (DFS: parent first)
                self._insert_param(solid_start, "shell.thickness", t, args[0])
                solid_desc = {"type": "shell", "child": solid_desc}

            elif method == 'cut':
                if solid_desc is None:
                    raise ValueError("cut() without a solid")
                sub_parser = CadQueryParser()
                sub_parser._assigns_seq = self._assigns_seq; sub_parser._resolution_index = self._resolution_index
                sub_chain = sub_parser._unroll_chain(args[0])
                if sub_chain:
                    right_desc = sub_parser._process_chain(sub_chain)
                    # Append right child params (left already in place)
                    self._take_revolve_bounds(sub_parser)
                    self._take_sub_ties(sub_parser)
                    self.params.extend(sub_parser.params)
                    self.slots.extend(sub_parser.slots)
                    self.names.extend(sub_parser.names)
                    solid_desc = {"type": "subtract", "left": solid_desc, "right": right_desc}
                    # solid_start unchanged (subtract has 0 self-params)

            elif method == 'union':
                if solid_desc is None:
                    raise ValueError("union() without a solid")
                sub_parser = CadQueryParser()
                sub_parser._assigns_seq = self._assigns_seq; sub_parser._resolution_index = self._resolution_index
                sub_chain = sub_parser._unroll_chain(args[0])
                if sub_chain:
                    right_desc = sub_parser._process_chain(sub_chain)
                    # DEDUPE TIE: identical .union(<chain>) calls (same params,
                    # same tree shape) duplicate the geometry — no-op for
                    # cadquery but creates separate tunable param sets that
                    # Adam drifts apart.  Hash (params, tree_desc) and tie
                    # all redundant calls to the first occurrence.
                    def _tree_str(d):
                        if not isinstance(d, dict): return repr(d)
                        keys = sorted(k for k in d if k not in ('left','right','child'))
                        parts = [f'{k}={d[k]}' for k in keys]
                        if 'left' in d:  parts.append('L:'+_tree_str(d['left']))
                        if 'right' in d: parts.append('R:'+_tree_str(d['right']))
                        if 'child' in d: parts.append('C:'+_tree_str(d['child']))
                        return '{'+','.join(parts)+'}'
                    key = (tuple(round(p,9) for p in sub_parser.params), _tree_str(right_desc))
                    union_map = getattr(self, '_union_key_to_idx', None)
                    if union_map is None:
                        union_map = {}; self._union_key_to_idx = union_map
                    first_idx = union_map.get(key)
                    sub_start = len(self.params)
                    self._take_revolve_bounds(sub_parser)
                    self._take_sub_ties(sub_parser)
                    self.params.extend(sub_parser.params)
                    self.slots.extend(sub_parser.slots)
                    self.names.extend(sub_parser.names)
                    if first_idx is not None:
                        n = len(sub_parser.params)
                        for k in range(n):
                            self.tied_slots.append((first_idx + k, sub_start + k))
                    else:
                        union_map[key] = sub_start
                    solid_desc = {"type": "union", "left": solid_desc, "right": right_desc}

            elif method == 'attach_at':
                # cadgen attach_at: apply a chain string at each anchor point.
                # Signature: attach_at(anchors_list, chain_str, combine='s'/'a').
                # We model this by, for each anchor, synthesising
                #   cq.Workplane('XY', origin=(x,y,z)) + chain_str
                # and parsing it as a sub-chain.  Each sub-result is then
                # combined into solid_desc via subtract/union.
                if solid_desc is None or len(args) < 2:
                    continue
                # arg0: list of [x,y,z] anchors.  Elements may be literal
                # lists OR variable names bound earlier (point_sel0 = [-36,1,41];
                # r.attach_at([point_sel0], ...)) -- resolve names through the
                # assignment map (was: Name elements silently dropped the whole
                # attach_at -> loft_10 cuts missing).
                anchors = []
                if isinstance(args[0], ast.List):
                    _amap = {n: v for n, v in getattr(self, '_assigns_seq', [])}
                    for elt in args[0].elts:
                        if isinstance(elt, ast.Name) and elt.id in _amap:
                            elt = _amap[elt.id]
                        if isinstance(elt, ast.List) and len(elt.elts) == 3:
                            xs = _extract_num(elt.elts[0])
                            ys = _extract_num(elt.elts[1])
                            zs = _extract_num(elt.elts[2])
                            if None not in (xs, ys, zs):
                                anchors.append((xs, ys, zs, elt.elts[0],
                                                elt.elts[1], elt.elts[2]))
                # arg1: chain string literal
                chain_s = None
                if isinstance(args[1], ast.Constant) and isinstance(args[1].value, str):
                    chain_s = args[1].value
                # combine kwarg
                a_combine = 's'
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'combine' and isinstance(kw.value, ast.Constant):
                        a_combine = 's' if kw.value.value == 's' else 'a'
                if not anchors or chain_s is None:
                    continue
                # Resolve the TRUE face tangent frame per anchor: cadquery_addons
                # attach_at builds the sub-chain on the surface-normal frame at the
                # anchor (tangent_plane_from_face_at_point), NOT on a root-XY plane
                # at the anchor point -- on curved faces the XY fallback mis-orients
                # the feature (loft_10 cuts).  Measure the frame exactly like the
                # helix anchor does: exec the prefix program and call the addons'
                # own frame function.  Gated + try/except: on any failure fall back
                # to the previous XY-at-anchor synthesis.
                _aframes = {}
                import os as _os_a
                if not _os_a.environ.get('CAD_NO_ATTACH_FRAME'):
                    try:
                        import cadquery as _cqm
                        from cadquery_addons.selectors import PointOnFaceSelector as _POFS
                        from cadquery_addons.attach_array import (
                            tangent_plane_from_face_at_point as _TPF)
                        import cadquery_addons  # noqa: registers attach_at & friends
                        _src = getattr(self, '_src_code', None)
                        _ln = getattr(node, 'lineno', None)
                        if _src and _ln:
                            _ns = {'cq': _cqm, 'PointOnFaceSelector': _POFS}
                            try:
                                from cadquery_addons.selectors import (
                                    PointOnEdgeSelector as _POES)
                                _ns['PointOnEdgeSelector'] = _POES
                            except Exception:
                                pass
                            exec('\n'.join(_src.splitlines()[:_ln - 1]), _ns)
                            _base = eval(ast.get_source_segment(_src, node.func.value), _ns)
                            # x_hint kwarg (literal tuple), default (0,0,1)
                            _xh = (0.0, 0.0, 1.0)
                            for kw in getattr(node, 'keywords', []):
                                if kw.arg == 'x_hint':
                                    try:
                                        _xh = ast.literal_eval(kw.value)
                                    except Exception:
                                        pass
                            _sgn = 1 if a_combine == 'a' else -1
                            for (_ax, _ay, _az, *_rest) in anchors:
                                _face = _base.faces(_POFS([_ax, _ay, _az])).val()
                                _pln = _TPF(_face, _cqm.Vector(_ax, _ay, _az),
                                            x_hint=_cqm.Vector(*_xh), normal=_sgn)
                                _aframes[(_ax, _ay, _az)] = (
                                    _pln.origin, _pln.xDir, _pln.zDir)
                    except Exception:
                        _aframes = {}
                # For each anchor, build a synthetic chain and merge.
                for ax, ay, az, ax_n, ay_n, az_n in anchors:
                    _fr = _aframes.get((ax, ay, az))
                    if _fr is not None:
                        _o, _x, _n = _fr
                        syn_code = (
                            f"cq.Workplane(cq.Plane(origin=cq.Vector({_o.x!r},{_o.y!r},{_o.z!r}),"
                            f" xDir=cq.Vector({_x.x!r},{_x.y!r},{_x.z!r}),"
                            f" normal=cq.Vector({_n.x!r},{_n.y!r},{_n.z!r})))"
                            f"{chain_s}")
                    else:
                        syn_code = (f"cq.Workplane('XY', origin=({ax},{ay},{az}))"
                                    f"{chain_s}")
                    try:
                        syn_ast = ast.parse(syn_code, mode='eval').body
                    except SyntaxError:
                        continue
                    sub_parser = CadQueryParser()
                    sub_parser._assigns_seq = self._assigns_seq
                    sub_parser._resolution_index = self._resolution_index
                    sub_chain = sub_parser._unroll_chain(syn_ast)
                    if not sub_chain:
                        continue
                    try:
                        right_desc = sub_parser._process_chain(sub_chain)
                    except Exception:
                        continue
                    if right_desc is None:
                        continue
                    # The sub-chain's params have no source positions to
                    # write back to.  If Adam tunes them, our SDF model
                    # would diverge from cadquery's render of the ORIGINAL
                    # (unchanged) attach_at line — making IoU drop while
                    # SDF loss decreases.  Freeze them entirely (prefix
                    # name with 'frozen.' so the optimizer's freeze mask
                    # excludes them) and mark slots as synthetic so write-
                    # back also skips them.
                    self._take_revolve_bounds(sub_parser)
                    self.params.extend(sub_parser.params)
                    for s in sub_parser.slots:
                        s.line = 0; s.col = 0; s.end_col = 0
                    self.slots.extend(sub_parser.slots)
                    self.names.extend(['frozen.attach.' + n for n in sub_parser.names])
                    if a_combine == 's':
                        solid_desc = {"type": "subtract",
                                      "left": solid_desc, "right": right_desc}
                    else:
                        solid_desc = {"type": "union",
                                      "left": solid_desc, "right": right_desc}

            elif method == 'circular_pattern':
                # cadquery_addons circular_pattern(point, span, n, expr, combine):
                # at runtime it finds the face at `point`, walks n points around
                # its U-parameter (= n points equally spaced around the base
                # revolve axis over `span` deg), and attaches sub-program `expr`
                # on each point's surface tangent frame (radial normal).  We model
                # it as N copies of `expr`, each built on a cq.Plane whose origin
                # lies on the ring and whose normal is radial, combined with base.
                import math as _m
                axis = getattr(self, '_last_revolve_axis', None)
                if solid_desc is None or axis is None or len(args) < 4:
                    continue
                if not (isinstance(args[0], ast.List) and len(args[0].elts) == 3):
                    continue
                sx_ = _extract_num(args[0].elts[0]); sy_ = _extract_num(args[0].elts[1])
                sz_ = _extract_num(args[0].elts[2])
                span = _extract_num(args[1]); ncp = _extract_num(args[2])
                if None in (sx_, sy_, sz_, span, ncp) or int(ncp) < 1:
                    continue
                ncp = int(ncp)
                expr = (args[3].value if isinstance(args[3], ast.Constant)
                        and isinstance(args[3].value, str) else None)
                if not expr or not expr.lstrip().startswith('.'):
                    continue
                cmb = 'a'
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'combine' and isinstance(kw.value, ast.Constant):
                        cmb = 's' if kw.value.value == 's' else 'a'
                (cx, cy, cz), (nx, ny, nz), rev_ang = axis
                _nl = _m.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
                nx, ny, nz = nx/_nl, ny/_nl, nz/_nl
                rx, ry, rz = sx_-cx, sy_-cy, sz_-cz
                along = rx*nx + ry*ny + rz*nz
                px, py, pz = rx-along*nx, ry-along*ny, rz-along*nz
                radius = _m.sqrt(px*px + py*py + pz*pz)
                if radius < 1e-6:
                    continue
                r0 = (px/radius, py/radius, pz/radius)
                t0 = (ny*r0[2]-nz*r0[1], nz*r0[0]-nx*r0[2], nx*r0[1]-ny*r0[0])
                bc = (cx+along*nx, cy+along*ny, cz+along*nz)
                # Surface normal of the BASE solid at the seed, from its SDF
                # gradient: a hole on a thin washer's FLAT face must punch axially,
                # not radially.  The gradient gives the true face normal (flat ->
                # axial, cylindrical -> radial).  Fall back to radial if the C++
                # backend isn't importable (parser used standalone) or degenerate.
                nrm0 = r0
                try:
                    import _cad_grad as _cg
                    _bt = _cg.create_tree(solid_desc); _bt.set_params(self.params)
                    _e = 1e-2
                    _g = (_bt.eval_sdf(sx_+_e, sy_, sz_) - _bt.eval_sdf(sx_-_e, sy_, sz_),
                          _bt.eval_sdf(sx_, sy_+_e, sz_) - _bt.eval_sdf(sx_, sy_-_e, sz_),
                          _bt.eval_sdf(sx_, sy_, sz_+_e) - _bt.eval_sdf(sx_, sy_, sz_-_e))
                    _gl = _m.sqrt(_g[0]**2 + _g[1]**2 + _g[2]**2)
                    if _gl > 1e-6:
                        nrm0 = (_g[0]/_gl, _g[1]/_gl, _g[2]/_gl)
                except Exception:
                    pass
                # The runtime walks the base face's U-range (= the revolve's own
                # angular extent), taking the span/360 fraction of it.  So a 136-deg
                # partial-revolve base spreads its copies over ~136 deg, not 360.
                _ext = rev_ang * (1.0 if span >= 359.0 else span / 360.0)
                _step = _ext / ncp if span >= 359.0 else _ext / max(ncp - 1, 1)
                angles = [j * _step for j in range(ncp)]
                for th in angles:
                    a = _m.radians(th); ca, sa = _m.cos(a), _m.sin(a)
                    rh = (ca*r0[0]+sa*t0[0], ca*r0[1]+sa*t0[1], ca*r0[2]+sa*t0[2])
                    ox = bc[0]+radius*rh[0]; oy = bc[1]+radius*rh[1]; oz = bc[2]+radius*rh[2]
                    # Rodrigues-rotate the seed normal around the axis by th.
                    _cdv = nx*nrm0[0] + ny*nrm0[1] + nz*nrm0[2]
                    _cr = (ny*nrm0[2]-nz*nrm0[1], nz*nrm0[0]-nx*nrm0[2], nx*nrm0[1]-ny*nrm0[0])
                    nk = (nrm0[0]*ca + _cr[0]*sa + nx*_cdv*(1-ca),
                          nrm0[1]*ca + _cr[1]*sa + ny*_cdv*(1-ca),
                          nrm0[2]*ca + _cr[2]*sa + nz*_cdv*(1-ca))
                    nd = nk if cmb == 'a' else (-nk[0], -nk[1], -nk[2])
                    xd = (ny*nd[2]-nz*nd[1], nz*nd[0]-nx*nd[2], nx*nd[1]-ny*nd[0])
                    _xl = _m.sqrt(xd[0]**2+xd[1]**2+xd[2]**2)
                    if _xl < 1e-9:        # normal parallel to axis -> use radial as xDir
                        xd, _xl = rh, 1.0
                    xd = (xd[0]/_xl, xd[1]/_xl, xd[2]/_xl)
                    synth = (f"cq.Workplane(cq.Plane("
                             f"origin=cq.Vector({ox:.6g},{oy:.6g},{oz:.6g}),"
                             f"xDir=cq.Vector({xd[0]:.6g},{xd[1]:.6g},{xd[2]:.6g}),"
                             f"normal=cq.Vector({nd[0]:.6g},{nd[1]:.6g},{nd[2]:.6g}))){expr}")
                    try:
                        syn_ast = ast.parse(synth, mode='eval').body
                    except SyntaxError:
                        continue
                    sub_parser = CadQueryParser()
                    sub_parser._assigns_seq = self._assigns_seq
                    sub_parser._resolution_index = self._resolution_index
                    sub_chain = sub_parser._unroll_chain(syn_ast)
                    if not sub_chain:
                        continue
                    try:
                        right_desc = sub_parser._process_chain(sub_chain)
                    except Exception:
                        continue
                    if right_desc is None:
                        continue
                    # Freeze all copy params (placement + feature dims): structural,
                    # no source literal to write back to, shared across copies.
                    self._take_revolve_bounds(sub_parser)
                    self.params.extend(sub_parser.params)
                    for s in sub_parser.slots:
                        s.line = 0; s.col = 0; s.end_col = 0
                    self.slots.extend(sub_parser.slots)
                    self.names.extend(['frozen.cpat.' + nm for nm in sub_parser.names])
                    if cmb == 's':
                        solid_desc = {"type": "subtract", "left": solid_desc, "right": right_desc}
                    else:
                        solid_desc = {"type": "union", "left": solid_desc, "right": right_desc}

            elif method == 'intersect':
                if solid_desc is None:
                    raise ValueError("intersect() without a solid")
                sub_parser = CadQueryParser()
                sub_parser._assigns_seq = self._assigns_seq; sub_parser._resolution_index = self._resolution_index
                sub_chain = sub_parser._unroll_chain(args[0])
                if sub_chain:
                    right_desc = sub_parser._process_chain(sub_chain)
                    self._take_revolve_bounds(sub_parser)
                    self._take_sub_ties(sub_parser)
                    self.params.extend(sub_parser.params)
                    self.slots.extend(sub_parser.slots)
                    self.names.extend(sub_parser.names)
                    solid_desc = {"type": "intersect", "left": solid_desc, "right": right_desc}

            elif method == 'translate':
                if solid_desc is None:
                    raise ValueError("translate() without a solid")
                tup = args[0]
                if isinstance(tup, ast.Tuple) and len(tup.elts) == 3:
                    tx = _extract_num(tup.elts[0])
                    ty = _extract_num(tup.elts[1])
                    tz = _extract_num(tup.elts[2])
                    if None in (tx, ty, tz):
                        raise ValueError("translate args must be numeric")
                    # INSERT translate params BEFORE child params (DFS: parent first)
                    self._insert_param(solid_start, "translate.z", tz, tup.elts[2])
                    self._insert_param(solid_start, "translate.y", ty, tup.elts[1])
                    self._insert_param(solid_start, "translate.x", tx, tup.elts[0])
                    solid_desc = {"type": "translate3d", "child": solid_desc}

            elif method == 'mirror':
                if solid_desc is None:
                    raise ValueError("mirror() without a solid")
                plane = args[0]
                if isinstance(plane, ast.Constant):
                    p = plane.value
                    if p == "XY":
                        solid_desc = {"type": "mirror_xy", "child": solid_desc}
                    elif p == "XZ":
                        solid_desc = {"type": "mirror_xz", "child": solid_desc}
                    elif p == "YZ":
                        solid_desc = {"type": "mirror_yz", "child": solid_desc}
                # Mirror has 0 self-params, solid_start unchanged

            elif method == 'offset2D':
                # offset2D(amount) — expand or shrink a 2D sketch boundary
                if sketch_desc is None:
                    raise ValueError("offset2D() without a sketch")
                amt = _extract_num(args[0])
                if amt is None:
                    raise ValueError("offset2D() arg must be numeric literal")
                # INSERT offset BEFORE sketch params (DFS: parent first)
                self._insert_param(sketch_start, "offset2d.amount", amt, args[0])
                sketch_desc = {"type": "offset", "child": sketch_desc}

            elif method == 'spline':
                # spline(pts) — approximate spline with polyline segments
                if not args:
                    raise ValueError("spline() needs a list argument")
                lst = args[0]
                control_pts = []
                if isinstance(lst, ast.List):
                    for elt in lst.elts:
                        if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                            x = _extract_num(elt.elts[0])
                            y = _extract_num(elt.elts[1])
                            if x is None or y is None:
                                raise ValueError("spline coords must be numeric")
                            control_pts.append((x, y))
                if not control_pts:
                    raise ValueError("spline() needs at least 2 control points")

                # If there are pending vertices, include them as start of spline
                if pending_vertices:
                    start_pt = (pending_vertices[-1][0], pending_vertices[-1][1])
                    all_pts = [start_pt] + control_pts
                else:
                    all_pts = control_pts
                    pending_vertices = []

                # Approximate spline with Catmull-Rom polyline
                spline_pts = _spline_to_points(all_pts, n_seg_per_span=8)
                for px, py in spline_pts:
                    pending_vertices.append((px, py, lst, lst))

            elif method == 'loft':
                # loft() — create solid by interpolating between two cross-sections
                # The first sketch goes to bottom, current sketch goes to top
                if sketch_desc is None and not loft_sketches:
                    raise ValueError("loft() without sketches")

                if len(loft_sketches) >= 1 and sketch_desc is not None:
                    # All cross-sections: accumulated ones (with their z) + the
                    # current one at the top (z = loft_height).
                    profs = list(loft_sketches) + [(sketch_desc, sketch_start,
                                  len(self.params) - sketch_start, loft_height)]
                    solid_start = profs[0][1]
                    if len(profs) == 2:
                        # Single linear loft, profiles stay optimisable (as before;
                        # the model centres these at the origin so no extra wrap).
                        hh = loft_height / 2.0 if loft_height > 0 else 0.5
                        self._insert_param(profs[0][1], "loft.hh", hh, node)
                        solid_desc = {"type": "loft", "left": profs[0][0], "right": profs[1][0]}
                    else:
                        # >=3 sections: chain one binary loft per consecutive pair and
                        # union them (LoftNode is binary).  Shared middle profiles must
                        # be DUPLICATED in the param vector, so freeze all loft params
                        # (structural; duplicated profiles have no unique source literal).
                        import copy as _copy
                        base = profs[0][1]
                        pdata = [(d, list(self.params[st:st+ct]), list(self.names[st:st+ct]), z)
                                 for (d, st, ct, z) in profs]
                        del self.params[base:]; del self.slots[base:]; del self.names[base:]
                        def _frz(nm, val):
                            self._add_param('frozen.loft.' + nm, val, _SyntheticNode())
                            self.slots[-1].line = 0; self.slots[-1].col = 0; self.slots[-1].end_col = 0
                        segs = []
                        for i in range(len(pdata) - 1):
                            d_lo, v_lo, n_lo, z_lo = pdata[i]
                            d_hi, v_hi, n_hi, z_hi = pdata[i+1]
                            hh = (z_hi - z_lo) / 2.0; zc = (z_lo + z_hi) / 2.0
                            if hh <= 1e-9: hh = 0.5
                            _frz('tx', 0.0); _frz('ty', 0.0); _frz('tz', zc); _frz('hh', hh)
                            for nm, val in zip(n_lo, v_lo): _frz(nm, val)
                            for nm, val in zip(n_hi, v_hi): _frz(nm, val)
                            segs.append({"type": "translate3d",
                                         "child": {"type": "loft",
                                                   "left": _copy.deepcopy(d_lo),
                                                   "right": _copy.deepcopy(d_hi)}})
                        solid_desc = segs[0]
                        for s in segs[1:]:
                            solid_desc = {"type": "union", "left": solid_desc, "right": s}
                        # Segments span local [0, total]; apply the workplane origin
                        # (which the model sets to ~-total/2) to centre it.
                        solid_desc = self._wrap_with_current_wp(solid_desc, base)
                    sketch_desc = None
                    loft_sketches = []
                    loft_height = 0.0
                elif sketch_desc is not None:
                    raise ValueError("loft() needs at least 2 cross-sections")

            elif method == 'sweep':
                # sweep(path_wire) — sweep profile along a path
                if sketch_desc is None:
                    raise ValueError("sweep() without a sketch")

                # cq.Wire.makeHelix(pitch, height, radius, center=, dir=) path:
                # springs/coils.  Generate a helical polyline (world coords) and
                # sweep the radius-r circle along it (was falling back to a straight
                # extrude -> a solid cylinder, faith 0.25).
                def _get_makehelix(call):
                    if not isinstance(call, ast.Call):
                        return None
                    fn = call.func
                    nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', '')
                    if nm != 'makeHelix':
                        return None
                    kw = {k.arg: k.value for k in call.keywords}
                    pos = list(call.args)
                    def g(key, idx):
                        if key in kw: return _extract_num(kw[key])
                        if idx < len(pos): return _extract_num(pos[idx])
                        return None
                    pitch = g('pitch', 0); height = g('height', 1); radius = g('radius', 2)
                    ang_t = g('angle', 5)
                    cen = _vec_args(kw['center'])[0] if 'center' in kw else (0.0, 0.0, 0.0)
                    drv = _vec_args(kw['dir'])[0] if 'dir' in kw else (0.0, 0.0, 1.0)
                    if None in (pitch, height, radius) or cen is None or drv is None:
                        return None
                    return pitch, height, radius, cen, drv, (ang_t or 0.0)
                helix = _get_makehelix(args[0]) if args else None
                if helix is not None:
                    import math as _m
                    pitch, height, radius, cen, drv, hangle = helix
                    dl = _m.sqrt(sum(c*c for c in drv)) or 1.0
                    du = tuple(c/dl for c in drv)
                    # Phase: cadquery attaches the profile at the helix start, so the
                    # phase-0 radial direction points from the helix axis to the
                    # profile's WORLD centre.  Matching it aligns the coils (an
                    # arbitrary phase gives balanced FP/FN ~0.85 faith).
                    pln = getattr(self, '_current_wp_plane', 'XY') or 'XY'
                    wo = getattr(self, '_current_wp_origin', None) or (0.0, 0.0, 0.0)
                    sxw, syw, _nw = _PLANE_BASIS.get(pln, _PLANE_BASIS['XY'])
                    cx0 = self.params[sketch_start] if sketch_start < len(self.params) else 0.0
                    cy0 = self.params[sketch_start+1] if sketch_start+1 < len(self.params) else 0.0
                    pw = (wo[0]+cx0*sxw[0]+cy0*syw[0], wo[1]+cx0*sxw[1]+cy0*syw[1], wo[2]+cx0*sxw[2]+cy0*syw[2])
                    rel = (pw[0]-cen[0], pw[1]-cen[1], pw[2]-cen[2])
                    pj = rel[0]*du[0]+rel[1]*du[1]+rel[2]*du[2]
                    r0 = (rel[0]-pj*du[0], rel[1]-pj*du[1], rel[2]-pj*du[2])
                    rl = _m.sqrt(r0[0]**2+r0[1]**2+r0[2]**2)
                    if rl > 1e-6:
                        u = (r0[0]/rl, r0[1]/rl, r0[2]/rl)
                    else:
                        axv = (1.0, 0.0, 0.0) if abs(du[0]) < 0.9 else (0.0, 1.0, 0.0)
                        cxp = (axv[1]*du[2]-axv[2]*du[1], axv[2]*du[0]-axv[0]*du[2], axv[0]*du[1]-axv[1]*du[0])
                        cl = _m.sqrt(sum(c*c for c in cxp)) or 1.0
                        u = tuple(c/cl for c in cxp)
                    v = (du[1]*u[2]-du[2]*u[1], du[2]*u[0]-du[0]*u[2], du[0]*u[1]-du[1]*u[0])
                    tan_a = _m.tan(_m.radians(hangle)) if 0.0 < abs(hangle) < 89.0 else 0.0
                    nturns = abs(height/pitch) if abs(pitch) > 1e-6 else 1.0
                    N = max(8, min(int(nturns*24)+1, 400))
                    pathv = []
                    for i in range(N+1):
                        ax_ = (i/float(N))*height
                        rr = radius + ax_*tan_a   # conical taper (angle param)
                        ang = 2.0*_m.pi*ax_/pitch if abs(pitch) > 1e-6 else 0.0
                        ca, sa = _m.cos(ang), _m.sin(ang)
                        pathv.append((cen[0]+ax_*du[0]+rr*(ca*u[0]+sa*v[0]),
                                      cen[1]+ax_*du[1]+rr*(ca*u[1]+sa*v[1]),
                                      cen[2]+ax_*du[2]+rr*(ca*u[2]+sa*v[2])))
                    # OCC's sweep places the swept solid axially offset from the
                    # makeHelix spine by a frame/pitch-dependent amount that is NOT
                    # predictable from the code (measured: -pitch/2..-pitch, and
                    # workplane-dependent).  Query OCC directly: rebuild THIS sweep,
                    # read the true solid centre, and rigidly shift the analytic path
                    # to match.  Falls back to the on-spine path if unavailable.
                    try:
                        import cadquery as _cq, numpy as _np, ast as _ast, os as _os
                        if _os.environ.get('CAD_NO_HELIX_ANCHOR'):
                            raise ValueError('anchor disabled')
                        _isfr = False
                        for _k in getattr(node, 'keywords', []):
                            if getattr(_k, 'arg', None) == 'isFrenet':
                                _isfr = bool(getattr(_k.value, 'value', False))
                        _w = _cq.Workplane(pln, origin=tuple(wo)).center(cx0, cy0)
                        _st = sketch_desc.get("type") if sketch_desc else None
                        if _st == 'rect':
                            _w = _w.rect(2*self.params[sketch_start+2], 2*self.params[sketch_start+3])
                        elif _st == 'circle':
                            _w = _w.circle(self.params[sketch_start+2])
                        else:
                            raise ValueError('helix profile not rebuildable')
                        _hel = eval(_ast.unparse(args[0]), {'cq': _cq})
                        # The offset is a pure axial translation, but centroid/bbox are
                        # biased when the analytic tube isn't a perfect translate of the
                        # solid.  Instead pick the axial shift that maximizes overlap of
                        # the path centerline with the ACTUAL swept solid (voxel-fill).
                        import trimesh as _tm
                        _shp = _w.sweep(_hel, isFrenet=_isfr).val()
                        _vs, _fs = _shp.tessellate(0.3)
                        _msh = _tm.Trimesh([[q.x, q.y, q.z] for q in _vs], _fs)
                        # coarse voxels are plenty for pitch-scale axial alignment and
                        # keep the parse fast (fine grids -> multi-second .fill() per parse)
                        _vpitch = max(float(_np.linalg.norm(_msh.extents))/55.0, 1.0)
                        _fill = _msh.voxelized(pitch=_vpitch).fill()
                        # Solve the AXIAL placement: the swept solid sits one frame-
                        # dependent axial offset from the makeHelix spine (measured, not
                        # predictable from the code).  Slide the path along the axis and
                        # take the centre of the largest max-count plateau of path-
                        # centerline-inside-solid (flat top = the coil's axial thickness).
                        # NB the AZIMUTHAL phase is left as-generated: for dense multi-turn
                        # round-wire springs it is only weakly observable (near-symmetric)
                        # and every cheap occupancy proxy mis-pins it (regressed sweep_23);
                        # voxel-fill IoU already masks it and pick-best covers the residual.
                        _pv = _np.array(pathv); _dun = _np.array(du)
                        _grid = _np.linspace(-1.5*pitch, 1.5*pitch, 241)
                        _cnt = _np.array([int(_fill.is_filled(_pv + _d*_dun).sum()) for _d in _grid])
                        _mx = _cnt.max(); _ismax = _cnt >= _mx
                        _brun = (0, 0, 1); _i = 0
                        while _i < len(_ismax):
                            if _ismax[_i]:
                                _j = _i
                                while _j < len(_ismax) and _ismax[_j]: _j += 1
                                if _j - _i > _brun[0]: _brun = (_j - _i, _i, _j)
                                _i = _j
                            else:
                                _i += 1
                        _dax = float(_grid[_brun[1]:_brun[2]].mean()); _sh = _dax * _dun
                        if _os.environ.get('CAD_HELIX_DEBUG'):
                            print('HELIX_ANCHOR axial shift =', round(_dax, 1),
                                  'plateau', _brun[0], 'inside', int(_mx), '/', len(_pv))
                        pathv = [(p[0]+_sh[0], p[1]+_sh[1], p[2]+_sh[2]) for p in pathv]
                    except Exception:
                        pass
                    # Tube is centred on the path -> zero the profile centre.
                    if sketch_start < len(self.params): self.params[sketch_start] = 0.0
                    if sketch_start+1 < len(self.params): self.params[sketch_start+1] = 0.0
                    n_path = len(pathv)
                    for i in range(n_path-1, -1, -1):
                        vx, vy, vz = pathv[i]
                        self._insert_param(sketch_start, 'frozen.sweep_path.z', vz, _SyntheticNode())
                        self._insert_param(sketch_start, 'frozen.sweep_path.y', vy, _SyntheticNode())
                        self._insert_param(sketch_start, 'frozen.sweep_path.x', vx, _SyntheticNode())
                    solid_start = sketch_start
                    # makeHelix springs are swept with a Frenet frame (isFrenet=True):
                    # the profile normal stays radial (rotates with the coil).  A
                    # rotation-minimizing frame mis-orients asymmetric profiles here.
                    solid_desc = {"type": "sweep", "n_path_verts": n_path,
                                  "frenet": True, "child": sketch_desc}
                    sketch_desc = None
                    continue   # helix path is in world coords -> no workplane wrap

                # Parse the path from the argument
                sub_parser = CadQueryParser()
                sub_parser._assigns_seq = self._assigns_seq; sub_parser._resolution_index = self._resolution_index
                sub_chain = sub_parser._unroll_chain(args[0])
                path_vertices = []
                if sub_chain:
                    # Determine path plane for coordinate mapping
                    # XZ plane: (u,v) -> (u, 0, v), XY: (u,v) -> (u, v, 0), YZ: (u,v) -> (0, u, v)
                    plane = "XY"
                    def _to_3d(u, v):
                        if plane == "XZ": return (u, 0.0, v)
                        elif plane == "YZ": return (0.0, u, v)
                        else: return (u, v, 0.0)

                    pu, pv = 0.0, 0.0
                    path_vertices.append(_to_3d(pu, pv))
                    for m, a, n in sub_chain:
                        if m == 'Workplane':
                            if a and isinstance(a[0], ast.Constant):
                                plane = a[0].value
                            continue
                        elif m == 'lineTo':
                            nu = _extract_num(a[0])
                            nv = _extract_num(a[1])
                            if nu is not None and nv is not None:
                                pu, pv = nu, nv
                                path_vertices.append(_to_3d(pu, pv))
                        elif m == 'moveTo':
                            nu = _extract_num(a[0])
                            nv = _extract_num(a[1])
                            if nu is not None and nv is not None:
                                pu, pv = nu, nv
                                if not path_vertices:
                                    path_vertices.append(_to_3d(pu, pv))
                                else:
                                    path_vertices[-1] = _to_3d(pu, pv)
                        elif m in ('wire', 'close', 'Wire'):
                            continue

                if len(path_vertices) < 2:
                    # Fallback: simple Z extrude
                    d = 1.0  # default depth
                    self._insert_param(sketch_start, "extrude.depth", d, node)
                    solid_start = sketch_start
                    solid_desc = {"type": "extrude", "child": sketch_desc}
                else:
                    # Build sweep node with path vertices
                    n_path = len(path_vertices)
                    # INSERT path vertex params BEFORE sketch params (DFS: parent first)
                    for i in range(n_path - 1, -1, -1):
                        vx, vy, vz = path_vertices[i]
                        self._insert_param(sketch_start, f"sweep_path.{i}.z", vz, node)
                        self._insert_param(sketch_start, f"sweep_path.{i}.y", vy, node)
                        self._insert_param(sketch_start, f"sweep_path.{i}.x", vx, node)
                    solid_start = sketch_start
                    solid_desc = {"type": "sweep", "n_path_verts": n_path, "child": sketch_desc}

                sketch_desc = None

            elif method == 'fillet':
                r = _extract_num(args[0])
                if r is None:
                    raise ValueError("fillet() arg must be numeric literal")
                # If a selector preceded this fillet (e.g. PointOnEdgeSelector
                # or edges('>X')), cadquery only fillets that specific edge.
                # Our offset/fillet_* SDF rounds ALL edges, which is wildly
                # wrong for selective fillets — stacking several selector
                # fillets shrinks the whole shape.  Skip the operation when
                # a selector is pending (geometry stays un-filleted, closer
                # to the real shape than a wrongly-shrunk one).
                selective = (pending_selector is not None
                             and pending_selector.get("kind") == "edges")
                pending_selector = None
                if selective:
                    continue
                if solid_desc and solid_desc.get("type") in ("union", "subtract", "intersect"):
                    old_type = solid_desc["type"]
                    # INSERT fillet radius BEFORE subtree (DFS: self-param first)
                    self._insert_param(solid_start, "fillet.r", r, args[0])
                    if old_type == "union":
                        solid_desc["type"] = "fillet_union"
                    elif old_type == "intersect":
                        solid_desc["type"] = "fillet_intersect"
                    elif old_type == "subtract":
                        solid_desc["type"] = "fillet_intersect"
                        solid_desc["right"] = {"type": "inverse", "child": solid_desc["right"]}
                elif solid_desc:
                    # Non-boolean solid (box, cylinder, extrude, etc.)
                    # Use offset node: sdf = child_sdf - r (rounds all edges)
                    self._insert_param(solid_start, "fillet.r", r, args[0])
                    solid_desc = {"type": "offset", "child": solid_desc}

            elif method == 'chamfer':
                s = _extract_num(args[0])
                if s is None:
                    raise ValueError("chamfer() arg must be numeric literal")
                # Same selective-chamfer issue as fillet — skip if selector pending.
                selective = (pending_selector is not None
                             and pending_selector.get("kind") == "edges")
                pending_selector = None
                if selective:
                    # SELECTIVE CHAMFER: instead of dropping the op (leaves the
                    # un-cut corner wedge in the analytic solid, chamfer_20),
                    # subtract the measured triangular bevel prism(s).  All
                    # cutter params frozen (derived geometry, no write-back).
                    for _syn in self._try_selective_bevel(node):
                        try:
                            _syn_ast = ast.parse(_syn, mode='eval').body
                        except SyntaxError:
                            continue
                        _sp = CadQueryParser()
                        _sp._assigns_seq = self._assigns_seq
                        _sp._resolution_index = self._resolution_index
                        _sc = _sp._unroll_chain(_syn_ast)
                        if not _sc:
                            continue
                        try:
                            _rd = _sp._process_chain(_sc)
                        except Exception:
                            continue
                        if _rd is None:
                            continue
                        self.params.extend(_sp.params)
                        for _sl in _sp.slots:
                            _sl.line = 0; _sl.col = 0; _sl.end_col = 0
                        self.slots.extend(_sp.slots)
                        self.names.extend(['frozen.chamfer.' + n for n in _sp.names])
                        solid_desc = {"type": "subtract",
                                      "left": solid_desc, "right": _rd}
                    continue
                if solid_desc and solid_desc.get("type") in ("union", "subtract", "intersect"):
                    old_type = solid_desc["type"]
                    self._insert_param(solid_start, "chamfer.s", s, args[0])
                    if old_type == "intersect":
                        solid_desc["type"] = "chamfer_intersect"
                    elif old_type == "subtract":
                        solid_desc["type"] = "chamfer_intersect"
                        solid_desc["right"] = {"type": "inverse", "child": solid_desc["right"]}
                elif solid_desc:
                    # Non-boolean solid: approximate chamfer as small fillet via offset
                    self._insert_param(solid_start, "chamfer.s", s, args[0])
                    solid_desc = {"type": "offset", "child": solid_desc}

            elif method in ('edges', 'faces', 'wires', 'vertices'):
                # Parse selector and store for next operation
                sel = _parse_selector(args) or {}
                sel["kind"] = method
                pending_selector = sel

            elif method == 'center':
                # .center(dx, dy): shift the workplane-local origin for the NEXT
                # direct profile (loft cross-sections use this to offset circles/
                # rects).  Previously dropped -> every loft profile was centred at
                # the axis, collapsing the loft.
                cdx = _extract_num(args[0]) if len(args) >= 1 else 0.0
                cdy = _extract_num(args[1]) if len(args) >= 2 else 0.0
                # RELATIVE shift (cadquery semantics): accumulate into the persistent
                # in-plane origin so it survives primitive resets and .workplane()
                # inheritance, and seed the next primitive from the accumulated origin.
                loft_base_center = (loft_base_center[0] + (cdx if cdx is not None else 0.0),
                                    loft_base_center[1] + (cdy if cdy is not None else 0.0))
                pending_center = loft_base_center

            elif method in ('pushPoints', 'push'):
                # In-plane offset for the NEXT primitive on a (usually face-relative)
                # workplane: .pushPoints([(x,y)]).hole(...) and .push([(x,y)]).circle()/
                # .rect() in 02_cut etc.  Was dropped -> face cuts sat at the plane
                # origin.  Uses the FIRST point of the list.
                try:
                    lst = args[0]
                    pt = lst.elts[0] if getattr(lst, 'elts', None) else None
                    if pt is not None and getattr(pt, 'elts', None) and len(pt.elts) >= 2:
                        px = _extract_num(pt.elts[0]); py = _extract_num(pt.elts[1])
                        pending_center = (px if px is not None else 0.0,
                                          py if py is not None else 0.0)
                except Exception:
                    pass

            elif method == 'workplane':
                # Track workplane offset for loft
                offset_val = None
                for kw in getattr(node, 'keywords', []):
                    if getattr(kw, 'arg', None) == 'offset':
                        offset_val = _extract_num(kw.value)
                if offset_val is None and args:
                    offset_val = _extract_num(args[0])
                # FACE-RELATIVE WORKPLANE: .faces(">Z").workplane() puts the sketch
                # plane on the selected bbox face of the current solid, so a following
                # .cutBlind / .extrude / sketch lands ON that face instead of the root
                # z=0.  Was unresolved -> face-relative cuts/holes were misplaced (the
                # 02_cut blocker; cutBlind depth couldn't optimise because the cut sat
                # at the wrong Z and the loss landscape had no minimum at the true
                # depth).  Map face direction -> (plane string, face-centre origin);
                # _wrap_with_current_wp already handles those plane strings.  offset=
                # shifts the plane outward along the face normal.
                if (pending_selector and pending_selector.get('kind') == 'faces'
                        and pending_selector.get('op') in ('>', '<')
                        and current_bbox is not None):
                    _op = pending_selector['op']; _ax = pending_selector['axis']
                    _xmn, _xmx, _ymn, _ymx, _zmn, _zmx = current_bbox
                    _cx = (_xmn + _xmx) / 2.0; _cy = (_ymn + _ymx) / 2.0
                    _cz = (_zmn + _zmx) / 2.0
                    _off = offset_val or 0.0
                    # (plane, origin, outward-normal axis index, sign)
                    _FACE = {
                        ('>', 'Z'): ('XY', [_cx, _cy, _zmx + _off]),
                        ('<', 'Z'): ('YX', [_cx, _cy, _zmn - _off]),
                        ('>', 'X'): ('YZ', [_xmx + _off, _cy, _cz]),
                        ('<', 'X'): ('ZY', [_xmn - _off, _cy, _cz]),
                        ('>', 'Y'): ('ZX', [_cx, _ymx + _off, _cz]),
                        ('<', 'Y'): ('XZ', [_cx, _ymn - _off, _cz]),
                    }
                    if (_op, _ax) in _FACE:
                        _pl, _org = _FACE[(_op, _ax)]
                        self._current_wp_plane = _pl
                        self._current_wp_euler = None
                        self._current_wp_origin = tuple(_org)
                        self._current_wp_origin_nodes = None
                        offset_val = None   # consumed as a face placement, not a loft step
                        loft_base_center = (0.0, 0.0)  # new face plane -> fresh origin
                    pending_selector = None
                # PLACEMENT offset: a plain Workplane('P').workplane(offset=o)
                # BEFORE any sketch shifts the plane origin along its normal
                # (cadquery semantics; desugared cadgen chains rely on it).
                # Previously the offset was only tracked as a loft step, so
                # such extrudes landed at offset 0 (faith ~0.5 on i2c parts).
                # Loft flows are untouched: their offsets come AFTER a sketch.
                if (offset_val is not None and sketch_desc is None
                        and not loft_sketches
                        and getattr(self, '_current_wp_euler', None) is None):
                    _NRM = {'XY': (0, 0, 1), 'YX': (0, 0, -1),
                            'YZ': (1, 0, 0), 'ZY': (-1, 0, 0),
                            'ZX': (0, 1, 0), 'XZ': (0, -1, 0)}
                    _pl = getattr(self, '_current_wp_plane', 'XY')
                    if _pl in _NRM:
                        _n = _NRM[_pl]
                        _o = list(getattr(self, '_current_wp_origin', None)
                                  or (0.0, 0.0, 0.0))
                        self._current_wp_origin = (
                            _o[0] + _n[0] * offset_val,
                            _o[1] + _n[1] * offset_val,
                            _o[2] + _n[2] * offset_val)
                        self._current_wp_origin_nodes = None
                        offset_val = None   # consumed as placement, not loft
                # Save the current profile at its z (loft_height so far) BEFORE
                # advancing, so multi-section lofts know each profile's height.
                if sketch_desc is not None:
                    loft_sketches.append((sketch_desc, sketch_start,
                                          len(self.params) - sketch_start, loft_height))
                    sketch_desc = None
                if offset_val is not None:
                    loft_height += abs(offset_val)
                # A parallel .workplane(offset=) INHERITS the accumulated in-plane
                # origin (cadquery), so the next loft profile stays over the carried
                # centre instead of snapping to the axis.  Face-relative planes reset
                # loft_base_center above.
                pending_center = loft_base_center

            elif method in ('finalize', 'assemble', 'val', 'end',
                            'tag', 'first', 'last', 'item', 'clean', 'combine',
                            'solids', 'compounds', 'all', 'toFreecad',
                            'workplaneFromTagged', 'copyWorkplane', 'transformed',
                            'toPending', 'forConstruction', 'add', 'each',
                            'rotate', 'rotateAboutCenter', 'Wire', 'wire'):
                continue

        # Per-extrude wrapping (in `_wrap_with_current_wp`) already handles
        # the workplane orientation/origin for every extrude in the chain.
        # Skip the legacy outer wrap to avoid double-applying the transform.
        return solid_desc or sketch_desc


def parse_cadquery(code: str) -> ParseResult:
    """Parse CadQuery code and return tree description + parameter vector."""
    parser = CadQueryParser()
    return parser.parse_chain(code)


def inject_params_simple(code: str, param_names: List[str],
                         old_params: List[float],
                         new_params: List[float]) -> str:
    """Replace numerical parameter values in CadQuery code.

    Uses simple text substitution: for each named parameter that appears
    as a top-level assignment (e.g. ``width = 30.0``), replace the old
    value with the new one.  Falls back to a global literal replacement
    when the assignment pattern is not found.
    """
    import re
    lines = code.split('\n')

    for name, old_val, new_val in zip(param_names, old_params, new_params):
        if old_val == new_val:
            continue
        new_val_str = f"{new_val:.6f}".rstrip('0').rstrip('.')

        # Try assignment pattern first: ``name = <number>``
        pat = re.compile(
            r'^(\s*' + re.escape(name.split('.')[-1]) + r'\s*=\s*)-?[\d.]+',
        )
        replaced = False
        for i, line in enumerate(lines):
            m = pat.match(line)
            if m:
                lines[i] = m.group(1) + new_val_str
                replaced = True
                break

        if not replaced:
            # Fallback: replace first occurrence of the literal value
            old_lit = f"{old_val:.6f}".rstrip('0').rstrip('.')
            code_joined = '\n'.join(lines)
            code_joined = code_joined.replace(old_lit, new_val_str, 1)
            lines = code_joined.split('\n')

    return '\n'.join(lines)
