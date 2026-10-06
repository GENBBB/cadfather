import logging
import math

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_UniformAbscissa

# from cadquery_addons import *

from .base import BaseFactory, BaseOperation
from .registry import factories
from .utils import float_to_string

logger = logging.getLogger(__name__)

# Plane axis index for each named workplane (YZ->0, ZX->1, XY->2), used to pick
# the helix axis direction.
_WORKPLANE_OFFSET_INDEX = {"XY": 2, "YZ": 0, "ZX": 1}

# Direction of the helix axis for each workplane axis index (matches the
# legacy emission in SweepInit.to_string).
_HELIX_DIR_BY_AXIS = {
    0: (0.0, 0.0, 1.0),  # YZ plane
    1: (1.0, 0.0, 0.0),  # ZX plane
    2: (0.0, 1.0, 0.0),  # XY plane
}


def _rotate_about_axis(point, center, axis, angle):
    """Rodrigues rotation of ``point`` about the line (``center``, ``axis``)."""
    v = point - center
    k = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return center + v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def _spring_tail_edges(start_pt, start_tan, leg_length, bend_deg, bend_radius,
                       axis_dir, bend_dir=1.0):
    """Build a G1-continuous end-feature ("tail") wire off a helix terminal.

    Starting at ``start_pt`` heading along the unit tangent ``start_tan``, lay a
    straight leg of ``leg_length`` then bend by ``bend_deg`` at ``bend_radius``.
    Continuous coverage: ``bend_deg`` 0 -> straight leg (torsion), ~180-270 ->
    open hook, ~360 -> closed loop/eye (extension). The bend turns in the plane
    spanned by the tangent and a perpendicular derived from the helix axis, so
    loops curl toward the axis. Returns a list of ``cq.Edge``.
    """
    edges = []
    t = np.asarray(start_tan, dtype=float)
    t = t / (np.linalg.norm(t) or 1.0)
    p = np.asarray(start_pt, dtype=float)

    if leg_length and leg_length > 1e-6:
        p1 = p + t * leg_length
        edges.append(cq.Edge.makeLine(cq.Vector(*p), cq.Vector(*p1)))
        p = p1

    if bend_deg and bend_deg > 1e-3 and bend_radius and bend_radius > 1e-6:
        ax = np.asarray(axis_dir, dtype=float)
        bend_axis = np.cross(t, ax)
        if np.linalg.norm(bend_axis) < 1e-6:  # tangent ~parallel to helix axis
            bend_axis = np.cross(t, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(bend_axis) < 1e-6:
                bend_axis = np.cross(t, np.array([0.0, 1.0, 0.0]))
        # bend_dir mirrors the curl: the start terminal must curl AWAY from the
        # coil body (toward -axis), the opposite sense to the end terminal,
        # else the hook wraps back into the coil.
        bend_axis = bend_dir * bend_axis / np.linalg.norm(bend_axis)
        side = np.cross(bend_axis, t)
        side = side / (np.linalg.norm(side) or 1.0)  # points toward arc centre
        center = p + side * bend_radius
        total = math.radians(bend_deg)
        # 3-point arcs degrade past ~180 deg; split into <=120 deg pieces.
        n_seg = max(1, int(math.ceil(bend_deg / 120.0)))
        for i in range(n_seg):
            a0 = total * i / n_seg
            a1 = total * (i + 1) / n_seg
            s0 = _rotate_about_axis(p, center, bend_axis, a0)
            sm = _rotate_about_axis(p, center, bend_axis, 0.5 * (a0 + a1))
            s1 = _rotate_about_axis(p, center, bend_axis, a1)
            edges.append(
                cq.Edge.makeThreePointArc(
                    cq.Vector(*s0), cq.Vector(*sm), cq.Vector(*s1)
                )
            )
    return edges


def _spring_frame(axis_dir):
    """Orthonormal (u, v, axis) with the last vector along ``axis_dir``."""
    a = np.asarray(axis_dir, float)
    a = a / (np.linalg.norm(a) or 1.0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(a, u)
    return u, v, a


def _compression_spine(point, axis_dir, radius, wire_extent, turns, pitch,
                       closed_turns, pts_per_turn=40):
    """Spline spine of a compression coil: constant radius, the pitch collapsing
    toward ~wire diameter over ``closed_turns`` at each end (squared ends)."""
    u, v, a = _spring_frame(axis_dir)
    p0 = np.asarray(point, float)
    closed_pitch = 1.15 * 2.0 * wire_extent
    ct = max(closed_turns, 1e-3)

    def local_pitch(tf):
        if tf < ct:
            return closed_pitch + (pitch - closed_pitch) * (tf / ct)
        if tf > turns - ct:
            return closed_pitch + (pitch - closed_pitch) * ((turns - tf) / ct)
        return pitch

    M = max(8, int(turns * pts_per_turn))
    pts = []
    z = 0.0
    prev = 0.0
    for i in range(M + 1):
        tf = turns * i / M
        if i > 0:
            z += local_pitch(tf) * (tf - prev)
        prev = tf
        th = 2.0 * math.pi * tf
        pos = p0 + radius * (math.cos(th) * u + math.sin(th) * v) + z * a
        pts.append(cq.Vector(*pos))
    return cq.Edge.makeSpline(pts)


def _profiled_spine(point, axis_dir, radius, turns, pitch, bulge, pitch_grade,
                    pts_per_turn=40):
    """Spline spine of a profiled coil: radius follows a half-sine envelope
    ``R(t)=radius*(1+bulge*sin(pi*t/turns))`` -- ``bulge>0`` barrel (bulges in
    the middle), ``bulge<0`` hourglass (pinches). ``pitch_grade`` grades the
    local pitch linearly along the axis for variable-pitch coils (0 = constant).
    """
    u, v, a = _spring_frame(axis_dir)
    p0 = np.asarray(point, float)
    M = max(8, int(turns * pts_per_turn))
    pts = []
    z = 0.0
    prev = 0.0
    for i in range(M + 1):
        tf = turns * i / M
        if i > 0:
            mid = 0.5 * (tf + prev) / turns
            lp = pitch * (1.0 + pitch_grade * (mid - 0.5))
            z += lp * (tf - prev)
        prev = tf
        rr = radius * (1.0 + bulge * math.sin(math.pi * tf / turns))
        th = 2.0 * math.pi * tf
        pos = p0 + rr * (math.cos(th) * u + math.sin(th) * v) + z * a
        pts.append(cq.Vector(*pos))
    return cq.Edge.makeSpline(pts)


def _garter_spine(point, axis_dir, coil_radius, ring_radius, turns,
                  frac=0.985, pts_per_turn=16):
    """Spline spine of a garter (toroidal) coil: a small coil of ``coil_radius``
    wound ``turns`` times around a ring of ``ring_radius`` in the plane
    perpendicular to ``axis_dir``. Left a hair open (``frac``) so the swept tube
    stays a watertight solid instead of failing the closed-loop seam."""
    u, v, a = _spring_frame(axis_dir)
    p0 = np.asarray(point, float)
    M = max(16, int(turns * pts_per_turn))
    last = int(frac * M)
    pts = []
    for i in range(last + 1):
        th = 2.0 * math.pi * i / M
        psi = turns * th
        rr = ring_radius + coil_radius * math.cos(psi)
        pos = (p0 + rr * (math.cos(th) * u + math.sin(th) * v)
               + coil_radius * math.sin(psi) * a)
        pts.append(cq.Vector(*pos))
    return cq.Edge.makeSpline(pts)


def _profile_at_spine_start(spine, profile):
    """Build the wire profile centred on the spine start, perpendicular to it."""
    q = spine.positionAt(0.0)
    tq = spine.tangentAt(0.0)
    wp = cq.Workplane(cq.Plane(origin=(q.x, q.y, q.z), normal=(tq.x, tq.y, tq.z)))
    if profile["type"] == "circle":
        return wp.circle(profile["r"])
    return wp.rect(profile["w"], profile["h"])


def _grind_ends(body, axis_dir, depth):
    """Flatten both ends of ``body`` with planes perpendicular to ``axis_dir``
    (ground compression-spring bearing seats). ``axis_dir`` is axis-aligned."""
    bb = body.val().BoundingBox()
    ai = int(np.argmax(np.abs(np.asarray(axis_dir, float))))
    lo = [bb.xmin, bb.ymin, bb.zmin][ai]
    hi = [bb.xmax, bb.ymax, bb.zmax][ai]
    if hi - lo <= 2.5 * depth:
        return body
    big = 4.0 * max(bb.xlen, bb.ylen, bb.zlen) + 10.0
    dims = [big, big, big]
    dims[ai] = (hi - depth) - (lo + depth)
    cen = [0.5 * (bb.xmin + bb.xmax), 0.5 * (bb.ymin + bb.ymax),
           0.5 * (bb.zmin + bb.zmax)]
    cen[ai] = 0.5 * ((lo + depth) + (hi - depth))
    box = cq.Workplane("XY").box(*dims).translate(tuple(cen))
    return body.intersect(box)


def _centered_hook_edges(P, t, axis_dir, C, hook_radius, gap, span_deg, n=18):
    """Realistic extension-spring HOOK off a coil terminal.

    The wire rises from the coil end ``P`` and curls into an open "C" centred on
    the spring axis, lying in the diametral plane through ``P`` -- so the hook
    points at the coil centre (a machine hook), not off to the side. ``C`` is the
    axis point at the terminal's height; pass ``axis_dir`` flipped at the start
    terminal so the hook lifts away from the body. The arc spans ``span_deg``
    with the opening centred at the bottom. Returns a single smooth spline edge
    (the spline absorbs the tangential->hook quarter-twist, keeping it G1)."""
    P = np.asarray(P, float)
    t = np.asarray(t, float)
    t = t / (np.linalg.norm(t) or 1.0)
    axis = np.asarray(axis_dir, float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    C = np.asarray(C, float)
    radial = P - C
    radial = radial / (np.linalg.norm(radial) or 1.0)
    O = C + axis * (hook_radius + gap)  # hook centre on the axis, clear of coil
    a_lo = math.radians(90.0 - 0.5 * span_deg)  # opening centred at the bottom
    span = math.radians(span_deg)
    pts = [cq.Vector(*P)]
    for k in range(n + 1):
        a = a_lo + span * k / n
        pts.append(
            cq.Vector(*(O + hook_radius * (math.cos(a) * radial + math.sin(a) * axis)))
        )
    last = np.asarray((pts[-1] - pts[-2]).toTuple(), float)
    return cq.Edge.makeSpline(pts, tangents=[cq.Vector(*t), cq.Vector(*last)])


def spring(
    r,
    point: tuple[float, float, float],
    workplane: str,
    profile: dict,
    pitch: float,
    height: float,
    radius: float,
    angle: float | None,
    center: tuple[float, float, float],
    sketch_shift: tuple[float, float],
    start_tail: tuple[float, float, float] | None = None,
    end_tail: tuple[float, float, float] | None = None,
    body_mode: str = "helix",
    ring_radius: float | None = None,
    turns: float | None = None,
    closed_turns: float = 1.0,
    grind: float = 0.0,
    hook_radius: float | None = None,
    hook_span: float = 290.0,
    bulge: float = 0.0,
    pitch_grade: float = 0.0,
):
    """Runtime DSL helper: build a spring body and union it onto ``r``.

    Self-contained like ``extrude``/``revolve``: rebuilds the workplane from the
    named ``workplane`` axis and ``point`` (no external ``w{i}`` variable).
    ``body_mode`` selects the coil family, all one continuous space:
      * ``"helix"`` -- straight-axis coil via ``cq.Wire.makeHelix`` (cylindrical
        or conical), with optional ``start_tail``/``end_tail`` end features each
        ``(leg_length, bend_deg, bend_radius)`` -> torsion legs / extension
        hooks & loops.
      * ``"compression"`` -- spline spine whose pitch collapses over
        ``closed_turns`` at each end (squared ends), optionally ground flat by
        ``grind`` (planar end cut) -> compression springs.
      * ``"garter"`` -- toroidal spline spine: a coil of ``radius`` wound
        ``turns`` times around a ring of ``ring_radius`` -> garter / oil-seal
        rings.
      * ``"extension"`` -- close-wound coil terminated by ``_centered_hook_edges``
        open "C" hooks centred on the axis at both ends -> extension springs.
    """
    axis_index = _WORKPLANE_OFFSET_INDEX[workplane]
    direction = _HELIX_DIR_BY_AXIS.get(axis_index, (0.0, 0.0, 1.0))
    wire_extent = (
        profile["r"]
        if profile["type"] == "circle"
        else 0.5 * max(profile["w"], profile["h"])
    )

    if body_mode == "profiled":
        n_turns = turns if turns is not None else max(4.0, height / max(pitch, 1e-6))
        spine = _profiled_spine(
            point, direction, radius, n_turns, pitch, bulge, pitch_grade
        )
        body = _profile_at_spine_start(spine, profile).sweep(spine, isFrenet=True)

    elif body_mode == "extension":
        n_turns = turns if turns is not None else max(4.0, height / max(pitch, 1e-6))
        helix_wire = cq.Wire.makeHelix(
            pitch=pitch, height=pitch * n_turns, radius=radius,
            center=cq.Vector(*point), dir=cq.Vector(*direction),
        )
        rh = hook_radius if hook_radius is not None else 0.75 * radius
        gap = 2.0 * wire_extent
        axis = np.asarray(direction, float)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        p0 = np.asarray(point, float)

        def _axis_pt(P):
            Pv = np.array([P.x, P.y, P.z])
            return p0 + axis * float(np.dot(Pv - p0, axis))

        s_pt = helix_wire.positionAt(0.0); s_tan = helix_wire.tangentAt(0.0)
        e_pt = helix_wire.positionAt(1.0); e_tan = helix_wire.tangentAt(1.0)
        e_start = _centered_hook_edges(
            (s_pt.x, s_pt.y, s_pt.z), (-s_tan.x, -s_tan.y, -s_tan.z),
            -axis, _axis_pt(s_pt), rh, gap, hook_span,
        )
        e_end = _centered_hook_edges(
            (e_pt.x, e_pt.y, e_pt.z), (e_tan.x, e_tan.y, e_tan.z),
            axis, _axis_pt(e_pt), rh, gap, hook_span,
        )
        spine = cq.Wire.assembleEdges(
            [e_start] + list(helix_wire.Edges()) + [e_end]
        )
        body = _profile_at_spine_start(spine, profile).sweep(spine, isFrenet=True)

    elif body_mode == "garter":
        n_turns = turns if turns is not None else 24.0
        r_ring = ring_radius if ring_radius is not None else 2.2 * radius
        spine = _garter_spine(point, direction, radius, r_ring, n_turns)
        body = _profile_at_spine_start(spine, profile).sweep(spine, isFrenet=True)

    elif body_mode == "compression":
        n_turns = turns if turns is not None else max(3.0, height / max(pitch, 1e-6))
        spine = _compression_spine(
            point, direction, radius, wire_extent, n_turns, pitch, closed_turns
        )
        body = _profile_at_spine_start(spine, profile).sweep(spine, isFrenet=True)
        if grind and grind > 0.0:
            body = _grind_ends(body, direction, grind * wire_extent)

    else:  # "helix"
        def _profile_wp():
            wp = cq.Workplane(workplane, origin=tuple(point))
            wp = wp.center(radius + sketch_shift[0], sketch_shift[1])
            if profile["type"] == "circle":
                return wp.circle(profile["r"])
            return wp.rect(profile["w"], profile["h"])

        helix_kwargs = {} if angle is None else {"angle": angle}
        helix_wire = cq.Wire.makeHelix(
            pitch=pitch,
            height=height,
            radius=radius,
            center=cq.Vector(*center),
            dir=cq.Vector(*direction),
            **helix_kwargs,
        )

        # Assemble one continuous spine: [start tail] + helix + [end tail].
        # Sweeping the whole path in a single pass (rather than unioning separate
        # tail solids) keeps every junction G1 and watertight, bends included.
        tail_edges = []
        if start_tail is not None:
            s_pt = helix_wire.positionAt(0.0)
            s_tan = helix_wire.tangentAt(0.0)
            tail_edges += _spring_tail_edges(
                (s_pt.x, s_pt.y, s_pt.z),
                (-s_tan.x, -s_tan.y, -s_tan.z),  # leave the coil heading outward
                start_tail[0], start_tail[1], start_tail[2], direction,
                bend_dir=-1.0,  # mirror the curl so it opens away from the body
            )
        spine_edges = list(tail_edges) + list(helix_wire.Edges())
        if end_tail is not None:
            end_pt = helix_wire.positionAt(1.0)
            end_tan = helix_wire.tangentAt(1.0)
            spine_edges += _spring_tail_edges(
                (end_pt.x, end_pt.y, end_pt.z),
                (end_tan.x, end_tan.y, end_tan.z),
                end_tail[0], end_tail[1], end_tail[2], direction,
            )

        if start_tail is None and end_tail is None:
            # Plain coil: keep the validated named-workplane profile placement.
            body = _profile_wp().sweep(helix_wire, isFrenet=True)
        else:
            path_wire = cq.Wire.assembleEdges(spine_edges)
            q = path_wire.positionAt(0.0)
            tq = path_wire.tangentAt(0.0)
            prof_plane = cq.Plane(origin=(q.x, q.y, q.z), normal=(tq.x, tq.y, tq.z))
            wp = cq.Workplane(prof_plane)
            if profile["type"] == "circle":
                wp = wp.circle(profile["r"])
            else:
                wp = wp.rect(profile["w"], profile["h"])
            body = wp.sweep(path_wire, isFrenet=True)

    if r is None:
        return body
    if isinstance(r, cq.Shape):
        return cq.Workplane().add(r).union(body)
    return r.union(body)


class SweepInit(BaseOperation):
    def __init__(
        self,
        pitch: float,
        height: float,
        radius: float,
        angle: float | None,
        n_per_turn: int,
        profile: dict,
        center: tuple[float, float, float] = [0.0, 0.0, 0.0],
        sketch_shift: tuple[float, float] = [0.0, 0.0],
        start_tail: tuple[float, float, float] | None = None,
        end_tail: tuple[float, float, float] | None = None,
        body_mode: str = "helix",
        ring_radius: float | None = None,
        turns: float | None = None,
        closed_turns: float = 1.0,
        grind: float = 0.0,
        hook_radius: float | None = None,
        hook_span: float = 290.0,
        bulge: float = 0.0,
        pitch_grade: float = 0.0,
    ):
        self.pitch = pitch
        self.height = height
        self.radius = radius
        self.angle = angle
        self.n_per_turn = n_per_turn
        self.profile = profile
        self.center = center
        self.sketch_shift = sketch_shift
        # End features: each tail is (leg_length, bend_deg, bend_radius) or None.
        # A continuous operator covering torsion legs / extension hooks & loops.
        self.start_tail = list(start_tail) if start_tail is not None else None
        self.end_tail = list(end_tail) if end_tail is not None else None
        # Body family: "helix" (straight axis, +tails), "compression" (squared/
        # ground ends via pitch-collapse + grind), "garter" (toroidal ring).
        self.body_mode = body_mode
        self.ring_radius = ring_radius      # garter: ring (torus) radius
        self.turns = turns                  # garter/compression/extension: # coils
        self.closed_turns = closed_turns    # compression: end coils that close
        self.grind = grind                  # compression: ground end-cut depth
        self.hook_radius = hook_radius      # extension: hook ("C") radius
        self.hook_span = hook_span          # extension: hook arc degrees (open C)
        # profiled body (dimensionless ratios -- never rounded to int):
        self.bulge = bulge                  # +barrel / -hourglass radius envelope
        self.pitch_grade = pitch_grade      # variable-pitch linear grade

    def to_string(
        self,
        plane: str = "cq.Workplane('XY')",
        axis: int | None = None,
        use_literals: bool = True,
    ) -> str:
        x_shift = self.sketch_shift[0]
        y_shift = self.sketch_shift[1]
        if use_literals:
            center_x = float_to_string(self.radius + x_shift)
            center_y = float_to_string(y_shift)
            center_plane = f"{plane}.center({center_x}, {center_y})"
        else:
            radius_str = float_to_string(self.radius)
            base_expr = f"(sweep_radius:={radius_str})"
            eps = 1e-9
            if abs(x_shift) > eps:
                base_expr = f"{base_expr}+{float_to_string(x_shift)}"
            y_expr = "0" if abs(y_shift) <= eps else float_to_string(y_shift)
            center_plane = f"{plane}.center({base_expr}, {y_expr})"
        if self.profile["type"] == "circle":
            profile_expr = f"{center_plane}.circle({self.profile['r']})"
        else:
            profile_expr = (
                f"{center_plane}.rect({self.profile['w']}, {self.profile['h']})"
            )
        angle_param = f", angle={self.angle}" if self.angle is not None else ""
        dir_map = {
            0: "cq.Vector(0, 0, 1)",  # YZ plane
            1: "cq.Vector(1, 0, 0)",  # ZX plane
            2: "cq.Vector(0, 1, 0)",  # XY plane
        }
        dir_expr = (
            dir_map.get(axis, "cq.Vector(0, 0, 1)")
            if axis is not None
            else "cq.Vector(0, 0, 1)"
        )
        if use_literals:
            center_expr = (
                f"cq.Vector({float_to_string(self.center[0])}, "
                f"{float_to_string(self.center[1])}, "
                f"{float_to_string(self.center[2])})"
            )
            radius_expr = float_to_string(self.radius)
        else:
            center_expr = (
                f"cq.Vector({self.center[0]}, {self.center[1]}, {self.center[2]})"
            )
            radius_expr = "sweep_radius"
        helix_expr = (
            f"cq.Wire.makeHelix(pitch={self.pitch}, height={self.height}, "
            f"radius={radius_expr}{angle_param}, center={center_expr}, dir={dir_expr})"
        )
        expr = f"{profile_expr}.sweep({helix_expr}, isFrenet=True)\n"
        return expr

    def to_call_string(
        self, point: tuple[float, float, float], workplane_axis: str
    ) -> str:
        """Emit a single-line ``helix(r, <point>, '<axis>', ...)`` DSL call.

        Self-contained like extrude/revolve: only the ``r`` variable is used and
        the workplane is rebuilt inside ``helix`` from ``point`` + the named axis
        (no external ``w{i}`` variable).
        """
        fts = float_to_string
        if self.profile["type"] == "circle":
            profile_str = f"{{'type': 'circle', 'r': {fts(self.profile['r'])}}}"
        else:
            profile_str = (
                f"{{'type': 'rect', 'w': {fts(self.profile['w'])}, "
                f"'h': {fts(self.profile['h'])}}}"
            )
        point_str = f"({fts(point[0])}, {fts(point[1])}, {fts(point[2])})"
        center_str = (
            f"({fts(self.center[0])}, {fts(self.center[1])}, {fts(self.center[2])})"
        )
        sketch_shift_str = (
            f"({fts(self.sketch_shift[0])}, {fts(self.sketch_shift[1])})"
        )
        angle_str = "None" if self.angle is None else fts(self.angle)

        def _tail_str(tail):
            if tail is None:
                return "None"
            return f"({fts(tail[0])}, {fts(tail[1])}, {fts(tail[2])})"

        extra = ""
        if self.start_tail is not None or self.end_tail is not None:
            extra += (
                f", start_tail={_tail_str(self.start_tail)}, "
                f"end_tail={_tail_str(self.end_tail)}"
            )
        if self.body_mode != "helix":
            ring_str = "None" if self.ring_radius is None else fts(self.ring_radius)
            turns_str = "None" if self.turns is None else fts(self.turns)
            extra += f", body_mode={self.body_mode!r}"
            if self.body_mode == "garter":
                extra += f", ring_radius={ring_str}, turns={turns_str}"
            elif self.body_mode == "compression":
                extra += (
                    f", turns={turns_str}, closed_turns={fts(self.closed_turns)}, "
                    f"grind={fts(self.grind)}"
                )
            elif self.body_mode == "extension":
                hr_str = "None" if self.hook_radius is None else fts(self.hook_radius)
                extra += (
                    f", turns={turns_str}, hook_radius={hr_str}, "
                    f"hook_span={fts(self.hook_span)}"
                )
            elif self.body_mode == "profiled":
                extra += (
                    f", turns={turns_str}, bulge={self.bulge:.4f}, "
                    f"pitch_grade={self.pitch_grade:.4f}"
                )
        return (
            f"spring(r, {point_str}, {workplane_axis!r}, {profile_str}, "
            f"{fts(self.pitch)}, {fts(self.height)}, {fts(self.radius)}, "
            f"{angle_str}, {center_str}, {sketch_shift_str}{extra})\n"
        )

    def transform(
        self, shift: list[float], scale: float, plane_axis: int | None = 2
    ) -> None:
        self.pitch *= scale
        self.height *= scale
        self.radius *= scale
        self.center = [
            (self.center[0] + shift[0]) * scale,
            (self.center[1] + shift[1]) * scale,
            (self.center[2] + shift[2]) * scale,
        ]
        if plane_axis == 0:
            plane_shift = [shift[1], shift[2]]
        elif plane_axis == 1:
            plane_shift = [shift[2], shift[0]]
        else:
            plane_shift = [shift[0], shift[1]]
        self.sketch_shift = [
            (self.sketch_shift[0] + plane_shift[0]) * scale,
            (self.sketch_shift[1] + plane_shift[1]) * scale,
        ]
        if self.profile["type"] == "circle":
            self.profile["r"] *= scale
        else:
            self.profile["w"] *= scale
            self.profile["h"] *= scale
        # Tail lengths scale with the part; the bend angle (deg) is invariant.
        if self.start_tail is not None:
            self.start_tail = [
                self.start_tail[0] * scale, self.start_tail[1], self.start_tail[2] * scale
            ]
        if self.end_tail is not None:
            self.end_tail = [
                self.end_tail[0] * scale, self.end_tail[1], self.end_tail[2] * scale
            ]
        # Ring/hook radius are lengths; turns/closed_turns/grind/span are scale-free.
        if self.ring_radius is not None:
            self.ring_radius *= scale
        if self.hook_radius is not None:
            self.hook_radius *= scale

    def round(self) -> None:
        self.pitch = round(self.pitch)
        self.height = round(self.height)
        self.radius = round(self.radius)
        self.center = [round(v) for v in self.center]
        self.sketch_shift = [round(v) for v in self.sketch_shift]
        if self.profile["type"] == "circle":
            self.profile["r"] = round(self.profile["r"])
        else:
            self.profile["w"] = round(self.profile["w"])
            self.profile["h"] = round(self.profile["h"])
        if self.angle is not None:
            self.angle = round(self.angle)
        # Round tail leg length & bend radius; keep a 1-deg resolution on bend.
        if self.start_tail is not None:
            self.start_tail = [
                round(self.start_tail[0]), round(self.start_tail[1]), round(self.start_tail[2])
            ]
        if self.end_tail is not None:
            self.end_tail = [
                round(self.end_tail[0]), round(self.end_tail[1]), round(self.end_tail[2])
            ]
        if self.ring_radius is not None:
            self.ring_radius = round(self.ring_radius)
        if self.hook_radius is not None:
            self.hook_radius = round(self.hook_radius)

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "SweepInit",
            "pitch": self.pitch,
            "height": self.height,
            "radius": self.radius,
            "angle": self.angle,
            "n_per_turn": self.n_per_turn,
            "profile": self.profile,
            "center": self.center,
            "sketch_shift": self.sketch_shift,
            "start_tail": self.start_tail,
            "end_tail": self.end_tail,
            "body_mode": self.body_mode,
            "ring_radius": self.ring_radius,
            "turns": self.turns,
            "closed_turns": self.closed_turns,
            "grind": self.grind,
            "hook_radius": self.hook_radius,
            "hook_span": self.hook_span,
            "bulge": self.bulge,
            "pitch_grade": self.pitch_grade,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SweepInit":
        assert entity["type"] in (
            "SweepInit",
            "Spring",
        ), f"Trying to build SweepInit from type {entity['type']}"
        return SweepInit(
            entity["pitch"],
            entity["height"],
            entity["radius"],
            entity["angle"],
            entity["n_per_turn"],
            entity["profile"],
            entity["center"],
            entity["sketch_shift"],
            entity.get("start_tail"),
            entity.get("end_tail"),
            entity.get("body_mode", "helix"),
            entity.get("ring_radius"),
            entity.get("turns"),
            entity.get("closed_turns", 1.0),
            entity.get("grind", 0.0),
            entity.get("hook_radius"),
            entity.get("hook_span", 290.0),
            entity.get("bulge", 0.0),
            entity.get("pitch_grade", 0.0),
        )


class SweepInitFactory(BaseFactory):

    def __init__(
        self,
        *,
        pitch_range: tuple[float, float] = (5.0, 20.0),
        height_range: tuple[float, float] = (30.0, 120.0),
        base_radius_range: tuple[float, float] = (10.0, 40.0),
        cone_angle_range_deg: tuple[float, float] = (0.0, 22.0),
        profile_circle_radius_range: tuple[float, float] = (1.0, 5.0),
        profile_rect_size_range: tuple[float, float] = (2.0, 8.0),
        helix_angle_probability: float = 0.5,
        end_feature_probability: float = 0.5,
        compression_probability: float = 0.25,
        garter_probability: float = 0.1,
        extension_probability: float = 0.2,
        profiled_probability: float = 0.12,
    ):
        self.pitch_range = pitch_range
        self.height_range = height_range
        self.base_radius_range = base_radius_range
        self.cone_angle_range_deg = cone_angle_range_deg
        self.profile_circle_radius_range = profile_circle_radius_range
        self.profile_rect_size_range = profile_rect_size_range
        self.helix_angle_probability = helix_angle_probability
        self.end_feature_probability = end_feature_probability
        # Body-mode mix: helix (plain/conical + torsion legs) is the remainder.
        self.compression_probability = compression_probability
        self.garter_probability = garter_probability
        self.extension_probability = extension_probability
        self.profiled_probability = profiled_probability

    def generate(self) -> SweepInit:
        roll = np.random.rand()
        c1 = self.garter_probability
        c2 = c1 + self.compression_probability
        c3 = c2 + self.extension_probability
        c4 = c3 + self.profiled_probability
        if roll < c1:
            return self._generate_garter()
        if roll < c2:
            return self._generate_compression()
        if roll < c3:
            return self._generate_extension()
        if roll < c4:
            return self._generate_profiled()
        return self._generate_helix()

    def _generate_profiled(self) -> SweepInit:
        """Barrel / hourglass / variable-pitch coil via a profiled spline-spine."""
        radius = float(np.random.uniform(*self.base_radius_range))
        index = float(np.random.uniform(5.0, 10.0))
        wire_r = radius / index
        turns = float(np.random.uniform(5.0, 9.0))
        pitch = float(np.random.uniform(2.6, 4.5)) * 2.0 * wire_r
        kind = np.random.choice(["barrel", "hourglass", "varpitch"])
        bulge = 0.0
        pitch_grade = 0.0
        if kind == "barrel":
            bulge = float(np.random.uniform(0.12, 0.30))
        elif kind == "hourglass":
            bulge = -float(np.random.uniform(0.12, 0.26))
        else:  # variable pitch: keep min local pitch above the wire diameter
            grade = float(np.random.uniform(0.5, 1.1))
            grade = min(grade, 2.0 * (1.0 - 1.1 * 2.0 * wire_r / pitch))
            pitch_grade = grade if np.random.rand() < 0.5 else -grade
        profile = {"type": "circle", "r": wire_r}
        return SweepInit(
            pitch=pitch, height=turns * pitch, radius=radius, angle=None,
            n_per_turn=120, profile=profile,
            body_mode="profiled", turns=turns, bulge=bulge, pitch_grade=pitch_grade,
        )

    def _generate_extension(self) -> SweepInit:
        """Close-wound extension coil terminated by centred open "C" hooks."""
        radius = float(np.random.uniform(*self.base_radius_range))
        index = float(np.random.uniform(5.0, 10.0))  # spring index C = D/d
        wire_r = radius / index
        # Close-wound body: coils nearly touching (pitch just above wire dia).
        pitch = float(np.random.uniform(2.05, 2.35)) * wire_r
        turns = float(np.random.uniform(6.0, 14.0))
        profile = {"type": "circle", "r": wire_r}
        hook_radius = float(np.random.uniform(0.65, 0.9)) * radius
        hook_span = float(np.random.uniform(255.0, 310.0))
        return SweepInit(
            pitch=pitch, height=turns * pitch, radius=radius, angle=None,
            n_per_turn=120, profile=profile,
            body_mode="extension", turns=turns,
            hook_radius=hook_radius, hook_span=hook_span,
        )

    def _generate_garter(self) -> SweepInit:
        """Toroidal coil ring (garter / oil-seal spring)."""
        coil_radius = float(np.random.uniform(5.0, 13.0))
        index = float(np.random.uniform(4.0, 8.0))
        wire_r = coil_radius / index
        ring_radius = float(np.random.uniform(2.5, 4.5)) * coil_radius
        # Cap turns so adjacent coils around the ring don't interpenetrate
        # (ring pitch = 2*pi*ring_radius/turns must exceed the wire diameter).
        max_turns = 0.6 * math.pi * ring_radius / wire_r
        turns = float(np.random.uniform(16.0, min(40.0, max(18.0, max_turns))))
        profile = {"type": "circle", "r": wire_r}
        return SweepInit(
            pitch=2.5 * wire_r, height=0.0, radius=coil_radius, angle=None,
            n_per_turn=120, profile=profile,
            body_mode="garter", ring_radius=ring_radius, turns=turns,
        )

    def _generate_compression(self) -> SweepInit:
        """Cylindrical compression coil with squared (and usually ground) ends."""
        radius = float(np.random.uniform(*self.base_radius_range))
        index = float(np.random.uniform(5.0, 10.0))  # spring index C = D/d
        wire_r = radius / index
        d = 2.0 * wire_r
        turns = float(np.random.uniform(4.0, 9.0))
        # Open pitch with a real coil gap; keep slenderness L0/D <~4 and pitch>d.
        pitch = float(np.random.uniform(2.2, 4.0)) * d
        pitch = float(np.clip(pitch, 1.4 * d, 8.0 * radius / turns))
        if np.random.rand() < 0.8:
            profile = {"type": "circle", "r": wire_r}
        else:  # rectangular/square die-spring wire
            profile = {"type": "rect", "w": float(np.random.uniform(1.4, 2.6)) * wire_r,
                       "h": float(np.random.uniform(1.4, 2.6)) * wire_r}
        closed_turns = float(np.random.uniform(0.75, 1.25))
        grind = 0.55 if np.random.rand() < 0.7 else 0.0  # ground vs squared-only
        return SweepInit(
            pitch=pitch, height=turns * pitch, radius=radius, angle=None,
            n_per_turn=120, profile=profile,
            body_mode="compression", turns=turns,
            closed_turns=closed_turns, grind=grind,
        )

    def _generate_helix(self) -> SweepInit:
        pitch = float(np.random.uniform(*self.pitch_range))
        height = float(np.random.uniform(*self.height_range))
        # Ensure at least ~3 turns so it reads as a coil, not a part-loop.
        height = max(height, 3.0 * pitch)
        base_radius = float(np.random.uniform(*self.base_radius_range))

        # Real taper: most coils cylindrical, some CONICAL/tapered. Clamp the
        # cone angle so the radius stays positive over the height
        # (height*tan(angle) < ~0.8*radius), else makeHelix inverts/fails.
        if np.random.random() < self.helix_angle_probability:
            max_angle = math.degrees(math.atan(0.8 * base_radius / height))
            hi = min(self.cone_angle_range_deg[1], max_angle)
            lo = min(self.cone_angle_range_deg[0], hi)
            angle_deg = float(np.random.uniform(lo, hi))
            if angle_deg < 0.5:
                angle_deg = None
        else:
            angle_deg = None

        # Wire: round wire is by far the most common; rectangular/square is the
        # heavy-duty case.
        if np.random.rand() < 0.7:
            profile = {
                "type": "circle",
                "r": float(np.random.uniform(*self.profile_circle_radius_range)),
            }
        else:
            w = float(np.random.uniform(*self.profile_rect_size_range))
            h = float(np.random.uniform(*self.profile_rect_size_range))
            profile = {"type": "rect", "w": w, "h": h}

        # END FEATURES on the helix body = TORSION LEGS (straight or slightly
        # bent tangent legs). Extension hooks are a separate realistic body mode
        # (centred machine hooks), so the helix tail family is legs only here.
        wire_extent = (
            profile["r"]
            if profile["type"] == "circle"
            else 0.5 * max(profile["w"], profile["h"])
        )
        start_tail = end_tail = None
        if np.random.rand() < self.end_feature_probability:

            def _mk_leg():
                leg = float(np.random.uniform(0.8, 2.0)) * base_radius
                bend = (
                    float(np.random.uniform(15.0, 70.0))
                    if np.random.rand() < 0.4
                    else 0.0
                )
                rad = max(
                    float(np.random.uniform(0.5, 1.0)) * base_radius,
                    1.6 * wire_extent,
                )
                return [leg, bend, rad]

            both = np.random.rand() < 0.7  # torsion springs have legs at both ends
            end_tail = _mk_leg()
            if both:
                start_tail = _mk_leg()
            elif np.random.rand() < 0.3:
                start_tail, end_tail = end_tail, None

        return SweepInit(
            pitch=pitch,
            height=height,
            radius=base_radius,
            angle=angle_deg,
            n_per_turn=120,
            profile=profile,
            start_tail=start_tail,
            end_tail=end_tail,
        )

    @staticmethod
    def discretize_wire_continuous(
        wire: cq.Wire, n_per_loop: int = 100
    ) -> list[tuple[float, float, float]]:
        """
        Discretize a closed wire into exactly n_per_loop points WITHOUT duplicating the last one.
        Suitable for smooth interpolation.
        """
        # Collect all edges into one list
        edges = wire.Edges()
        total_len = sum(e.Length() for e in edges)  # type: ignore

        # Distribute points in proportion to each edge's length
        pts = []
        for e in edges:
            frac = e.Length() / total_len
            n_local = max(2, int(round(frac * n_per_loop)))
            d = SweepInitFactory.discretize_edge(e, n=n_local)
            # Add ALL points, including the first and last; duplicates are removed afterwards
            pts.extend(d)

        # Remove duplicates at edge joints (keep only the start of the loop)
        cleaned = [pts[0]]
        for p in pts[1:]:
            if not (
                abs(p[0] - cleaned[-1][0]) < 1e-6
                and abs(p[1] - cleaned[-1][1]) < 1e-6
                and abs(p[2] - cleaned[-1][2]) < 1e-6
            ):
                cleaned.append(p)

        # Make sure it is closed: if the first is about the last, drop the last
        if len(cleaned) > 1 and all(
            abs(cleaned[0][i] - cleaned[-1][i]) < 1e-6 for i in range(3)
        ):
            cleaned = cleaned[:-1]

        return cleaned

    @staticmethod
    def discretize_edge(edge: cq.Edge, n: int = 50) -> list[tuple[float, float, float]]:
        c = BRepAdaptor_Curve(edge.wrapped)
        algo = GCPnts_UniformAbscissa(c, n)
        pts = []
        for i in range(1, algo.NbPoints() + 1):
            p = c.Value(algo.Parameter(i))
            pts.append((p.X(), p.Y(), p.Z()))
        return pts

    @staticmethod
    def make_smooth_spiral_path(
        sketch_wp: cq.Workplane,
        pitch: float = 1.5,
        height: float = 10.0,
        radius: float = 5.0,
        angle: float = 10.0,
        n_per_turn: int = 120,
        workplane: cq.Workplane = cq.Workplane("XY"),
    ) -> cq.Workplane:
        """
        Build a smooth 3D helix path from an arbitrary closed sketch.
        If the sketch is a circle, uses makeHelix directly.
        Otherwise builds a helix following the sketch shape.

        Args:
            sketch_wp: Workplane with a closed wire/sketch
            pitch: Distance between helix turns
            height: Total helix height
            radius: Helix radius (used for a circle or as a base point for other shapes)
            angle: Cone angle in degrees (for a conical helix)
            n_per_turn: Number of points per turn (for non-circle cases)

        Returns:
            cq.Workplane with the path for sweep
        """
        val = sketch_wp.val()
        if isinstance(val, cq.Edge):
            wire = cq.Wire.assembleEdges([val])
        elif isinstance(val, cq.Wire):
            wire = val
        else:
            raise ValueError("Input must be a closed Wire or Edge")

        if not wire.IsClosed():  # type: ignore
            raise ValueError("Wire must be closed for a spiral!")

        # Check whether the wire is a circle
        edges = wire.Edges()
        if len(edges) == 1:
            edge = edges[0]
            # Check that it is a circle
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GeomAbs import GeomAbs_Circle

            adaptor = BRepAdaptor_Curve(edge.wrapped)
            if adaptor.GetType() == GeomAbs_Circle:
                # A circle: use makeHelix directly
                helix_wire = cq.Wire.makeHelix(
                    pitch=pitch, height=height, radius=radius, angle=angle
                )
                return workplane.add(helix_wire)

        # Non-circle case: build a helix following the sketch shape
        # Compute the number of turns from height and pitch
        turns = int(height / pitch) if pitch > 0 else 1

        # Discretize ONE turn
        base_pts = SweepInitFactory.discretize_wire_continuous(
            wire, n_per_loop=n_per_turn
        )  # [(x,y,0), ...]
        if not base_pts:
            raise ValueError("No points sampled — check sketch")

        # Collect 3D points for all turns: z grows smoothly
        pts_3d = []
        n = len(base_pts)

        for k in range(turns * n):
            turn_frac = k / (turns * n)  # from 0 to 1
            z = turn_frac * height
            i = k % n
            pts_3d.append((*base_pts[i][:2], z))

        # Build a spline through all points; CadQuery makes a cubic spline
        return workplane.spline(pts_3d)

    def to_dict(self) -> dict:
        return {
            "type": "SweepInitFactory",
            "pitch_range": self.pitch_range,
            "height_range": self.height_range,
            "base_radius_range": self.base_radius_range,
            "cone_angle_range_deg": self.cone_angle_range_deg,
            "profile_circle_radius_range": self.profile_circle_radius_range,
            "profile_rect_size_range": self.profile_rect_size_range,
            "helix_angle_probability": self.helix_angle_probability,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SweepInitFactory":
        return SweepInitFactory(
            pitch_range=entity["pitch_range"],
            height_range=entity["height_range"],
            base_radius_range=entity["base_radius_range"],
            cone_angle_range_deg=entity["cone_angle_range_deg"],
            profile_circle_radius_range=entity["profile_circle_radius_range"],
            profile_rect_size_range=entity["profile_rect_size_range"],
            helix_angle_probability=entity["helix_angle_probability"],
        )


# The "helix" element renamed to "spring" for clarity. Preferred names + the
# new registry key; old names kept as aliases for back-compat (configs, data).
Spring = SweepInit
SpringFactory = SweepInitFactory
helix = spring  # back-compat alias for the runtime DSL function

factories.register("sweep_init", SweepInitFactory)
factories.register("spring", SweepInitFactory)
