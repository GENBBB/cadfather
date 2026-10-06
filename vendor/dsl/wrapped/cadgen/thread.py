import logging
import math

import cadquery as cq
import numpy as np

from .base import BaseFactory, BaseOperation
from .registry import factories
from .utils import float_to_string

logger = logging.getLogger(__name__)

# Workplane axis -> (rotation axis, angle) that turns a +Z-built thread onto it.
_THREAD_AXIS_ROT = {
    "XY": None,
    "YZ": ((0.0, 1.0, 0.0), 90.0),   # Z -> X
    "ZX": ((1.0, 0.0, 0.0), -90.0),  # Z -> Y
}


def _build_threaded_cylinder(R, H, pitch, depth, profile, cone_semi, internal,
                             thickness, thr_z0, thr_len, left_hand, flat_ratio):
    """Threaded cylinder/cone built along +Z (ported from the reference Cylinder).

    External thread = a helical ridge swept and unioned; internal = a helical
    groove cut (guarded). ``cone_semi`` > 0 tapers the thread (NPT-style) via the
    makeHelix cone half-angle. Validity clamps keep the helical sweep stable.
    """
    base = cq.Workplane("XY")
    if thickness > 0.0:
        core = base.circle(R).circle(max(0.1, R - thickness)).extrude(H)
    else:
        core = base.circle(R).extrude(H)

    z0 = float(max(0.0, min(H, thr_z0)))
    # Length clamp: cap turns (pitch*25), tighter for steep cones (400/semi deg).
    max_turns = 25.0 if cone_semi <= 10.0 else max(2.0, 400.0 / cone_semi)
    L = float(max(0.0, min(H - z0, thr_len, pitch * max_turns)))
    if L < pitch * 0.5:
        return core

    min_feat = max(0.1, pitch * 0.05)
    base_min = max(0.3, pitch * 0.15)
    # Depth clamp: <= pitch*0.45 and <= R*0.5 (and <= wall for internal).
    real_th = min(depth, R * 0.5, pitch * 0.45)
    if internal:
        real_th = min(real_th, max(0.1, thickness * 0.6))
    real_th = max(real_th, base_min)
    if real_th < min_feat:
        return core

    # Sweep the thread at the surface radius (cylinder R, or bore R-t internal),
    # NOT offset outward -- the profile base then overlaps INTO the surface so the
    # thread fuses to it with no gap.
    Rpath = R if not internal else max(0.1, R - thickness)
    pitch_eff = -pitch if left_hand else pitch
    helix_kwargs = {} if cone_semi <= 1e-6 else {"angle": cone_semi}
    helix = cq.Wire.makeHelix(
        pitch=pitch_eff, height=L, radius=Rpath,
        center=cq.Vector(0, 0, z0), dir=cq.Vector(0, 0, 1), **helix_kwargs,
    )
    helix_wp = cq.Workplane(obj=helix)

    # For a tapered (conical) thread, the helix radius drifts with z; rebuild the
    # core as a matching frustum so the thread doesn't separate from the surface.
    if cone_semi > 1e-6 and not internal:
        try:
            p0 = helix.positionAt(0.0)
            p1 = helix.positionAt(1.0)
            r0 = (p0.x ** 2 + p0.y ** 2) ** 0.5
            r1 = (p1.x ** 2 + p1.y ** 2) ** 0.5
            slope = (r1 - r0) / L if L > 1e-9 else 0.0
            r_bot = max(0.5, R - slope * z0)
            r_top = max(0.5, R + slope * (H - z0))
            cone = (
                cq.Workplane("XY").circle(r_bot)
                .workplane(offset=H).circle(r_top).loft()
            )
            if thickness > 0.0:
                # Hollow tapered tube = outer cone minus an inner cone (a
                # cutThruAll on a lofted cone removes the whole wall).
                ib = max(0.3, r_bot - thickness)
                it = max(0.3, r_top - thickness)
                inner = (
                    cq.Workplane("XY").circle(ib)
                    .workplane(offset=H).circle(it).loft()
                )
                cone = cone.cut(inner)
            if cone.val().isValid():
                core = cone
        except Exception:
            logger.debug("cone core build failed; straight core", exc_info=True)

    # Profile base reaches OVERLAP inside the surface (toward the body) and the
    # crest reaches ``real_th`` out; both in local +x = radially outward.
    overlap = max(min_feat, 0.25 * real_th)
    d = -real_th if internal else real_th  # internal ridge points into the bore
    base_x = -math.copysign(overlap, d)    # base sits inside the body material
    hb = max(real_th * 0.5, min_feat)
    if profile == "trap":
        hf = max(0.0, min(real_th * 0.5 * max(0.0, min(1.0, flat_ratio)), hb))
        coords = [(base_x, -hb), (d, -hf), (d, hf), (base_x, hb)]
    else:  # "tri" V-profile
        coords = [(base_x, -hb), (d, 0), (base_x, hb)]

    profile_wp = (
        cq.Workplane("XZ", origin=(0, 0, z0)).center(Rpath, 0).polyline(coords).close()
    )
    thread = profile_wp.sweep(helix_wp, isFrenet=True, clean=False)

    if internal:
        # Internal threads are OCC-fragile; fall back to the plain bore if the
        # union of the inward helical ridge does not yield a valid solid.
        try:
            result = core.union(thread)
            if result.val().isValid():
                return result
        except Exception:
            logger.debug("internal thread failed; plain bore", exc_info=True)
        return core
    return core.union(thread)


def thread(
    r,
    point: tuple[float, float, float],
    workplane: str,
    R: float,
    H: float,
    pitch: float,
    depth: float,
    profile: str = "tri",
    cone_semi: float = 0.0,
    internal: bool = False,
    thickness: float = 0.0,
    thr_z0: float = 0.0,
    thr_len: float | None = None,
    left_hand: bool = False,
    flat_ratio: float = 0.25,
):
    """Runtime DSL helper: a threaded cylinder/cone, unioned onto ``r``.

    Self-contained like extrude/revolve. Covers external V (``tri``) and Acme
    (``trap``) threads on straight cylinders and tapered cones (``cone_semi`` deg,
    NPT-style), optionally hollow (``thickness``) with a best-effort internal
    thread. Built along +Z then rotated onto the named ``workplane`` axis.
    """
    tl = H if thr_len is None else thr_len
    body = _build_threaded_cylinder(
        R, H, pitch, depth, profile, cone_semi, internal, thickness,
        thr_z0, tl, left_hand, flat_ratio,
    )
    rot = _THREAD_AXIS_ROT.get(workplane)
    if rot is not None:
        axis, angle = rot
        body = body.rotate((0, 0, 0), axis, angle)
    body = body.translate(tuple(point))

    if r is None:
        return body
    if isinstance(r, cq.Shape):
        return cq.Workplane().add(r).union(body)
    return r.union(body)


class Thread(BaseOperation):
    def __init__(self, R, H, pitch, depth, profile="tri", cone_semi=0.0,
                 internal=False, thickness=0.0, thr_z0=0.0, thr_len=None,
                 left_hand=False, flat_ratio=0.25,
                 center: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0)):
        self.R = R
        self.H = H
        self.pitch = pitch
        self.depth = depth
        self.profile = profile
        self.cone_semi = cone_semi
        self.internal = internal
        self.thickness = thickness
        self.thr_z0 = thr_z0
        self.thr_len = thr_len
        self.left_hand = left_hand
        self.flat_ratio = flat_ratio
        self.center = list(center)

    def to_string(self, *args, **kwargs) -> str:
        raise NotImplementedError("Thread only supports the thread(...) DSL emission")

    def to_call_string(
        self, point: tuple[float, float, float], workplane_axis: str
    ) -> str:
        fts = float_to_string
        tl = "None" if self.thr_len is None else fts(self.thr_len)
        point_str = f"({fts(point[0])}, {fts(point[1])}, {fts(point[2])})"
        return (
            f"thread(r, {point_str}, {workplane_axis!r}, {fts(self.R)}, {fts(self.H)}, "
            f"{fts(self.pitch)}, {fts(self.depth)}, {self.profile!r}, "
            f"{fts(self.cone_semi)}, {bool(self.internal)}, {fts(self.thickness)}, "
            f"{fts(self.thr_z0)}, {tl}, {bool(self.left_hand)}, "
            f"{self.flat_ratio:.4f})\n"
        )

    def transform(
        self, shift: list[float], scale: float, plane_axis: int | None = 2
    ) -> None:
        # Lengths scale; profile/cone_semi/flat_ratio/booleans are scale-free.
        for key in ("R", "H", "pitch", "depth", "thickness", "thr_z0"):
            setattr(self, key, getattr(self, key) * scale)
        if self.thr_len is not None:
            self.thr_len *= scale
        self.center = [self.center[i] * scale for i in range(3)]
        if plane_axis is None:
            for i in range(3):
                self.center[i] += shift[i] * scale
        else:
            for i in range(3):
                if i != plane_axis:
                    self.center[i] += shift[i] * scale

    def round(self) -> None:
        self.R = max(1, round(self.R))
        self.H = max(1, round(self.H))
        self.pitch = max(1, round(self.pitch))
        self.depth = max(1, round(self.depth))
        self.thickness = max(0, round(self.thickness))
        self.thr_z0 = max(0, round(self.thr_z0))
        if self.thr_len is not None:
            self.thr_len = max(1, round(self.thr_len))
        self.center = [round(v) for v in self.center]
        # cone_semi / flat_ratio stay float (dimensionless / small angle).

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "Thread",
            "R": self.R, "H": self.H, "pitch": self.pitch, "depth": self.depth,
            "profile": self.profile, "cone_semi": self.cone_semi,
            "internal": self.internal, "thickness": self.thickness,
            "thr_z0": self.thr_z0, "thr_len": self.thr_len,
            "left_hand": self.left_hand, "flat_ratio": self.flat_ratio,
            "center": self.center,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Thread":
        assert entity["type"] == "Thread", f"bad type {entity['type']}"
        return Thread(
            entity["R"], entity["H"], entity["pitch"], entity["depth"],
            entity.get("profile", "tri"), entity.get("cone_semi", 0.0),
            entity.get("internal", False), entity.get("thickness", 0.0),
            entity.get("thr_z0", 0.0), entity.get("thr_len"),
            entity.get("left_hand", False), entity.get("flat_ratio", 0.25),
            entity.get("center", (0.0, 0.0, 0.0)),
        )


class ThreadFactory(BaseFactory):
    def __init__(
        self,
        *,
        radius_range: tuple[float, float] = (6.0, 55.0),
        height_range: tuple[float, float] = (16.0, 130.0),
        trap_probability: float = 0.3,
        cone_probability: float = 0.25,
        internal_probability: float = 0.18,
        left_hand_probability: float = 0.15,
    ):
        self.radius_range = radius_range
        self.height_range = height_range
        self.trap_probability = trap_probability
        self.cone_probability = cone_probability
        self.internal_probability = internal_probability
        self.left_hand_probability = left_hand_probability

    def generate(self) -> Thread:
        R = float(np.random.uniform(*self.radius_range))
        H = float(np.random.uniform(*self.height_range))
        # Pitch as a fraction of diameter (coarse..fine), bounded to real ranges.
        pitch = float(np.clip(np.random.uniform(0.08, 0.28) * 2.0 * R, 1.5, 14.0))
        depth = float(np.random.uniform(0.5, 0.62)) * pitch  # ~ ISO thread height
        profile = "trap" if np.random.rand() < self.trap_probability else "tri"

        cone_semi = 0.0
        if np.random.rand() < self.cone_probability:
            # NPT-ish gentle taper up to a steeper machined cone.
            cone_semi = float(np.random.uniform(1.5, 9.0))

        internal = np.random.rand() < self.internal_probability
        thickness = 0.0
        # Mostly SOLID cylinders (bolts/screws/studs); occasional hollow tube
        # with a SUBSTANTIAL wall so it never reads as thin floating rings.
        if internal or np.random.rand() < 0.15:
            thickness = float(np.random.uniform(0.35, 0.6)) * R
        if internal:
            thickness = max(thickness, depth + 0.18 * R)

        # Threaded length: a portion of the cylinder, capped at ~25 turns.
        thr_z0 = float(np.random.uniform(0.0, 0.15)) * H
        thr_len = float(np.random.uniform(0.45, 1.0)) * (H - thr_z0)
        thr_len = min(thr_len, pitch * 25.0)

        return Thread(
            R=R, H=H, pitch=pitch, depth=depth, profile=profile,
            cone_semi=cone_semi, internal=internal, thickness=thickness,
            thr_z0=thr_z0, thr_len=thr_len,
            left_hand=np.random.rand() < self.left_hand_probability,
            flat_ratio=float(np.random.uniform(0.2, 0.4)),
        )

    def to_dict(self) -> dict:
        return {
            "type": "ThreadFactory",
            "radius_range": self.radius_range,
            "height_range": self.height_range,
            "trap_probability": self.trap_probability,
            "cone_probability": self.cone_probability,
            "internal_probability": self.internal_probability,
            "left_hand_probability": self.left_hand_probability,
        }

    @staticmethod
    def from_dict(entity: dict) -> "ThreadFactory":
        return ThreadFactory(
            radius_range=entity["radius_range"],
            height_range=entity["height_range"],
            trap_probability=entity.get("trap_probability", 0.3),
            cone_probability=entity.get("cone_probability", 0.25),
            internal_probability=entity.get("internal_probability", 0.18),
            left_hand_probability=entity.get("left_hand_probability", 0.15),
        )


factories.register("thread", ThreadFactory)
