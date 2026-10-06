import logging

import cadquery as cq
import numpy as np

from .base import BaseFactory, BaseOperation
from .registry import factories
from .utils import float_to_string

logger = logging.getLogger(__name__)


def _rib_wall(workplane, point, cx, cy, length, thickness, height, top_ratio,
              angle, z0):
    """A straight reinforcement rib: a drafted wall (loft from a wide base
    rectangle to a narrower top rectangle = side draft) standing on the base."""
    top_t = max(0.15 * thickness, thickness * top_ratio)
    return (
        cq.Workplane(workplane, origin=tuple(point))
        .workplane(offset=z0)
        .center(cx, cy)
        .transformed(rotate=(0.0, 0.0, angle))
        .rect(length, thickness)
        .workplane(offset=height)
        .rect(length, top_t)
        .loft()
    )


def _gusset(workplane, point, cx, cy, length, height, thickness, angle, z0):
    """A triangular gusset brace: right-triangle side profile (tall edge against
    a wall, sloping hypotenuse) extruded by ``thickness``."""
    pts = [(-0.5 * length, 0.0), (0.5 * length, 0.0), (-0.5 * length, height)]
    return (
        cq.Workplane(workplane, origin=tuple(point))
        .workplane(offset=z0)
        .center(cx, cy)
        .transformed(rotate=(0.0, 0.0, angle))
        .transformed(rotate=(90.0, 0.0, 0.0))
        .polyline(pts)
        .close()
        .extrude(thickness)
    )


def rib(r, point: tuple[float, float, float], workplane: str, base: dict,
        ribs: list[dict]):
    """Runtime DSL helper: build a ribbed structure (base plate + ribs) and union
    onto ``r``.

    Self-contained like extrude/revolve. ``base`` is a box plate or disk; each
    entry in ``ribs`` is a drafted straight rib (``kind="wall"``) or a triangular
    ``kind="gusset"`` brace, positioned at ``(cx, cy)`` and rotated by ``angle``.
    Covers reinforcement-rib arrays, radial spokes, and gusset braces. Each rib
    union is guarded so one bad rib can't sink the whole part.
    """
    wp = cq.Workplane(workplane, origin=tuple(point))
    if base["type"] == "box":
        body = wp.box(base["w"], base["d"], base["h"], centered=(True, True, False))
    else:  # disk
        body = wp.circle(base["r"]).extrude(base["h"])
    z0 = base["h"]

    for rb in ribs:
        try:
            if rb.get("kind") == "gusset":
                solid = _gusset(
                    workplane, point, rb["cx"], rb["cy"], rb["length"],
                    rb["height"], rb["thickness"], rb.get("angle", 0.0), z0,
                )
            else:
                solid = _rib_wall(
                    workplane, point, rb["cx"], rb["cy"], rb["length"],
                    rb["thickness"], rb["height"], rb.get("top_ratio", 0.8),
                    rb.get("angle", 0.0), z0,
                )
            body = body.union(solid)
        except Exception:
            logger.debug("rib union failed; skipping one rib", exc_info=True)

    if r is None:
        return body
    if isinstance(r, cq.Shape):
        return cq.Workplane().add(r).union(body)
    return r.union(body)


class Rib(BaseOperation):
    def __init__(self, base: dict, ribs: list[dict]):
        self.base = base
        self.ribs = ribs

    def to_string(self, *args, **kwargs) -> str:
        # Rib is emitted only through the self-contained ``rib(...)`` DSL call
        # (use_literals path -> to_call_string), like spring/loft/gear.
        raise NotImplementedError("Rib only supports the rib(...) DSL emission")

    def to_call_string(
        self, point: tuple[float, float, float], workplane_axis: str
    ) -> str:
        fts = float_to_string
        if self.base["type"] == "box":
            base_str = (
                f"{{'type': 'box', 'w': {fts(self.base['w'])}, "
                f"'d': {fts(self.base['d'])}, 'h': {fts(self.base['h'])}}}"
            )
        else:
            base_str = (
                f"{{'type': 'disk', 'r': {fts(self.base['r'])}, "
                f"'h': {fts(self.base['h'])}}}"
            )
        rib_strs = []
        for rb in self.ribs:
            parts = [f"'kind': {rb.get('kind', 'wall')!r}"]
            for key in ("cx", "cy", "length", "thickness", "height", "angle"):
                parts.append(f"'{key}': {fts(rb[key])}")
            if rb.get("kind") != "gusset":
                parts.append(f"'top_ratio': {fts(rb.get('top_ratio', 0.8))}")
            rib_strs.append("{" + ", ".join(parts) + "}")
        ribs_str = "[" + ", ".join(rib_strs) + "]"
        point_str = f"({fts(point[0])}, {fts(point[1])}, {fts(point[2])})"
        return f"rib(r, {point_str}, {workplane_axis!r}, {base_str}, {ribs_str})\n"

    def transform(
        self, shift: list[float], scale: float, plane_axis: int | None = 2
    ) -> None:
        for key in ("w", "d", "h", "r"):
            if key in self.base:
                self.base[key] *= scale
        for rb in self.ribs:
            for key in ("cx", "cy", "length", "thickness", "height"):
                rb[key] *= scale
            # top_ratio (draft) and angle are scale-free.

    def round(self) -> None:
        for key in ("w", "d", "h", "r"):
            if key in self.base:
                self.base[key] = max(1, round(self.base[key]))
        for rb in self.ribs:
            for key in ("cx", "cy", "length", "thickness", "height"):
                rb[key] = round(rb[key])
            if "angle" in rb:
                rb["angle"] = round(rb["angle"])
            # top_ratio is a dimensionless draft ratio -> keep as float.

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {"type": "Rib", "base": self.base, "ribs": self.ribs}

    @staticmethod
    def from_dict(entity: dict) -> "Rib":
        assert entity["type"] == "Rib", f"Trying to build Rib from type {entity['type']}"
        return Rib(base=entity["base"], ribs=entity["ribs"])


class RibFactory(BaseFactory):
    def __init__(
        self,
        *,
        plate_size_range: tuple[float, float] = (55.0, 120.0),
        plate_thickness_range: tuple[float, float] = (4.0, 12.0),
        disk_probability: float = 0.35,
        gusset_probability: float = 0.2,
    ):
        self.plate_size_range = plate_size_range
        self.plate_thickness_range = plate_thickness_range
        self.disk_probability = disk_probability
        self.gusset_probability = gusset_probability

    def generate(self) -> Rib:
        wall = float(np.random.uniform(*self.plate_thickness_range))  # plate thickness
        # DFM-ish rib proportions relative to the base wall thickness.
        rib_t = float(np.random.uniform(0.5, 0.8)) * wall
        rib_h = float(np.random.uniform(1.5, 3.0)) * wall
        top_ratio = float(np.random.uniform(0.6, 0.9))  # side draft

        use_disk = np.random.rand() < self.disk_probability
        ribs: list[dict] = []

        if not use_disk and np.random.rand() < self.gusset_probability:
            # Base plate + vertical wall + triangular gusset braces along it.
            w = float(np.random.uniform(*self.plate_size_range))
            d = float(np.random.uniform(*self.plate_size_range))
            base = {"type": "box", "w": w, "d": d, "h": wall}
            wall_x = -0.5 * w + max(2.0 * wall, 0.12 * w)
            ribs.append({
                "kind": "wall", "cx": wall_x, "cy": 0.0, "length": d * 0.9,
                "thickness": wall, "height": float(np.random.uniform(3.0, 6.0)) * wall,
                "top_ratio": 1.0, "angle": 90.0,
            })
            g_len = float(np.random.uniform(2.0, 3.5)) * wall
            g_h = float(np.random.uniform(2.5, 5.0)) * wall
            n = int(np.random.randint(2, 6))
            for i in range(n):
                cy = -0.35 * d + i * (0.7 * d) / max(1, n - 1)
                ribs.append({
                    "kind": "gusset", "cx": wall_x + 0.5 * g_len + 0.5 * wall,
                    "cy": cy, "length": g_len, "thickness": rib_t, "height": g_h,
                    "angle": 0.0, "top_ratio": top_ratio,
                })
            return Rib(base, ribs)

        if use_disk:
            R = 0.5 * float(np.random.uniform(*self.plate_size_range))
            base = {"type": "disk", "r": R, "h": wall}
            n = int(np.random.randint(3, 9))  # radial spokes
            span = 1.85 * R
            for i in range(n):
                ribs.append({
                    "kind": "wall", "cx": 0.0, "cy": 0.0, "length": span,
                    "thickness": rib_t, "height": rib_h, "top_ratio": top_ratio,
                    "angle": i * 180.0 / n,
                })
            return Rib(base, ribs)

        # Box plate + linear array of straight reinforcement ribs.
        w = float(np.random.uniform(*self.plate_size_range))
        d = float(np.random.uniform(*self.plate_size_range))
        base = {"type": "box", "w": w, "d": d, "h": wall}
        spacing_min = 2.5 * rib_t
        n_max = max(2, int((d * 0.85) / spacing_min))
        n = int(np.random.randint(2, min(9, n_max) + 1))
        along_x = np.random.rand() < 0.5
        rib_len = (w if along_x else d) * 0.9
        usable = (d if along_x else w) * 0.8
        for i in range(n):
            pos = -0.5 * usable + i * usable / max(1, n - 1)
            ribs.append({
                "kind": "wall",
                "cx": 0.0 if along_x else pos,
                "cy": pos if along_x else 0.0,
                "length": rib_len, "thickness": rib_t, "height": rib_h,
                "top_ratio": top_ratio, "angle": 0.0 if along_x else 90.0,
            })
        return Rib(base, ribs)

    def to_dict(self) -> dict:
        return {
            "type": "RibFactory",
            "plate_size_range": self.plate_size_range,
            "plate_thickness_range": self.plate_thickness_range,
            "disk_probability": self.disk_probability,
            "gusset_probability": self.gusset_probability,
        }

    @staticmethod
    def from_dict(entity: dict) -> "RibFactory":
        return RibFactory(
            plate_size_range=entity["plate_size_range"],
            plate_thickness_range=entity["plate_thickness_range"],
            disk_probability=entity.get("disk_probability", 0.35),
            gusset_probability=entity.get("gusset_probability", 0.2),
        )


factories.register("rib", RibFactory)
