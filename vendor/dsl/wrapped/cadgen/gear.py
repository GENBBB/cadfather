import logging
import math
import re
from copy import deepcopy

import numpy as np

from .base import BaseFactory, BaseOperation
from .registry import factories

logger = logging.getLogger(__name__)


def gear(
    workplane,
    outer_radius,
    inner_radius,
    cylinder_height,
    number_outer_teeth,
    outer_tooth_profile,
    number_inner_teeth,
    inner_tooth_profile,
    tooth_depth,
    offset_from_top_edge,
    translation=(0.0, 0.0, 0.0),
):
    """
    Build a gear from sketches on ``workplane``.

    The base cylinder is an extruded circle sketch, optionally with an inner
    circle cut from the same sketch. Each tooth sketch is evaluated in its own
    local XY workplane: local X is tangent to the cylinder and local Y points
    away from the outer surface or into the inner hole. ``gear_sketch`` may be
    an open integer-coordinate contour; this function closes it with a precise
    cylinder-surface arc before extrusion.
    """
    outer_radius = float(outer_radius)
    cylinder_height = float(cylinder_height)
    tooth_depth = float(tooth_depth)
    offset_from_top_edge = abs(float(offset_from_top_edge))
    inner_radius = None if inner_radius in (None, 0) else float(inner_radius)

    if outer_radius <= 0:
        raise ValueError("outer_radius must be positive")
    if math.isclose(cylinder_height, 0.0):
        raise ValueError("cylinder_height must be non-zero")
    if inner_radius is not None and not 0 < inner_radius < outer_radius:
        raise ValueError("inner_radius must be between 0 and outer_radius")
    translation = tuple(float(v) for v in translation)

    def _finish(body):
        return body.translate(translation)

    cylinder_sketch = workplane.sketch().circle(outer_radius)
    if inner_radius is not None:
        cylinder_sketch = cylinder_sketch.circle(inner_radius, mode="s")
    gear = cylinder_sketch.finalize().extrude(cylinder_height)

    direction = 1.0 if cylinder_height > 0 else -1.0
    height = abs(cylinder_height)
    tooth_plane_distance = abs(offset_from_top_edge)
    tooth_extent = abs(tooth_depth)
    if tooth_extent <= 0 or tooth_plane_distance > height:
        return _finish(gear)

    tooth_plane_offset = direction * tooth_plane_distance

    def _to_tuple(value):
        if hasattr(value, "toTuple"):
            value = value.toTuple()
        return tuple(float(v) for v in value)

    def _add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def _scale(vec, value):
        return (vec[0] * value, vec[1] * value, vec[2] * value)

    def _cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _unit(vec):
        length = math.sqrt(sum(v * v for v in vec))
        if math.isclose(length, 0.0):
            raise ValueError("Cannot create gear tooth workplane from zero vector")
        return tuple(v / length for v in vec)

    base_origin = _to_tuple(workplane.plane.origin)
    base_x_dir = _unit(_to_tuple(workplane.plane.xDir))
    base_y_dir = _unit(_to_tuple(workplane.plane.yDir))
    base_z_dir = _unit(_to_tuple(workplane.plane.zDir))

    def _profile_value(profile, key, default):
        if isinstance(profile, dict):
            return profile.get(key, default)
        return default

    def _tooth_workplane(radius, angle, inner=False):
        import cadquery as cq

        radial = _unit(
            _add(
                _scale(base_x_dir, math.cos(angle)),
                _scale(base_y_dir, math.sin(angle)),
            )
        )
        profile_y_dir = _scale(radial, -1.0 if inner else 1.0)
        x_dir = _unit(_cross(profile_y_dir, base_z_dir))
        origin = _add(
            _add(base_origin, _scale(base_z_dir, tooth_plane_offset)),
            _scale(radial, radius),
        )
        plane = cq.Plane(origin=origin, xDir=x_dir, normal=base_z_dir)
        return cq.Workplane(plane)

    def _sketch_points(expr):
        number = r"-?(?:\d+\.\d*|\d+|\.\d+)"
        command = re.compile(
            rf"\.(moveTo|lineTo)\(\s*({number})\s*,\s*({number})\s*\)"
            rf"|\.threePointArc\(\s*\(\s*{number}\s*,\s*{number}\s*\)\s*,"
            rf"\s*\(\s*({number})\s*,\s*({number})\s*\)\s*\)"
        )
        points = []
        for match in command.finditer(expr):
            if match.group(1):
                points.append((float(match.group(2)), float(match.group(3))))
            else:
                points.append((float(match.group(4)), float(match.group(5))))
        if not points:
            raise ValueError("gear_sketch must contain moveTo/lineTo/threePointArc commands")
        return points[0], points[-1]

    def _surface_point(radius, x, inner=False):
        limit = radius * 0.999
        x = min(limit, max(-limit, float(x)))
        sagitta = radius - math.sqrt(max(radius * radius - x * x, 0.0))
        y = sagitta if inner else -sagitta
        return (x, y)

    def _almost_same(a, b):
        return math.isclose(a[0], b[0], abs_tol=1e-9) and math.isclose(
            a[1],
            b[1],
            abs_tol=1e-9,
        )

    def _apply_gear_sketch(tooth_workplane, gear_sketch, radius, inner=False):
        if not isinstance(gear_sketch, str) or not gear_sketch.strip():
            raise ValueError("gear_sketch must be a non-empty CadQuery expression")

        expr = gear_sketch.strip().replace(".close()", "")
        if expr.startswith("."):
            expr = "tooth_workplane" + expr
        elif not expr.startswith(("tooth_workplane.", "wp.")):
            expr = "tooth_workplane." + expr

        start, end = _sketch_points(expr)
        start_on_surface = _surface_point(radius, start[0], inner=inner)
        end_on_surface = _surface_point(radius, end[0], inner=inner)

        sketch = eval(
            expr,
            {"__builtins__": {}},
            {"tooth_workplane": tooth_workplane, "wp": tooth_workplane},
        )
        if not _almost_same(end, end_on_surface):
            sketch = sketch.lineTo(*end_on_surface)
        return sketch.threePointArc((0.0, 0.0), start_on_surface).close()

    def _make_tooth(radius, angle, profile, inner=False):
        tooth_workplane = _tooth_workplane(radius, angle, inner=inner)
        gear_sketch = _profile_value(profile, "gear_sketch", "")
        sketch = _apply_gear_sketch(tooth_workplane, gear_sketch, radius, inner=inner)
        return sketch.extrude(direction * tooth_extent)

    def _add_teeth(body, radius, count, profile, inner=False):
        count = int(count or 0)
        if count <= 0 or not profile:
            return body
        first_angle = math.radians(float(_profile_value(profile, "first_gear_angle", 0.0)))
        pitch_angle = 2.0 * math.pi / count
        for i in range(count):
            angle = first_angle + i * pitch_angle
            body = body.union(_make_tooth(radius, angle, profile, inner=inner))
        return body

    gear = _add_teeth(
        gear,
        outer_radius,
        number_outer_teeth,
        outer_tooth_profile,
        inner=False,
    )
    if inner_radius is not None:
        gear = _add_teeth(
            gear,
            inner_radius,
            number_inner_teeth,
            inner_tooth_profile,
            inner=True,
        )

    return _finish(gear)


class Gear(BaseOperation):
    def __init__(
        self,
        outer_radius: float,
        inner_radius: float | None,
        cylinder_height: float,
        number_outer_teeth: int,
        outer_tooth_profile: dict,
        number_inner_teeth: int,
        inner_tooth_profile: dict,
        tooth_depth: float,
        offset_from_top_edge: float,
        center: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    ):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.cylinder_height = cylinder_height
        self.number_outer_teeth = number_outer_teeth
        self.outer_tooth_profile = outer_tooth_profile
        self.number_inner_teeth = number_inner_teeth
        self.inner_tooth_profile = inner_tooth_profile
        self.tooth_depth = tooth_depth
        self.offset_from_top_edge = offset_from_top_edge
        self.center = list(center)

    def to_string(self, plane: str) -> str:
        expr = (
            f"gear({plane}, "
            f"outer_radius={self.outer_radius!r}, "
            f"inner_radius={self.inner_radius!r}, "
            f"cylinder_height={self.cylinder_height!r}, "
            f"number_outer_teeth={self.number_outer_teeth!r}, "
            f"outer_tooth_profile={self.outer_tooth_profile!r}, "
            f"number_inner_teeth={self.number_inner_teeth!r}, "
            f"inner_tooth_profile={self.inner_tooth_profile!r}, "
            f"tooth_depth={self.tooth_depth!r}, "
            f"offset_from_top_edge={self.offset_from_top_edge!r}, "
            f"translation=({self.center[0]!r}, {self.center[1]!r}, {self.center[2]!r}))"
        )
        return expr

    def transform(
        self,
        shift: list[float],
        scale: float,
        plane_axis: int | None = None,
    ) -> None:
        self.outer_radius *= scale
        if self.inner_radius is not None:
            self.inner_radius *= scale
        self.cylinder_height *= scale
        self.tooth_depth *= scale
        self.offset_from_top_edge *= scale
        self.outer_tooth_profile = self._scale_profile(self.outer_tooth_profile, scale)
        self.inner_tooth_profile = self._scale_profile(self.inner_tooth_profile, scale)
        if plane_axis is None:
            self.center = [(self.center[i] + shift[i]) * scale for i in range(3)]
        else:
            self.center = [self.center[i] * scale for i in range(3)]
            for i in range(3):
                if i != plane_axis:
                    self.center[i] += shift[i] * scale

    @staticmethod
    def _scale_profile(profile: dict, scale: float) -> dict:
        return Gear._map_profile_sketch_numbers(profile, lambda value: value * scale)

    def round(self) -> None:
        self.outer_radius = max(1, round(self.outer_radius))
        if self.inner_radius is not None:
            self.inner_radius = round(self.inner_radius)
            if self.inner_radius <= 0:
                self.inner_radius = None
            elif self.inner_radius >= self.outer_radius:
                self.inner_radius = max(1, self.outer_radius - 1)

        self.cylinder_height = round(self.cylinder_height)
        if self.cylinder_height == 0:
            self.cylinder_height = 1
        self.tooth_depth = max(1, round(self.tooth_depth))
        self.offset_from_top_edge = max(0, round(self.offset_from_top_edge))
        max_offset = abs(self.cylinder_height)
        if self.offset_from_top_edge >= max_offset:
            self.offset_from_top_edge = max(0, max_offset - 1)

        self.number_outer_teeth = max(0, int(round(self.number_outer_teeth)))
        self.number_inner_teeth = max(0, int(round(self.number_inner_teeth)))
        self.outer_tooth_profile = self._round_profile(self.outer_tooth_profile)
        self.inner_tooth_profile = self._round_profile(self.inner_tooth_profile)
        self.center = [round(v) for v in self.center]

    @staticmethod
    def _round_profile(profile: dict) -> dict:
        if not profile:
            return {}
        rounded = dict(profile)
        if "first_gear_angle" in rounded:
            rounded["first_gear_angle"] = int(round(float(rounded["first_gear_angle"])))
        return Gear._map_profile_sketch_numbers(rounded, round)

    def fix(self) -> None:
        height = abs(self.cylinder_height)
        self.offset_from_top_edge = abs(self.offset_from_top_edge)
        self.tooth_depth = abs(self.tooth_depth)
        if self.offset_from_top_edge > height:
            self.offset_from_top_edge = height
        if self.offset_from_top_edge + self.tooth_depth > height:
            self.tooth_depth = max(1e-6, height - self.offset_from_top_edge)

        if self.inner_radius is not None:
            if self.inner_radius <= 0 or self.inner_radius >= self.outer_radius:
                self.inner_radius = None
                self.number_inner_teeth = 0
                self.inner_tooth_profile = {}

        self.outer_tooth_profile = self._fix_profile(
            self.outer_tooth_profile,
        )
        self.inner_tooth_profile = self._fix_profile(
            self.inner_tooth_profile,
        )

        if self.number_outer_teeth <= 0:
            self.outer_tooth_profile = {}
        elif not self.outer_tooth_profile:
            self.number_outer_teeth = 0

        if self.inner_radius is None or self.number_inner_teeth <= 0:
            self.number_inner_teeth = 0
            self.inner_tooth_profile = {}
        elif not self.inner_tooth_profile:
            self.number_inner_teeth = 0

        if self.number_outer_teeth <= 0 and self.number_inner_teeth <= 0:
            raise ValueError("gear must have outer or inner teeth")

    @staticmethod
    def _fix_profile(
        profile: dict,
    ) -> dict:
        if not profile:
            return {}

        fixed = dict(profile)
        if not fixed.get("gear_sketch"):
            return {}
        fixed["first_gear_angle"] = int(round(float(fixed.get("first_gear_angle", 0.0)))) % 360
        fixed["gear_sketch"] = str(fixed["gear_sketch"]).strip()
        return fixed

    @staticmethod
    def _format_number(value: float) -> str:
        value = float(value)
        if math.isclose(value, 0.0, abs_tol=1e-9):
            return "0"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _map_profile_sketch_numbers(profile: dict, mapper) -> dict:
        mapped = deepcopy(profile)
        sketch = mapped.get("gear_sketch")
        if not sketch:
            return mapped

        number = r"(?<![\w.])-?(?:\d+\.\d*|\d+|\.\d+)(?![\w.])"

        def replace(match):
            return Gear._format_number(mapper(float(match.group(0))))

        mapped["gear_sketch"] = re.sub(number, replace, str(sketch))
        return mapped

    def to_dict(self) -> dict:
        return {
            "type": "Gear",
            "outer_radius": self.outer_radius,
            "inner_radius": self.inner_radius,
            "cylinder_height": self.cylinder_height,
            "number_outer_teeth": self.number_outer_teeth,
            "outer_tooth_profile": self.outer_tooth_profile,
            "number_inner_teeth": self.number_inner_teeth,
            "inner_tooth_profile": self.inner_tooth_profile,
            "tooth_depth": self.tooth_depth,
            "offset_from_top_edge": self.offset_from_top_edge,
            "center": self.center,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Gear":
        assert entity["type"] == "Gear", f"Trying to build Gear from type {entity['type']}"
        return Gear(
            entity["outer_radius"],
            entity.get("inner_radius"),
            entity["cylinder_height"],
            entity["number_outer_teeth"],
            entity["outer_tooth_profile"],
            entity["number_inner_teeth"],
            entity["inner_tooth_profile"],
            entity["tooth_depth"],
            entity["offset_from_top_edge"],
            entity.get("center", (0.0, 0.0, 0.0)),
        )


class GearFactory(BaseFactory):
    def __init__(
        self,
        *,
        hole_probability: float = 0.75,
        outer_teeth_probability: float = 0.9,
        inner_teeth_probability: float = 0.45,
        trapezoid_probability: float = 0.7,
    ):
        self.hole_probability = hole_probability
        self.outer_teeth_probability = outer_teeth_probability
        self.inner_teeth_probability = inner_teeth_probability
        self.trapezoid_probability = trapezoid_probability
        self.world_size = 200.0

    def generate(self) -> Gear:
        world_size = float(self.world_size)
        outer_radius = float(np.random.uniform(0.1 * world_size, 0.95 * world_size))
        sign = float(np.random.choice([-1, 1]))
        cylinder_height = sign * float(
            np.random.uniform(0.05 * world_size, world_size)
        )
        height = abs(cylinder_height)

        if np.random.random() < 0.1:
            offset_from_top_edge = 0.0
        else:
            offset_from_top_edge = float(np.random.uniform(0.0, 0.48 * height))

        max_tooth_depth = max(0.0, height - offset_from_top_edge)
        tooth_depth = float(np.random.uniform(0.0, max_tooth_depth))

        inner_radius = self._sample_inner_radius(outer_radius)

        number_outer_teeth = 0
        outer_tooth_profile = {}
        if np.random.random() < self.outer_teeth_probability:
            outer_tooth_profile, number_outer_teeth = self._sample_teeth(
                outer_radius,
                inner=False,
            )

        number_inner_teeth = 0
        inner_tooth_profile = {}
        inner_teeth_required = number_outer_teeth <= 0
        if inner_teeth_required and inner_radius is None:
            inner_radius = self._sample_inner_radius(outer_radius, required=True)

        if (
            inner_radius is not None
            and (
                inner_teeth_required
                or np.random.random() < self.inner_teeth_probability
            )
        ):
            inner_tooth_profile, number_inner_teeth = self._sample_teeth(
                inner_radius,
                inner=True,
            )

        gear = Gear(
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            cylinder_height=cylinder_height,
            number_outer_teeth=number_outer_teeth,
            outer_tooth_profile=outer_tooth_profile,
            number_inner_teeth=number_inner_teeth,
            inner_tooth_profile=inner_tooth_profile,
            tooth_depth=tooth_depth,
            offset_from_top_edge=offset_from_top_edge,
        )
        logger.info(
            "Generated gear: outer_radius=%s inner_radius=%s height=%s outer_teeth=%s inner_teeth=%s",
            outer_radius,
            inner_radius,
            cylinder_height,
            number_outer_teeth,
            number_inner_teeth,
        )
        return gear

    def _sample_inner_radius(
        self,
        outer_radius: float,
        *,
        required: bool = False,
    ) -> float | None:
        if not required and np.random.random() >= self.hole_probability:
            return None

        min_inner = 0.1 * outer_radius
        max_inner = 0.9 * outer_radius
        if max_inner <= min_inner:
            return None
        return float(np.random.uniform(min_inner, max_inner))

    def _sample_teeth(
        self,
        radius: float,
        *,
        inner: bool,
    ) -> tuple[dict, int]:
        tooth_profile, tooth_width = self._make_tooth_profile(
            radius,
            inner=inner,
        )
        tooth_count = self._sample_tooth_count(
            radius,
            tooth_width,
        )
        tooth_profile = self._set_first_gear_angle(
            tooth_profile,
            tooth_count,
        )
        return tooth_profile, tooth_count

    @staticmethod
    def _randint_inclusive(bounds: tuple[int, int]) -> int:
        low, high = bounds
        return int(np.random.randint(int(low), int(high) + 1))

    def _sample_tooth_count(
        self,
        radius: float,
        tooth_width: float,
    ) -> int:
        if radius <= 0 or tooth_width <= 0:
            return 0

        pitch_fill = 0.78
        q = tooth_width / (2.0 * radius * pitch_fill)
        if q >= 1.0:
            return 0

        max_by_width = int(math.floor(math.pi / math.asin(q)))
        low, high = 3, max_by_width
        if high < 3:
            return 0
        return self._randint_inclusive((low, high))

    @staticmethod
    def _set_first_gear_angle(profile: dict, tooth_count: int) -> dict:
        if tooth_count <= 0:
            return {}
        profile = dict(profile)
        profile["first_gear_angle"] = int(
            np.random.randint(0, max(1, round(360.0 / tooth_count)))
        )
        return profile

    def _make_tooth_profile(
        self,
        radius: float,
        *,
        inner: bool,
    ) -> tuple[dict, float]:
        min_base_half_width = max(1, int(math.ceil(radius * 0.055)))
        max_base_half_width = max(min_base_half_width, int(math.floor(radius * 0.17)))
        base_half_width = self._randint_inclusive(
            (min_base_half_width, max_base_half_width)
        )
        if np.random.random() < self.trapezoid_probability:
            tip_half_width = int(round(base_half_width * float(np.random.uniform(0.35, 0.85))))
        else:
            tip_half_width = int(round(base_half_width * float(np.random.uniform(0.85, 1.05))))
        sketch_kind = int(np.random.randint(6))

        base_half_width = min(base_half_width, max(1, int(radius * 0.85)))
        tip_half_width = max(1, min(tip_half_width, base_half_width))
        surface_sagitta = radius - math.sqrt(
            max(radius * radius - base_half_width * base_half_width, 0.0)
        )
        min_radial_depth = max(1, int(math.ceil(radius * 0.08)))
        max_radial_depth = max(min_radial_depth, int(math.floor(radius * 0.22)))
        if inner:
            min_radial_depth = max(
                min_radial_depth,
                int(math.ceil(surface_sagitta + max(1.0, radius * 0.02))),
            )
            max_radial_depth = max(max_radial_depth, int(math.ceil(min_radial_depth * 1.4)))
            max_radial_depth = min(max_radial_depth, max(min_radial_depth, int(radius * 0.65)))
        radial_depth = self._randint_inclusive((min_radial_depth, max_radial_depth))

        sketch = f".moveTo({-base_half_width}, 0)"

        x_values = [-base_half_width, base_half_width, 0.0]
        if sketch_kind == 0:
            sketch += f".lineTo(0, {radial_depth}).lineTo({base_half_width}, 0)"
        elif sketch_kind == 1:
            tip_bulge = max(1, int(round(radial_depth * float(np.random.uniform(0.08, 0.25)))))
            x_values.extend([tip_half_width, -tip_half_width, 0.0])
            sketch += (
                f".lineTo({-tip_half_width}, {radial_depth})"
                f".threePointArc((0, {radial_depth + tip_bulge}), "
                f"({tip_half_width}, {radial_depth}))"
                f".lineTo({base_half_width}, 0)"
            )
        elif sketch_kind == 2:
            shoulder_y = max(1, int(round(radial_depth * float(np.random.uniform(0.45, 0.7)))))
            shoulder_half_width = max(
                tip_half_width,
                int(round(0.5 * (base_half_width + tip_half_width))),
            )
            x_values.extend([
                shoulder_half_width,
                tip_half_width,
                -tip_half_width,
                -shoulder_half_width,
            ])
            sketch += (
                f".lineTo({-shoulder_half_width}, {shoulder_y})"
                f".lineTo({-tip_half_width}, {radial_depth})"
                f".lineTo({tip_half_width}, {radial_depth})"
                f".lineTo({shoulder_half_width}, {shoulder_y})"
                f".lineTo({base_half_width}, 0)"
            )
        elif sketch_kind == 3:
            shoulder_y = max(1, int(round(radial_depth * float(np.random.uniform(0.45, 0.75)))))
            shoulder_half_width = max(
                tip_half_width,
                int(round(0.5 * (base_half_width + tip_half_width))),
            )
            x_values.extend([
                -shoulder_half_width,
                -tip_half_width,
                tip_half_width,
            ])
            sketch += (
                f".threePointArc(({-shoulder_half_width}, {shoulder_y}), "
                f"({-tip_half_width}, {radial_depth}))"
                f".lineTo({tip_half_width}, {radial_depth})"
                f".lineTo({base_half_width}, 0)"
            )
        elif sketch_kind == 4:
            x_values.extend([-tip_half_width, tip_half_width])
            sketch += (
                f".threePointArc(({-tip_half_width}, {radial_depth}), "
                f"(0, {radial_depth}))"
                f".threePointArc(({tip_half_width}, {radial_depth}), "
                f"({base_half_width}, 0))"
            )
        else:
            shoulder_y = max(1, int(round(radial_depth * float(np.random.uniform(0.45, 0.75)))))
            tip_bulge = max(1, int(round(radial_depth * float(np.random.uniform(0.08, 0.25)))))
            x_values.extend([-tip_half_width, tip_half_width, 0.0])
            sketch += (
                f".threePointArc(({-tip_half_width}, {shoulder_y}), "
                f"({-tip_half_width}, {radial_depth}))"
                f".threePointArc((0, {radial_depth + tip_bulge}), "
                f"({tip_half_width}, {radial_depth}))"
                f".threePointArc(({tip_half_width}, {shoulder_y}), "
                f"({base_half_width}, 0))"
            )

        tooth_width = max(x_values) - min(x_values)
        return {"first_gear_angle": 0, "gear_sketch": sketch}, tooth_width

    @staticmethod
    def from_dict(entity: dict) -> "GearFactory":
        return GearFactory(
            hole_probability=entity.get("hole_probability", 0.75),
            outer_teeth_probability=entity.get("outer_teeth_probability", 0.9),
            inner_teeth_probability=entity.get("inner_teeth_probability", 0.45),
            trapezoid_probability=entity.get("trapezoid_probability", 0.7),
        )

    def to_dict(self) -> dict:
        return {
            "type": "GearFactory",
            "hole_probability": self.hole_probability,
            "outer_teeth_probability": self.outer_teeth_probability,
            "inner_teeth_probability": self.inner_teeth_probability,
            "trapezoid_probability": self.trapezoid_probability,
        }


factories.register("gear", GearFactory)
