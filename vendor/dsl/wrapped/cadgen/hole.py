import ast
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np

if TYPE_CHECKING:
    from cadquery import Workplane

# from cadquery_addons import *

from .base import BaseFactory, BaseOperation
from .extrude import (
    WORKPLANE_BY_NORMAL,
    WORKPLANE_OFFSET_INDEX,
    _double_quoted_string,
    _format_number,
    _format_point,
    _join_workplane_and_sketch,
)
from .registry import factories
from .sketch import Sketch, SketchFactory
from .sketch_global_coords import (
    sketch_chain_only,
    sketch_to_global_coords,
    transform_sketch_local_axes,
)
from .surface_sampling import (
    AxisWorkplane,
    PreparedSurfaceSampler,
    SampledSite,
    axis_vector,
    closest_point_on_face_surface,
    closest_surface_point_from_compound,
    normal_at_face_uv,
    point_is_on_face,
    precompute_surface_compound,
    shape_from_cad_object,
)
from .utils import (
    compound_to_mesh,
    get_face_2d_radius,
    get_sketch_radius_range,
    is_face_outer,
    normal_distance_to_next_face,
    put_sketch_on_face_simplified,
)

logger = logging.getLogger(__name__)


WORKPLANE_AXES = {
    "XY": ("X", "Y"),
    "YZ": ("Y", "Z"),
    "ZX": ("Z", "X"),
}


def _flatten_method_chain(
    node: ast.AST,
) -> tuple[ast.AST, list[tuple[str, list, list]]]:
    calls = []
    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        calls.append(
            (current.func.attr, list(current.args), list(current.keywords))
        )
        current = current.func.value
    calls.reverse()
    return current, calls


def _rebuild_method_chain(
    base: ast.AST,
    calls: list[tuple[str, list, list]],
) -> ast.AST:
    current = base
    for name, args, keywords in calls:
        current = ast.Call(
            func=ast.Attribute(value=current, attr=name, ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )
    return current


def _number_node(value: float) -> ast.Constant:
    value = float(value)
    return ast.Constant(int(value) if value.is_integer() else value)


def _surface_normal_at_point(
    shape: cq.Shape,
    surface_point: cq.Vector,
) -> np.ndarray:
    point = np.array(surface_point.toTuple(), dtype=float)
    faces = [
        face
        for face in shape.Faces()
        if point_is_on_face(face, point, tol=1e-5)
    ] or shape.Faces()
    closest = None
    closest_distance = float("inf")
    for face in faces:
        try:
            projected, u, v = closest_point_on_face_surface(face, point)
        except RuntimeError:
            continue
        distance = float(np.linalg.norm(projected - point))
        if distance < closest_distance:
            closest = (face, u, v)
            closest_distance = distance

    if closest is None:
        raise RuntimeError("Could not compute normal at closest surface point")

    face, u, v = closest
    return normal_at_face_uv(face, u, v)


def _dominant_axis_selector(
    shape: cq.Shape,
    surface_point: cq.Vector,
) -> tuple[str, str]:
    normal = _surface_normal_at_point(shape, surface_point)
    axis_index = int(np.argmax(np.abs(normal)))
    axis = ("X", "Y", "Z")[axis_index]
    sign = ">" if normal[axis_index] >= 0 else "<"
    return sign, axis


def hole(
    r: cq.Workplane | cq.Shape,
    point: tuple[float, float, float],
    workplane: str,
    sketch: str,
    extrude_height: float,
):
    if workplane not in WORKPLANE_OFFSET_INDEX:
        raise ValueError("workplane must be one of 'XY', 'YZ', or 'ZX'")

    shape = shape_from_cad_object(r)
    surface_compound = precompute_surface_compound(shape)
    surface_point = closest_surface_point_from_compound(surface_compound, point)
    offset = surface_point.toTuple()[WORKPLANE_OFFSET_INDEX[workplane]]
    prefix = f"cq.Workplane('{workplane}').workplane(offset={offset})"
    expression = (
        _join_workplane_and_sketch(prefix, sketch)
        + f".extrude({extrude_height})"
    )
    cutter = eval(expression, {"cq": cq})

    sign, axis = _dominant_axis_selector(shape, surface_point)
    wire = cutter.faces(f"{sign}{axis}").val().outerWire()
    swept_vector = axis_vector(axis if sign == ">" else f"-{axis}").multiply(10)
    swept_solid = cq.Solid.extrudeLinear(
        wire,
        [],
        swept_vector,
    )
    cutter = cutter.union(swept_solid)

    base = r if isinstance(r, cq.Workplane) else cq.Workplane("XY").add(shape)
    return base.cut(cutter)


class Hole(BaseOperation):
    SKETCH_HOLE_TYPES = {"blind", "through", "cut_thru_all"}
    ATTACH_AT_FACE_OFFSET = 1.0
    BLIND_SINGLE_CIRCLE_PROBABILITY = 0.2
    BLIND_INNER_LOOP_PROBABILITY = 0.2
    BLIND_INNER_RADIUS_MIN_RATIO = 0.8
    BLIND_INNER_RADIUS_MAX_RATIO = 0.98
    BLIND_INNER_RADIUS_MARGIN = 2

    def __init__(
        self,
        radius: float | None = None,
        cbore_radius: float | None = None,
        csk_radius: float | None = None,
        depth: float | None = None,
        cbore_depth: float | None = None,
        csk_angle: int | None = None,
        point: list[float] | None = None,
        sketch: Sketch | None = None,
        world_size: float | None = None,
        hole_type: str | None = None,
        multi_cbore_radii: list[float] | None = None,
        multi_cbore_depths: list[float] | None = None,
        workplane: str | None = None,
        normal_axis: str | None = None,
        normal_sign: float | None = None,
        blind_inner_radius: float | None = None,
    ):
        self.radius = radius
        self.cbore_radius = cbore_radius
        self.csk_radius = csk_radius
        self.depth = depth
        self.cbore_depth = cbore_depth
        self.csk_angle = csk_angle
        self.point = point
        self.sketch = sketch
        self.world_size = world_size
        self.multi_cbore_radii = multi_cbore_radii
        self.multi_cbore_depths = multi_cbore_depths
        self.workplane = workplane
        self.normal_axis = normal_axis
        self.normal_sign = normal_sign
        self.blind_inner_radius = blind_inner_radius
        self.hole_type = hole_type or self._infer_hole_type()

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{float(value):g}"

    @classmethod
    def _format_point(cls, point) -> str:
        return "(" + ", ".join(cls._format_number(value) for value in point) + ")"

    def _has_world_point(self) -> bool:
        return self.point is not None and len(self.point) == 3

    def _infer_hole_type(self) -> str:
        if self.cbore_radius is not None:
            return "cbore"
        if self.csk_radius is not None:
            return "csk"
        if self.sketch is not None:
            return "blind"
        if self.multi_cbore_radii is not None:
            return "multi_cbore"
        return "legacy_default"

    @staticmethod
    def _normal_sign_from_axis(axis: str) -> float:
        return -1.0 if axis.startswith("-") else 1.0

    @staticmethod
    def _workplane_axis_from_normal(normal_axis: str) -> str:
        return WORKPLANE_BY_NORMAL[normal_axis[-1]]

    @staticmethod
    def _default_source_axes(workplane_axis: str) -> tuple[str, str]:
        try:
            return WORKPLANE_AXES[workplane_axis]
        except KeyError as exc:
            raise ValueError(
                "workplane must be one of 'XY', 'YZ', or 'ZX'"
            ) from exc

    def _hole_workplane_and_sign(
        self,
        plane,
    ) -> tuple[str, float]:
        if self.workplane is not None:
            workplane_axis = self.workplane
            normal_sign = self.normal_sign
            if normal_sign is None:
                normal_sign = 1.0
            return workplane_axis, float(normal_sign)

        if self.normal_axis is not None:
            return (
                self._workplane_axis_from_normal(self.normal_axis),
                float(self.normal_sign)
                if self.normal_sign is not None
                else self._normal_sign_from_axis(self.normal_axis),
            )

        if isinstance(plane, AxisWorkplane):
            return (
                self._workplane_axis_from_normal(plane.normal),
                self._normal_sign_from_axis(plane.normal),
            )

        raise ValueError(
            f"{self.hole_type} holes require sampled surface normal metadata"
        )

    def _sketch_arg_for_hole(
        self,
        workplane_axis: str,
        point: tuple[float, float, float],
        sketch: Sketch,
    ) -> str:
        local_sketch = sketch.to_string(f"cq.Workplane('{workplane_axis}')")
        source_x_dir, source_y_dir = self._default_source_axes(workplane_axis)
        aligned_sketch = transform_sketch_local_axes(
            local_sketch,
            source_x_dir,
            source_y_dir,
            workplane_axis,
        )
        global_sketch = sketch_to_global_coords(
            workplane_axis,
            point,
            aligned_sketch,
        )
        return sketch_chain_only(global_sketch)

    @staticmethod
    def _single_circle_wire(sketch: Sketch | None) -> dict | None:
        if (
            sketch is None
            or sketch.array_pattern_sketch is not None
            or len(sketch.wires) != 1
        ):
            return None
        wire = sketch.wires[0]
        if wire.get("type") != "circle" or not wire.get("outer", False):
            return None
        return wire

    @staticmethod
    def _blind_single_circle_sketch() -> Sketch:
        return Sketch(
            [
                {
                    "type": "circle",
                    "outer": True,
                    "center": [0, 0],
                    "radius": 1,
                }
            ]
        )

    @classmethod
    def maybe_sample_blind_single_circle_sketch(cls) -> Sketch | None:
        if np.random.random() >= cls.BLIND_SINGLE_CIRCLE_PROBABILITY:
            return None
        return cls._blind_single_circle_sketch()

    @classmethod
    def _sample_blind_inner_radius(cls, outer_radius: float) -> int | None:
        max_inner_radius = (
            int(round(float(outer_radius))) - cls.BLIND_INNER_RADIUS_MARGIN
        )
        if max_inner_radius <= 0:
            return None
        sampled_radius = int(
            round(
                np.random.uniform(
                    cls.BLIND_INNER_RADIUS_MIN_RATIO * outer_radius,
                    cls.BLIND_INNER_RADIUS_MAX_RATIO * outer_radius,
                )
            )
        )
        inner_radius = min(sampled_radius, max_inner_radius)
        if inner_radius <= 0:
            return None
        return inner_radius

    @classmethod
    def maybe_sample_blind_inner_radius(cls, sketch: Sketch) -> int | None:
        wire = cls._single_circle_wire(sketch)
        if wire is None:
            return None
        if np.random.random() >= cls.BLIND_INNER_LOOP_PROBABILITY:
            return None
        return cls._sample_blind_inner_radius(float(wire["radius"]))

    @staticmethod
    def _insert_blind_inner_circle(
        sketch_arg: str,
        inner_radius: float,
    ) -> str:
        source = sketch_arg.strip()
        leading_dot_base = "__sketch_base__"
        has_leading_dot = source.startswith(".")
        if has_leading_dot:
            source = leading_dot_base + source

        expression = ast.parse(source, mode="eval")
        base, calls = _flatten_method_chain(expression.body)
        circle_indices = [
            index for index, (name, _, _) in enumerate(calls) if name == "circle"
        ]
        if len(circle_indices) != 1:
            return sketch_arg

        circle_index = circle_indices[0]
        if circle_index == 0 or calls[circle_index - 1][0] != "push":
            return sketch_arg

        _, push_args, push_keywords = calls[circle_index - 1]
        inner_circle_call = (
            "circle",
            [_number_node(inner_radius)],
            [ast.keyword(arg="mode", value=ast.Constant("s"))],
        )
        calls[circle_index + 1:circle_index + 1] = [
            ("push", deepcopy(push_args), deepcopy(push_keywords)),
            inner_circle_call,
        ]
        rebuilt = ast.Expression(body=_rebuild_method_chain(base, calls))
        ast.fix_missing_locations(rebuilt)
        result = ast.unparse(rebuilt)
        if has_leading_dot and result.startswith(leading_dot_base):
            return result[len(leading_dot_base):]
        return result

    def _with_blind_inner_circle(self, sketch_arg: str) -> str:
        if self.hole_type != "blind" or self.blind_inner_radius is None:
            return sketch_arg
        if self._single_circle_wire(self.sketch) is None:
            return sketch_arg
        return self._insert_blind_inner_circle(
            sketch_arg,
            self.blind_inner_radius,
        )

    def _hole_call_string(
        self,
        point: tuple[float, float, float],
        workplane_axis: str,
        sketch_arg: str,
        cut_depth: float,
        normal_sign: float,
    ) -> str:
        extrude_height = -float(normal_sign) * float(cut_depth)
        args = (
            f"r, {_format_point(point)}, {workplane_axis!r}, "
            f"{_double_quoted_string(sketch_arg)}, {_format_number(extrude_height)}"
        )
        return f"r = hole({args})\n"

    def to_string(self, plane) -> str:
        logger.info(
            "Building hole string with type=%s, radius=%s, cbore_radius=%s, depth=%s, cbore_depth=%s, csk_angle=%s",
            self.hole_type,
            self.radius,
            self.cbore_radius,
            self.depth,
            self.cbore_depth,
            self.csk_angle,
        )

        if self.hole_type in self.SKETCH_HOLE_TYPES:
            assert self.sketch is not None, "sketch is required for sketch-based holes"
            if self._has_world_point():
                point = tuple(float(value) for value in self.point)
            else:
                assert isinstance(plane, str) and plane.startswith("point"), (
                    f"{self.hole_type} holes require a point"
                )
                point_expr = plane
                if self.hole_type == "cut_thru_all":
                    assert self.world_size is not None, "world_size is required"
                    depth = 2 * self.world_size
                else:
                    assert self.depth is not None, "depth is required"
                    depth = self.depth

                sketch_str = self.sketch.to_string_one_sketch(
                    "", self.sketch.wires, skip_on_one_wire=True
                )
                sketch_str = self._with_blind_inner_circle(sketch_str)
                cut_depth = abs(depth) + self.ATTACH_AT_FACE_OFFSET
                extrude_expr = f".extrude({cut_depth:g})"
                return (
                    f"r=r.attach_at([{point_expr}], "
                    f"'{sketch_str}.finalize(){extrude_expr}',"
                    "combine='s')\n"
                )

            if self.hole_type == "cut_thru_all":
                assert self.world_size is not None, "world_size is required"
                depth = 2 * self.world_size
            else:
                assert self.depth is not None, "depth is required"
                depth = self.depth

            workplane_axis, normal_sign = self._hole_workplane_and_sign(plane)
            sketch_arg = self._sketch_arg_for_hole(
                workplane_axis,
                point,
                self.sketch,
            )
            sketch_arg = self._with_blind_inner_circle(sketch_arg)
            cut_depth = abs(depth) + self.ATTACH_AT_FACE_OFFSET
            return self._hole_call_string(
                point,
                workplane_axis,
                sketch_arg,
                cut_depth,
                normal_sign,
            )

        if self.hole_type == "multi_cbore":
            if self._has_world_point():
                point = tuple(float(value) for value in self.point)
            else:
                assert isinstance(plane, str) and plane.startswith("point"), (
                    "multi_cbore holes require a point"
                )
                point_expr = plane
                assert self.multi_cbore_radii is not None
                assert self.multi_cbore_depths is not None
                assert len(self.multi_cbore_radii) == len(self.multi_cbore_depths)
                expr = ""
                for radius, depth in zip(self.multi_cbore_radii, self.multi_cbore_depths):
                    cut_depth = abs(depth) + self.ATTACH_AT_FACE_OFFSET
                    expr += (
                        f"r=r.attach_at([{point_expr}], "
                        f"'.sketch().circle({radius}).finalize().extrude({cut_depth:g})',"
                        "combine='s')\n"
                    )
                return expr
            assert self.multi_cbore_radii is not None
            assert self.multi_cbore_depths is not None
            assert len(self.multi_cbore_radii) == len(self.multi_cbore_depths)
            workplane_axis, normal_sign = self._hole_workplane_and_sign(plane)
            expr = ""
            for radius, depth in zip(self.multi_cbore_radii, self.multi_cbore_depths):
                cut_depth = abs(depth) + self.ATTACH_AT_FACE_OFFSET
                sketch = Sketch(
                    [
                        {
                            "type": "circle",
                            "outer": True,
                            "center": [0, 0],
                            "radius": radius,
                        }
                    ]
                )
                sketch_arg = self._sketch_arg_for_hole(
                    workplane_axis,
                    point,
                    sketch,
                )
                expr += self._hole_call_string(
                    point,
                    workplane_axis,
                    sketch_arg,
                    cut_depth,
                    normal_sign,
                )
            return expr

        expr = "r=r"

        assert self.radius is not None, "radius is required"
        assert self.depth is not None, "depth is required"
        if self._has_world_point():
            point_expr = self._format_point(self.point)
            expr += f".copyWorkplane(workplane_by_point(r, {point_expr}))"
        else:
            assert plane is not None, f"{self.hole_type} holes require a workplane"
            center = self.point if self.point else (0, 0)
            tag = "r.workplane(origin=r.val().Center())" if "faces" in plane else plane
            push_points = (
                f".pushPoints([({center[0]}, {center[1]})])" if center != (0, 0) else ""
            )
            expr += f".copyWorkplane({tag}){push_points}"
        if self.hole_type == "cbore":
            assert self.cbore_radius is not None
            assert self.cbore_depth is not None
            expr += f".cboreHole({2*self.radius}, {2*self.cbore_radius}, {self.cbore_depth}, {self.depth})\n"
        elif self.hole_type == "csk":
            assert self.csk_radius is not None
            assert self.csk_angle is not None
            expr += f".cskHole({2*self.radius}, {2*self.csk_radius}, {self.csk_angle}, {self.depth})\n"
        elif self.hole_type == "legacy_default":
            expr += f".hole({2*self.radius},{self.depth})\n"
        else:
            raise ValueError(f"Unsupported circular hole type: {self.hole_type}")

        return expr

    def transform(self, shift: list[float], scale: float) -> None:
        if self.sketch is not None:
            self.sketch.transform([0, 0], scale)
        if self.radius:
            self.radius *= scale
        if self.cbore_radius:
            self.cbore_radius *= scale
        if self.csk_radius:
            self.csk_radius *= scale
        if self.depth:
            self.depth *= scale
        if self.cbore_depth:
            self.cbore_depth *= scale
        if self.multi_cbore_radii is not None:
            self.multi_cbore_radii = [
                radius * scale for radius in self.multi_cbore_radii
            ]
        if self.multi_cbore_depths is not None:
            self.multi_cbore_depths = [
                depth * scale for depth in self.multi_cbore_depths
            ]
        if self.blind_inner_radius is not None:
            self.blind_inner_radius *= scale

        if self.point is not None:
            if len(self.point) == 3:
                self.point = tuple((self.point[i] + shift[i]) * scale for i in range(3))
            else:
                self.point = (self.point[0] * scale, self.point[1] * scale)

    def round(self) -> None:
        if self.sketch is not None:
            self.sketch.round()

        if self.radius:
            self.radius = round(self.radius)
            if np.allclose(self.radius, 0):
                self.radius = 1
        if self.cbore_radius:
            self.cbore_radius = round(self.cbore_radius)
            if np.allclose(self.cbore_radius, 0):
                self.cbore_radius = 1
        if self.csk_radius:
            self.csk_radius = round(self.csk_radius)
            if np.allclose(self.csk_radius, 0):
                self.csk_radius = 1
        if self.depth:
            self.depth = round(self.depth)
            if np.allclose(self.depth, 0):
                self.depth = 1
        if self.cbore_depth:
            self.cbore_depth = round(self.cbore_depth)
            if np.allclose(self.cbore_depth, 0):
                self.cbore_depth = 1
        if self.multi_cbore_radii is not None:
            radii = [max(1, round(radius)) for radius in self.multi_cbore_radii]
            for i in range(1, len(radii)):
                if radii[i] <= radii[i - 1]:
                    radii[i] = radii[i - 1] + 1
            self.multi_cbore_radii = radii
        if self.multi_cbore_depths is not None:
            depths = [max(1, round(depth)) for depth in self.multi_cbore_depths]
            for i in range(1, len(depths)):
                if depths[i] >= depths[i - 1]:
                    depths[i] = depths[i - 1] - 1
                if depths[i] < 1:
                    depths[i] = 1
            self.multi_cbore_depths = depths
        if self.blind_inner_radius is not None:
            outer_wire = self._single_circle_wire(self.sketch)
            if outer_wire is None:
                self.blind_inner_radius = None
            else:
                max_inner_radius = (
                    round(float(outer_wire["radius"])) - self.BLIND_INNER_RADIUS_MARGIN
                )
                if max_inner_radius <= 0:
                    self.blind_inner_radius = None
                else:
                    self.blind_inner_radius = min(
                        max(1, round(self.blind_inner_radius)),
                        max_inner_radius,
                    )

        if self.point is not None:
            if len(self.point) == 3:
                self.point = tuple(round(value) for value in self.point)
            else:
                self.point = (round(self.point[0]), round(self.point[1]))

    def fix(self) -> None:
        if self.sketch is not None:
            self.sketch.fix()

    def to_dict(self) -> dict:
        return {
            "type": "Hole",
            "hole_type": self.hole_type,
            "radius": self.radius,
            "cbore_radius": self.cbore_radius,
            "csk_radius": self.csk_radius,
            "depth": self.depth,
            "cbore_depth": self.cbore_depth,
            "csk_angle": self.csk_angle,
            "point": self.point,
            "sketch": self.sketch.to_dict() if self.sketch is not None else None,
            "world_size": self.world_size,
            "multi_cbore_radii": self.multi_cbore_radii,
            "multi_cbore_depths": self.multi_cbore_depths,
            "workplane": self.workplane,
            "normal_axis": self.normal_axis,
            "normal_sign": self.normal_sign,
            "blind_inner_radius": self.blind_inner_radius,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Hole":
        assert (
            entity["type"] == "Hole"
        ), f"Trying to build Hole from type {entity['type']}"
        return Hole(
            entity["radius"],
            entity["cbore_radius"],
            entity["csk_radius"],
            entity["depth"],
            entity["cbore_depth"],
            entity["csk_angle"],
            entity["point"],
            Sketch.from_dict(entity["sketch"]) if entity.get("sketch") else None,
            entity.get("world_size"),
            entity.get("hole_type"),
            entity.get("multi_cbore_radii"),
            entity.get("multi_cbore_depths"),
            entity.get("workplane"),
            entity.get("normal_axis"),
            entity.get("normal_sign"),
            entity.get("blind_inner_radius"),
        )


class HoleFactory(BaseFactory):
    HOLE_TYPE_ALIASES = {
        "default": "blind",
        "cutThruAll": "cut_thru_all",
        "cut_through_all": "cut_thru_all",
    }
    HOLE_TYPES = {
        "blind",
        "through",
        "cut_thru_all",
        "cbore",
        "csk",
        "multi_cbore",
    }
    SKETCH_HOLE_TYPES = {"blind", "through", "cut_thru_all"}
    THROUGH_DEPTH_MARGIN = 1.1
    MIN_DEPTH = 1.0
    MAX_BLIND_DEPTH_RATIO = 0.95
    MIN_RADIUS = 1.0
    MIN_MULTI_CBORE_CYLINDERS = 3
    MAX_MULTI_CBORE_CYLINDERS = 5

    def __init__(
        self,
        hole_type_probabilities: dict[str, float],
        sketch_factory: SketchFactory | dict | None = None,
        mesh_deflection: float = 0.005,
    ):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)

        self.hole_type_probabilities = self._normalize_hole_type_probabilities(
            hole_type_probabilities
        )
        self.sketch_factory = sketch_factory
        self.mesh_deflection = mesh_deflection

    @classmethod
    def _normalize_hole_type(cls, hole_type: str) -> str:
        return cls.HOLE_TYPE_ALIASES.get(hole_type, hole_type)

    @classmethod
    def _normalize_hole_type_probabilities(
        cls, hole_type_probabilities: dict[str, float]
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for hole_type, probability in hole_type_probabilities.items():
            hole_type = cls._normalize_hole_type(hole_type)
            if hole_type not in cls.HOLE_TYPES:
                raise ValueError(f"Unsupported hole type: {hole_type}")
            normalized[hole_type] = normalized.get(hole_type, 0.0) + probability

        total = sum(normalized.values())
        assert total > 0, "hole_type_probabilities must contain positive mass"
        return {
            hole_type: probability / total
            for hole_type, probability in normalized.items()
        }

    def _generate_sketch(self) -> Sketch:
        assert self.sketch_factory is not None, "sketch_factory is required"
        sketch = self.sketch_factory.generate_one_sketch(
            self.sketch_factory.min_n_commands,
            self.sketch_factory.max_n_commands,
            zero_center=True,
        )
        sketch.wires = [wire for wire in sketch.wires if wire["outer"]]
        return sketch

    @staticmethod
    def _world_length_to_generation_length(
        length: float, world_size: float | None
    ) -> float:
        if world_size is None:
            return length
        return length / world_size

    @staticmethod
    def _point_from_face_triangles(face, triangles):
        for a, b, c, _ in sorted(triangles, key=lambda item: item[3], reverse=True):
            mesh_point = (a + b + c) / 3.0
            try:
                point, _, _ = closest_point_on_face_surface(face, mesh_point)
            except Exception:
                continue
            return tuple(float(value) for value in point)
        return None

    def _sample_site(self, workplane: "Workplane") -> SampledSite:
        sampler = PreparedSurfaceSampler.from_cad_object(
            workplane,
            mesh_deflection=self.mesh_deflection,
        )
        outer_indices = []
        for i, face in enumerate(sampler.faces):
            point_on_face = self._point_from_face_triangles(face, sampler.triangles[i])
            if point_on_face is None:
                continue
            try:
                if is_face_outer(workplane, face, point_on_face=point_on_face):
                    outer_indices.append(i)
            except Exception:
                logger.debug("Skipping face during hole outer-face filter", exc_info=True)
        if not outer_indices:
            raise ValueError("HoleFactory found no outer faces to sample")
        sampler = PreparedSurfaceSampler(
            shape=sampler.shape,
            surface_compound=sampler.surface_compound,
            faces=[sampler.faces[i] for i in outer_indices],
            areas=[sampler.areas[i] for i in outer_indices],
            triangles=[sampler.triangles[i] for i in outer_indices],
        )
        return sampler.sample_site()

    def _generate_sketch_hole_on_site(
        self,
        hole_type: str,
        workplane: "Workplane",
        site: SampledSite,
        world_size: float | None,
    ) -> Hole:
        blind_single_circle_sketch = None
        if hole_type == "blind":
            blind_single_circle_sketch = Hole.maybe_sample_blind_single_circle_sketch()
        sketch = blind_single_circle_sketch or self._generate_sketch()
        point_on_face = site.point.toTuple()
        face = site.face
        min_radius = self._world_length_to_generation_length(
            self.MIN_RADIUS, world_size
        )
        min_depth = self._world_length_to_generation_length(self.MIN_DEPTH, world_size)
        put_sketch_on_face_simplified(
            sketch,
            face,
            point_on_face,
            min_radius=min_radius,
        )

        depth = None
        if hole_type in {"blind", "through"}:
            extent = normal_distance_to_next_face(workplane, face, point=point_on_face)
            if hole_type == "blind":
                max_depth = self.MAX_BLIND_DEPTH_RATIO * extent
                if max_depth <= min_depth:
                    raise ValueError(
                        f"Blind hole extent too small: max_depth={max_depth}"
                    )
                depth = np.random.uniform(min_depth, max_depth)
            else:
                depth = max(min_depth, extent * self.THROUGH_DEPTH_MARGIN)

        blind_inner_radius = None
        if blind_single_circle_sketch is not None:
            blind_inner_radius = Hole.maybe_sample_blind_inner_radius(sketch)

        return Hole(
            depth=depth,
            point=point_on_face,
            sketch=sketch,
            world_size=world_size,
            hole_type=hole_type,
            workplane=WORKPLANE_BY_NORMAL[site.normal_axis],
            normal_axis=site.normal_axis,
            normal_sign=site.normal_sign,
            blind_inner_radius=blind_inner_radius,
        )

    def _generate_multi_cbore_on_site(
        self,
        workplane: "Workplane",
        site: SampledSite,
        world_size: float | None,
    ) -> Hole:
        point_on_face = site.point.toTuple()
        face = site.face
        min_radius = self._world_length_to_generation_length(
            self.MIN_RADIUS, world_size
        )
        min_depth = self._world_length_to_generation_length(self.MIN_DEPTH, world_size)
        min_step = self._world_length_to_generation_length(1.0, world_size)

        n_cylinders = np.random.randint(
            self.MIN_MULTI_CBORE_CYLINDERS,
            self.MAX_MULTI_CBORE_CYLINDERS + 1,
        )

        radius_limit = 0.95 * get_face_2d_radius(face, point_on_face)
        first_radius_max = radius_limit - min_step * (n_cylinders - 1)
        if first_radius_max <= min_radius:
            raise ValueError(
                f"multi_cbore radius range too small: max={radius_limit}, n={n_cylinders}"
            )

        radii = [np.random.uniform(min_radius, first_radius_max)]
        for i in range(1, n_cylinders):
            remaining = n_cylinders - i - 1
            min_next_radius = radii[-1] + min_step
            max_next_radius = min(radii[-1] * 2, radius_limit - min_step * remaining)
            if max_next_radius <= min_next_radius:
                raise ValueError(
                    f"multi_cbore radius step range too small: min={min_next_radius}, max={max_next_radius}"
                )
            radii.append(np.random.uniform(min_next_radius, max_next_radius))

        extent = normal_distance_to_next_face(workplane, face, point=point_on_face)
        min_first_depth = max(
            min_depth + min_step * (n_cylinders - 1),
            min_depth * 2 ** (n_cylinders - 1),
        )
        first_cut_through = bool(np.random.randint(2))
        if first_cut_through:
            first_depth = max(min_first_depth, extent * self.THROUGH_DEPTH_MARGIN)
        else:
            max_first_depth = self.MAX_BLIND_DEPTH_RATIO * extent
            if max_first_depth <= min_first_depth:
                raise ValueError(
                    f"multi_cbore blind depth range too small: max={max_first_depth}, n={n_cylinders}"
                )
            first_depth = np.random.uniform(min_first_depth, max_first_depth)

        depths = [first_depth]
        for i in range(1, n_cylinders):
            remaining = n_cylinders - i - 1
            min_next_depth = max(
                min_depth + min_step * remaining,
                min_depth * 2 ** remaining,
                depths[-1] / 2,
            )
            max_next_depth = depths[-1] - min_step
            if max_next_depth <= min_next_depth:
                raise ValueError(
                    f"multi_cbore depth step range too small: min={min_next_depth}, max={max_next_depth}"
                )
            depths.append(np.random.uniform(min_next_depth, max_next_depth))

        return Hole(
            point=point_on_face,
            world_size=world_size,
            hole_type="multi_cbore",
            multi_cbore_radii=radii,
            multi_cbore_depths=depths,
            workplane=WORKPLANE_BY_NORMAL[site.normal_axis],
            normal_axis=site.normal_axis,
            normal_sign=site.normal_sign,
        )

    @staticmethod
    def _assert_solid_count_unchanged(
        s: str, hole: Hole, plane: str | None = None
    ) -> None:
        exec(s, globals())
        mesh_before = compound_to_mesh(globals()["r"].val())
        exec(s + hole.to_string(plane), globals())
        mesh_after = compound_to_mesh(globals()["r"].val())
        assert len(mesh_before.split()) == len(
            mesh_after.split()
        ), "number of solids changed"

    def generate(
        self,
        s: str | None = None,
        plane: str | None = None,
        point_plane: str | None = None,
        world_size: float | None = None,
    ) -> Hole | None:
        if s is None:
            return None

        exec(s, globals())
        _w = globals()["r"]

        hole_type = np.random.choice(
            list(self.hole_type_probabilities.keys()),
            p=list(self.hole_type_probabilities.values()),
        )
        site = self._sample_site(_w)

        if hole_type in self.SKETCH_HOLE_TYPES:
            hole = self._generate_sketch_hole_on_site(
                hole_type,
                _w,
                site,
                world_size,
            )
            self._assert_solid_count_unchanged(s, hole)
            return hole

        if hole_type == "multi_cbore":
            hole = self._generate_multi_cbore_on_site(
                _w,
                site,
                world_size,
            )
            self._assert_solid_count_unchanged(s, hole)
            return hole

        face = site.face
        point_on_face = site.point.toTuple()
        extent = normal_distance_to_next_face(_w, face, point=point_on_face)

        min_extent, max_extent = get_sketch_radius_range(face, point_on_face)

        radius = None
        cbore_radius = None
        csk_radius = None
        depth = None
        cbore_depth = None
        csk_angle = None

        if hole_type == "cbore":
            min_radius = max(
                self._world_length_to_generation_length(self.MIN_RADIUS, world_size),
                0.5 * max_extent,
                min_extent,
            )
            max_radius = 0.95 * max_extent
            if max_radius <= min_radius:
                raise ValueError(
                    f"Counterbore radius range too small: min={min_radius}, max={max_radius}"
                )
            radius = np.random.uniform(
                min_radius, max_radius
            )
            cbore_radius = np.random.uniform(radius, max_radius)
            min_depth = self._world_length_to_generation_length(
                self.MIN_DEPTH, world_size
            )
            max_depth = 0.95 * extent
            if max_depth <= min_depth:
                raise ValueError(
                    f"Counterbore depth range too small: max_depth={max_depth}"
                )
            depth = np.random.uniform(min_depth, max_depth)
            cbore_depth = np.random.uniform(min_depth, depth)
        elif hole_type == "csk":
            min_radius = max(
                self._world_length_to_generation_length(self.MIN_RADIUS, world_size),
                0.5 * max_extent,
                min_extent,
            )
            max_radius = 0.95 * max_extent
            if max_radius <= min_radius:
                raise ValueError(
                    f"Countersink radius range too small: min={min_radius}, max={max_radius}"
                )
            radius = np.random.uniform(
                min_radius, max_radius
            )
            csk_radius = np.random.uniform(radius, max_radius)
            min_depth = self._world_length_to_generation_length(
                self.MIN_DEPTH, world_size
            )
            max_depth = 0.95 * extent
            if max_depth <= min_depth:
                raise ValueError(
                    f"Countersink depth range too small: max_depth={max_depth}"
                )
            depth = np.random.uniform(min_depth, max_depth)
            csk_angle = np.random.randint(30, 150)

        logger.info(
            "Generated hole with type=%s, radius=%s, cbore_radius=%s, depth=%s, cbore_depth=%s, csk_angle=%s",
            hole_type,
            radius,
            cbore_radius,
            depth,
            cbore_depth,
            csk_angle,
        )
        return Hole(
            radius,
            cbore_radius,
            csk_radius,
            depth,
            cbore_depth,
            csk_angle,
            point_on_face,
            None,
            world_size,
            hole_type,
        )

    def to_dict(self) -> dict:
        return {
            "type": "HoleFactory",
            "hole_type_probabilities": self.hole_type_probabilities,
            "sketch_factory": self.sketch_factory.to_dict()
            if self.sketch_factory is not None
            else None,
            "mesh_deflection": self.mesh_deflection,
        }

    @staticmethod
    def from_dict(entity: dict) -> "HoleFactory":
        return HoleFactory(
            hole_type_probabilities=entity["hole_type_probabilities"],
            sketch_factory=SketchFactory.from_dict(entity["sketch_factory"])
            if entity.get("sketch_factory")
            else None,
            mesh_deflection=entity.get("mesh_deflection", 0.005),
        )


factories.register("hole", HoleFactory)
