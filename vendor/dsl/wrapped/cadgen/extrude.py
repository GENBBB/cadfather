import logging
import math
from copy import deepcopy
from dataclasses import dataclass

import cadquery as cq
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

# from cadquery_addons import *

from .base import BaseFactory, BaseOperation
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
    PositiveAxisName,
    axis_name_from_vector,
    SampledSite,
    axis_vector,
    closest_point_on_face_surface,
    orthogonal_axes,
    precompute_surface_compound,
    remaining_axis,
    signed_axis,
    closest_surface_point_from_compound,
    normal_at_face_uv,
    point_is_on_face,
    shape_from_cad_object,
    workplane_at_closest_point,
)
from .utils import shape_to_area

logger = logging.getLogger(__name__)

NON_SKETCHGRAPH_CONTACT_FALLBACK_RETRIES = 5

WORKPLANE_BY_NORMAL = {
    "X": "YZ",
    "Y": "ZX",
    "Z": "XY",
}

WORKPLANE_AXES = {
    "XY": ("X", "Y", "Z"),
    "YZ": ("Y", "Z", "X"),
    "ZX": ("Z", "X", "Y"),
}

WORKPLANE_OFFSET_INDEX = {
    "XY": 2,
    "YZ": 0,
    "ZX": 1,
}


def _format_number(value: float) -> str:
    value = float(value)
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:g}"


def _format_point(point: tuple[float, float, float]) -> str:
    return "(" + ", ".join(_format_number(value) for value in point) + ")"


def _double_quoted_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _join_workplane_and_sketch(prefix: str, sketch: str) -> str:
    sketch = sketch.strip()
    if not sketch:
        raise ValueError("sketch must not be empty")
    if sketch.startswith("."):
        return prefix + sketch
    return prefix + "." + sketch


def _world_size_from_shape(shape: cq.Shape) -> float:
    bbox = shape.BoundingBox()
    return max(
        bbox.xmax - bbox.xmin,
        bbox.ymax - bbox.ymin,
        bbox.zmax - bbox.zmin,
    )


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


def _surface_translation(
    shape: cq.Shape,
    surface_point: cq.Vector,
    world_size: float,
    inward: bool = True,
) -> tuple[float, float, float]:
    normal = _surface_normal_at_point(shape, surface_point)
    axis_index = int(np.argmax(np.abs(normal)))
    if inward:
        dir_to_solid = -float(normal[axis_index])
    else:
        dir_to_solid = float(normal[axis_index])
    distance = 0.0000025 * float(world_size)
    translation = [0.0, 0.0, 0.0]
    translation[axis_index] = math.copysign(distance, dir_to_solid)
    return tuple(translation)


def extrude(
    r: cq.Workplane | cq.Shape | None,
    point: tuple[float, float, float],
    workplane: str,
    sketch: str,
    extrude_height: float,
    point_on_surface: bool = True,
    world_size: float | None = 200,
):
    if workplane not in WORKPLANE_OFFSET_INDEX:
        raise ValueError("workplane must be one of 'XY', 'YZ', or 'ZX'")

    point = tuple(float(value) for value in point)
    if r is None or not point_on_surface:
        axis_point = point
        offset = axis_point[WORKPLANE_OFFSET_INDEX[workplane]]
        shape = None
        surface_point = None
    else:
        shape = shape_from_cad_object(r)
        # if world_size is None:
        #     world_size = _world_size_from_shape(shape)

        surface_compound = precompute_surface_compound(shape)
        surface_point = closest_surface_point_from_compound(surface_compound, point)
        axis_point = tuple(float(value) for value in surface_point.toTuple())
        offset = axis_point[WORKPLANE_OFFSET_INDEX[workplane]]

    prefix = f"cq.Workplane('{workplane}').workplane(offset={offset})"
    expression = _join_workplane_and_sketch(
        prefix,
        sketch,
    ) + f".extrude({extrude_height})"
    body = eval(expression, {"cq": cq})
    if shape is not None and surface_point is not None:
        body = body.translate(
            _surface_translation(shape, surface_point, world_size)
        )

    if r is None:
        return cq.Workplane(workplane).add(body)
    if isinstance(r, cq.Shape):
        return cq.Workplane(workplane).add(r).union(body)
    return r.union(body)


@dataclass(frozen=True)
class ExtrudePlacement:
    site: SampledSite
    workplane: AxisWorkplane
    tangent_to_surface: bool


@dataclass(frozen=True)
class WorkplaneBounds:
    x_neg: float
    x_pos: float
    y_neg: float
    y_pos: float

    @property
    def symmetric_width(self) -> float:
        return 2.0 * min(self.x_neg, self.x_pos)

    @property
    def symmetric_height(self) -> float:
        return 2.0 * min(self.y_neg, self.y_pos)


class Extrude(BaseOperation):
    def __init__(
        self,
        sketch: Sketch,
        extent: float,
        point_on_surface: bool = True,
    ):
        self.sketch = sketch
        self.extent = extent
        self.point_on_surface = point_on_surface

    def to_string(
        self,
        plane: str | AxisWorkplane,
        world_size: float | None = None,
    ) -> str:
        logger.info("Building extrude string with extent=%f", self.extent)
        if isinstance(plane, AxisWorkplane):
            return self._to_factory_call_string(plane, world_size)
        expr = self.sketch.to_string(plane)
        expr += f".extrude({self.extent})\n"
        return expr

    def to_global_workplane_string(
        self,
        workplane_axis: str,
        origin: tuple[float, float, float],
    ) -> str:
        local_sketch = self.sketch.to_string(f"cq.Workplane('{workplane_axis}')")
        global_sketch = sketch_to_global_coords(
            workplane_axis,
            origin,
            local_sketch,
        )
        sketch_arg = sketch_chain_only(global_sketch)
        return self._factory_call_string(
            tuple(float(value) for value in origin),
            workplane_axis,
            sketch_arg,
            self.extent,
            point_on_surface=self.point_on_surface,
            world_size=None,
        )

    def _to_factory_call_string(
        self,
        workplane: AxisWorkplane,
        world_size: float | None,
    ) -> str:
        normal_name = workplane.normal[-1]
        normal_sign = -1.0 if workplane.normal.startswith("-") else 1.0
        workplane_axis = WORKPLANE_BY_NORMAL[normal_name]

        normal_vec = axis_vector(workplane.normal)
        x_dir_vec = axis_vector(workplane.xDir)
        y_dir_vec = normal_vec.cross(x_dir_vec)
        y_axis = axis_name_from_vector(y_dir_vec)
        if y_axis is None:
            raise ValueError("Could not resolve workplane yDir to a global axis")
        source_y_dir = signed_axis(*y_axis)

        local_sketch = self.sketch.to_string(f"cq.Workplane('{workplane_axis}')")
        logger.debug(
            "extrude sketch before axis/global transform: %s",
            sketch_chain_only(local_sketch),
        )
        aligned_sketch = transform_sketch_local_axes(
            local_sketch,
            workplane.xDir,
            source_y_dir,
            workplane_axis,
        )
        global_sketch = sketch_to_global_coords(
            workplane_axis,
            workplane.point,
            aligned_sketch,
        )
        sketch_arg = sketch_chain_only(global_sketch)
        logger.debug("extrude sketch after axis/global transform: %s", sketch_arg)

        extent = self.extent * normal_sign
        return self._factory_call_string(
            tuple(float(value) for value in workplane.point),
            workplane_axis,
            sketch_arg,
            extent,
            point_on_surface=self.point_on_surface,
            world_size=world_size,
        )

    def _factory_call_string(
        self,
        point: tuple[float, float, float],
        workplane_axis: str,
        sketch_arg: str,
        extent: float,
        point_on_surface: bool = True,
        world_size: float | None = None,
    ) -> str:
        point_expr = _format_point(point)
        args = (
            f"r, {point_expr}, {workplane_axis!r}, "
            f"{_double_quoted_string(sketch_arg)}, {_format_number(extent)}"
        )
        if not point_on_surface:
            args += ", False"
        elif world_size is not None:
            args += ", True"
        if world_size is not None:
            args += f", {_format_number(world_size)}"
        return f"extrude({args})\n"

    def transform(self, shift: list[float], scale: float) -> None:
        self.extent *= scale

    def round(self) -> None:
        self.sketch.round()
        self.extent = round(self.extent)

    def fix(self) -> None:
        self.sketch.fix()

    def to_dict(self) -> dict:
        return {
            "type": "Extrude",
            "sketch": self.sketch.to_dict(),
            "extent": self.extent,
            "point_on_surface": self.point_on_surface,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Extrude":
        assert (
            entity["type"] == "Extrude"
        ), f"Trying to build Extrude from type {entity['type']}"
        return Extrude(
            Sketch.from_dict(entity["sketch"]),
            entity["extent"],
            entity.get("point_on_surface", True),
        )


class TwoExtrudes(BaseOperation):
    def __init__(
        self,
        first: Extrude,
        second: Extrude,
        second_center: tuple[float, float],
        second_normal_offset: float,
    ):
        self.first = first
        self.second = second
        self.second_center = tuple(float(value) for value in second_center)
        self.second_normal_offset = float(second_normal_offset)

    @property
    def sketch(self) -> Sketch:
        return self.first.sketch

    @property
    def extent(self) -> float:
        return self.first.extent

    def to_string(
        self,
        plane: str | AxisWorkplane,
        world_size: float | None = None,
    ) -> str:
        if isinstance(plane, AxisWorkplane):
            return self._to_factory_call_string(plane, world_size)
        raise ValueError("TwoExtrudes operations require a sampled or indexed workplane")

    def to_global_workplane_string(
        self,
        workplane_axis: str,
        origin: tuple[float, float, float],
    ) -> str:
        first = self.first.to_global_workplane_string(workplane_axis, origin)
        second_point = self._second_point_for_indexed_workplane(
            workplane_axis,
            origin,
        )
        second_sketch = self._global_sketch_arg(
            self.second,
            workplane_axis,
            second_point,
        )
        second = self.second._factory_call_string(
            second_point,
            workplane_axis,
            second_sketch,
            self.second.extent,
            point_on_surface=False,
            world_size=None,
        )
        return first + "r=" + second

    def _to_factory_call_string(
        self,
        workplane: AxisWorkplane,
        world_size: float | None,
    ) -> str:
        first = self.first.to_string(workplane, world_size)
        second_workplane = self._second_axis_workplane(workplane)
        second = self.second.to_string(second_workplane, world_size)
        return first + "r=" + second

    @staticmethod
    def _global_sketch_arg(
        extrude_op: Extrude,
        workplane_axis: str,
        point: tuple[float, float, float],
    ) -> str:
        local_sketch = extrude_op.sketch.to_string(f"cq.Workplane('{workplane_axis}')")
        global_sketch = sketch_to_global_coords(
            workplane_axis,
            point,
            local_sketch,
        )
        return sketch_chain_only(global_sketch)

    def _second_point_for_indexed_workplane(
        self,
        workplane_axis: str,
        origin: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        delta = ExtrudeFactory._factory_axes_delta_to_world(
            workplane_axis,
            self.second_center[0],
            self.second_center[1],
            self.second_normal_offset,
        )
        return tuple(float(value + shift) for value, shift in zip(origin, delta))

    def _second_axis_workplane(self, workplane: AxisWorkplane) -> AxisWorkplane:
        normal = axis_vector(workplane.normal)
        x_dir = axis_vector(workplane.xDir)
        y_dir = normal.cross(x_dir)
        point = cq.Vector(*workplane.point)
        point = point.add(x_dir.multiply(self.second_center[0]))
        point = point.add(y_dir.multiply(self.second_center[1]))
        point = point.add(normal.multiply(self.second_normal_offset))
        return AxisWorkplane(
            point=point.toTuple(),
            normal=workplane.normal,
            xDir=workplane.xDir,
        )

    def transform(self, shift: list[float], scale: float) -> None:
        self.first.extent *= scale
        self.second.extent *= scale
        self.first.sketch.transform(shift, scale)
        self.second.sketch.transform([0.0, 0.0], scale)
        self.second_center = (
            (self.second_center[0] + shift[0]) * scale,
            (self.second_center[1] + shift[1]) * scale,
        )
        self.second_normal_offset *= scale

    def round(self) -> None:
        self.first.round()
        self.second.round()
        self.second_center = (
            round(self.second_center[0]),
            round(self.second_center[1]),
        )
        self.second_normal_offset = round(self.second_normal_offset)

    def fix(self) -> None:
        self.first.fix()
        self.second.fix()

    def to_dict(self) -> dict:
        return {
            "type": "TwoExtrudes",
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "second_center": self.second_center,
            "second_normal_offset": self.second_normal_offset,
        }

    @staticmethod
    def from_dict(entity: dict) -> "TwoExtrudes":
        assert (
            entity["type"] == "TwoExtrudes"
        ), f"Trying to build TwoExtrudes from type {entity['type']}"
        return TwoExtrudes(
            Extrude.from_dict(entity["first"]),
            Extrude.from_dict(entity["second"]),
            tuple(entity["second_center"]),
            entity["second_normal_offset"],
        )


class ExtrudeFactory(BaseFactory):
    def __init__(
        self,
        sketch_factory: SketchFactory | dict,
        tangent_plane_prob: float = 0.5,
        out_of_face_prob: float = 0.5,
        point_out_of_surface_probability: float = 0.0,
        mesh_deflection: float = 0.005,
    ):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)
        self.sketch_factory = sketch_factory
        self.r_min, self.r_max = 0.01, 1.0
        self.tangent_plane_prob = tangent_plane_prob
        self.out_of_face_prob = out_of_face_prob
        self.point_out_of_surface_probability = float(
            np.clip(point_out_of_surface_probability, 0.0, 1.0)
        )
        self.mesh_deflection = mesh_deflection

    def generate(self) -> Extrude:
        sketch = self.sketch_factory.generate()
        sign = float(np.random.choice([-1, 1]))
        extent = sign * np.random.uniform(self.r_min, self.r_max)

        logger.info("Generated extrude with extent=%f", extent)
        return Extrude(sketch, extent)

    def prepare_existing_sampler(
        self,
        cad_object: cq.Workplane | cq.Shape,
    ) -> PreparedSurfaceSampler:
        return PreparedSurfaceSampler.from_cad_object(
            cad_object,
            mesh_deflection=self.mesh_deflection,
        )

    def generate_on_existing(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        world_size: float,
        generation_world_half: float = 1.0,
    ) -> tuple[Extrude, AxisWorkplane]:
        placement = self._sample_placement(sampler)
        workplane = workplane_at_closest_point(
            cad_object,
            placement.site.point,
            placement.workplane.normal,
            placement.workplane.xDir,
            surface_compound=sampler.surface_compound,
        )

        sketch = self.sketch_factory.generate()
        min_dim = self._min_generation_dimension(world_size)
        world_bounds = self._workplane_world_bounds(workplane, generation_world_half)
        if not placement.tangent_to_surface and placement.site.face_kind == "plane":
            target_width, target_depth, depth_sign = self._sample_planar_contact_box(
                world_bounds,
                min_dim,
            )
            self._place_sketch_on_planar_contact(
                sketch,
                target_width,
                target_depth,
                depth_sign,
                min_dim,
            )
        elif (
            not placement.tangent_to_surface
            and placement.site.face_kind == "cylinder"
            and placement.site.cylinder_radius is not None
        ):
            target_width, target_depth = self._sample_cylinder_contact_box(
                world_bounds,
                placement.site.cylinder_radius,
                min_dim,
            )
            self._place_sketch_on_cylinder_contact(
                sketch,
                target_width,
                target_depth,
                placement.site.cylinder_radius,
                min_dim,
            )
        else:
            target_width, target_height = self._sample_sketch_dimensions(
                sketch_workplane=workplane,
                site=placement.site,
                tangent_to_surface=placement.tangent_to_surface,
                min_dim=min_dim,
                world_bounds=world_bounds,
            )
            self._fit_sketch_to_box(sketch, target_width, target_height, min_dim)
        self._assert_sketch_inside_world_box(sketch, world_bounds)

        extent = self._sample_extent(
            workplane=workplane,
            placement=placement,
            world_size=world_size,
            generation_world_half=generation_world_half,
        )
        logger.info(
            "Generated placed extrude with extent=%f, normal=%s, xDir=%s",
            extent,
            placement.workplane.normal,
            placement.workplane.xDir,
        )
        factory_workplane = placement.workplane
        point_on_surface = True
        if np.random.random() < self.point_out_of_surface_probability:
            try:
                factory_workplane = self._move_point_out_of_surface(
                    sketch=sketch,
                    extent=extent,
                    placement=placement,
                    workplane=placement.workplane,
                    generation_world_half=generation_world_half,
                    min_dim=min_dim,
                )
                point_on_surface = False
            except Exception as exc:
                logger.debug(
                    "Could not move extrude point out of surface; keeping surface point",
                    exc_info=True,
                )
        return Extrude(sketch, extent, point_on_surface), factory_workplane

    def _move_point_out_of_surface(
        self,
        sketch: Sketch,
        extent: float,
        placement: ExtrudePlacement,
        workplane: AxisWorkplane,
        generation_world_half: float,
        min_dim: float,
    ) -> AxisWorkplane:
        workplane_axis = self._factory_workplane_axis(workplane)
        lateral_axis, lateral_delta = self._sample_point_out_lateral_delta(
            sketch=sketch,
            placement=placement,
            workplane=workplane,
            workplane_axis=workplane_axis,
            generation_world_half=generation_world_half,
            min_dim=min_dim,
        )
        delta_u = lateral_delta if lateral_axis == "u" else 0.0
        delta_v = lateral_delta if lateral_axis == "v" else 0.0
        delta_normal = self._sample_point_out_normal_delta(
            point=workplane.point,
            extent=extent,
            workplane_axis=workplane_axis,
            generation_world_half=generation_world_half,
            min_dim=min_dim,
        )
        delta_world = self._factory_axes_delta_to_world(
            workplane_axis,
            delta_u,
            delta_v,
            delta_normal,
        )
        point = tuple(
            float(value + delta)
            for value, delta in zip(workplane.point, delta_world)
        )
        logger.debug(
            "Moved extrude point out of surface: workplane=%s, lateral_axis=%s, "
            "lateral_delta=%s, normal_delta=%s, point=%s",
            workplane_axis,
            lateral_axis,
            lateral_delta,
            delta_normal,
            point,
        )
        return AxisWorkplane(point=point, normal=workplane.normal, xDir=workplane.xDir)

    def _sample_point_out_lateral_delta(
        self,
        sketch: Sketch,
        placement: ExtrudePlacement,
        workplane: AxisWorkplane,
        workplane_axis: str,
        generation_world_half: float,
        min_dim: float,
    ) -> tuple[str, float]:
        u_distances, v_distances = self._face_axis_distances(
            placement.site.face,
            placement.site.point.toTuple(),
            workplane_axis,
        )
        candidates = [
            ("u", -1.0, u_distances[0]),
            ("u", 1.0, u_distances[1]),
            ("v", -1.0, v_distances[0]),
            ("v", 1.0, v_distances[1]),
        ]
        candidates = [
            (axis, sign, float(distance))
            for axis, sign, distance in candidates
            if float(distance) > 1e-8
        ]
        if not candidates:
            raise ValueError("Selected face has no lateral boundary distance")

        u_limits, v_limits = self._shift_limits_for_sketch(
            sketch,
            workplane,
            workplane_axis,
            generation_world_half,
        )
        epsilon = max(0.25 * min_dim, 1e-6)
        valid_candidates = []
        for axis, sign, face_distance in candidates:
            lower, upper = u_limits if axis == "u" else v_limits
            max_lateral = upper if sign > 0 else -lower
            if max_lateral > face_distance + epsilon:
                valid_candidates.append((axis, sign, face_distance, max_lateral))
        if not valid_candidates:
            raise ValueError("No room to move extrude point outside selected face")

        axis, sign, face_distance, max_lateral = min(
            valid_candidates,
            key=lambda item: item[2],
        )
        sketch_opposite = self._sketch_span_opposite_direction(
            sketch,
            workplane,
            workplane_axis,
            axis,
            sign,
        )
        low = face_distance + epsilon
        high = min(max_lateral, face_distance + max(sketch_opposite, epsilon))
        if high <= low:
            distance = low
        else:
            distance = float(np.random.uniform(low, high))
        if np.random.random() < 0.2:
            distance = min(max_lateral, distance * float(np.random.uniform(1.0, 2.0)))
        return axis, sign * distance

    @staticmethod
    def _sample_point_out_normal_delta(
        point: tuple[float, float, float],
        extent: float,
        workplane_axis: str,
        generation_world_half: float,
        min_dim: float,
    ) -> float:
        if np.random.random() >= 0.2:
            return 0.0

        lower, upper = ExtrudeFactory._normal_shift_limits(
            point,
            extent,
            workplane_axis,
            generation_world_half,
        )
        epsilon = max(0.25 * min_dim, 1e-6)
        cap = max(epsilon, min(generation_world_half * 0.25, abs(extent) * 0.5))
        candidates = []
        if upper > epsilon:
            candidates.append((1.0, min(upper, cap)))
        if -lower > epsilon:
            candidates.append((-1.0, min(-lower, cap)))
        if not candidates:
            return 0.0
        sign, high = candidates[int(np.random.randint(0, len(candidates)))]
        if high <= epsilon:
            return 0.0
        return sign * float(np.random.uniform(epsilon, high))

    @staticmethod
    def _factory_workplane_axis(workplane: AxisWorkplane) -> str:
        return WORKPLANE_BY_NORMAL[workplane.normal[-1]]

    @staticmethod
    def _factory_axis_vectors(
        workplane_axis: str,
    ) -> tuple[cq.Vector, cq.Vector, cq.Vector]:
        try:
            u_axis, v_axis, normal_axis = WORKPLANE_AXES[workplane_axis]
        except KeyError as exc:
            raise ValueError(f"unknown workplane axis {workplane_axis!r}") from exc
        return axis_vector(u_axis), axis_vector(v_axis), axis_vector(normal_axis)

    @staticmethod
    def _standard_factory_workplane(
        point: tuple[float, float, float],
        workplane_axis: str,
    ) -> cq.Workplane:
        u_axis, _, normal_axis = WORKPLANE_AXES[workplane_axis]
        plane = cq.Plane(
            origin=cq.Vector(*point),
            xDir=axis_vector(u_axis),
            normal=axis_vector(normal_axis),
        )
        return cq.Workplane(plane)

    @staticmethod
    def _axis_distances_from_local_face(
        local_face: cq.Face,
        axis: str,
    ) -> tuple[float, float]:
        bbox = local_face.BoundingBox()
        span = max(
            abs(float(bbox.xmin)),
            abs(float(bbox.xmax)),
            abs(float(bbox.ymin)),
            abs(float(bbox.ymax)),
            abs(float(bbox.zmin)),
            abs(float(bbox.zmax)),
            1.0,
        )
        direction = {
            "x": cq.Vector(1, 0, 0),
            "y": cq.Vector(0, 1, 0),
        }[axis]
        probe = cq.Edge.makeLine(
            direction.multiply(-2.0 * span),
            direction.multiply(2.0 * span),
        )
        intersection = local_face.intersect(probe)
        candidates: list[tuple[float, float]] = []
        for edge in intersection.Edges():
            start = edge.startPoint()
            end = edge.endPoint()
            t0 = start.x if axis == "x" else start.y
            t1 = end.x if axis == "x" else end.y
            lo, hi = sorted((float(t0), float(t1)))
            if lo <= 1e-7 and hi >= -1e-7 and hi - lo > 1e-7:
                candidates.append((-lo, hi))

        if candidates:
            return max(candidates, key=lambda item: item[0] + item[1])

        if axis == "x":
            return max(0.0, -float(bbox.xmin)), max(0.0, float(bbox.xmax))
        return max(0.0, -float(bbox.ymin)), max(0.0, float(bbox.ymax))

    @staticmethod
    def _face_axis_distances(
        face: cq.Face,
        point: tuple[float, float, float],
        workplane_axis: str,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        standard_workplane = ExtrudeFactory._standard_factory_workplane(
            point,
            workplane_axis,
        )
        local_face = face.transformShape(standard_workplane.plane.fG)
        return (
            ExtrudeFactory._axis_distances_from_local_face(local_face, "x"),
            ExtrudeFactory._axis_distances_from_local_face(local_face, "y"),
        )

    @staticmethod
    def _local_sketch_point_to_factory_axes(
        point: np.ndarray,
        workplane: AxisWorkplane,
        workplane_axis: str,
    ) -> tuple[float, float]:
        source_x = axis_vector(workplane.xDir)
        source_normal = axis_vector(workplane.normal)
        source_y = source_normal.cross(source_x)
        target_u, target_v, _ = ExtrudeFactory._factory_axis_vectors(workplane_axis)
        delta = source_x.multiply(float(point[0])).add(
            source_y.multiply(float(point[1]))
        )
        return float(delta.dot(target_u)), float(delta.dot(target_v))

    @staticmethod
    def _sketch_bbox_in_factory_axes(
        sketch: Sketch,
        workplane: AxisWorkplane,
        workplane_axis: str,
    ) -> tuple[float, float, float, float]:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        corners = [
            np.array([bbox.xmin, bbox.ymin], dtype=float),
            np.array([bbox.xmin, bbox.ymax], dtype=float),
            np.array([bbox.xmax, bbox.ymin], dtype=float),
            np.array([bbox.xmax, bbox.ymax], dtype=float),
        ]
        points = [
            ExtrudeFactory._local_sketch_point_to_factory_axes(
                corner,
                workplane,
                workplane_axis,
            )
            for corner in corners
        ]
        u_values = [point[0] for point in points]
        v_values = [point[1] for point in points]
        return min(u_values), max(u_values), min(v_values), max(v_values)

    @staticmethod
    def _shift_limits_for_sketch(
        sketch: Sketch,
        workplane: AxisWorkplane,
        workplane_axis: str,
        generation_world_half: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        target_u, target_v, _ = ExtrudeFactory._factory_axis_vectors(workplane_axis)
        point_vec = cq.Vector(*workplane.point)
        point_u = float(point_vec.dot(target_u))
        point_v = float(point_vec.dot(target_v))
        bbox_u_min, bbox_u_max, bbox_v_min, bbox_v_max = (
            ExtrudeFactory._sketch_bbox_in_factory_axes(
                sketch,
                workplane,
                workplane_axis,
            )
        )
        bbox_u_min = min(0.0, bbox_u_min)
        bbox_u_max = max(0.0, bbox_u_max)
        bbox_v_min = min(0.0, bbox_v_min)
        bbox_v_max = max(0.0, bbox_v_max)
        half = float(generation_world_half)
        u_limits = (
            -half - point_u - bbox_u_min,
            half - point_u - bbox_u_max,
        )
        v_limits = (
            -half - point_v - bbox_v_min,
            half - point_v - bbox_v_max,
        )
        return u_limits, v_limits

    @staticmethod
    def _explicit_sketch_points_for_point_out(sketch: Sketch) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        for wire in sketch.wires:
            if wire["type"] == "circle":
                points.append(np.array(wire["center"], dtype=float))
                continue

            rectangle = Sketch.wire_to_rectangle(wire)
            if rectangle is not None:
                points.append(np.array([rectangle[0], rectangle[1]], dtype=float))
                continue

            if wire["type"] != "polygon":
                continue
            points.extend(np.array(vertex, dtype=float) for vertex in wire["vertices"])
            for edge in wire["edges"]:
                if edge["type"] == "arc":
                    points.append(np.array(edge["point"], dtype=float))

        if sketch.point is not None:
            points.append(np.array(sketch.point, dtype=float))
        if not points:
            points.append(np.array([0.0, 0.0], dtype=float))
        return points

    @staticmethod
    def _sketch_span_opposite_direction(
        sketch: Sketch,
        workplane: AxisWorkplane,
        workplane_axis: str,
        lateral_axis: str,
        direction_sign: float,
    ) -> float:
        axis_index = 0 if lateral_axis == "u" else 1
        spans = []
        for point in ExtrudeFactory._explicit_sketch_points_for_point_out(sketch):
            projected = ExtrudeFactory._local_sketch_point_to_factory_axes(
                point,
                workplane,
                workplane_axis,
            )
            spans.append(max(0.0, -direction_sign * projected[axis_index]))
        return max(spans, default=0.0)

    @staticmethod
    def _normal_shift_limits(
        point: tuple[float, float, float],
        extent: float,
        workplane_axis: str,
        generation_world_half: float,
    ) -> tuple[float, float]:
        _, _, normal = ExtrudeFactory._factory_axis_vectors(workplane_axis)
        point_normal = float(cq.Vector(*point).dot(normal))
        body_min = min(0.0, float(extent))
        body_max = max(0.0, float(extent))
        half = float(generation_world_half)
        return (
            -half - point_normal - body_min,
            half - point_normal - body_max,
        )

    @staticmethod
    def _factory_axes_delta_to_world(
        workplane_axis: str,
        delta_u: float,
        delta_v: float,
        delta_normal: float,
    ) -> tuple[float, float, float]:
        u_axis, v_axis, normal_axis = ExtrudeFactory._factory_axis_vectors(
            workplane_axis
        )
        delta = (
            u_axis.multiply(float(delta_u))
            .add(v_axis.multiply(float(delta_v)))
            .add(normal_axis.multiply(float(delta_normal)))
        )
        return delta.toTuple()

    def _sample_placement(self, sampler: PreparedSurfaceSampler) -> ExtrudePlacement:
        site = sampler.sample_site()
        tangent_to_surface = bool(np.random.random() < self.tangent_plane_prob)

        if tangent_to_surface:
            normal = site.normal_axis
            x_dir = self._standard_workplane_x_dir(normal)
        else:
            if (
                site.face_kind == "cylinder"
                and site.cylinder_axis is not None
                and site.cylinder_axis != site.normal_axis
            ):
                normal = site.cylinder_axis
                x_dir = signed_axis(site.normal_axis, site.normal_sign)
            else:
                normal = np.random.choice(orthogonal_axes(site.normal_axis)).item()
                x_dir = remaining_axis(site.normal_axis, normal)

        workplane = AxisWorkplane(
            point=site.point.toTuple(),
            normal=normal,
            xDir=x_dir,
        )
        return ExtrudePlacement(site, workplane, tangent_to_surface)

    def _sample_sketch_dimensions(
        self,
        sketch_workplane: cq.Workplane,
        site: SampledSite,
        tangent_to_surface: bool,
        min_dim: float,
        world_bounds: WorkplaneBounds,
    ) -> tuple[float, float]:
        world_width = world_bounds.symmetric_width
        world_height = world_bounds.symmetric_height
        if world_width < min_dim or world_height < min_dim:
            raise ValueError("Workplane has too little room inside world bounds")

        face_width = face_height = None
        if site.face_kind == "plane" and tangent_to_surface:
            face_width, face_height = self._planar_face_box(site.face, sketch_workplane)

        out_of_face = (
            site.face_kind != "plane"
            or not tangent_to_surface
            or np.random.random() < self.out_of_face_prob
        )
        if not out_of_face and face_width is not None and face_height is not None:
            max_width = min(face_width, world_width)
            max_height = min(face_height, world_height)
            return (
                self._sample_inside_size(max_width, min_dim),
                self._sample_inside_size(max_height, min_dim),
            )

        boundary_width = face_width if face_width is not None else min_dim
        boundary_height = face_height if face_height is not None else min_dim
        return (
            self._sample_out_of_face_size(boundary_width, world_width, min_dim),
            self._sample_out_of_face_size(boundary_height, world_height, min_dim),
        )

    def _sample_planar_contact_box(
        self,
        bounds: WorkplaneBounds,
        min_dim: float,
    ) -> tuple[float, float, float]:
        width_limit = bounds.symmetric_width
        depth_pos = bounds.y_pos
        depth_neg = bounds.y_neg
        if width_limit < min_dim or max(depth_pos, depth_neg) < min_dim:
            raise ValueError("Workplane has too little room for contact sketch")

        if depth_neg > depth_pos:
            depth_limit = depth_neg
            depth_sign = -1.0
        else:
            depth_limit = depth_pos
            depth_sign = 1.0

        return (
            self._sample_out_of_face_size(min_dim, width_limit, min_dim),
            self._sample_out_of_face_size(min_dim, depth_limit, min_dim),
            depth_sign,
        )

    def _sample_cylinder_contact_box(
        self,
        bounds: WorkplaneBounds,
        cylinder_radius: float,
        min_dim: float,
    ) -> tuple[float, float]:
        radius = max(abs(cylinder_radius), 1e-6)
        inward_overlap = min(min_dim, 0.1 * radius)
        y_half_limit = min(bounds.y_neg, bounds.y_pos)
        if y_half_limit < 0.5 * min_dim or bounds.x_pos < min_dim:
            raise ValueError("Workplane has too little room for cylindrical contact sketch")

        theta_from_y = math.asin(min(1.0, y_half_limit / radius))
        inward_room = max(0.0, bounds.x_neg - inward_overlap)
        theta_from_x_neg = math.acos(max(-1.0, min(1.0, 1.0 - inward_room / radius)))
        theta_limit = min(1.0, theta_from_y, theta_from_x_neg)
        width_limit = 2.0 * radius * theta_limit
        min_sagitta = min(0.25 * radius, 2.0 * min_dim)
        theta_min = math.acos(max(-1.0, min(1.0, 1.0 - min_sagitta / radius)))
        width_min = max(min_dim, 2.0 * radius * theta_min)
        if width_limit < width_min:
            raise ValueError("Cylindrical contact arc is smaller than minimum sketch size")

        return (
            max(
                width_min,
                self._sample_out_of_face_size(width_min, width_limit, min_dim),
            ),
            self._sample_out_of_face_size(min_dim, bounds.x_pos, min_dim),
        )

    def _sample_extent(
        self,
        workplane: cq.Workplane,
        placement: ExtrudePlacement,
        world_size: float,
        generation_world_half: float,
    ) -> float:
        min_extent = self._min_generation_dimension(world_size)
        normal_vec = axis_vector(placement.workplane.normal)
        if placement.tangent_to_surface:
            sign = placement.site.normal_sign
        else:
            sign = float(np.random.choice([-1.0, 1.0]))

        distance = self._distance_to_world_border(
            workplane.plane.origin,
            normal_vec.multiply(sign),
            generation_world_half,
        )
        if distance < min_extent:
            raise ValueError("Too little room for extrude extent")

        if np.random.random() < 0.6:
            high = max(min_extent, 0.5 * distance)
            height = np.random.uniform(min_extent, high)
        else:
            mean = 0.5 * distance
            sigma = max((distance - mean) / 2.0, 1e-12)
            height = float(np.random.normal(mean, sigma))
            height = float(np.clip(height, min_extent, distance))
        return sign * height

    @staticmethod
    def _standard_workplane_x_dir(normal: PositiveAxisName) -> PositiveAxisName:
        return {"X": "Y", "Y": "Z", "Z": "X"}[normal]  # type: ignore[return-value]

    @staticmethod
    def _min_generation_dimension(world_size: float) -> float:
        return 2.0 / max(float(world_size), 1.0)

    @staticmethod
    def _sample_inside_size(max_size: float, min_dim: float) -> float:
        if max_size < min_dim:
            raise ValueError("Face boundary is smaller than the minimum sketch size")
        mean = 0.5 * max_size
        sigma = max((max_size - mean) / 2.0, min_dim)
        return float(np.clip(np.random.normal(mean, sigma), min_dim, max_size))

    @staticmethod
    def _sample_out_of_face_size(
        boundary_size: float,
        world_size: float,
        min_dim: float,
    ) -> float:
        if world_size < min_dim:
            raise ValueError("World boundary is smaller than the minimum sketch size")
        lower_mean = min(max(boundary_size, min_dim), world_size)
        mean = float(np.random.uniform(lower_mean, world_size))
        sigma = max((world_size - mean) / 3.0, min_dim)
        return float(np.clip(np.random.normal(mean, sigma), min_dim, world_size))

    @staticmethod
    def _workplane_world_bounds(
        workplane: cq.Workplane,
        generation_world_half: float,
    ) -> WorkplaneBounds:
        return WorkplaneBounds(
            x_neg=ExtrudeFactory._distance_to_world_border(
                workplane.plane.origin,
                workplane.plane.xDir.multiply(-1),
                generation_world_half,
            ),
            x_pos=ExtrudeFactory._distance_to_world_border(
                workplane.plane.origin,
                workplane.plane.xDir,
                generation_world_half,
            ),
            y_neg=ExtrudeFactory._distance_to_world_border(
                workplane.plane.origin,
                workplane.plane.yDir.multiply(-1),
                generation_world_half,
            ),
            y_pos=ExtrudeFactory._distance_to_world_border(
                workplane.plane.origin,
                workplane.plane.yDir,
                generation_world_half,
            ),
        )

    @staticmethod
    def _distance_to_world_border(
        origin: cq.Vector,
        direction: cq.Vector,
        generation_world_half: float,
    ) -> float:
        direction = direction.normalized()
        origin_values = origin.toTuple()
        direction_values = direction.toTuple()
        distances = []
        for origin_value, direction_value in zip(origin_values, direction_values):
            if abs(direction_value) <= 1e-12:
                continue
            if direction_value > 0:
                distance = (generation_world_half - origin_value) / direction_value
            else:
                distance = (-generation_world_half - origin_value) / direction_value
            if distance >= 0:
                distances.append(distance)
        if not distances:
            return float("inf")
        return float(min(distances))

    @staticmethod
    def _planar_face_box(
        face: cq.Face,
        workplane: cq.Workplane,
    ) -> tuple[float, float]:
        local_face = face.transformShape(workplane.plane.fG)
        bbox = local_face.BoundingBox()
        width = 2.0 * min(abs(bbox.xmin), abs(bbox.xmax))
        height = 2.0 * min(abs(bbox.ymin), abs(bbox.ymax))
        if width <= 0 or height <= 0:
            raise ValueError("Planar face projection has no usable sketch area")
        return width, height

    @staticmethod
    def _fit_sketch_to_box(
        sketch: Sketch,
        target_width: float,
        target_height: float,
        min_dim: float,
    ) -> None:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin
        if width <= 0 or height <= 0:
            raise ValueError("Generated sketch has an empty bounding box")
        scale = min(target_width / width, target_height / height)
        if scale <= 0:
            raise ValueError("Invalid sketch scale")
        sketch.transform(
            [-(bbox.xmin + bbox.xmax) / 2.0, -(bbox.ymin + bbox.ymax) / 2.0],
            scale,
        )
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        if bbox.xmax - bbox.xmin < min_dim or bbox.ymax - bbox.ymin < min_dim:
            raise ValueError("Generated sketch is smaller than minimum dimensions")

    def _place_sketch_on_planar_contact(
        self,
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        depth_sign: float,
        min_dim: float,
    ) -> None:
        if self._try_place_sketch_on_planar_contact(
            sketch,
            target_width,
            target_depth,
            depth_sign,
            min_dim,
        ):
            return
        if self._replace_with_non_sketchgraph_contact_sketch(
            sketch,
            lambda candidate: self._try_place_sketch_on_planar_contact(
                candidate,
                target_width,
                target_depth,
                depth_sign,
                min_dim,
            ),
        ):
            return
        raise ValueError("Could not place a non-sketchgraph planar contact sketch")

    @staticmethod
    def _try_place_sketch_on_planar_contact(
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        depth_sign: float,
        min_dim: float,
    ) -> bool:
        if ExtrudeFactory._remap_polygon_sketch_to_contact(
            sketch,
            target_width,
            target_depth,
            contact_kind="line",
            depth_sign=depth_sign,
            min_dim=min_dim,
        ):
            return True
        if ExtrudeFactory._splice_sketch_to_planar_contact(
            sketch,
            target_width,
            target_depth,
            depth_sign,
            min_dim=min_dim,
        ):
            return True
        return False

    def _place_sketch_on_cylinder_contact(
        self,
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        cylinder_radius: float,
        min_dim: float,
    ) -> None:
        if self._try_place_sketch_on_cylinder_contact(
            sketch,
            target_width,
            target_depth,
            cylinder_radius,
            min_dim,
        ):
            return
        if self._replace_with_non_sketchgraph_contact_sketch(
            sketch,
            lambda candidate: self._try_place_sketch_on_cylinder_contact(
                candidate,
                target_width,
                target_depth,
                cylinder_radius,
                min_dim,
            ),
        ):
            return
        raise ValueError("Could not place a non-sketchgraph cylindrical contact sketch")

    @staticmethod
    def _try_place_sketch_on_cylinder_contact(
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        cylinder_radius: float,
        min_dim: float,
    ) -> bool:
        if ExtrudeFactory._remap_polygon_sketch_to_contact(
            sketch,
            target_width,
            target_depth,
            contact_kind="arc",
            cylinder_radius=cylinder_radius,
            min_dim=min_dim,
        ):
            return True
        if ExtrudeFactory._splice_sketch_to_cylinder_contact(
            sketch,
            target_width,
            target_depth,
            cylinder_radius=cylinder_radius,
            min_dim=min_dim,
        ):
            return True
        return False

    def _replace_with_non_sketchgraph_contact_sketch(
        self,
        sketch: Sketch,
        try_place_contact,
    ) -> bool:
        for _ in range(NON_SKETCHGRAPH_CONTACT_FALLBACK_RETRIES):
            try:
                candidate = self._generate_non_sketchgraph_sketch()
                if try_place_contact(candidate):
                    sketch.__dict__.clear()
                    sketch.__dict__.update(deepcopy(candidate.__dict__))
                    return True
            except Exception:
                continue
        return False

    def _generate_non_sketchgraph_sketch(self) -> Sketch:
        from_sketchgraph = self.sketch_factory.from_sketchgraph
        from_sketchgraph_probability = self.sketch_factory.from_sketchgraph_probability
        self.sketch_factory.from_sketchgraph = False
        self.sketch_factory.from_sketchgraph_probability = None
        try:
            return self.sketch_factory.generate()
        finally:
            self.sketch_factory.from_sketchgraph = from_sketchgraph
            self.sketch_factory.from_sketchgraph_probability = (
                from_sketchgraph_probability
            )

    @staticmethod
    def _splice_sketch_to_planar_contact(
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        depth_sign: float,
        min_dim: float,
    ) -> bool:
        state = deepcopy(sketch.__dict__)
        try:
            gap, body_depth = ExtrudeFactory._contact_gap_and_body_depth(
                target_depth,
                min_dim,
            )
            ExtrudeFactory._fit_sketch_to_box(sketch, target_width, body_depth, min_dim)
            bbox = cq.Shape(sketch.to_shape()).BoundingBox()
            if depth_sign >= 0:
                shift_y = gap - bbox.ymin
            else:
                shift_y = -gap - bbox.ymax
            sketch.transform([0.0, shift_y], 1.0)

            half_width = 0.5 * target_width
            return ExtrudeFactory._splice_first_outer_wire_to_contact(
                sketch,
                contact_vertices=[[-half_width, 0.0], [half_width, 0.0]],
                contact_edge={"type": "line"},
                near_axis=1,
                near_sign=depth_sign,
            )
        except Exception:
            sketch.__dict__.clear()
            sketch.__dict__.update(state)
            return False

    @staticmethod
    def _splice_sketch_to_cylinder_contact(
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        cylinder_radius: float,
        min_dim: float,
    ) -> bool:
        state = deepcopy(sketch.__dict__)
        try:
            gap, body_depth = ExtrudeFactory._contact_gap_and_body_depth(
                target_depth,
                min_dim,
            )
            arc_x, arc_y = ExtrudeFactory._cylinder_contact_arc(
                target_width,
                cylinder_radius,
                min_dim,
            )
            body_width = 2.0 * arc_y
            if body_width < min_dim:
                raise ValueError("Cylindrical contact has too little chord width")

            ExtrudeFactory._fit_sketch_to_box(sketch, body_depth, body_width, min_dim)
            bbox = cq.Shape(sketch.to_shape()).BoundingBox()
            sketch.transform([gap - bbox.xmin, -(bbox.ymin + bbox.ymax) / 2.0], 1.0)

            return ExtrudeFactory._splice_first_outer_wire_to_contact(
                sketch,
                contact_vertices=[[arc_x, -arc_y], [arc_x, arc_y]],
                contact_edge={"type": "arc", "point": [0.0, 0.0]},
                near_axis=0,
                near_sign=1.0,
            )
        except Exception:
            sketch.__dict__.clear()
            sketch.__dict__.update(state)
            return False

    @staticmethod
    def _contact_gap_and_body_depth(
        target_depth: float,
        min_dim: float,
    ) -> tuple[float, float]:
        gap = min(max(min_dim, 0.1 * target_depth), max(0.0, target_depth - min_dim))
        body_depth = target_depth - gap
        if gap <= 0 or body_depth < min_dim:
            raise ValueError("Contact sketch has too little room for lead-in")
        return gap, body_depth

    @staticmethod
    def _cylinder_contact_arc(
        target_width: float,
        cylinder_radius: float,
        min_dim: float,
    ) -> tuple[float, float]:
        radius = max(abs(cylinder_radius), 1e-6)
        theta = float(np.clip(target_width / (2.0 * radius), 0.03, 1.0))
        x = radius * (np.cos(theta) - 1.0) - min(min_dim, 0.1 * radius)
        y = radius * np.sin(theta)
        return x, y

    @staticmethod
    def _splice_first_outer_wire_to_contact(
        sketch: Sketch,
        contact_vertices: list[list[float]],
        contact_edge: dict,
        near_axis: int,
        near_sign: float,
    ) -> bool:
        first_outer_idx = next(
            (idx for idx, wire in enumerate(sketch.wires) if wire.get("outer")),
            None,
        )
        if first_outer_idx is None:
            return False

        wire = deepcopy(sketch.wires[first_outer_idx])
        if wire["type"] == "circle":
            wire = ExtrudeFactory._circle_wire_to_arc_polygon(wire)
        if wire["type"] != "polygon" or len(wire["vertices"]) < 3:
            return False
        if len(wire["vertices"]) != len(wire["edges"]):
            return False

        break_edge = ExtrudeFactory._nearest_contact_edge_index(
            wire,
            near_axis,
            near_sign,
        )
        vertices = wire["vertices"]
        edges = wire["edges"]
        n_edges = len(edges)
        a = np.array(vertices[break_edge], dtype=float)
        b = np.array(vertices[(break_edge + 1) % n_edges], dtype=float)
        contact_start, contact_end = ExtrudeFactory._paired_contact_vertices(
            contact_vertices,
            a,
            b,
        )

        path_vertices = [deepcopy(vertices[(break_edge + 1) % n_edges])]
        path_edges = []
        idx = (break_edge + 1) % n_edges
        while idx != break_edge:
            path_edges.append(deepcopy(edges[idx]))
            idx = (idx + 1) % n_edges
            path_vertices.append(deepcopy(vertices[idx]))

        sketch.wires[first_outer_idx] = {
            "type": "polygon",
            "outer": True,
            "vertices": [contact_start, contact_end] + path_vertices,
            "edges": [
                deepcopy(contact_edge),
                {"type": "line"},
                *path_edges,
                {"type": "line"},
            ],
        }
        cq.Shape(sketch.to_shape()).BoundingBox()
        return True

    @staticmethod
    def _paired_contact_vertices(
        contact_vertices: list[list[float]],
        a: np.ndarray,
        b: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        first = np.array(contact_vertices[0], dtype=float)
        second = np.array(contact_vertices[1], dtype=float)
        direct = np.linalg.norm(a - first) + np.linalg.norm(b - second)
        swapped = np.linalg.norm(a - second) + np.linalg.norm(b - first)
        if direct <= swapped:
            return deepcopy(contact_vertices[0]), deepcopy(contact_vertices[1])
        return deepcopy(contact_vertices[1]), deepcopy(contact_vertices[0])

    @staticmethod
    def _nearest_contact_edge_index(
        wire: dict,
        near_axis: int,
        near_sign: float,
    ) -> int:
        scores = []
        vertices = wire["vertices"]
        for idx, edge in enumerate(wire["edges"]):
            if edge["type"] == "arc":
                midpoint = np.array(edge["point"], dtype=float)
            else:
                start = np.array(vertices[idx], dtype=float)
                end = np.array(vertices[(idx + 1) % len(vertices)], dtype=float)
                midpoint = 0.5 * (start + end)
            scores.append(float(near_sign * midpoint[near_axis]))
        return int(np.argmin(scores))

    @staticmethod
    def _circle_wire_to_arc_polygon(wire: dict, n_segments: int = 4) -> dict:
        center = np.array(wire["center"], dtype=float)
        radius = float(wire["radius"])
        if radius <= 0:
            raise ValueError("Circle wire radius must be positive")

        vertices = []
        edges = []
        for idx in range(n_segments):
            angle = 2.0 * math.pi * (idx + 0.5) / n_segments
            mid_angle = 2.0 * math.pi * (idx + 1.0) / n_segments
            vertices.append(
                [
                    float(center[0] + radius * math.cos(angle)),
                    float(center[1] + radius * math.sin(angle)),
                ]
            )
            edges.append(
                {
                    "type": "arc",
                    "point": [
                        float(center[0] + radius * math.cos(mid_angle)),
                        float(center[1] + radius * math.sin(mid_angle)),
                    ],
                }
            )

        return {
            "type": "polygon",
            "outer": wire["outer"],
            "vertices": vertices,
            "edges": edges,
        }

    @staticmethod
    def _remap_polygon_sketch_to_contact(
        sketch: Sketch,
        target_width: float,
        target_depth: float,
        *,
        contact_kind: str,
        min_dim: float,
        depth_sign: float = 1.0,
        cylinder_radius: float | None = None,
    ) -> bool:
        if not sketch.wires or any(wire["type"] != "polygon" for wire in sketch.wires):
            return False

        first_wire = sketch.wires[0]
        if len(first_wire["vertices"]) < 3:
            return False

        v0 = np.array(first_wire["vertices"][0], dtype=float)
        v1 = np.array(first_wire["vertices"][1], dtype=float)
        edge = v1 - v0
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 1e-9:
            return False

        center = 0.5 * (v0 + v1)
        tangent = edge / edge_length
        side = np.array([-tangent[1], tangent[0]], dtype=float)

        points = ExtrudeFactory._sketch_points(sketch)
        side_values = [float(np.dot(point - center, side)) for point in points]
        if abs(min(side_values)) > max(side_values):
            side = -side
            side_values = [-value for value in side_values]

        side_max = max(side_values)
        if side_max <= 1e-9:
            return False

        if contact_kind == "line":
            half_width = 0.5 * target_width
            if half_width < 0.5 * min_dim or target_depth < min_dim:
                return False

            def map_point(point: np.ndarray) -> list[float]:
                along = float(np.dot(point - center, tangent))
                away = max(0.0, float(np.dot(point - center, side)))
                x = np.clip(along * target_width / edge_length, -half_width, half_width)
                y = depth_sign * min(target_depth, away * target_depth / side_max)
                return [float(x), float(y)]

            first_edge = {"type": "line"}
            first_vertices = [[-half_width, 0.0], [half_width, 0.0]]

        elif contact_kind == "arc":
            if cylinder_radius is None:
                return False
            radius = max(abs(cylinder_radius), 1e-6)
            theta = float(np.clip(target_width / (2.0 * radius), 0.03, 1.0))
            arc_x = radius * (np.cos(theta) - 1.0) - min(min_dim, 0.1 * radius)
            arc_y = radius * np.sin(theta)
            if 2.0 * arc_y < min_dim or target_depth < min_dim:
                return False

            def map_point(point: np.ndarray) -> list[float]:
                along = float(np.dot(point - center, tangent))
                away = max(0.0, float(np.dot(point - center, side)))
                x = arc_x + min(target_depth - arc_x, away * (target_depth - arc_x) / side_max)
                y = np.clip(along * (2.0 * arc_y) / edge_length, -arc_y, arc_y)
                return [float(x), float(y)]

            first_edge = {"type": "arc", "point": [0.0, 0.0]}
            first_vertices = [[arc_x, -arc_y], [arc_x, arc_y]]

        else:
            raise ValueError(f"Unsupported contact kind: {contact_kind}")

        for wire in sketch.wires:
            for vertex in wire["vertices"]:
                mapped = map_point(np.array(vertex, dtype=float))
                vertex[0], vertex[1] = mapped
            for edge_data in wire["edges"]:
                if edge_data["type"] == "arc":
                    mapped = map_point(np.array(edge_data["point"], dtype=float))
                    edge_data["point"][0], edge_data["point"][1] = mapped

        first_wire["vertices"][0] = first_vertices[0]
        first_wire["vertices"][1] = first_vertices[1]
        first_wire["edges"][0] = first_edge
        return True

    @staticmethod
    def _sketch_points(sketch: Sketch) -> list[np.ndarray]:
        points = []
        for wire in sketch.wires:
            if wire["type"] != "polygon":
                continue
            points.extend(np.array(vertex, dtype=float) for vertex in wire["vertices"])
            for edge in wire["edges"]:
                if edge["type"] == "arc":
                    points.append(np.array(edge["point"], dtype=float))
        return points

    @staticmethod
    def _assert_sketch_inside_world_box(
        sketch: Sketch,
        bounds: WorkplaneBounds,
        tolerance: float = 1e-7,
    ) -> None:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        if (
            bbox.xmin < -bounds.x_neg - tolerance
            or bbox.xmax > bounds.x_pos + tolerance
            or bbox.ymin < -bounds.y_neg - tolerance
            or bbox.ymax > bounds.y_pos + tolerance
        ):
            raise ValueError("Sketch exceeds local world bounds")

    @staticmethod
    def from_dict(entity: dict) -> "ExtrudeFactory":
        return ExtrudeFactory(
            SketchFactory.from_dict(entity["sketch_factory"]),
            tangent_plane_prob=entity.get("tangent_plane_prob", 0.5),
            out_of_face_prob=entity.get("out_of_face_prob", 0.5),
            point_out_of_surface_probability=entity.get(
                "point_out_of_surface_probability",
                0.0,
            ),
            mesh_deflection=entity.get("mesh_deflection", 0.005),
        )

    def to_dict(self) -> dict:
        return {
            "type": "ExtrudeFactory",
            "sketch_factory": self.sketch_factory.to_dict(),
            "tangent_plane_prob": self.tangent_plane_prob,
            "out_of_face_prob": self.out_of_face_prob,
            "point_out_of_surface_probability": (
                self.point_out_of_surface_probability
            ),
            "mesh_deflection": self.mesh_deflection,
        }


class TwoExtrudesFactory(ExtrudeFactory):
    def __init__(
        self,
        sketch_factory: SketchFactory | dict,
        tangent_plane_prob: float = 0.5,
        out_of_face_prob: float = 0.5,
        point_out_of_surface_probability: float = 0.0,
        mesh_deflection: float = 0.005,
        second_new_sketch_probability: float = 0.8,
        second_new_sketch_retries: int = 20,
    ):
        super().__init__(
            sketch_factory=sketch_factory,
            tangent_plane_prob=tangent_plane_prob,
            out_of_face_prob=out_of_face_prob,
            point_out_of_surface_probability=point_out_of_surface_probability,
            mesh_deflection=mesh_deflection,
        )
        self.second_new_sketch_probability = float(
            np.clip(second_new_sketch_probability, 0.0, 1.0)
        )
        self.second_new_sketch_retries = max(0, int(second_new_sketch_retries))

    def _make_second_extrude(self, first: Extrude) -> TwoExtrudes:
        center = self._sketch_center(first.sketch)
        first_area = self._sketch_area(first.sketch)
        area_ratio = float(np.random.uniform(1.1, 3.0))
        second_sketch = self._sample_second_sketch(
            first.sketch,
            center,
            first_area * area_ratio,
        )

        height_divisor = float(np.random.uniform(2.0, 10.0))
        second_extent = first.extent / height_divisor
        second_normal_offset = float(
            np.random.uniform(
                min(0.0, first.extent),
                max(0.0, first.extent),
            )
        )
        second = Extrude(
            second_sketch,
            second_extent,
            point_on_surface=False,
        )
        return TwoExtrudes(first, second, center, second_normal_offset)

    def _sample_second_sketch(
        self,
        first_sketch: Sketch,
        center: tuple[float, float],
        target_area: float,
    ) -> Sketch:
        if np.random.random() < self.second_new_sketch_probability:
            for _ in range(self.second_new_sketch_retries):
                try:
                    candidate = self.sketch_factory.generate()
                    self._center_and_scale_sketch(candidate, target_area)
                    if self._contains_first_sketch(first_sketch, candidate, center):
                        return candidate
                except Exception:
                    continue
            raise ValueError(
                "Could not sample a fresh second extrude sketch containing the first"
            )

        second_sketch = deepcopy(first_sketch)
        self._center_and_scale_sketch(second_sketch, target_area)
        return second_sketch

    @staticmethod
    def _center_and_scale_sketch(sketch: Sketch, target_area: float) -> None:
        area = TwoExtrudesFactory._sketch_area(sketch)
        center = TwoExtrudesFactory._sketch_center(sketch)
        scale = math.sqrt(float(target_area) / area)
        sketch.transform([-center[0], -center[1]], scale)

    @staticmethod
    def _contains_first_sketch(
        first_sketch: Sketch,
        second_sketch: Sketch,
        second_center: tuple[float, float],
    ) -> bool:
        placed_second = deepcopy(second_sketch)
        placed_second.transform([second_center[0], second_center[1]], 1.0)
        first_shape = first_sketch.to_shape()
        remaining = BRepAlgoAPI_Cut(first_shape, placed_second.to_shape()).Shape()
        first_area = TwoExtrudesFactory._sketch_area(first_sketch)
        remaining_area = abs(shape_to_area(remaining))
        return remaining_area <= max(first_area * 1e-5, 1e-8)

    @staticmethod
    def _sketch_area(sketch: Sketch) -> float:
        area = abs(float(shape_to_area(sketch.to_shape())))
        if area <= 0 or np.isclose(area, 0):
            raise ValueError("Generated sketch has zero area")
        return area

    def _resample_first_extent(
        self,
        first: Extrude,
        max_abs_extent: float | None = None,
    ) -> None:
        bbox_size = self._sketch_bbox_max_size(first.sketch)
        min_abs_extent = 1.5 * bbox_size
        high_abs_extent = 10.0 * bbox_size
        if max_abs_extent is not None:
            high_abs_extent = min(high_abs_extent, float(max_abs_extent))
        if high_abs_extent < min_abs_extent:
            raise ValueError(
                "TwoExtrudes first extent has too little room for required "
                "sketch-height ratio"
            )
        sign = math.copysign(1.0, first.extent if first.extent else 1.0)
        first.extent = sign * float(np.random.uniform(min_abs_extent, high_abs_extent))

    @staticmethod
    def _sketch_bbox_max_size(sketch: Sketch) -> float:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        width = float(bbox.xmax - bbox.xmin)
        height = float(bbox.ymax - bbox.ymin)
        if width <= 0 or height <= 0:
            raise ValueError("Generated sketch has an empty bounding box")
        return max(width, height)

    @staticmethod
    def _sketch_center(sketch: Sketch) -> tuple[float, float]:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        return (
            0.5 * (float(bbox.xmin) + float(bbox.xmax)),
            0.5 * (float(bbox.ymin) + float(bbox.ymax)),
        )

    def generate(self) -> TwoExtrudes:
        first = super().generate()
        self._resample_first_extent(first)
        return self._make_second_extrude(first)

    def generate_on_existing(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        world_size: float,
        generation_world_half: float = 1.0,
    ) -> tuple[TwoExtrudes, AxisWorkplane]:
        first, workplane = super().generate_on_existing(
            cad_object,
            sampler,
            world_size,
            generation_world_half,
        )
        direction = axis_vector(workplane.normal).multiply(
            math.copysign(1.0, first.extent if first.extent else 1.0)
        )
        max_abs_extent = self._distance_to_world_border(
            cq.Vector(*workplane.point),
            direction,
            generation_world_half,
        )
        self._resample_first_extent(first, max_abs_extent)
        return self._make_second_extrude(first), workplane

    @staticmethod
    def from_dict(entity: dict) -> "TwoExtrudesFactory":
        return TwoExtrudesFactory(
            SketchFactory.from_dict(entity["sketch_factory"]),
            tangent_plane_prob=entity.get("tangent_plane_prob", 0.5),
            out_of_face_prob=entity.get("out_of_face_prob", 0.5),
            point_out_of_surface_probability=entity.get(
                "point_out_of_surface_probability",
                0.0,
            ),
            mesh_deflection=entity.get("mesh_deflection", 0.005),
            second_new_sketch_probability=entity.get(
                "second_new_sketch_probability",
                0.8,
            ),
            second_new_sketch_retries=entity.get("second_new_sketch_retries", 20),
        )

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "TwoExtrudesFactory"
        data["second_new_sketch_probability"] = self.second_new_sketch_probability
        data["second_new_sketch_retries"] = self.second_new_sketch_retries
        return data


factories.register("extrude", ExtrudeFactory)
factories.register("two_extrudes", TwoExtrudesFactory)
