import logging
import cadquery as cq
import numpy as np
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from .base import BaseFactory, BaseOperation
from .extrude import (
    ExtrudeFactory,
    WORKPLANE_BY_NORMAL,
    WORKPLANE_OFFSET_INDEX,
    WorkplaneBounds,
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
    axis_name_from_vector,
    axis_vector,
    closest_surface_point_from_compound,
    point_vertex,
    positive_axis,
    precompute_surface_compound,
    shape_from_cad_object,
    signed_axis,
    workplane_at_closest_point,
)
from .utils import _is_right_half_plane

logger = logging.getLogger(__name__)

WORKPLANE_COORD_AXES = {
    "XY": ("X", "Y"),
    "YZ": ("Y", "Z"),
    "ZX": ("Z", "X"),
}


def _point_on_workplane(
    point: tuple[float, float, float],
    workplane: str,
) -> tuple[float, float]:
    if workplane == "XY":
        return (point[0], point[1])
    if workplane == "ZX":
        return (point[2], point[0])
    if workplane == "YZ":
        return (point[1], point[2])
    raise ValueError("workplane must be one of 'XY', 'YZ', or 'ZX'")


def _revolve_axis_points(
    point: tuple[float, float, float],
    workplane: str,
    revolve_axis: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    axis = positive_axis(revolve_axis)
    workplane_axes = WORKPLANE_COORD_AXES.get(workplane)
    if workplane_axes is None:
        raise ValueError("workplane must be one of 'XY', 'YZ', or 'ZX'")
    if axis not in workplane_axes:
        raise ValueError(
            f"revolve_axis={revolve_axis!r} must lie in workplane {workplane!r}"
        )

    dir_point_0 = _point_on_workplane(point, workplane)
    dir_point_1 = list(dir_point_0)
    dir_point_1[workplane_axes.index(axis)] += 1.0
    return dir_point_0, (dir_point_1[0], dir_point_1[1])


def revolve(
    r: cq.Workplane | cq.Shape | None,
    point: tuple[float, float, float],
    workplane: str,
    sketch: str,
    rotation_angle: float,
    revolve_axis: str,
):
    if workplane not in WORKPLANE_OFFSET_INDEX:
        raise ValueError("workplane must be one of 'XY', 'YZ', or 'ZX'")

    point = tuple(float(value) for value in point)
    if r is None:
        offset = point[WORKPLANE_OFFSET_INDEX[workplane]]
        axis_point = point
    else:
        shape = shape_from_cad_object(r)
        surface_compound = precompute_surface_compound(shape)
        surface_point = closest_surface_point_from_compound(surface_compound, point)
        axis_point = tuple(float(value) for value in surface_point.toTuple())
        offset = axis_point[WORKPLANE_OFFSET_INDEX[workplane]]

    prefix = f"cq.Workplane('{workplane}').workplane(offset={offset})"
    dir_point_0, dir_point_1 = _revolve_axis_points(
        axis_point,
        workplane,
        revolve_axis,
    )
    expression = (
        _join_workplane_and_sketch(prefix, sketch)
        + f".revolve({rotation_angle}, {dir_point_0}, {dir_point_1}).val()"
    )
    body = eval(expression, {"cq": cq})

    if r is None:
        return cq.Workplane(workplane).add(body)
    if isinstance(r, cq.Shape):
        return cq.Workplane(workplane).add(r).union(body)
    return r.union(body)


def workplane_for_revolve(normal: str, x_dir: str) -> str:
    normal_name = positive_axis(normal)
    if normal_name not in WORKPLANE_BY_NORMAL:
        raise ValueError(f"unknown workplane normal {normal!r}")
    axis_vector(x_dir)
    return WORKPLANE_BY_NORMAL[normal_name]


def _source_y_dir(normal: str, x_dir: str) -> str:
    normal_vec = axis_vector(normal)
    x_dir_vec = axis_vector(x_dir)
    y_dir_vec = normal_vec.cross(x_dir_vec)
    y_axis = axis_name_from_vector(y_dir_vec)
    if y_axis is None:
        raise ValueError("Could not resolve workplane yDir to a global axis")
    return signed_axis(*y_axis)


def _is_horizontal_axis(
    axis: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    (x1, y1), (x2, y2) = axis
    return abs(float(x2) - float(x1)) >= abs(float(y2) - float(y1))


def _positive_axis_name(axis: str) -> str:
    return positive_axis(axis)


def _point_shifted_along_axis(
    point: tuple[float, float, float],
    axis: str,
    distance: float,
) -> tuple[float, float, float]:
    vector = axis_vector(axis)
    return (
        point[0] + vector.x * distance,
        point[1] + vector.y * distance,
        point[2] + vector.z * distance,
    )


def compute_revolve_axis_points(
    sketch: Sketch,
    dist_to_axis: float,
    vertical_axis_line: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Build revolve axis segment (p1, p2) in sketch 2D from profile bbox and offset."""
    bbox = cq.Face(sketch.to_shape()).BoundingBox()
    x_min, y_min, x_max, y_max = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
    if _is_right_half_plane(sketch.to_shape(), (0, 0, 0), (0, 1, 0)):
        x1 = x_min - dist_to_axis
        y1 = y_min - dist_to_axis
        if vertical_axis_line:
            x2, y2 = x1, y1 - 1.0
        else:
            x2, y2 = x1 - 1.0, y1
    else:
        x1 = x_max + dist_to_axis
        y1 = y_max + dist_to_axis
        if vertical_axis_line:
            x2, y2 = x1, y1 + 1.0
        else:
            x2, y2 = x1 + 1.0, y1
    return ((float(x1), float(y1)), (float(x2), float(y2)))


def compute_revolve_axis_points_at_origin(
    vertical_axis_line: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Build a local revolve axis through the workplane origin."""
    if vertical_axis_line:
        return ((0.0, 0.0), (0.0, 1.0))
    return ((0.0, 0.0), (1.0, 0.0))


class Revolve(BaseOperation):
    def __init__(
        self,
        sketch: Sketch,
        angle_degrees: float = 360,
        axis: tuple[tuple[float, float], tuple[float, float]] | None = None,
        zero_dist_to_axis: bool = False,
    ):
        self.sketch = sketch
        self.angle_degrees = angle_degrees
        self.axis = axis
        self.zero_dist_to_axis = zero_dist_to_axis

    def to_string(self, s: str, plane: str | AxisWorkplane) -> str:
        if isinstance(plane, AxisWorkplane):
            return self._to_factory_call_string(plane)

        assert self.axis is not None
        logger.info("Building revolve string with axis=%s", self.axis)
        expr = self.sketch.to_string(plane)
        (x1, y1), (x2, y2) = self.axis
        expr += f".revolve({self.angle_degrees}, {(x1, y1)}, {(x2, y2)})\n"
        return expr

    def to_global_workplane_string(
        self,
        workplane_axis: str,
        origin: tuple[float, float, float],
    ) -> str:
        assert self.axis is not None
        local_sketch = self.sketch.to_string(f"cq.Workplane('{workplane_axis}')")
        global_sketch = sketch_to_global_coords(
            workplane_axis,
            origin,
            local_sketch,
        )
        sketch_arg = sketch_chain_only(global_sketch)
        revolve_axis = self._revolve_axis_from_workplane_axes(
            WORKPLANE_COORD_AXES[workplane_axis][0],
            WORKPLANE_COORD_AXES[workplane_axis][1],
        )
        if _is_horizontal_axis(self.axis):
            fixed_axis = WORKPLANE_COORD_AXES[workplane_axis][1]
            fixed_offset = float(self.axis[0][1])
        else:
            fixed_axis = WORKPLANE_COORD_AXES[workplane_axis][0]
            fixed_offset = float(self.axis[0][0])
        point = _point_shifted_along_axis(
            tuple(float(value) for value in origin),
            fixed_axis,
            fixed_offset,
        )
        return self._factory_call_string(
            point,
            workplane_axis,
            sketch_arg,
            revolve_axis,
        )

    def _to_factory_call_string(self, workplane: AxisWorkplane) -> str:
        normal_name = positive_axis(workplane.normal)
        workplane_axis = WORKPLANE_BY_NORMAL[normal_name]
        source_y_dir = _source_y_dir(workplane.normal, workplane.xDir)

        local_sketch = self.sketch.to_string(f"cq.Workplane('{workplane_axis}')")
        logger.debug(
            "revolve sketch before axis/global transform: %s",
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
        logger.debug("revolve sketch after axis/global transform: %s", sketch_arg)

        revolve_axis = self._revolve_axis_from_workplane_axes(
            workplane.xDir,
            source_y_dir,
        )
        return self._factory_call_string(
            tuple(float(value) for value in workplane.point),
            workplane_axis,
            sketch_arg,
            revolve_axis,
        )

    def _revolve_axis_from_workplane_axes(
        self,
        source_x_dir: str,
        source_y_dir: str,
    ) -> str:
        assert self.axis is not None
        if _is_horizontal_axis(self.axis):
            return _positive_axis_name(source_x_dir)
        return _positive_axis_name(source_y_dir)

    def _factory_call_string(
        self,
        point: tuple[float, float, float],
        workplane_axis: str,
        sketch_arg: str,
        revolve_axis: str,
    ) -> str:
        point_expr = _format_point(point)
        args = (
            f"r, {point_expr}, {workplane_axis!r}, "
            f"{_double_quoted_string(sketch_arg)}, "
            f"{_format_number(self.angle_degrees)}, {revolve_axis!r}"
        )
        return f"revolve({args})\n"

    def transform(
        self,
        shift: list[float],
        scale: float,
        *,
        sketch_shift: list[float] | None = None,
    ) -> None:
        assert self.axis is not None
        if sketch_shift is None:
            sketch_shift = [0.0, 0.0]
        sx, sy = float(sketch_shift[0]), float(sketch_shift[1])
        (x1, y1), (x2, y2) = self.axis
        self.axis = (
            ((x1 + sx) * scale, (y1 + sy) * scale),
            ((x2 + sx) * scale, (y2 + sy) * scale),
        )
        if self.axis[0][0] == self.axis[1][0]:
            self.axis = (self.axis[0], (self.axis[1][0], self.axis[0][1] + 1.0))
        else:
            self.axis = (self.axis[0], (self.axis[0][0] + 1.0, self.axis[1][1]))

    def round(self) -> None:
        assert self.axis is not None
        self.sketch.round()
        (x1, y1), (x2, y2) = self.axis
        self.axis = (
            (round(x1), round(y1)),
            (round(x2), round(y2)),
        )

    def fix(self) -> None:
        self.sketch.fix()

    def to_dict(self) -> dict:
        return {
            "type": "Revolve",
            "sketch": self.sketch.to_dict(),
            "angle_degrees": self.angle_degrees,
            "axis": self.axis,
            "zero_dist_to_axis": self.zero_dist_to_axis,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Revolve":
        assert (
            entity["type"] == "Revolve"
        ), f"Trying to build Revolve from type {entity['type']}"
        return Revolve(
            Sketch.from_dict(entity["sketch"]),
            entity["angle_degrees"],
            entity["axis"],
            entity["zero_dist_to_axis"],
        )


class RevolveFactory(BaseFactory):
    SIMPLE_SKETCH_RESAMPLE_ATTEMPTS = 5
    SIMPLE_SKETCH_RESAMPLE_PROBABILITY = 0.9

    def __init__(
        self,
        sketch_factory: SketchFactory | dict,
        full_revolve_probability: float | None = None,
        zero_dist_to_axis_probability: float = 0.5,
        tangent_plane_prob: float = 0.5,
        mesh_deflection: float = 0.05,
        min_revolve_angle: int = 30,
    ):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)
        if min_revolve_angle < 1 or min_revolve_angle >= 355:
            raise ValueError("min_revolve_angle must be in [1, 354] degrees")
        self.sketch_factory = sketch_factory
        self.full_revolve_probability = full_revolve_probability
        self.zero_dist_to_axis_probability = zero_dist_to_axis_probability
        self.min_revolve_angle = int(min_revolve_angle)
        self.tangent_plane_prob = tangent_plane_prob
        self.mesh_deflection = mesh_deflection
        self.r_min, self.r_max = 0.01, 1.0
        self.world_size: float | None = None

    @staticmethod
    def _sketch_has_circle_or_rect(sketch: Sketch) -> bool:
        sketch_string = sketch.to_string("cq.Workplane('XY')")
        return "circle" in sketch_string or "rect" in sketch_string

    def _sample_revolve_sketch(self) -> Sketch:
        sketch = None
        for _ in range(self.SIMPLE_SKETCH_RESAMPLE_ATTEMPTS):
            sketch = self.sketch_factory.generate()
            if (
                not self._sketch_has_circle_or_rect(sketch)
                or np.random.random() >= self.SIMPLE_SKETCH_RESAMPLE_PROBABILITY
            ):
                return sketch
        assert sketch is not None
        return sketch

    def generate(self) -> Revolve:
        sketch = self._sample_revolve_sketch()
        return self._make_revolve(sketch)

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
    ) -> tuple[Revolve, AxisWorkplane]:
        placement_helper = ExtrudeFactory(
            sketch_factory=self.sketch_factory,
            tangent_plane_prob=0.0,
            mesh_deflection=self.mesh_deflection,
        )
        placement = placement_helper._sample_placement(sampler)
        axis_parallel_to_workplane_x = (
            positive_axis(placement.workplane.xDir) == placement.site.normal_axis
        )
        workplane = workplane_at_closest_point(
            cad_object,
            placement.site.point,
            placement.workplane.normal,
            placement.workplane.xDir,
            surface_compound=sampler.surface_compound,
        )

        sketch = self._sample_revolve_sketch()
        min_dim = placement_helper._min_generation_dimension(world_size)
        world_bounds = placement_helper._workplane_world_bounds(
            workplane,
            generation_world_half,
        )
        revolution_radius = self._sample_revolution_radius(
            placement.site.face,
            workplane.plane.origin,
            world_size,
            generation_world_half,
        )
        revolve = self._make_revolve(
            sketch,
            vertical_axis_line=not axis_parallel_to_workplane_x,
            axis_through_origin=True,
        )
        self._linearize_polygon_arcs(revolve.sketch)
        self._resize_sketch_for_revolution_radius(
            revolve.sketch,
            vertical_axis_line=not axis_parallel_to_workplane_x,
            revolution_radius=revolution_radius,
            bounds=world_bounds,
            min_dim=min_dim,
            axis_outward_sign=self._axis_outward_sign(
                workplane,
                placement.site.outward,
                vertical_axis_line=not axis_parallel_to_workplane_x,
            ),
        )
        minimum_span = self._minimum_span_before_normalization(world_size)
        self._ensure_min_sketch_span(revolve.sketch, minimum_span)
        self._place_sketch_on_revolve_side(
            revolve.sketch,
            vertical_axis_line=not axis_parallel_to_workplane_x,
            min_dim=min_dim,
            axis_outward_sign=self._axis_outward_sign(
                workplane,
                placement.site.outward,
                vertical_axis_line=not axis_parallel_to_workplane_x,
            ),
        )
        placement_helper._assert_sketch_inside_world_box(revolve.sketch, world_bounds)
        logger.info(
            "Generated placed revolve with angle=%s, axis=%s, normal=%s, xDir=%s",
            revolve.angle_degrees,
            revolve.axis,
            placement.workplane.normal,
            placement.workplane.xDir,
        )
        return revolve, placement.workplane

    @staticmethod
    def _sample_revolution_radius(
        face: cq.Face,
        origin: cq.Vector,
        world_size: float,
        generation_world_half: float,
    ) -> float:
        border_distance = RevolveFactory._distance_to_face_border(face, origin)
        min_radius = 15.0 / max(float(world_size), 1.0)
        max_radius = float(generation_world_half)
        if max_radius < min_radius:
            raise ValueError("World size is too small for minimum revolve radius")
        sigma = max(border_distance, 1e-12)
        radius = float(np.random.normal(border_distance, sigma))
        if radius < border_distance:
            if border_distance <= min_radius:
                radius = min_radius
            else:
                radius = float(np.random.uniform(min_radius, border_distance))
        return float(np.clip(radius, min_radius, max_radius))

    @staticmethod
    def _distance_to_face_border(face: cq.Face, origin: cq.Vector) -> float:
        edges = face.Edges()
        if not edges:
            raise ValueError("Cannot sample revolve radius from a face with no borders")
        edge_compound = cq.Compound.makeCompound(edges)
        extrema = BRepExtrema_DistShapeShape(
            point_vertex(origin).wrapped,
            edge_compound.wrapped,
        )
        extrema.Perform()
        if not extrema.IsDone() or extrema.NbSolution() < 1:
            raise ValueError("Could not compute distance from workplane origin to face border")
        return float(extrema.Value())

    @staticmethod
    def _linearize_polygon_arcs(sketch: Sketch) -> None:
        for wire in sketch.wires:
            if wire["type"] != "polygon":
                continue
            for edge in wire["edges"]:
                if edge["type"] == "arc":
                    edge.clear()
                    edge["type"] = "line"

    @staticmethod
    def _resize_sketch_for_revolution_radius(
        sketch: Sketch,
        vertical_axis_line: bool,
        revolution_radius: float,
        bounds: WorkplaneBounds,
        min_dim: float,
        axis_outward_sign: float = 1.0,
    ) -> None:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        tol = 1e-9
        clearance = max(0.25 * min_dim, 1e-6)
        radial_target = revolution_radius - clearance
        if radial_target <= 0:
            raise ValueError("Revolve radius is too small for sketch clearance")
        if vertical_axis_line:
            width = bbox.xmax - bbox.xmin
            if width <= 1e-12:
                raise ValueError("Sketch has empty radial bounds around revolve axis")
            sketch.transform([0.0, 0.0], radial_target / width)
            bbox = cq.Shape(sketch.to_shape()).BoundingBox()
            if revolution_radius <= bounds.x_pos + tol:
                sketch.transform([clearance - bbox.xmin, 0.0], 1.0)
            elif revolution_radius <= bounds.x_neg + tol:
                sketch.transform([-clearance - bbox.xmax, 0.0], 1.0)
            else:
                raise ValueError("Sketch has too little one-sided room around revolve axis")
            RevolveFactory._anchor_sketch_to_workplane_origin(
                sketch,
                vertical_axis_line=True,
                axis_outward_sign=axis_outward_sign,
            )
        else:
            height = bbox.ymax - bbox.ymin
            if height <= 1e-12:
                raise ValueError("Sketch has empty radial bounds around revolve axis")
            sketch.transform([0.0, 0.0], radial_target / height)
            bbox = cq.Shape(sketch.to_shape()).BoundingBox()
            if revolution_radius <= bounds.y_pos + tol:
                sketch.transform([0.0, clearance - bbox.ymin], 1.0)
            elif revolution_radius <= bounds.y_neg + tol:
                sketch.transform([0.0, -clearance - bbox.ymax], 1.0)
            else:
                raise ValueError("Sketch has too little one-sided room around revolve axis")
            RevolveFactory._anchor_sketch_to_workplane_origin(
                sketch,
                vertical_axis_line=False,
                axis_outward_sign=axis_outward_sign,
            )

    @staticmethod
    def _axis_outward_sign(
        workplane: cq.Workplane,
        outward: cq.Vector,
        vertical_axis_line: bool,
    ) -> float:
        axis_dir = workplane.plane.yDir if vertical_axis_line else workplane.plane.xDir
        return 1.0 if axis_dir.normalized().dot(outward.normalized()) >= 0 else -1.0

    @staticmethod
    def _anchor_sketch_to_workplane_origin(
        sketch: Sketch,
        vertical_axis_line: bool,
        axis_outward_sign: float,
    ) -> None:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        if vertical_axis_line:
            shift_y = -bbox.ymin if axis_outward_sign >= 0 else -bbox.ymax
            sketch.transform([0.0, shift_y], 1.0)
        else:
            shift_x = -bbox.xmin if axis_outward_sign >= 0 else -bbox.xmax
            sketch.transform([shift_x, 0.0], 1.0)

    @staticmethod
    def _place_sketch_on_revolve_side(
        sketch: Sketch,
        vertical_axis_line: bool,
        min_dim: float,
        axis_outward_sign: float,
    ) -> None:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        clearance = max(0.25 * min_dim, 1e-6)
        if vertical_axis_line:
            radial_positive = abs(bbox.xmax) >= abs(bbox.xmin)
            shift_x = clearance - bbox.xmin if radial_positive else -clearance - bbox.xmax
            sketch.transform([shift_x, 0.0], 1.0)
        else:
            radial_positive = abs(bbox.ymax) >= abs(bbox.ymin)
            shift_y = clearance - bbox.ymin if radial_positive else -clearance - bbox.ymax
            sketch.transform([0.0, shift_y], 1.0)
        RevolveFactory._anchor_sketch_to_workplane_origin(
            sketch,
            vertical_axis_line,
            axis_outward_sign,
        )

    @staticmethod
    def _minimum_span_before_normalization(world_size: float | None) -> float:
        if world_size is None:
            return 0.0
        return 10.0 / max(float(world_size), 1.0)

    @staticmethod
    def _stretch_sketch_point(
        point: list[float],
        center_x: float,
        center_y: float,
        scale_x: float,
        scale_y: float,
    ) -> None:
        point[0] = center_x + (point[0] - center_x) * scale_x
        point[1] = center_y + (point[1] - center_y) * scale_y

    @staticmethod
    def _ensure_min_sketch_span(sketch: Sketch, minimum_span: float) -> None:
        if minimum_span <= 0.0:
            return

        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        span_x = bbox.xmax - bbox.xmin
        span_y = bbox.ymax - bbox.ymin
        if span_x <= 1e-12 or span_y <= 1e-12:
            raise ValueError("Cannot stretch sketch with empty span")

        scale_x = max(1.0, minimum_span / span_x)
        scale_y = max(1.0, minimum_span / span_y)
        if scale_x == 1.0 and scale_y == 1.0:
            return

        center_x = 0.5 * (bbox.xmin + bbox.xmax)
        center_y = 0.5 * (bbox.ymin + bbox.ymax)
        for wire in sketch.wires:
            if wire["type"] == "circle":
                RevolveFactory._stretch_sketch_point(
                    wire["center"],
                    center_x,
                    center_y,
                    scale_x,
                    scale_y,
                )
                wire["radius"] *= max(scale_x, scale_y)
                continue

            for vertex in wire["vertices"]:
                RevolveFactory._stretch_sketch_point(
                    vertex,
                    center_x,
                    center_y,
                    scale_x,
                    scale_y,
                )
            for edge in wire["edges"]:
                if edge["type"] == "arc":
                    RevolveFactory._stretch_sketch_point(
                        edge["point"],
                        center_x,
                        center_y,
                        scale_x,
                        scale_y,
                    )

        if sketch.point is not None:
            point = [sketch.point[0], sketch.point[1]]
            RevolveFactory._stretch_sketch_point(
                point,
                center_x,
                center_y,
                scale_x,
                scale_y,
            )
            sketch.point = (point[0], point[1])
        if sketch.array_pattern_dx is not None:
            sketch.array_pattern_dx *= scale_x
        if sketch.array_pattern_dy is not None:
            sketch.array_pattern_dy *= scale_y
        if sketch.array_pattern_radius is not None:
            sketch.array_pattern_radius *= max(scale_x, scale_y)
        if sketch.array_pattern_sketch is not None:
            RevolveFactory._ensure_min_sketch_span(
                sketch.array_pattern_sketch,
                minimum_span,
            )

    @staticmethod
    def _sample_partial_revolve_angle(min_revolve_angle: int) -> int:
        angles = np.arange(45, 360, 45)
        angles = angles[angles >= min_revolve_angle]
        if len(angles) == 0:
            return 315
        return int(np.random.choice(angles))

    def _make_revolve(
        self,
        sketch: Sketch,
        vertical_axis_line: bool = True,
        axis_through_origin: bool = False,
        minimum_span: float | None = None,
    ) -> Revolve:
        # leave only outer wires
        outer_wire = None
        for wire in sketch.wires:
            if wire["outer"]:
                outer_wire = wire
                break
        sketch.wires = [outer_wire]
        if minimum_span is None:
            minimum_span = self._minimum_span_before_normalization(self.world_size)
        self._ensure_min_sketch_span(sketch, minimum_span)

        assert self.full_revolve_probability is not None
        angle_degrees = (
            360
            if np.random.random() < self.full_revolve_probability
            else self._sample_partial_revolve_angle(self.min_revolve_angle)
        )
        if axis_through_origin:
            dist_to_axis = 0.0
            zero_dist_to_axis = True
            axis = compute_revolve_axis_points_at_origin(vertical_axis_line)
        elif np.random.uniform() < self.zero_dist_to_axis_probability:
            dist_to_axis = 0.0
            zero_dist_to_axis = True
            axis = compute_revolve_axis_points(
                sketch,
                dist_to_axis,
                vertical_axis_line,
            )
        else:
            dist_to_axis = float(np.random.uniform(self.r_min, self.r_max))
            zero_dist_to_axis = False
            axis = compute_revolve_axis_points(
                sketch,
                dist_to_axis,
                vertical_axis_line,
            )

        logger.info(
            "Generated revolve: angle=%s, axis=%s (dist_to_axis used at gen=%s, zero_dist_to_axis=%s)",
            angle_degrees,
            axis,
            dist_to_axis,
            zero_dist_to_axis,
        )
        return Revolve(sketch, angle_degrees, axis, zero_dist_to_axis)

    @staticmethod
    def from_dict(entity: dict) -> "RevolveFactory":
        return RevolveFactory(
            SketchFactory.from_dict(entity["sketch_factory"]),
            entity["full_revolve_probability"],
            entity.get("zero_dist_to_axis_probability", 0.5),
            min_revolve_angle=entity.get("min_revolve_angle", 30),
            tangent_plane_prob=entity.get("tangent_plane_prob", 0.5),
            mesh_deflection=entity.get("mesh_deflection", 0.05),
        )

    def to_dict(self) -> dict:
        return {
            "type": "RevolveFactory",
            "sketch_factory": self.sketch_factory.to_dict(),
            "full_revolve_probability": self.full_revolve_probability,
            "zero_dist_to_axis_probability": self.zero_dist_to_axis_probability,
            "min_revolve_angle": self.min_revolve_angle,
            "tangent_plane_prob": self.tangent_plane_prob,
            "mesh_deflection": self.mesh_deflection,
        }


factories.register("revolve", RevolveFactory)
