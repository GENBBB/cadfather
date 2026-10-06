import logging
from dataclasses import dataclass

import cadquery as cq
import numpy as np

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
    PositiveAxisName,
    SampledSite,
    axis_name_from_vector,
    axis_vector,
    closest_point_on_face_surface,
    closest_surface_point_from_compound,
    normal_at_face_uv,
    orthogonal_axes,
    point_is_on_face,
    precompute_surface_compound,
    shape_from_cad_object,
    signed_axis,
    workplane_at_closest_point,
)
from .utils import is_face_outer

logger = logging.getLogger(__name__)


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


def orto_cut(
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


@dataclass(frozen=True)
class OrtoCutPlacement:
    site: SampledSite
    workplane: AxisWorkplane


class OrtoCut(BaseOperation):
    def __init__(
        self,
        sketch: Sketch,
        extent: float,
        through_all: bool = False,
        world_size: float | None = None,
    ):
        self.sketch = sketch
        self.extent = extent
        self.through_all = through_all
        self.world_size = world_size

    def to_string(
        self,
        plane: AxisWorkplane,
        world_size: float | None = None,
    ) -> str:
        if not isinstance(plane, AxisWorkplane):
            raise ValueError("OrtoCut requires an AxisWorkplane placement")
        return self._to_factory_call_string(plane, world_size)

    def _to_factory_call_string(
        self,
        workplane: AxisWorkplane,
        world_size: float | None,
    ) -> str:
        normal_name = workplane.normal[-1]
        workplane_axis = WORKPLANE_BY_NORMAL[normal_name]

        normal_vec = axis_vector(workplane.normal)
        x_dir_vec = axis_vector(workplane.xDir)
        y_dir_vec = normal_vec.cross(x_dir_vec)
        y_axis = axis_name_from_vector(y_dir_vec)
        if y_axis is None:
            raise ValueError("Could not resolve workplane yDir to a global axis")
        source_y_dir = signed_axis(*y_axis)

        local_sketch = self.sketch.to_string(f"cq.Workplane('{workplane_axis}')")
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

        point_expr = _format_point(tuple(float(value) for value in workplane.point))
        extent = self._effective_extent(world_size)
        args = (
            f"r, {point_expr}, {workplane_axis!r}, "
            f"{_double_quoted_string(sketch_arg)}, {_format_number(extent)}"
        )
        return f"r=orto_cut({args})\n"

    def _effective_extent(self, world_size: float | None) -> float:
        if self.through_all:
            size = world_size if world_size is not None else self.world_size
            if size is not None:
                return 2.0 * float(size) + 1.0
        return self.extent

    def transform(self, shift: list[float], scale: float) -> None:
        if not self.through_all:
            self.extent *= scale

    def round(self) -> None:
        self.sketch.round()
        if self.through_all and self.world_size is not None:
            self.extent = 2.0 * float(self.world_size) + 1.0
        else:
            sign = -1 if self.extent < 0 else 1
            self.extent = round(self.extent)
            if np.allclose(self.extent, 0):
                self.extent = sign

    def fix(self) -> None:
        self.sketch.fix()
        self.sketch.wires = [wire for wire in self.sketch.wires if wire["outer"]]

    def to_dict(self) -> dict:
        return {
            "type": "OrtoCut",
            "sketch": self.sketch.to_dict(),
            "extent": self.extent,
            "through_all": self.through_all,
            "world_size": self.world_size,
        }

    @staticmethod
    def from_dict(entity: dict) -> "OrtoCut":
        assert (
            entity["type"] == "OrtoCut"
        ), f"Trying to build OrtoCut from type {entity['type']}"
        return OrtoCut(
            Sketch.from_dict(entity["sketch"]),
            entity["extent"],
            through_all=entity.get("through_all", False),
            world_size=entity.get("world_size"),
        )


class OrtoCutFactory(BaseFactory):
    MIN_EXTRUDE_HEIGHT = 5.0
    MIN_TRIANGLE_HEIGHT = 4.0
    MIN_ARC_HEIGHT = 5.0
    MIN_X_SPAN = 10.0
    VERTICAL_SEGMENT_PROBABILITY = 0.5

    def __init__(
        self,
        sketch_factory: SketchFactory | dict | None = None,
        inner_cut_probability: float = 0.5,
        through_all_probability: float | None = None,
        mesh_deflection: float = 0.005,
    ):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)
        self.sketch_factory = sketch_factory or SketchFactory(
            min_n_commands=1,
            max_n_commands=4,
            n_outer_probabilities=[1.0],
            rotation_probability=0.0,
            array_pattern_probability=0.0,
        )
        self.inner_cut_probability = inner_cut_probability
        self.through_all_probability = through_all_probability
        self.mesh_deflection = mesh_deflection

    def generate(self) -> OrtoCut | None:
        return None

    @staticmethod
    def _world_length_to_generation_length(
        length: float,
        world_size: float | None,
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

    def prepare_existing_sampler(
        self,
        cad_object: cq.Workplane | cq.Shape,
    ) -> PreparedSurfaceSampler:
        sampler = PreparedSurfaceSampler.from_cad_object(
            cad_object,
            mesh_deflection=self.mesh_deflection,
        )
        workplane = (
            cad_object
            if isinstance(cad_object, cq.Workplane)
            else cq.Workplane("XY").add(sampler.shape)
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
                logger.debug(
                    "Skipping face during orto_cut outer-face filter",
                    exc_info=True,
                )
        if not outer_indices:
            raise ValueError("OrtoCutFactory found no outer faces to sample")
        return PreparedSurfaceSampler(
            shape=sampler.shape,
            surface_compound=sampler.surface_compound,
            faces=[sampler.faces[i] for i in outer_indices],
            areas=[sampler.areas[i] for i in outer_indices],
            triangles=[sampler.triangles[i] for i in outer_indices],
        )

    @staticmethod
    def _x_dir_for_inside_y(
        normal: PositiveAxisName,
        site: SampledSite,
    ) -> str:
        inward = axis_vector(signed_axis(site.normal_axis, -site.normal_sign))
        normal_vec = axis_vector(normal)
        x_dir = inward.cross(normal_vec)
        x_axis = axis_name_from_vector(x_dir)
        if x_axis is None:
            raise ValueError("Could not resolve orto_cut xDir")
        return signed_axis(*x_axis)

    def _sample_placement(self, site: SampledSite) -> OrtoCutPlacement:
        normal = np.random.choice(orthogonal_axes(site.normal_axis)).item()
        x_dir = self._x_dir_for_inside_y(normal, site)
        workplane = AxisWorkplane(
            point=site.point.toTuple(),
            normal=normal,
            xDir=x_dir,
        )
        return OrtoCutPlacement(site, workplane)

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
            "z": cq.Vector(0, 0, 1),
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
            t0 = start.x if axis == "x" else start.z
            t1 = end.x if axis == "x" else end.z
            lo, hi = sorted((float(t0), float(t1)))
            if lo <= 1e-7 and hi >= -1e-7 and hi - lo > 1e-7:
                candidates.append((-lo, hi))

        if candidates:
            return max(candidates, key=lambda item: item[0] + item[1])

        if axis == "x":
            return max(0.0, -float(bbox.xmin)), max(0.0, float(bbox.xmax))
        return max(0.0, -float(bbox.zmin)), max(0.0, float(bbox.zmax))

    @staticmethod
    def _face_axis_distances(
        face: cq.Face,
        workplane: cq.Workplane,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        local_face = face.transformShape(workplane.plane.fG)
        x_distances = OrtoCutFactory._axis_distances_from_local_face(local_face, "x")
        normal_distances = OrtoCutFactory._axis_distances_from_local_face(
            local_face,
            "z",
        )
        if (
            min(x_distances) <= 0
            or max(normal_distances) <= 0
            or min(normal_distances) <= 0
        ):
            raise ValueError("Selected face has too little orto_cut room")
        return normal_distances, x_distances

    @staticmethod
    def _inside_depth_to_next_face(
        cad_object: cq.Workplane | cq.Shape,
        point: tuple[float, float, float],
        inside_direction: cq.Vector,
    ) -> float:
        tolerance = 1e-7
        shape = shape_from_cad_object(cad_object)
        direction = inside_direction.normalized()
        p_start = cq.Vector(*point).add(direction.multiply(tolerance))
        p_end = cq.Vector(*point).add(direction.multiply(10000))
        probe_ray = cq.Edge.makeLine(p_start, p_end)
        intersection = shape.intersect(probe_ray)

        segments = []
        for edge in intersection.Edges():
            length = edge.Length()
            if length <= tolerance:
                continue
            dist = min(
                (edge.startPoint() - p_start).Length,
                (edge.endPoint() - p_start).Length,
            )
            segments.append((dist, length))
        if not segments:
            raise ValueError("No body intersection found along inward face normal")
        return float(min(segments, key=lambda item: item[0])[1])

    @staticmethod
    def _sample_capped_uniform(
        low: float,
        high: float,
        cap: float | None = None,
    ) -> float:
        if high <= low:
            raise ValueError(f"Invalid sampling range: low={low}, high={high}")
        value = float(np.random.uniform(low, high))
        if cap is not None:
            value = min(value, cap)
        if value <= 0:
            raise ValueError("Sampled non-positive orto_cut dimension")
        return value

    @staticmethod
    def _polygon_sketch(vertices: list[list[float]], edges: list[dict]) -> Sketch:
        return Sketch(
            [
                {
                    "type": "polygon",
                    "outer": True,
                    "vertices": vertices,
                    "edges": edges,
                }
            ]
        )

    @staticmethod
    def _line_edges(n_edges: int) -> list[dict]:
        return [{"type": "line"} for _ in range(n_edges)]

    @staticmethod
    def _depth_ceiling(depth_len: float) -> float:
        ceiling = 0.95 * float(depth_len)
        if ceiling <= 0:
            raise ValueError("Selected face has non-positive orto_cut depth")
        return ceiling

    @staticmethod
    def _sample_middle_x(left_x: float, right_x: float) -> float:
        position = np.random.choice(
            ["between", "left", "right"],
            p=[0.5, 0.25, 0.25],
        )
        if position == "left":
            return left_x
        if position == "right":
            return right_x
        return float(np.random.uniform(left_x, right_x))

    def _vertical_offset(self, peak_y: float, depth_len: float) -> float | None:
        if np.random.random() >= self.VERTICAL_SEGMENT_PROBABILITY:
            return None
        max_offset = self._depth_ceiling(depth_len) - peak_y
        if max_offset <= 1e-9:
            return None
        offset = float(np.random.uniform(0.0, max_offset))
        if offset <= 1e-9:
            offset = 0.5 * max_offset
        return offset

    def _generate_triangle_sketch(
        self,
        x_span: float,
        depth_len: float,
        world_size: float | None,
    ) -> Sketch:
        left_x = -0.5 * x_span
        right_x = 0.5 * x_span
        min_y = self._world_length_to_generation_length(
            self.MIN_TRIANGLE_HEIGHT,
            world_size,
        )
        apex_y = self._sample_capped_uniform(min_y, self._depth_ceiling(depth_len))
        apex_x = self._sample_middle_x(left_x, right_x)
        offset = self._vertical_offset(apex_y, depth_len)
        if offset is None:
            return self._polygon_sketch(
                [[left_x, 0.0], [apex_x, apex_y], [right_x, 0.0]],
                self._line_edges(3),
            )

        return self._polygon_sketch(
            [
                [left_x, 0.0],
                [left_x, offset],
                [apex_x, apex_y + offset],
                [right_x, offset],
                [right_x, 0.0],
            ],
            self._line_edges(5),
        )

    def _generate_arc_sketch(
        self,
        x_span: float,
        depth_len: float,
        world_size: float | None,
    ) -> Sketch:
        half_span = 0.5 * x_span
        min_y = self._world_length_to_generation_length(
            self.MIN_ARC_HEIGHT,
            world_size,
        )
        cap = min(half_span, self._depth_ceiling(depth_len))
        if cap < min_y:
            raise ValueError("Selected face has too little orto_cut depth")
        arc_y = self._sample_capped_uniform(min_y, 1.2 * half_span, cap=cap)
        offset = self._vertical_offset(arc_y, depth_len)
        if offset is None:
            return self._polygon_sketch(
                [[-half_span, 0.0], [half_span, 0.0]],
                [
                    {"type": "arc", "point": [0.0, arc_y]},
                    {"type": "line"},
                ],
            )

        return self._polygon_sketch(
            [
                [-half_span, 0.0],
                [-half_span, offset],
                [half_span, offset],
                [half_span, 0.0],
            ],
            [
                {"type": "line"},
                {"type": "arc", "point": [0.0, arc_y + offset]},
                {"type": "line"},
                {"type": "line"},
            ],
        )

    def _generate_rect_sketch(self, x_span: float, y_span: float) -> Sketch:
        half_span = 0.5 * x_span
        return self._polygon_sketch(
            [
                [-half_span, 0.0],
                [half_span, 0.0],
                [half_span, y_span],
                [-half_span, y_span],
            ],
            self._line_edges(4),
        )

    def _generate_orto_sketch(
        self,
        x_span: float,
        depth_len: float,
        world_size: float | None,
    ) -> tuple[str, Sketch]:
        kind = str(np.random.choice(["triangle", "arc"]))
        if kind == "triangle":
            return kind, self._generate_triangle_sketch(
                x_span,
                depth_len,
                world_size,
            )
        return kind, self._generate_arc_sketch(x_span, depth_len, world_size)

    def _sample_x_span(self, x_dir_len: float, world_size: float | None) -> float:
        min_x_span = self._world_length_to_generation_length(
            self.MIN_X_SPAN,
            world_size,
        )
        return self._sample_capped_uniform(min_x_span, 1.96 * x_dir_len)

    def _sample_cylinder_rect_dimensions(
        self,
        cylinder_diameter: float,
        depth_len: float,
    ) -> tuple[float, float]:
        if cylinder_diameter <= 0:
            raise ValueError("Cylinder diameter must be positive for rect orto_cut")

        x_span = self._sample_capped_uniform(
            0.3 * cylinder_diameter,
            1.2 * cylinder_diameter,
        )
        max_y_span = min(0.25 * cylinder_diameter, self._depth_ceiling(depth_len))
        if max_y_span <= 0:
            raise ValueError("Cylinder rect orto_cut has non-positive y span")
        min_y_span = min(0.05 * cylinder_diameter, 0.5 * max_y_span)
        y_span = self._sample_capped_uniform(min_y_span, max_y_span)
        return x_span, y_span

    def _generate_outer_sketch(
        self,
        *,
        site: SampledSite,
        x_span: float,
        depth_len: float,
        world_size: float | None,
    ) -> tuple[str, Sketch]:
        if (
            site.face_kind == "cylinder"
            and site.cylinder_radius is not None
            and np.random.random() < 0.5
        ):
            cylinder_diameter = 2.0 * abs(float(site.cylinder_radius))
            rect_x_span, rect_y_span = self._sample_cylinder_rect_dimensions(
                cylinder_diameter,
                depth_len,
            )
            return "rect", self._generate_rect_sketch(rect_x_span, rect_y_span)

        return self._generate_orto_sketch(x_span, depth_len, world_size)

    def _cylinder_extrude_limit(self, site: SampledSite) -> float | None:
        if site.face_kind == "cylinder" and site.cylinder_radius is not None:
            return 2.0 * abs(float(site.cylinder_radius))
        return None

    def generate_on_existing(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        world_size: float,
        generation_world_half: float = 1.0,
    ) -> tuple[OrtoCut, AxisWorkplane]:
        site = sampler.sample_site()
        placement = self._sample_placement(site)
        workplane = workplane_at_closest_point(
            cad_object,
            placement.site.point,
            placement.workplane.normal,
            placement.workplane.xDir,
            surface_compound=sampler.surface_compound,
        )

        normal_distances, x_distances = self._face_axis_distances(site.face, workplane)
        cylinder_extrude_limit = self._cylinder_extrude_limit(site)
        x_dir_len = min(x_distances)
        inside_direction = axis_vector(
            signed_axis(site.normal_axis, -site.normal_sign)
        )
        depth_len = self._inside_depth_to_next_face(
            cad_object,
            site.point.toTuple(),
            inside_direction,
        )

        min_extrude_height = self._world_length_to_generation_length(
            self.MIN_EXTRUDE_HEIGHT,
            world_size,
        )
        min_x_span = self._world_length_to_generation_length(
            self.MIN_X_SPAN,
            world_size,
        )
        margin = self._world_length_to_generation_length(1.0, world_size)

        inner_cut = bool(np.random.random() < self.inner_cut_probability)
        if inner_cut:
            extrude_h_len = (
                cylinder_extrude_limit
                if cylinder_extrude_limit is not None
                else min(normal_distances)
            )
            extrude_h = self._sample_capped_uniform(
                min_extrude_height,
                extrude_h_len,
            )
            extrude_h *= float(np.random.choice([-1.0, 1.0]))
            x_span = self._sample_capped_uniform(min_x_span, 0.98 * x_dir_len)
            sketch_kind, sketch = self._generate_orto_sketch(
                x_span,
                depth_len,
                world_size,
            )
        else:
            if cylinder_extrude_limit is not None:
                extrude_h_len = cylinder_extrude_limit
                extrude_sign = float(np.random.choice([-1.0, 1.0]))
            elif np.random.random() < 0.5:
                extrude_h_len = normal_distances[1]
                extrude_sign = 1.0
            else:
                extrude_h_len = normal_distances[0]
                extrude_sign = -1.0

            extrude_h_abs = extrude_h_len + margin
            x_span = x_dir_len + margin
            if np.random.random() < 0.75:
                if np.random.random() < 0.5:
                    extrude_h_abs = self._sample_capped_uniform(
                        min_extrude_height,
                        0.98 * extrude_h_len,
                    )
                else:
                    x_span = self._sample_capped_uniform(
                        min_x_span,
                        0.98 * x_dir_len,
                    )
            extrude_h = extrude_sign * extrude_h_abs
            sketch_kind, sketch = self._generate_outer_sketch(
                site=site,
                x_span=x_span,
                depth_len=depth_len,
                world_size=world_size,
            )

        op = OrtoCut(
            sketch=sketch,
            extent=extrude_h,
            through_all=False,
            world_size=world_size,
        )
        logger.info(
            "Generated orto_cut with extent=%s, inner_cut=%s, sketch=%s, normal=%s, xDir=%s",
            extrude_h,
            inner_cut,
            sketch_kind,
            placement.workplane.normal,
            placement.workplane.xDir,
        )
        return op, placement.workplane

    @staticmethod
    def from_dict(entity: dict) -> "OrtoCutFactory":
        return OrtoCutFactory(
            sketch_factory=SketchFactory.from_dict(entity["sketch_factory"])
            if entity.get("sketch_factory")
            else None,
            inner_cut_probability=entity.get("inner_cut_probability", 0.5),
            through_all_probability=entity.get("through_all_probability"),
            mesh_deflection=entity.get("mesh_deflection", 0.005),
        )

    def to_dict(self) -> dict:
        return {
            "type": "OrtoCutFactory",
            "sketch_factory": self.sketch_factory.to_dict()
            if self.sketch_factory is not None
            else None,
            "inner_cut_probability": self.inner_cut_probability,
            "mesh_deflection": self.mesh_deflection,
        }


factories.register("orto_cut", OrtoCutFactory)
