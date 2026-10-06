import math

import cadquery as cq
import numpy as np

# from cadquery_addons import *

from .extrude import (
    Extrude,
    ExtrudeFactory,
    WORKPLANE_OFFSET_INDEX,
    _double_quoted_string,
    _format_number,
    _format_point,
    _join_workplane_and_sketch, _surface_translation,
)
from .hole import _dominant_axis_selector
from .registry import factories
from .sketch import Sketch, SketchFactory
from .surface_sampling import (
    AxisWorkplane,
    PreparedSurfaceSampler,
    closest_surface_point_from_compound,
    face_kind_name,
    precompute_surface_compound,
    shape_from_cad_object,
)


WORKPLANE_NORMAL_AXIS = {
    "XY": "Z",
    "YZ": "X",
    "ZX": "Y",
}


def shell(
    r: cq.Workplane | cq.Shape | None,
    point: tuple[float, float, float],
    workplane: str,
    sketch: str,
    extrude_height: float,
    wall_thickness: float,
    bottom_thickness: float,
):
    if workplane not in WORKPLANE_OFFSET_INDEX:
        raise ValueError("workplane must be one of 'XY', 'YZ', or 'ZX'")

    point = tuple(float(value) for value in point)
    extrude_height = float(extrude_height)
    wall_thickness = float(wall_thickness)
    bottom_thickness = max(0.0, float(bottom_thickness))

    if r is None:
        axis_point = point
        offset = axis_point[WORKPLANE_OFFSET_INDEX[workplane]]
        # axis = WORKPLANE_NORMAL_AXIS[workplane]
        shape = None
        surface_point = None
    else:
        shape = shape_from_cad_object(r)
        surface_compound = precompute_surface_compound(shape)
        surface_point = closest_surface_point_from_compound(surface_compound, point)
        axis_point = tuple(float(value) for value in surface_point.toTuple())
        offset = axis_point[WORKPLANE_OFFSET_INDEX[workplane]]
        # _, axis = _dominant_axis_selector(shape, surface_point)

    prefix = f"cq.Workplane('{workplane}').workplane(offset={offset})"
    profile = eval(_join_workplane_and_sketch(prefix, sketch), {"cq": cq})

    axis = WORKPLANE_NORMAL_AXIS[workplane]

    shelled = (
        profile.extrude(extrude_height)
        .faces(f">{axis} or <{axis}")
        .shell(-wall_thickness, kind="intersection")
    )

    if bottom_thickness > 0.000001:
        bottom_extrude_height = math.copysign(bottom_thickness, extrude_height)
        bottom = profile.extrude(bottom_extrude_height)
        shelled = shelled.union(bottom)

    if shape is not None and surface_point is not None:
        shelled = shelled.translate(
            _surface_translation(shape, surface_point, 200)
        )

    if r is None:
        return cq.Workplane(workplane).add(shelled)
    if isinstance(r, cq.Shape):
        return cq.Workplane(workplane).add(r).union(shelled)
    return r.union(shelled)


class Shell(Extrude):
    def __init__(
        self,
        sketch: Sketch,
        extent: float,
        wall_thickness: float = 0.01,
        bottom_thickness: float = 0.0,
    ):
        super().__init__(sketch, extent)
        self.wall_thickness = float(wall_thickness)
        self.bottom_thickness = max(0.0, float(bottom_thickness))

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
            f"{_double_quoted_string(sketch_arg)}, {_format_number(extent)}, "
            f"{_format_number(self.wall_thickness)}, "
            f"{_format_number(self.bottom_thickness)}"
        )
        return f"shell({args})\n"

    def transform(self, shift: list[float], scale: float) -> None:
        super().transform(shift, scale)
        scale = abs(float(scale))
        self.wall_thickness *= scale
        self.bottom_thickness *= scale

    def round(self) -> None:
        super().round()
        self.wall_thickness = round(self.wall_thickness)
        self.bottom_thickness = round(self.bottom_thickness)

    def to_dict(self) -> dict:
        return {
            "type": "Shell",
            "sketch": self.sketch.to_dict(),
            "extent": self.extent,
            "wall_thickness": self.wall_thickness,
            "bottom_thickness": self.bottom_thickness,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Shell":
        assert (
            entity["type"] == "Shell"
        ), f"Trying to build Shell from type {entity['type']}"
        return Shell(
            Sketch.from_dict(entity["sketch"]),
            entity["extent"],
            entity.get("wall_thickness", 0.01),
            entity.get("bottom_thickness", 0.0),
        )


class ShellFactory(ExtrudeFactory):
    SIMPLE_SKETCH_RESAMPLE_ATTEMPTS = 5

    def __init__(
        self,
        sketch_factory: SketchFactory | dict,
        tangent_plane_prob: float = 0.5,
        out_of_face_prob: float = 0.5,
        mesh_deflection: float = 0.005,
        wall_thickness: float | None = None,
        bottom_thickness: float | None = None,
        bottom_probability: float | None = None,
        world_size: float = 200,
        wall_thickness_max_fraction: float = 0.1,
        bottom_thickness_max_fraction: float = 0.15,
    ):
        super().__init__(
            sketch_factory=sketch_factory,
            tangent_plane_prob=tangent_plane_prob,
            out_of_face_prob=out_of_face_prob,
            mesh_deflection=mesh_deflection,
        )
        self.wall_thickness = (
            None if wall_thickness is None else max(0.0, float(wall_thickness))
        )
        self.bottom_thickness = (
            None if bottom_thickness is None else max(0.0, float(bottom_thickness))
        )
        if bottom_probability is None:
            bottom_probability = (
                1.0
                if self.bottom_thickness is not None and self.bottom_thickness > 0
                else 0.0
            )
        self.bottom_probability = float(np.clip(bottom_probability, 0.0, 1.0))
        self.world_size = float(world_size)
        self.wall_thickness_max_fraction = float(wall_thickness_max_fraction)
        self.bottom_thickness_max_fraction = float(bottom_thickness_max_fraction)

    def prepare_existing_sampler(
        self,
        cad_object: cq.Workplane | cq.Shape,
    ) -> PreparedSurfaceSampler:
        sampler = super().prepare_existing_sampler(cad_object)
        planar_indices = [
            i for i, face in enumerate(sampler.faces) if face_kind_name(face) == "plane"
        ]
        if not planar_indices:
            raise ValueError("ShellFactory found no planar faces to sample")
        return PreparedSurfaceSampler(
            shape=sampler.shape,
            surface_compound=sampler.surface_compound,
            faces=[sampler.faces[i] for i in planar_indices],
            areas=[sampler.areas[i] for i in planar_indices],
            triangles=[sampler.triangles[i] for i in planar_indices],
        )

    @staticmethod
    def _sketch_has_circle_or_rect(sketch: Sketch) -> bool:
        sketch_string = sketch.to_string("cq.Workplane('XY')")
        return "circle" in sketch_string or "rect" in sketch_string

    def _sample_shell_sketch(self, generate_sketch) -> Sketch:
        sketch = None
        for _ in range(self.SIMPLE_SKETCH_RESAMPLE_ATTEMPTS):
            sketch = generate_sketch()
            if not self._sketch_has_circle_or_rect(sketch):
                return sketch
        assert sketch is not None
        return sketch

    def _with_shell_sketch_sampling(self, callback):
        original_generate = self.sketch_factory.generate

        def generate_shell_sketch(*args, **kwargs):
            return self._sample_shell_sketch(
                lambda: original_generate(*args, **kwargs)
            )

        self.sketch_factory.generate = generate_shell_sketch
        try:
            return callback()
        finally:
            self.sketch_factory.generate = original_generate

    @staticmethod
    def _minimum_thickness(world_size: float) -> float:
        world_size = float(world_size)
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        return 2.0 / world_size

    @staticmethod
    def _sketch_min_bbox_side(sketch: Sketch) -> float:
        bbox = cq.Shape(sketch.to_shape()).BoundingBox()
        return min(float(bbox.xmax - bbox.xmin), float(bbox.ymax - bbox.ymin))

    @staticmethod
    def _sample_thickness(min_thickness: float, max_thickness: float, label: str) -> float:
        if max_thickness < min_thickness:
            raise ValueError(
                f"{label} max thickness {max_thickness:g} is smaller than "
                f"minimum {min_thickness:g}"
            )
        if np.isclose(max_thickness, min_thickness):
            return float(min_thickness)
        return float(np.random.uniform(min_thickness, max_thickness))

    def _sample_wall_thickness(self, sketch: Sketch, world_size: float) -> float:
        if self.wall_thickness is not None:
            return self.wall_thickness
        min_thickness = self._minimum_thickness(world_size)
        max_thickness = (
            self.wall_thickness_max_fraction * self._sketch_min_bbox_side(sketch)
        )
        return self._sample_thickness(min_thickness, max_thickness, "Wall")

    def _sample_bottom_thickness(self, extent: float, world_size: float) -> float:
        if np.random.random() >= self.bottom_probability:
            return 0.0
        if self.bottom_thickness is not None:
            return self.bottom_thickness
        min_thickness = self._minimum_thickness(world_size)
        max_thickness = self.bottom_thickness_max_fraction * abs(float(extent))
        return self._sample_thickness(min_thickness, max_thickness, "Bottom")

    def generate(self) -> Shell:
        def build_shell():
            extrude = ExtrudeFactory.generate(self)
            wall_thickness = self._sample_wall_thickness(
                extrude.sketch,
                self.world_size,
            )
            return Shell(
                extrude.sketch,
                extrude.extent,
                wall_thickness,
                self._sample_bottom_thickness(extrude.extent, self.world_size),
            )

        return self._with_shell_sketch_sampling(build_shell)

    def generate_on_existing(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        world_size: float,
        generation_world_half: float = 1.0,
    ) -> tuple[Shell, AxisWorkplane]:
        def build_shell():
            extrude, workplane = ExtrudeFactory.generate_on_existing(
                self,
                cad_object,
                sampler,
                world_size,
                generation_world_half,
            )
            wall_thickness = self._sample_wall_thickness(extrude.sketch, world_size)
            return (
                Shell(
                    extrude.sketch,
                    extrude.extent,
                    wall_thickness,
                    self._sample_bottom_thickness(extrude.extent, world_size),
                ),
                workplane,
            )

        return self._with_shell_sketch_sampling(build_shell)

    @staticmethod
    def from_dict(entity: dict) -> "ShellFactory":
        return ShellFactory(
            SketchFactory.from_dict(entity["sketch_factory"]),
            tangent_plane_prob=entity.get("tangent_plane_prob", 0.5),
            out_of_face_prob=entity.get("out_of_face_prob", 0.5),
            mesh_deflection=entity.get("mesh_deflection", 0.005),
            wall_thickness=entity.get("wall_thickness"),
            bottom_thickness=entity.get("bottom_thickness"),
            bottom_probability=entity.get("bottom_probability"),
            world_size=entity.get("world_size", 200),
            wall_thickness_max_fraction=entity.get(
                "wall_thickness_max_fraction", 0.1
            ),
            bottom_thickness_max_fraction=entity.get(
                "bottom_thickness_max_fraction", 0.15
            ),
        )

    def to_dict(self) -> dict:
        return {
            "type": "ShellFactory",
            "sketch_factory": self.sketch_factory.to_dict(),
            "tangent_plane_prob": self.tangent_plane_prob,
            "out_of_face_prob": self.out_of_face_prob,
            "mesh_deflection": self.mesh_deflection,
            "wall_thickness": self.wall_thickness,
            "bottom_thickness": self.bottom_thickness,
            "bottom_probability": self.bottom_probability,
            "world_size": self.world_size,
            "wall_thickness_max_fraction": self.wall_thickness_max_fraction,
            "bottom_thickness_max_fraction": self.bottom_thickness_max_fraction,
        }


factories.register("shell", ShellFactory)
