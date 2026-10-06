from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from typing import Sequence

import cadquery as cq
import numpy as np

# from cadquery_addons import *

from .base import BaseFactory, BaseOperation
from .extrude import (
    WORKPLANE_BY_NORMAL,
    ExtrudeFactory,
    _double_quoted_string,
    _format_number,
    _format_point,
    _surface_translation,
)
from .orto_cut import OrtoCutFactory
from .registry import factories
from .sketch import SketchFactory
from .sketch_global_coords import (
    AXIS_BASIS,
    sketch_to_global_coords,
    transform_sketch_local_axes,
)
from .surface_sampling import (
    AxisName,
    AxisWorkplane,
    PositiveAxisName,
    PreparedSurfaceSampler,
    SampledSite,
    axis_name_from_vector,
    axis_vector,
    closest_surface_point_from_compound,
    orthogonal_axes,
    precompute_surface_compound,
    shape_from_cad_object,
    signed_axis,
    workplane_at_closest_point,
)

logger = logging.getLogger(__name__)


PROFILE_KIND_PROBABILITIES = ("circle", "circle_hole", "rect", "rect_hole")
OFFSET_CALL_RE = re.compile(r"\.offset\(")
MIN_PROFILE_CIRCLE_RADIUS = 3.0
MIN_PROFILE_RECT_SIDE = 5.0
MIN_PROFILE_HOLE_MARGIN = 1.0


@dataclass
class _ParsedCall:
    name: str
    args: list[ast.expr]
    keywords: list[ast.keyword]


def _normalize_offset_calls(source: str) -> str:
    return OFFSET_CALL_RE.sub(".workplane(offset=", source)


def _display_offset_calls(source: str) -> str:
    return source.replace(".workplane(offset=", ".offset(")


def _literal_number(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    raise ValueError("Expected a literal number")


def _literal_point2(node: ast.AST) -> tuple[float, float]:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 2:
        raise ValueError("Expected a 2D point literal")
    return (_literal_number(node.elts[0]), _literal_number(node.elts[1]))


def _literal_point3(node: ast.AST) -> tuple[float, float, float]:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 3:
        raise ValueError("Expected a 3D point literal")
    return (
        _literal_number(node.elts[0]),
        _literal_number(node.elts[1]),
        _literal_number(node.elts[2]),
    )


def _flatten_chain(node: ast.AST) -> tuple[ast.AST, list[_ParsedCall]]:
    calls: list[_ParsedCall] = []
    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        calls.append(
            _ParsedCall(current.func.attr, list(current.args), list(current.keywords))
        )
        current = current.func.value
    calls.reverse()
    return current, calls


def _parse_workplane_base(
    base: ast.AST,
    calls: list[_ParsedCall],
) -> tuple[str, list[_ParsedCall]]:
    if (
        isinstance(base, ast.Call)
        and isinstance(base.func, ast.Attribute)
        and isinstance(base.func.value, ast.Name)
        and base.func.value.id == "cq"
        and base.func.attr == "Workplane"
        and base.args
        and isinstance(base.args[0], ast.Constant)
        and isinstance(base.args[0].value, str)
    ):
        axis = base.args[0].value
        if axis not in AXIS_BASIS:
            raise ValueError(f"Unsupported workplane axis {axis!r}")
        return axis, calls
    if (
        isinstance(base, ast.Name)
        and base.id == "cq"
        and calls
        and calls[0].name == "Workplane"
        and calls[0].args
        and isinstance(calls[0].args[0], ast.Constant)
        and isinstance(calls[0].args[0].value, str)
    ):
        axis = calls[0].args[0].value
        if axis not in AXIS_BASIS:
            raise ValueError(f"Unsupported workplane axis {axis!r}")
        return axis, calls[1:]
    raise ValueError("Path must start with cq.Workplane('<axis>')")


def _call_offset(call: _ParsedCall) -> float | None:
    if call.name != "workplane":
        return None
    for keyword in call.keywords:
        if keyword.arg == "offset":
            return _literal_number(keyword.value)
    if call.args:
        return _literal_number(call.args[0])
    return None


def _local_to_global(
    workplane_axis: str,
    offset: float,
    point: tuple[float, float],
) -> tuple[float, float, float]:
    x_dir, y_dir, z_dir = AXIS_BASIS[workplane_axis]
    u, v = point
    return tuple(
        offset * z_dir[i] + u * x_dir[i] + v * y_dir[i] for i in range(3)
    )


def _edge_from_points(
    kind: str,
    workplane_axis: str,
    offset: float,
    start: tuple[float, float],
    end: tuple[float, float],
    mid: tuple[float, float] | None = None,
) -> cq.Edge:
    start_v = cq.Vector(*_local_to_global(workplane_axis, offset, start))
    end_v = cq.Vector(*_local_to_global(workplane_axis, offset, end))
    if kind == "segment":
        return cq.Edge.makeLine(start_v, end_v)
    if mid is None:
        raise ValueError("Arc edge is missing a midpoint")
    mid_v = cq.Vector(*_local_to_global(workplane_axis, offset, mid))
    return cq.Edge.makeThreePointArc(start_v, mid_v, end_v)


def _parse_make_line(path: str) -> tuple[cq.Edge, tuple[float, float, float]]:
    expr = ast.parse(path.strip(), mode="eval").body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        raise ValueError("makeLine path must be a function call")
    if expr.func.id != "makeLine" or len(expr.args) != 2:
        raise ValueError("makeLine path must have exactly two point arguments")
    p1 = _literal_point3(expr.args[0])
    p2 = _literal_point3(expr.args[1])
    return cq.Edge.makeLine(cq.Vector(*p1), cq.Vector(*p2)), p1


def _path_wire_from_sketch(path: str) -> tuple[cq.Edge | cq.Wire, tuple[float, float, float]]:
    module = ast.parse(_normalize_offset_calls(path.strip()), mode="eval")
    base, calls = _flatten_chain(module.body)
    workplane_axis, calls = _parse_workplane_base(base, calls)
    offset = 0.0
    in_sketch = False
    current_point: tuple[float, float] | None = None
    first_point: tuple[float, float] | None = None
    edges: list[cq.Edge] = []

    for call in calls:
        call_offset = _call_offset(call)
        if call_offset is not None and not in_sketch:
            offset = call_offset
            continue
        if call.name == "sketch":
            in_sketch = True
            continue
        if not in_sketch:
            continue
        if call.name in {"assemble", "finalize", "val"}:
            continue
        if call.name == "segment":
            if len(call.args) >= 2:
                start = _literal_point2(call.args[0])
                end = _literal_point2(call.args[1])
            elif len(call.args) >= 1 and current_point is not None:
                start = current_point
                end = _literal_point2(call.args[0])
            else:
                raise ValueError("segment requires a start point or previous point")
            edges.append(_edge_from_points("segment", workplane_axis, offset, start, end))
        elif call.name in {"arc", "threePointArc"}:
            if len(call.args) >= 3:
                start = _literal_point2(call.args[0])
                mid = _literal_point2(call.args[1])
                end = _literal_point2(call.args[2])
            elif len(call.args) >= 2 and current_point is not None:
                start = current_point
                mid = _literal_point2(call.args[0])
                end = _literal_point2(call.args[1])
            else:
                raise ValueError("arc requires a start point or previous point")
            edges.append(
                _edge_from_points("arc", workplane_axis, offset, start, end, mid)
            )
        else:
            raise ValueError(f"Unsupported path sketch call {call.name!r}")
        if first_point is None:
            first_point = start
        current_point = end

    if not edges or first_point is None:
        raise ValueError("Path sketch must contain at least one segment or arc")
    start_global = _local_to_global(workplane_axis, offset, first_point)
    if len(edges) == 1:
        return edges[0], start_global
    return cq.Wire.assembleEdges(edges), start_global


def _path_3d(path: str) -> tuple[cq.Edge | cq.Wire, tuple[float, float, float]]:
    path = path.strip()
    if path.startswith("makeLine"):
        return _parse_make_line(path)
    return _path_wire_from_sketch(path)


def sweep(
    r: cq.Workplane | cq.Shape | None,
    profile: str,
    path: str,
    world_size: float | None = 200,
):
    path_3d, path_start = _path_3d(path)
    profile_r = eval(_normalize_offset_calls(profile), {"cq": cq})
    sweep_body = profile_r.sweep(path_3d, makeSolid=True, isFrenet=True)
    if r is None:
        return sweep_body

    shape = shape_from_cad_object(r)
    surface_compound = precompute_surface_compound(shape)
    surface_point = closest_surface_point_from_compound(surface_compound, path_start)
    sweep_body = sweep_body.translate(
        _surface_translation(shape, surface_point, world_size or 200)
    )
    base = r if isinstance(r, cq.Workplane) else cq.Workplane("XY").add(shape)
    return base.union(sweep_body)


def _workplane_source_y_dir(workplane: AxisWorkplane) -> AxisName:
    normal_vec = axis_vector(workplane.normal)
    x_dir_vec = axis_vector(workplane.xDir)
    y_dir_vec = normal_vec.cross(x_dir_vec)
    y_axis = axis_name_from_vector(y_dir_vec)
    if y_axis is None:
        raise ValueError("Could not resolve workplane yDir to a global axis")
    return signed_axis(*y_axis)


def _globalize_workplane_sketch(workplane: AxisWorkplane, sketch_expr: str) -> str:
    workplane_axis = WORKPLANE_BY_NORMAL[workplane.normal[-1]]
    aligned = transform_sketch_local_axes(
        sketch_expr,
        workplane.xDir,
        _workplane_source_y_dir(workplane),
        workplane_axis,
    )
    globalized = sketch_to_global_coords(workplane_axis, workplane.point, aligned)
    return _display_offset_calls(globalized)


def _profile_expression(workplane: AxisWorkplane, profile: dict) -> str:
    workplane_axis = WORKPLANE_BY_NORMAL[workplane.normal[-1]]
    local = f"cq.Workplane('{workplane_axis}').sketch()"
    if profile["type"] == "circle":
        local += f".circle({_format_number(profile['radius'])})"
        if profile.get("inner_radius") is not None:
            local += f".circle({_format_number(profile['inner_radius'])}, mode='s')"
    elif profile["type"] == "rect":
        local += (
            f".rect({_format_number(profile['width'])}, "
            f"{_format_number(profile['height'])})"
        )
        if profile.get("inner_width") is not None and profile.get("inner_height") is not None:
            local += (
                f".rect({_format_number(profile['inner_width'])}, "
                f"{_format_number(profile['inner_height'])}, mode='s')"
            )
    else:
        raise ValueError(f"Unknown sweep profile type {profile['type']!r}")
    local += ".finalize()"
    return _globalize_workplane_sketch(workplane, local)


def _format_point2(point: Sequence[float]) -> str:
    return "(" + ", ".join(_format_number(float(value)) for value in point[:2]) + ")"


def _path_expression_from_edges(
    workplane: AxisWorkplane,
    local_edges: list[dict],
) -> str:
    if not local_edges:
        raise ValueError("Path sketch requires at least one edge")
    workplane_axis = WORKPLANE_BY_NORMAL[workplane.normal[-1]]
    local = f"cq.Workplane('{workplane_axis}').sketch()"
    current: tuple[float, float] | None = None
    for index, edge in enumerate(local_edges):
        start = tuple(float(value) for value in edge["start"])
        end = tuple(float(value) for value in edge["end"])
        if edge["type"] == "line":
            if index == 0 or current is None:
                local += f".segment({_format_point2(start)}, {_format_point2(end)})"
            else:
                local += f".segment({_format_point2(end)})"
        else:
            mid = tuple(float(value) for value in edge["mid"])
            if index == 0 or current is None:
                local += (
                    f".arc({_format_point2(start)}, {_format_point2(mid)}, "
                    f"{_format_point2(end)})"
                )
            else:
                local += f".arc({_format_point2(mid)}, {_format_point2(end)})"
        current = end
    return _globalize_workplane_sketch(workplane, local)


class Sweep(BaseOperation):
    def __init__(
        self,
        profile_workplane: AxisWorkplane,
        profile: dict,
        *,
        path_kind: str,
        path_workplane: AxisWorkplane | None = None,
        path_edges: list[dict] | None = None,
        line_points: tuple[tuple[float, float, float], tuple[float, float, float]]
        | None = None,
    ):
        self.profile_workplane = profile_workplane
        self.profile = profile
        self.path_kind = path_kind
        self.path_workplane = path_workplane
        self.path_edges = path_edges
        self.line_points = line_points

    def profile_string(self) -> str:
        return _profile_expression(self.profile_workplane, self.profile)

    def path_string(self) -> str:
        if self.path_kind == "line":
            if self.line_points is None:
                raise ValueError("line sweep path is missing endpoints")
            p1, p2 = self.line_points
            return f"makeLine({_format_point(p1)}, {_format_point(p2)})"
        if self.path_workplane is None or self.path_edges is None:
            raise ValueError("sketch sweep path is missing workplane or edges")
        return _path_expression_from_edges(self.path_workplane, self.path_edges)

    def to_string(self, world_size: float | None = None) -> str:
        args = (
            f"r, {_double_quoted_string(self.profile_string())}, "
            f"{_double_quoted_string(self.path_string())}"
        )
        return f"r=sweep({args})\n"

    def to_shape(self) -> cq.Shape:
        path_3d, _ = _path_3d(self.path_string())
        profile_r = eval(_normalize_offset_calls(self.profile_string()), {"cq": cq})
        return shape_from_cad_object(
            profile_r.sweep(path_3d, makeSolid=True, isFrenet=True)
        )

    def transform(self, shift: list[float], scale: float, *args, **kwargs) -> None:
        self.profile_workplane.transform(shift, scale)
        if self.path_workplane is not None:
            self.path_workplane.transform(shift, scale)
        if self.line_points is not None:
            self.line_points = (
                tuple((self.line_points[0][i] + shift[i]) * scale for i in range(3)),
                tuple((self.line_points[1][i] + shift[i]) * scale for i in range(3)),
            )
        for key in (
            "radius",
            "inner_radius",
            "width",
            "height",
            "inner_width",
            "inner_height",
        ):
            if self.profile.get(key) is not None:
                self.profile[key] *= scale
        if self.path_edges is not None:
            for edge in self.path_edges:
                for key in ("start", "end", "mid"):
                    if key in edge and edge[key] is not None:
                        edge[key] = [value * scale for value in edge[key]]

    def round(self) -> None:
        self.profile_workplane.round()
        if self.path_workplane is not None:
            self.path_workplane.round()
        if self.line_points is not None:
            self.line_points = (
                tuple(round(value) for value in self.line_points[0]),
                tuple(round(value) for value in self.line_points[1]),
            )
        for key in (
            "radius",
            "inner_radius",
            "width",
            "height",
            "inner_width",
            "inner_height",
        ):
            if self.profile.get(key) is not None:
                self.profile[key] = max(1, round(self.profile[key]))
        if self.profile["type"] == "circle" and self.profile.get("inner_radius") is not None:
            if self.profile["radius"] <= 1:
                self.profile.pop("inner_radius", None)
            else:
                self.profile["inner_radius"] = min(
                    self.profile["inner_radius"],
                    self.profile["radius"] - 1,
                )
        if self.profile["type"] == "rect" and self.profile.get("inner_width") is not None:
            if self.profile["width"] <= 1 or self.profile["height"] <= 1:
                self.profile.pop("inner_width", None)
                self.profile.pop("inner_height", None)
            else:
                self.profile["inner_width"] = min(
                    self.profile["inner_width"],
                    self.profile["width"] - 1,
                )
                self.profile["inner_height"] = min(
                    self.profile["inner_height"],
                    self.profile["height"] - 1,
                )
        if self.path_edges is not None:
            for edge in self.path_edges:
                for key in ("start", "end", "mid"):
                    if key in edge and edge[key] is not None:
                        edge[key] = [round(value) for value in edge[key]]

    def fix(self) -> None:
        self.to_shape()

    def to_dict(self) -> dict:
        return {
            "type": "Sweep",
            "profile_workplane": self.profile_workplane.to_dict(),
            "profile": self.profile,
            "path_kind": self.path_kind,
            "path_workplane": self.path_workplane.to_dict()
            if self.path_workplane is not None
            else None,
            "path_edges": self.path_edges,
            "line_points": self.line_points,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Sweep":
        assert (
            entity["type"] == "Sweep"
        ), f"Trying to build Sweep from type {entity['type']}"
        return Sweep(
            AxisWorkplane.from_dict(entity["profile_workplane"]),
            dict(entity["profile"]),
            path_kind=entity["path_kind"],
            path_workplane=AxisWorkplane.from_dict(entity["path_workplane"])
            if entity.get("path_workplane") is not None
            else None,
            path_edges=entity.get("path_edges"),
            line_points=tuple(tuple(point) for point in entity["line_points"])
            if entity.get("line_points") is not None
            else None,
        )


@dataclass(frozen=True)
class SweepPlacement:
    site: SampledSite
    workplane: AxisWorkplane


class SweepFactory(BaseFactory):
    MIN_PATH_SPAN = 5.0

    def __init__(
        self,
        inclined_line_path_probability: float = 0.0,
        mesh_deflection: float = 0.005,
    ):
        self.inclined_line_path_probability = inclined_line_path_probability
        self.mesh_deflection = mesh_deflection
        self._orto_sampler = OrtoCutFactory(
            inner_cut_probability=1.0,
            mesh_deflection=mesh_deflection,
        )
        self._profile_sizer = ExtrudeFactory(
            SketchFactory(
                min_n_commands=1,
                max_n_commands=1,
                n_outer_probabilities=[1.0],
                rotation_probability=0.0,
                array_pattern_probability=0.0,
            ),
            tangent_plane_prob=1.0,
            out_of_face_prob=0.0,
            mesh_deflection=mesh_deflection,
        )

    def generate(
        self,
        world_size: float | None = 200,
        generation_world_half: float = 1.0,
    ) -> Sweep:
        point = tuple(
            float(
                np.random.uniform(
                    -generation_world_half / 4.0,
                    generation_world_half / 4.0,
                )
            )
            for _ in range(3)
        )
        profile_normal = np.random.choice(("X", "Y", "Z")).item()
        profile_workplane = AxisWorkplane(
            point=point,
            normal=profile_normal,
            xDir=ExtrudeFactory._standard_workplane_x_dir(profile_normal),
        )

        path_normal = np.random.choice(orthogonal_axes(profile_normal)).item()
        path_workplane = AxisWorkplane(
            point=point,
            normal=path_normal,
            xDir=self._x_dir_for_profile_normal(profile_normal, path_normal),
        )
        path_cq_workplane = self._cq_workplane(path_workplane)
        profile, profile_offset = self._sample_free_profile(
            path_cq_workplane,
            world_size or 200,
            generation_world_half,
        )

        if np.random.random() < self.inclined_line_path_probability:
            min_length = self._world_length_to_generation_length(
                self.MIN_PATH_SPAN,
                world_size,
            )
            p2 = self._sample_line_endpoint(
                cq.Vector(*point),
                path_workplane,
                generation_world_half,
                min_length,
            )
            return Sweep(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="line",
                line_points=(point, p2),
            )

        edges = self._sample_sketch_path(
            path_cq_workplane,
            profile_offset,
            world_size or 200,
            generation_world_half,
        )
        return Sweep(
            profile_workplane=profile_workplane,
            profile=profile,
            path_kind="sketch",
            path_workplane=path_workplane,
            path_edges=edges,
        )

    def prepare_existing_sampler(
        self,
        cad_object: cq.Workplane | cq.Shape,
    ) -> PreparedSurfaceSampler:
        return self._orto_sampler.prepare_existing_sampler(cad_object)

    @staticmethod
    def _x_dir_for_outside_y(
        normal: PositiveAxisName,
        site: SampledSite,
    ) -> AxisName:
        return SweepFactory._x_dir_for_profile_normal(
            signed_axis(site.normal_axis, site.normal_sign),
            normal,
        )

    @staticmethod
    def _x_dir_for_profile_normal(
        profile_normal: AxisName,
        path_normal: PositiveAxisName,
    ) -> AxisName:
        outward = axis_vector(profile_normal)
        normal_vec = axis_vector(path_normal)
        x_dir = outward.cross(normal_vec)
        x_axis = axis_name_from_vector(x_dir)
        if x_axis is None:
            raise ValueError("Could not resolve sweep xDir")
        return signed_axis(*x_axis)

    @staticmethod
    def _cq_workplane(workplane: AxisWorkplane) -> cq.Workplane:
        plane = cq.Plane(
            origin=cq.Vector(*workplane.point),
            xDir=axis_vector(workplane.xDir),
            normal=axis_vector(workplane.normal),
        )
        return cq.Workplane(plane)

    def _sample_placement(self, site: SampledSite) -> SweepPlacement:
        normal = np.random.choice(orthogonal_axes(site.normal_axis)).item()
        x_dir = self._x_dir_for_outside_y(normal, site)
        return SweepPlacement(
            site=site,
            workplane=AxisWorkplane(
                point=site.point.toTuple(),
                normal=normal,
                xDir=x_dir,
            ),
        )

    @staticmethod
    def _profile_workplane(site: SampledSite) -> AxisWorkplane:
        return AxisWorkplane(
            point=site.point.toTuple(),
            normal=site.normal_axis,
            xDir=ExtrudeFactory._standard_workplane_x_dir(site.normal_axis),
        )

    @staticmethod
    def _world_length_to_generation_length(
        length: float,
        world_size: float | None,
    ) -> float:
        if world_size is None:
            return length
        return length / world_size

    def _sample_profile(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        site: SampledSite,
        workplane: AxisWorkplane,
        world_size: float,
        generation_world_half: float,
    ) -> tuple[dict, float]:
        cq_workplane = workplane_at_closest_point(
            cad_object,
            site.point,
            workplane.normal,
            workplane.xDir,
            surface_compound=sampler.surface_compound,
        )
        kind = np.random.choice(PROFILE_KIND_PROBABILITIES).item()
        min_circle_radius = self._world_length_to_generation_length(
            MIN_PROFILE_CIRCLE_RADIUS,
            world_size,
        )
        min_rect_side = self._world_length_to_generation_length(
            MIN_PROFILE_RECT_SIDE,
            world_size,
        )
        min_hole_margin = self._world_length_to_generation_length(
            MIN_PROFILE_HOLE_MARGIN,
            world_size,
        )
        min_dim = 2.0 * min_circle_radius if kind.startswith("circle") else min_rect_side
        world_bounds = ExtrudeFactory._workplane_world_bounds(
            cq_workplane,
            generation_world_half,
        )
        if site.face_kind == "plane":
            target_width, target_height = self._profile_sizer._sample_sketch_dimensions(
                sketch_workplane=cq_workplane,
                site=site,
                tangent_to_surface=True,
                min_dim=min_dim,
                world_bounds=world_bounds,
            )
        else:
            max_face_dim = self._face_bbox_max_dimension(site.face)
            target_width = ExtrudeFactory._sample_inside_size(max_face_dim, min_dim)
            target_height = ExtrudeFactory._sample_inside_size(max_face_dim, min_dim)
        if kind.startswith("circle"):
            radius = max(min_circle_radius, 0.5 * min(target_width, target_height))
            profile = {"type": "circle", "radius": radius}
            if kind == "circle_hole":
                max_inner_radius = radius - min_hole_margin
                if max_inner_radius >= min_circle_radius:
                    profile["inner_radius"] = float(
                        np.random.uniform(min_circle_radius, max_inner_radius)
                    )
            return profile, radius

        target_width = max(min_rect_side, target_width)
        target_height = max(min_rect_side, target_height)
        profile = {"type": "rect", "width": target_width, "height": target_height}
        if kind == "rect_hole":
            max_inner_width = target_width - min_hole_margin
            max_inner_height = target_height - min_hole_margin
            if max_inner_width >= min_rect_side and max_inner_height >= min_rect_side:
                profile["inner_width"] = float(
                    np.random.uniform(min_rect_side, max_inner_width)
                )
                profile["inner_height"] = float(
                    np.random.uniform(min_rect_side, max_inner_height)
                )
        return profile, 0.5 * target_height

    @staticmethod
    def _face_bbox_max_dimension(face: cq.Face) -> float:
        bbox = face.BoundingBox()
        max_dim = max(
            float(bbox.xmax - bbox.xmin),
            float(bbox.ymax - bbox.ymin),
            float(bbox.zmax - bbox.zmin),
        )
        if max_dim <= 0:
            raise ValueError("Sampled face has empty bounding box")
        return max_dim

    def _sample_free_profile(
        self,
        workplane: cq.Workplane,
        world_size: float,
        generation_world_half: float,
    ) -> tuple[dict, float]:
        kind = np.random.choice(PROFILE_KIND_PROBABILITIES).item()
        min_circle_radius = self._world_length_to_generation_length(
            MIN_PROFILE_CIRCLE_RADIUS,
            world_size,
        )
        min_rect_side = self._world_length_to_generation_length(
            MIN_PROFILE_RECT_SIDE,
            world_size,
        )
        min_hole_margin = self._world_length_to_generation_length(
            MIN_PROFILE_HOLE_MARGIN,
            world_size,
        )
        bounds = ExtrudeFactory._workplane_world_bounds(
            workplane,
            generation_world_half,
        )
        min_dim = 2.0 * min_circle_radius if kind.startswith("circle") else min_rect_side
        if bounds.symmetric_width < min_dim or bounds.symmetric_height < min_dim:
            raise ValueError("Workplane has too little room inside world bounds")

        target_width = ExtrudeFactory._sample_out_of_face_size(
            min_dim,
            bounds.symmetric_width,
            min_dim,
        )
        target_height = ExtrudeFactory._sample_out_of_face_size(
            min_dim,
            bounds.symmetric_height,
            min_dim,
        )
        if kind.startswith("circle"):
            radius = max(min_circle_radius, 0.5 * min(target_width, target_height))
            profile = {"type": "circle", "radius": radius}
            if kind == "circle_hole":
                max_inner_radius = radius - min_hole_margin
                if max_inner_radius >= min_circle_radius:
                    profile["inner_radius"] = float(
                        np.random.uniform(min_circle_radius, max_inner_radius)
                    )
            return profile, radius

        target_width = max(min_rect_side, target_width)
        target_height = max(min_rect_side, target_height)
        profile = {"type": "rect", "width": target_width, "height": target_height}
        if kind == "rect_hole":
            max_inner_width = target_width - min_hole_margin
            max_inner_height = target_height - min_hole_margin
            if max_inner_width >= min_rect_side and max_inner_height >= min_rect_side:
                profile["inner_width"] = float(
                    np.random.uniform(min_rect_side, max_inner_width)
                )
                profile["inner_height"] = float(
                    np.random.uniform(min_rect_side, max_inner_height)
                )
        return profile, 0.5 * target_height

    @staticmethod
    def _sample_line_endpoint(
        origin: cq.Vector,
        workplane: AxisWorkplane,
        generation_world_half: float,
        min_length: float,
    ) -> tuple[float, float, float]:
        theta = math.radians(float(np.random.uniform(0.0, 360.0)))
        phi = math.radians(float(np.random.uniform(5.0, 65.0)))
        x_vec = axis_vector(workplane.xDir)
        y_vec = axis_vector(_workplane_source_y_dir(workplane))
        z_vec = axis_vector(workplane.normal)
        direction = (
            x_vec.multiply(math.sin(phi) * math.cos(theta))
            .add(y_vec.multiply(math.cos(phi)))
            .add(z_vec.multiply(math.sin(phi) * math.sin(theta)))
        ).normalized()
        distance = ExtrudeFactory._distance_to_world_border(
            origin,
            direction,
            generation_world_half,
        )
        if distance < min_length:
            raise ValueError("Too little room for sweep line path")
        high = max(min_length, 0.7 * distance)
        length = float(np.random.uniform(min_length, high))
        return origin.add(direction.multiply(length)).toTuple()

    @staticmethod
    def _sample_path_length(min_length: float, max_length: float) -> float:
        if max_length <= min_length:
            raise ValueError("Too little room for sweep sketch path")
        return float(np.random.uniform(min_length, max_length))

    @staticmethod
    def _path_sampling_high(
        bounds,
        profile_offset: float,
    ) -> float:
        x_room = max(0.0, min(bounds.x_neg, bounds.x_pos))
        y_room = max(0.0, bounds.y_pos - profile_offset)
        return 0.85 * min(x_room, y_room)

    @staticmethod
    def _path_points_from_edges(path_edges: list[dict]) -> list[list[float]]:
        points: list[list[float]] = []
        for edge in path_edges:
            points.append(edge["start"])
            if edge["type"] == "arc":
                points.append(edge["mid"])
            points.append(edge["end"])
        return points

    @staticmethod
    def _normalize_path_second_coordinate(
        points: list[list[float]],
        profile_offset: float,
    ) -> None:
        min_value = min(point[1] for point in points)
        for point in points:
            point[1] = point[1] - min_value + profile_offset

    @staticmethod
    def _invert_path_x(points: list[list[float]]) -> None:
        for point in points:
            point[0] *= -1.0

    @staticmethod
    def _swap_path_xy(points: list[list[float]]) -> None:
        for point in points:
            point[0], point[1] = point[1], point[0]

    @staticmethod
    def _path_edges_fit_bounds(path_edges: list[dict], bounds) -> bool:
        points = SweepFactory._path_points_from_edges(path_edges)
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        return (
            min_x >= -bounds.x_neg
            and max_x <= bounds.x_pos
            and min_y >= -bounds.y_neg
            and max_y <= bounds.y_pos
        )

    def _sample_arc_path_edges(
        self,
        min_length: float,
        max_length: float,
        profile_offset: float,
    ) -> list[dict]:
        radius = self._sample_path_length(min_length, max_length)
        c45 = math.cos(math.radians(45.0))
        s45 = math.sin(math.radians(45.0))
        points = [
            [0.0, 0.0],
            [(c45 - 1.0) * radius, s45 * radius],
            [-radius, radius],
        ]

        edges: list[dict]
        if np.random.random() < 0.5:
            l1 = self._sample_path_length(min_length, max_length)
            points.insert(0, [0.0, -l1])
            for point in points:
                point[1] += l1
            edges = [
                {"type": "line", "start": points[0], "end": points[1]},
                {
                    "type": "arc",
                    "start": points[1],
                    "mid": points[2],
                    "end": points[3],
                },
            ]
        else:
            edges = [
                {
                    "type": "arc",
                    "start": points[0],
                    "mid": points[1],
                    "end": points[2],
                }
            ]

        if np.random.random() < 0.5:
            l2 = self._sample_path_length(min_length, max_length)
            last = points[-1]
            end = [last[0] - l2, last[1]]
            points.append(end)
            edges.append({"type": "line", "start": last, "end": end})

        if np.random.random() < 0.5:
            self._invert_path_x(points)
        if np.random.random() < 0.5:
            self._swap_path_xy(points)
        self._normalize_path_second_coordinate(points, profile_offset)
        return edges

    def _sample_segment_path_edges(
        self,
        min_length: float,
        max_length: float,
        profile_offset: float,
    ) -> list[dict]:
        length_1 = self._sample_path_length(min_length, max_length)
        length_2 = self._sample_path_length(min_length, max_length)
        alpha_1 = math.radians(float(np.random.uniform(0.0, 90.0)))
        alpha_2 = math.radians(float(np.random.uniform(0.0, 90.0)))
        points = [
            [0.0, 0.0],
            [length_1 * math.cos(alpha_1), length_1 * math.sin(alpha_1)],
            [
                length_1 * math.cos(alpha_1) + length_2 * math.cos(alpha_2),
                length_1 * math.sin(alpha_1) + length_2 * math.sin(alpha_2),
            ],
        ]
        if np.random.random() < 0.5:
            self._invert_path_x(points)
        self._normalize_path_second_coordinate(points, profile_offset)
        return [
            {"type": "line", "start": points[0], "end": points[1]},
            {"type": "line", "start": points[1], "end": points[2]},
        ]

    def _sample_sketch_path(
        self,
        workplane: cq.Workplane,
        profile_offset: float,
        world_size: float,
        generation_world_half: float,
    ) -> list[dict]:
        min_length = self._world_length_to_generation_length(8.0, world_size)
        bounds = ExtrudeFactory._workplane_world_bounds(
            workplane,
            generation_world_half,
        )
        max_length = self._path_sampling_high(bounds, profile_offset)
        for _ in range(30):
            if np.random.random() < 0.5:
                edges = self._sample_arc_path_edges(
                    min_length,
                    max_length,
                    profile_offset,
                )
            else:
                edges = self._sample_segment_path_edges(
                    min_length,
                    max_length,
                    profile_offset,
                )
            if self._path_edges_fit_bounds(edges, bounds):
                return edges
        raise ValueError("Could not sample a sweep sketch path inside world bounds")

    def generate_on_existing(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        world_size: float,
        generation_world_half: float = 1.0,
    ) -> tuple[Sweep, AxisWorkplane]:
        site = sampler.sample_site()
        placement = self._sample_placement(site)
        profile_workplane = self._profile_workplane(site)
        path_cq_workplane = workplane_at_closest_point(
            cad_object,
            site.point,
            placement.workplane.normal,
            placement.workplane.xDir,
            surface_compound=sampler.surface_compound,
        )
        profile, profile_offset = self._sample_profile(
            cad_object,
            sampler,
            site,
            profile_workplane,
            world_size,
            generation_world_half,
        )
        if np.random.random() < self.inclined_line_path_probability:
            min_length = self._world_length_to_generation_length(
                self.MIN_PATH_SPAN,
                world_size,
            )
            p1 = site.point.toTuple()
            p2 = self._sample_line_endpoint(
                site.point,
                placement.workplane,
                generation_world_half,
                min_length,
            )
            op = Sweep(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="line",
                line_points=(p1, p2),
            )
        else:
            edges = self._sample_sketch_path(
                path_cq_workplane,
                profile_offset,
                world_size,
                generation_world_half,
            )
            op = Sweep(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="sketch",
                path_workplane=placement.workplane,
                path_edges=edges,
            )
        return op, placement.workplane

    def to_dict(self) -> dict:
        return {
            "type": "SweepFactory",
            "inclined_line_path_probability": self.inclined_line_path_probability,
            "mesh_deflection": self.mesh_deflection,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SweepFactory":
        return SweepFactory(
            inclined_line_path_probability=entity.get(
                "inclined_line_path_probability",
                0.0,
            ),
            mesh_deflection=entity.get("mesh_deflection", 0.005),
        )


factories.register("sweep", SweepFactory)
