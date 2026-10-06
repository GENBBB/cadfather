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


def _parse_make_spline(path: str) -> tuple[cq.Edge, tuple[float, float, float]]:
    expr = ast.parse(path.strip(), mode="eval").body
    if (
        not isinstance(expr, ast.Call)
        or not isinstance(expr.func, ast.Name)
        or expr.func.id != "makeSpline"
        or len(expr.args) < 1
    ):
        raise ValueError("makeSpline path must be makeSpline([points][, [tangents]])")
    list_node = expr.args[0]
    if not isinstance(list_node, (ast.List, ast.Tuple)):
        raise ValueError("makeSpline argument must be a list of points")
    points = [_literal_point3(elt) for elt in list_node.elts]
    if len(points) < 2:
        raise ValueError("makeSpline needs at least two points")
    vectors = [cq.Vector(*point) for point in points]
    # Optional [start_tangent, end_tangent] so the start tangent is fixed (the
    # profile stays perpendicular to the path start).
    if len(expr.args) >= 2 and isinstance(expr.args[1], (ast.List, ast.Tuple)):
        tangents = [cq.Vector(*_literal_point3(elt)) for elt in expr.args[1].elts]
        edge = cq.Edge.makeSpline(vectors, tangents=tangents)
    else:
        edge = cq.Edge.makeSpline(vectors)
    return edge, points[0]


def _path_3d(path: str) -> tuple[cq.Edge | cq.Wire, tuple[float, float, float]]:
    path = path.strip()
    if path.startswith("makeLine"):
        return _parse_make_line(path)
    if path.startswith("makeSpline"):
        return _parse_make_spline(path)
    return _path_wire_from_sketch(path)


def _rotate_vector(vector, axis, angle: float):
    """Rodrigues rotation of ``vector`` about ``axis`` by ``angle`` radians."""
    v = np.asarray(vector, dtype=float)
    k = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(k)
    if norm < 1e-9:
        return v
    k = k / norm
    return (
        v * math.cos(angle)
        + np.cross(k, v) * math.sin(angle)
        + k * float(np.dot(k, v)) * (1.0 - math.cos(angle))
    )


def sweep_adv(
    r: cq.Workplane | cq.Shape | None,
    profile: str,
    path: str,
    world_size: float | None = 200,
    up: tuple[float, float, float] | None = None,
):
    path = path.strip()
    path_3d, path_start = _path_3d(path)
    profile_r = eval(_normalize_offset_calls(profile), {"cq": cq})
    if path.startswith("makeSpline"):
        # 3D/planar spline spine: use a corrected (non-Frenet) frame, plus a
        # fixed up-vector when given, so a non-symmetric profile does NOT twist
        # along the path (strict Frenet spins the section at inflections / in 3D).
        if up is not None:
            sweep_body = profile_r.sweep(
                path_3d, makeSolid=True, isFrenet=False, normal=cq.Vector(*up)
            )
        else:
            sweep_body = profile_r.sweep(path_3d, makeSolid=True, isFrenet=False)
    else:
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
    elif profile["type"] == "polygon":
        # Closed polygon (<=8 points), built from segments + close so the
        # globalizer handles it.
        pts = profile["points"]
        local += f".segment({_format_point2(pts[0])}, {_format_point2(pts[1])})"
        for point in pts[2:]:
            local += f".segment({_format_point2(point)})"
        local += ".close().assemble()"
    elif profile["type"] == "stadium":
        # Smooth obround (two tangent semicircles + two straight sides), built
        # from segment+arc so the globalizer handles it. width = straight length,
        # radius = end-cap radius; full size = (width + 2*radius) x (2*radius).
        half_w = profile["width"] / 2.0
        cap = profile["radius"]
        hw, nhw = _format_number(half_w), _format_number(-half_w)
        rr, nrr = _format_number(cap), _format_number(-cap)
        hwr, nhwr = _format_number(half_w + cap), _format_number(-(half_w + cap))
        local += (
            f".segment(({nhw}, {nrr}), ({hw}, {nrr}))"
            f".arc(({hw}, {nrr}), ({hwr}, 0), ({hw}, {rr}))"
            f".segment(({hw}, {rr}), ({nhw}, {rr}))"
            f".arc(({nhw}, {rr}), ({nhwr}, 0), ({nhw}, {nrr}))"
            f".close().assemble()"
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


class SweepAdv(BaseOperation):
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
        spline_points: list[tuple[float, float, float]] | None = None,
        spline_up: tuple[float, float, float] | None = None,
    ):
        self.profile_workplane = profile_workplane
        self.profile = profile
        self.path_kind = path_kind
        self.path_workplane = path_workplane
        self.path_edges = path_edges
        self.line_points = line_points
        # Global 3D control points for a 'spline' path (Family B: planar + 3D).
        self.spline_points = spline_points
        # Fixed up-vector (binormal axis) for a spline sweep so a non-symmetric
        # profile does not twist; None for symmetric (circle) profiles.
        self.spline_up = spline_up

    def profile_string(self) -> str:
        return _profile_expression(self.profile_workplane, self.profile)

    def path_string(self) -> str:
        if self.path_kind == "line":
            if self.line_points is None:
                raise ValueError("line sweep path is missing endpoints")
            p1, p2 = self.line_points
            return f"makeLine({_format_point(p1)}, {_format_point(p2)})"
        if self.path_kind == "spline":
            if not self.spline_points:
                raise ValueError("spline sweep path is missing control points")
            sp = self.spline_points
            pts = ", ".join(_format_point(point) for point in sp)
            # Fix the start tangent to the first chord (which runs along the
            # profile normal axis) so the profile is perpendicular to the path
            # start; fix the end tangent to the last chord for a natural finish.
            t0 = tuple(sp[1][i] - sp[0][i] for i in range(3))
            t1 = tuple(sp[-1][i] - sp[-2][i] for i in range(3))
            return f"makeSpline([{pts}], [{_format_point(t0)}, {_format_point(t1)}])"
        if self.path_workplane is None or self.path_edges is None:
            raise ValueError("sketch sweep path is missing workplane or edges")
        return _path_expression_from_edges(self.path_workplane, self.path_edges)

    def to_string(self, world_size: float | None = None) -> str:
        args = (
            f"r, {_double_quoted_string(self.profile_string())}, "
            f"{_double_quoted_string(self.path_string())}"
        )
        if self.path_kind == "spline" and self.spline_up is not None:
            args += f", up={_format_point(self.spline_up)}"
        return f"r=sweep_adv({args})\n"

    def to_shape(self) -> cq.Shape:
        path_str = self.path_string()
        path_3d, _ = _path_3d(path_str)
        profile_r = eval(_normalize_offset_calls(self.profile_string()), {"cq": cq})
        if path_str.strip().startswith("makeSpline"):
            if self.spline_up is not None:
                body = profile_r.sweep(
                    path_3d,
                    makeSolid=True,
                    isFrenet=False,
                    normal=cq.Vector(*self.spline_up),
                )
            else:
                body = profile_r.sweep(path_3d, makeSolid=True, isFrenet=False)
        else:
            body = profile_r.sweep(path_3d, makeSolid=True, isFrenet=True)
        return shape_from_cad_object(body)

    def transform(self, shift: list[float], scale: float, *args, **kwargs) -> None:
        self.profile_workplane.transform(shift, scale)
        if self.path_workplane is not None:
            self.path_workplane.transform(shift, scale)
        if self.line_points is not None:
            self.line_points = (
                tuple((self.line_points[0][i] + shift[i]) * scale for i in range(3)),
                tuple((self.line_points[1][i] + shift[i]) * scale for i in range(3)),
            )
        if self.spline_points is not None:
            self.spline_points = [
                tuple((point[i] + shift[i]) * scale for i in range(3))
                for point in self.spline_points
            ]
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
        if self.profile.get("type") == "polygon":
            self.profile["points"] = [
                [coord * scale for coord in point]
                for point in self.profile["points"]
            ]
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
        if self.spline_points is not None:
            self.spline_points = [
                tuple(round(value) for value in point) for point in self.spline_points
            ]
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
        if self.profile.get("type") == "polygon":
            self.profile["points"] = [
                [round(coord) for coord in point]
                for point in self.profile["points"]
            ]
        if self.path_edges is not None:
            for edge in self.path_edges:
                for key in ("start", "end", "mid"):
                    if key in edge and edge[key] is not None:
                        edge[key] = [round(value) for value in edge[key]]

    def fix(self) -> None:
        self.to_shape()

    def to_dict(self) -> dict:
        return {
            "type": "SweepAdv",
            "profile_workplane": self.profile_workplane.to_dict(),
            "profile": self.profile,
            "path_kind": self.path_kind,
            "path_workplane": self.path_workplane.to_dict()
            if self.path_workplane is not None
            else None,
            "path_edges": self.path_edges,
            "line_points": self.line_points,
            "spline_points": self.spline_points,
            "spline_up": self.spline_up,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SweepAdv":
        assert (
            entity["type"] == "SweepAdv"
        ), f"Trying to build SweepAdv from type {entity['type']}"
        return SweepAdv(
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
            spline_points=[tuple(point) for point in entity["spline_points"]]
            if entity.get("spline_points") is not None
            else None,
            spline_up=tuple(entity["spline_up"])
            if entity.get("spline_up") is not None
            else None,
        )


@dataclass(frozen=True)
class SweepAdvPlacement:
    site: SampledSite
    workplane: AxisWorkplane


class SweepAdvFactory(BaseFactory):
    MIN_PATH_SPAN = 5.0

    def __init__(
        self,
        inclined_line_path_probability: float = 0.0,
        mesh_deflection: float = 0.005,
        freeform_path_probability: float = 0.0,
        spline_path_probability: float = 0.5,
        existing_spline_probability: float = 0.0,
    ):
        self.inclined_line_path_probability = inclined_line_path_probability
        self.mesh_deflection = mesh_deflection
        # Probability that a standalone sweep uses a FREEFORM path. Freeform
        # paths are deliberately not a single line/arc/helix (those would
        # duplicate extrude/revolve/spring) -- see TODO.md Step 1.
        self.freeform_path_probability = freeform_path_probability
        # Within a freeform sweep, probability of Family B (3D B-spline spine)
        # vs Family A (composite planar line+arc chain).
        self.spline_path_probability = spline_path_probability
        # Probability that a COMBINED (face-attach) sweep uses a B-spline path.
        # Default 0: combined sweeps use the fast/robust line+arc face paths
        # (the colleague's approach). Spline face-paths are correct (grow out
        # of the profile along its outward normal) but OCC's spline pipe-sweep +
        # the validity rebuild are slow, so they are OPT-IN.
        self.existing_spline_probability = existing_spline_probability
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
    ) -> SweepAdv:
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

        if np.random.random() < self.freeform_path_probability:
            # General sweep: a small, smooth, no-hole profile swept along a
            # FREEFORM path. The profile is kept small so the curved path has
            # room (and so bends don't self-intersect the cross-section).
            profile, profile_offset = self._sample_freeform_profile(
                generation_world_half
            )
            if np.random.random() < self.spline_path_probability:
                # Family B: smooth 3D (or planar) B-spline spine.
                spline_points, spline_up = self._sample_spline_path(
                    profile_workplane,
                    profile_offset,
                    generation_world_half,
                )
                # Circle profiles are symmetric -> no fixed up needed (no twist).
                if profile["type"] == "circle":
                    spline_up = None
                return SweepAdv(
                    profile_workplane=profile_workplane,
                    profile=profile,
                    path_kind="spline",
                    spline_points=spline_points,
                    spline_up=spline_up,
                )
            # Family A: composite planar line+arc chain (dogleg/U/S/serpentine).
            edges = self._sample_freeform_path(
                path_cq_workplane,
                profile_offset,
                world_size or 200,
                generation_world_half,
            )
            return SweepAdv(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="sketch",
                path_workplane=path_workplane,
                path_edges=edges,
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
            return SweepAdv(
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
        return SweepAdv(
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
        return SweepAdvFactory._x_dir_for_profile_normal(
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

    def _sample_placement(self, site: SampledSite) -> SweepAdvPlacement:
        normal = np.random.choice(orthogonal_axes(site.normal_axis)).item()
        x_dir = self._x_dir_for_outside_y(normal, site)
        return SweepAdvPlacement(
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
        points = SweepAdvFactory._path_points_from_edges(path_edges)
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

    @staticmethod
    def _sample_freeform_profile(
        generation_world_half: float,
    ) -> tuple[dict, float]:
        """A small, simple, single-loop (no-hole) cross-section for a freeform
        sweep. Returns (profile dict, profile_offset) where profile_offset is the
        half-extent that the bend radius must clear to avoid self-intersection.
        """
        half = float(np.random.uniform(0.04, 0.09)) * generation_world_half
        roll = np.random.random()
        if roll < 0.15:
            return {"type": "circle", "radius": half}, half
        if roll < 0.30:
            # Annulus (hollow circle) -> pipe / tube cross-section. Symmetric, so
            # no twist; profile_offset stays the outer radius.
            inner = half * float(np.random.uniform(0.4, 0.8))
            return {"type": "circle", "radius": half, "inner_radius": inner}, half
        if roll < 0.80:
            # REGULAR polygon, 3-8 sides (triangle..octagon), random size + start
            # rotation, tiny jitter only -> a clean regular polygon (the user's
            # "at least regular polygons"). Chunky (> circle radius) so it reads
            # as a hex/oct rod, not a thin ribbon.
            n = int(np.random.randint(3, 9))
            step = 2.0 * math.pi / n
            base = half * float(np.random.uniform(1.3, 1.9))
            angle0 = float(np.random.uniform(0.0, 2.0 * math.pi))
            points = []
            for i in range(n):
                angle = angle0 + i * step + float(np.random.uniform(-0.05, 0.05)) * step
                radius = base * float(np.random.uniform(0.95, 1.0))
                points.append(
                    (float(radius * math.cos(angle)), float(radius * math.sin(angle)))
                )
            return {"type": "polygon", "points": points}, base
        # Smooth obround (stadium): flat-ish, rounded, slide-like.
        cap = half * float(np.random.uniform(0.5, 1.0))
        straight = 2.0 * half * float(np.random.uniform(0.8, 2.2))
        offset = 0.5 * straight + cap  # half of full width
        return {"type": "stadium", "width": straight, "radius": cap}, offset

    def _sample_freeform_path_edges(
        self,
        min_length: float,
        max_length: float,
        profile_offset: float,
    ) -> list[dict]:
        """Family A: a FREEFORM PLANAR path = a G1 tangent-continuous chain of
        straight segments joined by tangent fillet arcs (polyline-with-fillets).
        Dogleg/offset, U-bend, ogee/S and serpentine all fall out as parameter
        regimes. Built with a 'turtle' so every join is kink-free (a kinked path
        self-intersects when swept). Deliberately NOT a single line (=extrude),
        single arc (=revolve) or helix (=spring); see the dedup rule in TODO.md.
        """

        def rotate_about(pt: list[float], center: tuple[float, float], angle: float) -> list[float]:
            ca, sa = math.cos(angle), math.sin(angle)
            dx, dy = pt[0] - center[0], pt[1] - center[1]
            return [center[0] + dx * ca - dy * sa, center[1] + dx * sa + dy * ca]

        # Fillet radius must exceed the profile half-extent or the swept profile
        # self-intersects on the inside of the bend; keep radius >= 3*offset.
        min_radius = max(min_length, 3.0 * profile_offset)
        n_turns = int(np.random.randint(2, 5))
        alpha_sign = 1.0 if np.random.random() < 0.5 else -1.0
        position = [0.0, 0.0]
        tangent = (0.0, 1.0)  # start heading +y
        points: list[list[float]] = [position]
        edges: list[dict] = []
        for _ in range(n_turns):
            if np.random.random() < 0.6:  # straight run before the turn (tangent)
                run = self._sample_path_length(min_length, max_length)
                new = [position[0] + tangent[0] * run, position[1] + tangent[1] * run]
                edges.append({"type": "line", "start": position, "end": new})
                points.append(new)
                position = new
            radius = self._sample_path_length(min_radius, max_length)
            alpha = alpha_sign * math.radians(float(np.random.uniform(20.0, 110.0)))
            left_normal = (-tangent[1], tangent[0])
            side = 1.0 if alpha > 0 else -1.0
            center = (
                position[0] + radius * side * left_normal[0],
                position[1] + radius * side * left_normal[1],
            )
            mid = rotate_about(position, center, alpha / 2.0)
            end = rotate_about(position, center, alpha)
            edges.append({"type": "arc", "start": position, "mid": mid, "end": end})
            points.append(mid)
            points.append(end)
            ca, sa = math.cos(alpha), math.sin(alpha)
            tangent = (
                tangent[0] * ca - tangent[1] * sa,
                tangent[0] * sa + tangent[1] * ca,
            )
            position = end
            if np.random.random() < 0.7:  # mostly alternate (S/serpentine)
                alpha_sign = -alpha_sign

        if np.random.random() < 0.5:  # tangent (kink-free) lead-out line
            lead = self._sample_path_length(min_length, max_length)
            new_end = [position[0] + tangent[0] * lead, position[1] + tangent[1] * lead]
            edges.append({"type": "line", "start": position, "end": new_end})
            points.append(new_end)

        # Keep the path starting along +y (maps to the profile normal in 3D, so
        # the profile is perpendicular to the path start). _invert_path_x keeps
        # that +y start; _swap_path_xy would rotate it to +x, so it is NOT used.
        if np.random.random() < 0.5:
            self._invert_path_x(points)
        self._normalize_path_second_coordinate(points, profile_offset)
        return edges

    def _sample_spline_path(
        self,
        profile_workplane: AxisWorkplane,
        profile_offset: float,
        generation_world_half: float,
        direction_override: "np.ndarray | None" = None,
    ) -> list[tuple[float, float, float]]:
        """Family B: smooth B-spline spine through global 3D control points.
        Starts heading along the profile's normal axis (so the axis-aligned
        profile is ~perpendicular to the start tangent), then drifts with gentle,
        low-curvature turns -- planar (40%) or genuinely 3D (60%). Gentle turns
        keep the local radius of curvature well above the profile extent.
        """
        axis_map = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        axis_letter = profile_workplane.normal[-1]
        start = np.array(profile_workplane.point, dtype=float)
        # Direction the spline first heads. For a face-attach sweep we MUST use
        # the face's signed OUTWARD vector (profile_workplane.normal is only a
        # POSITIVE axis name, so on a -X face it would head +X = into the solid,
        # giving a buried "did not change volume" boss). Standalone callers omit
        # the override and use the positive profile-normal axis (grows into open
        # space, which is correct there).
        if direction_override is not None:
            direction = np.asarray(direction_override, dtype=float)
            direction = direction / (float(np.linalg.norm(direction)) or 1.0)
        else:
            nv = axis_vector(profile_workplane.normal)
            direction = np.array([nv.x, nv.y, nv.z], dtype=float)
        n_points = int(np.random.randint(4, 8))
        planar = np.random.random() < 0.4
        perp_names = list(orthogonal_axes(axis_letter))
        plane_normal = np.array(
            axis_map[perp_names[int(np.random.randint(0, len(perp_names)))][-1]]
        )
        half = generation_world_half
        # Gentle curvature (turns <=32 deg) keeps the radius well above the
        # profile, so the step margin can be modest; cap the offset term so a
        # wide stadium profile still fits inside the world box.
        eff_offset = min(profile_offset, 0.12 * half)
        min_step = max(3.0 * eff_offset, 0.13 * half)
        max_step = 0.5 * half
        lim = 0.95 * half
        for _ in range(40):
            pts = [start.copy()]
            d = direction.copy()  # = signed profile normal
            for _ in range(n_points - 1):
                # Advance FIRST, then turn: the first segment runs exactly along
                # the profile normal, so the spline starts perpendicular to the
                # profile. CLAMP each step to the room left inside the world box
                # (instead of failing the whole sample) -- the path just ends
                # early at the boundary, so spline paths reliably generate.
                room = max_step
                for i in range(3):
                    if d[i] > 1e-9:
                        room = min(room, (lim - pts[-1][i]) / d[i])
                    elif d[i] < -1e-9:
                        room = min(room, (-lim - pts[-1][i]) / d[i])
                if room < min_step:
                    break
                step = float(np.random.uniform(min_step, room))
                pts.append(pts[-1] + d * step)
                turn = math.radians(float(np.random.uniform(8.0, 32.0)))
                if planar:
                    sign = 1.0 if np.random.random() < 0.5 else -1.0
                    d = _rotate_vector(d, plane_normal, sign * turn)
                else:
                    d = _rotate_vector(d, np.random.uniform(-1.0, 1.0, size=3), turn)
                norm = float(np.linalg.norm(d))
                if norm < 1e-9:
                    break
                d = d / norm
            if len(pts) >= 3:
                # Fixed up = the plane normal, but ONLY for a PLANAR spline:
                # there the tangent is always perpendicular to it, so the section
                # neither twists nor distorts. For a genuinely 3D spline a fixed
                # binormal distorts the sweep where the tangent turns toward it,
                # so use None (corrected, non-Frenet frame: minimal twist, no
                # distortion).
                up = tuple(float(c) for c in plane_normal) if planar else None
                return [tuple(float(c) for c in p) for p in pts], up
        raise ValueError("Could not sample a spline sweep path inside world bounds")

    def _sample_freeform_path(
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
        if max_length <= min_length:
            raise ValueError("Workplane has too little room for a freeform path")
        for _ in range(40):
            try:
                edges = self._sample_freeform_path_edges(
                    min_length,
                    max_length,
                    profile_offset,
                )
            except ValueError:
                continue
            if self._path_edges_fit_bounds(edges, bounds):
                return edges
        raise ValueError("Could not sample a freeform sweep path inside world bounds")

    def generate_on_existing(
        self,
        cad_object: cq.Workplane | cq.Shape,
        sampler: PreparedSurfaceSampler,
        world_size: float,
        generation_world_half: float = 1.0,
    ) -> tuple[SweepAdv, AxisWorkplane]:
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
        roll = float(np.random.random())
        if roll < self.inclined_line_path_probability:
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
            op = SweepAdv(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="line",
                line_points=(p1, p2),
            )
        elif roll < self.inclined_line_path_probability + self.existing_spline_probability:
            # SPLINE path on an existing face: same approach as the colleague's
            # sketch path (profile on a face, path grows from the profile centre
            # PERPENDICULAR to the profile), but the path is a smooth freeform
            # B-spline. _sample_spline_path starts at profile_workplane.point
            # along its (signed) normal, so the perpendicular-start holds. Bound
            # by the world half so the boss can grow out beyond the parent.
            spline_points, spline_up = self._sample_spline_path(
                profile_workplane,
                profile_offset,
                0.5 * world_size,
                direction_override=np.array(
                    [site.outward.x, site.outward.y, site.outward.z], dtype=float
                ),
            )
            op = SweepAdv(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="spline",
                spline_points=spline_points,
                spline_up=spline_up,
            )
        else:
            edges = self._sample_sketch_path(
                path_cq_workplane,
                profile_offset,
                world_size,
                generation_world_half,
            )
            op = SweepAdv(
                profile_workplane=profile_workplane,
                profile=profile,
                path_kind="sketch",
                path_workplane=placement.workplane,
                path_edges=edges,
            )
        return op, placement.workplane

    def to_dict(self) -> dict:
        return {
            "type": "SweepAdvFactory",
            "inclined_line_path_probability": self.inclined_line_path_probability,
            "mesh_deflection": self.mesh_deflection,
            "freeform_path_probability": self.freeform_path_probability,
            "spline_path_probability": self.spline_path_probability,
            "existing_spline_probability": self.existing_spline_probability,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SweepAdvFactory":
        return SweepAdvFactory(
            inclined_line_path_probability=entity.get(
                "inclined_line_path_probability",
                0.0,
            ),
            mesh_deflection=entity.get("mesh_deflection", 0.005),
            freeform_path_probability=entity.get("freeform_path_probability", 0.0),
            spline_path_probability=entity.get("spline_path_probability", 0.5),
            existing_spline_probability=entity.get("existing_spline_probability", 0.0),
        )


factories.register("sweep_adv", SweepAdvFactory)
