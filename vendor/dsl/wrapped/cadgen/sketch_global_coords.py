from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Iterable


AXIS_BASIS = {
    "XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "ZX": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}

POINT_METHODS = {"push", "segment", "arc", "threePointArc"}
NEED_PUSH_METHODS = {"rect", "circle"}


@dataclass
class ChainCall:
    name: str
    args: list[ast.expr]
    keywords: list[ast.keyword]


Point2D = tuple[float, float]
Vector3D = tuple[float, float, float]
Matrix2D = tuple[tuple[float, float], tuple[float, float]]


@dataclass
class PathEdge:
    kind: str
    start: Point2D
    end: Point2D
    mid: Point2D | None = None
    extra_args: list[ast.expr] | None = None
    keywords: list[ast.keyword] | None = None
    from_close: bool = False

    def reversed(self) -> "PathEdge":
        return PathEdge(
            self.kind,
            self.end,
            self.start,
            mid=self.mid,
            extra_args=list(self.extra_args or []),
            keywords=list(self.keywords or []),
            from_close=self.from_close,
        )


def _dot(a: Vector3D, b: Vector3D) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _axis_vector(axis: str) -> Vector3D:
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[-1]
    if name == "X":
        return (sign, 0.0, 0.0)
    if name == "Y":
        return (0.0, sign, 0.0)
    if name == "Z":
        return (0.0, 0.0, sign)
    raise ValueError(f"unknown axis {axis!r}")


def _number_node(value: float) -> ast.Constant:
    if float(value).is_integer():
        return ast.Constant(int(value))
    return ast.Constant(value)


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def _transform_point_node(node: ast.AST, du: float, dv: float) -> ast.AST:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return node
    values = [_literal_number(elt) for elt in node.elts]
    if len(node.elts) == 2 and all(value is not None for value in values):
        return ast.Tuple(
            elts=[
                _number_node(values[0] + du),
                _number_node(values[1] + dv),
            ],
            ctx=ast.Load(),
        )
    transformed = [_transform_point_node(elt, du, dv) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return ast.Tuple(elts=transformed, ctx=ast.Load())
    return ast.List(elts=transformed, ctx=ast.Load())


def _matrix_transform_point_node(node: ast.AST, matrix: Matrix2D) -> ast.AST:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return node
    values = [_literal_number(elt) for elt in node.elts]
    if len(node.elts) == 2 and all(value is not None for value in values):
        x, y = values
        return ast.Tuple(
            elts=[
                _number_node(matrix[0][0] * x + matrix[0][1] * y),
                _number_node(matrix[1][0] * x + matrix[1][1] * y),
            ],
            ctx=ast.Load(),
        )
    transformed = [_matrix_transform_point_node(elt, matrix) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return ast.Tuple(elts=transformed, ctx=ast.Load())
    return ast.List(elts=transformed, ctx=ast.Load())


def _point_from_node(node: ast.AST) -> Point2D | None:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 2:
        return None
    values = [_literal_number(elt) for elt in node.elts]
    if any(value is None for value in values):
        return None
    return (values[0], values[1])


def _point_node(point: Point2D) -> ast.Tuple:
    return ast.Tuple(
        elts=[_number_node(point[0]), _number_node(point[1])],
        ctx=ast.Load(),
    )


def _same_point(left: Point2D, right: Point2D, tol: float = 1e-9) -> bool:
    return abs(left[0] - right[0]) <= tol and abs(left[1] - right[1]) <= tol


def _cross(left: Point2D, right: Point2D) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _normalize_angle_positive(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _arc_center(start: Point2D, mid: Point2D, end: Point2D) -> Point2D | None:
    ax, ay = start
    bx, by = mid
    cx, cy = end
    determinant = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(determinant) <= 1e-12:
        return None

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / determinant
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / determinant
    return (ux, uy)


def _arc_delta(start: Point2D, mid: Point2D, end: Point2D, center: Point2D) -> float:
    t0 = math.atan2(start[1] - center[1], start[0] - center[0])
    tm = math.atan2(mid[1] - center[1], mid[0] - center[0])
    t1 = math.atan2(end[1] - center[1], end[0] - center[0])
    ccw_end = _normalize_angle_positive(t1 - t0)
    ccw_mid = _normalize_angle_positive(tm - t0)
    if ccw_mid <= ccw_end + 1e-9:
        return ccw_end
    return ccw_end - 2.0 * math.pi


def _edge_signed_area(edge: PathEdge) -> float:
    if edge.kind == "segment" or edge.mid is None:
        return 0.5 * _cross(edge.start, edge.end)

    center = _arc_center(edge.start, edge.mid, edge.end)
    if center is None:
        return 0.5 * _cross(edge.start, edge.end)

    radius = math.dist(center, edge.start)
    t0 = math.atan2(edge.start[1] - center[1], edge.start[0] - center[0])
    delta = _arc_delta(edge.start, edge.mid, edge.end, center)
    t1 = t0 + delta
    integral = radius * (
        center[0] * (math.sin(t1) - math.sin(t0))
        - center[1] * (math.cos(t1) - math.cos(t0))
    ) + radius * radius * delta
    return 0.5 * integral


def _path_signed_area(edges: list[PathEdge]) -> float:
    return sum(_edge_signed_area(edge) for edge in edges)


def _parse_path_edges(path_calls: list[ChainCall]) -> tuple[list[PathEdge], bool] | None:
    edges: list[PathEdge] = []
    start_point: Point2D | None = None
    current_point: Point2D | None = None
    closed = False

    for call in path_calls:
        if call.name == "segment":
            if len(call.args) >= 2:
                start = _point_from_node(call.args[0])
                end = _point_from_node(call.args[1])
                point_count = 2
            elif len(call.args) >= 1 and current_point is not None:
                start = current_point
                end = _point_from_node(call.args[0])
                point_count = 1
            else:
                return None
            if start is None or end is None:
                return None
            if start_point is None:
                start_point = start
            edges.append(
                PathEdge(
                    "segment",
                    start,
                    end,
                    extra_args=list(call.args[point_count:]),
                    keywords=list(call.keywords),
                )
            )
            current_point = end
            continue

        if call.name in {"arc", "threePointArc"}:
            if len(call.args) >= 3:
                start = _point_from_node(call.args[0])
                mid = _point_from_node(call.args[1])
                end = _point_from_node(call.args[2])
                point_count = 3
            elif len(call.args) >= 2 and current_point is not None:
                start = current_point
                mid = _point_from_node(call.args[0])
                end = _point_from_node(call.args[1])
                point_count = 2
            else:
                return None
            if start is None or mid is None or end is None:
                return None
            if start_point is None:
                start_point = start
            edges.append(
                PathEdge(
                    "arc",
                    start,
                    end,
                    mid=mid,
                    extra_args=list(call.args[point_count:]),
                    keywords=list(call.keywords),
                )
            )
            current_point = end
            continue

        if call.name == "close":
            if start_point is None or current_point is None:
                return None
            if not _same_point(current_point, start_point):
                edges.append(
                    PathEdge(
                        "segment",
                        current_point,
                        start_point,
                        keywords=list(call.keywords),
                        from_close=True,
                    )
                )
            current_point = start_point
            closed = True
            continue

        return None

    if start_point is not None and current_point is not None and _same_point(
        current_point, start_point
    ):
        closed = True

    return edges, closed


def _rotate_edges_to_lowest_start(edges: list[PathEdge]) -> list[PathEdge]:
    start_index = min(
        range(len(edges)),
        key=lambda index: (edges[index].start[0], edges[index].start[1]),
    )
    return edges[start_index:] + edges[:start_index]


def _make_edge_call(edge: PathEdge, is_first: bool) -> ChainCall:
    extra_args = list(edge.extra_args or [])
    keywords = list(edge.keywords or [])
    if edge.kind == "segment":
        args = (
            [_point_node(edge.start), _point_node(edge.end)]
            if is_first
            else [_point_node(edge.end)]
        )
        return ChainCall("segment", args + extra_args, keywords)

    if edge.mid is None:
        raise ValueError("arc edge is missing midpoint")

    args = (
        [_point_node(edge.start), _point_node(edge.mid), _point_node(edge.end)]
        if is_first
        else [_point_node(edge.mid), _point_node(edge.end)]
    )
    return ChainCall("arc", args + extra_args, keywords)


def _path_edges_to_calls(edges: list[PathEdge]) -> list[ChainCall]:
    calls: list[ChainCall] = []
    for index, edge in enumerate(edges):
        is_first = index == 0
        is_last = index == len(edges) - 1
        if (
            is_last
            and edge.kind == "segment"
            and _same_point(edge.end, edges[0].start)
        ):
            calls.append(ChainCall("close", [], list(edge.keywords or [])))
            continue
        calls.append(_make_edge_call(edge, is_first))
    return calls


def _normalize_path_calls(path_calls: list[ChainCall]) -> list[ChainCall]:
    parsed = _parse_path_edges(path_calls)
    if parsed is None:
        return path_calls

    edges, closed = parsed
    if not closed or not edges:
        return path_calls

    edges = _rotate_edges_to_lowest_start(edges)
    if _path_signed_area(edges) > 0.0:
        edges = [edge.reversed() for edge in reversed(edges)]
    return _path_edges_to_calls(edges)


def _normalize_sketch_path_calls(calls: list[ChainCall]) -> list[ChainCall]:
    normalized: list[ChainCall] = []
    in_sketch = False
    index = 0
    edge_methods = {"segment", "arc", "threePointArc"}
    path_methods = edge_methods | {"close"}

    while index < len(calls):
        call = calls[index]
        if call.name == "sketch":
            in_sketch = True
            normalized.append(call)
            index += 1
            continue

        if in_sketch and call.name in edge_methods:
            path_calls: list[ChainCall] = []
            while index < len(calls) and calls[index].name in path_methods:
                path_calls.append(calls[index])
                index += 1

            if index < len(calls) and calls[index].name == "assemble":
                normalized.extend(_normalize_path_calls(path_calls))
                normalized.append(calls[index])
                index += 1
                continue

            normalized.extend(path_calls)
            continue

        normalized.append(call)
        if in_sketch and call.name == "finalize":
            in_sketch = False
        index += 1

    return normalized


def _push_call(u: float, v: float) -> ChainCall:
    point = ast.Tuple(elts=[_number_node(u), _number_node(v)], ctx=ast.Load())
    points = ast.List(elts=[point], ctx=ast.Load())
    return ChainCall("push", [points], [])


def _workplane_offset_call(offset: float) -> ChainCall:
    return ChainCall(
        "workplane",
        [],
        [ast.keyword(arg="offset", value=_number_node(offset))],
    )


def _flatten_chain(node: ast.AST) -> tuple[ast.AST, list[ChainCall]]:
    calls: list[ChainCall] = []
    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        calls.append(
            ChainCall(current.func.attr, list(current.args), list(current.keywords))
        )
        current = current.func.value
    calls.reverse()
    return current, calls


def _rebuild_chain(base: ast.AST, calls: Iterable[ChainCall]) -> ast.AST:
    current = base
    for call in calls:
        current = ast.Call(
            func=ast.Attribute(value=current, attr=call.name, ctx=ast.Load()),
            args=call.args,
            keywords=call.keywords,
        )
    return current


def _transform_sketch_chain(
    base: ast.AST,
    calls: list[ChainCall],
    u_offset: float,
    v_offset: float,
    normal_offset: float,
    add_normal_workplane: bool,
) -> ast.AST:
    calls = _normalize_sketch_path_calls(calls)
    transformed_calls: list[ChainCall] = []
    in_sketch = False
    added_normal_workplane = False
    for call in calls:
        call = ChainCall(
            call.name,
            [_transform_nested_sketches(arg, u_offset, v_offset) for arg in call.args],
            [
                ast.keyword(
                    arg=keyword.arg,
                    value=_transform_nested_sketches(keyword.value, u_offset, v_offset),
                )
                for keyword in call.keywords
            ],
        )

        if call.name == "sketch":
            if (
                add_normal_workplane
                and not added_normal_workplane
                and _is_nonzero(normal_offset)
            ):
                transformed_calls.append(_workplane_offset_call(normal_offset))
                added_normal_workplane = True
            in_sketch = True
            transformed_calls.append(call)
            continue

        if in_sketch and call.name in NEED_PUSH_METHODS:
            if not transformed_calls or transformed_calls[-1].name != "push":
                transformed_calls.append(_push_call(u_offset, v_offset))
            transformed_calls.append(call)
            continue

        if in_sketch and call.name in POINT_METHODS:
            call = ChainCall(
                call.name,
                [_transform_point_node(arg, u_offset, v_offset) for arg in call.args],
                [
                    ast.keyword(
                        arg=keyword.arg,
                        value=_transform_point_node(keyword.value, u_offset, v_offset),
                    )
                    for keyword in call.keywords
                ],
            )
            transformed_calls.append(call)
            continue

        transformed_calls.append(call)
        if in_sketch and call.name == "finalize":
            in_sketch = False

    return _rebuild_chain(base, transformed_calls)


class _NestedSketchTransformer(ast.NodeTransformer):
    def __init__(self, u_offset: float, v_offset: float) -> None:
        self.u_offset = u_offset
        self.v_offset = v_offset

    def visit_Call(self, node: ast.Call) -> ast.AST:
        base, calls = _flatten_chain(node)
        if calls and any(call.name == "sketch" for call in calls):
            return _transform_sketch_chain(
                base,
                calls,
                self.u_offset,
                self.v_offset,
                0.0,
                add_normal_workplane=False,
            )
        return self.generic_visit(node)


def _transform_nested_sketches(node: ast.AST, u_offset: float, v_offset: float) -> ast.AST:
    return _NestedSketchTransformer(u_offset, v_offset).visit(node)


def _parse_assignment_or_expr(source: str) -> tuple[ast.Module, ast.AST]:
    module = ast.parse(source.strip())
    if len(module.body) != 1:
        raise ValueError("sketch_string must contain exactly one Python statement")
    stmt = module.body[0]
    if isinstance(stmt, ast.Assign):
        return module, stmt.value
    if isinstance(stmt, ast.Expr):
        return module, stmt.value
    raise ValueError("sketch_string must be an assignment or expression")


def _replace_assignment_or_expr(module: ast.Module, new_value: ast.AST) -> str:
    stmt = module.body[0]
    if isinstance(stmt, ast.Assign):
        stmt.value = new_value
    elif isinstance(stmt, ast.Expr):
        stmt.value = new_value
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def _axis_offsets(
    workplane_axis: str, origin: tuple[float, float, float]
) -> tuple[float, float, float]:
    try:
        x_dir, y_dir, z_dir = AXIS_BASIS[workplane_axis]
    except KeyError as exc:
        raise ValueError(f"unknown workplane_axis {workplane_axis!r}") from exc
    u = _dot(origin, x_dir)
    v = _dot(origin, y_dir)
    normal_offset = _dot(origin, z_dir)
    return u, v, normal_offset


def _is_nonzero(value: float, tol: float = 1e-12) -> bool:
    return abs(value) > tol


def sketch_to_global_coords(
    workplane_axis: str,
    local_workplane_origin: tuple[float, float, float],
    sketch_string: str,
) -> str:
    """Return code for the same sketch on the zero-origin global workplane."""

    origin = tuple(float(value) for value in local_workplane_origin)
    if len(origin) != 3:
        raise ValueError("local_workplane_origin must have exactly three values")
    u_offset, v_offset, normal_offset = _axis_offsets(workplane_axis, origin)

    module, value = _parse_assignment_or_expr(sketch_string)
    base, calls = _flatten_chain(value)
    if not calls:
        raise ValueError("sketch_string does not contain a method chain")

    new_value = _transform_sketch_chain(
        base,
        calls,
        u_offset,
        v_offset,
        normal_offset,
        add_normal_workplane=True,
    )
    return _replace_assignment_or_expr(module, new_value)


def _local_axis_transform_matrix(
    source_x_dir: str,
    source_y_dir: str,
    workplane_axis: str,
) -> Matrix2D:
    target_x_dir, target_y_dir, _ = AXIS_BASIS[workplane_axis]
    source_x = _axis_vector(source_x_dir)
    source_y = _axis_vector(source_y_dir)
    return (
        (_dot(source_x, target_x_dir), _dot(source_y, target_x_dir)),
        (_dot(source_x, target_y_dir), _dot(source_y, target_y_dir)),
    )


def _transform_rect_call(call: ChainCall, matrix: Matrix2D) -> ChainCall:
    if len(call.args) < 2:
        return call
    width = _literal_number(call.args[0])
    height = _literal_number(call.args[1])
    if width is None or height is None:
        return call

    x_axis_target = (matrix[0][0], matrix[1][0])
    if abs(x_axis_target[1]) > abs(x_axis_target[0]):
        args = [_number_node(height), _number_node(width), *call.args[2:]]
    else:
        args = list(call.args)
    return ChainCall(call.name, args, call.keywords)


def _transform_matrix_sketch_chain(
    base: ast.AST,
    calls: list[ChainCall],
    matrix: Matrix2D,
) -> ast.AST:
    transformed_calls: list[ChainCall] = []
    in_sketch = False
    for call in calls:
        call = ChainCall(
            call.name,
            [_transform_nested_sketches_by_matrix(arg, matrix) for arg in call.args],
            [
                ast.keyword(
                    arg=keyword.arg,
                    value=_transform_nested_sketches_by_matrix(keyword.value, matrix),
                )
                for keyword in call.keywords
            ],
        )

        if call.name == "sketch":
            in_sketch = True
            transformed_calls.append(call)
            continue

        if in_sketch and call.name in POINT_METHODS:
            call = ChainCall(
                call.name,
                [_matrix_transform_point_node(arg, matrix) for arg in call.args],
                [
                    ast.keyword(
                        arg=keyword.arg,
                        value=_matrix_transform_point_node(keyword.value, matrix),
                    )
                    for keyword in call.keywords
                ],
            )
            transformed_calls.append(call)
            continue

        if in_sketch and call.name == "rect":
            transformed_calls.append(_transform_rect_call(call, matrix))
            continue

        transformed_calls.append(call)
        if in_sketch and call.name == "finalize":
            in_sketch = False

    return _rebuild_chain(base, transformed_calls)


class _NestedMatrixSketchTransformer(ast.NodeTransformer):
    def __init__(self, matrix: Matrix2D) -> None:
        self.matrix = matrix

    def visit_Call(self, node: ast.Call) -> ast.AST:
        base, calls = _flatten_chain(node)
        if calls and any(call.name == "sketch" for call in calls):
            return _transform_matrix_sketch_chain(base, calls, self.matrix)
        return self.generic_visit(node)


def _transform_nested_sketches_by_matrix(node: ast.AST, matrix: Matrix2D) -> ast.AST:
    return _NestedMatrixSketchTransformer(matrix).visit(node)


def transform_sketch_local_axes(
    sketch_string: str,
    source_x_dir: str,
    source_y_dir: str,
    workplane_axis: str,
) -> str:
    """Rewrite local 2D sketch coordinates into a positive global workplane basis."""

    matrix = _local_axis_transform_matrix(source_x_dir, source_y_dir, workplane_axis)
    module, value = _parse_assignment_or_expr(sketch_string)
    base, calls = _flatten_chain(value)
    if not calls:
        raise ValueError("sketch_string does not contain a method chain")

    new_value = _transform_matrix_sketch_chain(base, calls, matrix)
    return _replace_assignment_or_expr(module, new_value)


def sketch_chain_only(sketch_string: str) -> str:
    """Return only the sketch() method chain from a full Workplane sketch expression."""

    module, value = _parse_assignment_or_expr(sketch_string)
    base_node, calls = _flatten_chain(value)
    if (
        isinstance(base_node, ast.Call)
        and isinstance(base_node.func, ast.Name)
        and base_node.func.id == "sketch"
    ):
        ast.fix_missing_locations(module)
        return ast.unparse(_rebuild_chain(base_node, calls))
    for index, call in enumerate(calls):
        if call.name == "sketch":
            base = ast.Call(
                func=ast.Name(id="sketch", ctx=ast.Load()),
                args=call.args,
                keywords=call.keywords,
            )
            ast.fix_missing_locations(module)
            return ast.unparse(_rebuild_chain(base, calls[index + 1 :]))
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return ast.unparse(value)
    raise ValueError("sketch_string does not contain a sketch() call")
