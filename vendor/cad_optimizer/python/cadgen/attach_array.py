from __future__ import annotations
import ast

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from .sselectors import PointOnFaceSelector

# ---------- geometry helpers (your proven attachment principle) ----------


def closest_point_on_face(face: cq.Face, p: cq.Vector) -> cq.Vector:
    v = BRepBuilderAPI_MakeVertex(p.toPnt()).Vertex()
    dss = BRepExtrema_DistShapeShape(v, face.wrapped)
    dss.Perform()
    if not dss.IsDone() or dss.NbSolution() < 1:
        raise ValueError("Could not compute closest point on face.")
    p_on = dss.PointOnShape2(1)
    return cq.Vector(p_on.X(), p_on.Y(), p_on.Z())


def tangent_plane_from_face_at_point(
    face: cq.Face, near_pt: cq.Vector, x_hint=cq.Vector(0, 0, 1)
) -> cq.Plane:
    p_on = closest_point_on_face(face, near_pt)
    n = face.normalAt(p_on).normalized()

    xh = x_hint.normalized()
    if abs(xh.dot(n)) > 0.95:
        xh = cq.Vector(1, 0, 0)

    xDir = (xh - n.multiply(xh.dot(n))).normalized()
    return cq.Plane(origin=p_on, xDir=xDir, normal=n)


# ---------- safe DSL: ".circle(5).extrude(10)" ----------

_ALLOWED_METHODS = {
    # sketch
    "circle",
    "rect",
    "polyline",
    "lineTo",
    "moveTo",
    "close",
    "text",
    "finalize",
    "assemble",
    "sketch",
    "segment",
    "arc",
    "close",
    "push",
    # transforms
    "center",
    "move",
    "translate",
    "rotate",
    # profiles
    "toPending",
    # 3D ops
    "extrude",
    "revolve",
    "loft",
    "sweep",
    # booleans on workplane solids (optional)
    "union",
    "cut",
    "intersect",
    # edge ops (optional)
    "fillet",
    "chamfer",
}


def _literal(node):
    # allow only literals: numbers, bool, None, strings, tuples/lists/dicts of literals
    try:
        return ast.literal_eval(node)
    except Exception:
        raise ValueError(
            "Only literal arguments are allowed (numbers/bool/None/str/tuples/lists)."
        )


def _parse_chain(expr: str):
    """
    Parse ".circle(5).extrude(10)" into [("circle",[5],{}), ("extrude",[10],{})]
    """
    s = expr.strip()
    if not s.startswith("."):
        raise ValueError(
            "Expression must start with '.', e.g. '.circle(5).extrude(10)'"
        )

    # Make it a valid python expression by prefixing a dummy name
    dummy = "WP" + s
    tree = ast.parse(dummy, mode="eval")

    calls = []
    node = tree.body  # expression AST

    # unwind nested Call/Attribute chain: WP.circle(...).extrude(...)
    while isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only chained method calls are allowed.")
        method = node.func.attr
        if method not in _ALLOWED_METHODS:
            raise ValueError(f"Method '{method}' is not allowed.")

        args = [_literal(a) for a in node.args]
        kwargs = {kw.arg: _literal(kw.value) for kw in node.keywords}
        calls.append((method, args, kwargs))

        node = node.func.value  # go to previous link in chain

    if not (isinstance(node, ast.Name) and node.id == "WP"):
        raise ValueError(
            "Expression must be a simple call chain, like '.circle(5).extrude(10)'"
        )

    calls.reverse()
    return calls


def _run_chain(wp: cq.Workplane, calls):
    cur = wp
    for method, args, kwargs in calls:
        cur = getattr(cur, method)(*args, **kwargs)
    return cur


def _wp_attach_at(
    self: cq.Workplane,
    points,
    expr: str,
    *,
    x_hint=(0, 0, 1),
    face_selector=None,
    combine="a",
    _combine=True,
):
    """
    points: list of (x,y,z) or cq.Vector
    expr: string like ".circle(5).extrude(10)" executed on a tangent wp per point
    _combine: if True => union features into base solid; else => return features compound WP
    """
    calls = _parse_chain(expr)

    base = self
    xh = cq.Vector(*x_hint) if not isinstance(x_hint, cq.Vector) else x_hint

    feats = []
    for pt in points:
        p = cq.Vector(*pt) if not isinstance(pt, cq.Vector) else pt

        if face_selector is None:
            face = base.faces(PointOnFaceSelector([p.x, p.y, p.z])).val()
        else:
            face = (
                base.faces(face_selector)
                .faces(PointOnFaceSelector([p.x, p.y, p.z]))
                .val()
            )

        pln = tangent_plane_from_face_at_point(face, p, x_hint=xh)
        wp_local = cq.Workplane(pln)

        res = _run_chain(wp_local, calls)

        # normalize to a solid/shape
        if isinstance(res, cq.Workplane):
            solids = res.solids().vals()
            if not solids:
                raise ValueError(
                    f"Expression did not produce a solid at point {tuple(p)}."
                )
            feats.append(solids[0])
        else:
            feats.append(res)

    comp = cq.Compound.makeCompound(feats)
    feats_wp = cq.Workplane(obj=comp)

    if not _combine:
        return feats_wp

    # union into base
    if combine == "a":
        return base.union(comp)
    elif combine == "s":
        return base.cut(comp)


setattr(cq.Workplane, "attach_at", _wp_attach_at)
