from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cadquery as cq
import numpy as np
import trimesh
from cq_parser.parser import CADQuerySyntacticParser  # type: ignore
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
)
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_State,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face

if TYPE_CHECKING:
    from .sketch import Sketch


def compound_to_mesh(compound):
    # cq.Compound
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def shape_to_area(shape):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return props.Mass()


def shape_to_volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def shape_to_bbox(shape):
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    return bbox.Get()


def distance_shape_to_shape(shape1, shape2):
    ext = BRepExtrema_DistShapeShape(shape1, shape2)
    dist = ext.Value()
    return dist


def distance_shape_to_shape_ocp(shape1, shape2):
    ext = BRepExtrema_DistShapeShape(shape1.wrapped, shape2.wrapped)
    dist = ext.Value()
    return dist


def get_outer_and_inner_wires(face: TopoDS_Face):
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    max_area = -1.0

    outer_wire = None
    inner_wires = []

    idx_outer = -1
    i = 0
    while exp.More():
        wire = TopoDS.Wire_s(exp.Current())
        tmp_face = BRepBuilderAPI_MakeFace(wire).Face()
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(tmp_face, props)
        area = props.Mass()
        if area > max_area:
            max_area = area
            idx_outer = i
        exp.Next()
        i += 1

    i = 0
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    while exp.More():
        wire = TopoDS.Wire_s(exp.Current())
        if idx_outer == i:
            outer_wire = wire
        else:
            inner_wires.append(wire)
        exp.Next()
        i += 1

    return outer_wire, inner_wires


def get_face_center(face: TopoDS_Face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    center = props.CentreOfMass()
    return (center.X(), center.Y(), center.Z())


def get_sketch_radius_range(
    sketch: Sketch | TopoDS_Face | cq.Face,
    point: tuple | str = (0, 0, 0),
) -> tuple[float, float]:

    if isinstance(sketch, cq.Face):
        sketch_occ = sketch.wrapped
    elif isinstance(sketch, TopoDS_Face):
        sketch_occ = sketch
    else:
        sketch_occ = sketch.to_shape()

    if point == "center":
        point = get_face_center(sketch_occ)  # type: ignore

    outer_wire, inner_wires = get_outer_and_inner_wires(sketch_occ)  # type: ignore
    max_extent = distance_shape_to_shape(
        BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex(), outer_wire
    )

    inner_radii = [
        (
            distance_shape_to_shape(
                BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex(), inner_wire
            )
            if is_point_in_wire(inner_wire, gp_Pnt(*point))
            else 0
        )
        for inner_wire in inner_wires
    ]
    min_extent = max(inner_radii) if inner_radii else 0
    return min_extent, max_extent


def is_face_outer(part: cq.Workplane, face: cq.Face, tolerance=1e-3) -> bool:
    point_on_face = get_point_on_face(face)
    normal = face.normalAt(point_on_face).normalized()  # type: ignore

    p_start = cq.Vector(point_on_face).add(normal.multiply(tolerance))
    p_end = cq.Vector(point_on_face).add(
        normal.multiply(10000)
    )  # Arbitrary large distance

    probe_ray = cq.Edge.makeLine(p_start, p_end)

    intersection = part.val().intersect(probe_ray)  # type: ignore

    s = set()
    for b in intersection.Edges():
        s.add(b.startPoint().toTuple())
        s.add(b.endPoint().toTuple())
    return len(s) % 2 == 0


def normal_distance_to_next_face(part: cq.Workplane, face: cq.Face) -> float:
    point_on_face = cq.Vector(get_point_on_face(face))
    normal = face.normalAt(point_on_face).normalized()  # type: ignore

    p_start = point_on_face
    p_end = point_on_face.add(normal.multiply(-10000))  # Arbitrary large distance

    probe_ray = cq.Edge.makeLine(p_start, p_end)

    intersection = part.val().intersect(probe_ray)  # type: ignore

    return intersection.Edges()[0].Length()


def is_face_on_workplane(
    face: cq.Face, workplane: cq.Workplane, tol: float = 1e-9
) -> bool:
    if face.geomType() != "PLANE":
        return False

    wp_plane = workplane.plane
    wp_normal = wp_plane.zDir
    wp_origin = wp_plane.origin

    point_on_face = get_point_on_face(face)
    face_normal = face.normalAt(point_on_face)  # type: ignore

    n1 = wp_normal.normalized()
    n2 = face_normal.normalized()

    dot_prod = n1.dot(n2)
    if abs(dot_prod - 1.0) > tol:
        return False

    vec_to_face = cq.Vector(point_on_face) - wp_origin
    distance = vec_to_face.dot(n1)

    if abs(distance) > tol:
        return False

    return True


def get_faces_on_plane(r: cq.Workplane, workplane: cq.Workplane):
    faces = r.val().Faces()  # type: ignore

    faces_on_plane = [
        face for face in faces if is_face_on_workplane(face, workplane)  # type: ignore
    ]
    return faces_on_plane


def get_point_on_sketch_and_radius_range(
    sketch: Sketch | TopoDS_Face | cq.Face, p_center=0.2
) -> tuple[tuple[float, float, float], float, float]:
    if isinstance(sketch, cq.Face):
        sketch_occ = sketch.wrapped
    elif isinstance(sketch, TopoDS_Face):
        sketch_occ = sketch
    else:
        sketch_occ = sketch.to_shape()
    # center = get_face_center(sketch_occ)

    if np.random.random() < p_center:
        point = get_face_center(sketch_occ)  # type: ignore
    else:
        point = get_point_on_face(sketch_occ)
    min_extent, max_extent = get_sketch_radius_range(sketch_occ, point)  # type: ignore
    # point = (point[0] - center[0], point[1] - center[1])
    return point, min_extent, max_extent


def put_sketch_on_face(
    sketch: Sketch, face: TopoDS_Face, workplane: cq.Workplane
) -> None:
    center = get_face_center(sketch.to_shape())  # type: ignore
    point = get_point_on_face(face)

    side_cut_sketch_radius_min, side_cut_sketch_radius_max = get_sketch_radius_range(
        face, point
    )
    side_cut_sketch_radius = np.random.uniform(
        max(side_cut_sketch_radius_min, 0.5 * side_cut_sketch_radius_max),
        side_cut_sketch_radius_max,
    )

    sketch_bbox = cq.Face(sketch.to_shape()).BoundingBox()  # type: ignore
    sketch_radius = (
        max(sketch_bbox.xmax - sketch_bbox.xmin, sketch_bbox.ymax - sketch_bbox.ymin)
        / 2
    )
    side_cut_sketch_scale = side_cut_sketch_radius / sketch_radius

    origin = np.array(
        [
            workplane.plane.origin.x,
            workplane.plane.origin.y,
            workplane.plane.origin.z,
        ]
    )
    x_axis = np.array(
        [workplane.plane.xDir.x, workplane.plane.xDir.y, workplane.plane.xDir.z]
    )
    y_axis = np.array(
        [workplane.plane.yDir.x, workplane.plane.yDir.y, workplane.plane.yDir.z]
    )

    # center_local = np.array(center) - origin
    # center = [center_local.dot(x_axis), center_local.dot(y_axis)]
    sketch.transform((-center[0], -center[1]), side_cut_sketch_scale)

    point_local = np.array(point) - origin
    point = [point_local.dot(x_axis), point_local.dot(y_axis)]
    sketch.transform((point[0], point[1]), 1)


def get_inner_holes_radii(
    r_start: float, n_inner_holes: int, r_min: float = 0.0
) -> list[float]:
    inner_holes_radii = []
    r_cur = r_start
    for _ in range(n_inner_holes):
        cur_min = max(r_min, 0.5 * r_cur)
        r_new = np.random.uniform(cur_min, 0.95 * r_cur)
        inner_holes_radii.append(r_new)
        r_cur = r_new

    return inner_holes_radii


def get_hole_extents(hole_outer_radius, hole_inner_radii, extent):
    full = np.random.uniform() < 0.5
    total_extent = 0
    if not full:
        extent = np.random.uniform(0.25 * extent, extent)
    if hole_outer_radius is not None:
        outer_extent = np.random.uniform(0.05 * extent, 0.25 * extent)
    else:
        outer_extent = 0
    total_extent += outer_extent
    cur_extent = extent - outer_extent
    inner_extents = []
    if hole_inner_radii is not None:
        for _ in range(len(hole_inner_radii) - 1):
            inner_extent = np.random.uniform(0.1 * cur_extent, 0.5 * cur_extent)
            total_extent += inner_extent
            cur_extent -= inner_extent
            inner_extents.append(inner_extent)
        inner_extents.append(cur_extent)
        total_extent += cur_extent
    return outer_extent, inner_extents, full, total_extent


def shape_intersects_shape(shape1, shape2):
    common = BRepAlgoAPI_Common(shape1.wrapped, shape2.wrapped)
    common.Build()
    if not common.IsDone():
        return False
    result = common.Shape()

    for t in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE):
        exp = TopExp_Explorer(result, t)
        a = exp.More()
        if a:
            return True

    return False


def is_projection_of_hole(
    face, obj, direction: tuple[int, int, int], length: float = 1000
):
    parser = CADQuerySyntacticParser()
    workplane = cq.Workplane(cq.Plane(origin=(0, 0, 0), normal=direction))

    point = parser.get_point_on_face(face, workplane=workplane)
    point = cq.Vector(*point)
    direction_vec = cq.Vector(*direction)
    p_start = point - length * direction_vec
    p_end = point + length * direction_vec
    line = cq.Edge.makeLine(p_start, p_end)

    return not shape_intersects_shape(line, obj)


def get_wire_boxes_and_areas(wires):
    bboxes = []
    areas = []
    for wire in wires:
        wire_shape = cq.Shape(wire)
        bbox = wire_shape.BoundingBox()
        # ? TODO why cq.Shape
        area = cq.Face.makeFromWires(wire_shape).Area()  # type: ignore
        bboxes.append(bbox)
        areas.append(area)
    return bboxes, areas


def rearrange_wires(wires, ids=None):
    if len(wires) == 0:
        return {}
    if ids is None:
        ids = list(range(len(wires)))
    boxes, areas = get_wire_boxes_and_areas(wires)
    order = np.argsort(areas)[::-1]
    res = {}
    done = [False] * len(boxes)
    for i in order:
        if done[i]:
            continue
        res[ids[i]] = []
        for j in order:
            if i == j or done[j]:
                continue
            if (
                boxes[i].xmin <= boxes[j].xmin
                and boxes[i].xmax >= boxes[j].xmax
                and boxes[i].ymin <= boxes[j].ymin
                and boxes[i].ymax >= boxes[j].ymax
                and boxes[i].zmin <= boxes[j].zmin
                and boxes[i].zmax >= boxes[j].zmax
            ):
                res[ids[i]].append(j)
                done[j] = True
        done[i] = True
    for k, v in res.items():
        res[k] = rearrange_wires([wires[j] for j in v], [ids[j] for j in v])
    return res


def traverse_wire_arrangement(
    wire_arrangement, i, parent=None, wire_index_to_add=None
) -> tuple[Any, Any]:
    if len(wire_arrangement) == 0:
        return None, None
    if i in wire_arrangement:
        if wire_index_to_add is not None:
            wire_arrangement[i][wire_index_to_add] = {}
            return None, None
        else:
            return parent, wire_arrangement
    for k, v in wire_arrangement.items():
        p, wa = traverse_wire_arrangement(v, i, k, wire_index_to_add)
        if (p, wa) != (None, None):
            return p, wa
        return None, None

    return None, None


def update_side_cut_sketch(wires_dict, wire_arrangement_dict, sketch, w, direction):
    wires = wires_dict[direction]
    wire_arrangement = wire_arrangement_dict[direction]

    for q in range(1):
        if q == 2:
            raise ValueError("Got only holes.")

        chosen_wire_idx = np.random.randint(0, len(wires))
        parent_wire_idx, chosen_wire_arrangement = traverse_wire_arrangement(
            wire_arrangement, chosen_wire_idx
        )
        if parent_wire_idx is None or np.random.uniform() < 0.5:
            outer_wire_idx = chosen_wire_idx
            face = cq.Face.makeFromWires(
                cq.Wire(wires[chosen_wire_idx]),
                [cq.Wire(wires[i]) for i in chosen_wire_arrangement[chosen_wire_idx]],
            )
        else:
            outer_wire_idx = parent_wire_idx
            face = cq.Face.makeFromWires(
                cq.Wire(wires[parent_wire_idx]),
                [cq.Wire(wires[i]) for i in chosen_wire_arrangement],
            )

        if isinstance(face.wrapped, TopoDS_Compound):
            faces = []
            exp = TopExp_Explorer(face.wrapped, TopAbs_FACE)
            while exp.More():
                face_tmp = TopoDS.Face_s(exp.Current())
                faces.append(face_tmp)
                exp.Next()
            face = cq.Face(np.random.choice(faces))

        if is_projection_of_hole(face, w, direction):
            continue

        face_radius_min, face_radius_max = get_sketch_radius_range(face, point="center")
        face_offset = offset_face_inward(
            face, 0.2 * (face_radius_max - face_radius_min)
        )

        if isinstance(face_offset.wrapped, TopoDS_Compound):
            faces = []
            exp = TopExp_Explorer(face_offset.wrapped, TopAbs_FACE)
            while exp.More():
                face_tmp = TopoDS.Face_s(exp.Current())
                faces.append(face_tmp)
                exp.Next()
            face_offset = cq.Face(np.random.choice(faces))

        if direction == (1, 0, 0):
            workplane = cq.Workplane("ZY")
        elif direction == (0, 1, 0):
            workplane = cq.Workplane("ZX")
        else:
            workplane = cq.Workplane("XY")

        parser = CADQuerySyntacticParser()
        point = parser.get_point_on_face(face_offset)

        s = sketch.to_string_one_sketch(
            "workplane", sketch.wires, skip_on_one_wire=True
        )
        exec("r=" + s + ".finalize()", {}, locals())
        new_sketch_wire = locals()["r"].val()
        new_wire = new_sketch_wire._faces.Faces()[0].Wires()[0]
        wires.append(new_wire.wrapped)
        traverse_wire_arrangement(
            wire_arrangement,
            outer_wire_idx,
            parent=None,
            wire_index_to_add=len(wires) - 1,
        )
        wires_dict[direction] = wires
        wire_arrangement_dict[direction] = wire_arrangement

        center = get_face_center(new_sketch_wire._faces.Faces()[0].wrapped)

        side_cut_sketch_radius_min, side_cut_sketch_radius_max = (
            get_sketch_radius_range(face, point)
        )
        side_cut_sketch_radius = np.random.uniform(
            max(side_cut_sketch_radius_min, 0.5 * side_cut_sketch_radius_max),
            side_cut_sketch_radius_max,
        )

        sketch_bbox = cq.Face(sketch.to_shape()).BoundingBox()
        sketch_radius = (
            max(
                sketch_bbox.xmax - sketch_bbox.xmin, sketch_bbox.ymax - sketch_bbox.ymin
            )
            / 2
        )
        side_cut_sketch_scale = side_cut_sketch_radius / sketch_radius

        sketch.transform((-center[0], -center[1]), side_cut_sketch_scale)

        if direction == (1, 0, 0):
            sketch.transform((point[0], -point[1]), 1)
        elif direction == (0, 1, 0):
            sketch.transform((point[0], point[1]), 1)
        else:
            sketch.transform((point[0], point[1]), 1)

        break


def update_side_cut_sketches(
    wires_dict,
    wire_arrangement_dict,
    sketches,
    w,
    directions: list[tuple[int, int, int]],
):
    for i in range(len(sketches)):
        update_side_cut_sketch(
            wires_dict, wire_arrangement_dict, sketches[i], w, directions[i]
        )


def is_point_in_wire(wire, point: gp_Pnt, tolerance=1e-6):
    face = BRepBuilderAPI_MakeFace(wire, True).Face()

    classifier = BRepClass_FaceClassifier()
    classifier.Perform(face, point, tolerance)

    state = classifier.State()

    return state in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON)


def get_edge_midpoint(edge: cq.Edge) -> tuple[float, float, float]:
    topo_edge = edge.wrapped

    curve_adaptor = BRepAdaptor_Curve(topo_edge)

    first = curve_adaptor.FirstParameter()
    last = curve_adaptor.LastParameter()
    mid = (first + last) / 2.0

    p = curve_adaptor.Value(mid)

    return p.X(), p.Y(), p.Z()


def get_point_on_face(
    face, n_samples=1000, tol=1e-3, sketch=None, workplane=None, get_wrapped=True
):
    # _face = face.wrapped if get_wrapped else face
    _face = face.wrapped if isinstance(face, cq.Face) else face

    adaptor = BRepAdaptor_Surface(_face)
    classifier = BRepClass_FaceClassifier()
    # umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
    # vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(_face)

    for _ in range(n_samples):
        u = np.random.uniform(umin, umax)
        v = np.random.uniform(vmin, vmax)

        value = adaptor.Value(u, v)
        classifier.Perform(_face, value, tol)

        if classifier.State() in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON):
            point = (value.X(), value.Y(), value.Z())
            if sketch is not None:
                point = sketch.parent.plane.toWorldCoords(point).toTuple()
            elif workplane is not None:
                point = workplane.plane.toWorldCoords(point).toTuple()
            return point

    raise ValueError(
        f"Of {n_samples} sampled points, none is in the interior "
        f"of face within {tol} tolerance."
    )


def is_planar_face(face):
    return face.geomType() == "PLANE"


def get_face_2d_radius(
    face: cq.Face,
    point_on_face: tuple[float, float, float],
    return_transform: bool = False,
):
    geom_surf = BRep_Tool.Surface_s(face.wrapped)

    p = face.Center()

    proj = GeomAPI_ProjectPointOnSurf(
        gp_Pnt(point_on_face[0], point_on_face[1], point_on_face[2]), geom_surf
    )

    u, v = proj.LowerDistanceParameters()
    surf = BRepAdaptor_Surface(face.wrapped)
    pnt = gp_Pnt()
    du = gp_Vec()
    dv = gp_Vec()

    surf.D1(u, v, pnt, du, dv)

    origin = cq.Vector(*point_on_face)
    t1 = cq.Vector(du).normalized()
    dv = cq.Vector(dv)
    normal = t1.cross(dv).normalized()
    t2 = normal.cross(t1).normalized()

    pts2d = []

    for e in face.Edges():
        for t in [i / 20 for i in range(21)]:
            p = e.positionAt(t)
            vec = cq.Vector(p) - origin
            x = vec.dot(t1)
            y = vec.dot(t2)

            pts2d.append((x, y))

    pts2d = np.array(pts2d)
    bbox = [pts2d.min(axis=0), pts2d.max(axis=0)]
    w = min(abs(bbox[0][0]), abs(bbox[1][0]))
    h = min(abs(bbox[0][1]), abs(bbox[1][1]))
    transform = (origin, t1, t2)
    if return_transform:
        return min(w, h).item(), *transform
    else:
        return min(w, h).item()


def put_sketch_on_face_simplified(
    sketch: Sketch, face: cq.Face, point_on_face: tuple[float, float, float]
) -> None:
    radius, origin, x_axis, y_axis = get_face_2d_radius(
        face, point_on_face, return_transform=True
    )

    center = get_face_center(sketch.to_shape())  # type: ignore
    # center_local = np.array(center) - np.array(origin.toTuple())
    # center = [center_local.dot(x_axis.toTuple()), center_local.dot(y_axis.toTuple())]

    side_cut_sketch_radius = np.random.uniform(
        0.5 * radius,
        0.95 * radius,
    )

    sketch_bbox = cq.Face(sketch.to_shape()).BoundingBox()  # type: ignore
    sketch_radius = (
        max(sketch_bbox.xmax - sketch_bbox.xmin, sketch_bbox.ymax - sketch_bbox.ymin)
        / 2
    )
    side_cut_sketch_scale = side_cut_sketch_radius / sketch_radius

    sketch.transform((-center[0], -center[1]), side_cut_sketch_scale)


def offset_face_inward(face: cq.Face, offset: float) -> cq.Face:
    try:
        outer_wire_offset = face.outerWire().offset2D(-offset)[0]

        inner_wires_offset = [
            inner_wire.offset2D(offset)[0] for inner_wire in face.innerWires()
        ]

        offset_face = cq.Face.makeFromWires(outer_wire_offset, inner_wires_offset)

        return offset_face
    except Exception as e:
        raise ValueError(
            f"Failed to offset face by {offset}. "
            f"The offset may be too large. Original error: {e}"
        )


def couple_list(lst):
    res = []
    for i in range(0, len(lst) - 1, 2):
        res.append((lst[i], lst[i + 1]))
    if len(lst) % 2:
        res.append((lst[-1], None))
    return res


def argsort_points(points):
    return np.lexsort(np.array(points).round(7).transpose()[[1, 0]])


def float_to_string(n):
    if np.isclose(n, round(n)):
        return str(round(n))
    return f"{n:.1f}"
