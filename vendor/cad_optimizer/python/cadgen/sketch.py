from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import cadlib.core as clc  # type: ignore
import cadlib.features as clf  # type: ignore
import numpy as np
from cadaxt.core import CADAxtProfile
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GC import GC_MakeArcOfCircle
from OCP.GeomAbs import GeomAbs_CurveType
from OCP.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Trsf
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from sg_parser.sample_sketch import sample_profile_from_sketch_dict  # type: ignore

from .base import BaseFactory, BaseOperation
from .registry import factories
from .utils import (
    argsort_points,
    float_to_string,
    get_point_on_sketch_and_radius_range,
    shape_to_area,
    shape_to_bbox,
)

logger = logging.getLogger(__name__)


class Sketch(BaseOperation):
    prev_center: tuple[float, float] | None = None
    array_pattern_sketch: Sketch | None = None
    array_pattern_n: int | None = None
    array_pattern_radius: float | None = None
    array_pattern_dx: float | None = None
    array_pattern_dy: float | None = None
    array_pattern_nx: int | None = None
    array_pattern_ny: int | None = None
    array_pattern_type: str | None = None
    array_pattern_mode: str | None = None
    array_pattern_start_angle: float | None = None
    array_pattern_span: float | None = None
    """
    wires = [{
            'type': 'polygon',
            'outer': True,
            'vertices': [[0, 0], [1, 1], ...]
            'edges': [
                {'type': 'line'},
                {'type': 'arc', 'point': [2, 2]}, ...]
        }, {
            'type': 'circle',
            'outer': False,
            'center': [1, 1],
            'radius': 2
        }, ...]
    """

    def __init__(self, wires):
        self.wires = wires

    @staticmethod
    def wire_to_rectangle(wire):
        # -> (x, y, w, h) or None
        vertices = wire["vertices"]
        if len(vertices) == 4:
            if set(e["type"] for e in wire["edges"]) == set(["line"]):
                x, y = zip(*vertices)
                if len(set(x)) == len(set(y)) == 2:
                    return (
                        (min(x) + max(x)) / 2,
                        (min(y) + max(y)) / 2,
                        max(x) - min(x),
                        max(y) - min(y),
                    )
        return None

    def to_string_one_sketch(
        self, plane, wires, skip_on_one_wire=False, short_mode=False, use_tag=False
    ):
        context = [0, 0]
        s_short = ""
        # tag = "r.workplane(origin=r.val().Center())" if "faces" in plane else plane
        if not use_tag:
            tag = plane.split("=")[-1] if "faces" in plane else plane
        else:
            tag = plane.split("\n")[2].split("=")[0] if "faces" in plane else plane
        if tag.startswith(".copyWorkplane"):
            tag = "w0" + tag
        if not use_tag:
            s = f"{plane}.sketch()" if not short_mode else tag
        else:
            s = tag
            s += ".sketch()" if not short_mode else ""

        for i, wire in enumerate(wires):
            m = ",mode='s'" if not wire["outer"] else ""
            if wire["type"] == "circle":
                x, y = wire["center"]
                if not np.allclose([x, y], context):
                    context = [x, y]
                    s += f".push([({x},{y})])"
                r = wire["radius"]
                if short_mode:
                    s_short += f".circle({r}{m})"
                else:
                    s += f".circle({r}{m})"
            else:  # 'polygon'
                rectangle = self.wire_to_rectangle(wire)
                # rectangle
                if rectangle is not None:
                    x, y, w, h = rectangle
                    if not np.allclose([x, y], context):
                        context = [x, y]
                        x = float_to_string(x)
                        y = float_to_string(y)
                        s += f".push([({x},{y})])"
                    if short_mode:
                        s_short += f".rect({w},{h}{m})"
                    else:
                        s += f".rect({w},{h}{m})"
                else:
                    t = ""
                    for j, edge in enumerate(wire["edges"]):
                        if j == 0:
                            x, y = wire["vertices"][0]
                            u = f"({x},{y}),"
                        else:
                            u = ""

                        n_edges = len(wire["edges"])
                        x, y = wire["vertices"][(j + 1) % n_edges]
                        if edge["type"] == "line":
                            if j < n_edges - 1:
                                t += f".segment({u}({x},{y}))"
                            else:
                                t += ".close()"
                        else:  # 'arc'
                            mx, my = edge["point"]
                            t += f".arc({u}({mx},{my}),({x},{y}))"

                    t += ".assemble()"
                    if i == 0:
                        s += t
                    else:
                        s += f".reset().face({tag}.sketch(){t}{m})"

        if short_mode and s_short != "":
            return s_short
        else:
            return s

    def to_string(
        self,
        plane,
    ):
        s = self.to_string_one_sketch(
            plane,
            self.wires,
            skip_on_one_wire=self.array_pattern_sketch is not None,
        )
        # TODO think where this should go
        # if ".faces" in s:
        #     extent = abs(extent)
        #     if hole_outer_extent:
        #         hole_outer_extent = abs(hole_outer_extent)
        #     if hole_inner_extents:
        #         for i in range(len(hole_inner_extents)):
        #             hole_inner_extents[i] = abs(hole_inner_extents[i])

        point = self.point if hasattr(self, "point") else (0, 0)

        if self.array_pattern_sketch is not None:
            if self.array_pattern_mode == "s":
                self.array_pattern_sketch.wires[0]["outer"] = False
            array_pattern_string = self.to_string_one_sketch(
                plane,
                self.array_pattern_sketch.wires,
                skip_on_one_wire=True,
                short_mode=True,
            )
            self.array_pattern_sketch.wires[0]["outer"] = True
            reset = ".reset()" if "push" not in s else ""
            if point and point != (0, 0):
                point_push = f".push([({point[0]},{point[1]})])"
                point = (0, 0)
            else:
                point_push = ""

            if self.array_pattern_type == "parray":
                s += f"{reset}{point_push}.parray({self.array_pattern_radius}, {self.array_pattern_start_angle}, {self.array_pattern_span}, {self.array_pattern_n})"
            elif self.array_pattern_type == "rarray":
                s += f"{reset}{point_push}.rarray({self.array_pattern_dx}, {self.array_pattern_dy}, {self.array_pattern_nx}, {self.array_pattern_ny})"
            if (
                "rect" in array_pattern_string or "circle" in array_pattern_string
            ) and "push" not in array_pattern_string:
                s += array_pattern_string
            else:
                array_pattern_string = self.to_string_one_sketch(
                    plane,
                    self.array_pattern_sketch.wires,
                    skip_on_one_wire=True,
                    use_tag=True,
                )
                s += f".face({array_pattern_string}.wires(), mode='{self.array_pattern_mode}')"

        s += ".finalize()"

        # check that solid
        logger.info("Sketch code: %s", s)
        if self.array_pattern_sketch is not None:
            pass
            # exec(
            #     f"import cadquery as cq\nw0=cq.Workplane('XY')\nw1=cq.Workplane('XY')\nr={s}",
            #     {},
            #     locals(),
            # )
            # w = locals()["r"].val()
            # mesh = compound_to_mesh(w)
            # assert len(mesh.split()) == 1, f"Mesh has {len(mesh.split())} parts"

        return s

    def cq_bounding_box(self, sketch):
        boxes = []
        for face in sketch.val()._faces.Faces():
            face_box = face.BoundingBox()
            boxes.append((face_box.xmin, face_box.ymin, face_box.xmax, face_box.ymax))

        xmin = min(boxes, key=lambda x: x[0])[0]
        ymin = min(boxes, key=lambda x: x[1])[1]
        xmax = max(boxes, key=lambda x: x[2])[2]
        ymax = max(boxes, key=lambda x: x[3])[3]
        return xmin, ymin, xmax, ymax

    def to_shape(self):
        # -> TopoDS_Compound
        compound = None
        face_builder: BRepBuilderAPI_MakeFace | None = None
        for i, wire in enumerate(self.wires):
            edges = self.wire_to_edges(wire)
            wire_builder = BRepBuilderAPI_MakeWire(edges[0])
            for edge in edges[1:]:
                wire_builder.Add(edge)

            if wire["outer"]:
                face_builder = BRepBuilderAPI_MakeFace(wire_builder.Wire())
            else:
                assert face_builder is not None, "face_builder is required"
                face_builder.Add(TopoDS.Wire_s(wire_builder.Wire().Reversed()))

            # make face if it is last inner wire
            if i == len(self.wires) - 1 or self.wires[i + 1]["outer"]:
                if compound is not None:
                    compound = BRepAlgoAPI_Fuse(compound, face_builder.Face()).Shape()
                else:
                    compound = face_builder.Face()
        return compound

    def transform(self, shift, scale):
        for wire in self.wires:
            if wire["type"] == "circle":
                wire["center"][0] = (wire["center"][0] + shift[0]) * scale
                wire["center"][1] = (wire["center"][1] + shift[1]) * scale
                wire["radius"] = wire["radius"] * scale
            else:  # 'polygon'
                for vertex in wire["vertices"]:
                    vertex[0] = (vertex[0] + shift[0]) * scale
                    vertex[1] = (vertex[1] + shift[1]) * scale
                for edge in wire["edges"]:
                    if edge["type"] == "arc":
                        edge["point"][0] = (edge["point"][0] + shift[0]) * scale
                        edge["point"][1] = (edge["point"][1] + shift[1]) * scale

        assert self.prev_center is not None
        prev_center_x, prev_center_y = self.prev_center
        prev_center_x = (prev_center_x + shift[0]) * scale
        prev_center_y = (prev_center_y + shift[1]) * scale
        self.prev_center = (prev_center_x, prev_center_y)

        if hasattr(self, "point"):
            point_x, point_y = self.point
            point_x = (point_x + shift[0]) * scale
            point_y = (point_y + shift[1]) * scale
            self.point = (point_x, point_y)

        # Transform sketch attributes
        if self.array_pattern_radius:
            self.array_pattern_radius *= scale
        if self.array_pattern_dx:
            self.array_pattern_dx *= scale
        if self.array_pattern_dy:
            self.array_pattern_dy *= scale

        if self.array_pattern_sketch:
            self.array_pattern_sketch.transform([0, 0], scale)

    def round(self):
        for wire in self.wires:
            if wire["type"] == "circle":
                wire["center"][0] = round(wire["center"][0])
                wire["center"][1] = round(wire["center"][1])
                wire["radius"] = round(wire["radius"])
            else:  # 'polygon'
                for vertex in wire["vertices"]:
                    vertex[0] = round(vertex[0])
                    vertex[1] = round(vertex[1])
                for edge in wire["edges"]:
                    if edge["type"] == "arc":
                        edge["point"][0] = round(edge["point"][0])
                        edge["point"][1] = round(edge["point"][1])
        assert self.prev_center is not None
        self.prev_center = (round(self.prev_center[0]), round(self.prev_center[1]))

        if hasattr(self, "point"):
            self.point = (round(self.point[0]), round(self.point[1]))

        # Round sketch attributes
        if self.array_pattern_radius:
            self.array_pattern_radius = round(self.array_pattern_radius)
        if self.array_pattern_dx:
            self.array_pattern_dx = round(self.array_pattern_dx)
        if self.array_pattern_dy:
            self.array_pattern_dy = round(self.array_pattern_dy)

        if self.array_pattern_sketch:
            self.array_pattern_sketch.round()

    def reorder(self):
        n_wires = len(self.wires)
        bbox_corners = list()
        for wire in self.wires:
            outer_wire = deepcopy(wire)
            outer_wire["outer"] = True
            outer_face = Sketch([outer_wire]).to_shape()
            x_min, y_min, _, _, _, _ = shape_to_bbox(outer_face)
            bbox_corners.append([x_min, y_min])
        bbox_corners = np.array(bbox_corners)
        order = argsort_points(bbox_corners)

        # reorder outer wires by bottom left bbox corner
        # reorder inner wires by bottom left bbox corner
        wires = list()
        for i in order:
            if self.wires[i]["outer"]:
                wires.append(self.wires[i])
                inner_ids = list()
                for j in range(i + 1, len(self.wires)):
                    if self.wires[j]["outer"]:
                        break
                    inner_ids.append(j)
                if not len(inner_ids):
                    continue
                inner_order = argsort_points(bbox_corners[inner_ids])
                for j in inner_order:
                    wires.append(self.wires[inner_ids[j]])
        assert len(wires) == n_wires
        self.wires = wires

        # reorder vertices to start from bottom left
        for wire in self.wires:
            if wire["type"] == "circle":
                continue

            first = argsort_points(wire["vertices"])[0]
            n = len(wire["vertices"])
            wire["vertices"] = wire["vertices"][first:n] + wire["vertices"][:first]
            wire["edges"] = wire["edges"][first:n] + wire["edges"][:first]

    def orient(self):
        # all wires to be counterclockwise
        for wire in self.wires:
            if wire["type"] == "circle":
                continue

            points = list()
            for vertex, edge in zip(wire["vertices"], wire["edges"]):
                points.append(vertex)
                if edge["type"] == "arc":
                    points.append(edge["point"])

            points = np.array(points)
            x, y = points[:, 0], points[:, 1]
            # shoelace formula
            signed_area = np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            assert not np.isclose(signed_area, 0)
            if signed_area > 0:
                wire["vertices"] = wire["vertices"][::-1]
                wire["edges"] = wire["edges"][:-1][::-1] + [wire["edges"][-1]]

    @staticmethod
    def wire_to_edges(wire):
        edges = list()
        if wire["type"] == "circle":
            x, y = wire["center"]
            circ = gp_Circ(gp_Ax2(gp_Pnt(x, y, 0), gp_Dir(0, 0, 1)), wire["radius"])
            edges.append(BRepBuilderAPI_MakeEdge(circ).Edge())
        else:  # 'polygon'
            for j, edge in enumerate(wire["edges"]):
                x, y = wire["vertices"][j]
                xyz_first = gp_Pnt(x, y, 0)
                x, y = wire["vertices"][(j + 1) % len(wire["edges"])]
                xyz_last = gp_Pnt(x, y, 0)
                if edge["type"] == "line":
                    edge_builder = BRepBuilderAPI_MakeEdge(xyz_first, xyz_last)
                else:  # 'arc'
                    x, y = edge["point"]
                    xyz_middle = gp_Pnt(x, y, 0)
                    curve = GC_MakeArcOfCircle(xyz_first, xyz_middle, xyz_last).Value()
                    edge_builder = BRepBuilderAPI_MakeEdge(curve)
                edges.append(edge_builder.Edge())
        return edges

    def try_fix_wire(self, wire, state):
        # skip inner wires if outer failed
        assert wire["outer"] or state["face"] is not None

        # remove zero edges
        if wire["type"] == "circle":
            assert not np.isclose(wire["radius"], 0)
        else:
            ids = [0]
            for i in range(1, len(wire["vertices"])):
                if not np.allclose(wire["vertices"][i], wire["vertices"][i - 1]):
                    ids.append(i)
            if len(ids) > 1:
                if np.allclose(wire["vertices"][0], wire["vertices"][ids[-1]]):
                    ids = ids[:-1]
            if len(ids) != len(wire["vertices"]):
                assert len(ids) > 1
                wire["vertices"] = itemgetter(*ids)(wire["vertices"])
                wire["edges"] = itemgetter(*ids)(wire["edges"])

        # skip wires with invalid arcs
        # they could be replaced with lines, but then probably need to orient again
        try:
            edges = self.wire_to_edges(wire)
        except Exception as e:
            if str(e).startswith("StdFail_NotDoneGC_MakeArcOfCircle"):
                assert False, "invalid arc"
            # what can go wrong?
            raise

        # check intersection of all edges
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                n = 0
                if j == i + 1:
                    n += 1
                if (j + 1) % len(edges) == i:
                    n += 1
                extrema = BRepExtrema_DistShapeShape(edges[i], edges[j])
                if np.isclose(extrema.Value(), 0):
                    assert n == extrema.NbSolution()
                else:
                    assert n == 0

        # skip invalid wires
        outer_wire = deepcopy(wire)
        outer_wire["outer"] = True
        outer_face = Sketch([outer_wire]).to_shape()
        outer_area = shape_to_area(outer_face)
        assert not np.isclose(outer_area, 0) and outer_area > 0

        new_face = Sketch(state["face_wires"] + [wire]).to_shape()

        # skip intersecting wires
        if not wire["outer"]:
            # todo: we should check distance between wires not areas of faces
            face_area = shape_to_area(state["face"])
            new_area = shape_to_area(new_face)
            assert not np.isclose(new_area, 0) and new_area > 0
            assert np.isclose(face_area - outer_area, new_area)

        state["face"] = new_face
        state["face_wires"].append(wire)

    def fix_one_sketch(self, wires):
        # skip zero edges, invalid wires, intersecting wires, intersecting faces
        compound, wires_cur = None, list()
        state = dict(face=None, face_wires=list())

        for i, wire in enumerate(wires):
            try:
                self.try_fix_wire(deepcopy(wire), state)
            except AssertionError:
                pass

            if i == len(wires) - 1 or wires[i + 1]["outer"]:
                if state["face"] is not None:
                    new_wires = wires_cur + state["face_wires"]  # type: ignore
                    new_compound = Sketch(new_wires).to_shape()

                    # skip intersecting faces
                    if compound is not None:
                        compound_area = shape_to_area(compound)
                        face_area = shape_to_area(
                            Sketch(state["face_wires"]).to_shape()
                        )
                        new_area = shape_to_area(new_compound)
                        if not np.isclose(compound_area + face_area, new_area):
                            continue

                    wires_cur = new_wires
                    compound = new_compound
                state["face"] = None
                state["face_wires"] = list()

        assert len(wires_cur) > 0
        return wires_cur

    def fix(self):
        # skip zero edges, invalid wires, intersecting wires, intersecting faces
        self.wires = self.fix_one_sketch(self.wires)
        if self.array_pattern_sketch is not None:
            self.array_pattern_sketch.wires = self.fix_one_sketch(
                self.array_pattern_sketch.wires
            )


class SketchFactory(BaseFactory):
    try:
        sketches_root = Path(
            os.environ.get("SKETCHGRAPHS_ROOT", "")
        )
        sketches = json.loads((sketches_root / "valid_files.json").read_text())
        long_sketches = json.loads((sketches_root / "long_sketches.json").read_text())
    except:
        logger.info("Sketchgraph folder not found.")
    sketches_root: Path
    sketches: list[str | Path]
    long_sketches: list[str | Path]

    def __init__(
        self,
        min_n_commands,
        max_n_commands,
        n_outer_probabilities,
        rotation_probability,
        array_pattern_probability=0.0,
        array_pattern_sketch_min_n_commands=1,
        array_pattern_sketch_max_n_commands=1,
        array_pattern_sketch_min_rel_scale=0.01,
        array_pattern_cut_probability=1,
        parray_probability=0.5,
        from_sketchgraph: bool = False,
    ):
        # n_commands: number of rotated rectangles / circles
        # n_outer_probabilities: number of outer wires probablities
        # rotation_probability: probablity to rotate rectangle
        self.min_n_commands = min_n_commands
        self.max_n_commands = max_n_commands
        self.n_outer_probabilities = n_outer_probabilities
        self.rotation_probability = rotation_probability
        self.array_pattern_probability = array_pattern_probability
        self.array_pattern_sketch_min_n_commands = array_pattern_sketch_min_n_commands
        self.array_pattern_sketch_max_n_commands = array_pattern_sketch_max_n_commands
        self.array_pattern_sketch_min_rel_scale = array_pattern_sketch_min_rel_scale
        self.array_pattern_cut_probability = array_pattern_cut_probability
        self.parray_probability = parray_probability
        self.from_sketchgraph = from_sketchgraph

    @staticmethod
    def make_rectangle_face(x, y, w, h, angle):
        polygon = BRepBuilderAPI_MakePolygon()
        polygon.Add(gp_Pnt(x - w / 2, y - h / 2, 0))
        polygon.Add(gp_Pnt(x + w / 2, y - h / 2, 0))
        polygon.Add(gp_Pnt(x + w / 2, y + h / 2, 0))
        polygon.Add(gp_Pnt(x - w / 2, y + h / 2, 0))
        polygon.Close()
        wire = polygon.Wire()

        if not np.isclose(angle, 0):
            trsf = gp_Trsf()
            rotation_axis = gp_Ax1(gp_Pnt(x, y, 0), gp_Dir(0, 0, 1))
            trsf.SetRotation(rotation_axis, angle)
            wire = TopoDS.Wire_s(BRepBuilderAPI_Transform(wire, trsf, True).Shape())

        face = BRepBuilderAPI_MakeFace(wire).Face()
        return face

    @staticmethod
    def make_circle_face(x, y, radius):
        circ = gp_Circ(gp_Ax2(gp_Pnt(x, y, 0), gp_Dir(0, 0, 1)), radius)
        edge = BRepBuilderAPI_MakeEdge(circ).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        return face

    def try_update_compound(self, compound, state, only_cuts=False, zero_center=False):
        # try add a rectangle or circle to existing face
        r_min, r_max = 0.01, 0.5
        if np.random.randint(2) < 1:
            xy = list(state["prev_center"])
        else:
            xy = np.random.uniform(-1, 1, 2).tolist()
        if zero_center:
            xy = [0, 0]

        if (np.random.randint(2) < 1 or only_cuts) and compound is not None:
            algo = BRepAlgoAPI_Cut
            # try to be inside bounding box
            if np.random.randint(2) < 1:
                x_min, y_min, _, x_max, y_max, _ = shape_to_bbox(compound)
                if not x_min < xy[0] < x_max:
                    xy[0] = np.random.uniform(x_min, x_max)  # type: ignore
                if not y_min < xy[1] < y_max:
                    xy[1] = np.random.uniform(y_min, y_max)  # type: ignore
                r_max = min(x_max - xy[0], xy[0] - x_min, y_max - xy[1], xy[1] - y_min)
        else:
            algo = BRepAlgoAPI_Fuse

        r_max = max(r_min, r_max)
        if not self.from_sketchgraph:
            if np.random.randint(2) < 1:
                radius = np.random.uniform(r_min, r_max)
                face = self.make_circle_face(xy[0], xy[1], radius)
            else:
                wh = np.random.uniform(r_min, r_max * 2, 2)
                if np.random.rand() < self.rotation_probability:
                    angle = np.random.uniform(0, np.pi)
                else:
                    angle = 0
                face = self.make_rectangle_face(xy[0], xy[1], wh[0], wh[1], angle)
        else:
            face = None
            bbox = np.random.uniform(r_min, r_max) * 2
            while face is None:
                try:
                    if np.random.uniform() > 0.10:
                        sketchgraph_file = (
                            self.sketches_root / random.choice(self.sketches)
                            if self.sketches
                            else None
                        )
                    else:
                        sketchgraph_file = Path(random.choice(self.long_sketches))

                    sample = sample_profile_from_sketch_dict(sketchgraph_file)
                    profile = CADAxtProfile(
                        **list(sample["profiles"].values())[0], transform={}
                    )
                    face = clc.CAD().ocp.move_lower_left(
                        clc.CAD().ocp.rescale_to_area(
                            clf.Profile.from_dict(
                                data=profile.model_dump(),
                                sketch_plane=clc.CoordSystem(),
                            ).profile2face(),
                            bbox**2,
                        ),
                        xy[0] - bbox / 2,
                        xy[1] - bbox / 2,
                    )
                except Exception:
                    continue

        if compound is None:
            compound = face
        else:
            new_compound = algo(compound, face).Shape()
            unifier = ShapeUpgrade_UnifySameDomain(new_compound, True, True, True)
            unifier.Build()
            new_compound = unifier.Shape()
            new_area = shape_to_area(new_compound)
            # check that not everything is deleted
            assert not np.isclose(new_area, 0)
            # check that something is changed
            assert not np.isclose(
                shape_to_area(BRepAlgoAPI_Cut(face, new_compound).Shape()), 0
            ) or not np.isclose(
                shape_to_area(BRepAlgoAPI_Cut(new_compound, face).Shape()), 0
            )
            compound = new_compound

        return compound, dict(prev_center=xy)

    @staticmethod
    def compound_to_wires(compound):
        face_explorer = TopExp_Explorer(compound, TopAbs_FACE)  # type: ignore
        wires = list()
        while face_explorer.More():
            face = face_explorer.Current()
            face_wires, bbox_sizes = list(), list()
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)  # type: ignore
            while wire_explorer.More():
                wire = TopoDS.Wire_s(wire_explorer.Current())
                x_min, y_min, _, x_max, y_max, _ = shape_to_bbox(
                    BRepBuilderAPI_MakeFace(wire).Face()  # type: ignore
                )
                bbox_size = max(x_max - x_min, y_max - y_min)
                face_wires.append(wire)
                bbox_sizes.append(bbox_size)
                wire_explorer.Next()

            for i, wire_id in enumerate(np.argsort(bbox_sizes)[::-1]):
                outer = i == 0
                wires.append(dict(wire=face_wires[wire_id], outer=outer))
            face_explorer.Next()
        return wires

    @staticmethod
    def sample_wires(wires):
        # max_n_outer_wires = np.random.randint(1, self.max_n_outer_wires + 1)
        # p = self.n_outer_probabilities
        # max_n_outer_wires = np.random.choice(len(p), p=p) + 1
        max_n_outer_wires = 1
        n_outer_wires = 0
        i = -1
        for i, wire in enumerate(wires):
            if wire["outer"]:
                n_outer_wires += 1
            if n_outer_wires > max_n_outer_wires:
                i = i - 1
                break
        return wires[: i + 1]

    @staticmethod
    def wires_to_sketch(wires, state):
        sketch = list()
        for wire in wires:
            edges = list()
            edge_explorer = BRepTools_WireExplorer(wire["wire"])
            while edge_explorer.More():
                edge = TopoDS.Edge_s(edge_explorer.Current())
                curve = BRepAdaptor_Curve(edge)
                u_first = curve.FirstParameter()
                u_last = curve.LastParameter()
                # edge = topods.Edge(edge_explorer.Current())
                # curve, u_first, u_last = BRep_Tool.Curve(edge)
                xy_first = curve.Value(u_first)
                xy_first = [xy_first.X(), xy_first.Y()]

                xy_middle = curve.Value((u_first + u_last) / 2)
                xy_middle = [xy_middle.X(), xy_middle.Y()]
                xy_last = curve.Value(u_last)
                xy_last = [xy_last.X(), xy_last.Y()]
                edges.append(
                    dict(
                        curve=curve,
                        xy_first=xy_first,
                        xy_middle=xy_middle,
                        xy_last=xy_last,
                    )
                )
                edge_explorer.Next()

            assert len(edges) > 0
            if len(edges) == 1:  # circle
                edge = edges[0]
                curve_type = edge["curve"].GetType()
                assert curve_type == GeomAbs_CurveType.GeomAbs_Circle
                assert np.allclose(edge["xy_first"], edge["xy_last"])
                circ = edge["curve"].Circle()

                radius = circ.Radius()
                center = circ.Location()
                center = [center.X(), center.Y()]
                sketch.append(
                    dict(
                        type="circle", outer=wire["outer"], center=center, radius=radius
                    )
                )
            else:  # polygon
                # if end if 1s edge is not begin or end ot the 2nd, than reverse 1st
                if not np.allclose(
                    edges[0]["xy_last"], edges[1]["xy_first"]
                ) and not np.allclose(edges[0]["xy_last"], edges[1]["xy_last"]):
                    xy = edges[0]["xy_first"]
                    edges[0]["xy_first"] = edges[0]["xy_last"]
                    edges[0]["xy_last"] = xy
                vertices = [edges[0]["xy_first"]]
                segments = list()
                for edge in edges:
                    assert not np.allclose(edge["xy_first"], edge["xy_last"])
                    # if end of begin edge is not end of previous, than reverse this
                    if not np.allclose(edge["xy_first"], vertices[-1]):
                        xy = edge["xy_first"]
                        edge["xy_first"] = edge["xy_last"]
                        edge["xy_last"] = xy
                    assert np.allclose(edge["xy_first"], vertices[-1])
                    vertices.append(edge["xy_last"])
                    if (
                        edge["curve"].GetType() == GeomAbs_CurveType.GeomAbs_Line
                    ):  # line
                        segments.append(dict(type="line"))
                    else:  # arc
                        segments.append(dict(type="arc", point=edge["xy_middle"]))
                assert np.allclose(vertices[0], vertices[-1])
                sketch.append(
                    dict(
                        type="polygon",
                        outer=wire["outer"],
                        vertices=vertices[:-1],
                        edges=segments,
                    )
                )
        sketch = Sketch(sketch)
        sketch.prev_center = state["prev_center"]
        return sketch

    def generate_one_sketch(
        self, min_n_commands, max_n_commands, only_cuts=False, zero_center=False
    ):
        prev_center = (0, 0)

        # construct Sketch from random rectangles and circles
        compound = None
        state = dict(prev_center=prev_center)
        n_commands = np.random.randint(min_n_commands, 1 + max_n_commands)

        for _ in range(n_commands):
            try:
                compound, state = self.try_update_compound(
                    compound, state, only_cuts=only_cuts, zero_center=zero_center
                )
            except AssertionError:
                pass

        wires = self.compound_to_wires(compound)
        wires = self.sample_wires(wires)
        sketch = self.wires_to_sketch(wires, state)
        logger.info("Generated sketch: %s", sketch.__dict__)

        return sketch

    def generate(self, zero_center=False):
        do_array_pattern = np.random.random() < self.array_pattern_probability

        array_pattern_sketch = None
        array_pattern_start_angle = None
        array_pattern_span = None
        array_pattern_radius = None
        array_pattern_mode = None
        array_pattern_sketch = None
        array_pattern_n = None
        array_pattern_type = None
        array_pattern_dx = None
        array_pattern_dy = None
        array_pattern_nx = None
        array_pattern_ny = None
        min_extent = None
        point = (0, 0)
        sketch: Sketch | None = None

        if do_array_pattern:
            logger.info("Array pattern enabled")
            array_pattern_sketch = self.generate_one_sketch(
                self.array_pattern_sketch_min_n_commands,
                self.array_pattern_sketch_max_n_commands,
                zero_center=True,
            )
            array_pattern_n = np.random.randint(2, 40)
            array_pattern_start_angle = np.random.randint(0, 360)
            array_pattern_span = (
                360 if np.random.random() < 0.2 else np.random.randint(0, 360)
            )

            array_pattern_type = (
                "parray" if np.random.random() < self.parray_probability else "rarray"
            )
            array_pattern_mode = (
                "s"
                if np.random.uniform() < self.array_pattern_cut_probability
                or array_pattern_type == "rarray"
                else "a"
            )
            logger.info(
                f"Array pattern: type={array_pattern_type}, n={array_pattern_n}, mode={array_pattern_mode}, start_angle={array_pattern_start_angle}, span={array_pattern_span}"
            )

            if array_pattern_type == "parray":
                sketch = self.generate_one_sketch(
                    1, 1 + self.max_n_commands, only_cuts=True, zero_center=zero_center
                )
                point, min_extent, max_extent = get_point_on_sketch_and_radius_range(
                    sketch
                )
                if array_pattern_mode == "s":

                    array_pattern_radius = np.random.uniform(
                        (
                            min_extent
                            if min_extent > 0.5 * max_extent
                            else 0.5 * max_extent
                        ),
                        max_extent * 0.95,
                    )
                    k = array_pattern_span / 360
                    possible_tseconds = (
                        2
                        * array_pattern_radius
                        * np.sin(np.pi * k / np.arange(2, array_pattern_n + 1))
                    )
                    # tmp = possible_tseconds[
                    #     possible_tseconds > self.array_pattern_sketch_min_rel_scale
                    # ]
                    tmp = possible_tseconds
                    # creating more space so it is not narrowly stacked
                    tsecond = (
                        tmp[-2]
                        if len(tmp) > 2
                        else tmp[-1] if len(tmp) > 1 else tmp[-1]
                    )
                    array_pattern_n = len(tmp)

                    t = min(
                        1 - array_pattern_radius / max_extent,
                        tsecond,
                    )
                    array_pattern_sketch_rel_scale = tsecond
                    # array_pattern_sketch_rel_scale = np.random.uniform(
                    #     self.array_pattern_sketch_min_rel_scale, t
                    # )
                else:
                    t = np.random.uniform(0.1, 0.7)
                    array_pattern_radius = np.random.uniform(
                        max_extent * (1 - 0.5 * t), max_extent * (1 + 0.5 * t)
                    )
                    array_pattern_sketch_rel_scale = t

            elif array_pattern_type == "rarray":
                xmin, ymin, _, xmax, ymax, _ = shape_to_bbox(
                    array_pattern_sketch.to_shape()
                )
                bbox_extents = [xmax - xmin, ymax - ymin]

                tmp = self.rotation_probability
                self.rotation_probability = 0
                sketch = self.generate_one_sketch(
                    1,
                    1 + self.max_n_commands,
                    only_cuts=True,
                    zero_center=zero_center,  # 1
                )  # self.min_n_commands, self.max_n_commands
                point, min_extent, max_extent = get_point_on_sketch_and_radius_range(
                    sketch
                )
                self.rotation_probability = tmp

                rect = (
                    sketch.wire_to_rectangle(sketch.wires[0])
                    if sketch.wires[0]["type"] == "polygon"
                    else None
                )
                if len(sketch.wires) == 1 and rect is not None:
                    x, y, w, h = rect
                    mi, ma = min(
                        w - 2 * abs(point[0] - x), h - 2 * abs(point[1] - y)
                    ), max(w - 2 * (point[0] - x), h - 2 * abs(point[1] - y))
                    n_mi = np.random.randint(1, 5)
                    n_ma = np.random.randint(2, 10)
                    dmi = mi / (n_mi)
                    dma = ma / (n_ma)

                    if mi == h - 2 * abs(point[1] - y):
                        array_pattern_dx = np.random.uniform(0.8 * dma, dma)
                        array_pattern_dy = np.random.uniform(0.8 * dmi, dmi)
                        array_pattern_nx = n_ma
                        array_pattern_ny = n_mi
                    else:
                        array_pattern_dx = np.random.uniform(0.8 * dmi, dmi)
                        array_pattern_dy = np.random.uniform(0.8 * dma, dma)
                        array_pattern_nx = n_mi
                        array_pattern_ny = n_ma

                    alpha_x = 0.8 * array_pattern_dx / bbox_extents[0]
                    alpha_y = 0.8 * array_pattern_dy / bbox_extents[1]
                else:
                    array_pattern_radius = max_extent
                    mi = ma = array_pattern_radius * 1.5
                    n_mi = np.random.randint(1, 5)
                    n_ma = np.random.randint(2, 10)
                    dmi = mi / (n_mi)
                    dma = ma / (n_ma)
                    array_pattern_dx = np.random.uniform(0.8 * dma, dma)
                    array_pattern_dy = np.random.uniform(0.8 * dmi, dmi)
                    array_pattern_nx = n_ma
                    array_pattern_ny = n_mi
                    alpha_x = 0.8 * array_pattern_dx / bbox_extents[0]
                    alpha_y = 0.8 * array_pattern_dy / bbox_extents[1]

                array_pattern_sketch_rel_scale = min(alpha_x, alpha_y)

            array_pattern_sketch.transform([0, 0], array_pattern_sketch_rel_scale)  # type: ignore

            # TODO maybe remove this whole part during sketch disentanglement
            # if do_hole:
            #     if do_outer_hole:
            #         assert (
            #             array_pattern_radius is not None and min_extent is not None
            #         ), "array_pattern_radius and min_extent are required"
            #         outer_hole_radius = np.random.uniform(
            #             array_pattern_radius, 0.95 * min_extent
            #         )
            #     inner_holes_radii = get_inner_holes_radii(
            #         array_pattern_radius, n_inner_holes
            #     )

        else:
            sketch = self.generate_one_sketch(
                self.min_n_commands, self.max_n_commands, zero_center=zero_center
            )

        assert sketch is not None
        sketch.array_pattern_sketch = array_pattern_sketch
        sketch.array_pattern_n = array_pattern_n
        sketch.array_pattern_start_angle = array_pattern_start_angle
        sketch.array_pattern_span = array_pattern_span
        sketch.array_pattern_radius = array_pattern_radius
        sketch.array_pattern_dx = array_pattern_dx
        sketch.array_pattern_dy = array_pattern_dy
        sketch.array_pattern_nx = array_pattern_nx
        sketch.array_pattern_ny = array_pattern_ny
        sketch.array_pattern_type = array_pattern_type
        sketch.array_pattern_mode = array_pattern_mode
        sketch.point = point[:2]

        return sketch


factories.register("sketch", SketchFactory)
