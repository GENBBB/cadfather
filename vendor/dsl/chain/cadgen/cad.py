import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from operator import itemgetter

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from omegaconf import OmegaConf

from .attach_array import *
from .control_flow import BlockFactory, EitherFactory, StopFactory
from .cut_thru_all import CutThruAllFactory
from .edge_operations import FilletChamferFactory
from .extrude import ExtrudeFactory
from .face_operations import FaceFilletChamferFactory
from .hole import HoleFactory
from .loft import LoftFactory
from .plane_logic import (
    generate_cut_thru_all_plane,
    generate_extrude_plane,
    generate_face_fillet_chamfer_plane,
    generate_fillet_chamfer_plane,
    generate_hole_plane,
    generate_loft_plane,
    generate_revolve_plane,
    generate_selected_plane,
    generate_sweep_plane,
)
from .registry import factories as factories_registry
from .revolve import RevolveFactory
from .selected_face_plane import SelectedFacePlaneFactory
from .sselectors import *
from .sweep import SweepFactory
from .utils import shape_to_bbox, shape_to_volume

logger = logging.getLogger(__name__)


class CAD:
    """
    planes = [{'axis': 0, 'origin': -10}, ...]
    sketches = [{'sketch': Sketch, 'extent': 1, 'plane': 0}, ...]
    """

    def __init__(
        self,
        planes,
        cs,
        world_size: float,
        max_string_length=None,
        use_literals: bool = True,
    ):
        self.planes = planes
        self.cs = cs
        self.world_size = world_size
        self.max_string_length = max_string_length
        self.use_literals = use_literals
        logger.info(
            "Initialized CAD with %d planes, %d operations, world_size=%s",
            len(self.planes),
            len(self.cs),
            self.world_size,
        )

    def to_string(self):
        logger.info(
            "Building CAD string representation for %d operations across %d planes",
            len(self.cs),
            len(self.planes),
        )

        s = "import cadquery as cq\n"

        for i, plane in enumerate(self.planes):
            axis = plane["axis"]
            origin = ["0", "0", "0"]
            origin[axis] = str(plane["origin"])
            origin = ",".join(origin)
            axis = ["YZ", "ZX", "XY"][axis]
            s += f"w{i}=cq.Workplane('{axis}',origin=({origin}))\n"

        s += "r="

        for i, op in enumerate(self.cs):

            if op["type"] == "Extrude":
                extrude = op["extrude"]
                plane = op["plane"]

                logger.info(
                    "Processing extrude operation %d/%d on plane %d",
                    i + 1,
                    len(self.cs),
                    plane,
                )

                t = extrude.to_string(
                    f"w{plane}" if isinstance(plane, int) else plane,
                )

                logger.info(
                    "Extrude parameters: extent=%s",
                    extrude.extent,
                )

                if i > 0:
                    if ".faces" not in t:
                        t = f"r=r.union({t[:-1]})\n"
                s += t

            elif op["type"] == "Revolve":
                revolve = op["revolve"]
                plane = op["plane"]

                t = revolve.to_string(f"w{plane}")

                logger.info(
                    "Revolve parameters: dist_to_axis=%s, angle=%s",
                    revolve.dist_to_axis,
                    revolve.angle_degrees,
                )

                if i > 0:
                    if ".faces" not in t:
                        t = f"r=r.union({t[:-1]})\n"
                s += t

            elif op["type"] == "Sweep":
                logger.info("Processing sweep operation %d/%d", i + 1, len(self.cs))
                sweep_plane = op.get("plane")
                if sweep_plane is not None:
                    plane_expr = f"w{sweep_plane}"
                    axis_idx = self.planes[sweep_plane]["axis"]
                else:
                    plane_expr = "cq.Workplane('XY')"
                    axis_idx = 2
                t = op["sweep"].to_string(
                    plane=plane_expr, axis=axis_idx, use_literals=self.use_literals
                )
                if i > 0:
                    t = f".union({t[:-1]})"
                s += t

            elif op["type"] == "Loft":
                logger.info("Processing loft operation %d/%d", i + 1, len(self.cs))
                loft_plane = op.get("plane")
                if loft_plane is not None:
                    plane_expr = f"w{loft_plane}"
                    axis_idx = self.planes[loft_plane]["axis"]
                else:
                    plane_expr = "cq.Workplane('XY')"
                    axis_idx = 2
                t = op["loft"].to_string(
                    plane=plane_expr, axis=axis_idx, use_literals=self.use_literals
                )
                if i > 0:
                    t = f".union({t[:-1]})"
                s += t

            elif op["type"] == "FaceFillet":
                face_fillet = op["face_fillet"]
                selection_str = face_fillet.ineq_sign + face_fillet.axis
                t = f".faces('{selection_str}').fillet({face_fillet.radius})"
                s += t

            elif op["type"] == "FaceChamfer":
                face_chamfer = op["face_chamfer"]
                selection_str = face_chamfer.ineq_sign + face_chamfer.axis
                t = f".faces('{selection_str}')"
                if face_chamfer.width2 is None:
                    t += f".chamfer({face_chamfer.width1})"
                else:
                    t += f".chamfer({face_chamfer.width1}, {face_chamfer.width2})"
                s += t

            elif op["type"] == "Fillet":
                fillet = op["fillet"]
                t = fillet.to_string()
                s += t

            elif op["type"] == "Chamfer":
                chamfer = op["chamfer"]
                t = chamfer.to_string()
                s += t

            elif op["type"] == "Hole":
                hole = op["hole"]
                plane = op["plane"]

                t = hole.to_string(plane)

                logger.info(
                    "Hole parameters:",
                )

                s += t

            elif op["type"] == "cutThruAll":
                cut_thru_all = op["cut_thru_all"]

                t = cut_thru_all.to_string()

                logger.info(
                    "cutThruAll parameters:",
                )

                s += t

            elif op["type"] == "SelectedFacePlane":
                selected_face_plane = op["selected_face_plane"]
                t = selected_face_plane.to_string()
                s += t

            # try:
            #     exec(s)
            # except Exception as e:
            #     print(f"Failure on op {op['type']} number {i + 1}")
            #     print(s)
            #     raise e

        return s

    def to_shape(self):
        # -> TopoDS_Compound
        logger.info("Converting CAD to shape for %d operations", len(self.cs))
        shape = None
        fused_ops = 0
        for op in self.cs:
            if op["type"] == "Sketch":
                sketch = op
                compound = sketch["sketch"].to_shape()
                plane = self.planes[sketch["plane"]]
                axis = plane["axis"]
                origin = plane["origin"]

                if axis == 0:  # 'YZ'
                    dir_n = gp_Dir(1, 0, 0)
                    dir_x = gp_Dir(0, 1, 0)
                elif axis == 1:  # 'ZX'
                    dir_n = gp_Dir(0, 1, 0)
                    dir_x = gp_Dir(0, 0, 1)
                else:  # 'XY'
                    dir_n = gp_Dir(0, 0, 1)
                    dir_x = gp_Dir(1, 0, 0)

                pnt = [0, 0, 0]
                pnt[axis] = origin
                pnt = gp_Pnt(*pnt)

                # rotate sketch from XY to its plane and translate to origin
                trsf = gp_Trsf()
                trsf.SetTransformation(
                    gp_Ax3(pnt, dir_n, dir_x),
                    gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)),
                )
                compound = BRepBuilderAPI_Transform(compound, trsf).Shape()

                extent = gp_Vec(dir_n).Multiplied(sketch["extent"])
                compound = BRepPrimAPI_MakePrism(compound, extent).Shape()
                if shape is not None:
                    shape = BRepAlgoAPI_Fuse(shape, compound).Shape()
                else:
                    shape = compound
                fused_ops += 1
            else:
                continue
        logger.info("Finished shape conversion; fused %d sketch operations", fused_ops)
        return shape

    def normalize(self, s=None):
        if s is None:
            shape = self.to_shape()
            try:
                x_min, y_min, z_min, x_max, y_max, z_max = shape_to_bbox(shape)
            except Exception as e:
                assert str(e).startswith("Standard_ConstructionErrorBnd_Box")
                assert False, "Invalid bbox in normalize"
        else:
            exec(s, globals())
            w = globals()["r"].val()
            bbox = w.BoundingBox()
            x_min, y_min, z_min, x_max, y_max, z_max = (
                bbox.xmin,
                bbox.ymin,
                bbox.zmin,
                bbox.xmax,
                bbox.ymax,
                bbox.zmax,
            )

        bbox_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
        assert not np.isclose(bbox_size, 0)
        shift = [-(x_min + x_max) / 2, -(y_min + y_max) / 2, -(z_min + z_max) / 2]
        scale = self.world_size / bbox_size  # type: ignore
        logger.info(
            "Normalizing CAD: bbox_size=%s, shift=%s, scale=%s",
            bbox_size,
            shift,
            scale,
        )

        self.transform(shift, scale)

    def transform(self, shift: list[float], scale: float) -> None:
        """Apply global shift/scale to all planes and operations."""
        for op in self.cs:
            if op["type"] == "Extrude" or op["type"] == "Revolve":
                if isinstance(op["plane"], int):
                    axis = self.planes[op["plane"]]["axis"]
                    if axis == 0:  # 'YZ'
                        sketch_shift = [shift[1], shift[2]]
                    elif axis == 1:  # 'ZX'
                        sketch_shift = [shift[2], shift[0]]
                    else:  # 'XY'
                        sketch_shift = [shift[0], shift[1]]
                else:
                    sketch_shift = [0, 0]

                op[op["type"].lower()].transform(shift, scale)
                op[op["type"].lower()].sketch.transform(sketch_shift, scale)

            elif op["type"] == "Sweep":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if plane_idx is not None else 2
                )
                op["sweep"].transform(shift, scale, plane_axis)

            elif op["type"] == "Loft":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if plane_idx is not None else 2
                )
                op["loft"].transform(shift, scale, plane_axis)

            elif op["type"] == "FaceFillet":
                op["face_fillet"].transform(shift, scale)

            elif op["type"] == "FaceChamfer":
                op["face_chamfer"].transform(shift, scale)

            elif op["type"] == "Fillet":
                op["fillet"].transform(shift, scale)

            elif op["type"] == "Chamfer":
                op["chamfer"].transform(shift, scale)

            elif op["type"] == "Hole":
                op["hole"].transform(shift, scale)

            elif op["type"] == "cutThruAll":
                op["cut_thru_all"].transform(shift, scale)

            elif op["type"] == "SelectedFacePlane":
                op["selected_face_plane"].transform(shift, scale)

        for plane in self.planes:
            plane["origin"] = (plane["origin"] + shift[plane["axis"]]) * scale

    def round(self):
        logger.info("Rounding CAD geometry")
        for plane in self.planes:
            plane["origin"] = round(plane["origin"])

        for op in self.cs:
            if op["type"] == "Extrude" or op["type"] == "Revolve":
                op[op["type"].lower()].round()

            elif op["type"] == "Sweep":
                op["sweep"].round()
            elif op["type"] == "Loft":
                op["loft"].round()
            elif op["type"] == "FaceFillet":
                op["face_fillet"].round()
            elif op["type"] == "FaceChamfer":
                op["face_chamfer"].round()
            elif op["type"] == "Fillet":
                op["fillet"].round()
            elif op["type"] == "Chamfer":
                op["chamfer"].round()
            elif op["type"] == "Hole":
                op["hole"].round()
            elif op["type"] == "cutThruAll":
                op["cut_thru_all"].round()
            elif op["type"] == "SelectedFacePlane":
                op["selected_face_plane"].round()
            else:
                continue

    def fix(self):
        # skip invalid sketches, sketches adding nothing, unused planes, long texts
        logger.info("Fixing CAD operations: starting with %d ops", len(self.cs))
        compound, cs = None, list()
        for op in self.cs:
            if op["type"] == "Sketch":
                sketch = op
                # skip invalid sketches
                try:
                    sketch["sketch"].fix()
                except AssertionError:
                    raise
                    # continue

                new_cs = cs + [sketch]
                cad = CAD(
                    self.planes,
                    new_cs,
                    self.world_size,
                    self.max_string_length,
                    self.use_literals,
                )
                # new_compound = cad.to_shape()
                new_string = cad.to_string()
                exec(new_string, globals())
                new_compound = globals()["r"].val()
                # skip long strings:
                # if len(new_string) > self.max_string_length:  # type: ignore
                #     continue

                if new_compound is None:
                    raise
                    # continue
                new_volume = shape_to_volume(new_compound.wrapped)
                if np.isclose(new_volume, 0) or new_volume < 0:
                    raise
                    # continue

                # skip sketches adding nothing
                if compound is not None:
                    compound_volume = shape_to_volume(compound.wrapped)
                    if np.isclose(compound_volume, new_volume):
                        raise
                        # continue

                cs = new_cs
                compound = new_compound
            else:
                cs.append(op)

        assert len(cs)
        self.cs = cs

        # skip unused planes
        planes = sorted(
            set(s["plane"] for s in self.cs if isinstance(s.get("plane", None), int))
        )
        if planes:
            self.planes = (
                itemgetter(*planes)(self.planes)
                if len(planes) > 1
                else [self.planes[planes[0]]]
            )
            if len(planes) != len(self.planes):
                mapping = dict(zip(planes, range(len(planes))))
                for op in self.cs:
                    if op.get("plane", None) is not None:
                        op["plane"] = mapping[op["plane"]]
        logger.info("Fix complete: %d ops remain after cleanup", len(self.cs))

    def finalize(self):
        # iterate: normalize, round, fix
        n_tries = 5
        # r = r"\b\d+\.?\d*\b"
        r = r"(?<![\w.])-?(?:\d+\.\d*|\d+|\.\d+)(?![\w.])"
        logger.info("Finalizing CAD across up to %d iterations", n_tries)
        for _ in range(n_tries):
            s = self.to_string()
            self.normalize(s)
            self.round()
            self.fix()

            if re.sub(r, "", s) == re.sub(r, "", self.to_string()):
                self.reorder()
                logger.info("Finalization successful")
                return
        assert False, "finalize iterations exceeded"

    def reorder(self):
        # reorder wires and edges inside each sketch
        logger.info("Reordering wires and edges in sketches")
        for op in self.cs:
            if op.get("sketch", None) is not None:
                op["sketch"].reorder()


class CADFactory:
    def __init__(
        self,
        factories,
        world_size,
        max_string_length,
        use_literals: bool = True,
    ):
        if isinstance(list(factories[0].values())[0], Mapping):
            factories = CADFactory._factories_from_mappings(factories)
        self.factories = factories
        self.world_size = world_size
        self.max_string_length = max_string_length
        self.use_literals = use_literals

    @staticmethod
    def _factories_from_mappings(factories):
        factory_objects = []
        for value in factories.values():
            factory_name = list(value.keys())[0]
            factory_kwargs = OmegaConf.to_container(value[factory_name])
            factory = factory_kwargs.pop("factory")  # type: ignore
            factory_objects.append(
                dict(
                    factory=factories_registry.get(factory_name)(**factory),  # type: ignore
                    **factory_kwargs,  # type: ignore
                )
            )
        return factory_objects

    def generate(self):
        r_min, r_max = 0.01, 1.0
        cs, planes, face_planes = list(), list(), list()
        previous_extrude = -1

        _factories = deepcopy(self.factories)

        for factory_number, factory in enumerate(_factories):
            if isinstance(factory["factory"], EitherFactory):
                chosen_factories = factory["factory"].generate()
                _factories[factory_number] = chosen_factories

        result = []
        for item in _factories:
            result.extend(item if isinstance(item, list) else [item])

        _factories = result

        logger.info(
            "Starting CADFactory generation with %d factory configurations",
            len(_factories),
        )

        n_retries = 10

        for factory_number, factory in enumerate(_factories):
            if isinstance(factory["factory"], BlockFactory):
                if np.random.rand() > factory["factory"].factories[0]["probability"]:
                    logger.info(
                        "Skipping factory %s due to probability filter",
                        type(factory["factory"].factories[0]["factory"]).__name__,
                    )
                    continue
            else:
                if np.random.rand() > factory["probability"]:
                    logger.info(
                        "Skipping factory %s due to probability filter",
                        type(factory["factory"]).__name__,
                    )
                    continue

            if isinstance(factory["factory"], StopFactory):
                break

            if isinstance(factory["factory"], BlockFactory):
                for block_factory_idx, block_factory in enumerate(
                    factory["factory"].factories
                ):
                    if (
                        np.random.rand() > block_factory["probability"]
                        and block_factory_idx
                    ):
                        logger.info(
                            "Skipping factory %s due to probability filter",
                            type(factory["factory"]).__name__,
                        )
                        continue

                    if isinstance(block_factory["factory"], ExtrudeFactory):
                        for retry in range(1):
                            try:
                                extrude = block_factory["factory"].generate()
                                generate_extrude_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    extrude,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                previous_extrude = extrude
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], RevolveFactory):
                        for retry in range(n_retries):
                            try:
                                revolve = block_factory["factory"].generate()
                                generate_revolve_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    revolve,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], SweepFactory):
                        for retry in range(n_retries):
                            try:
                                sweep = block_factory["factory"].generate()
                                generate_sweep_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    sweep,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], LoftFactory):
                        for retry in range(n_retries):
                            try:
                                loft = block_factory["factory"].generate()
                                generate_loft_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    loft,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], HoleFactory):
                        for retry in range(n_retries):
                            try:
                                hole = block_factory["factory"].generate(
                                    plane=None, s=None
                                )
                                generate_hole_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    hole,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], CutThruAllFactory):
                        for retry in range(n_retries):
                            try:
                                cut_thru_all = block_factory["factory"].generate(
                                    plane=None, s=None
                                )
                                generate_cut_thru_all_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    cut_thru_all,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], FaceFilletChamferFactory):
                        for retry in range(n_retries):
                            try:
                                face_fillet_chamfer = block_factory[
                                    "factory"
                                ].generate()
                                generate_face_fillet_chamfer_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    face_fillet_chamfer,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], FilletChamferFactory):
                        for retry in range(n_retries):
                            try:
                                fillet_chamfer = block_factory["factory"].generate()
                                generate_fillet_chamfer_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    fillet_chamfer,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], SelectedFacePlaneFactory):
                        for retry in range(n_retries):
                            try:
                                selected_face_plane = block_factory[
                                    "factory"
                                ].generate()
                                generate_selected_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    selected_face_plane,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                break
                            except Exception:
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{n_retries}"
                                # )
                                if retry == n_retries - 1:
                                    raise
                                else:
                                    continue

            elif isinstance(factory["factory"], ExtrudeFactory):
                for retry in range(1):
                    try:
                        extrude = factory["factory"].generate()

                        generate_extrude_plane(
                            factory,
                            planes,
                            face_planes,
                            extrude,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                        previous_extrude = extrude
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], RevolveFactory):
                for retry in range(n_retries):
                    try:
                        revolve = factory["factory"].generate()

                        generate_revolve_plane(
                            factory,
                            planes,
                            face_planes,
                            revolve,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], SweepFactory):
                for retry in range(n_retries):
                    try:
                        sweep = factory["factory"].generate()

                        generate_sweep_plane(
                            factory,
                            planes,
                            face_planes,
                            sweep,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], LoftFactory):
                for retry in range(n_retries):
                    try:
                        loft = factory["factory"].generate()

                        generate_loft_plane(
                            factory,
                            planes,
                            face_planes,
                            loft,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], HoleFactory):
                for retry in range(n_retries):
                    try:
                        hole = factory["factory"].generate(plane=None, s=None)

                        generate_hole_plane(
                            factory,
                            planes,
                            face_planes,
                            hole,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], CutThruAllFactory):
                for retry in range(n_retries):
                    try:
                        cut_thru_all = factory["factory"].generate(plane=None, s=None)

                        generate_cut_thru_all_plane(
                            factory,
                            planes,
                            face_planes,
                            cut_thru_all,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], FaceFilletChamferFactory):
                for retry in range(n_retries):
                    try:
                        face_fillet_chamfer = factory["factory"].generate()
                        generate_face_fillet_chamfer_plane(
                            factory,
                            planes,
                            face_planes,
                            face_fillet_chamfer,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], FilletChamferFactory):
                for retry in range(n_retries):
                    try:
                        fillet_chamfer = factory["factory"].generate()
                        generate_fillet_chamfer_plane(
                            factory,
                            planes,
                            face_planes,
                            fillet_chamfer,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

            elif isinstance(factory["factory"], SelectedFacePlaneFactory):
                for retry in range(n_retries):
                    try:
                        selected_face_plane = factory["factory"].generate()
                        generate_selected_plane(
                            factory,
                            planes,
                            face_planes,
                            selected_face_plane,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        break
                    except Exception:
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{n_retries}"
                        # )
                        if retry == n_retries - 1:
                            raise
                        else:
                            continue

        # sort per plane sketches by extent
        sorted_cs = list()
        for i in range(len(planes)):
            plane_sketches = [
                s for s in cs if s.get("plane", None) == i and s["type"] == "Sketch"
            ]
            if len(plane_sketches) == 0:
                continue
            if len(plane_sketches) == 1:
                sorted_cs.extend(plane_sketches)
            else:
                plane_extents = [s["extent"] for s in plane_sketches]
                ids = np.argsort(plane_extents)
                sorted_cs.extend(itemgetter(*ids)(plane_sketches))
        for i, op in enumerate(cs):
            if op["type"] != "Sketch" or isinstance(op.get("plane", None), str):
                sorted_cs.insert(i, op)
        cs = sorted_cs

        logger.info(
            "CADFactory assembled CAD with %d planes and %d operations",
            len(planes),
            len(cs),
        )
        return CAD(
            planes,
            cs,
            self.world_size,
            self.max_string_length,
            self.use_literals,
        )
