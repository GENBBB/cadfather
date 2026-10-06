import logging
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from operator import itemgetter

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from omegaconf import OmegaConf
# from cadquery_addons import *

from .circular_pattern import CircularPatternFactory, CircularPattern
from .control_flow import (
    BlockFactory,
    EitherFactory,
    StopFactory,
    expand_factory_entries,
)
from .cut_thru_all import CutThruAllFactory, CutThruAll
from .edge_operations import FilletChamferFactory, Fillet, Chamfer
from .extrude import ExtrudeFactory, Extrude, TwoExtrudes
from .face_operations import FaceFilletChamferFactory, FaceFillet, FaceChamfer
from .gear import GearFactory, Gear
from .hole import HoleFactory, Hole
from .loft import LoftFactory, Loft
from .orto_cut import OrtoCutFactory, OrtoCut
from .plane_logic import (
    generate_circular_pattern_plane,
    generate_cut_thru_all_plane,
    generate_extrude_plane,
    generate_face_fillet_chamfer_plane,
    generate_fillet_chamfer_plane,
    generate_gear_plane,
    generate_hole_plane,
    generate_loft_plane,
    generate_revolve_plane,
    generate_rib_plane,
    generate_selected_plane,
    generate_sweep_init_plane,
    generate_thread_plane,
)
from .registry import factories as factories_registry
from .revolve import RevolveFactory, Revolve
from .rib import RibFactory, Rib
from .thread import ThreadFactory, Thread
from .selected_face_plane import SelectedFacePlaneFactory, SelectedFacePlane
from .shell import ShellFactory, Shell
from .surface_sampling import (
    AxisWorkplane,
    shape_from_cad_object,
)
from .sweep import SweepFactory, Sweep
from .sweep_adv import SweepAdvFactory, SweepAdv
from .sweep_init import SweepInitFactory, SweepInit
from .utils import (
    get_face_center,
    shape_to_area,
    shape_to_bbox,
    shape_to_volume,
)

logger = logging.getLogger(__name__)


class CADFinalFixTooDestructive(RuntimeError):
    """Raised when final cascade cleanup would discard too much history."""


class EmptyCADGeneratedError(RuntimeError):
    """Raised when CADFactory finishes without constructing any operations."""


@dataclass
class _ShapeCache:
    loaded: bool = False
    shape: object | None = None
    faces: list | None = None
    volume: float | None = None


def _plane_to_string(plane):
    if isinstance(plane, AxisWorkplane):
        return plane.to_string()
    return plane


def _result_shape_from_cad_object(cad_object):
    if hasattr(cad_object, "val"):
        value = cad_object.val()
        if hasattr(value, "wrapped"):
            return value
    return shape_from_cad_object(cad_object)


def _indexed_plane_expr(plane: dict) -> str:
    axis_index = plane["axis"]
    workplane_axis = ["YZ", "ZX", "XY"][axis_index]
    offset = plane["origin"]
    return f"cq.Workplane('{workplane_axis}').workplane(offset={offset})"


def _indexed_plane_workplane_and_origin(
    plane: dict,
) -> tuple[str, tuple[float, float, float]]:
    axis_index = plane["axis"]
    workplane_axis = ["YZ", "ZX", "XY"][axis_index]
    origin = [0.0, 0.0, 0.0]
    origin[axis_index] = float(plane["origin"])
    return workplane_axis, tuple(origin)


def _indexed_plane_ref(
    planes: list[dict],
    plane_index: int,
    inline_plane_indices: set[int],
) -> str:
    if plane_index in inline_plane_indices:
        return _indexed_plane_expr(planes[plane_index])
    return f"w{plane_index}"


class CAD:
    """
    planes = [{'axis': 0, 'origin': -10}, ...]
    sketches = [{'sketch': Sketch, 'extent': 1, 'plane': 0}, ...]
    """

    def __init__(
        self,
        planes,
        face_planes,
        cs,
        sampled_factories: list[dict] | None = None,
        world_size: float = 200,
        max_string_length=None,
        use_literals: bool = True,
    ):
        self.planes = planes
        self.face_planes = face_planes
        self.cs = cs
        self.sampled_factories = sampled_factories
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
        s += "from cadgen.selectors import PointOnEdgeSelector\n"
        s += "from cadgen.extrude import extrude\n"
        s += "from cadgen.shell import shell\n"
        s += "from cadgen.hole import hole\n"
        s += "from cadgen.revolve import revolve\n"
        s += "from cadgen.orto_cut import orto_cut\n"
        s += "from cadgen.sweep import sweep\n"
        s += "from cadgen.sweep_adv import sweep_adv\n"
        s += "from cadgen.spring import spring\n"
        s += "from cadgen.loft import loft\n"
        s += "from cadgen.gear import gear\n"
        s += "from cadgen.rib import rib\n"
        s += "from cadgen.thread import thread\n"
        s += "r = None\n"

        inline_plane_indices: set[int] = set()
        if (
            self.cs
            and self.cs[0]["type"]
            in {"Extrude", "Revolve", "Shell", "TwoExtrudes", "SweepInit", "Loft",
                "Rib", "Thread"}
            and isinstance(self.cs[0].get("plane"), int)
        ):
            inline_plane_indices.add(self.cs[0]["plane"])

        for i, plane in enumerate(self.planes):
            if i in inline_plane_indices:
                continue
            axis = plane["axis"]
            origin = ["0", "0", "0"]
            origin[axis] = str(plane["origin"])
            origin = ",".join(origin)
            axis = ["YZ", "ZX", "XY"][axis]
            s += f"w{i}=cq.Workplane('{axis}',origin=({origin}))\n"

        s += "r = "

        for i, op in enumerate(self.cs):
            if op["type"] == "TwoExtrudes":
                two_extrudes = op["op"]
                plane = op["plane"]

                logger.info(
                    "Processing two_extrudes operation %d/%d on plane %s",
                    i + 1,
                    len(self.cs),
                    plane,
                )

                if isinstance(plane, int):
                    workplane_axis, origin = _indexed_plane_workplane_and_origin(
                        self.planes[plane]
                    )
                    t = two_extrudes.to_global_workplane_string(
                        workplane_axis,
                        origin,
                    )
                elif isinstance(plane, AxisWorkplane):
                    t = two_extrudes.to_string(plane)
                else:
                    raise ValueError(
                        "TwoExtrudes operations require an indexed or sampled workplane"
                    )

                logger.info(
                    "TwoExtrudes parameters: first_extent=%s second_extent=%s",
                    two_extrudes.first.extent,
                    two_extrudes.second.extent,
                )

                if i == 0:
                    s = s[: -len("r = ")] + "r="
                else:
                    t = f"r={t}"
                s += t

            elif op["type"] == "Extrude":
                extrude = op["op"]
                plane = op["plane"]
                uses_factory_call = False

                logger.info(
                    "Processing extrude operation %d/%d on plane %s",
                    i + 1,
                    len(self.cs),
                    plane,
                )

                if isinstance(plane, int):
                    if plane in inline_plane_indices:
                        workplane_axis, origin = _indexed_plane_workplane_and_origin(
                            self.planes[plane]
                        )
                        t = extrude.to_global_workplane_string(
                            workplane_axis,
                            origin,
                        )
                        uses_factory_call = True
                    else:
                        plane_expr = _indexed_plane_ref(
                            self.planes,
                            plane,
                            inline_plane_indices,
                        )
                        t = extrude.to_string(plane_expr)
                elif isinstance(plane, AxisWorkplane):
                    t = extrude.to_string(plane)
                    uses_factory_call = True
                else:
                    plane_expr = plane
                    t = extrude.to_string(plane_expr)

                logger.info(
                    "Extrude parameters: extent=%s",
                    extrude.extent,
                )

                if i == 0 and uses_factory_call:
                    s = s[: -len("r = ")] + "r="
                elif i > 0:
                    if uses_factory_call:
                        t = f"r={t}"
                    elif ".faces" not in t:
                        t = f"r=r.union({t[:-1]})\n"
                s += t

            elif op["type"] == "Shell":
                shell_op = op["op"]
                plane = op["plane"]

                logger.info(
                    "Processing shell operation %d/%d on plane %s",
                    i + 1,
                    len(self.cs),
                    plane,
                )

                if isinstance(plane, int):
                    workplane_axis, origin = _indexed_plane_workplane_and_origin(
                        self.planes[plane]
                    )
                    t = shell_op.to_global_workplane_string(
                        workplane_axis,
                        origin,
                    )
                elif isinstance(plane, AxisWorkplane):
                    t = shell_op.to_string(plane)
                else:
                    raise ValueError(
                        "Shell operations require an indexed or sampled workplane"
                    )

                logger.info(
                    "Shell parameters: extent=%s wall_thickness=%s bottom_thickness=%s",
                    shell_op.extent,
                    shell_op.wall_thickness,
                    shell_op.bottom_thickness,
                )

                if i == 0:
                    s = s[: -len("r = ")] + "r="
                else:
                    t = f"r={t}"
                s += t

            elif op["type"] == "Revolve":
                revolve = op["op"]
                plane = op["plane"]
                uses_factory_call = False

                if isinstance(plane, int):
                    if plane in inline_plane_indices:
                        workplane_axis, origin = _indexed_plane_workplane_and_origin(
                            self.planes[plane]
                        )
                        t = revolve.to_global_workplane_string(
                            workplane_axis,
                            origin,
                        )
                        uses_factory_call = True
                    else:
                        plane_expr = _indexed_plane_ref(
                            self.planes,
                            plane,
                            inline_plane_indices,
                        )
                        t = revolve.to_string(s, plane_expr)
                elif isinstance(plane, AxisWorkplane):
                    t = revolve.to_string(s, plane)
                    uses_factory_call = True
                else:
                    plane_expr = _plane_to_string(plane)
                    t = revolve.to_string(s, plane_expr)

                logger.info(
                    "Revolve parameters: axis=%s, angle=%s",
                    revolve.axis,
                    revolve.angle_degrees,
                )

                if i > 0:
                    if uses_factory_call:
                        t = f"r = {t}"
                    elif ".faces" not in t:
                        t = f"r=r.union({t[:-1]}.val())\n"
                s += t

            elif op["type"] in ("Sweep", "SweepAdv"):
                logger.info("Processing attached sweep operation %d/%d", i + 1, len(self.cs))
                if i == 0:
                    s = s[: -len("r = ")]
                    t = op["op"].to_string()
                else:
                    t = op["op"].to_string()
                s += t

            elif op["type"] == "SweepInit":
                logger.info("Processing sweep operation %d/%d", i + 1, len(self.cs))
                sweep_plane = op.get("plane")
                if self.use_literals:
                    # One line per 3D op, self-contained (only `r`): rebuild the
                    # workplane inside `helix` from point + named axis.
                    if isinstance(sweep_plane, int):
                        workplane_axis, origin = _indexed_plane_workplane_and_origin(
                            self.planes[sweep_plane]
                        )
                    else:
                        workplane_axis, origin = "XY", (0.0, 0.0, 0.0)
                    t = op["op"].to_call_string(origin, workplane_axis)
                    if i == 0:
                        s = s[: -len("r = ")] + "r="
                    else:
                        t = f"r={t}"
                else:
                    # Parametric (walrus) mode: keep the raw multi-line chain.
                    if sweep_plane is not None:
                        plane_expr = _indexed_plane_ref(
                            self.planes, sweep_plane, inline_plane_indices
                        )
                        axis_idx = self.planes[sweep_plane]["axis"]
                    else:
                        plane_expr = "cq.Workplane('XY')"
                        axis_idx = 2
                    t = op["op"].to_string(
                        plane=plane_expr, axis=axis_idx, use_literals=False
                    )
                    if i > 0:
                        t = f".union({t[:-1]})"
                s += t

            elif op["type"] == "Loft":
                logger.info("Processing loft operation %d/%d", i + 1, len(self.cs))
                loft_plane = op.get("plane")
                # One line per 3D op, self-contained (only `r`): rebuild the base
                # workplane inside `loft` from point + named axis.
                if isinstance(loft_plane, int):
                    workplane_axis, origin = _indexed_plane_workplane_and_origin(
                        self.planes[loft_plane]
                    )
                else:
                    workplane_axis, origin = "XY", (0.0, 0.0, 0.0)
                t = op["op"].to_call_string(origin, workplane_axis)
                if i == 0:
                    s = s[: -len("r = ")] + "r="
                else:
                    t = f"r={t}"
                s += t

            elif op["type"] == "Rib":
                logger.info("Processing rib operation %d/%d", i + 1, len(self.cs))
                rib_plane = op.get("plane")
                if isinstance(rib_plane, int):
                    workplane_axis, origin = _indexed_plane_workplane_and_origin(
                        self.planes[rib_plane]
                    )
                else:
                    workplane_axis, origin = "XY", (0.0, 0.0, 0.0)
                t = op["op"].to_call_string(origin, workplane_axis)
                if i == 0:
                    s = s[: -len("r = ")] + "r="
                else:
                    t = f"r={t}"
                s += t

            elif op["type"] == "Thread":
                logger.info("Processing thread operation %d/%d", i + 1, len(self.cs))
                thr_plane = op.get("plane")
                if isinstance(thr_plane, int):
                    workplane_axis, origin = _indexed_plane_workplane_and_origin(
                        self.planes[thr_plane]
                    )
                else:
                    workplane_axis, origin = "XY", (0.0, 0.0, 0.0)
                t = op["op"].to_call_string(origin, workplane_axis)
                if i == 0:
                    s = s[: -len("r = ")] + "r="
                else:
                    t = f"r={t}"
                s += t

            elif op["type"] == "Gear":
                logger.info("Processing gear operation %d/%d", i + 1, len(self.cs))
                gear = op["op"]
                plane = op.get("plane")
                plane_expr = (
                    _indexed_plane_ref(self.planes, plane, inline_plane_indices)
                    if isinstance(plane, int)
                    else str(plane)
                )
                t = gear.to_string(plane_expr)
                if i > 0:
                    t = f"r=r.union({t})\n"
                else:
                    t += "\n"
                s += t

            elif op["type"] == "FaceFillet":
                face_fillet = op["op"]
                selection_str = face_fillet.ineq_sign + face_fillet.axis
                t = f".faces('{selection_str}').fillet({face_fillet.radius})"
                s += t

            elif op["type"] == "FaceChamfer":
                face_chamfer = op["op"]
                selection_str = face_chamfer.ineq_sign + face_chamfer.axis
                t = f".faces('{selection_str}')"
                if face_chamfer.width2 is None:
                    t += f".chamfer({face_chamfer.width1})"
                else:
                    t += f".chamfer({face_chamfer.width1}, {face_chamfer.width2})"
                s += t

            elif op["type"] == "Fillet":
                fillet = op["op"]
                t = fillet.to_string()
                s += t

            elif op["type"] == "Chamfer":
                chamfer = op["op"]
                t = chamfer.to_string()
                s += t

            elif op["type"] == "Hole":
                hole = op["op"]
                plane = op["plane"]

                t = hole.to_string(plane)

                logger.info(
                    "Hole parameters:",
                )

                s += t

            elif op["type"] == "OrtoCut":
                orto_cut_op = op["op"]
                plane = op["plane"]
                t = orto_cut_op.to_string(plane, self.world_size)
                logger.info(
                    "OrtoCut parameters: extent=%s",
                    orto_cut_op.extent,
                )
                s += t

            elif op["type"] == "cutThruAll":
                cut_thru_all = op["op"]
                plane = op["plane"]

                t = cut_thru_all.to_string(plane)

                logger.info(
                    "cutThruAll parameters:",
                )

                s += t

            elif op["type"] == "SelectedFacePlane":
                selected_face_plane = op["op"]
                t = selected_face_plane.to_string()
                s += t

            elif op["type"] == "CircularPattern":
                circular_pattern = op["op"]
                t = circular_pattern.to_string()
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
            if op["type"] in {"Extrude", "Revolve", "Shell", "TwoExtrudes"}:
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
                    if isinstance(op.get("plane", None), AxisWorkplane):
                        op["plane"].transform(shift, scale)

                if op["type"] == "TwoExtrudes":
                    op["op"].transform(sketch_shift, scale)
                elif op["type"] in {"Extrude", "Shell"}:
                    op["op"].transform(shift, scale)
                    op["op"].sketch.transform(sketch_shift, scale)
                else:
                    op["op"].transform(shift, scale, sketch_shift=sketch_shift)
                    op["op"].sketch.transform(sketch_shift, scale)

            elif op["type"] in ("Sweep", "SweepAdv"):
                op["op"].transform(shift, scale)

            elif op["type"] == "SweepInit":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if plane_idx is not None else 2
                )
                op["op"].transform(shift, scale, plane_axis)

            elif op["type"] == "Loft":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if plane_idx is not None else 2
                )
                op["op"].transform(shift, scale, plane_axis)

            elif op["type"] == "Rib":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if plane_idx is not None else 2
                )
                op["op"].transform(shift, scale, plane_axis)

            elif op["type"] == "Thread":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if plane_idx is not None else 2
                )
                op["op"].transform(shift, scale, plane_axis)

            elif op["type"] == "Gear":
                plane_idx = op.get("plane")
                plane_axis = (
                    self.planes[plane_idx]["axis"] if isinstance(plane_idx, int) else None
                )
                op["op"].transform(shift, scale, plane_axis)

            elif op["type"] == "FaceFillet":
                op["op"].transform(shift, scale)

            elif op["type"] == "FaceChamfer":
                op["op"].transform(shift, scale)

            elif op["type"] == "Fillet":
                op["op"].transform(shift, scale)

            elif op["type"] == "Chamfer":
                op["op"].transform(shift, scale)

            elif op["type"] == "Hole":
                op["op"].transform(shift, scale)

            elif op["type"] == "OrtoCut":
                if isinstance(op.get("plane", None), AxisWorkplane):
                    op["plane"].transform(shift, scale)
                op["op"].transform(shift, scale)
                op["op"].sketch.transform([0, 0], scale)

            elif op["type"] == "cutThruAll":
                op["op"].transform(shift, scale)

            elif op["type"] == "SelectedFacePlane":
                op["op"].transform(shift, scale)

            elif op["type"] == "CircularPattern":
                op["op"].transform(shift, scale)

        for plane in self.planes:
            plane["origin"] = (plane["origin"] + shift[plane["axis"]]) * scale

    def round(self):
        logger.info("Rounding CAD geometry")
        for plane in self.planes:
            plane["origin"] = round(plane["origin"])

        for op in self.cs:
            if op["type"] in {"Extrude", "Revolve", "Shell", "TwoExtrudes"}:
                op["op"].round()
                if isinstance(op.get("plane", None), AxisWorkplane):
                    op["plane"].round()

            elif op["type"] in ("Sweep", "SweepAdv"):
                op["op"].round()
            elif op["type"] == "SweepInit":
                op["op"].round()
            elif op["type"] == "Loft":
                op["op"].round()
            elif op["type"] == "Rib":
                op["op"].round()
            elif op["type"] == "Thread":
                op["op"].round()
            elif op["type"] == "Gear":
                op["op"].round()
            elif op["type"] == "FaceFillet":
                op["op"].round()
            elif op["type"] == "FaceChamfer":
                op["op"].round()
            elif op["type"] == "Fillet":
                op["op"].round()
            elif op["type"] == "Chamfer":
                op["op"].round()
            elif op["type"] == "Hole":
                op["op"].round()
            elif op["type"] == "OrtoCut":
                op["op"].round()
                if isinstance(op.get("plane", None), AxisWorkplane):
                    op["plane"].round()
            elif op["type"] == "cutThruAll":
                op["op"].round()
            elif op["type"] == "SelectedFacePlane":
                op["op"].round()
            elif op["type"] == "CircularPattern":
                op["op"].round()
            else:
                continue

    def fix(self):
        # skip invalid sketches, sketches adding nothing, unused planes, long texts
        logger.info("Fixing CAD operations: starting with %d ops", len(self.cs))

        assert len(self.cs) == len(
            self.sampled_factories
        ), f"len(self.cs) != len(self.sampled_factories): {len(self.cs)} != {len(self.sampled_factories)}, self.cs: {self.cs}, self.sampled_factories: {self.sampled_factories}"

        fixed_cs = []
        fixed_sampled_factories = []
        for i, op in enumerate(self.cs):
            try:
                op["op"].fix()
            except Exception as e:
                logger.warning(
                    "CAD.fix removed %s at op index %d: operation fix failed: %s",
                    op["type"],
                    i,
                    e,
                )
                continue
            fixed_cs.append(op)
            fixed_sampled_factories.append(self.sampled_factories[i])

        self.cs = fixed_cs
        self.sampled_factories = fixed_sampled_factories

        compound, cs, sampled_factories = None, list(), list()
        # s_executed = ""
        assert len(self.cs) == len(
            self.sampled_factories
        ), f"len(self.cs) != len(self.sampled_factories): {len(self.cs)} != {len(self.sampled_factories)}, self.cs: {self.cs}, self.sampled_factories: {self.sampled_factories}"
        for i, op in enumerate(self.cs):
            if op["type"] != "SelectedFacePlane":
                new_cs = cs + [op]
                new_sampled_factories = sampled_factories + [self.sampled_factories[i]]
                cad = CAD(
                    self.planes,
                    self.face_planes,
                    new_cs,
                    new_sampled_factories,
                    self.world_size,
                    self.max_string_length,
                    self.use_literals,
                )
                new_string = cad.to_string()
                # lines_to_execute = new_string.split("\n")[len(s_executed.split("\n")) :]
                # exec("\n".join(lines_to_execute), globals())
                exec(new_string, globals())
                new_compound = globals()["r"].val()
                # skip long strings:
                # if len(new_string) > self.max_string_length:  # type: ignore
                #     continue

                if new_compound is None:
                    # raise ValueError("new compound is None", op["type"], new_string)
                    continue
                new_volume = shape_to_volume(new_compound.wrapped)
                if np.isclose(new_volume, 0) or new_volume < 0:
                    # raise ValueError("volume is 0", op["type"], new_string)
                    continue

                # skip sketches adding nothing
                if compound is not None:
                    compound_volume = shape_to_volume(compound.wrapped)
                    if np.isclose(compound_volume, new_volume, rtol=1e-8):
                        if op["type"] == "Hole":
                            msg = (
                                f"CAD.fix removed Hole at op index {i}: "
                                "operation did not change model volume"
                            )
                            print(msg, flush=True)
                            logger.warning(msg)
                        # raise ValueError(
                        #     "op added nothing",
                        #     op["type"],
                        #     new_string,
                        #     id(compound),
                        #     id(new_compound),
                        # )
                        continue

                cs = new_cs
                sampled_factories = new_sampled_factories
                compound = new_compound
            else:
                cs.append(op)
                sampled_factories.append(self.sampled_factories[i])

        assert len(cs)
        self.cs = cs
        self.sampled_factories = sampled_factories

        # skip unused planes; keep first-seen index order in cs (not sorted) so
        # plane labels match introduction order for prefixes / remapping.
        planes: list[int] = []
        _seen_plane: set[int] = set()
        for s in self.cs:
            p = s.get("plane", None)
            if isinstance(p, int) and p not in _seen_plane:
                _seen_plane.add(p)
                planes.append(p)
        if planes:
            mapping = dict(zip(planes, range(len(planes))))
            for op in self.cs:
                if isinstance(op.get("plane", None), int):
                    op["plane"] = mapping[op["plane"]]
            self.planes = (
                itemgetter(*planes)(self.planes)
                if len(planes) > 1
                else [self.planes[planes[0]]]
            )
        else:
            planes = self.planes

        # skip unused face planes (exclude SelectedFacePlane: self-referential plane string)
        face_planes: list[int] = []
        _seen_fp: set[int] = set()
        for s in self.cs:
            if (
                isinstance(s.get("plane", None), str)
                and s["type"] != "SelectedFacePlane"
                and s["plane"].startswith("face_w")
            ):
                idx = int(s["plane"].split("_")[-1].lstrip("selw"))
                if idx not in _seen_fp:
                    _seen_fp.add(idx)
                    face_planes.append(idx)
        new_cs = []
        new_sampled_factories = []

        _face_planes_need_remap = len(face_planes) != len(
            self.face_planes
        ) or face_planes != sorted(face_planes)
        if face_planes:
            if _face_planes_need_remap:
                mapping = dict(zip(face_planes, range(len(face_planes))))
                for i, op in enumerate(self.cs):
                    if (
                        isinstance(op.get("plane", None), str)
                        and op["type"] != "SelectedFacePlane"
                        and op["plane"].startswith("face_w")
                    ):
                        old_ending = op["plane"].split("_")[-1].lstrip("selw")
                        new_ending = mapping[int(old_ending)]
                        op["plane"] = op["plane"][: -len(old_ending)] + str(new_ending)
                    elif op["type"] == "SelectedFacePlane":
                        old_ending = (
                            op["op"].face_plane_name.split("_")[-1].lstrip("selw")
                        )
                        if int(old_ending) not in mapping:
                            continue
                        else:
                            new_ending = mapping[int(old_ending)]
                            op["op"].face_plane_name = op["op"].face_plane_name[
                                : -len(old_ending)
                            ] + str(new_ending)
                            op["op"].face_name = op["op"].face_name[
                                : -len(old_ending)
                            ] + str(new_ending)
                            op["op"].point_name = op["op"].point_name[
                                : -len(old_ending)
                            ] + str(new_ending)
                    new_cs.append(op)
                    new_sampled_factories.append(self.sampled_factories[i])
                self.cs = new_cs
                self.sampled_factories = new_sampled_factories
        else:
            for i, op in enumerate(self.cs):
                if op["type"] != "SelectedFacePlane":
                    new_cs.append(op)
                    new_sampled_factories.append(self.sampled_factories[i])
            self.cs = new_cs
            self.sampled_factories = new_sampled_factories

        if len(face_planes) > 0:
            self.face_planes = (
                itemgetter(*face_planes)(self.face_planes)
                if len(face_planes) > 1
                else [self.face_planes[face_planes[0]]]
            )
        else:
            self.face_planes = []

        logger.info("Fix complete: %d ops remain after cleanup", len(self.cs))

    def fix_operations_in_place(self) -> None:
        """Run per-operation fixers without deleting or reordering operations."""
        for i, op in enumerate(self.cs):
            try:
                op["op"].fix()
            except Exception as exc:
                raise ValueError(
                    f"{op['type']} at op index {i} failed non-deleting fix"
                ) from exc

    def _delete_operations_from(
        self,
        delete_from: int,
        initial_count: int,
        reason: str,
        max_delete_ratio: float,
    ) -> None:
        delete_count = len(self.cs) - delete_from
        total_delete_count = initial_count - delete_from
        delete_ratio = total_delete_count / initial_count if initial_count else 0.0
        if delete_ratio > max_delete_ratio:
            raise CADFinalFixTooDestructive(
                "Final cascade cleanup would delete "
                f"{total_delete_count}/{initial_count} operations "
                f"({delete_ratio:.1%}), starting at op index {delete_from}: {reason}"
            )

        logger.warning(
            "Final cascade cleanup deleted %d/%d operations from op index %d: %s",
            delete_count,
            initial_count,
            delete_from,
            reason,
        )
        self.cs = self.cs[:delete_from]
        if self.sampled_factories is not None:
            self.sampled_factories = self.sampled_factories[:delete_from]
        if not self.cs:
            raise CADFinalFixTooDestructive(
                "Final cascade cleanup would leave no operations"
            )

    def _delete_operation_at(
        self,
        delete_at: int,
        initial_count: int,
        reason: str,
    ) -> None:
        logger.warning(
            "Final cascade cleanup deleted 1/%d operation at op index %d: %s",
            initial_count,
            delete_at,
            reason,
        )
        del self.cs[delete_at]
        if self.sampled_factories is not None:
            del self.sampled_factories[delete_at]
        if not self.cs:
            raise CADFinalFixTooDestructive(
                "Final cascade cleanup would leave no operations"
            )

    def _prefix_shape_and_volume(self, cs: list[dict]):
        cad = CAD(
            self.planes,
            self.face_planes,
            cs,
            world_size=self.world_size,
            max_string_length=self.max_string_length,
            use_literals=self.use_literals,
        )
        namespace: dict[str, object] = {}
        exec(cad.to_string(), globals(), namespace)
        shape = _result_shape_from_cad_object(namespace["r"])
        volume = abs(shape_to_volume(shape.wrapped))
        return shape, volume

    def fix_cascade(
        self,
        *,
        max_delete_ratio: float = 0.3,
    ) -> None:
        """Run final cleanup, pruning bad ops.

        OrtoCut failures remove only that operation; other failures remove the
        first bad operation and all following operations.
        """
        initial_count = len(self.cs)
        if initial_count == 0:
            return
        if self.sampled_factories is not None:
            assert len(self.cs) == len(
                self.sampled_factories
            ), f"len(self.cs) != len(self.sampled_factories): {len(self.cs)} != {len(self.sampled_factories)}"

        i = 0
        while i < len(self.cs):
            op = self.cs[i]
            try:
                op["op"].fix()
            except Exception as exc:
                reason = f"{op['type']} operation fix failed: {exc}"
                if op["type"] == "OrtoCut":
                    self._delete_operation_at(
                        i,
                        initial_count,
                        reason,
                    )
                    continue
                self._delete_operations_from(i, initial_count, reason, max_delete_ratio)
                return
            i += 1

        previous_volume: float | None = None
        i = 0
        while i < len(self.cs):
            op = self.cs[i]
            try:
                _, volume = self._prefix_shape_and_volume(self.cs[: i + 1])
            except Exception as exc:
                reason = f"{op['type']} prefix execution failed: {exc}"
                if op["type"] == "OrtoCut":
                    self._delete_operation_at(
                        i,
                        initial_count,
                        reason,
                    )
                    continue
                self._delete_operations_from(i, initial_count, reason, max_delete_ratio)
                return

            if np.isclose(volume, 0) or volume <= 0:
                reason = f"{op['type']} produced non-positive volume"
                if op["type"] == "OrtoCut":
                    self._delete_operation_at(
                        i,
                        initial_count,
                        reason,
                    )
                    continue
                self._delete_operations_from(i, initial_count, reason, max_delete_ratio)
                return

            if op["type"] == "SelectedFacePlane":
                if previous_volume is None:
                    previous_volume = volume
                i += 1
                continue

            if previous_volume is not None and np.isclose(
                previous_volume,
                volume,
                rtol=1e-8,
            ):
                reason = f"{op['type']} did not change model volume"
                if op["type"] == "OrtoCut":
                    self._delete_operation_at(
                        i,
                        initial_count,
                        reason,
                    )
                    continue
                self._delete_operations_from(i, initial_count, reason, max_delete_ratio)
                return
            previous_volume = volume
            i += 1

    def finalize(self, do_normalize: bool = True, skip_fix: bool = False):
        # iterate: normalize, round, fix
        n_tries = 5
        r = r"(?<![\w.])-?(?:(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)(?![\w.])"
        logger.info("Finalizing CAD across up to %d iterations", n_tries)
        for _ in range(n_tries):
            s = self.to_string()
            if do_normalize:
                self.normalize(s)
            self.round()
            if skip_fix:
                self.fix_operations_in_place()
            else:
                self.fix_cascade(max_delete_ratio=0.3)

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

    def to_dict(self) -> dict:
        op_dict = []
        for op in self.cs:
            op_dict.append(op.copy())
            op_dict[-1]["op"] = op_dict[-1]["op"].to_dict()
            if isinstance(op_dict[-1].get("plane", None), AxisWorkplane):
                op_dict[-1]["plane"] = op_dict[-1]["plane"].to_dict()

        return {
            "type": "CAD",
            "planes": self.planes,
            "face_planes": self.face_planes,
            "cs": op_dict,
            "world_size": self.world_size,
            "max_string_length": self.max_string_length,
            "use_literals": self.use_literals,
        }

    @staticmethod
    def from_dict(entity: dict) -> "CAD":
        assert (
            entity["type"] == "CAD"
        ), f"Trying to build CAD from type {entity['type']}"
        cs = deepcopy(entity["cs"])
        for op in cs:
            op["op"] = eval(op["op"]["type"]).from_dict(op["op"])
            if isinstance(op.get("plane", None), dict) and op["plane"].get("type") == "AxisWorkplane":
                op["plane"] = AxisWorkplane.from_dict(op["plane"])

        return CAD(
            planes=entity["planes"],
            face_planes=entity["face_planes"],
            cs=cs,
            world_size=entity["world_size"],
            max_string_length=entity["max_string_length"],
            use_literals=entity["use_literals"],
        )


class CADFactory:
    def __init__(
        self,
        factories,
        world_size,
        max_string_length,
        use_literals: bool = True,
        n_retries: int = 10,
        n_extrude_retries: int = 1,
        log_exceptions: bool = False,
        skip_fix: bool = False,
        single_solid_check: bool = True,
    ):
        if isinstance(list(factories[0].values())[0], Mapping):
            factories = CADFactory._factories_from_mappings(factories)
        self.factories = factories
        self.world_size = world_size
        self.max_string_length = max_string_length
        self.use_literals = use_literals
        self.n_retries = n_retries
        self.n_extrude_retries = n_extrude_retries
        self.log_exceptions = log_exceptions
        self.skip_fix = skip_fix
        self.single_solid_check = single_solid_check
        print(
            f"n_retries: {self.n_retries}, n_extrude_retries: {self.n_extrude_retries}",
            flush=True,
        )

    @classmethod
    def validation_only(
        cls,
        *,
        world_size,
        max_string_length,
        use_literals: bool = True,
        single_solid_check: bool = True,
    ) -> "CADFactory":
        validator = cls.__new__(cls)
        validator.factories = []
        validator.world_size = world_size
        validator.max_string_length = max_string_length
        validator.use_literals = use_literals
        validator.n_retries = 0
        validator.n_extrude_retries = 0
        validator.log_exceptions = False
        validator.skip_fix = False
        validator.single_solid_check = single_solid_check
        return validator

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

    def _existing_model_and_sampler(
        self,
        factory,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
    ):
        try:
            s = CAD(
                planes,
                face_planes,
                cs,
                world_size=self.world_size,
                max_string_length=self.max_string_length,
                use_literals=self.use_literals,
            ).to_string()
        except Exception:
            if self.log_exceptions:
                logger.warning(
                    "Surface sampler prep failed while serializing existing CAD "
                    "(factory=%s, existing_ops=%d)",
                    type(factory).__name__,
                    len(cs),
                    exc_info=True,
                )
            raise
        namespace: dict[str, object] = {}
        try:
            exec(s, globals(), namespace)
        except Exception:
            if self.log_exceptions:
                logger.warning(
                    "Surface sampler prep failed while executing existing CAD "
                    "(factory=%s, existing_ops=%d)",
                    type(factory).__name__,
                    len(cs),
                    exc_info=True,
                )
            raise
        r = namespace["r"]
        try:
            sampler = factory.prepare_existing_sampler(r)
        except Exception:
            if self.log_exceptions:
                logger.warning(
                    "Surface sampler prep failed while building sampler "
                    "(factory=%s, existing_ops=%d)",
                    type(factory).__name__,
                    len(cs),
                    exc_info=True,
                )
            raise
        existing_shape = _result_shape_from_cad_object(r)
        return r, sampler, sampler.faces, abs(shape_to_volume(existing_shape.wrapped))

    def _log_extrude_retry_failure(
        self,
        *,
        factory_number: int,
        retry: int,
        max_retries: int,
        error: Exception,
        using_existing_sampler: bool,
        block_factory_idx: int | None = None,
    ) -> None:
        if not self.log_exceptions:
            return
        factory_label = str(factory_number + 1)
        if block_factory_idx is not None:
            factory_label += f".{block_factory_idx + 1}"
        logger.warning(
            "Extrude retry failed: factory=%s retry=%d/%d "
            "using_existing_sampler=%s error=%s: %s",
            factory_label,
            retry + 1,
            max_retries,
            using_existing_sampler,
            type(error).__name__,
            error,
        )

    def _append_existing_extrude(
        self,
        factory: dict,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cad_object,
        sampler,
        r_max: float,
    ):
        extrude, workplane = factory["factory"].generate_on_existing(
            cad_object,
            sampler,
            world_size=self.world_size,
            generation_world_half=r_max,
        )
        cs.append(
            dict(
                type=type(extrude).__name__,
                op=extrude,
                plane=workplane,
                plane_axes=None,
            )
        )
        return extrude

    def _append_existing_revolve(
        self,
        factory: dict,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cad_object,
        sampler,
        r_max: float,
    ):
        revolve, workplane = factory["factory"].generate_on_existing(
            cad_object,
            sampler,
            world_size=self.world_size,
            generation_world_half=r_max,
        )
        cs.append(
            dict(
                type="Revolve",
                op=revolve,
                plane=workplane,
                plane_axes=None,
            )
        )
        return revolve

    def _append_existing_orto_cut(
        self,
        factory: dict,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cad_object,
        sampler,
        r_max: float,
    ):
        orto_cut_op, workplane = factory["factory"].generate_on_existing(
            cad_object,
            sampler,
            world_size=self.world_size,
            generation_world_half=r_max,
        )
        cs.append(
            dict(
                type="OrtoCut",
                op=orto_cut_op,
                plane=workplane,
                plane_axes=None,
            )
        )
        return orto_cut_op

    def _append_existing_sweep(
        self,
        factory: dict,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cad_object,
        sampler,
        r_max: float,
    ):
        sweep_op, _ = factory["factory"].generate_on_existing(
            cad_object,
            sampler,
            world_size=self.world_size,
            generation_world_half=r_max,
        )
        cs.append(
            dict(
                type=type(sweep_op).__name__,
                op=sweep_op,
                plane=None,
                plane_axes=None,
            )
        )
        return sweep_op

    def _append_initial_sweep(
        self,
        factory: dict,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        r_max: float,
    ):
        sweep_op = factory["factory"].generate(
            world_size=self.world_size,
            generation_world_half=r_max,
        )
        cs.append(
            dict(
                type=type(sweep_op).__name__,
                op=sweep_op,
                plane=None,
                plane_axes=None,
            )
        )
        return sweep_op

    def _shape_for_ops(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
    ):
        if not cs:
            return None
        s = CAD(
            planes,
            face_planes,
            cs,
            world_size=self.world_size,
            max_string_length=self.max_string_length,
            use_literals=self.use_literals,
        ).to_string()
        namespace: dict[str, object] = {}
        exec(s, globals(), namespace)
        return _result_shape_from_cad_object(namespace["r"])

    def _shape_for_ops_cached(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        shape_cache: _ShapeCache | None = None,
    ):
        if shape_cache is None:
            return self._shape_for_ops(planes, face_planes, cs)
        if not shape_cache.loaded:
            shape_cache.shape = self._shape_for_ops(planes, face_planes, cs)
            shape_cache.loaded = True
        return shape_cache.shape

    def _volume_for_ops_cached(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        shape_cache: _ShapeCache | None = None,
    ) -> float:
        if shape_cache is None:
            shape = self._shape_for_ops(planes, face_planes, cs)
            return 0.0 if shape is None else abs(shape_to_volume(shape.wrapped))
        if shape_cache.volume is None:
            shape = self._shape_for_ops_cached(planes, face_planes, cs, shape_cache)
            shape_cache.volume = (
                0.0 if shape is None else abs(shape_to_volume(shape.wrapped))
            )
        return shape_cache.volume

    def _rounded_cad_for_ops(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        shape,
    ) -> "CAD":
        cad = CAD(
            deepcopy(planes),
            deepcopy(face_planes),
            deepcopy(cs),
            world_size=self.world_size,
            max_string_length=self.max_string_length,
            use_literals=self.use_literals,
        )

        bbox = shape.BoundingBox()
        x_min, y_min, z_min, x_max, y_max, z_max = (
            bbox.xmin,
            bbox.ymin,
            bbox.zmin,
            bbox.xmax,
            bbox.ymax,
            bbox.zmax,
        )
        bbox_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if np.isclose(bbox_size, 0):
            raise ValueError("Rounded-prefix validation found zero-size bbox")

        shift = [-(x_min + x_max) / 2, -(y_min + y_max) / 2, -(z_min + z_max) / 2]
        scale = self.world_size / bbox_size
        cad.transform(shift, scale)
        cad.round()
        return cad

    def _assert_rounded_operations_fast(self, cad: "CAD") -> None:
        """Validate rounded operations directly, before generated-code execution."""
        cad.fix_operations_in_place()

    def _assert_rounded_operations_survive_final_cleanup(self, cad: "CAD") -> None:
        """Fail generation if final cleanup would remove the rounded latest op."""
        expected_count = len(cad.cs)
        try:
            cad.fix_cascade(max_delete_ratio=0.3)
        except Exception as exc:
            raise ValueError(
                "Rounded-prefix final cleanup would reject the candidate operation"
            ) from exc
        if len(cad.cs) != expected_count:
            raise ValueError(
                "Rounded-prefix final cleanup deleted "
                f"{expected_count - len(cad.cs)} operation(s)"
            )

    def _faces_for_ops(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        shape_cache: _ShapeCache | None = None,
    ) -> list:
        if shape_cache is None:
            shape = self._shape_for_ops(planes, face_planes, cs)
            return [] if shape is None else shape.Faces()
        if shape_cache.faces is None:
            shape = self._shape_for_ops_cached(planes, face_planes, cs, shape_cache)
            shape_cache.faces = [] if shape is None else shape.Faces()
        return shape_cache.faces

    def _solid_count_for_ops(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
    ) -> int:
        s = CAD(
            planes,
            face_planes,
            cs,
            world_size=self.world_size,
            max_string_length=self.max_string_length,
            use_literals=self.use_literals,
        ).to_string()
        namespace: dict[str, object] = {}
        exec(s, globals(), namespace)
        return len(namespace["r"].val().Solids())

    def _assert_solid_count_unchanged(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
    ) -> None:
        if cs_len_before == 0:
            raise ValueError("Cannot compare solid count without an existing model")
        before_count = self._solid_count_for_ops(
            planes,
            face_planes,
            cs[:cs_len_before],
        )
        after_count = self._solid_count_for_ops(planes, face_planes, cs)
        if before_count != after_count:
            raise ValueError(
                f"operation changed solid count from {before_count} to {after_count}"
            )

    def _assert_solid_count_unchanged_for_operation(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        *,
        required: bool = False,
    ) -> None:
        if required or self.single_solid_check:
            self._assert_solid_count_unchanged(
                planes,
                face_planes,
                cs,
                cs_len_before,
            )

    @staticmethod
    def _face_bbox_values(face) -> np.ndarray:
        bbox = face.BoundingBox()
        return np.array(
            [bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax]
        )

    @staticmethod
    def _face_center_values(face) -> np.ndarray:
        return np.array(face.Center().toTuple())

    @staticmethod
    def _faces_match(existing_face, candidate_face, tolerance: float = 1e-7) -> bool:
        if (
            existing_face.wrapped.IsSame(candidate_face.wrapped)
            or existing_face.wrapped.IsEqual(candidate_face.wrapped)
        ):
            return True
        if existing_face.geomType() != candidate_face.geomType():
            return False
        return (
            np.isclose(
                existing_face.Area(),
                candidate_face.Area(),
                rtol=tolerance,
                atol=tolerance,
            )
            and np.allclose(
                CADFactory._face_center_values(existing_face),
                CADFactory._face_center_values(candidate_face),
                rtol=tolerance,
                atol=tolerance,
            )
            and np.allclose(
                CADFactory._face_bbox_values(existing_face),
                CADFactory._face_bbox_values(candidate_face),
                rtol=tolerance,
                atol=tolerance,
            )
        )

    @staticmethod
    def _changed_existing_face_count(existing_faces: list, candidate_faces: list) -> int:
        remaining_candidates = list(candidate_faces)
        changed_faces = 0
        for existing_face in existing_faces:
            match_idx = next(
                (
                    i
                    for i, candidate_face in enumerate(remaining_candidates)
                    if CADFactory._faces_match(existing_face, candidate_face)
                ),
                None,
            )
            if match_idx is None:
                changed_faces += 1
            else:
                remaining_candidates.pop(match_idx)
        return changed_faces

    @staticmethod
    def _extrude_volume(extrude: Extrude) -> float:
        return abs(shape_to_area(extrude.sketch.to_shape()) * extrude.extent)

    def _shell_volume(
        self,
        planes: list[dict],
        face_planes: list[dict],
        op: dict,
    ) -> float:
        shell_shape = self._shape_for_ops(
            planes,
            face_planes,
            [op],
        )
        if shell_shape is None:
            return 0.0
        return abs(shape_to_volume(shell_shape.wrapped))

    @staticmethod
    def _revolve_volume(revolve: Revolve) -> float:
        assert revolve.axis is not None
        sketch_shape = revolve.sketch.to_shape()
        area = abs(shape_to_area(sketch_shape))
        cx, cy, _ = get_face_center(sketch_shape)
        (x1, y1), (x2, y2) = revolve.axis
        dx, dy = x2 - x1, y2 - y1
        axis_length = math.hypot(dx, dy)
        if axis_length <= 1e-12:
            return 0.0
        radius = abs(dx * (cy - y1) - dy * (cx - x1)) / axis_length
        angle_radians = math.radians(abs(revolve.angle_degrees))
        return area * radius * angle_radians

    @staticmethod
    def _sweep_volume(sweep: Sweep) -> float:
        return abs(shape_to_volume(sweep.to_shape().wrapped))

    def _operation_volume(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
    ) -> float:
        op = cs[cs_len_before]
        if op["type"] == "Extrude":
            return self._extrude_volume(op["op"])
        if op["type"] == "Shell":
            return self._shell_volume(planes, face_planes, op)
        if op["type"] == "TwoExtrudes":
            operation_shape = self._shape_for_ops(
                planes,
                face_planes,
                [op],
            )
            return (
                0.0
                if operation_shape is None
                else abs(shape_to_volume(operation_shape.wrapped))
            )
        if op["type"] == "Revolve":
            return self._revolve_volume(op["op"])
        if op["type"] in ("Sweep", "SweepAdv"):
            return self._sweep_volume(op["op"])
        raise ValueError(f"Unsupported operation volume check for {op['type']}")

    def _operation_intersection_ratio(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_volume: float | None = None,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> float:
        if len(cs) != cs_len_before + 1 or cs_len_before == 0:
            return 0.0
        operation_volume = self._operation_volume(planes, face_planes, cs, cs_len_before)
        if operation_volume <= 1e-12:
            return 0.0
        if existing_volume is None:
            existing_shape = self._shape_for_ops(
                planes,
                face_planes,
                cs[:cs_len_before],
            )
            if existing_shape is None:
                return 0.0
            existing_volume = abs(shape_to_volume(existing_shape.wrapped))
        union_volume = self._volume_for_ops_cached(
            planes,
            face_planes,
            cs,
            candidate_shape_cache,
        )
        intersection_volume = max(0.0, existing_volume + operation_volume - union_volume)
        return intersection_volume / operation_volume

    def _extrude_intersection_ratio(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_volume: float | None = None,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> float:
        return self._operation_intersection_ratio(
            planes,
            face_planes,
            cs,
            cs_len_before,
            existing_volume,
            candidate_shape_cache,
        )

    def _fix_latest_operation(self, cs: list[dict], cs_len_before: int) -> None:
        op = cs[cs_len_before]
        try:
            op["op"].fix()
        except Exception as exc:
            raise ValueError(
                f"{op['type']} at op index {cs_len_before} failed operation fix"
            ) from exc

    def _assert_operation_change_allowed(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_faces: list | None = None,
        existing_volume: float | None = None,
        max_changed_faces: int = 2,
        max_intersection_ratio: float = 0.01,
        operation_name: str = "Operation",
        intersection_ratio_fn=None,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> None:
        if len(cs) != cs_len_before + 1:
            return
        self._fix_latest_operation(cs, cs_len_before)
        if cs_len_before == 0:
            return
        if intersection_ratio_fn is None:
            intersection_ratio_fn = self._operation_intersection_ratio
        if existing_faces is None:
            existing_faces = self._faces_for_ops(
                planes,
                face_planes,
                cs[:cs_len_before],
            )
        candidate_faces = self._faces_for_ops(
            planes,
            face_planes,
            cs,
            candidate_shape_cache,
        )
        changed_faces = self._changed_existing_face_count(existing_faces, candidate_faces)
        intersection_ratio = intersection_ratio_fn(
            planes,
            face_planes,
            cs,
            cs_len_before,
            existing_volume,
            candidate_shape_cache,
        )
        if (
            changed_faces > max_changed_faces
            and intersection_ratio > max_intersection_ratio
        ):
            raise ValueError(
                f"{operation_name} changed "
                f"{changed_faces} existing faces and intersects "
                f"{intersection_ratio:.4%} of its volume; maximums are "
                f"{max_changed_faces} face and {max_intersection_ratio:.2%} volume"
            )

    def _assert_extrude_change_allowed(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_faces: list | None = None,
        existing_volume: float | None = None,
        max_changed_faces: int = 3,
        max_intersection_ratio: float = 0.01,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> None:
        operation_type = (
            cs[cs_len_before].get("type") if len(cs) > cs_len_before else "Extrude"
        )
        operation_name = {
            "Shell": "Shell",
            "TwoExtrudes": "TwoExtrudes",
        }.get(operation_type, "Extrude")
        self._assert_operation_change_allowed(
            planes,
            face_planes,
            cs,
            cs_len_before,
            existing_faces,
            existing_volume,
            max_changed_faces,
            max_intersection_ratio,
            operation_name=operation_name,
            intersection_ratio_fn=self._extrude_intersection_ratio,
            candidate_shape_cache=candidate_shape_cache,
        )

    def _assert_revolve_change_allowed(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_faces: list | None = None,
        existing_volume: float | None = None,
        max_changed_faces: int = 2,
        max_intersection_ratio: float = 0.01,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> None:
        self._assert_operation_change_allowed(
            planes,
            face_planes,
            cs,
            cs_len_before,
            existing_faces,
            existing_volume,
            max_changed_faces,
            max_intersection_ratio,
            operation_name="Revolve",
            candidate_shape_cache=candidate_shape_cache,
        )

    def _assert_sweep_change_allowed(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_faces: list | None = None,
        existing_volume: float | None = None,
        max_changed_faces: int = 2,
        max_intersection_ratio: float = 0.01,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> None:
        self._assert_operation_change_allowed(
            planes,
            face_planes,
            cs,
            cs_len_before,
            existing_faces,
            existing_volume,
            max_changed_faces,
            max_intersection_ratio,
            operation_name="Sweep",
            candidate_shape_cache=candidate_shape_cache,
        )

    def _assert_orto_cut_change_allowed(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        existing_faces: list | None = None,
        candidate_shape_cache: _ShapeCache | None = None,
        max_changed_faces: int = 4,
    ) -> None:
        if len(cs) != cs_len_before + 1:
            return
        self._fix_latest_operation(cs, cs_len_before)
        # Changed-face rejection is disabled for OrtoCut. These cuts often
        # legitimately touch several existing faces, especially after a prior
        # cut has created narrow side faces.
        # if cs_len_before == 0:
        #     return
        # if existing_faces is None:
        #     existing_faces = self._faces_for_ops(
        #         planes,
        #         face_planes,
        #         cs[:cs_len_before],
        #     )
        # candidate_faces = self._faces_for_ops(
        #     planes,
        #     face_planes,
        #     cs,
        #     candidate_shape_cache,
        # )
        # changed_faces = self._changed_existing_face_count(
        #     existing_faces,
        #     candidate_faces,
        # )
        # if changed_faces > max_changed_faces:
        #     raise ValueError(
        #         f"OrtoCut changed {changed_faces} existing faces; "
        #         f"maximum is {max_changed_faces}"
        #     )

    def _assert_latest_operation_valid(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        previous_volume: float | None = None,
        require_volume_change: bool | None = None,
        candidate_shape_cache: _ShapeCache | None = None,
        operation_already_fixed: bool = False,
    ) -> float:
        if len(cs) != cs_len_before + 1:
            raise ValueError("Expected exactly one newly appended operation")

        op = cs[cs_len_before]
        if not operation_already_fixed:
            self._fix_latest_operation(cs, cs_len_before)

        if require_volume_change is None:
            require_volume_change = op["type"] != "SelectedFacePlane"

        candidate_shape = self._shape_for_ops_cached(
            planes,
            face_planes,
            cs,
            candidate_shape_cache,
        )
        if candidate_shape is None:
            raise ValueError(f"{op['type']} produced no shape")

        candidate_volume = self._volume_for_ops_cached(
            planes,
            face_planes,
            cs,
            candidate_shape_cache,
        )
        if np.isclose(candidate_volume, 0) or candidate_volume <= 0:
            raise ValueError(f"{op['type']} produced non-positive volume")

        rounded_cad = self._rounded_cad_for_ops(
            planes,
            face_planes,
            cs,
            candidate_shape,
        )
        try:
            self._assert_rounded_operations_fast(rounded_cad)
            self._assert_rounded_operations_survive_final_cleanup(rounded_cad)
        except Exception as exc:
            raise ValueError(
                f"{op['type']} failed fast rounded operation validation"
            ) from exc

        if require_volume_change and cs_len_before > 0:
            if previous_volume is None:
                previous_shape = self._shape_for_ops(
                    planes,
                    face_planes,
                    cs[:cs_len_before],
                )
                previous_volume = (
                    0.0
                    if previous_shape is None
                    else abs(shape_to_volume(previous_shape.wrapped))
                )
            if np.isclose(previous_volume, candidate_volume, rtol=1e-8):
                raise ValueError(f"{op['type']} did not change model volume")

        return candidate_volume

    def _assert_operation_attempt_valid(
        self,
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        cs_len_before: int,
        previous_volume: float | None = None,
        *,
        existing_faces: list | None = None,
        existing_volume: float | None = None,
        require_volume_change: bool | None = None,
        require_solid_count_unchanged: bool = False,
        candidate_shape_cache: _ShapeCache | None = None,
    ) -> float:
        if len(cs) != cs_len_before + 1:
            raise ValueError("Expected exactly one newly appended operation")

        if candidate_shape_cache is None:
            candidate_shape_cache = _ShapeCache()

        op_type = cs[cs_len_before]["type"]
        operation_already_fixed = False
        if previous_volume is None:
            previous_volume = existing_volume

        if op_type in {"Extrude", "Shell", "TwoExtrudes"}:
            self._assert_extrude_change_allowed(
                planes,
                face_planes,
                cs,
                cs_len_before,
                existing_faces,
                existing_volume,
                candidate_shape_cache=candidate_shape_cache,
            )
            operation_already_fixed = True
        elif op_type == "Revolve":
            self._assert_revolve_change_allowed(
                planes,
                face_planes,
                cs,
                cs_len_before,
                existing_faces,
                existing_volume,
                candidate_shape_cache=candidate_shape_cache,
            )
            operation_already_fixed = True
        elif op_type in ("Sweep", "SweepAdv"):
            self._assert_sweep_change_allowed(
                planes,
                face_planes,
                cs,
                cs_len_before,
                existing_faces,
                existing_volume,
                candidate_shape_cache=candidate_shape_cache,
            )
            operation_already_fixed = True
        elif op_type == "OrtoCut":
            self._assert_orto_cut_change_allowed(
                planes,
                face_planes,
                cs,
                cs_len_before,
                existing_faces,
                candidate_shape_cache=candidate_shape_cache,
            )
            require_solid_count_unchanged = True
            operation_already_fixed = True
        elif op_type == "Hole":
            require_solid_count_unchanged = True

        if require_solid_count_unchanged:
            self._assert_solid_count_unchanged_for_operation(
                planes,
                face_planes,
                cs,
                cs_len_before,
                required=True,
            )

        return self._assert_latest_operation_valid(
            planes,
            face_planes,
            cs,
            cs_len_before,
            previous_volume,
            require_volume_change=(
                require_volume_change
                if require_volume_change is not None
                else op_type != "SelectedFacePlane"
            ),
            candidate_shape_cache=candidate_shape_cache,
            operation_already_fixed=operation_already_fixed,
        )

    @staticmethod
    def _restore_generation_state(
        planes: list[dict],
        face_planes: list[dict],
        cs: list[dict],
        planes_len: int,
        face_planes_len: int,
        cs_len: int,
    ) -> None:
        del planes[planes_len:]
        del face_planes[face_planes_len:]
        del cs[cs_len:]

    def generate(self):
        r_min, r_max = 0.01, 1.0
        cs, planes, face_planes = list(), list(), list()
        sampled_factories = []
        previous_extrude = -1
        current_volume = 0.0

        _factories = expand_factory_entries(deepcopy(self.factories))

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

            cs_len_before = len(cs)
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
                        existing_context = None
                        if cs:
                            try:
                                existing_context = self._existing_model_and_sampler(
                                    block_factory["factory"],
                                    planes,
                                    face_planes,
                                    cs,
                                )
                            except Exception as e:
                                if self.log_exceptions:
                                    logger.warning(
                                        "Skipping placed extrude: failed to prepare "
                                        "surface sampler (factory=%d.%d, existing_ops=%d, "
                                        "error=%s: %s)",
                                        factory_number + 1,
                                        block_factory_idx + 1,
                                        len(cs),
                                        type(e).__name__,
                                        e,
                                    )
                                continue
                        for retry in range(self.n_extrude_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                if existing_context is not None:
                                    extrude = self._append_existing_extrude(
                                        block_factory,
                                        planes,
                                        face_planes,
                                        cs,
                                        existing_context[0],
                                        existing_context[1],
                                        r_max,
                                    )
                                else:
                                    plane_type = None
                                    if np.random.uniform() < block_factory.get(
                                        "reuse_plane_probability", 0
                                    ):
                                        plane_type = "reuse"
                                    elif np.random.uniform() < block_factory.get(
                                        "selected_plane_probability", 0
                                    ):
                                        plane_type = "selected"
                                    else:
                                        plane_type = "default"
                                    if isinstance(block_factory["factory"], ShellFactory):
                                        block_factory["factory"].world_size = (
                                            self.world_size
                                        )
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
                                        plane_type=plane_type,
                                    )
                                candidate_shape_cache = _ShapeCache()
                                self._assert_extrude_change_allowed(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[2]
                                    if existing_context is not None
                                    else None,
                                    existing_context[3]
                                    if existing_context is not None
                                    else None,
                                    candidate_shape_cache=candidate_shape_cache,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[3]
                                    if existing_context is not None
                                    else current_volume,
                                    candidate_shape_cache=candidate_shape_cache,
                                    operation_already_fixed=True,
                                )
                                current_volume = candidate_volume
                                previous_extrude = extrude
                                break
                            except Exception as e:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                self._log_extrude_retry_failure(
                                    factory_number=factory_number,
                                    block_factory_idx=block_factory_idx,
                                    retry=retry,
                                    max_retries=self.n_extrude_retries,
                                    error=e,
                                    using_existing_sampler=existing_context is not None,
                                )
                                if retry == self.n_extrude_retries - 1:
                                    if self.log_exceptions:
                                        logger.warning("Skipping failed extrude after %d retries", self.n_extrude_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], OrtoCutFactory):
                        if not cs:
                            continue
                        try:
                            existing_context = self._existing_model_and_sampler(
                                block_factory["factory"],
                                planes,
                                face_planes,
                                cs,
                            )
                        except Exception as e:
                            if self.log_exceptions:
                                logger.warning(
                                    "Skipping orto_cut: failed to prepare surface "
                                    "sampler (factory=%d.%d, existing_ops=%d, "
                                    "error=%s: %s)",
                                    factory_number + 1,
                                    block_factory_idx + 1,
                                    len(cs),
                                    type(e).__name__,
                                    e,
                                )
                            continue
                        for retry in range(self.n_extrude_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                self._append_existing_orto_cut(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    cs,
                                    existing_context[0],
                                    existing_context[1],
                                    r_max,
                                )
                                candidate_shape_cache = _ShapeCache()
                                self._assert_orto_cut_change_allowed(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[2],
                                    candidate_shape_cache=candidate_shape_cache,
                                )
                                self._assert_solid_count_unchanged_for_operation(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    required=True,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[3],
                                    candidate_shape_cache=candidate_shape_cache,
                                    operation_already_fixed=True,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception as e:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                if self.log_exceptions:
                                    logger.warning(
                                        "OrtoCut retry failed: factory=%d.%d "
                                        "retry=%d/%d error=%s: %s",
                                        factory_number + 1,
                                        block_factory_idx + 1,
                                        retry + 1,
                                        self.n_extrude_retries,
                                        type(e).__name__,
                                        e,
                                    )
                                if retry == self.n_extrude_retries - 1:
                                    if self.log_exceptions:
                                        logger.warning(
                                            "Skipping failed orto_cut after %d retries",
                                            self.n_extrude_retries,
                                        )
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], RevolveFactory):
                        existing_context = None
                        if cs:
                            try:
                                existing_context = self._existing_model_and_sampler(
                                    block_factory["factory"],
                                    planes,
                                    face_planes,
                                    cs,
                                )
                            except Exception:
                                if self.log_exceptions:
                                    logger.warning(
                                        "Skipping placed revolve: failed to prepare surface sampler",
                                        exc_info=True,
                                    )
                                continue
                        max_retries = (
                            self.n_extrude_retries
                            if existing_context is not None
                            else self.n_retries
                        )
                        for retry in range(max_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                if existing_context is not None:
                                    revolve = self._append_existing_revolve(
                                        block_factory,
                                        planes,
                                        face_planes,
                                        cs,
                                        existing_context[0],
                                        existing_context[1],
                                        r_max,
                                    )
                                else:
                                    block_factory["factory"].world_size = self.world_size
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
                                candidate_shape_cache = _ShapeCache()
                                self._assert_revolve_change_allowed(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[2]
                                    if existing_context is not None
                                    else None,
                                    existing_context[3]
                                    if existing_context is not None
                                    else None,
                                    candidate_shape_cache=candidate_shape_cache,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[3]
                                    if existing_context is not None
                                    else current_volume,
                                    candidate_shape_cache=candidate_shape_cache,
                                    operation_already_fixed=True,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == max_retries - 1:
                                    if self.log_exceptions:
                                        logger.warning(
                                            "Skipping failed operation after %d retries",
                                            max_retries,
                                        )
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], (SweepFactory, SweepAdvFactory)):
                        if not cs:
                            for retry in range(self.n_extrude_retries):
                                attempt_state = (len(planes), len(face_planes), len(cs))
                                try:
                                    self._append_initial_sweep(
                                        block_factory,
                                        planes,
                                        face_planes,
                                        cs,
                                        r_max,
                                    )
                                    candidate_shape_cache = _ShapeCache()
                                    candidate_volume = self._assert_operation_attempt_valid(
                                        planes,
                                        face_planes,
                                        cs,
                                        attempt_state[2],
                                        current_volume,
                                        candidate_shape_cache=candidate_shape_cache,
                                    )
                                    current_volume = candidate_volume
                                    break
                                except Exception as e:
                                    self._restore_generation_state(
                                        planes,
                                        face_planes,
                                        cs,
                                        *attempt_state,
                                    )
                                    if self.log_exceptions:
                                        logger.warning(
                                            "Initial sweep retry failed: factory=%d.%d "
                                            "retry=%d/%d error=%s: %s",
                                            factory_number + 1,
                                            block_factory_idx + 1,
                                            retry + 1,
                                            self.n_extrude_retries,
                                            type(e).__name__,
                                            e,
                                        )
                                    if retry == self.n_extrude_retries - 1:
                                        if self.log_exceptions:
                                            logger.warning(
                                                "Skipping failed sweep after %d retries",
                                                self.n_extrude_retries,
                                            )
                                        break
                                    else:
                                        continue
                            if len(cs) == cs_len_before + 1:
                                factory_dict = {}
                                for k, v in factory.items():
                                    if k != "factory":
                                        factory_dict[k] = v
                                factory_dict["factory"] = block_factory[
                                    "factory"
                                ].to_dict()
                                sampled_factories.append(factory_dict)
                                cs_len_before += 1
                            continue
                        try:
                            existing_context = self._existing_model_and_sampler(
                                block_factory["factory"],
                                planes,
                                face_planes,
                                cs,
                            )
                        except Exception as e:
                            if self.log_exceptions:
                                logger.warning(
                                    "Skipping sweep: failed to prepare surface sampler "
                                    "(factory=%d.%d, existing_ops=%d, error=%s: %s)",
                                    factory_number + 1,
                                    block_factory_idx + 1,
                                    len(cs),
                                    type(e).__name__,
                                    e,
                                )
                            continue
                        for retry in range(self.n_extrude_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                self._append_existing_sweep(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    cs,
                                    existing_context[0],
                                    existing_context[1],
                                    r_max,
                                )
                                candidate_shape_cache = _ShapeCache()
                                self._assert_sweep_change_allowed(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[2],
                                    existing_context[3],
                                    candidate_shape_cache=candidate_shape_cache,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    existing_context[3],
                                    candidate_shape_cache=candidate_shape_cache,
                                    operation_already_fixed=True,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception as e:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                if self.log_exceptions:
                                    logger.warning(
                                        "Sweep retry failed: factory=%d.%d "
                                        "retry=%d/%d error=%s: %s",
                                        factory_number + 1,
                                        block_factory_idx + 1,
                                        retry + 1,
                                        self.n_extrude_retries,
                                        type(e).__name__,
                                        e,
                                    )
                                if retry == self.n_extrude_retries - 1:
                                    if self.log_exceptions:
                                        logger.warning(
                                            "Skipping failed sweep after %d retries",
                                            self.n_extrude_retries,
                                        )
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], SweepInitFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                sweep = block_factory["factory"].generate()
                                generate_sweep_init_plane(
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
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], LoftFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
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
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], RibFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                rib_op = block_factory["factory"].generate()
                                generate_rib_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    rib_op,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], ThreadFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                thread_op = block_factory["factory"].generate()
                                generate_thread_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    thread_op,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], GearFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                block_factory["factory"].world_size = self.world_size
                                gear = block_factory["factory"].generate()
                                generate_gear_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    gear,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], HoleFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
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
                                self._assert_solid_count_unchanged_for_operation(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    required=True,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning(
                                        "Skipping failed hole after %d retries",
                                        self.n_retries,
                                        exc_info=True,
                                    )
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], CutThruAllFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
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
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], CircularPatternFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
                            try:
                                circular_pattern = block_factory["factory"].generate()
                                generate_circular_pattern_plane(
                                    block_factory,
                                    planes,
                                    face_planes,
                                    circular_pattern,
                                    r_min,
                                    r_max,
                                    cs,
                                    self.world_size,
                                    self.max_string_length,
                                    self.use_literals,
                                    CAD,
                                )
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], FaceFilletChamferFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
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
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], FilletChamferFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
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
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue

                    elif isinstance(block_factory["factory"], SelectedFacePlaneFactory):
                        for retry in range(self.n_retries):
                            attempt_state = (len(planes), len(face_planes), len(cs))
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
                                candidate_volume = self._assert_latest_operation_valid(
                                    planes,
                                    face_planes,
                                    cs,
                                    attempt_state[2],
                                    current_volume,
                                    require_volume_change=False,
                                )
                                current_volume = candidate_volume
                                break
                            except Exception:
                                self._restore_generation_state(
                                    planes,
                                    face_planes,
                                    cs,
                                    *attempt_state,
                                )
                                # print(
                                #     f"Failure on op {block_factory['factory'].__class__.__name__} number {factory_number + 1}.{block_factory_idx + 1}, retry {retry + 1}/{self.n_retries}"
                                # )
                                if retry == self.n_retries - 1:
                                    logger.warning("Skipping failed operation after %d retries", self.n_retries)
                                    break
                                else:
                                    continue
                    if len(cs) == cs_len_before + 1:
                        factory_dict = {}
                        for k, v in factory.items():
                            if k != "factory":
                                factory_dict[k] = v
                        factory_dict["factory"] = block_factory["factory"].to_dict()
                        sampled_factories.append(factory_dict)
                        cs_len_before += 1
                continue

            elif isinstance(factory["factory"], ExtrudeFactory):
                existing_context = None
                if cs:
                    try:
                        existing_context = self._existing_model_and_sampler(
                            factory["factory"],
                            planes,
                            face_planes,
                            cs,
                        )
                    except Exception as e:
                        if self.log_exceptions:
                            logger.warning(
                                "Skipping placed extrude: failed to prepare surface "
                                "sampler (factory=%d, existing_ops=%d, error=%s: %s)",
                                factory_number + 1,
                                len(cs),
                                type(e).__name__,
                                e,
                            )
                        continue
                for retry in range(self.n_extrude_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        if existing_context is not None:
                            extrude = self._append_existing_extrude(
                                factory,
                                planes,
                                face_planes,
                                cs,
                                existing_context[0],
                                existing_context[1],
                                r_max,
                            )
                        else:
                            if isinstance(factory["factory"], ShellFactory):
                                factory["factory"].world_size = self.world_size
                            extrude = factory["factory"].generate()
                            plane_type = None
                            if np.random.uniform() < factory.get(
                                "reuse_plane_probability", 0
                            ):
                                plane_type = "reuse"
                            elif np.random.uniform() < factory.get(
                                "selected_plane_probability", 0
                            ):
                                plane_type = "selected"
                            else:
                                plane_type = "default"

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
                                plane_type=plane_type,
                            )
                        candidate_shape_cache = _ShapeCache()
                        self._assert_extrude_change_allowed(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[2]
                            if existing_context is not None
                            else None,
                            existing_context[3]
                            if existing_context is not None
                            else None,
                            candidate_shape_cache=candidate_shape_cache,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[3]
                            if existing_context is not None
                            else current_volume,
                            candidate_shape_cache=candidate_shape_cache,
                            operation_already_fixed=True,
                        )
                        current_volume = candidate_volume
                        previous_extrude = extrude
                        break
                    except Exception as e:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        self._log_extrude_retry_failure(
                            factory_number=factory_number,
                            retry=retry,
                            max_retries=self.n_extrude_retries,
                            error=e,
                            using_existing_sampler=existing_context is not None,
                        )
                        if retry == self.n_extrude_retries - 1:
                            if self.log_exceptions:
                                logger.warning("Skipping failed extrude after %d retries", self.n_extrude_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], OrtoCutFactory):
                if not cs:
                    continue
                try:
                    existing_context = self._existing_model_and_sampler(
                        factory["factory"],
                        planes,
                        face_planes,
                        cs,
                    )
                except Exception as e:
                    if self.log_exceptions:
                        logger.warning(
                            "Skipping orto_cut: failed to prepare surface sampler "
                            "(factory=%d, existing_ops=%d, error=%s: %s)",
                            factory_number + 1,
                            len(cs),
                            type(e).__name__,
                            e,
                        )
                    continue
                for retry in range(self.n_extrude_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        self._append_existing_orto_cut(
                            factory,
                            planes,
                            face_planes,
                            cs,
                            existing_context[0],
                            existing_context[1],
                            r_max,
                        )
                        candidate_shape_cache = _ShapeCache()
                        self._assert_orto_cut_change_allowed(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[2],
                            candidate_shape_cache=candidate_shape_cache,
                        )
                        self._assert_solid_count_unchanged_for_operation(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            required=True,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[3],
                            candidate_shape_cache=candidate_shape_cache,
                            operation_already_fixed=True,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception as e:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        if self.log_exceptions:
                            logger.warning(
                                "OrtoCut retry failed: factory=%d retry=%d/%d "
                                "error=%s: %s",
                                factory_number + 1,
                                retry + 1,
                                self.n_extrude_retries,
                                type(e).__name__,
                                e,
                            )
                        if retry == self.n_extrude_retries - 1:
                            if self.log_exceptions:
                                logger.warning(
                                    "Skipping failed orto_cut after %d retries",
                                    self.n_extrude_retries,
                                )
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], RevolveFactory):
                existing_context = None
                if cs:
                    try:
                        existing_context = self._existing_model_and_sampler(
                            factory["factory"],
                            planes,
                            face_planes,
                            cs,
                        )
                    except Exception:
                        if self.log_exceptions:
                            logger.warning(
                                "Skipping placed revolve: failed to prepare surface sampler",
                                exc_info=True,
                            )
                        continue
                max_retries = (
                    self.n_extrude_retries
                    if existing_context is not None
                    else self.n_retries
                )
                for retry in range(max_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        if existing_context is not None:
                            revolve = self._append_existing_revolve(
                                factory,
                                planes,
                                face_planes,
                                cs,
                                existing_context[0],
                                existing_context[1],
                                r_max,
                            )
                        else:
                            factory["factory"].world_size = self.world_size
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
                        candidate_shape_cache = _ShapeCache()
                        self._assert_revolve_change_allowed(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[2]
                            if existing_context is not None
                            else None,
                            existing_context[3]
                            if existing_context is not None
                            else None,
                            candidate_shape_cache=candidate_shape_cache,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[3]
                            if existing_context is not None
                            else current_volume,
                            candidate_shape_cache=candidate_shape_cache,
                            operation_already_fixed=True,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == max_retries - 1:
                            if self.log_exceptions:
                                logger.warning(
                                    "Skipping failed operation after %d retries",
                                    max_retries,
                                )
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], (SweepFactory, SweepAdvFactory)):
                if not cs:
                    for retry in range(self.n_extrude_retries):
                        attempt_state = (len(planes), len(face_planes), len(cs))
                        try:
                            self._append_initial_sweep(
                                factory,
                                planes,
                                face_planes,
                                cs,
                                r_max,
                            )
                            candidate_shape_cache = _ShapeCache()
                            candidate_volume = self._assert_operation_attempt_valid(
                                planes,
                                face_planes,
                                cs,
                                attempt_state[2],
                                current_volume,
                                candidate_shape_cache=candidate_shape_cache,
                            )
                            current_volume = candidate_volume
                            break
                        except Exception as e:
                            self._restore_generation_state(
                                planes,
                                face_planes,
                                cs,
                                *attempt_state,
                            )
                            if self.log_exceptions:
                                logger.warning(
                                    "Initial sweep retry failed: factory=%d "
                                    "retry=%d/%d error=%s: %s",
                                    factory_number + 1,
                                    retry + 1,
                                    self.n_extrude_retries,
                                    type(e).__name__,
                                    e,
                                )
                            if retry == self.n_extrude_retries - 1:
                                if self.log_exceptions:
                                    logger.warning(
                                        "Skipping failed sweep after %d retries",
                                        self.n_extrude_retries,
                                    )
                                break
                            else:
                                continue
                    if len(cs) == cs_len_before + 1:
                        factory_dict = {}
                        for k, v in factory.items():
                            if k != "factory":
                                factory_dict[k] = v
                        factory_dict["factory"] = factory["factory"].to_dict()
                        sampled_factories.append(factory_dict)
                    continue
                try:
                    existing_context = self._existing_model_and_sampler(
                        factory["factory"],
                        planes,
                        face_planes,
                        cs,
                    )
                except Exception as e:
                    if self.log_exceptions:
                        logger.warning(
                            "Skipping sweep: failed to prepare surface sampler "
                            "(factory=%d, existing_ops=%d, error=%s: %s)",
                            factory_number + 1,
                            len(cs),
                            type(e).__name__,
                            e,
                        )
                    continue
                for retry in range(self.n_extrude_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        self._append_existing_sweep(
                            factory,
                            planes,
                            face_planes,
                            cs,
                            existing_context[0],
                            existing_context[1],
                            r_max,
                        )
                        candidate_shape_cache = _ShapeCache()
                        self._assert_sweep_change_allowed(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[2],
                            existing_context[3],
                            candidate_shape_cache=candidate_shape_cache,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            existing_context[3],
                            candidate_shape_cache=candidate_shape_cache,
                            operation_already_fixed=True,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception as e:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        if self.log_exceptions:
                            logger.warning(
                                "Sweep retry failed: factory=%d retry=%d/%d "
                                "error=%s: %s",
                                factory_number + 1,
                                retry + 1,
                                self.n_extrude_retries,
                                type(e).__name__,
                                e,
                            )
                        if retry == self.n_extrude_retries - 1:
                            if self.log_exceptions:
                                logger.warning(
                                    "Skipping failed sweep after %d retries",
                                    self.n_extrude_retries,
                                )
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], SweepInitFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        sweep = factory["factory"].generate()

                        generate_sweep_init_plane(
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
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], LoftFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
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
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], RibFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        rib_op = factory["factory"].generate()
                        generate_rib_plane(
                            factory,
                            planes,
                            face_planes,
                            rib_op,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], ThreadFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        thread_op = factory["factory"].generate()
                        generate_thread_plane(
                            factory,
                            planes,
                            face_planes,
                            thread_op,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], GearFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        factory["factory"].world_size = self.world_size
                        gear = factory["factory"].generate()

                        generate_gear_plane(
                            factory,
                            planes,
                            face_planes,
                            gear,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], HoleFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
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
                        self._assert_solid_count_unchanged_for_operation(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            required=True,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning(
                                "Skipping failed hole after %d retries",
                                self.n_retries,
                                exc_info=True,
                            )
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], CutThruAllFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
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
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], FaceFilletChamferFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
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
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], FilletChamferFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
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
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], SelectedFacePlaneFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
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
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                            require_volume_change=False,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue

            elif isinstance(factory["factory"], CircularPatternFactory):
                for retry in range(self.n_retries):
                    attempt_state = (len(planes), len(face_planes), len(cs))
                    try:
                        circular_pattern = factory["factory"].generate()
                        generate_circular_pattern_plane(
                            factory,
                            planes,
                            face_planes,
                            circular_pattern,
                            r_min,
                            r_max,
                            cs,
                            self.world_size,
                            self.max_string_length,
                            self.use_literals,
                            CAD,
                        )
                        candidate_volume = self._assert_latest_operation_valid(
                            planes,
                            face_planes,
                            cs,
                            attempt_state[2],
                            current_volume,
                        )
                        current_volume = candidate_volume
                        break
                    except Exception:
                        self._restore_generation_state(
                            planes,
                            face_planes,
                            cs,
                            *attempt_state,
                        )
                        # print(
                        #     f"Failure on op {factory['factory']} number {factory_number + 1}, retry {retry + 1}/{self.n_retries}"
                        # )
                        if retry == self.n_retries - 1:
                            logger.warning("Skipping failed operation after %d retries", self.n_retries)
                            break
                        else:
                            continue
            if len(cs) == cs_len_before + 1:
                # factory_dict = factory["factory"].to_dict()
                factory_dict = {}
                for k, v in factory.items():
                    if k != "factory":
                        factory_dict[k] = v
                factory_dict["factory"] = factory["factory"].to_dict()
                sampled_factories.append(factory_dict)
        # sort per plane sketches by extent
        # sorted_cs = list()
        # for i in range(len(planes)):
        #     plane_sketches = [
        #         s for s in cs if s.get("plane", None) == i and s["type"] == "Sketch"
        #     ]
        #     if len(plane_sketches) == 0:
        #         continue
        #     if len(plane_sketches) == 1:
        #         sorted_cs.extend(plane_sketches)
        #     else:
        #         plane_extents = [s["extent"] for s in plane_sketches]
        #         ids = np.argsort(plane_extents)
        #         sorted_cs.extend(itemgetter(*ids)(plane_sketches))
        # for i, op in enumerate(cs):
        #     if op["type"] != "Sketch" or isinstance(op.get("plane", None), str):
        #         sorted_cs.insert(i, op)
        # cs = sorted_cs

        logger.info(
            "CADFactory assembled CAD with %d planes and %d operations",
            len(planes),
            len(cs),
        )
        if not cs:
            raise EmptyCADGeneratedError(
                "Empty CAD generated: no operation was successfully constructed"
            )
        cad = CAD(
            planes,
            face_planes,
            cs,
            sampled_factories,
            self.world_size,
            self.max_string_length,
            self.use_literals,
        )
        # sampled_cad_factory = {
        #     "type": "CADFactory",
        #     "factories": sampled_factories,
        #     "world_size": self.world_size,
        #     "max_string_length": self.max_string_length,
        #     "use_literals": self.use_literals,
        #     "n_retries": self.n_retries,
        #     "n_extrude_retries": self.n_extrude_retries,
        # }

        # if return_factory:
        #     return cad, sampled_cad_factory
        # else:
        return cad

    @staticmethod
    def from_dict(entity: dict) -> "CADFactory":
        assert entity["type"] == "CADFactory"
        return CADFactory(
            factories=entity["factories"],
            world_size=entity["world_size"],
            max_string_length=entity["max_string_length"],
            use_literals=entity["use_literals"],
            n_retries=entity["n_retries"],
            n_extrude_retries=entity["n_extrude_retries"],
            log_exceptions=entity.get("log_exceptions", False),
            skip_fix=entity.get("skip_fix", False),
            single_solid_check=entity.get("single_solid_check", True),
        )

    def to_dict(self) -> dict:
        return {
            "type": "CADFactory",
            "factories": [factory.to_dict() for factory in self.factories],
            "world_size": self.world_size,
            "max_string_length": self.max_string_length,
            "use_literals": self.use_literals,
            "n_retries": self.n_retries,
            "n_extrude_retries": self.n_extrude_retries,
            "log_exceptions": self.log_exceptions,
            "skip_fix": self.skip_fix,
            "single_solid_check": self.single_solid_check,
        }
