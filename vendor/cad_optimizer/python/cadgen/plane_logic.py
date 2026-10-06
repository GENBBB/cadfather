from __future__ import annotations
import logging
from typing import Any

import numpy as np

from .cut_thru_all import CutThruAll
from .edge_operations import Chamfer, Fillet
from .extrude import Extrude
from .face_operations import FaceChamfer, FaceFillet
from .hole import Hole
from .loft import Loft
from .revolve import Revolve
from .selected_face_plane import SelectedFacePlane
from .sselectors import *
from .sweep import Sweep
from .utils import (
    get_point_on_face,
    is_face_outer,
    is_planar_face,
)

logger = logging.getLogger(__name__)


def generate_selected_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    selected_face_plane: SelectedFacePlane,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    s = cad_cls(
        planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()
    exec(s, globals())
    r = globals()["r"]

    # areas = np.array([f.Area() for f in r.val().Faces()])
    # areas = areas / areas.sum()
    # order = np.argsort(-areas)
    # faces = np.array(r.val().Faces())[
    #     order[np.cumsum(areas[order]) <= selected_face_plane.top_p]
    # ].tolist()
    faces = r.val().Faces()
    faces = [f for f in faces if is_face_outer(r, f)]
    areas = np.array([f.Area() for f in faces])
    areas = areas / areas.sum()
    order = np.argsort(-areas)
    faces = np.array(faces)[
        order[np.cumsum(areas[order]) <= selected_face_plane.top_p]
    ].tolist()

    face = np.random.choice(faces)
    point_on_face = get_point_on_face(face)

    face_name = f"face_sel{len(face_planes)}"
    face_plane_name = f"face_w{len(face_planes)}"
    point_name = f"point_sel{len(face_planes)}"

    face_str = (
        f"{point_name} = [{point_on_face[0]}, {point_on_face[1]}, {point_on_face[2]}]\n"
    )
    face_str += f"{face_name} = r.faces(PointOnFaceSelector({point_name}))\n"
    face_planar = is_planar_face(face)

    if face_planar:
        plane = f"{face_str}{face_plane_name}={face_name}.workplane(origin={face_name}.val().Center())\n"
    else:
        plane = face_str

    selected_face_plane.point_on_face = point_on_face
    selected_face_plane.face_name = face_name
    selected_face_plane.point_name = point_name
    selected_face_plane.face_plane_name = face_plane_name
    selected_face_plane.planar = face_planar

    exec(plane, globals())

    if face_planar:
        workplane = globals()[f"{face_plane_name}"]

        origin = workplane.plane.origin.toTuple()
        xdir = workplane.plane.xDir.toTuple()
        ydir = workplane.plane.yDir.toTuple()
        zdir = workplane.plane.zDir.toTuple()
        plane_axes = (xdir, ydir, zdir)

        face_planes.append(dict(origin=origin, plane_axes=plane_axes))
    else:
        origin = None
        plane_axes = None

        face_planes.append(dict(origin=None, plane_axes=None))

    cs.append(
        dict(
            type="SelectedFacePlane",
            selected_face_plane=selected_face_plane,
            plane=face_plane_name if face_planar else point_name,
            plane_axes=plane_axes,
        )
    )


def generate_extrude_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    extrude: Extrude,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    if factory["plane"] > len(planes) - 1:
        axis = np.random.randint(3)
        # why 4?
        origin = np.random.uniform(-r_max / 4, r_max / 4)
        planes.append(dict(axis=axis, origin=origin))
        logger.info(
            "Created new plane %d (axis=%d, origin=%s)",
            len(planes) - 1,
            axis,
            origin,
        )

    if np.random.uniform() < factory["selected_plane_probability"]:
        ax = np.random.choice(["X", "Y", "Z"])
        sign = np.random.choice(["<", ">"])
        face = f"r=r.faces('{sign}{ax}')"
        face_plane_name = f"face_w{len(face_planes)}"
        plane = f"{face}\n{face_plane_name}=r.workplane(origin=r.val().Center())\nr={face_plane_name}"

        s = cad_cls(
            planes,
            cs,
            world_size,
            max_string_length,
            use_literals,
        ).to_string()

        s += face + "\nworkplane=r.workplane(origin=r.val().Center())"
        exec(s, globals())
        w = globals()["r"].val()
        workplane = globals()["workplane"]

        assert is_planar_face(w), "Cannot select non-planar face for new Workplane"
        w_bbox = w.BoundingBox()

        origin = workplane.plane.origin.toTuple()
        xdir = workplane.plane.xDir.toTuple()
        ydir = workplane.plane.yDir.toTuple()
        zdir = workplane.plane.zDir.toTuple()
        plane_axes = (xdir, ydir, zdir)
        extrude.extent = abs(extrude.extent)

        face_string = (
            s
            + "\n"
            + "face_center="
            + extrude.sketch.to_string_one_sketch("workplane", extrude.sketch.wires)
            + ".finalize().val()._faces.BoundingBox()\n"
            + "face_center=[0.5*(face_center.xmin+face_center.xmax), 0.5*(face_center.ymin+face_center.ymax)]"
        )

        exec(face_string, globals())
        face_center = globals()["face_center"]

        if np.random.uniform() < factory["selected_plane_centered_probability"]:
            add = [0, 0]
        else:
            if "X" in plane:
                middle1 = 0.5 * (w_bbox.ymin + w_bbox.ymax)
                middle2 = 0.5 * (w_bbox.zmin + w_bbox.zmax)
                add = [
                    np.random.uniform(w_bbox.ymin, w_bbox.ymax) - middle1,
                    np.random.uniform(w_bbox.zmin, w_bbox.zmax) - middle2,
                ]
            elif "Y" in plane:
                middle1 = 0.5 * (w_bbox.xmin + w_bbox.xmax)
                middle2 = 0.5 * (w_bbox.zmin + w_bbox.zmax)
                add = [
                    np.random.uniform(w_bbox.xmin, w_bbox.xmax) - middle1,
                    np.random.uniform(w_bbox.zmin, w_bbox.zmax) - middle2,
                ]
            else:
                middle1 = 0.5 * (w_bbox.xmin + w_bbox.xmax)
                middle2 = 0.5 * (w_bbox.ymin + w_bbox.ymax)
                add = [
                    np.random.uniform(w_bbox.xmin, w_bbox.xmax) - middle1,
                    np.random.uniform(w_bbox.ymin, w_bbox.ymax) - middle2,
                ]

        extrude.sketch.transform(
            [-face_center[0] + add[0], -face_center[1] + add[1]], 1
        )
        face_string = (
            s
            + "\n"
            + "face_center="
            + extrude.sketch.to_string_one_sketch("workplane", extrude.sketch.wires)
            + ".finalize().val()._faces.BoundingBox()\n"
            + "face_center=[0.5*(face_center.xmin+face_center.xmax), 0.5*(face_center.ymin+face_center.ymax)]"
        )
        exec(face_string, globals())
        face_center2 = globals()["face_center"]
        assert np.allclose(face_center2, add)

        extrude.sketch.orient()
        face_planes.append(dict(origin=origin, plane_axes=plane_axes))
    else:
        plane = factory["plane"]
        plane_axes = None

        extrude.sketch.orient()

    logger.info("Generated extrude with extent=%f", extrude.extent)

    cs.append(
        dict(
            type="Extrude",
            extrude=extrude,
            plane=plane,
            plane_axes=plane_axes,
        )
    )


def generate_revolve_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    revolve: Revolve,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    if factory["plane"] > len(planes) - 1:
        axis = np.random.randint(3)
        # why 4?
        origin = np.random.uniform(-r_max / 4, r_max / 4)
        planes.append(dict(axis=axis, origin=origin))
        logger.info(
            "Created new plane %d (axis=%d, origin=%s)",
            len(planes) - 1,
            axis,
            origin,
        )

    plane = factory["plane"]

    cs.append(
        dict(
            type="Revolve",
            revolve=revolve,
            plane=plane,
        )
    )
    logger.info(
        "Revolve sketch appended: dist_to_axis=%s, plane=%d",
        revolve.dist_to_axis,
        plane,
    )


def generate_sweep_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    sweep: Sweep,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    target_plane = factory.get("plane", len(planes))
    while target_plane > len(planes) - 1:
        axis = np.random.randint(3)
        origin = np.random.uniform(-r_max / 4, r_max / 4)
        planes.append(dict(axis=axis, origin=origin))
        logger.info(
            "Created base plane %d (axis=%d, origin=%s) for sweep",
            len(planes) - 1,
            axis,
            origin,
        )

    plane = min(target_plane, len(planes) - 1)

    cs.append(
        dict(
            type="Sweep",
            sweep=sweep,
            plane=plane,
        )
    )
    logger.info(
        "Added sweep operation: pitch=%s, height=%s, radius=%s",
        sweep.pitch,
        sweep.height,
        sweep.radius,
    )


def generate_loft_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    loft: Loft,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    target_plane = factory.get("plane", len(planes))
    while target_plane > len(planes) - 1:
        axis = np.random.randint(3)
        origin = np.random.uniform(-r_max / 4, r_max / 4)
        planes.append(dict(axis=axis, origin=origin))
        logger.info(
            "Created base plane %d (axis=%d, origin=%s) for loft",
            len(planes) - 1,
            axis,
            origin,
        )

    plane = min(target_plane, len(planes) - 1)
    cs.append(
        dict(
            type="Loft",
            loft=loft,
            plane=plane,
        )
    )

    logger.info(
        "Added loft operation with %d sections",
        len(loft.sections),
    )


def generate_hole_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    hole: Hole,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    good_face_indices = [
        i for i in range(len(face_planes)) if face_planes[i]["origin"] is not None
    ]
    try:
        plane = f"face_w{np.random.choice(good_face_indices)}"
    except:
        return

    s = cad_cls(
        planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    hole = factory["factory"].generate(s=s, plane=plane)

    cs.append(dict(type="Hole", hole=hole, plane=plane))

    logger.info(
        "Hole appended: plane=%d",
        plane,
    )


def generate_cut_thru_all_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    cut_thru_all: CutThruAll,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    try:
        face_plane_idx = np.random.choice(np.arange(len(face_planes)))
    except:
        return

    if face_planes[face_plane_idx]["origin"] is not None:
        plane = f"face_w{face_plane_idx}"
    else:
        plane = f"point_sel{face_plane_idx}"

    s = cad_cls(
        planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    cutThruAll = factory["factory"].generate(s=s, plane=plane, world_size=world_size)

    cs.append(dict(type="cutThruAll", cut_thru_all=cutThruAll, plane=plane))

    logger.info(
        "cutThruAll appended: plane=%d",
        plane,
    )


def generate_face_fillet_chamfer_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    face_fillet_chamfer: FaceFillet | FaceChamfer,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    s = cad_cls(
        planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    face_fillet_chamfer = factory["factory"].generate(s)
    if isinstance(face_fillet_chamfer, FaceFillet):
        cs.append(
            dict(
                type="FaceFillet",
                face_fillet=face_fillet_chamfer,
            )
        )
    else:
        cs.append(dict(type="FaceChamfer", face_chamfer=face_fillet_chamfer))
    logger.info(
        "Added face operation: %s on axis %s",
        cs[-1]["type"],
        face_fillet_chamfer.axis,
    )


def generate_fillet_chamfer_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    fillet_chamfer: Fillet | Chamfer,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    s = cad_cls(
        planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    fillet_chamfer = factory["factory"].generate(s)

    if isinstance(fillet_chamfer, Fillet):
        cs.append(dict(type="Fillet", fillet=fillet_chamfer))
    else:
        cs.append(dict(type="Chamfer", chamfer=fillet_chamfer))
    logger.info(
        "Added edge operation: %s on point %s",
        cs[-1]["type"],
        fillet_chamfer.point,
    )
