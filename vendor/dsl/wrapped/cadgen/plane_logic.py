import logging
from typing import Any

import numpy as np

# from cadquery_addons import *

from .circular_pattern import CircularPattern
from .cut_thru_all import CutThruAll
from .edge_operations import Chamfer, Fillet
from .extrude import Extrude
from .face_operations import FaceChamfer, FaceFillet
from .gear import Gear
from .hole import Hole
from .loft import Loft
from .revolve import Revolve
from .rib import Rib
from .thread import Thread
from .selected_face_plane import SelectedFacePlane
from .sweep_init import SweepInit
from .utils import (
    is_planar_face,
)
from .surface_sampling import PreparedSurfaceSampler

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
        face_planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()
    exec(s, globals())
    r = globals()["r"]

    site = PreparedSurfaceSampler.from_cad_object(r).sample_site()
    face = site.face
    point_on_face = site.point.toTuple()

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
            op=selected_face_plane,
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
    plane_type: str | None = None,
):
    if factory["plane"] > len(planes) - 1:
        axis = np.random.randint(3)
        # why 4?
        origin = np.random.uniform(-r_max / 4, r_max / 4)
        planes.append(dict(axis=axis, origin=origin))
        plane = len(planes) - 1
        logger.info(
            "Created new plane %d (axis=%d, origin=%s)",
            len(planes) - 1,
            axis,
            origin,
        )
        # plane = factory["plane"]
        plane_axes = None
    else:
        plane = factory["plane"]
        plane_axes = None

    if plane_type is None:
        if np.random.uniform() < factory.get("reuse_plane_probability", 0):
            plane_type = "reuse"
        elif np.random.uniform() < factory.get("selected_plane_probability", 0):
            plane_type = "selected"
        else:
            plane_type = "default"

    if plane_type == "reuse":
        good_face_planes = [
            f"face_w{i}"
            for i, plane in enumerate(face_planes)
            if plane["origin"] is not None
        ]
        if np.random.uniform() < 0.5 and len(good_face_planes) > 0:
            good_planes = good_face_planes
        else:
            good_planes = [i for i in range(len(planes))]
        plane = good_planes[np.random.randint(len(good_planes))]
        # plane = factory["plane"]
        plane_axes = None

    elif plane_type == "selected":
        ax = np.random.choice(["X", "Y", "Z"])
        sign = np.random.choice(["<", ">"])
        face = f"r=r.faces('{sign}{ax}')"
        face_plane_name = f"face_w{len(face_planes)}"
        plane = f"{face}\n{face_plane_name}=r.workplane(origin=r.val().Center())\nr={face_plane_name}"

        s = cad_cls(
            planes,
            face_planes,
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

        extrude.sketch.orient()

    logger.info("Generated extrude with extent=%f", extrude.extent)

    cs.append(
        dict(
            type=type(extrude).__name__,
            op=extrude,
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
        plane = len(planes) - 1
        logger.info(
            "Created new plane %d (axis=%d, origin=%s)",
            len(planes) - 1,
            axis,
            origin,
        )
    else:
        plane = factory["plane"]

    cs.append(
        dict(
            type="Revolve",
            op=revolve,
            plane=plane,
        )
    )
    logger.info(
        "Revolve sketch appended: axis=%s, plane=%d",
        revolve.axis,
        plane,
    )


def generate_sweep_init_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    sweep: SweepInit,
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
            "Created base plane %d (axis=%d, origin=%s) for sweep_init",
            len(planes) - 1,
            axis,
            origin,
        )

    plane = min(target_plane, len(planes) - 1)

    cs.append(
        dict(
            type="SweepInit",
            op=sweep,
            plane=plane,
        )
    )
    logger.info(
        "Added sweep_init operation: pitch=%s, height=%s, radius=%s",
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
            op=loft,
            plane=plane,
        )
    )

    logger.info(
        "Added loft operation with %d sections",
        len(loft.sections),
    )


def generate_rib_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    rib: Rib,
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
            "Created base plane %d (axis=%d, origin=%s) for rib",
            len(planes) - 1,
            axis,
            origin,
        )

    plane = min(target_plane, len(planes) - 1)
    cs.append(dict(type="Rib", op=rib, plane=plane))
    logger.info("Added rib operation with %d ribs", len(rib.ribs))


def generate_thread_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    thread: Thread,
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
            "Created base plane %d (axis=%d, origin=%s) for thread",
            len(planes) - 1,
            axis,
            origin,
        )

    plane = min(target_plane, len(planes) - 1)
    cs.append(dict(type="Thread", op=thread, plane=plane))
    logger.info(
        "Added thread operation: R=%s H=%s pitch=%s profile=%s",
        thread.R, thread.H, thread.pitch, thread.profile,
    )


def generate_gear_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    gear: Gear,
    r_min: float,
    r_max: float,
    cs: list[dict],
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    cad_cls,
):
    if factory.get("plane", len(planes)) > len(planes) - 1:
        axis = np.random.randint(3)
        origin = np.random.uniform(-r_max / 4, r_max / 4)
        planes.append(dict(axis=axis, origin=origin))
        plane = len(planes) - 1
    else:
        plane = factory.get("plane", 0)

    if np.random.uniform() < factory.get("reuse_plane_probability", 0):
        good_planes = [i for i in range(len(planes))]
        if good_planes:
            plane = good_planes[np.random.randint(len(good_planes))]

    cs.append(dict(type="Gear", op=gear, plane=plane))
    logger.info(
        "Added gear operation: plane=%s, outer_radius=%s, inner_radius=%s, outer_teeth=%s, inner_teeth=%s",
        plane,
        gear.outer_radius,
        gear.inner_radius,
        gear.number_outer_teeth,
        gear.number_inner_teeth,
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
    s = cad_cls(
        planes,
        face_planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    hole = factory["factory"].generate(
        s=s,
        plane=None,
        point_plane=None,
        world_size=world_size,
    )
    cs.append(dict(type="Hole", op=hole, plane=None))
    logger.info("Hole appended at sampled point=%s", hole.point)


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
    good_face_indices = [i for i in range(len(face_planes))]
    np.random.shuffle(good_face_indices)

    s = cad_cls(
        planes,
        face_planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    for good_face_index in good_face_indices:
        if face_planes[good_face_index]["origin"] is not None:
            plane = f"face_w{good_face_index}"
        else:
            plane = f"point_sel{good_face_index}"

        try:
            cutThruAll = factory["factory"].generate(
                s=s, plane=plane, world_size=world_size
            )

            cs.append(dict(type="cutThruAll", op=cutThruAll, plane=plane))

            logger.info(
                "cutThruAll appended: plane=%s",
                plane,
            )
            break
        except Exception as e:
            pass


def generate_circular_pattern_plane(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    circular_pattern: CircularPattern,
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
        face_planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    circular_pattern = factory["factory"].generate(s=s)

    cs.append(dict(type="CircularPattern", op=circular_pattern, plane=None))

    logger.info(
        "CircularPattern appended: plane=None",
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
    world_size_continue: float = 1.0,
):
    s = cad_cls(
        planes,
        face_planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    face_fillet_chamfer = factory["factory"].generate(s, world_size=world_size_continue)
    if isinstance(face_fillet_chamfer, FaceFillet):
        cs.append(
            dict(
                type="FaceFillet",
                op=face_fillet_chamfer,
            )
        )
    else:
        cs.append(dict(type="FaceChamfer", op=face_fillet_chamfer))
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
    world_size_continue: float | None = None,
):
    s = cad_cls(
        planes,
        face_planes,
        cs,
        world_size,
        max_string_length,
        use_literals,
    ).to_string()

    operation_world_size = (
        world_size if world_size_continue is None else world_size_continue
    )
    fillet_chamfer = factory["factory"].generate(
        s,
        world_size=operation_world_size,
    )

    if isinstance(fillet_chamfer, Fillet):
        cs.append(dict(type="Fillet", op=fillet_chamfer))
    else:
        cs.append(dict(type="Chamfer", op=fillet_chamfer))
    logger.info(
        "Added edge operation: %s on point %s",
        cs[-1]["type"],
        fillet_chamfer.point,
    )
