from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np

if TYPE_CHECKING:
    from cadquery import Workplane

from .base import BaseFactory, BaseOperation
from .registry import factories
from .sselectors import *
from .utils import (
    get_faces_on_plane,
    get_point_on_sketch_and_radius_range,
    is_face_outer,
    normal_distance_to_next_face,
)

logger = logging.getLogger(__name__)


class Hole(BaseOperation):
    def __init__(
        self,
        radius: float | None = None,
        cbore_radius: float | None = None,
        csk_radius: float | None = None,
        depth: float | None = None,
        cbore_depth: float | None = None,
        csk_angle: int | None = None,
        point: list[float] | None = None,
    ):
        self.radius = radius
        self.cbore_radius = cbore_radius
        self.csk_radius = csk_radius
        self.depth = depth
        self.cbore_depth = cbore_depth
        self.csk_angle = csk_angle
        self.point = point

    def to_string(self, plane) -> str:
        logger.info(
            "Building hole string with radius=%f, cbore_radius=%f, depth=%f, cbore_depth=%f, csk_angle=%d",
            self.radius,
            self.cbore_radius,
            self.depth,
            self.cbore_depth,
            self.csk_angle,
        )
        expr = "r=r"

        assert self.radius is not None, "radius is required"
        assert self.depth is not None, "depth is required"

        center = self.point if self.point else (0, 0)

        tag = "r.workplane(origin=r.val().Center())" if "faces" in plane else plane
        push_points = f".pushPoints([{center}])" if center != (0, 0) else ""

        expr += f".copyWorkplane({tag}){push_points}"
        if self.cbore_radius is not None:
            assert self.cbore_depth is not None
            expr += f".cboreHole({2*self.radius}, {2*self.cbore_radius}, {self.cbore_depth}, {self.depth})\n"
        elif self.csk_radius is not None:
            assert self.csk_angle is not None
            expr += f".cskHole({2*self.radius}, {2*self.csk_radius}, {self.csk_angle}, {self.depth})\n"
        else:
            assert self.depth is not None
            expr += f".hole({2*self.radius}, {self.depth})\n"

        return expr

    def transform(self, shift: list[float], scale: float) -> None:
        if self.radius:
            self.radius *= scale
        if self.cbore_radius:
            self.cbore_radius *= scale
        if self.csk_radius:
            self.csk_radius *= scale
        if self.depth:
            self.depth *= scale
        if self.cbore_depth:
            self.cbore_depth *= scale

        assert self.point is not None

        self.point = (self.point[0] * scale, self.point[1] * scale)

    def round(self) -> None:
        assert self.point is not None

        if self.radius:
            self.radius = round(self.radius)
            if np.allclose(self.radius, 0):
                self.radius = 1
        if self.cbore_radius:
            self.cbore_radius = round(self.cbore_radius)
            if np.allclose(self.cbore_radius, 0):
                self.cbore_radius = 1
        if self.csk_radius:
            self.csk_radius = round(self.csk_radius)
            if np.allclose(self.csk_radius, 0):
                self.csk_radius = 1
        if self.depth:
            self.depth = round(self.depth)
            if np.allclose(self.depth, 0):
                self.depth = 1
        if self.cbore_depth:
            self.cbore_depth = round(self.cbore_depth)
            if np.allclose(self.cbore_depth, 0):
                self.cbore_depth = 1

        self.point = (round(self.point[0]), round(self.point[1]))


class HoleFactory(BaseFactory):
    def __init__(
        self,
        hole_type_probabilities: dict[str, float],
    ):
        self.hole_type_probabilities = hole_type_probabilities

    def generate(
        self,
        s: str | None = None,
        plane: str | None = None,
    ) -> Hole | None:
        if s is None:
            return None

        exec(s, globals())
        _w = globals()["r"]

        assert plane is not None
        face_plane: Workplane = eval(plane, globals())

        faces_on_plane = get_faces_on_plane(_w, face_plane)
        faces_on_plane = [f for f in faces_on_plane if is_face_outer(_w, f)]
        faces_on_plane.sort(key=lambda x: x.Area())
        face = faces_on_plane[-1].wrapped
        extent = normal_distance_to_next_face(_w, cq.Face(face))

        point, min_extent, max_extent = get_point_on_sketch_and_radius_range(
            face, p_center=0.0
        )

        origin = np.array(
            [
                face_plane.plane.origin.x,
                face_plane.plane.origin.y,
                face_plane.plane.origin.z,
            ]
        )
        x_axis = np.array(
            [face_plane.plane.xDir.x, face_plane.plane.xDir.y, face_plane.plane.xDir.z]
        )
        y_axis = np.array(
            [face_plane.plane.yDir.x, face_plane.plane.yDir.y, face_plane.plane.yDir.z]
        )
        point_local = np.array(point) - origin
        point = [point_local.dot(x_axis), point_local.dot(y_axis)]

        hole_type = np.random.choice(
            list(self.hole_type_probabilities.keys()),
            p=list(self.hole_type_probabilities.values()),
        )

        radius = None
        cbore_radius = None
        csk_radius = None
        depth = None
        cbore_depth = None
        csk_angle = None

        if hole_type == "default":
            radius = np.random.uniform(
                max(0.5 * max_extent, min_extent), 0.95 * max_extent
            )
            depth = np.random.uniform(0, extent)
        elif hole_type == "cbore":
            radius = np.random.uniform(
                max(0.5 * max_extent, min_extent), 0.95 * max_extent
            )
            cbore_radius = np.random.uniform(radius, 0.95 * max_extent)
            depth = np.random.uniform(0, extent)
            cbore_depth = np.random.uniform(0, depth)
        elif hole_type == "csk":
            radius = np.random.uniform(
                max(0.5 * max_extent, min_extent), 0.95 * max_extent
            )
            csk_radius = np.random.uniform(radius, 0.95 * max_extent)
            depth = np.random.uniform(0, extent)
            csk_angle = np.random.randint(30, 150)

        logger.info(
            "Generated hole with radius=%f, cbore_radius=%f, depth=%f, cbore_depth=%f, csk_angle=%d",
            radius,
            cbore_radius,
            depth,
            cbore_depth,
            csk_angle,
        )
        return Hole(
            radius,
            cbore_radius,
            csk_radius,
            depth,
            cbore_depth,
            csk_angle,
            point[:2],
        )


factories.register("hole", HoleFactory)
