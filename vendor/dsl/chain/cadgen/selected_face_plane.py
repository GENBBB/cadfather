import logging

from .base import BaseFactory, BaseOperation
from .registry import factories

logger = logging.getLogger(__name__)


class SelectedFacePlane(BaseOperation):
    def __init__(self, top_p: float):
        self.top_p = top_p
        self.point_on_face: tuple[float, float, float] | None = None
        self.face_plane_name: str | None = None
        self.face_name: str | None = None
        self.point_name: str | None = None
        self.planar: bool | None = None

    def to_string(self) -> str:
        assert self.point_on_face is not None
        assert self.face_plane_name is not None
        face_str = f"{self.point_name} = [{self.point_on_face[0]}, {self.point_on_face[1]}, {self.point_on_face[2]}]\n"
        face_str += (
            f"{self.face_name} = r.faces(PointOnFaceSelector({self.point_name}))\n"
        )
        if self.planar:
            plane = f"{face_str}{self.face_plane_name}={self.face_name}.workplane(origin={self.face_name}.val().Center())\n"
        else:
            plane = face_str
        return plane

    def transform(self, shift: list[float], scale: float) -> None:
        assert self.point_on_face is not None
        self.point_on_face = (
            (self.point_on_face[0] + shift[0]) * scale,
            (self.point_on_face[1] + shift[1]) * scale,
            (self.point_on_face[2] + shift[2]) * scale,
        )

    def round(self) -> None:
        assert self.point_on_face is not None
        self.point_on_face = (
            round(self.point_on_face[0]),
            round(self.point_on_face[1]),
            round(self.point_on_face[2]),
        )


class SelectedFacePlaneFactory(BaseFactory):
    def __init__(self, top_p: float):
        self.top_p = top_p

    def generate(self) -> SelectedFacePlane:
        return SelectedFacePlane(self.top_p)


factories.register("selected_face_plane", SelectedFacePlaneFactory)
