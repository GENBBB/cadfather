import logging

from .base import BaseFactory, BaseOperation
from .registry import factories

logger = logging.getLogger(__name__)


class SelectedFacePlane(BaseOperation):
    def __init__(
        self,
        top_p: float,
        point_on_face: tuple[float, float, float] | None = None,
        face_plane_name: str | None = None,
        face_name: str | None = None,
        point_name: str | None = None,
        planar: bool | None = None,
    ):
        self.top_p = top_p
        self.point_on_face = point_on_face
        self.face_plane_name = face_plane_name
        self.face_name = face_name
        self.point_name = point_name
        self.planar = planar

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

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "SelectedFacePlane",
            "top_p": self.top_p,
            "point_on_face": self.point_on_face,
            "face_plane_name": self.face_plane_name,
            "face_name": self.face_name,
            "point_name": self.point_name,
            "planar": self.planar,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SelectedFacePlane":
        assert (
            entity["type"] == "SelectedFacePlane"
        ), f"Trying to build SelectedFacePlane from type {entity['type']}"
        return SelectedFacePlane(
            entity["top_p"],
            entity["point_on_face"],
            entity["face_plane_name"],
            entity["face_name"],
            entity["point_name"],
            entity["planar"],
        )


class SelectedFacePlaneFactory(BaseFactory):
    def __init__(self, top_p: float):
        self.top_p = top_p

    def generate(self) -> SelectedFacePlane:
        return SelectedFacePlane(self.top_p)

    def to_dict(self) -> dict:
        return {
            "type": "SelectedFacePlaneFactory",
            "top_p": self.top_p,
        }

    @staticmethod
    def from_dict(entity: dict) -> "SelectedFacePlaneFactory":
        return SelectedFacePlaneFactory(entity["top_p"])


factories.register("selected_face_plane", SelectedFacePlaneFactory)
