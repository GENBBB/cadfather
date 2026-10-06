from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadquery import Workplane

from .base import BaseFactory, BaseOperation
from .registry import factories
from .sselectors import *
from .sketch import Sketch, SketchFactory
from .utils import (
    get_faces_on_plane,
    is_face_outer,
    put_sketch_on_face,
    put_sketch_on_face_simplified,
)

logger = logging.getLogger(__name__)


class CutThruAll(BaseOperation):
    def __init__(self, sketch: Sketch, plane: str, world_size: float):
        self.sketch = sketch
        self.plane = plane
        self.world_size = world_size

    def to_string(self) -> str:
        logger.info("Building cutTruAll string")
        # if self.direction == (1, 0, 0):
        #     workplane = ".copyWorkplane(cq.Workplane('ZY'))"
        # elif self.direction == (0, 1, 0):
        #     workplane = ".copyWorkplane(cq.Workplane('ZX'))"
        # else:
        #     workplane = ".copyWorkplane(cq.Workplane('XY'))"
        if self.plane.startswith("point"):
            sketch_str = self.sketch.to_string_one_sketch(
                "", self.sketch.wires, skip_on_one_wire=True
            )
            expr = f"r=r.attach_at([{self.plane}], '{sketch_str}.finalize().extrude({self.world_size}, both=True)', combine='s')\n"
        else:
            workplane = f"r=r.copyWorkplane({self.plane})"

            expr = self.sketch.to_string_one_sketch(
                workplane, self.sketch.wires, skip_on_one_wire=True
            )
            expr += f".finalize().extrude({self.world_size}, combine='s', both=True)\n"  # .cutThruAll()
        return expr

    def transform(self, shift: list[float], scale: float) -> None:
        # if self.direction == (1, 0, 0):
        #     sketch_shift = [shift[2], shift[1]]
        # elif self.direction == (0, 1, 0):
        #     sketch_shift = [shift[2], shift[0]]
        # else:
        #     sketch_shift = [shift[0], shift[1]]
        # self.sketch.transform(sketch_shift, scale)
        self.sketch.transform([0, 0], scale)

    def round(self) -> None:
        self.sketch.round()


class CutThruAllFactory(BaseFactory):
    def __init__(self, sketch_factory: SketchFactory | dict):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)

        self.sketch_factory = sketch_factory

    def generate(
        self, s: str | None, plane: str | None = None, world_size: float | None = None
    ) -> CutThruAll | None:
        # possible_directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        # direction = possible_directions[np.random.randint(2)]
        if s is None:
            return

        assert s is not None

        sketch = self.sketch_factory.generate_one_sketch(
            self.sketch_factory.min_n_commands,
            self.sketch_factory.max_n_commands,
            zero_center=True,
        )
        sketch.wires = [wire for wire in sketch.wires if wire['outer']]

        exec(s, globals())
        _w = globals()["r"]

        assert plane is not None
        face_plane: Workplane = eval(plane)

        if plane.startswith("point"):
            point_on_face = face_plane
            face_sel_name = f"face_sel{plane.split('_')[-1].lstrip('sel')}"
            face = eval(face_sel_name)
            put_sketch_on_face_simplified(sketch, face.val(), point_on_face)
        else:
            faces_on_plane = get_faces_on_plane(_w, face_plane)
            faces_on_plane = [f for f in faces_on_plane if is_face_outer(_w, f)]
            faces_on_plane.sort(key=lambda x: x.Area())
            face = faces_on_plane[-1].wrapped

            put_sketch_on_face(sketch, face, face_plane)

        logger.info("Generated cutThruAll")
        assert world_size is not None
        return CutThruAll(sketch, plane, world_size)


factories.register("cut_thru_all", CutThruAllFactory)
