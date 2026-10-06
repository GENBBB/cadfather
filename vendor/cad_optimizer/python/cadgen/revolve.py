from __future__ import annotations
import logging

import numpy as np

from .base import BaseFactory, BaseOperation
from .registry import factories
from .sketch import Sketch, SketchFactory
from .sselectors import *

logger = logging.getLogger(__name__)


class Revolve(BaseOperation):
    def __init__(
        self,
        sketch: Sketch,
        angle_degrees: float = 360,
        revolve_axis: int = 0,
        dist_to_axis: float | None = None,
    ):
        self.sketch = sketch
        self.angle_degrees = angle_degrees
        self.revolve_axis = revolve_axis
        self.dist_to_axis = dist_to_axis

    def to_string(self, plane: str) -> str:
        logger.info("Building revolve string with dist_to_axis=%f", self.dist_to_axis)
        expr = self.sketch.to_string(plane)
        assert self.dist_to_axis is not None

        cq_import = (
            "import cadquery as cq\nw0=cq.Workplane('XY')\nw1=cq.Workplane('XY')\nr="
        )
        exec(cq_import + expr, globals())
        w = globals()["r"]
        bbox = self.sketch.cq_bounding_box(w)
        x1 = float(int(bbox[2])) + self.dist_to_axis
        y1 = float(int(bbox[3])) + self.dist_to_axis

        if self.revolve_axis == 0:
            x2 = x1 + 1
            y2 = y1
        else:
            x2 = x1
            y2 = y1 + 1

        if np.isclose(x1, round(x1)):
            x1 = int(x1)
        if np.isclose(y1, round(y1)):
            y1 = int(y1)
        if np.isclose(x2, round(x2)):
            x2 = int(x2)
        if np.isclose(y2, round(y2)):
            y2 = int(y2)

        expr += f".revolve({self.angle_degrees}, {(x1, y1)}, {(x2, y2)})\n"

        return expr

    def transform(self, shift: list[float], scale: float) -> None:
        assert self.dist_to_axis is not None
        self.dist_to_axis *= scale

    def round(self) -> None:
        assert self.dist_to_axis is not None
        self.sketch.round()
        self.dist_to_axis = round(self.dist_to_axis)


class RevolveFactory(BaseFactory):
    def __init__(
        self,
        sketch_factory: SketchFactory | dict,
        full_revolve_probability: float | None = None,
    ):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)
        self.sketch_factory = sketch_factory
        self.full_revolve_probability = full_revolve_probability
        self.r_min, self.r_max = 0.01, 1.0

    def generate(self) -> Revolve:
        sketch = self.sketch_factory.generate()
        # leave only outer wires
        outer_wire = None
        for wire in sketch.wires:
            if wire["outer"]:
                outer_wire = wire
                break
        sketch.wires = [outer_wire]

        revolve_axis = np.random.random() < 0.5
        assert self.full_revolve_probability is not None
        angle_degrees = (
            360
            if np.random.random() < self.full_revolve_probability
            else np.random.randint(10, 355)
        )
        dist_to_axis = np.random.uniform(self.r_min, self.r_max)

        logger.info("Generated extrude with dist_to_axis=%f", dist_to_axis)
        return Revolve(sketch, angle_degrees, revolve_axis, dist_to_axis)


factories.register("revolve", RevolveFactory)
