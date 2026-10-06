from __future__ import annotations
import logging

import numpy as np

from .base import BaseFactory, BaseOperation
from .registry import factories
from .sselectors import *
from .sketch import Sketch, SketchFactory

logger = logging.getLogger(__name__)


class Extrude(BaseOperation):
    def __init__(self, sketch: Sketch, extent: float):
        self.sketch = sketch
        self.extent = extent

    def to_string(
        self,
        plane: str,
    ) -> str:
        logger.info("Building extrude string with extent=%f", self.extent)
        expr = self.sketch.to_string(plane)
        expr += f".extrude({self.extent})\n"
        return expr

    def transform(self, shift: list[float], scale: float) -> None:
        self.extent *= scale

    def round(self) -> None:
        self.sketch.round()
        self.extent = round(self.extent)


class ExtrudeFactory(BaseFactory):
    def __init__(self, sketch_factory: SketchFactory | dict):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)
        self.sketch_factory = sketch_factory
        self.r_min, self.r_max = 0.01, 1.0

    def generate(self) -> Extrude:
        sketch = self.sketch_factory.generate()
        sign = float(np.random.choice([-1, 1]))
        extent = sign * np.random.uniform(self.r_min, self.r_max)

        logger.info("Generated extrude with extent=%f", extent)
        return Extrude(sketch, extent)


factories.register("extrude", ExtrudeFactory)
