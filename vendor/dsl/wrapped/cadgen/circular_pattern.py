import logging

import cadquery as cq
import numpy as np
from OCP.BRepTools import BRepTools

from .base import BaseFactory, BaseOperation
from .registry import factories
from .sketch import Sketch, SketchFactory
from .utils import (
    compute_length_of_u_isocurve,
    get_point_on_face,
    normal_distance_to_next_face,
)

logger = logging.getLogger(__name__)


class CircularPattern(BaseOperation):
    def __init__(
        self,
        sketch: Sketch,
        point: tuple[float, float, float],
        span: float = 360,
        n: int = 1,
        combine: str = "s",
        depth: float = 200,
    ):
        self.sketch = sketch
        self.point = point
        self.span = span
        self.n = n
        self.combine = combine
        self.depth = depth

    def to_string(self) -> str:
        sketch_str = self.sketch.to_string_one_sketch("", self.sketch.wires)
        expr = f"r=r.circular_pattern([{self.point[0]}, {self.point[1]}, {self.point[2]}], {self.span}, {self.n}, '{sketch_str}.finalize().extrude({self.depth}, both=False)', combine='{self.combine}')\n"
        return expr

    def transform(
        self,
        shift: list[float],
        scale: float,
    ) -> None:
        self.point = (
            (self.point[0] + shift[0]) * scale,
            (self.point[1] + shift[1]) * scale,
            (self.point[2] + shift[2]) * scale,
        )
        self.sketch.transform([0, 0], scale)
        self.depth *= scale

    def round(self) -> None:
        self.point = (
            round(self.point[0]),
            round(self.point[1]),
            round(self.point[2]),
        )
        self.sketch.round()
        self.depth = round(self.depth)

    def fix(self) -> None:
        self.sketch.fix()

    def to_dict(self) -> dict:
        return {
            "type": "CircularPattern",
            "sketch": self.sketch.to_dict(),
            "point": self.point,
            "span": self.span,
            "n": self.n,
            "combine": self.combine,
            "depth": self.depth,
        }

    @staticmethod
    def from_dict(entity: dict) -> "CircularPattern":
        assert (
            entity["type"] == "CircularPattern"
        ), f"Trying to build CircularPattern from type {entity['type']}"
        return CircularPattern(
            Sketch.from_dict(entity["sketch"]),
            entity["point"],
            entity["span"],
            entity["n"],
            entity["combine"],
            entity["depth"],
        )


class CircularPatternFactory(BaseFactory):
    def __init__(
        self,
        sketch_factory: SketchFactory | dict,
        full_circular_pattern_probability: float | None = None,
        cut_probability: float = 0.5,
    ):
        if isinstance(sketch_factory, dict):
            sketch_factory = SketchFactory(**sketch_factory)
        self.sketch_factory = sketch_factory
        self.full_circular_pattern_probability = full_circular_pattern_probability
        self.cut_probability = cut_probability

    def generate(self, s: str | None = None) -> CircularPattern | None:
        if s is None:
            return None
        combine = "s" if np.random.random() < self.cut_probability else "a"
        sketch = self.sketch_factory.generate_one_sketch(
            self.sketch_factory.min_n_commands,
            self.sketch_factory.max_n_commands,
            zero_center=True,
        )
        sketch.wires = [wire for wire in sketch.wires if wire["outer"]]
        # if combine == "s":
        #     # leave only outer wires
        #     outer_wire = None
        #     for wire in sketch.wires:
        #         if wire["outer"]:
        #             outer_wire = wire
        #             break
        #     sketch.wires = [outer_wire]

        exec(s, globals())
        w = globals()["r"]
        eligible_faces = [
            face for face in w.faces().vals() if face.geomType() in ["CYLINDER", "CONE"]
        ]
        face = np.random.choice(eligible_faces)
        point = get_point_on_face(face)
        circle_length = compute_length_of_u_isocurve(face, point)
        radius = circle_length / (2 * np.pi)
        n = np.random.randint(2, 40)

        assert self.full_circular_pattern_probability is not None
        span = (
            360
            if np.random.random() < self.full_circular_pattern_probability
            else np.random.randint(10, 355)
        )

        k = span / 360
        possible_tseconds = 2 * radius * np.sin(np.pi * k / np.arange(2, n + 1))

        tmp = possible_tseconds
        # creating more space so it is not narrowly stacked
        tsecond = tmp[-2] if len(tmp) > 2 else tmp[-1] if len(tmp) > 1 else tmp[-1]
        n = len(tmp)

        bbox = cq.Face(sketch.to_shape()).BoundingBox()
        xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        _, _, vmin, vmax = BRepTools.UVBounds_s(face.wrapped)
        scale_2 = (
            (vmax - vmin) / max(xmax - xmin, ymax - ymin) * np.random.uniform(1, 1.3)
        )
        sketch_rel_scale = min(tsecond / max(xmax - xmin, ymax - ymin), scale_2)

        sketch.transform([0, 0], sketch_rel_scale)

        depth = normal_distance_to_next_face(w, face, point)
        depth *= 1.25

        logger.info(
            "Generated circular pattern: angle=%s, n=%s",
            span,
            n,
        )
        return CircularPattern(sketch, point, span, n, combine, depth)

    @staticmethod
    def from_dict(entity: dict) -> "CircularPatternFactory":
        return CircularPatternFactory(
            SketchFactory.from_dict(entity["sketch_factory"]),
            entity["full_circular_pattern_probability"],
            entity["cut_probability"],
        )

    def to_dict(self) -> dict:
        return {
            "type": "CircularPatternFactory",
            "sketch_factory": self.sketch_factory.to_dict(),
            "full_circular_pattern_probability": self.full_circular_pattern_probability,
            "cut_probability": self.cut_probability,
        }


factories.register("circular_pattern", CircularPatternFactory)
