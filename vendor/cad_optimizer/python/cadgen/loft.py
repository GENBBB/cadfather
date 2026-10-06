from __future__ import annotations
import logging

import numpy as np

from .base import BaseFactory, BaseOperation
from .registry import factories
from .utils import float_to_string

logger = logging.getLogger(__name__)


class Loft(BaseOperation):
    def __init__(self, sections: list[dict]):
        if len(sections) < 2:
            raise ValueError("Loft requires at least two sections")
        self.sections = sections

    def _apply_profile(self, expr: str, profile: dict) -> str:
        if profile["type"] == "circle":
            radius = float_to_string(profile["r"])
            return f"{expr}.circle({radius})"
        width = float_to_string(profile["w"])
        height = float_to_string(profile["h"])
        return f"{expr}.rect({width}, {height})"

    def _apply_center(
        self,
        expr: str,
        center: list[float],
    ) -> str:
        eps = 1e-9
        if abs(center[0]) < eps and abs(center[1]) < eps:
            return expr
        cx = float_to_string(center[0])
        cy = float_to_string(center[1])
        return f"{expr}.center({cx}, {cy})"

    def to_string(
        self,
        plane: str = "cq.Workplane('XY')",
        axis: int | None = None,
        use_literals: bool = True,
    ) -> str:
        logger.info(
            "Building loft string for %d sections on axis %s",
            len(self.sections),
            axis,
        )
        expr = plane
        for idx, section in enumerate(self.sections):
            if idx > 0:
                offset = float_to_string(section["offset"])
                expr = f"{expr}.workplane(offset={offset})"
            expr = self._apply_center(expr, section["center"])
            expr = self._apply_profile(expr, section["profile"])
        expr = f"{expr}.loft()\n"
        return expr

    def transform(
        self, shift: list[float], scale: float, plane_axis: int | None = 2
    ) -> None:
        if plane_axis == 0:
            plane_shift = [shift[1], shift[2]]
        elif plane_axis == 1:
            plane_shift = [shift[2], shift[0]]
        else:
            plane_shift = [shift[0], shift[1]]

        for idx, section in enumerate(self.sections):
            section["center"] = [
                (section["center"][0] + plane_shift[0]) * scale,
                (section["center"][1] + plane_shift[1]) * scale,
            ]
            if idx > 0:
                section["offset"] *= scale
            profile = section["profile"]
            if profile["type"] == "circle":
                profile["r"] *= scale
            else:
                profile["w"] *= scale
                profile["h"] *= scale

    def round(self) -> None:
        for idx, section in enumerate(self.sections):
            section["center"] = [
                round(section["center"][0]),
                round(section["center"][1]),
            ]
            if idx > 0:
                section["offset"] = round(section["offset"])
            profile = section["profile"]
            if profile["type"] == "circle":
                profile["r"] = round(profile["r"])
            else:
                profile["w"] = round(profile["w"])
                profile["h"] = round(profile["h"])


class LoftFactory(BaseFactory):
    def __init__(
        self,
        *,
        n_sections_range: tuple[int, int] = (2, 4),
        offset_range: tuple[float, float] = (5.0, 30.0),
        profile_circle_radius_range: tuple[float, float] = (5.0, 20.0),
        profile_rect_size_range: tuple[float, float] = (5.0, 25.0),
        center_shift_range: tuple[float, float] = (-10.0, 10.0),
    ):
        self.n_sections_range = n_sections_range
        self.offset_range = offset_range
        self.profile_circle_radius_range = profile_circle_radius_range
        self.profile_rect_size_range = profile_rect_size_range
        self.center_shift_range = center_shift_range

    def _random_profile(self) -> dict:
        if np.random.rand() < 0.5:
            return {
                "type": "circle",
                "r": float(np.random.uniform(*self.profile_circle_radius_range)),
            }
        width = float(np.random.uniform(*self.profile_rect_size_range))
        height = float(np.random.uniform(*self.profile_rect_size_range))
        return {"type": "rect", "w": width, "h": height}

    def _random_center(self) -> list[float]:
        return [
            float(np.random.uniform(*self.center_shift_range)),
            float(np.random.uniform(*self.center_shift_range)),
        ]

    def generate(self) -> Loft:
        n_sections = int(
            np.random.randint(self.n_sections_range[0], self.n_sections_range[1] + 1)
        )
        sections: list[dict] = []
        for idx in range(n_sections):
            section = {
                "offset": (
                    0.0 if idx == 0 else float(np.random.uniform(*self.offset_range))
                ),
                "profile": self._random_profile(),
                "center": self._random_center(),
            }
            sections.append(section)
        logger.info("Generated loft with %d sections", len(sections))
        return Loft(sections)


factories.register("loft", LoftFactory)
