import logging

import cadquery as cq
import numpy as np

from .base import BaseFactory, BaseOperation
from .extrude import _surface_translation
from .registry import factories
from .surface_sampling import (
    closest_surface_point_from_compound,
    precompute_surface_compound,
    shape_from_cad_object,
)
from .utils import float_to_string

logger = logging.getLogger(__name__)


def _loft_profile(wp, profile: dict):
    """Draw a section profile on workplane ``wp``."""
    t = profile["type"]
    if t == "circle":
        return wp.circle(profile["r"])
    if t == "rect":
        return wp.rect(profile["w"], profile["h"])
    if t == "polygon":
        return wp.polygon(int(profile["n"]), profile["d"])
    if t == "ellipse":
        return wp.ellipse(profile["a"], profile["b"])
    raise ValueError(f"unknown loft profile type {t!r}")


def loft(
    r,
    point: tuple[float, float, float],
    workplane: str,
    sections: list[dict],
    ruled: bool = False,
):
    """Runtime DSL helper: loft through ``sections`` and union onto ``r``.

    Self-contained like extrude/revolve: rebuilds the base workplane from the
    named ``workplane`` axis and ``point`` (no external ``w{i}`` variable), then
    stacks each section's profile at its ``offset`` along the axis, with optional
    per-section ``center`` (eccentricity) and ``rotation`` (twist, deg). Profiles
    cover circle / rect / regular polygon / ellipse, so the continuous space
    spans reducers, round<->rectangular duct transitions, twisted prisms, and
    multi-section vase/funnel shapes. ``ruled`` = straight (ruled) vs smooth loft.
    """
    wp = cq.Workplane(workplane, origin=tuple(point))
    for idx, section in enumerate(sections):
        if idx > 0:
            wp = wp.workplane(offset=section["offset"])
        rotation = section.get("rotation", 0.0)
        if abs(rotation) > 1e-9:
            wp = wp.transformed(rotate=(0.0, 0.0, rotation))
        center = section.get("center", [0.0, 0.0])
        if abs(center[0]) > 1e-9 or abs(center[1]) > 1e-9:
            wp = wp.center(center[0], center[1])
        wp = _loft_profile(wp, section["profile"])
    body = wp.loft(ruled=ruled)

    if r is None:
        return body

    # Combined with a parent: nudge the loft toward the parent surface (closest
    # point to the loft base) so the boss boolean-unions cleanly, mirroring how
    # sweep attaches. Guarded -- on failure just union in place.
    try:
        shape = shape_from_cad_object(r)
        surface_compound = precompute_surface_compound(shape)
        surface_point = closest_surface_point_from_compound(
            surface_compound, tuple(point)
        )
        body = body.translate(_surface_translation(shape, surface_point, 200))
    except Exception:
        logger.debug("loft surface nudge failed; union in place", exc_info=True)

    if isinstance(r, cq.Shape):
        return cq.Workplane().add(r).union(body)
    return r.union(body)


class Loft(BaseOperation):
    def __init__(self, sections: list[dict], ruled: bool = False):
        if len(sections) < 2:
            raise ValueError("Loft requires at least two sections")
        self.sections = sections
        self.ruled = ruled

    def _apply_profile(self, expr: str, profile: dict) -> str:
        t = profile["type"]
        if t == "circle":
            return f"{expr}.circle({float_to_string(profile['r'])})"
        if t == "rect":
            return (
                f"{expr}.rect({float_to_string(profile['w'])}, "
                f"{float_to_string(profile['h'])})"
            )
        if t == "polygon":
            return f"{expr}.polygon({int(profile['n'])}, {float_to_string(profile['d'])})"
        if t == "ellipse":
            return (
                f"{expr}.ellipse({float_to_string(profile['a'])}, "
                f"{float_to_string(profile['b'])})"
            )
        raise ValueError(f"unknown loft profile type {t!r}")

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
            rotation = section.get("rotation", 0.0)
            if abs(rotation) > 1e-9:
                expr = f"{expr}.transformed(rotate=(0, 0, {float_to_string(rotation)}))"
            expr = self._apply_center(expr, section["center"])
            expr = self._apply_profile(expr, section["profile"])
        ruled_arg = "ruled=True" if self.ruled else ""
        expr = f"{expr}.loft({ruled_arg})\n"
        return expr

    def to_call_string(
        self, point: tuple[float, float, float], workplane_axis: str
    ) -> str:
        """Emit a single-line ``loft(r, <point>, '<axis>', [<sections>])`` call.

        Self-contained like extrude/revolve: only the ``r`` variable is used and
        the base workplane is rebuilt inside ``loft`` from ``point`` + the named
        axis (no external ``w{i}`` variable).
        """
        fts = float_to_string

        def _profile_str(profile: dict) -> str:
            t = profile["type"]
            if t == "circle":
                return f"{{'type': 'circle', 'r': {fts(profile['r'])}}}"
            if t == "rect":
                return (
                    f"{{'type': 'rect', 'w': {fts(profile['w'])}, "
                    f"'h': {fts(profile['h'])}}}"
                )
            if t == "polygon":
                return (
                    f"{{'type': 'polygon', 'n': {int(profile['n'])}, "
                    f"'d': {fts(profile['d'])}}}"
                )
            return (
                f"{{'type': 'ellipse', 'a': {fts(profile['a'])}, "
                f"'b': {fts(profile['b'])}}}"
            )

        section_strs = []
        for section in self.sections:
            center = section["center"]
            center_str = f"[{fts(center[0])}, {fts(center[1])}]"
            offset = section.get("offset", 0.0)
            rotation = section.get("rotation", 0.0)
            section_strs.append(
                f"{{'offset': {fts(offset)}, 'center': {center_str}, "
                f"'rotation': {fts(rotation)}, "
                f"'profile': {_profile_str(section['profile'])}}}"
            )
        sections_str = "[" + ", ".join(section_strs) + "]"
        point_str = f"({fts(point[0])}, {fts(point[1])}, {fts(point[2])})"
        ruled_arg = ", ruled=True" if self.ruled else ""
        return (
            f"loft(r, {point_str}, {workplane_axis!r}, {sections_str}{ruled_arg})\n"
        )

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
            t = profile["type"]
            if t == "circle":
                profile["r"] *= scale
            elif t == "rect":
                profile["w"] *= scale
                profile["h"] *= scale
            elif t == "polygon":
                profile["d"] *= scale  # n (sides) is scale-free; rotation invariant
            elif t == "ellipse":
                profile["a"] *= scale
                profile["b"] *= scale

    def round(self) -> None:
        for idx, section in enumerate(self.sections):
            section["center"] = [
                round(section["center"][0]),
                round(section["center"][1]),
            ]
            if idx > 0:
                section["offset"] = round(section["offset"])
            if "rotation" in section:
                section["rotation"] = round(section["rotation"])
            profile = section["profile"]
            t = profile["type"]
            if t == "circle":
                profile["r"] = round(profile["r"])
            elif t == "rect":
                profile["w"] = round(profile["w"])
                profile["h"] = round(profile["h"])
            elif t == "polygon":
                profile["d"] = round(profile["d"])
            elif t == "ellipse":
                profile["a"] = round(profile["a"])
                profile["b"] = round(profile["b"])

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {"type": "Loft", "sections": self.sections, "ruled": self.ruled}

    @staticmethod
    def from_dict(entity: dict) -> "Loft":
        assert (
            entity["type"] == "Loft"
        ), f"Trying to build Loft from type {entity['type']}"
        return Loft(sections=entity["sections"], ruled=entity.get("ruled", False))


class LoftFactory(BaseFactory):
    def __init__(
        self,
        *,
        n_sections_range: tuple[int, int] = (2, 5),
        offset_range: tuple[float, float] = (8.0, 30.0),
        profile_circle_radius_range: tuple[float, float] = (5.0, 20.0),
        profile_rect_size_range: tuple[float, float] = (8.0, 28.0),
        center_shift_range: tuple[float, float] = (-9.0, 9.0),
        twist_probability: float = 0.35,
        ruled_probability: float = 0.2,
        eccentric_probability: float = 0.5,
    ):
        self.n_sections_range = n_sections_range
        self.offset_range = offset_range
        self.profile_circle_radius_range = profile_circle_radius_range
        self.profile_rect_size_range = profile_rect_size_range
        self.center_shift_range = center_shift_range
        self.twist_probability = twist_probability
        self.ruled_probability = ruled_probability
        self.eccentric_probability = eccentric_probability

    def _random_profile(self) -> dict:
        roll = np.random.rand()
        if roll < 0.34:
            return {
                "type": "circle",
                "r": float(np.random.uniform(*self.profile_circle_radius_range)),
            }
        if roll < 0.58:
            w = float(np.random.uniform(*self.profile_rect_size_range))
            h = float(np.random.uniform(*self.profile_rect_size_range))
            return {"type": "rect", "w": w, "h": h}
        if roll < 0.82:
            return {
                "type": "polygon",
                "n": int(np.random.randint(3, 9)),
                "d": float(np.random.uniform(*self.profile_rect_size_range)),
            }
        a = float(np.random.uniform(*self.profile_circle_radius_range))
        b = float(np.random.uniform(*self.profile_circle_radius_range))
        return {"type": "ellipse", "a": a, "b": b}

    def _random_center(self) -> list[float]:
        if np.random.rand() >= self.eccentric_probability:
            return [0.0, 0.0]
        return [
            float(np.random.uniform(*self.center_shift_range)),
            float(np.random.uniform(*self.center_shift_range)),
        ]

    def _n_sections(self) -> int:
        # Mostly 2-section transitions; a tail of 3-5 for vase/funnel shapes.
        lo, hi = self.n_sections_range
        weights = np.array([0.5, 0.28, 0.14, 0.08][: hi - lo + 1], dtype=float)
        weights = weights / weights.sum()
        return int(np.random.choice(range(lo, hi + 1), p=weights))

    def generate(self) -> Loft:
        n_sections = self._n_sections()
        ruled = np.random.rand() < self.ruled_probability

        # TWIST mode: one shape repeated, rotated incrementally -> twisted prism.
        # Otherwise: varied shapes per section (reducers / round<->rect / vases).
        twist_mode = np.random.rand() < self.twist_probability
        if twist_mode:
            base = self._random_profile()
            if base["type"] == "circle":  # circles don't show twist
                base = {
                    "type": "polygon",
                    "n": int(np.random.randint(3, 8)),
                    "d": float(np.random.uniform(*self.profile_rect_size_range)),
                }
            n_sides = base.get("n", 4 if base["type"] == "rect" else 4)
            twist_per = float(np.random.uniform(15.0, min(70.0, 180.0 / n_sides)))
            taper = float(np.random.uniform(0.6, 1.0))

        sections: list[dict] = []
        for idx in range(n_sections):
            offset = 0.0 if idx == 0 else float(np.random.uniform(*self.offset_range))
            if twist_mode:
                profile = dict(base)
                s = taper ** idx
                for key in ("r", "w", "h", "d", "a", "b"):
                    if key in profile:
                        profile[key] = profile[key] * s
                rotation = idx * twist_per
                center = [0.0, 0.0]
            else:
                profile = self._random_profile()
                rotation = (
                    float(np.random.uniform(10.0, 80.0))
                    if (idx > 0 and profile["type"] != "circle"
                        and np.random.rand() < 0.35)
                    else 0.0
                )
                center = self._random_center()
            sections.append(
                {"offset": offset, "profile": profile, "center": center,
                 "rotation": rotation}
            )
        logger.info("Generated loft with %d sections", len(sections))
        return Loft(sections, ruled=ruled)

    def to_dict(self) -> dict:
        return {
            "type": "LoftFactory",
            "n_sections_range": self.n_sections_range,
            "offset_range": self.offset_range,
            "profile_circle_radius_range": self.profile_circle_radius_range,
            "profile_rect_size_range": self.profile_rect_size_range,
            "center_shift_range": self.center_shift_range,
            "twist_probability": self.twist_probability,
            "ruled_probability": self.ruled_probability,
            "eccentric_probability": self.eccentric_probability,
        }

    @staticmethod
    def from_dict(entity: dict) -> "LoftFactory":
        return LoftFactory(
            n_sections_range=entity["n_sections_range"],
            offset_range=entity["offset_range"],
            profile_circle_radius_range=entity["profile_circle_radius_range"],
            profile_rect_size_range=entity["profile_rect_size_range"],
            center_shift_range=entity["center_shift_range"],
            twist_probability=entity.get("twist_probability", 0.35),
            ruled_probability=entity.get("ruled_probability", 0.2),
            eccentric_probability=entity.get("eccentric_probability", 0.5),
        )


factories.register("loft", LoftFactory)
