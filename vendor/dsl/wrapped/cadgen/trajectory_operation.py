from __future__ import annotations

import logging
import math

import cadquery as cq
import numpy as np
# from cadquery_addons import *

from .base import BaseFactory, BaseOperation
from .registry import factories, operations

logger = logging.getLogger(__name__)


def _unit3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return (v / n).astype(float)


def _random_unit_perp_to(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = _unit3(t)
    for _ in range(16):
        a = rng.standard_normal(3)
        b = np.cross(t, a)
        nb = float(np.linalg.norm(b))
        if nb >= 1e-9:
            return (b / nb).astype(float)
    o = np.array([1.0, 0.0, 0.0] if abs(t[0]) < 0.9 else [0.0, 1.0, 0.0])
    return _unit3(np.cross(t, o))


def _random_unit_not_parallel_to(
    ref: np.ndarray,
    rng: np.random.Generator,
    *,
    min_turn_deg: float = 12.0,
    max_turn_deg: float = 90.0,
) -> np.ndarray:
    """Random unit 3D axis that is not nearly collinear with ref (|dot| < cos(min_turn))."""
    ref = _unit3(ref)
    lim_min = math.cos(math.radians(min_turn_deg))
    lim_max = math.cos(math.radians(max_turn_deg))
    for _ in range(64):
        v = _unit3(rng.standard_normal(3))
        if float(np.dot(v, ref)) < lim_min and float(np.dot(v, ref)) > lim_max:
            return v
    # nearly parallel to ref: orthogonalize the random vector
    a = rng.standard_normal(3)
    v = _unit3(a - ref * float(np.dot(a, ref)))
    if float(np.linalg.norm(v)) < 1e-9:
        return _random_unit_perp_to(ref, rng)
    return v


def _rotate_about_axis(v: np.ndarray, k: np.ndarray, ang: float) -> np.ndarray:
    """Rotate v about the unit axis k by ang (axis-angle, Rodrigues)."""
    k = _unit3(k)
    ca, sa = math.cos(ang), math.sin(ang)
    return (v * ca + np.cross(k, v) * sa + k * float(np.dot(k, v)) * (1.0 - ca)).astype(
        float
    )


def random_arc_line_wire_3d(
    *,
    n_segments: int = 12,
    line_len_range: tuple[float, float] = (4.0, 14.0),
    arc_deg_range: tuple[float, float] = (18.0, 180.0),
    arc_radius_range: tuple[float, float] = (
        14.0 / (2 * np.pi * (180 / 360)),
        14.0 / (2 * np.pi * (180 / 360)),
    ),
    origin_jitter: float = 8.0,
    line_probability: float = 0.60,
    min_turn_between_lines_deg: float = 15.0,
    rng: np.random.Generator | None = None,
    p: tuple[float, float, float] | cq.Vector | None = None,
    t: tuple[float, float, float] | cq.Vector | None = None,
) -> tuple[cq.Workplane, cq.Wire]:
    """
    Random path in R^3: cq.Edge.makeLine + cq.Edge.makeThreePointArc, cq.Wire.assembleEdges.

    Arcs: G1 at the arc start (the middle point is chosen accordingly). After an arc the
    tangent t is updated by a rotation about w.

    Segments: two consecutive **LINE** edges are not collinear: before a new segment, if the
    previous edge is also a segment, the direction t is re-chosen (not nearly parallel to the
    previous one). After an arc the segment follows the current t (G1 on the arc side).
    """
    rng = rng or np.random.default_rng()

    if p is None:
        p = rng.uniform(-origin_jitter, origin_jitter, size=3).astype(float)
    elif isinstance(p, cq.Vector):
        p = np.array([p.x, p.y, p.z])
    else:
        p = np.array(p)
    # p = np.round(p)

    if t is None:
        t = _unit3(rng.standard_normal(3))
    elif isinstance(t, cq.Vector):
        t = _unit3(np.array([t.x, t.y, t.z]))
    else:
        t = _unit3(np.array(t))

    edge_dicts: list[dict] = []

    for _ in range(n_segments):
        if rng.random() < line_probability:
            if edge_dicts and edge_dicts[-1]["type"] == "line":
                t = _random_unit_not_parallel_to(
                    t, rng, min_turn_deg=min_turn_between_lines_deg
                )
            L = float(rng.uniform(*line_len_range))
            p_end = p + t * L
            # p_end = np.round(p_end)
            edge_dicts.append(
                {
                    "type": "line",
                    "start_point": p,
                    "end_point": p_end,
                }
            )
            p = p_end
        else:
            R = float(rng.uniform(*arc_radius_range))
            deg = float(rng.uniform(*arc_deg_range))
            if rng.random() < 0.5:
                deg = -deg
            rad = math.radians(deg)
            th = abs(rad)
            if th < 1e-4:
                if edge_dicts and edge_dicts[-1]["type"] == "line":
                    t = _random_unit_not_parallel_to(
                        t, rng, min_turn_deg=min_turn_between_lines_deg
                    )
                L = float(rng.uniform(*line_len_range))
                p_end = p + t * L
                # p_end = np.round(p_end)
                edge_dicts.append(
                    {
                        "type": "line",
                        "start_point": p,
                        "end_point": p_end,
                    }
                )
                p = p_end
                continue

            w = _random_unit_perp_to(t, rng)
            e2 = _unit3(np.cross(w, t))
            C = p + R * e2
            v0 = p - C
            p_end = C + _rotate_about_axis(v0, w, rad)
            p_mid = C + _rotate_about_axis(v0, w, 0.5 * rad)
            # p_end = np.round(p_end)
            # p_mid = np.round(p_mid)

            Cneg = p - R * e2
            v0neg = p - Cneg
            p_endneg = Cneg + _rotate_about_axis(v0neg, w, rad)
            p_midneg = Cneg + _rotate_about_axis(v0neg, w, 0.5 * rad)
            # p_endneg = np.round(p_endneg)
            # p_midneg = np.round(p_midneg)

            chord_half = float(np.linalg.norm(p_end - p) / 2.0)
            if chord_half >= abs(R) * 0.999:
                if edge_dicts and edge_dicts[-1]["type"] == "line":
                    t = _random_unit_not_parallel_to(
                        t, rng, min_turn_deg=min_turn_between_lines_deg
                    )
                L = float(rng.uniform(*line_len_range))
                p_end = p + t * L
                edge_dicts.append(
                    {
                        "type": "line",
                        "start_point": p,
                        "end_point": p_end,
                    }
                )
                p = p_end
                continue

            arc = cq.Edge.makeThreePointArc(
                cq.Vector(float(p[0]), float(p[1]), float(p[2])),
                cq.Vector(float(p_mid[0]), float(p_mid[1]), float(p_mid[2])),
                cq.Vector(float(p_end[0]), float(p_end[1]), float(p_end[2])),
            )

            arcneg = cq.Edge.makeThreePointArc(
                cq.Vector(float(p[0]), float(p[1]), float(p[2])),
                cq.Vector(float(p_midneg[0]), float(p_midneg[1]), float(p_midneg[2])),
                cq.Vector(float(p_endneg[0]), float(p_endneg[1]), float(p_endneg[2])),
            )

            if (
                float(
                    arc.tangentAt(0.0, mode="length").dot(
                        cq.Vector(float(t[0]), float(t[1]), float(t[2]))
                    )
                )
                > 0.99
            ):
                edge_dicts.append(
                    {
                        "type": "arc",
                        "start_point": p,
                        "mid_point": p_mid,
                        "end_point": p_end,
                    }
                )
                p = p_end.astype(float)
                t = _unit3(_rotate_about_axis(t, w, rad))
            elif (
                float(
                    arcneg.tangentAt(0.0, mode="length").dot(
                        cq.Vector(float(t[0]), float(t[1]), float(t[2]))
                    )
                )
                > 0.99
            ):
                edge_dicts.append(
                    {
                        "type": "arc",
                        "start_point": p,
                        "mid_point": p_midneg,
                        "end_point": p_endneg,
                    }
                )
                p = p_endneg.astype(float)
                t = _unit3(_rotate_about_axis(t, w, rad))
            else:
                raise ValueError("No valid arc found")

    return edge_dicts


class TrajectoryOperation(BaseOperation):
    def __init__(
        self,
        edge_dicts: list[dict],
        profile: dict,
        plane: str,
        sketch_shift: tuple[float, float] | None = None,
    ):
        self.edge_dicts = edge_dicts
        self.profile = profile
        self.plane = plane
        self.sketch_shift = sketch_shift

    def to_string(
        self,
    ) -> str:
        wire_strs = []
        for edge_dict in self.edge_dicts:
            if edge_dict["type"] == "line":
                wire_strs.append(
                    f"cq.Edge.makeLine(cq.Vector([{edge_dict['start_point'][0]}, {edge_dict['start_point'][1]}, {edge_dict['start_point'][2]}]), cq.Vector([{edge_dict['end_point'][0]}, {edge_dict['end_point'][1]}, {edge_dict['end_point'][2]}]))"
                )
            elif edge_dict["type"] == "arc":
                wire_strs.append(
                    f"cq.Edge.makeThreePointArc(cq.Vector([{edge_dict['start_point'][0]}, {edge_dict['start_point'][1]}, {edge_dict['start_point'][2]}]), cq.Vector([{edge_dict['mid_point'][0]}, {edge_dict['mid_point'][1]}, {edge_dict['mid_point'][2]}]), cq.Vector([{edge_dict['end_point'][0]}, {edge_dict['end_point'][1]}, {edge_dict['end_point'][2]}]))"
                )
        wire_str = f"cq.Wire.assembleEdges([{', '.join(wire_strs)}])"

        plane_expr = f"w{self.plane}" if isinstance(self.plane, int) else self.plane
        if self.sketch_shift is not None:
            plane_expr = (
                f"{plane_expr}.center({self.sketch_shift[0]}, {self.sketch_shift[1]})"
            )
        if self.profile["type"] == "circle":
            profile_expr = f"{plane_expr}.circle({self.profile['r']})"
        else:
            profile_expr = (
                f"{plane_expr}.rect({self.profile['w']}, {self.profile['h']})"
            )

        expr = f"{profile_expr}.sweep({wire_str})\n"
        return expr

    def transform(
        self, shift: list[float], scale: float, plane_axis: int | None = None
    ) -> None:
        if self.sketch_shift is not None:
            if plane_axis == 0:
                plane_shift = [shift[1], shift[2]]
            elif plane_axis == 1:
                plane_shift = [shift[2], shift[0]]
            elif plane_axis == 2:
                plane_shift = [shift[0], shift[1]]
            else:
                plane_shift = [0, 0]
            
            self.sketch_shift = [
                (self.sketch_shift[0] + plane_shift[0]) * scale,
                (self.sketch_shift[1] + plane_shift[1]) * scale,
            ]
        for edge_dict in self.edge_dicts:
            if "start_point" in edge_dict:
                edge_dict["start_point"] = [
                    (edge_dict["start_point"][0] + shift[0]) * scale,
                    (edge_dict["start_point"][1] + shift[1]) * scale,
                    (edge_dict["start_point"][2] + shift[2]) * scale,
                ]
            if "end_point" in edge_dict:
                edge_dict["end_point"] = [
                    (edge_dict["end_point"][0] + shift[0]) * scale,
                    (edge_dict["end_point"][1] + shift[1]) * scale,
                    (edge_dict["end_point"][2] + shift[2]) * scale,
                ]
            if "mid_point" in edge_dict:
                edge_dict["mid_point"] = [
                    (edge_dict["mid_point"][0] + shift[0]) * scale,
                    (edge_dict["mid_point"][1] + shift[1]) * scale,
                    (edge_dict["mid_point"][2] + shift[2]) * scale,
                ]

        if self.profile["type"] == "circle":
            self.profile["r"] *= scale
        else:
            self.profile["w"] *= scale
            self.profile["h"] *= scale

    def round(self) -> None:
        if self.sketch_shift is not None:
            self.sketch_shift = [round(v) for v in self.sketch_shift]

        for edge_dict in self.edge_dicts:
            if "start_point" in edge_dict:
                edge_dict["start_point"] = [round(v) for v in edge_dict["start_point"]]
            if "end_point" in edge_dict:
                edge_dict["end_point"] = [round(v) for v in edge_dict["end_point"]]
            if "mid_point" in edge_dict:
                edge_dict["mid_point"] = [round(v) for v in edge_dict["mid_point"]]

        if self.profile["type"] == "circle":
            self.profile["r"] = round(self.profile["r"])
        else:
            self.profile["w"] = round(self.profile["w"])
            self.profile["h"] = round(self.profile["h"])

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "trajectory_operation",
            "edge_dicts": self.edge_dicts,
            "profile": self.profile,
            "plane": self.plane,
        }

    @staticmethod
    def from_dict(entity: dict) -> "TrajectoryOperation":
        assert (
            entity["type"] == "trajectory_operation"
        ), f"Trying to build TrajectoryOperation from type {entity['type']}"
        return TrajectoryOperation(
            entity["edge_dicts"],
            entity["profile"],
            entity["plane"],
        )


class TrajectoryOperationFactory(BaseFactory):

    def __init__(
        self,
        *,
        n_segments_range: tuple[int, int] = (2, 8),
        line_len_range: tuple[float, float] = (0.04, 0.14),
        arc_deg_range: tuple[float, float] = (18.0, 180.0),
        arc_radius_range: tuple[float, float] = (
            0.14 / (2 * np.pi * (180 / 360)),
            0.14 / (2 * np.pi * (180 / 360)),
        ),
        origin_jitter: float = 0.08,
        line_probability: float = 0.60,
        min_turn_between_lines_deg: float = 15.0,
        profile_circle_radius_range: tuple[float, float] = (0.005, 0.04),
        profile_rect_size_range: tuple[float, float] = (0.005, 0.04),
    ):
        self.n_segments_range = n_segments_range
        self.line_len_range = line_len_range
        self.arc_deg_range = arc_deg_range
        self.arc_radius_range = arc_radius_range
        self.origin_jitter = origin_jitter
        self.line_probability = line_probability
        self.min_turn_between_lines_deg = min_turn_between_lines_deg
        self.profile_circle_radius_range = profile_circle_radius_range
        self.profile_rect_size_range = profile_rect_size_range

    def generate(
        self,
        t: cq.Vector | None = None,
        p: cq.Vector | None = None,
        profile: dict | None = None,
        plane: str | None = None,
        sketch_shift: tuple[float, float] | None = None,
    ) -> TrajectoryOperation | None:
        if plane is None:
            return
        if profile is None:
            if np.random.rand() < 0.5:
                profile = {
                    "type": "circle",
                    "r": float(np.random.uniform(*self.profile_circle_radius_range)),
                }
            else:
                w = float(np.random.uniform(*self.profile_rect_size_range))
                h = float(np.random.uniform(*self.profile_rect_size_range))
                profile = {"type": "rect", "w": w, "h": h}

        n_segments = np.random.randint(*self.n_segments_range)
        edge_dicts = random_arc_line_wire_3d(
            n_segments=n_segments,
            line_len_range=self.line_len_range,
            arc_deg_range=self.arc_deg_range,
            arc_radius_range=self.arc_radius_range,
            origin_jitter=self.origin_jitter,
            line_probability=self.line_probability,
            min_turn_between_lines_deg=self.min_turn_between_lines_deg,
            p=p,
            t=t,
        )
        if sketch_shift is None:
            if isinstance(plane, int):
                sketch_shift = [0, 0]
            else:
                sketch_shift = None
        return TrajectoryOperation(
            edge_dicts=edge_dicts,
            profile=profile,
            plane=plane,
            sketch_shift=sketch_shift,
        )

    def to_dict(self) -> dict:
        return {
            "type": "trajectory_operation_factory",
            "n_segments_range": self.n_segments_range,
            "line_len_range": self.line_len_range,
            "arc_deg_range": self.arc_deg_range,
            "arc_radius_range": self.arc_radius_range,
            "origin_jitter": self.origin_jitter,
        }

    @staticmethod
    def from_dict(entity: dict) -> "TrajectoryOperationFactory":
        assert (
            entity["type"] == "trajectory_operation_factory"
        ), f"Trying to build TrajectoryOperationFactory from type {entity['type']}"
        return TrajectoryOperationFactory(
            n_segments_range=entity["n_segments_range"],
            line_len_range=entity["line_len_range"],
            arc_deg_range=entity["arc_deg_range"],
            arc_radius_range=entity["arc_radius_range"],
            origin_jitter=entity["origin_jitter"],
            line_probability=entity["line_probability"],
            min_turn_between_lines_deg=entity["min_turn_between_lines_deg"],
            profile_circle_radius_range=entity["profile_circle_radius_range"],
            profile_rect_size_range=entity["profile_rect_size_range"],
        )


factories.register("trajectory_operation", TrajectoryOperationFactory)
operations.register("trajectory_operation", TrajectoryOperation)
