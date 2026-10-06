import logging

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_UniformAbscissa

from .base import BaseFactory, BaseOperation
from .registry import factories
from .sselectors import *
from .utils import float_to_string

logger = logging.getLogger(__name__)


class Sweep(BaseOperation):
    def __init__(
        self,
        pitch: float,
        height: float,
        radius: float,
        angle: float | None,
        n_per_turn: int,
        profile: dict,
    ):
        self.pitch = pitch
        self.height = height
        self.radius = radius
        self.angle = angle
        self.n_per_turn = n_per_turn
        self.profile = profile
        self.center = [0.0, 0.0, 0.0]
        self.sketch_shift = [0.0, 0.0]

    def to_string(
        self,
        plane: str = "cq.Workplane('XY')",
        axis: int | None = None,
        use_literals: bool = True,
    ) -> str:
        x_shift = self.sketch_shift[0]
        y_shift = self.sketch_shift[1]
        if use_literals:
            center_x = float_to_string(self.radius + x_shift)
            center_y = float_to_string(y_shift)
            center_plane = f"{plane}.center({center_x}, {center_y})"
        else:
            radius_str = float_to_string(self.radius)
            base_expr = f"(sweep_radius:={radius_str})"
            eps = 1e-9
            if abs(x_shift) > eps:
                base_expr = f"{base_expr}+{float_to_string(x_shift)}"
            y_expr = "0" if abs(y_shift) <= eps else float_to_string(y_shift)
            center_plane = f"{plane}.center({base_expr}, {y_expr})"
        if self.profile["type"] == "circle":
            profile_expr = f"{center_plane}.circle({self.profile['r']})"
        else:
            profile_expr = (
                f"{center_plane}.rect({self.profile['w']}, {self.profile['h']})"
            )
        angle_param = f", angle={self.angle}" if self.angle is not None else ""
        dir_map = {
            0: "cq.Vector(0, 0, 1)",  # YZ plane
            1: "cq.Vector(1, 0, 0)",  # ZX plane
            2: "cq.Vector(0, 1, 0)",  # XY plane
        }
        dir_expr = (
            dir_map.get(axis, "cq.Vector(0, 0, 1)")
            if axis is not None
            else "cq.Vector(0, 0, 1)"
        )
        if use_literals:
            center_expr = (
                f"cq.Vector({float_to_string(self.center[0])}, "
                f"{float_to_string(self.center[1])}, "
                f"{float_to_string(self.center[2])})"
            )
            radius_expr = float_to_string(self.radius)
        else:
            center_expr = (
                f"cq.Vector({self.center[0]}, {self.center[1]}, {self.center[2]})"
            )
            radius_expr = "sweep_radius"
        helix_expr = (
            f"cq.Wire.makeHelix(pitch={self.pitch}, height={self.height}, "
            f"radius={radius_expr}{angle_param}, center={center_expr}, dir={dir_expr})"
        )
        expr = f"{profile_expr}.sweep({helix_expr}, isFrenet=True)\n"
        return expr

    def transform(
        self, shift: list[float], scale: float, plane_axis: int | None = 2
    ) -> None:
        self.pitch *= scale
        self.height *= scale
        self.radius *= scale
        self.center = [
            (self.center[0] + shift[0]) * scale,
            (self.center[1] + shift[1]) * scale,
            (self.center[2] + shift[2]) * scale,
        ]
        if plane_axis == 0:
            plane_shift = [shift[1], shift[2]]
        elif plane_axis == 1:
            plane_shift = [shift[2], shift[0]]
        else:
            plane_shift = [shift[0], shift[1]]
        self.sketch_shift = [
            (self.sketch_shift[0] + plane_shift[0]) * scale,
            (self.sketch_shift[1] + plane_shift[1]) * scale,
        ]
        if self.profile["type"] == "circle":
            self.profile["r"] *= scale
        else:
            self.profile["w"] *= scale
            self.profile["h"] *= scale

    def round(self) -> None:
        self.pitch = round(self.pitch)
        self.height = round(self.height)
        self.radius = round(self.radius)
        self.center = [round(v) for v in self.center]
        self.sketch_shift = [round(v) for v in self.sketch_shift]
        if self.profile["type"] == "circle":
            self.profile["r"] = round(self.profile["r"])
        else:
            self.profile["w"] = round(self.profile["w"])
            self.profile["h"] = round(self.profile["h"])
        if self.angle is not None:
            self.angle = round(self.angle)


class SweepFactory(BaseFactory):

    def __init__(
        self,
        *,
        pitch_range: tuple[float, float] = (5.0, 20.0),
        height_range: tuple[float, float] = (30.0, 120.0),
        base_radius_range: tuple[float, float] = (10.0, 40.0),
        cone_angle_range_deg: tuple[float, float] = (0.0, 10.0),
        profile_circle_radius_range: tuple[float, float] = (1.0, 5.0),
        profile_rect_size_range: tuple[float, float] = (2.0, 8.0),
        helix_angle_probability: float = 0.5,
    ):
        self.pitch_range = pitch_range
        self.height_range = height_range
        self.base_radius_range = base_radius_range
        self.cone_angle_range_deg = cone_angle_range_deg
        self.profile_circle_radius_range = profile_circle_radius_range
        self.profile_rect_size_range = profile_rect_size_range
        self.helix_angle_probability = helix_angle_probability

    def generate(self) -> Sweep:
        pitch = float(np.random.uniform(*self.pitch_range))
        height = float(np.random.uniform(*self.height_range))
        base_radius = float(np.random.uniform(*self.base_radius_range))
        angle_deg = 5.0 if np.random.uniform() < self.helix_angle_probability else None
        if np.random.rand() < 0.5:
            profile = {
                "type": "circle",
                "r": float(np.random.uniform(*self.profile_circle_radius_range)),
            }
        else:
            w = float(np.random.uniform(*self.profile_rect_size_range))
            h = float(np.random.uniform(*self.profile_rect_size_range))
            profile = {"type": "rect", "w": w, "h": h}

        return Sweep(
            pitch=pitch,
            height=height,
            radius=base_radius,
            angle=angle_deg,
            n_per_turn=120,
            profile=profile,
        )

    @staticmethod
    def discretize_wire_continuous(
        wire: cq.Wire, n_per_loop: int = 100
    ) -> list[tuple[float, float, float]]:
        """
        Discretize a closed wire into exactly n_per_loop points WITHOUT repeating the last one.
        Suitable for smooth interpolation.
        """
        # Collect all edges into one list
        edges = wire.Edges()
        total_len = sum(e.Length() for e in edges)  # type: ignore

        # Distribute points in proportion to each edge's length
        pts = []
        for e in edges:
            frac = e.Length() / total_len
            n_local = max(2, int(round(frac * n_per_loop)))
            d = SweepFactory.discretize_edge(e, n=n_local)
            # Add ALL points, including first and last; duplicates are removed below
            pts.extend(d)

        # Remove duplicates at edge joints (keep only the start of the loop)
        cleaned = [pts[0]]
        for p in pts[1:]:
            if not (
                abs(p[0] - cleaned[-1][0]) < 1e-6
                and abs(p[1] - cleaned[-1][1]) < 1e-6
                and abs(p[2] - cleaned[-1][2]) < 1e-6
            ):
                cleaned.append(p)

        # Make sure it is closed: if first is approximately last, drop the last
        if len(cleaned) > 1 and all(
            abs(cleaned[0][i] - cleaned[-1][i]) < 1e-6 for i in range(3)
        ):
            cleaned = cleaned[:-1]

        return cleaned

    @staticmethod
    def discretize_edge(edge: cq.Edge, n: int = 50) -> list[tuple[float, float, float]]:
        c = BRepAdaptor_Curve(edge.wrapped)
        algo = GCPnts_UniformAbscissa(c, n)
        pts = []
        for i in range(1, algo.NbPoints() + 1):
            p = c.Value(algo.Parameter(i))
            pts.append((p.X(), p.Y(), p.Z()))
        return pts

    @staticmethod
    def make_smooth_spiral_path(
        sketch_wp: cq.Workplane,
        pitch: float = 1.5,
        height: float = 10.0,
        radius: float = 5.0,
        angle: float = 10.0,
        n_per_turn: int = 120,
        workplane: cq.Workplane = cq.Workplane("XY"),
    ) -> cq.Workplane:
        """
        Create a smooth 3D spiral path from an arbitrary closed sketch.
        If the sketch is a circle, use makeHelix directly.
        Otherwise build a spiral from the sketch shape.

        Args:
            sketch_wp: Workplane with a closed wire/sketch
            pitch: Distance between spiral turns
            height: Total spiral height
            radius: Spiral radius (used for a circle, or as a base point for other shapes)
            angle: Cone angle in degrees (for a conical spiral)
            n_per_turn: Number of points per turn (for non-circle cases)

        Returns:
            cq.Workplane with the path for sweep
        """
        val = sketch_wp.val()
        if isinstance(val, cq.Edge):
            wire = cq.Wire.assembleEdges([val])
        elif isinstance(val, cq.Wire):
            wire = val
        else:
            raise ValueError("Input must be a closed Wire or Edge")

        if not wire.IsClosed():  # type: ignore
            raise ValueError("Wire must be closed for a spiral!")

        # Check whether the wire is a circle
        edges = wire.Edges()
        if len(edges) == 1:
            edge = edges[0]
            # Check that it is a circle
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GeomAbs import GeomAbs_Circle

            adaptor = BRepAdaptor_Curve(edge.wrapped)
            if adaptor.GetType() == GeomAbs_Circle:
                # A circle: use makeHelix directly
                helix_wire = cq.Wire.makeHelix(
                    pitch=pitch, height=height, radius=radius, angle=angle
                )
                return workplane.add(helix_wire)

        # Non-circle case: build a spiral from the sketch shape
        # Compute the number of turns from height and pitch
        turns = int(height / pitch) if pitch > 0 else 1

        # Discretize ONE turn
        base_pts = SweepFactory.discretize_wire_continuous(
            wire, n_per_loop=n_per_turn
        )  # [(x,y,0), ...]
        if not base_pts:
            raise ValueError("No points sampled — check sketch")

        # Collect 3D points for all turns: z grows smoothly
        pts_3d = []
        n = len(base_pts)

        for k in range(turns * n):
            turn_frac = k / (turns * n)  # from 0 to 1
            z = turn_frac * height
            i = k % n
            pts_3d.append((*base_pts[i][:2], z))

        # Build a spline through all points; CadQuery makes a cubic spline
        return workplane.spline(pts_3d)


factories.register("sweep", SweepFactory)
