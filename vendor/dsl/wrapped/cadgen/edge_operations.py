import logging

import numpy as np

from .registry import factories
from .selectors import PointOnEdgeSelector
from .utils import get_edge_midpoint, pending_normalization_scale
from .base import BaseFactory, BaseOperation

logger = logging.getLogger(__name__)

MIN_FILLET_CHAMFER_ARGUMENT = 2.0
MIN_NORMALIZED_EDGE_LENGTH = 8.0


class Fillet(BaseOperation):
    def __init__(self, point: tuple[float, float, float], radius: float):
        self.point = point
        self.radius = radius

    def round(self):
        self.radius = max(int(MIN_FILLET_CHAMFER_ARGUMENT), int(self.radius))
        self.point = list(self.point)
        self.point[0] = int(self.point[0])
        self.point[1] = int(self.point[1])
        self.point[2] = int(self.point[2])
        self.point = tuple(self.point)

    def transform(self, shift, scale):
        self.point = list(self.point)
        self.point[0] = (self.point[0] + shift[0]) * scale
        self.point[1] = (self.point[1] + shift[1]) * scale
        self.point[2] = (self.point[2] + shift[2]) * scale
        self.point = tuple(self.point)
        self.radius *= scale

    def to_string(self):
        t = f"r=r.edges(PointOnEdgeSelector([{self.point[0]}, {self.point[1]}, {self.point[2]}])).fillet({self.radius})\n"
        return t

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "Fillet",
            "point": self.point,
            "radius": self.radius,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Fillet":
        assert (
            entity["type"] == "Fillet"
        ), f"Trying to build Fillet from type {entity['type']}"
        return Fillet(point=entity["point"], radius=entity["radius"])


class Chamfer(BaseOperation):
    def __init__(
        self,
        point: tuple[float, float, float],
        width1: float,
        width2: float | None = None,
        equal_offsets: bool | None = None,
    ):
        self.point = point
        self.width1 = width1
        self.width2 = width2
        self.equal_offsets = equal_offsets

    def round(self):
        self.width1 = max(int(MIN_FILLET_CHAMFER_ARGUMENT), int(self.width1))
        if self.width2 is not None:
            self.width2 = max(int(MIN_FILLET_CHAMFER_ARGUMENT), int(self.width2))
        self.point = list(self.point)
        self.point[0] = int(self.point[0])
        self.point[1] = int(self.point[1])
        self.point[2] = int(self.point[2])
        self.point = tuple(self.point)

    def transform(self, shift, scale):
        self.point = list(self.point)
        self.point[0] = (self.point[0] + shift[0]) * scale
        self.point[1] = (self.point[1] + shift[1]) * scale
        self.point[2] = (self.point[2] + shift[2]) * scale
        self.point = tuple(self.point)
        self.width1 *= scale
        if self.width2 is not None:
            self.width2 *= scale

    def to_string(self):
        t = f"r=r.edges(PointOnEdgeSelector([{self.point[0]}, {self.point[1]}, {self.point[2]}]))"
        if self.width2 is None:
            t += f".chamfer({self.width1})\n"
        else:
            t += f".chamfer({self.width1}, {self.width2})\n"
        return t

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "Chamfer",
            "point": self.point,
            "width1": self.width1,
            "width2": self.width2,
            "equal_offsets": self.equal_offsets,
        }

    @staticmethod
    def from_dict(entity: dict) -> "Chamfer":
        assert (
            entity["type"] == "Chamfer"
        ), f"Trying to build Chamfer from type {entity['type']}"
        return Chamfer(
            point=entity["point"],
            width1=entity["width1"],
            width2=entity["width2"],
            equal_offsets=entity["equal_offsets"],
        )


class FilletChamferFactory(BaseFactory):
    def __init__(
        self,
        fillet_chamfer_probs: tuple[float, float] = (0.5, 0.5),
        equal_offsets_prob: float = 0.5,
    ):
        self.fillet_chamfer_probs = fillet_chamfer_probs
        self.equal_offsets_prob = equal_offsets_prob

    @staticmethod
    def _argument_bounds(
        world_size: float,
        normalization_scale: float,
    ) -> tuple[float, float]:
        min_value = MIN_FILLET_CHAMFER_ARGUMENT / normalization_scale
        max_value = 0.1 * float(world_size) / normalization_scale
        if max_value < min_value:
            raise ValueError("world_size is too small for fillet/chamfer minimum.")
        return min_value, max_value

    @staticmethod
    def _edge_sampling_probabilities(
        normalized_lengths: list[float],
    ) -> np.ndarray:
        weights = np.sqrt(np.asarray(normalized_lengths, dtype=float))
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("No positive edge lengths to sample.")
        return weights / total

    @staticmethod
    def _eligible_edges(
        w,
        world_size: float,
    ) -> tuple[list[tuple[object, float]], float]:
        normalization_scale = pending_normalization_scale(w, world_size)
        candidates = []
        for edge in w.edges().vals():
            normalized_length = float(edge.Length()) * normalization_scale
            if normalized_length >= MIN_NORMALIZED_EDGE_LENGTH:
                candidates.append((edge, normalized_length))
        return candidates, normalization_scale

    def generate(self, s=None, world_size: float = 1.0):
        if not s:
            return None

        world_size = float(world_size)
        op = np.random.choice(["Fillet", "Chamfer"], p=self.fillet_chamfer_probs)
        exec(s, globals())
        w = globals()["r"]

        equal_offsets = (
            np.random.random() < self.equal_offsets_prob if op == "Chamfer" else None
        )

        max_offset = 10 * world_size

        # L = 0
        # r = max_offset

        # while L + 1e-3 < r:
        #     v = 0.5 * (L + r)
        #     try:
        #         exec(s + t + f".{op.lower()}({v})")
        #         w = globals()["r"].val()
        #         if w.isValid():
        #             L = v
        #         else:
        #             r = v
        #     except Exception:
        #         r = v

        # maxv = L

        all_edges, normalization_scale = self._eligible_edges(w, world_size)
        minv, maxv = self._argument_bounds(world_size, normalization_scale)
        minv_chamfer, maxv_chamfer = minv, maxv

        if len(all_edges) == 0:
            raise ValueError(
                f"No edges with normalized length >= {MIN_NORMALIZED_EDGE_LENGTH}."
            )

        good = False
        total = len(all_edges)
        point_on_edge: tuple[float, float, float] | None = None
        v: float | None = None
        v1: float | None = None
        v2: float | None = None

        for _ in range(total):
            normalized_lengths = [length for _, length in all_edges]
            probabilities = self._edge_sampling_probabilities(normalized_lengths)
            idx = int(np.random.choice(len(all_edges), p=probabilities))
            edge, _ = all_edges[idx]
            point_on_edge = get_edge_midpoint(edge)

            t = f"r=r.edges(PointOnEdgeSelector([{point_on_edge[0]}, {point_on_edge[1]}, {point_on_edge[2]}]))"
            v = np.random.uniform(minv, maxv)
            v1 = np.random.uniform(minv_chamfer, maxv_chamfer)
            v2 = np.random.uniform(minv_chamfer, maxv_chamfer)
            try:
                code_to_exec = ""
                if op == "Fillet":
                    code_to_exec = s + t + f".{op.lower()}({v})\n"
                else:
                    if equal_offsets:
                        code_to_exec = s + t + f".{op.lower()}({v1})\n"
                    else:
                        code_to_exec = s + t + f".{op.lower()}({v1}, {v2})\n"
                exec(code_to_exec, globals())
                _w = globals()["r"].val()
                if _w.isValid():
                    good = True
                    break
            except Exception:
                all_edges.pop(idx)
                pass

        if not good:
            raise ValueError(f"No edges to {op}.")

        logger.info(
            "Applying edge operation: op=%s, point=%s, max_offset=%s",
            op,
            point_on_edge,
            max_offset,
        )
        assert point_on_edge is not None
        assert v is not None

        if op.lower() == "fillet":
            # radius = np.random.uniform(0, maxv)
            radius = v
            logger.info("Final fillet radius=%s", radius)
            return Fillet(point_on_edge, radius)
        else:
            # width1 = np.random.uniform(0, maxv)
            width1 = v1
            if equal_offsets:
                logger.info("Final chamfer width=%s (equal offsets)", width1)
                width2 = None
            else:
                # width2 = np.random.uniform(0, maxv)
                width2 = v2
                # if np.random.random() > 0.5:
                #     width2, width1 = width1, width2
                logger.info("Final chamfer widths=(%s, %s)", width1, width2)
            assert width1 is not None
            return Chamfer(point_on_edge, width1, width2, equal_offsets)

    def to_dict(self) -> dict:
        return {
            "type": "FilletChamferFactory",
            "fillet_chamfer_probs": self.fillet_chamfer_probs,
            "equal_offsets_prob": self.equal_offsets_prob,
        }

    @staticmethod
    def from_dict(entity: dict) -> "FilletChamferFactory":
        return FilletChamferFactory(
            fillet_chamfer_probs=entity["fillet_chamfer_probs"],
            equal_offsets_prob=entity["equal_offsets_prob"],
        )


factories.register("fillet_chamfer", FilletChamferFactory)
