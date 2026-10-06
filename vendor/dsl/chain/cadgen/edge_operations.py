import logging

import numpy as np

from .registry import factories
from .sselectors import *
from .utils import get_edge_midpoint

logger = logging.getLogger(__name__)


class Fillet:
    def __init__(self, point: tuple[float, float, float], radius: float):
        self.point = point
        self.radius = radius

    def round(self):
        self.radius = int(self.radius)
        if self.radius == 0:
            self.radius = 1
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


class Chamfer:
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
        self.width1 = int(self.width1)
        if self.width1 == 0:
            self.width1 = 1
        if self.width2 is not None:
            self.width2 = int(self.width2)
            if self.width2 == 0:
                self.width2 = 1
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


class FilletChamferFactory:
    def __init__(
        self,
        fillet_chamfer_probs: tuple[float, float] = (0.5, 0.5),
        equal_offsets_prob: float = 0.5,
    ):
        self.fillet_chamfer_probs = fillet_chamfer_probs
        self.equal_offsets_prob = equal_offsets_prob

    def generate(self, s=None):
        if not s:
            return None

        op = np.random.choice(["Fillet", "Chamfer"], p=self.fillet_chamfer_probs)
        exec(s, globals())
        w = globals()["r"]

        equal_offsets = (
            np.random.random() < self.equal_offsets_prob if op == "Chamfer" else None
        )

        max_offset = 10

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

        maxv = 0.1
        maxv_chamfer = 0.1

        all_edges = w.edges().vals()

        q = np.quantile([e.Length() for e in all_edges], 0.5)
        all_edges = [e for e in all_edges if e.Length() >= q]

        good = False
        total = len(all_edges)
        point_on_edge: tuple[float, float, float] | None = None
        v: float | None = None
        v1: float | None = None
        v2: float | None = None

        for _ in range(total):
            idx = np.random.choice(len(all_edges))
            edge = all_edges[idx]
            point_on_edge = get_edge_midpoint(edge)

            t = f"r=r.edges(PointOnEdgeSelector([{point_on_edge[0]}, {point_on_edge[1]}, {point_on_edge[2]}]))"
            v = np.random.uniform(0, maxv)
            v1 = np.random.uniform(0, maxv_chamfer)
            v2 = np.random.uniform(0, maxv_chamfer)
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


factories.register("fillet_chamfer", FilletChamferFactory)
