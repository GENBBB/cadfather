import logging

import numpy as np

# from cadquery_addons import *

from .registry import factories
from .base import BaseFactory, BaseOperation

logger = logging.getLogger(__name__)


class FaceFillet(BaseOperation):
    def __init__(self, ineq_sign: str, axis: str, radius: float):
        self.ineq_sign = ineq_sign
        self.axis = axis
        self.radius = radius

    def round(self):
        self.radius = int(self.radius)

    def transform(self, shift, scale):
        self.radius *= scale

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "FaceFillet",
            "ineq_sign": self.ineq_sign,
            "axis": self.axis,
            "radius": self.radius,
        }

    @staticmethod
    def from_dict(entity: dict) -> "FaceFillet":
        assert (
            entity["type"] == "FaceFillet"
        ), f"Trying to build FaceFillet from type {entity['type']}"
        return FaceFillet(
            entity["ineq_sign"],
            entity["axis"],
            entity["radius"],
        )


class FaceChamfer(BaseOperation):
    def __init__(
        self,
        ineq_sign: str,
        axis: str,
        width1: float,
        width2: float | None = None,
        equal_offsets: bool | None = False,
    ):
        self.ineq_sign = ineq_sign
        self.axis = axis
        self.width1 = width1
        self.width2 = width2
        self.equal_offsets = equal_offsets

    def round(self):
        self.width1 = int(self.width1)
        if self.width2 is not None:
            self.width2 = int(self.width2)

    def transform(self, shift, scale):
        self.width1 *= scale
        if self.width2 is not None:
            self.width2 *= scale

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": "FaceChamfer",
            "ineq_sign": self.ineq_sign,
            "axis": self.axis,
            "width1": self.width1,
            "width2": self.width2,
            "equal_offsets": self.equal_offsets,
        }

    @staticmethod
    def from_dict(entity: dict) -> "FaceChamfer":
        assert (
            entity["type"] == "FaceChamfer"
        ), f"Trying to build FaceChamfer from type {entity['type']}"
        return FaceChamfer(
            entity["ineq_sign"],
            entity["axis"],
            entity["width1"],
            entity["width2"],
            entity["equal_offsets"],
        )


class FaceFilletChamferFactory(BaseFactory):
    def __init__(
        self,
        fillet_chamfer_probs: tuple[float, float] = (0.5, 0.5),
        equal_offsets_prob: float = 0.5,
    ):
        self.fillet_chamfer_probs = fillet_chamfer_probs
        self.equal_offsets_prob = equal_offsets_prob

    def generate(self, s=None, world_size: float = 1.0):
        if not s:
            return None

        op = np.random.choice(["Fillet", "Chamfer"], p=self.fillet_chamfer_probs)
        axis = np.random.choice(["X", "Y", "Z"])
        ineq_sign = np.random.choice([">", "<"])

        equal_offsets = (
            np.random.random() < self.equal_offsets_prob if op == "Chamfer" else None
        )

        selection_str = ineq_sign + axis
        t = f".faces('{selection_str}')"

        # max_offset = 10
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

        maxv = 0.15 * world_size
        maxv_chamfer = 0.15 * world_size

        v = np.random.uniform(0, maxv)
        v1 = np.random.uniform(0, maxv_chamfer)
        v2 = np.random.uniform(0, maxv_chamfer)

        code_to_exec = ""
        if op == "Fillet":
            code_to_exec = s + f".{op.lower()}({v})"
        else:
            if equal_offsets:
                code_to_exec = s + f".{op.lower()}({v1})"
            else:
                code_to_exec = s + f".{op.lower()}({v1}, {v2})"
        exec(code_to_exec, globals())
        _w = globals()["r"].val()
        if _w.isValid():
            if op.lower() == "fillet":
                # radius = np.random.uniform(0, maxv)
                radius = v
                logger.info("Final face fillet radius=%s", radius)
                return FaceFillet(ineq_sign, axis, radius)
            else:
                # width1 = np.random.uniform(0, maxv)
                width1 = v1
                if equal_offsets:
                    logger.info("Final face chamfer width=%s (equal offsets)", width1)
                    width2 = None
                else:
                    # width2 = np.random.uniform(0, maxv)
                    width2 = v2
                    # if np.random.random() > 0.5:
                    #     width2, width1 = width1, width2
                    logger.info("Final face chamfer widths=(%s, %s)", width1, width2)
                return FaceChamfer(ineq_sign, axis, width1, width2, equal_offsets)

    def to_dict(self) -> dict:
        return {
            "type": "FaceFilletChamferFactory",
            "fillet_chamfer_probs": self.fillet_chamfer_probs,
            "equal_offsets_prob": self.equal_offsets_prob,
        }

    @staticmethod
    def from_dict(entity: dict) -> "FaceFilletChamferFactory":
        return FaceFilletChamferFactory(
            fillet_chamfer_probs=entity["fillet_chamfer_probs"],
            equal_offsets_prob=entity["equal_offsets_prob"],
        )


factories.register("face_fillet_chamfer", FaceFilletChamferFactory)
