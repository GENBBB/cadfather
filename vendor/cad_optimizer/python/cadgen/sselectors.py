from __future__ import annotations
from functools import partial
from typing import List, Sequence, TypeVar

import numpy as np
from cadquery.occ_impl.shape_protocols import ShapeProtocol
from cadquery.selectors import Selector
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt

Shape = TypeVar("Shape", bound=ShapeProtocol)


class PointSelector(Selector):
    def __init__(
        self,
        pnt: tuple[float, float, float] | list[tuple[float, float, float]],
        atol=np.inf,
    ):
        if isinstance(pnt[0], list):
            self.pnts = pnt
        else:
            self.pnts = [pnt]
        self.atol = atol

    def dist(self, obj: Shape, pnt: tuple[float, float, float]):
        raise NotImplementedError

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        ret = []
        for pnt in self.pnts:
            x = min(objectList, key=partial(self.dist, pnt=pnt))  # type: ignore
            if self.dist(x, pnt) < self.atol:  # type: ignore
                ret.append(x)
            else:
                raise ValueError("Point is not on any object.")
        return ret


class PointOnFaceSelector(PointSelector):
    def dist(self, obj, pnt: tuple[float, float, float]):
        vertex_shape = BRepBuilderAPI_MakeVertex(gp_Pnt(*pnt)).Vertex()
        dist_calc = BRepExtrema_DistShapeShape(obj.wrapped, vertex_shape)
        dist_calc.Perform()

        if not dist_calc.IsDone():
            raise ValueError("Distance calculation failed.")

        return dist_calc.Value()


class PointOnEdgeSelector(PointSelector):
    def dist(self, obj, pnt: tuple[float, float, float]):
        vertex_shape = BRepBuilderAPI_MakeVertex(gp_Pnt(*pnt)).Vertex()
        dist_calc = BRepExtrema_DistShapeShape(obj.wrapped, vertex_shape)
        dist_calc.Perform()

        if not dist_calc.IsDone():
            raise ValueError("Distance calculation failed.")

        return dist_calc.Value()
