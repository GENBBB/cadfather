"""det KD-tree: ``pykdtree`` in place of ``scipy.spatial.cKDTree``.

``cadfit_pass`` takes its tree from here directly, without calling
``import_det_deps``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DET_KDTREE_LEAFSIZE = 16


class PyKDTree:
    """``cadfit_pass`` only reads ``query(points, 1)``, ``.data`` and the bounding
    box ``.mins``/``.maxes`` from the tree (as with scipy). The search is exact,
    like scipy's, and a query is about twice as fast.
    """

    def __init__(self, data: Any, *args: Any, **kwargs: Any) -> None:
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.mins = self.data.min(axis=0)
        self.maxes = self.data.max(axis=0)
        self._tree = _KDTree(self.data, leafsize=DET_KDTREE_LEAFSIZE)

    def query(self, x: Any, k: int = 1, *args: Any, distance_upper_bound: Any = None,
              **kwargs: Any) -> Any:
        return self._tree.query(np.ascontiguousarray(x, dtype=np.float64), k=k,
                                distance_upper_bound=distance_upper_bound)


try:
    from pykdtree.kdtree import KDTree as _KDTree
except Exception as exc:  # fall back to scipy, but loudly: same output, different wall time
    logger.warning(
        "pykdtree cannot be imported (%s: %s); det stays on scipy cKDTree, "
        "det wall time is higher than usual", type(exc).__name__, exc,
    )
    from scipy.spatial import cKDTree as KDTree
else:
    KDTree = PyKDTree
