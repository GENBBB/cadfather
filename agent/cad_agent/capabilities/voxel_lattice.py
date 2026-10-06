"""Voxelization like trimesh's `subdivide`, without subdividing the mesh: same points, different bookkeeping.

`trimesh.voxel.creation.voxelize_subdivide` subdivides the mesh (`remesh.subdivide_to_size`)
until all edges are shorter than `pitch / edge_factor`, then takes the cells of the
vertices. Every level sorts the edges (`unique_rows`), recomputes all edge lengths and
rebuilds the arrays; on long thin faces (CAD tessellation) one face yields 4^k children,
which is what dominates `voxelized` in the det profile.

A face subdivided k times gives a barycentric lattice of step 2^-k, and every lattice
point is the midpoint of two neighbours of the previous level, exactly as in trimesh
((P+Q)/2 is the same value in either order). The lattice is built per whole face,
without shared vertices or sorting, and the points go into a dense occupancy map. A
face's level is computed from its longest edge divided by 2^j; trimesh checks child
lengths from their coordinates, so a mismatch is possible only for an edge sitting on
the threshold within rounding.
"""

from __future__ import annotations

import numpy as np
from trimesh import transformations as tr
from trimesh.voxel import base
from trimesh.voxel import encoding as enc

# Lattice points per chunk of faces: keeps a chunk's memory in the tens of MB.
_BUDGET = 4_000_000


def _depths(tri: np.ndarray, max_edge: float, max_iter: int) -> np.ndarray:
    edge = (np.diff(tri[:, [0, 1, 2, 0], :], axis=1) ** 2).sum(axis=2) ** 0.5
    cur = edge.max(axis=1)
    k = np.zeros(len(tri), dtype=np.int64)
    for j in range(max_iter + 2):
        long_ = cur > max_edge
        if not long_.any():
            return k
        if j >= max_iter:
            # Like `subdivide_to_size`: the caller (det) catches it and goes on without the grid.
            raise ValueError("max_iter exceeded!")
        k[long_] += 1
        cur = np.where(long_, cur / 2.0, cur)
    return k


def _refine(G: np.ndarray) -> np.ndarray:
    # G: (F, n, n, 3), vertex (i, j) for i + j <= n - 1; (0,0), (1,0), (0,1) are a, b, c.
    # Lattice edges: along i (a->b), along j (a->c) and the diagonal (i+1, j)-(i, j+1) (b->c).
    F, n = G.shape[0], G.shape[1]
    m = 2 * n - 1
    H = np.empty((F, m, m, 3))
    H[:, 0::2, 0::2] = G
    H[:, 1::2, 0::2] = (G[:, :-1, :] + G[:, 1:, :]) / 2.0
    H[:, 0::2, 1::2] = (G[:, :, :-1] + G[:, :, 1:]) / 2.0
    H[:, 1::2, 1::2] = (G[:, 1:, :-1] + G[:, :-1, 1:]) / 2.0
    return H


def voxelize_lattice(mesh, pitch: float, max_iter: int | None = 10, edge_factor: float = 2.0) -> base.VoxelGrid:
    """Drop-in replacement for `voxelize_subdivide(mesh, pitch, max_iter, edge_factor)` with the same result."""
    max_edge = pitch / edge_factor
    tri = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
    if max_iter is None:
        longest = float(np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1).max())
        max_iter = max(int(np.ceil(np.log2(longest / max_edge))), 0)
    k = _depths(tri, max_edge, max_iter)
    flat = tri.reshape(-1, 3)
    lo = np.floor(flat.min(axis=0) / pitch).astype(np.int64) - 1
    hi = np.ceil(flat.max(axis=0) / pitch).astype(np.int64) + 1
    occ = np.zeros(tuple(hi - lo + 1), dtype=bool)
    for depth in np.unique(k):
        idx = np.nonzero(k == depth)[0]
        n = 2 ** int(depth) + 1
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        valid = (ii + jj) <= n - 1
        step = max(1, _BUDGET // (n * n))
        for s in range(0, len(idx), step):
            t = tri[idx[s:s + step]]
            G = np.zeros((len(t), 2, 2, 3))
            G[:, 0, 0], G[:, 1, 0], G[:, 0, 1] = t[:, 0], t[:, 1], t[:, 2]
            for _ in range(int(depth)):
                G = _refine(G)
            h = np.round(G[:, valid].reshape(-1, 3) / pitch).astype(np.int64) - lo
            occ[h[:, 0], h[:, 1], h[:, 2]] = True
    occupied = np.argwhere(occ) + lo
    origin = occupied.min(axis=0)
    return base.VoxelGrid(
        enc.SparseBinaryEncoding(occupied - origin),
        transform=tr.scale_and_translate(scale=pitch, translate=origin * pitch))
