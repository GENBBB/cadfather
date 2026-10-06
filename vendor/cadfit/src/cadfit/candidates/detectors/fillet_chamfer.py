"""Fillet / chamfer edge-walk detector — CADFit-style edge modifier.

Algorithm (no OCCT introspection needed)
----------------------------------------
1.  Find sharp edges of the input mesh: any face-adjacency edge whose
    dihedral angle exceeds ``sharp_angle_deg``.
2.  Cluster sharp edges into connected chains via union-find on the
    shared vertices.  A chain corresponds to one "feature edge" in the
    CAD sense (e.g. one rim of a box, one rim around a hole).
3.  For each chain, compute a representative point (length-weighted
    midpoint) and a representative radius scale.
4.  Emit candidate blocks::

        _piece = (the existing r)
        r = _piece.edges(cq.selectors.NearestToPointSelector((cx, cy, cz))).fillet(R)

    for R in ``radii`` (default fractions of mesh diagonal).  Same for
    chamfer.

Notes
-----
These detectors are *block-emitting* and must run on the PREVIOUS
build's mesh (not the residual).  The proposer is responsible for
calling this on the rendered prev_code mesh, not on the ADD/CUT
residual.  The function still takes a `mesh` argument so it conforms
to the existing detector contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from .extrude import DetectorOutput


@dataclass
class _Chain:
    edges: list[tuple[int, int]]    # list of (v_i, v_j) sharp edges
    midpoint: np.ndarray            # 3D centroid (length-weighted)
    total_length: float


def _find_sharp_edges(mesh: trimesh.Trimesh,
                      sharp_angle_deg: float,
                      ) -> tuple[list[tuple[int, int]], trimesh.Trimesh]:
    """Return (sharp_edges, working_mesh).

    Sharp edges are face-adjacency edges whose dihedral angle exceeds
    sharp_angle_deg.  If the input mesh has duplicate vertices (as
    happens when STL is loaded with process=False), we silently merge
    them on a local copy so face_adjacency_* is populated.  The
    returned working_mesh uses the merged-vertex coordinate frame.
    """
    work = mesh
    try:
        if work.face_adjacency_edges.size == 0:
            work = mesh.copy()
            work.merge_vertices()
    except Exception:
        try:
            work = mesh.copy()
            work.merge_vertices()
        except Exception:
            return [], mesh
    try:
        adj_angles = np.asarray(work.face_adjacency_angles)
        adj_edges  = np.asarray(work.face_adjacency_edges)
    except Exception:
        return [], work
    if adj_angles.size == 0:
        return [], work
    rad = math.radians(sharp_angle_deg)
    sharp_mask = adj_angles > rad
    edges = adj_edges[sharp_mask]
    out = []
    for v0, v1 in edges:
        out.append((int(min(v0, v1)), int(max(v0, v1))))
    return out, work


def _cluster_edge_chains(edges: list[tuple[int, int]],
                         mesh: trimesh.Trimesh,
                         ) -> list[_Chain]:
    """Group connected sharp edges into chains via union-find on the
    shared vertices."""
    if not edges:
        return []
    # Union-find over vertex indices that appear in any sharp edge.
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for v0, v1 in edges:
        for v in (v0, v1):
            parent.setdefault(v, v)
        union(v0, v1)

    groups: dict[int, list[tuple[int, int]]] = {}
    for e in edges:
        root = find(e[0])
        groups.setdefault(root, []).append(e)

    verts = np.asarray(mesh.vertices)
    chains: list[_Chain] = []
    for es in groups.values():
        total_len = 0.0
        weighted_mid = np.zeros(3, dtype=np.float64)
        for v0, v1 in es:
            p0, p1 = verts[v0], verts[v1]
            ll = float(np.linalg.norm(p1 - p0))
            mid = 0.5 * (p0 + p1)
            weighted_mid += ll * mid
            total_len += ll
        if total_len < 1e-9:
            continue
        weighted_mid /= total_len
        chains.append(_Chain(es, weighted_mid, total_len))
    chains.sort(key=lambda c: c.total_length, reverse=True)
    return chains


def _emit_modifier_program(kind: str, pt: np.ndarray, r_or_d: float) -> str:
    """Emit a CadQuery snippet that fillets / chamfers an existing ``r``
    near `pt`.  This is a STANDALONE program (not yet wrapped into the
    iter-2+ block; the proposer will rewrite it).

    For the proposer's block-rewriter to work, we still emit a fake
    initial ``r``: the result variable just gets the same workplane
    chain, with the fillet/chamfer applied.  The rewriter replaces
    ``result`` -> ``_piece`` and adds ``r = r.union(_piece)`` (or
    similar) which is WRONG for a modifier.  We therefore mark the
    operation specially -- the proposer recognises "fillet" / "chamfer"
    detectors and emits a DIRECT modifier block instead of the usual
    union/cut wrapper.
    """
    cx, cy, cz = float(pt[0]), float(pt[1]), float(pt[2])
    if kind == "fillet":
        return (
            "import cadquery as cq\n"
            f"# fillet-modifier near=({cx:.4f},{cy:.4f},{cz:.4f}) r={r_or_d:.4f}\n"
            f"result = r.edges(cq.selectors.NearestToPointSelector("
            f"({cx:.4f},{cy:.4f},{cz:.4f}))).fillet({r_or_d:.4f})\n"
        )
    else:
        return (
            "import cadquery as cq\n"
            f"# chamfer-modifier near=({cx:.4f},{cy:.4f},{cz:.4f}) d={r_or_d:.4f}\n"
            f"result = r.edges(cq.selectors.NearestToPointSelector("
            f"({cx:.4f},{cy:.4f},{cz:.4f}))).chamfer({r_or_d:.4f})\n"
        )


def detect_fillet_chamfer(mesh: trimesh.Trimesh,
                          sharp_angle_deg: float = 25.0,
                          min_chain_length_frac: float = 0.05,
                          radii: list[float] | None = None,
                          max_chains: int = 8,
                          kinds: tuple[str, ...] = ("fillet", "chamfer"),
                          ) -> list[DetectorOutput]:
    """Edge-walk detector.

    Parameters
    ----------
    sharp_angle_deg : float
        Adjacent-face dihedral angle threshold for an edge to count as
        "sharp".  Default 25 degrees -- catches box rims and rims around
        holes; ignores nearly-coplanar mesh tessellation artifacts.
    min_chain_length_frac : float
        Skip chains whose total length is below this fraction of the
        mesh bbox diagonal.
    radii : list[float] | None
        Radii / chamfer distances to emit per chain, EXPRESSED AS
        FRACTIONS OF THE MESH bbox diagonal.  Default
        [0.02, 0.05, 0.10] -- three magnitudes.
    max_chains : int
        Drop chains beyond the top-K by total length.
    kinds : tuple[str, ...]
        Subset of {"fillet", "chamfer"} to emit.
    """
    if mesh is None or len(mesh.faces) < 4:
        return []
    if radii is None:
        radii = [0.02, 0.05, 0.10]

    edges, work = _find_sharp_edges(mesh, sharp_angle_deg)
    if not edges:
        return []
    chains = _cluster_edge_chains(edges, work)
    if not chains:
        return []

    lo, hi = mesh.bounds
    diag = float(np.linalg.norm(hi - lo))
    if diag < 1e-9:
        return []
    min_chain_len = diag * min_chain_length_frac

    outs: list[DetectorOutput] = []
    for ci, ch in enumerate(chains[:max_chains]):
        if ch.total_length < min_chain_len:
            continue
        for kind in kinds:
            for r_frac in radii:
                r_abs = max(diag * float(r_frac), 1e-4)
                program = _emit_modifier_program(kind, ch.midpoint, r_abs)
                # Higher = better; reward LONG chains, modest penalty
                # for radii far from the typical sweet spot (~5% diag).
                length_frac = float(min(ch.total_length / diag, 1.0))
                outs.append(DetectorOutput(
                    program=program,
                    score=length_frac - 0.1 * abs(r_frac - 0.05),
                    debug={
                        "chain_id": ci,
                        "kind": kind,
                        "midpoint": [round(float(x), 4) for x in ch.midpoint],
                        "chain_length_frac": ch.total_length / diag,
                        "n_edges": len(ch.edges),
                        "radius_frac": float(r_frac),
                        "radius_abs": r_abs,
                        "modifier": True,   # flag for the proposer
                    },
                ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
