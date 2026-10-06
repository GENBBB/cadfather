"""Local-extrude detector — CADFit-style stable-depth sweep.

The key difference from ``planar_cluster``: instead of extruding each
planar cluster's profile by an arbitrary depth fraction (0.25, 0.5,
1.0, 1.5 of mesh extent), this detector finds the **stable depth** at
which the extruded shape's LATERAL surface lies ON the target body's
surface.

Algorithm
---------
1.  Cluster mesh faces by normal direction (reuses
    ``planar_cluster._greedy_normal_clusters``).
2.  For each surviving cluster:
      a.  Project cluster triangles to 2D in the (u, v) plane
          perpendicular to the cluster normal.
      b.  Union them with shapely to get the 2D sketch outline.
      c.  Sweep extrude depths d in increasing steps.  At each d,
          sample points on the LATERAL surface of the extrude (the
          cylinder/box "side").  Compute their distance to the GT
          mesh's surface.  The DEEPEST d where most lateral surface
          points are within ``surface_tol`` of GT defines the stable
          interval.
3.  Emit ONE DetectorOutput per cluster, with mesh = extrude at the
    stable depth.  These are LOCAL operations: a stack of independent
    extrudes whose UNION captures the body exactly when the body is
    composed of axis-aligned slabs.

Also emits a single ``slab_union`` candidate whose mesh is the
UNION of all stable extrudes — the CADFit iter-0 big-union step.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, _cg
from .extrude import DetectorOutput, chain_of, emit_union_program
from .planar_cluster import (
    _Cluster,
    _greedy_normal_clusters,
    _plane_basis,
    _project_cluster_to_2d,
)


def _resample_perimeter(coords: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon perimeter to ``n`` evenly-spaced points."""
    edges = np.roll(coords, -1, axis=0) - coords
    lens = np.linalg.norm(edges, axis=1)
    total = float(lens.sum())
    if total < 1e-9:
        return coords
    cs = np.cumsum(lens)
    out = []
    for s in np.linspace(0.0, total, n, endpoint=False):
        idx = int(np.searchsorted(cs, s, side="right"))
        if idx >= len(coords):
            idx = len(coords) - 1
        e_start = cs[idx - 1] if idx > 0 else 0.0
        t = (s - e_start) / max(lens[idx], 1e-9)
        a = coords[idx]; b = coords[(idx + 1) % len(coords)]
        out.append(a + t * (b - a))
    return np.asarray(out, dtype=np.float64)


def _stable_depth(gt_tree, gt_diag: float,
                  origin: np.ndarray, normal: np.ndarray,
                  u: np.ndarray, v: np.ndarray,
                  sketch_coords: np.ndarray,
                  d_max: float,
                  n_depths: int = 16,
                  n_perim: int = 40,
                  surface_tol_frac: float = 0.03,
                  ) -> float:
    """Find the CADFit stable extrude depth using a prebuilt KD-tree.

    ``gt_tree`` is a scipy cKDTree over sampled GT surface points.
    For a band of candidate heights, sample lateral-surface points of
    the extrude and query their distance to the nearest GT surface
    point (one-sided chamfer D(h)).  Following CADFit:

        h* = min { h : dD/dh(h) >= tau  AND  sup_{h'<h} D(h') <= eps }

    i.e. the first height where, after a low-CD plateau, CD starts to
    rise sharply -- the natural end of the slab.
    """
    coords = np.asarray(sketch_coords, dtype=np.float64)
    if len(coords) < 3:
        return 0.0
    perim = _resample_perimeter(coords, n_perim)   # (P, 2)

    tol = surface_tol_frac * gt_diag
    depths = np.linspace(d_max / n_depths, d_max, n_depths)
    # D(h) = mean distance of the lateral ring at height h to GT surface.
    D = np.zeros(n_depths)
    for i, d in enumerate(depths):
        shift = d * normal
        ring = (origin[None, :] + perim[:, 0:1] * u[None, :]
                + perim[:, 1:2] * v[None, :] + shift[None, :])
        dist, _ = gt_tree.query(ring, k=1)
        D[i] = float(np.mean(dist))

    # Find the plateau-then-rise transition.
    # eps = tolerance for "covered"; tau = rise threshold.
    eps = tol
    best_d = 0.0
    prev_D = D[0]
    for i in range(n_depths):
        # Up to this height, was coverage good?
        if D[i] <= eps:
            best_d = float(depths[i])
            prev_D = D[i]
            continue
        # D[i] > eps: coverage broke.  If it rose sharply from a good
        # plateau, stop here (the slab ended just before).
        if best_d > 0.0:
            break
        prev_D = D[i]
    return best_d


def _build_gt_tree(gt: trimesh.Trimesh, n_samples: int = 4000):
    """Build a KD-tree over sampled GT surface points (fast CD queries)."""
    from scipy.spatial import cKDTree
    try:
        pts, _ = trimesh.sample.sample_surface(gt, n_samples)
    except Exception:
        pts = np.asarray(gt.vertices, dtype=np.float64)
    return cKDTree(pts), float(np.linalg.norm(gt.extents))


def _emit_local_extrude_chain(coords, origin, u, normal_out, depth) -> str:
    """Emit a CadQuery chain for an arbitrary-plane polygon extrude.

    The 2D ``coords`` are in the (u, v) basis with v = normal_out x u
    (the same basis ``_plane_basis(normal_out)`` and the fast-mesh
    builder use).  We sketch on cq.Plane(origin, xDir=u, normal=normal_out)
    -- whose local Y equals normal_out x u = v, so (cx, cy) map directly
    -- and extrude by -depth (INWARD, matching normal_in = -normal_out).
    """
    O = np.asarray(origin, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    nn = np.asarray(normal_out, dtype=np.float64)
    plane = (f"cq.Plane(origin=cq.Vector({O[0]:.5f},{O[1]:.5f},{O[2]:.5f}), "
             f"xDir=cq.Vector({uu[0]:.5f},{uu[1]:.5f},{uu[2]:.5f}), "
             f"normal=cq.Vector({nn[0]:.5f},{nn[1]:.5f},{nn[2]:.5f}))")
    # Drop consecutive duplicate/near-duplicate points (densified
    # perimeters can have sub-epsilon spacing -> zero-length OCCT edges
    # that crash makeLine).
    raw = [(float(c[0]), float(c[1])) for c in coords]
    pts = [raw[0]]
    for x, y in raw[1:]:
        if abs(x - pts[-1][0]) > 1e-4 or abs(y - pts[-1][1]) > 1e-4:
            pts.append((x, y))
    # Also drop a near-duplicate closing point.
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-4 and abs(pts[0][1] - pts[-1][1]) < 1e-4:
        pts = pts[:-1]
    if len(pts) < 3:
        return ""
    # CADRecode-style emission (native cadquery Sketch API) on the custom plane.
    from .. import cadrecode_emit as cre
    chain = (f'cq.Workplane({plane}).sketch(){cre.poly_frag(pts)}'
             f'.finalize().extrude({-float(depth):.5f})')
    return chain


def _build_local_extrude_mesh(coords, origin, normal, u, v, depth
                              ) -> Optional[trimesh.Trimesh]:
    """Build a Trimesh by extruding the 2D sketch from origin along
    normal by ``depth``.
    """
    try:
        from shapely.geometry import Polygon
        from trimesh.creation import extrude_polygon
        shell = [(float(c[0]), float(c[1])) for c in coords]
        if len(shell) < 3 or depth <= 0:
            return None
        poly = Polygon(shell)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        m3 = extrude_polygon(poly, float(depth))
        if m3 is None or len(m3.faces) == 0:
            return None
        # Rotate trimesh-(x, y, z) into world (u, v, normal).
        R = np.column_stack([u, v, np.asarray(normal, dtype=np.float64)])
        m3.vertices = (R @ np.asarray(m3.vertices, dtype=np.float64).T).T
        m3.apply_translation(np.asarray(origin, dtype=np.float64))
        try:
            if float(m3.volume) < 0:
                m3.invert()
        except Exception:
            pass
        return m3
    except Exception:
        return None


def detect_local_extrudes(mesh: trimesh.Trimesh,
                          angle_thresh_deg: float = 8.0,
                          min_area_frac: float = 0.03,
                          surface_tol_frac: float = 0.03,
                          max_clusters: int = 12,
                          n_depth_steps: int = 16,
                          ) -> list[DetectorOutput]:
    """Run planar clustering, then for each cluster find the stable
    extrude depth and emit a LOCAL extrude.  Also emit one ``union``
    candidate whose mesh is the boolean union of all local extrudes
    (the CADFit iter-0 big-union step).
    """
    if mesh is None or len(mesh.faces) == 0:
        return []
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fa = np.asarray(mesh.area_faces, dtype=np.float64)
    fc = np.asarray(mesh.triangles_center, dtype=np.float64)
    if len(fn) == 0:
        return []
    clusters = _greedy_normal_clusters(fn, fa, fc, angle_thresh_deg,
                                       min_area_frac)
    if not clusters:
        return []

    bbox_diag = float(np.linalg.norm(mesh.extents))
    gt_tree, gt_diag = _build_gt_tree(mesh)
    outs: list[DetectorOutput] = []
    all_pieces: list[trimesh.Trimesh] = []
    all_chains: list[str] = []

    for ci, c in enumerate(clusters[:max_clusters]):
        coords = _project_cluster_to_2d(mesh, c)
        if coords is None or len(coords) < 3:
            continue
        u, v = _plane_basis(c.normal)
        # The cluster's normal points OUTWARD; extrude INWARD by negating.
        normal_in = -np.asarray(c.normal, dtype=np.float64)
        # Find stable depth.  d_max = full bbox diagonal as upper bound.
        d_max = bbox_diag
        d_stable = _stable_depth(
            gt_tree, gt_diag, c.origin, normal_in, u, v, coords,
            d_max=d_max, n_depths=n_depth_steps,
            surface_tol_frac=surface_tol_frac)
        if d_stable <= 0:
            continue
        m_piece = _build_local_extrude_mesh(coords, c.origin, normal_in,
                                            u, v, d_stable)
        if m_piece is None or len(m_piece.faces) == 0:
            continue
        all_pieces.append(m_piece)
        # Emit a renderable CadQuery chain for this piece (outward normal
        # for the workplane basis, negative depth = inward extrude).
        ch = _emit_local_extrude_chain(coords, c.origin, u, c.normal, d_stable)
        all_chains.append(ch)
        # Score low; this detector wins on IoU at render time, not on
        # static score.  Keep scores ≤ 0.4 so it doesn't displace the
        # other detectors' candidates from the proposer's top-k slot.
        outs.append(DetectorOutput(
            program=emit_union_program([ch]) if ch else "",
            score=0.30 + 0.02 * min(ci, 5),
            debug={
                "cluster_id": ci,
                "n_clusters_total": len(clusters),
                "stable_depth": d_stable,
                "normal": [round(float(x), 3) for x in c.normal],
                "area_frac": float(c.total_area / fa.sum()),
            },
            mesh=m_piece,
        ))

    # CADFit iter-0: union all local pieces into a single candidate.
    if len(all_pieces) >= 2:
        try:
            union_mesh = trimesh.boolean.union(all_pieces)
        except Exception:
            try:
                union_mesh = trimesh.util.concatenate(all_pieces)
            except Exception:
                union_mesh = None
        if union_mesh is not None and len(union_mesh.faces) > 0:
            outs.append(DetectorOutput(
                program=emit_union_program(all_chains),
                score=0.45,
                debug={
                    "kind": "union",
                    "n_pieces": len(all_pieces),
                },
                mesh=union_mesh,
            ))

    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
