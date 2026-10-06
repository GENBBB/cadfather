"""Planar face clustering detector — CADFit-style sketch source.

Algorithm
---------
1.  Compute face normals (unit) and face areas.
2.  Greedy cluster: start with the unclustered face of largest area as
    the seed.  Add any unclustered face whose normal makes an angle
    < ``angle_thresh_deg`` with the cluster's running area-weighted
    representative normal.  Recompute the rep after each addition.
3.  After each cluster's grow phase, recompute (origin, normal) from
    the area-weighted centroid and mean normal of its triangles.
4.  Drop clusters whose total area is below ``min_area_frac`` of the
    full mesh area.
5.  For each surviving cluster:
       - project the cluster's triangle vertices onto the plane,
       - run shapely union to get the outer 2D contour,
       - fit the best primitive (CIRCLE / RECT / POLYGON) via C++,
       - emit a CadQuery ``cq.Workplane(Plane(origin, normal)) ...
         extrude(h)`` candidate, with h = mesh axial extent along
         the cluster normal.

The detector is **prismatic-shape oriented**: boxes / plates / parts
with planar faces get strong clusters.  Curved shapes (cylinders, cones)
naturally fall back to revolve / silhouette extrude detectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from ..section_analyzer import _HAS_CPP_FITS, _cg
from .extrude import DetectorOutput

_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


# ---------------------------- Clustering ----------------------------------

@dataclass
class _Cluster:
    indices: np.ndarray         # face indices
    normal: np.ndarray          # area-weighted mean unit normal
    origin: np.ndarray          # area-weighted centroid
    total_area: float


def _greedy_normal_clusters(face_normals: np.ndarray,
                            face_areas: np.ndarray,
                            face_centroids: np.ndarray,
                            angle_thresh_deg: float,
                            min_area_frac: float,
                            ) -> list[_Cluster]:
    """Greedy clustering of faces by normal direction.  Returns clusters
    sorted by total_area descending."""
    # C++ fast path -- this greedy O(n^2) loop is the det hot path (~37% of
    # make_candidates time on high-face meshes).  Identical algorithm in C++.
    if _HAS_CPP_FITS and hasattr(_cg, "cluster_normals"):
        try:
            cl = _cg.cluster_normals(
                np.ascontiguousarray(face_normals, dtype=np.float64),
                np.ascontiguousarray(face_areas, dtype=np.float64),
                np.ascontiguousarray(face_centroids, dtype=np.float64),
                float(angle_thresh_deg), float(min_area_frac))
            return [_Cluster(np.asarray(idx, dtype=np.int64),
                             np.asarray(nrm, dtype=np.float64),
                             np.asarray(org, dtype=np.float64), float(area))
                    for (idx, nrm, org, area) in cl]
        except Exception:
            pass  # fall back to the pure-Python loop below
    n = len(face_normals)
    if n == 0:
        return []
    cos_thresh = math.cos(math.radians(angle_thresh_deg))
    total = float(face_areas.sum())
    if total < 1e-12:
        return []
    min_area = total * min_area_frac

    assigned = np.zeros(n, dtype=bool)
    clusters: list[_Cluster] = []
    order = np.argsort(-face_areas, kind="stable")  # largest-first; ties -> idx asc

    for seed in order:
        if assigned[seed]:
            continue
        # Initialize cluster with the seed face.
        members = [int(seed)]
        rep_normal = face_normals[seed].copy()
        rep_area = float(face_areas[seed])
        assigned[seed] = True

        # One pass over the unassigned faces in area-descending order.
        for j in order:
            if assigned[j]:
                continue
            cos_a = float(np.dot(face_normals[j], rep_normal))
            if cos_a < cos_thresh:
                continue
            # Accept: update area-weighted running mean.
            members.append(int(j))
            assigned[j] = True
            new_area = rep_area + float(face_areas[j])
            rep_normal = (rep_normal * rep_area
                          + face_normals[j] * float(face_areas[j])) / new_area
            nlen = float(np.linalg.norm(rep_normal))
            if nlen > 1e-12:
                rep_normal = rep_normal / nlen
            rep_area = new_area

        if rep_area < min_area:
            continue

        idx = np.asarray(members, dtype=np.int64)
        weights = face_areas[idx]
        origin = (face_centroids[idx] * weights[:, None]).sum(axis=0) / weights.sum()
        clusters.append(_Cluster(idx, rep_normal, origin, rep_area))

    clusters.sort(key=lambda c: c.total_area, reverse=True)
    return clusters


# ---------------------------- 2D projection -------------------------------

def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal basis (u, v) on the plane perpendicular to
    `normal`.  u is picked deterministically (rotate the smallest
    component of `normal` to avoid degeneracy)."""
    n = normal / max(np.linalg.norm(normal), 1e-12)
    # Pick a non-collinear seed:
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    return u, v


def _project_cluster_to_2d(mesh: trimesh.Trimesh, cluster: _Cluster,
                           ) -> Optional[np.ndarray]:
    """Project all cluster triangles to 2D in the plane (origin, normal),
    union them with shapely, return the largest polygon's outer coords."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    from .extrude import fast_projected_poly

    u, v = _plane_basis(cluster.normal)

    # Fast path: build a submesh of the cluster faces and use the
    # edge-based projection (origin at the cluster centroid so coords
    # are relative, matching the old per-triangle behaviour).
    union = None
    try:
        sub = mesh.submesh([cluster.indices], append=True)
        if sub is not None and len(sub.faces) > 0:
            union = fast_projected_poly(sub, cluster.normal, u, v,
                                        origin=cluster.origin)
    except Exception:
        union = None

    if union is None:
        # Fallback: per-triangle union (slow).
        tris = mesh.triangles[cluster.indices]
        rel = tris - cluster.origin[None, None, :]
        uu = np.einsum("kij,j->ki", rel, u)
        vv = np.einsum("kij,j->ki", rel, v)
        polys: list[Polygon] = []
        for k in range(uu.shape[0]):
            ring = list(zip(uu[k], vv[k]))
            try:
                p = Polygon(ring)
                if p.is_valid and p.area > 1e-9:
                    polys.append(p)
            except Exception:
                pass
        if not polys:
            return None
        union = unary_union(polys).buffer(0)
    if isinstance(union, MultiPolygon):
        union = max(union.geoms, key=lambda g: g.area)
    if union.is_empty:
        return None
    coords = np.asarray(union.exterior.coords)[:-1]
    if len(coords) < 3:
        return None
    return _densify_polyline(coords.astype(np.float64), target_n=32)


def _densify_polyline(coords: np.ndarray, target_n: int) -> np.ndarray:
    """Insert mid-edge points until we have at least ``target_n``.

    Why: a 4-corner rectangle has all corners on the same circle, so
    fit_all picks CIRCLE on data that is actually a rectangle.  Mid-edge
    samples lie ON the rectangle but OFF the corner-circle, breaking the
    tie in favour of RECT.
    """
    pts = list(map(tuple, coords))
    while len(pts) < target_n:
        new_pts = []
        for i in range(len(pts)):
            p = pts[i]
            q = pts[(i + 1) % len(pts)]
            new_pts.append(p)
            new_pts.append(((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5))
        pts = new_pts
    return np.asarray(pts, dtype=np.float64)


# ---------------------------- Emission ------------------------------------

def _emit_workplane_extrude(origin: np.ndarray, normal: np.ndarray,
                            sketch_kind: str, sketch_params: list,
                            depth: float) -> str:
    """Emit a CadQuery program that defines an arbitrary-plane workplane
    at `origin` with `normal`, sketches the fitted primitive, and
    extrudes by `depth` along the plane normal."""
    # Compute xDir (orthogonal to normal) deterministically so the
    # 2D sketch coordinates we emit match the basis we used.
    u, _ = _plane_basis(normal)
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])

    plane_expr = (
        f'cq.Plane(origin=cq.Vector({ox:.4f},{oy:.4f},{oz:.4f}),'
        f' xDir=cq.Vector({ux:.4f},{uy:.4f},{uz:.4f}),'
        f' normal=cq.Vector({nx:.4f},{ny:.4f},{nz:.4f}))'
    )

    # CADRecode-style emission (native cadquery Sketch API) on the custom plane.
    from .. import cadrecode_emit as cre
    if sketch_kind == "CIRCLE":
        cx, cy, r = sketch_params
        frag = cre.circle_frag(cx, cy, r)
    elif sketch_kind == "RECT":
        cx, cy, w, h, th = sketch_params
        if abs(th) < 1e-9:
            frag = cre.rect_frag(cx, cy, w, h)
        else:
            ct = math.cos(th); st = math.sin(th)
            hw, hh = w * 0.5, h * 0.5
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            rot = [(cx + ct * cx_ - st * cy_,
                    cy + st * cx_ + ct * cy_) for cx_, cy_ in corners]
            frag = cre.poly_frag(rot)
    else:  # POLYGON or fallback
        pts = list(zip(sketch_params[0::2], sketch_params[1::2]))
        frag = cre.poly_frag(pts)

    if not frag:
        return ""
    return (
        "import cadquery as cq\n"
        f"result = cq.Workplane({plane_expr}).sketch(){frag}.finalize().extrude({depth:.4f})\n"
    )


def _build_planar_cluster_mesh(origin: np.ndarray, normal: np.ndarray,
                               kind: str, params: list, depth: float
                               ) -> Optional[trimesh.Trimesh]:
    """Build a trimesh approximation of a planar-cluster extrude.

    Same logic as ``_emit_workplane_extrude`` but produces a Trimesh in
    Python without going through cadquery.  Uses the same plane basis
    ``_plane_basis`` so the 2D sketch coords map to the same 3D plane.
    """
    try:
        from shapely.geometry import Polygon
        from trimesh.creation import extrude_polygon
        if kind == "CIRCLE":
            cx, cy, r = params
            n = 64
            shell = [(cx + r * math.cos(2 * math.pi * i / n),
                      cy + r * math.sin(2 * math.pi * i / n))
                     for i in range(n)]
        elif kind == "RECT":
            cx, cy, w, h, th = params
            ct = math.cos(th); st = math.sin(th)
            hw, hh = w * 0.5, h * 0.5
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            shell = [(cx + ct * cx_ - st * cy_,
                      cy + st * cx_ + ct * cy_) for cx_, cy_ in corners]
        elif kind == "POLYGON":
            pts = list(zip(params[0::2], params[1::2]))
            if len(pts) < 3:
                return None
            shell = [(float(x), float(y)) for x, y in pts]
        else:
            return None
        poly = Polygon(shell)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        m3 = extrude_polygon(poly, float(depth))
        if m3 is None or len(m3.faces) == 0:
            return None

        # Rotate from extrude-along-Z to extrude-along-normal using the
        # same (u, v, n) basis as the emit path.
        u, v = _plane_basis(normal)
        n_unit = np.asarray(normal, dtype=np.float64)
        n_unit = n_unit / max(np.linalg.norm(n_unit), 1e-12)
        R = np.column_stack([u, v, n_unit])
        v3 = np.asarray(m3.vertices, dtype=np.float64)
        m3.vertices = (R @ v3.T).T
        m3.apply_translation(np.asarray(origin, dtype=np.float64))
        return m3
    except Exception:
        return None


# ---------------------------- Top-level -----------------------------------

def detect_planar_clusters(mesh: trimesh.Trimesh,
                           angle_thresh_deg: float = 8.0,
                           min_area_frac: float = 0.05,
                           depth_fracs: list[float] | None = None,
                           max_candidates: int = 8,
                           ) -> list[DetectorOutput]:
    """Cluster the mesh's faces by normal direction; for each major
    cluster emit a workplane + primitive sketch + extrude candidate.

    Parameters
    ----------
    angle_thresh_deg : float
        Two faces are merged into a cluster if their normals make an
        angle below this threshold.  Default 8 degrees (CADFit-ish).
    min_area_frac : float
        Discard clusters whose total area is less than this fraction
        of the full mesh surface area.
    depth_fracs : list[float] | None
        Per (cluster, primitive) sketch, emit one candidate per depth
        fraction.  Default [0.25, 0.5, 1.0] -- three depth variants, all
        <= 1.0 so the prism never exceeds the residual extent along the
        normal.  Each value multiplies the mesh extent along the cluster
        normal, floored at 0.02 * max bbox extent and capped at the full
        extent.
    """
    if not _HAS_CPP_FITS:
        return []
    if mesh is None or len(mesh.faces) == 0:
        return []
    if depth_fracs is None:
        depth_fracs = [0.25, 0.5, 1.0]

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas   = np.asarray(mesh.area_faces, dtype=np.float64)
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    if len(face_normals) == 0:
        return []

    clusters = _greedy_normal_clusters(
        face_normals, face_areas, face_centroids,
        angle_thresh_deg, min_area_frac)
    if not clusters:
        return []

    lo, hi = mesh.bounds
    max_extent = float(np.max(hi - lo))
    total_area = float(face_areas.sum())
    outs: list[DetectorOutput] = []

    for ci, c in enumerate(clusters[:max_candidates]):
        coords = _project_cluster_to_2d(mesh, c)
        if coords is None or len(coords) < 3:
            continue
        fits = _cg.section.fit_all(coords, -1.0)
        if not fits:
            continue
        top = fits[0]
        kind = str(top.kind).split(".")[-1]
        if kind not in ("CIRCLE", "RECT", "POLYGON"):
            continue
        # Mesh extent along the cluster normal sets the base scale.
        v = mesh.vertices - c.origin[None, :]
        s = v @ c.normal
        smin = float(s.min()); smax = float(s.max())
        base_depth = smax - smin
        if base_depth < 1e-6:
            base_depth = max_extent
            smin = -0.5 * base_depth
        # BUGFIX: anchor the workplane at the residual's NEAR face (s == smin)
        # so a one-directional +depth extrude fills [smin, smin+depth] INSIDE
        # the residual.  Previously the workplane sat at the cluster CENTROID
        # and extruded +depth one-directionally, so the prism stuck out a full
        # half-extent past the residual's far face (and with frac>1.0 even
        # further) -> pieces 2.7-4x the residual volume that gouge valid
        # geometry on a cut and overhang the part on a union.
        anchor = c.origin + smin * c.normal

        for frac in depth_fracs:
            # Clamp so the piece never exceeds the residual extent along the
            # normal (floored to a minimum thickness for very thin residuals).
            depth = max(min(base_depth * float(frac), base_depth),
                        0.02 * max_extent)
            program = _emit_workplane_extrude(anchor, c.normal, kind,
                                              list(top.params), depth)
            if not program:
                continue
            fast_mesh = _build_planar_cluster_mesh(anchor, c.normal, kind,
                                                   list(top.params), depth)
            # Bias toward smaller depths when the primitive fit was good
            # (so the score still surfaces the best primitive first while
            # multiple depths compete on near-equal footing).
            # Normalize the C++ score (LOWER better) into HIGHER-better
            # via exp(-score), then weight by cluster area fraction so
            # the biggest planar feature wins ties.  Penalize depth
            # drift from 1.0 lightly.
            base = math.exp(-float(top.score))
            area_weight = c.total_area / max(total_area, 1e-9)
            depth_penalty = 0.05 * abs(float(frac) - 1.0)
            outs.append(DetectorOutput(
                program=program,
                score=float(base * area_weight - depth_penalty),
                debug={
                    "cluster_id": ci,
                    "kind": kind,
                    "total_area_frac": c.total_area / total_area,
                    "normal": [round(float(x), 4) for x in c.normal],
                    "primitive_residual": float(top.residual),
                    "depth": depth,
                    "depth_frac": float(frac),
                },
                mesh=fast_mesh,
            ))

    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
