"""Silhouette-extrude detector — adapted from CADAgent-Evolution
detect_silhouette_extrude.py, refactored as a library function so we can
import it instead of shelling out.

Project the mesh onto each principal plane (YZ, XZ, XY), union the 2D
footprints, simplify the outer ring + interior holes, extrude back along
the perpendicular axis.

Returns a list of (program, score, debug) - one entry per principal
axis tried.  The caller picks how many to keep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .. import cadrecode_emit as cre


_AXIS_TO_WP = {0: "YZ", 1: "XZ", 2: "XY"}

# Above this triangle count, the per-triangle shapely-union *fallback*
# (used only when trimesh's fast edge projection fails) is skipped: it
# costs ~0.7s on a 17k-face mesh for a speculative candidate that is
# essentially never the best fit.  The fallback is designed for the
# decimated-residual regime (~6k tris).
_FALLBACK_TRI_CAP = 8000


@dataclass
class DetectorOutput:
    program: str           # standalone CadQuery program (starts with imports)
    score: float           # higher = better fit
    debug: dict = field(default_factory=dict)
    # Optional pre-built trimesh approximation of `program`'s output.
    # When set, the eval can skip the cadquery render round-trip
    # (saves ~1-3 s per candidate).  For multi-step iterations the
    # caller still must render the cadquery code; this field is purely
    # an iter-1 fast path.
    mesh: Optional[trimesh.Trimesh] = None


# --------------------------------------------------------------------------
# Shared CadQuery-program helpers (used by the local_* detectors so they
# emit RENDERABLE, parameter-optimizable code -- not just a fast mesh).
# --------------------------------------------------------------------------

def chain_of(program: str) -> str:
    """Extract the ``<chain>`` expression from a single-result program of
    the form ``import cadquery as cq\\nresult = <chain>``.  Returns the
    chain (possibly multi-line with backslash continuations)."""
    if not program:
        return ""
    marker = "result = "
    idx = program.find(marker)
    if idx < 0:
        return ""
    return program[idx + len(marker):].strip()


def emit_union_program(chains: list[str], ops: Optional[list[str]] = None) -> str:
    """Combine per-piece CadQuery chains into ONE single-statement program.

    Produces ``result = (chain0).union((chain1)).cut((chain2))...`` as a
    SINGLE assignment (no intermediate ``_pN`` temporaries) so the block
    emitter can append it as one line ``r = r.<op>(<expr>)`` -- matching
    the one-line-per-step stepwise convention.

    ``ops[i]`` (default "union") chooses how piece i (>=1) combines.
    """
    chains = [c for c in chains if c]
    if not chains:
        return ""
    expr = f"({chains[0]})"
    for i in range(1, len(chains)):
        op = (ops[i] if ops and i < len(ops) else "union")
        meth = "cut" if op == "cut" else "union"
        expr = f"{expr}.{meth}(({chains[i]}))"
    return "import cadquery as cq\nresult = " + expr + "\n"


def _decimate(coords, max_pts: int):
    if len(coords) <= max_pts:
        return coords
    step = len(coords) / max_pts
    return [coords[int(i * step)] for i in range(max_pts)]


def _polygon_to_coords_and_holes(poly: Polygon, simplify: float, max_points: int):
    """Convert one shapely Polygon -> (outer_coords, area, holes) tuple.

    Returns None if the polygon is too small / degenerate to emit.
    """
    poly = poly.simplify(simplify, preserve_topology=True)
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        return None
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) < 3:
        return None
    coords = _decimate(coords, max_points)

    holes = []
    for interior in poly.interiors:
        hc = list(interior.coords)[:-1]
        if len(hc) < 3:
            continue
        hpoly = Polygon(hc)
        if hpoly.length < 1e-6:
            continue
        iso = 4 * math.pi * hpoly.area / (hpoly.length ** 2)
        if iso > 0.95 and len(hc) >= 8:
            cx = sum(p[0] for p in hc) / len(hc)
            cy = sum(p[1] for p in hc) / len(hc)
            r = math.sqrt(hpoly.area / math.pi)
            holes.append({"type": "circle", "cx": cx, "cy": cy, "r": r})
        else:
            holes.append({"type": "polygon", "coords": _decimate(hc, max_points)})

    return coords, float(poly.area), holes


def fast_projected_poly(mesh: trimesh.Trimesh, normal, u, v, origin=None):
    """Project ``mesh`` along ``normal`` and return a shapely
    (Multi)Polygon expressed in the explicit (u, v) basis (origin at
    ``origin``, default world 0).

    Uses ``trimesh.path.polygons.projected`` (edge-based, ~50x faster
    than per-triangle shapely union) then affine-maps trimesh's plane
    basis into the caller's (u, v) basis.  Returns None on failure so
    the caller can fall back.
    """
    from shapely.affinity import affine_transform
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    u = np.asarray(u, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    if not bool(getattr(mesh, "is_watertight", False)):
        return None   # caller falls back to per-triangle (robust on residuals)
    try:
        from trimesh.path import polygons as _tpoly
        import trimesh.geometry as _tg
        poly = _tpoly.projected(mesh, normal=n, origin=o)
        if poly is None or poly.is_empty:
            return None
        T = _tg.plane_transform(origin=o.tolist(), normal=n.tolist())

        def w2t(p3):
            return trimesh.transform_points(
                np.asarray([p3], dtype=np.float64), T)[0][:2]
        q0 = w2t(o); qa = w2t(o + u); qb = w2t(o + v)
        B = np.column_stack([qa - q0, qb - q0])
        A = np.linalg.inv(B)
        t = -A @ q0
        return affine_transform(
            poly, [A[0, 0], A[0, 1], A[1, 0], A[1, 1], t[0], t[1]])
    except Exception:
        return None


def _projected_outline(m: trimesh.Trimesh, axis_idx: int):
    """Return the 2D silhouette of ``m`` projected along the cardinal
    axis ``axis_idx``, in our (other0, other1) world-coordinate basis,
    as a shapely (Multi)Polygon.

    Uses ``trimesh.path.polygons.projected`` (boundary-edge based) which
    is ~50x faster than per-triangle shapely union, then re-expresses
    the result into our (world[other0], world[other1]) basis via a small
    affine map so the downstream emit/mesh builders stay correct.
    """
    from shapely.geometry import Polygon
    from shapely.affinity import affine_transform
    from shapely.ops import unary_union
    n = np.zeros(3); n[axis_idx] = 1.0
    other = [i for i in range(3) if i != axis_idx]

    poly = None
    # trimesh.projected is edge-based and assumes a clean watertight
    # surface; on non-watertight residual meshes (boolean / marching-
    # cubes output) it can drop components or fill holes, hurting IoU.
    # Use it only for watertight meshes; residuals fall back to the
    # robust per-triangle union (they are decimated to ~6k, so cheap).
    if bool(getattr(m, "is_watertight", False)):
        try:
            from trimesh.path import polygons as _tpoly
            poly = _tpoly.projected(m, normal=n)
        except Exception:
            poly = None

    if poly is None or poly.is_empty:
        # Fallback: per-triangle union (slow, rare).  Cap it -- unioning a
        # huge triangle soup dominates the whole pipeline; the fallback is
        # meant for decimated residuals (~6k tris), not raw 17k+ meshes.
        tris = np.asarray(m.triangles, dtype=np.float64)
        if len(tris) > _FALLBACK_TRI_CAP:
            return None
        polys = []
        for t in tris:
            ring = [(float(t[k][other[0]]), float(t[k][other[1]]))
                    for k in range(3)]
            try:
                p = Polygon(ring)
                if p.is_valid and p.area > 1e-9:
                    polys.append(p)
            except Exception:
                pass
        if not polys:
            return None
        return unary_union(polys).buffer(0)

    # Map trimesh's 2D basis -> our (world[other0], world[other1]) basis.
    try:
        import trimesh.geometry as _tg
        T = _tg.plane_transform(origin=[0.0, 0.0, 0.0], normal=n)

        def w2t(p3):
            return trimesh.transform_points(np.asarray([p3], dtype=np.float64),
                                            T)[0][:2]
        ea = np.zeros(3); ea[other[0]] = 1.0
        eb = np.zeros(3); eb[other[1]] = 1.0
        q0 = w2t([0.0, 0.0, 0.0]); qa = w2t(ea); qb = w2t(eb)
        B = np.column_stack([qa - q0, qb - q0])      # 2x2: my-basis -> trimesh
        A = np.linalg.inv(B)                         # trimesh -> my-basis
        t = -A @ q0
        # shapely affine_transform takes [a, b, d, e, xoff, yoff] for
        # x' = a*x + b*y + xoff ; y' = d*x + e*y + yoff
        poly = affine_transform(
            poly, [A[0, 0], A[0, 1], A[1, 0], A[1, 1], t[0], t[1]])
    except Exception:
        pass
    return poly


def _project_silhouette_components(m: trimesh.Trimesh, axis_idx: int,
                                   simplify: float, max_points: int,
                                   min_component_area_frac: float = 0.005,
                                   ) -> list[tuple]:
    """Return one (coords, area, holes) tuple per connected component of
    the silhouette union, sorted by area descending.

    Components smaller than ``min_component_area_frac`` of the total
    silhouette area are dropped (these are usually triangulation artifacts).
    """
    # Fast path: trimesh.path.polygons.projected computes the silhouette
    # outline directly from the mesh boundary edges -- ~50x faster than
    # building one shapely Polygon per triangle and unary_union'ing a
    # million of them (the old approach dominated the whole pipeline).
    union = _projected_outline(m, axis_idx)
    if union is None or union.is_empty:
        return []
    # Normalize to a list of disjoint Polygons.
    if isinstance(union, MultiPolygon):
        candidates = sorted(union.geoms, key=lambda p: p.area, reverse=True)
    elif isinstance(union, Polygon):
        candidates = [union]
    else:
        return []
    total_area = sum(p.area for p in candidates)
    if total_area <= 0:
        return []
    min_area = total_area * min_component_area_frac
    out = []
    for poly in candidates:
        if poly.area < min_area:
            continue
        triple = _polygon_to_coords_and_holes(poly, simplify, max_points)
        if triple is not None:
            out.append(triple)
    return out


def _emit_program(coords, axis_idx, lo, hi, holes=None,
                  min_thickness_frac: float = 0.02) -> str:
    """Emit a standalone extrude program.

    ``min_thickness_frac`` is a *relative* floor on the extrude depth as a
    fraction of the largest bbox extent.  This avoids zero-thickness
    extrudes for paper-thin residuals without hard-coding a unit scale
    (the original CADAgent-Evolution version capped at 4.0 units which
    over-extrudes by ~50x on [0,1]-normalized meshes).
    """
    wp = _AXIS_TO_WP[axis_idx]
    raw_thickness = float(hi[axis_idx] - lo[axis_idx])
    max_extent = float(max(hi - lo))
    thickness = max(raw_thickness, min_thickness_frac * max_extent)
    delta = (thickness - raw_thickness) / 2.0
    # Axial shift folded into the workplane ORIGIN (no trailing .translate).
    o = [0.0, 0.0, 0.0]
    o[axis_idx] = float(lo[axis_idx]) - delta
    # CADRecode-style emission (native cadquery Sketch API). Circle holes
    # become inner wires (mode='s') in the SAME sketch -> one clean statement,
    # no separate cut / thicker-hole hack.  Polygon holes stay as sketch-style
    # cuts (extruded thicker for a clean through-cut).
    circle_holes = [("circle", h["cx"], h["cy"], h["r"])
                    for h in (holes or []) if h.get("type") == "circle"]
    poly_holes = [h for h in (holes or [])
                  if h.get("type") == "polygon" and len(h.get("coords", [])) >= 3]

    chain = cre.poly_extrude(wp, o, coords, thickness, holes=circle_holes)
    if not chain:
        return ""
    lines = ["import cadquery as cq", f"result = {chain}"]

    if poly_holes:
        hole_pad = max(thickness * 0.1, max_extent * 0.005)
        hole_thickness = thickness + 2 * hole_pad
        oh = list(o); oh[axis_idx] -= hole_pad
        for hole in poly_holes:
            hchain = cre.poly_extrude(wp, oh, hole["coords"], hole_thickness)
            if hchain:
                lines.append(f"result = result.cut({hchain})")
    return "\n".join(lines)


def _pca_principal_axes(m: trimesh.Trimesh) -> list[np.ndarray]:
    """Return up to 3 principal-component axes of the surface vertices
    (unit vectors).  These are candidate "extrusion directions" for
    parts that aren't axis-aligned: a long thin slab's longest PC is
    its extrude axis."""
    try:
        v = np.asarray(m.vertices, dtype=np.float64)
        v = v - v.mean(axis=0, keepdims=True)
        cov = (v.T @ v) / max(len(v), 1)
        w, vec = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        return [vec[:, i] for i in order]
    except Exception:
        return []


def _pca_aligned_silhouette(m: trimesh.Trimesh, axis: np.ndarray,
                            simplify: float, max_points: int,
                            min_component_area_frac: float
                            ) -> list[tuple]:
    """Like _project_silhouette_components but along an arbitrary axis.

    Builds a basis (u, v, n) where n = axis, projects mesh vertices to
    (u, v), unions triangles, returns component polygons.  The caller
    converts back to 3D via the same basis.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    n = axis / max(np.linalg.norm(axis), 1e-12)
    seed = (np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9
            else np.array([0.0, 1.0, 0.0]))
    u = np.cross(n, seed); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)

    # Fast edge-based projection in our (u, v) basis.
    union = fast_projected_poly(m, n, u, v, origin=np.zeros(3))
    if union is None:
        # Fallback: per-triangle union (slow).  Cap it: unioning a huge
        # triangle soup (e.g. a 17k-face watertight mesh whose off-axis
        # projection trimesh can't handle) costs ~0.7s for a *speculative*
        # PCA candidate that is rarely useful.  The fallback exists for the
        # decimated-residual regime (~6k tris); above that, skip PCA.
        tris = np.asarray(m.triangles, dtype=np.float64)
        if len(tris) > _FALLBACK_TRI_CAP:
            return []
        uu = np.einsum("nij,j->ni", tris, u)
        vv = np.einsum("nij,j->ni", tris, v)
        polys: list[Polygon] = []
        for i in range(len(tris)):
            ring = [(float(uu[i, 0]), float(vv[i, 0])),
                    (float(uu[i, 1]), float(vv[i, 1])),
                    (float(uu[i, 2]), float(vv[i, 2]))]
            try:
                p = Polygon(ring)
                if p.is_valid and p.area > 1e-9:
                    polys.append(p)
            except Exception:
                pass
        if not polys:
            return []
        union = unary_union(polys).buffer(0)
    if union.is_empty:
        return []
    if isinstance(union, MultiPolygon):
        candidates = sorted(union.geoms, key=lambda p: p.area, reverse=True)
    elif isinstance(union, Polygon):
        candidates = [union]
    else:
        return []
    total_area = sum(p.area for p in candidates)
    if total_area <= 0:
        return []
    min_area = total_area * min_component_area_frac
    out = []
    for poly in candidates:
        if poly.area < min_area:
            continue
        triple = _polygon_to_coords_and_holes(poly, simplify, max_points)
        if triple is not None:
            out.append(triple)
    # Also compute the axial extent for the caller.
    verts = np.asarray(m.vertices, dtype=np.float64)
    proj = verts @ n
    z_min, z_max = float(proj.min()), float(proj.max())
    return [(c, a, h, z_min, z_max) for (c, a, h) in out]


def _build_pca_extrude_mesh(coords, axis_n, u_vec, v_vec, z_min, z_max,
                            holes, min_thickness_frac: float = 0.02
                            ) -> Optional[trimesh.Trimesh]:
    """Build a Trimesh by extruding 2D polygon along arbitrary axis."""
    try:
        from shapely.geometry import Polygon
        from trimesh.creation import extrude_polygon
        thickness = max(float(z_max - z_min),
                        min_thickness_frac * float(z_max - z_min))
        if thickness <= 1e-6:
            return None
        shell = [(float(c[0]), float(c[1])) for c in coords]
        if len(shell) < 3:
            return None
        hole_rings = []
        if holes:
            for hole in holes:
                if hole.get("type") == "circle":
                    cx, cy, r = hole["cx"], hole["cy"], hole["r"]
                    n_circ = 32
                    hole_rings.append([
                        (cx + r * math.cos(2 * math.pi * i / n_circ),
                         cy + r * math.sin(2 * math.pi * i / n_circ))
                        for i in range(n_circ)])
                elif hole.get("type") == "polygon":
                    hc = hole["coords"]
                    if len(hc) >= 3:
                        hole_rings.append([(float(c[0]), float(c[1]))
                                           for c in hc])
        poly = Polygon(shell, holes=hole_rings)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        m3 = extrude_polygon(poly, thickness)
        if m3 is None or len(m3.faces) == 0:
            return None
        # extrude_polygon makes vertices in (u, v, w) where w is the
        # extrude axis (local Z, 0 to thickness).  Rotate to align local
        # axes with (u_vec, v_vec, axis_n).
        R = np.column_stack([u_vec, v_vec, axis_n])
        m3.vertices = (R @ np.asarray(m3.vertices, dtype=np.float64).T).T
        # Translate along axis_n so the local-z=0 plane is at z_min.
        m3.apply_translation(z_min * axis_n)
        if float(m3.volume) < 0:
            m3.invert()
        return m3
    except Exception:
        return None


def detect_silhouette_extrude(m: trimesh.Trimesh,
                              simplify: float = 0.01,
                              max_points: int = 200,
                              min_component_area_frac: float = 0.005,
                              try_pca: bool = True,
                              ) -> list[DetectorOutput]:
    """Try all three principal axes; for each axis, emit one
    DetectorOutput **per connected component** of the silhouette union.

    This is what lets a multi-feature residual (e.g. 4 disjoint pads) be
    captured as 4 separate candidate blocks instead of one giant
    bounding silhouette.

    Components smaller than ``min_component_area_frac`` of the total
    per-axis silhouette area are dropped as triangulation noise.

    Score:  -|log( comp_area * thickness / mesh.volume )| —
    closer to 0 means the component's extrude volume best matches the
    mesh volume.  Smaller components get worse scores (further from 1.0
    fit-ratio); they still compete in the candidate pool.
    """
    lo, hi = m.bounds
    try:
        vol = float(abs(m.volume))
    except Exception:
        vol = 0.0

    outs: list[DetectorOutput] = []
    for axis_idx in (0, 1, 2):
        comps = _project_silhouette_components(
            m, axis_idx, simplify, max_points, min_component_area_frac)
        for comp_idx, (coords, area, holes) in enumerate(comps):
            thickness = float(hi[axis_idx] - lo[axis_idx])
            ratio = (area * thickness) / vol if vol > 1e-9 else float("inf")
            # Normalized score: 1.0 at fit_ratio == 1 (perfect extrude),
            # decays smoothly toward 0 as the ratio departs from 1.
            # Equivalent to min(ratio, 1/ratio); range [0, 1].
            score = math.exp(-abs(math.log(max(ratio, 1e-9))))
            program = _emit_program(coords, axis_idx, lo, hi, holes)
            fast_mesh = _build_extrude_mesh(coords, axis_idx, lo, hi, holes)
            outs.append(DetectorOutput(
                program=program,
                score=score,
                debug={
                    "axis": _AXIS_TO_WP[axis_idx],
                    "component": comp_idx,
                    "n_components": len(comps),
                    "n_points": len(coords),
                    "n_holes": len(holes),
                    "fit_ratio": ratio,
                    "comp_area": area,
                    "thickness": thickness,
                },
                mesh=fast_mesh,
            ))

    # PCA-aligned extrude: only fires when the body's principal direction
    # is meaningfully off the cardinal axes (max-cos < 0.95).
    if try_pca:
        pca_axes = _pca_principal_axes(m)
        for pca_idx, pca_axis in enumerate(pca_axes[:1]):  # only longest PC
            max_cos = float(max(abs(pca_axis[0]), abs(pca_axis[1]),
                                abs(pca_axis[2])))
            if max_cos > 0.95:
                continue  # already aligned with a cardinal axis
            triples = _pca_aligned_silhouette(
                m, pca_axis, simplify, max_points, min_component_area_frac)
            for comp_idx, (coords, area, holes, z_min, z_max) in enumerate(triples):
                thickness = float(z_max - z_min)
                ratio = (area * thickness) / vol if vol > 1e-9 else float("inf")
                score = math.exp(-abs(math.log(max(ratio, 1e-9))))
                # Reuse the cardinal-axis emit for now (cadquery-rendered
                # PCA workplane would need cq.Plane).  Skip program; only
                # provide the fast mesh which is what the eval uses.
                program = "# PCA extrude — no cadquery emit\n"
                # Build basis to feed into fast-mesh constructor.
                pca = pca_axis / max(np.linalg.norm(pca_axis), 1e-12)
                seed = (np.array([1.0, 0.0, 0.0]) if abs(pca[0]) < 0.9
                        else np.array([0.0, 1.0, 0.0]))
                u = np.cross(pca, seed); u /= np.linalg.norm(u) + 1e-12
                v = np.cross(pca, u)
                fast_mesh = _build_pca_extrude_mesh(coords, pca, u, v,
                                                    z_min, z_max, holes)
                outs.append(DetectorOutput(
                    program=program,
                    score=score * 0.95,  # slight discount vs cardinal
                    debug={
                        "axis": f"PCA{pca_idx}",
                        "component": comp_idx,
                        "fit_ratio": ratio,
                        "comp_area": area,
                        "thickness": thickness,
                        "pca_axis": [round(float(x), 3) for x in pca],
                    },
                    mesh=fast_mesh,
                ))
    return outs


def _build_extrude_mesh(coords, axis_idx: int, lo, hi, holes,
                        min_thickness_frac: float = 0.02
                        ) -> Optional[trimesh.Trimesh]:
    """Build a trimesh directly from the silhouette polygon (no cadquery).

    Same geometric semantics as ``_emit_program`` but the output is a
    Python-resident Trimesh suitable for IoU compute -- 1000x faster
    than rendering CadQuery to STL and re-loading.
    """
    try:
        from shapely.geometry import Polygon
        from trimesh.creation import extrude_polygon
        raw_thickness = float(hi[axis_idx] - lo[axis_idx])
        max_extent = float(max(hi - lo))
        thickness = max(raw_thickness, min_thickness_frac * max_extent)

        # Build a shapely polygon with holes.
        shell = [(float(c[0]), float(c[1])) for c in coords]
        if len(shell) < 3:
            return None
        hole_rings: list[list[tuple[float, float]]] = []
        if holes:
            for hole in holes:
                if hole.get("type") == "circle":
                    cx, cy, r = hole["cx"], hole["cy"], hole["r"]
                    import math as _math
                    n = 32
                    hole_rings.append([
                        (cx + r * _math.cos(2 * _math.pi * i / n),
                         cy + r * _math.sin(2 * _math.pi * i / n))
                        for i in range(n)])
                elif hole.get("type") == "polygon":
                    hc = hole["coords"]
                    if len(hc) >= 3:
                        hole_rings.append([(float(c[0]), float(c[1]))
                                           for c in hc])
        try:
            poly = Polygon(shell, holes=hole_rings)
        except Exception:
            poly = Polygon(shell)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None

        m3 = extrude_polygon(poly, thickness)
        if m3 is None or len(m3.faces) == 0:
            return None

        # extrude_polygon extrudes along Z in [0, thickness].  Map to
        # the chosen axis by reading the vertices through a permutation
        # whose det = +1 (preserves outward winding).
        #   axis 0:  permutation [2, 0, 1] is EVEN (det=+1)  -> no flip
        #   axis 1:  permutation [0, 2, 1] is ODD  (det=-1)  -> flip
        delta = (thickness - raw_thickness) / 2.0
        if axis_idx == 0:
            m3.vertices = m3.vertices[:, [2, 0, 1]]
            t = [float(lo[0]) - delta, 0.0, 0.0]
        elif axis_idx == 1:
            m3.vertices = m3.vertices[:, [0, 2, 1]]
            m3.invert()
            t = [0.0, float(lo[1]) - delta, 0.0]
        else:
            t = [0.0, 0.0, float(lo[2]) - delta]
        m3.apply_translation(t)
        # Safety: detect inverted winding via negative volume.
        try:
            if float(m3.volume) < 0:
                m3.invert()
        except Exception:
            pass
        return m3
    except Exception:
        return None
