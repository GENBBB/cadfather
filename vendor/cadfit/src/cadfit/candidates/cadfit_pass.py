"""CADFit-style single-pass reconstruction (arXiv 2605.01171).

Implements the paper's three core mechanisms that our detector set lacked:

1. PROFILES = exact mesh section loops WITH HOLES (Alg. 4-7 / App. I):
   slicing planes from (a) planar face clusters at +/-delta offsets and
   (b) axis-aligned planes at quantiles of the extent; closed intersection
   loops projected to the plane; outer boundaries grouped with contained
   interior holes (trimesh's ``polygons_full`` already gives shapely
   polygons with interiors).

2. DEPTH by one-sided chamfer SWEEP (Alg. 8 / App. K): slide the profile's
   boundary ring along the plane normal, measure mean sq. distance to the
   target surface per level, keep maximal "stable" runs (D <= eps) --
   i.e. extrusion intervals that do not overshoot the target; emit the
   largest interval (and the largest containing t=0) as candidates.

3. ASSEMBLY by union-of-all + BACKWARD MARGINAL-IoU PRUNING (Alg. 2):
   voxelize every candidate on a shared grid, start from the union of all,
   repeatedly remove the candidate whose removal does not decrease
   volumetric IoU (taking the best removal each round), stop when every
   removal hurts.  No greedy forward selection -> no init lock-in.

Pure trimesh/shapely/numpy (no cadquery needed): outputs fast meshes +
per-candidate plane/profile data; CadQuery emission can be wired on top.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shapely
import trimesh

from cadfit.candidates.kdtree import KDTree as cKDTree
from cadfit.voxel_lattice import voxelize_lattice


# --------------------------------------------------------------------------
# 1. Profile extraction: exact section loops with holes
# --------------------------------------------------------------------------

@dataclass
class Profile:
    polygon: object           # shapely Polygon (exterior + interiors), 2D plane coords
    to_3d: np.ndarray         # 4x4: (x, y, t) plane coords -> world
    source: str               # "cluster" | "axis"
    area: float
    sig: tuple = field(default=())


def _section_profiles(mesh: trimesh.Trimesh, origin, normal, source: str,
                      min_area: float) -> list[Profile]:
    """Slice mesh with one plane -> shapely polygons WITH holes -> Profiles."""
    out = []
    try:
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec is None:
            return out
        p2d, to_3d = sec.to_planar()
        if p2d is None:
            return out
        for poly in p2d.polygons_full:        # outer loops grouped w/ interiors
            if poly is None or poly.is_empty or poly.area < min_area:
                continue
            out.append(Profile(polygon=poly, to_3d=np.asarray(to_3d, float),
                               source=source, area=float(poly.area)))
    except Exception:
        pass
    return out


def extract_profiles(mesh: trimesh.Trimesh,
                     n_quantiles: int = 7,
                     max_cluster_planes: int = 20,
                     cluster_angle_deg: float = 8.0,
                     cluster_min_area_frac: float = 0.02,
                     offset_frac: float = 0.01,
                     min_area_frac: float = 5e-4,
                     ) -> list[Profile]:
    """Planar-cluster planes (+/- delta) + axis-aligned quantile planes."""
    lo, hi = mesh.bounds
    ext = float(np.max(hi - lo))
    min_area = min_area_frac * ext * ext
    delta = offset_frac * ext
    profiles: list[Profile] = []

    # (a) planar face clusters
    try:
        from .detectors.planar_cluster import _greedy_normal_clusters
        clusters = _greedy_normal_clusters(
            np.asarray(mesh.face_normals, float),
            np.asarray(mesh.area_faces, float),
            np.asarray(mesh.triangles_center, float),
            cluster_angle_deg, cluster_min_area_frac)
    except Exception:
        clusters = []
    for c in clusters[:max_cluster_planes]:
        for sgn in (-1.0, +1.0):
            o = np.asarray(c.origin, float) + sgn * delta * np.asarray(c.normal, float)
            profiles += _section_profiles(mesh, o, c.normal, "cluster", min_area)

    # (b) axis-aligned quantile planes
    qs = np.concatenate([[0.03, 0.97], np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]])
    for ax in range(3):
        n = np.zeros(3); n[ax] = 1.0
        for q in qs:
            o = lo.copy().astype(float)
            o[ax] = lo[ax] + q * (hi[ax] - lo[ax])
            profiles += _section_profiles(mesh, o, n, "axis", min_area)

    # dedup: same plane normal + similar area + similar world centroid
    seen, uniq = set(), []
    for p in profiles:
        nrm = p.to_3d[:3, 2]
        cen2 = np.array(p.polygon.centroid.coords[0])
        cen3 = (p.to_3d @ np.array([cen2[0], cen2[1], 0.0, 1.0]))[:3]
        sig = (tuple(np.round(np.abs(nrm), 1)),
               round(p.area / max(min_area, 1e-9), 0),
               tuple(np.round(cen3 / max(ext * 0.02, 1e-9), 0)))
        if sig in seen:
            continue
        seen.add(sig); p.sig = sig; uniq.append(p)
    return uniq


# --------------------------------------------------------------------------
# 2. Chamfer-sweep depth selection
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    mesh: trimesh.Trimesh
    profile: Profile
    t0: float
    t1: float
    ring_err: float


@dataclass
class PrimitiveCandidate:
    """A non-extrude op (revolve / sweep / loft) carried through the assembler
    and emitter by its MESH (for occupancy IoU) plus a ready CadQuery program.
    ``profile`` is None so extrude-specific paths (_occ_extrude, tip_extensions,
    _is_circle, the emit anchoring grammar) recognise and skip it."""
    mesh: trimesh.Trimesh
    program: str
    kind: str                      # "revolve" | "sweep" | "loft"
    profile: object = None
    t0: float = 0.0
    t1: float = 0.0
    ring_err: float = 0.0


def _boundary_ring(poly, n_pts: int = 160) -> np.ndarray:
    """Sample 2D points along the polygon exterior + interiors (arc-length)."""
    rings = [poly.exterior] + list(poly.interiors)
    total = sum(r.length for r in rings)
    pts = []
    for r in rings:
        length = r.length
        k = max(8, int(round(n_pts * length / max(total, 1e-9))))
        # one GEOS call per ring: bit-identical points to a per-point r.interpolate
        pts.append(shapely.get_coordinates(shapely.line_interpolate_point(r, length * np.arange(k) / k)))
    return np.concatenate(pts).astype(float)


def sweep_candidates(profile: Profile, gtree: cKDTree, ext: float,
                     n_levels: int = 96,
                     eps_frac: float = 0.012,
                     min_span_frac: float = 0.02,
                     gt_mesh: Optional[trimesh.Trimesh] = None,
                     gt_vox=None,
                     ) -> list[Candidate]:
    """Slide the profile boundary ring along the plane normal; keep maximal
    stable runs (ring distance <= eps) as extrusion intervals."""
    ring2 = _boundary_ring(profile.polygon)
    if len(ring2) < 8:
        return []
    T = profile.to_3d
    eps = (eps_frac * ext) ** 2
    # CIRCULAR profiles get a radius-scaled tolerance: a global eps (~2.4u of
    # drift) is WIDER than the 1-2u radius steps of multi-diameter rods, so the
    # sweep blurs straight through the step and the tip segments never form.
    # 10% of r resolves the step while tolerating mesh noise.
    fc = _fit_circle_2d(ring2)
    if fc is not None and fc[3] < 1.0:
        eps = min(eps, max((0.10 * fc[2]) ** 2, 1.0))

    # ring at level t in world coords:  T @ (x, y, t, 1) = W0 + t*dW (affine
    # in t), so ALL sweep levels can be scored with ONE KD query instead of
    # one query per level (ring_d dominated single-pass runtime).
    W0 = ring2 @ T[:3, :2].T + T[:3, 3]
    dW = T[:3, 2]

    def ring_d(t: float) -> float:
        dist, _ = gtree.query(W0 + t * dW, 1)
        return float(np.mean(dist ** 2))

    def ring_d_batch(tarr: np.ndarray) -> np.ndarray:
        W = W0[None, :, :] + tarr[:, None, None] * dW
        n = len(ring2)
        # Search bound: a point with d^2 > len(ring2)*eps alone pushes the level mean above eps, so
        # the level is unstable whatever its exact distance; the tree may return inf. On stable
        # levels all d^2 are under the bound and match the unbounded search.
        bound = np.sqrt(n * eps)
        # Pre-filter on every 8th point: if their d^2 sum already exceeds n*eps, the level is unstable.
        # Survivors are evaluated on all points as before; unstable levels get inf (their value is
        # never read: ring_err uses stable levels only).
        ds, _ = gtree.query(W[:, ::8].reshape(-1, 3), 1, distance_upper_bound=bound)
        alive = ~((ds ** 2).reshape(len(tarr), -1).sum(axis=1) > n * eps)
        d = np.full(len(tarr), np.inf)
        if alive.any():
            dist, _ = gtree.query(W[alive].reshape(-1, 3), 1, distance_upper_bound=bound)
            d[alive] = (dist ** 2).reshape(-1, n).mean(axis=1)
        return d

    # Sweep only the range the target can occupy along this plane's normal
    # (project the gt bbox corners onto the normal in PLANE coords) -- the
    # full [-ext, ext] range wastes >half the levels outside the part.
    Tin = np.linalg.inv(T)
    # bbox corners of the gt point cloud (tree mins/maxes = min/max of its data; do not recompute
    # on every sweep: the cloud is the same for the whole pass):
    bb_lo = gtree.mins; bb_hi = gtree.maxes
    corners = np.array([[x, y, z, 1.0] for x in (bb_lo[0], bb_hi[0])
                        for y in (bb_lo[1], bb_hi[1]) for z in (bb_lo[2], bb_hi[2])])
    t_corners = (Tin @ corners.T).T[:, 2]
    t_part_lo, t_part_hi = float(t_corners.min()), float(t_corners.max())
    t_span_part = t_part_hi - t_part_lo                       # part extent along this normal
    min_span = min_span_frac * ext
    # genuinely thin axis: the part is thinner than min_span ALONG THIS NORMAL, so
    # its clean single extrude is a flat sheet/plate the min_span rule would wrongly
    # reject.  Thin axes are handled by full-thickness snap (below): catch the run
    # with a LOOSE eps, then snap endpoints to the exact part bounds.
    thin_axis = t_span_part < min_span
    # THICK-axis eps cap: the global eps (~2.4u drift at ext=200) lets the ring
    # stay "stable" past sharp feature ends; capping drift to 25% of the span
    # resolves them (helps gear/bracket).  NOT applied to thin axes -- there the
    # cap would be tighter than mesh noise and the footprint sweep finds NO run.
    if not thin_axis and t_span_part > 1e-6:
        eps = min(eps, (0.25 * t_span_part) ** 2)
    t_lo, t_hi = float(t_corners.min()) - 2.0, float(t_corners.max()) + 2.0
    ts = np.linspace(t_lo, t_hi, n_levels)

    d = ring_d_batch(ts)
    stable = d <= eps
    runs, i = [], 0
    while i < len(ts):
        if stable[i]:
            j = i
            while j + 1 < len(ts) and stable[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1

    cands: list[Candidate] = []
    # min_span and thin_axis computed above (before the eps cap).
    step = ts[1] - ts[0]
    picked = []
    if runs:
        runs_sorted = sorted(runs, key=lambda r: ts[r[1]] - ts[r[0]], reverse=True)
        picked.append(runs_sorted[0])                       # largest stable run
        for r in runs:                                       # largest containing t=0
            if ts[r[0]] <= 0.0 <= ts[r[1]] and r not in picked:
                picked.append(r); break

    def refine(t_in: float, t_out: float) -> float:
        """Bisect the stable/unstable boundary between a stable level t_in
        and an unstable neighbour t_out (sub-cell endpoint precision)."""
        for _ in range(7):
            tm = 0.5 * (t_in + t_out)
            if ring_d(tm) <= eps:
                t_in = tm
            else:
                t_out = tm
        return t_in

    # interval-end containment trim: the sweep's eps tolerance lets interval
    # ends creep past chamfered/rounded feature tips (ring drifts ~sqrt(eps)
    # off-surface before tripping).  Pull each end inward until the end-cap
    # CENTROID sits INSIDE the target solid -- a candidate must never overhang
    # the part (overhang voxels are uniquely its own -> the IoU pruner would
    # kill the whole candidate for them).
    cen = profile.polygon.representative_point()   # guaranteed INSIDE (centroid
    cx_, cy_ = float(cen.x), float(cen.y)          # falls in the hole of an annulus)

    def _inside_batch(P3: np.ndarray) -> np.ndarray:
        if gt_vox is not None:
            try:
                return np.asarray(gt_vox.is_filled(P3), dtype=bool)
            except Exception:
                return np.ones(len(P3), dtype=bool)
        if gt_mesh is not None:
            try:
                return np.asarray(gt_mesh.contains(P3), dtype=bool)
            except Exception:
                return np.ones(len(P3), dtype=bool)
        return np.ones(len(P3), dtype=bool)

    def trim_end(t_end: float, direction: float, t_other: float) -> float:
        # batched replay of the sequential inward walk (12 steps of 1.5, probe
        # 1.0 inward of the current end): all 12 probe points go through ONE
        # inside-test call, then the walk's stop logic replays on the results.
        if gt_vox is None and gt_mesh is None:
            return t_end
        tk = t_end - 1.5 * direction * np.arange(13)
        probes = tk[:12] - 1.0 * direction
        pw = (np.array([cx_, cy_]) @ T[:3, :2].T + T[:3, 3])[None, :] \
            + probes[:, None] * T[:3, 2]
        inside = _inside_batch(pw)
        for k in range(12):
            if abs(tk[k] - t_other) < min_span or inside[k]:
                return float(tk[k])
        return float(tk[12])

    def emit(t0: float, t1: float, i0: int, i1: int):
        t0 = trim_end(t0, -1.0, t1)        # pull start inward (towards t1)
        t1 = trim_end(t1, +1.0, t0)        # pull end inward (towards t0)
        # min_span (0.02*ext) rejects sliver runs in the MIDDLE of a thick part,
        # but a run that spans (nearly) the WHOLE part along this normal IS the
        # part's thickness, not a sliver -- a flat sheet/plate <0.02*ext thick.
        # Keep it: this is the clean single extrude the X/Y strip-sweeps only
        # approximate.  (A genuine sliver covers <70% of the extent -> rejected.)
        full_thickness = thin_axis and (t1 - t0) >= 0.70 * t_span_part
        if (t1 - t0) < min_span and not full_thickness:
            return
        # snap a full-thickness extrude to the part's EXACT extent along the
        # normal: even the capped eps leaves a thin sheet's run over-/under-shot
        # by a fraction of its thickness (1u sheet -> 1.5u, IoU 0.67); the true
        # solid spans exactly [t_part_lo, t_part_hi] -> clean single extrude ~1.0.
        if full_thickness:
            t0, t1 = t_part_lo, t_part_hi
        try:
            m = trimesh.creation.extrude_polygon(profile.polygon, t1 - t0)
            m.apply_translation([0.0, 0.0, t0])
            m.apply_transform(profile.to_3d)
            if len(m.faces):
                cands.append(Candidate(mesh=m, profile=profile, t0=t0, t1=t1,
                                       ring_err=float(np.mean(d[i0:i1 + 1]))))
        except Exception:
            pass

    for (i0, i1) in picked:
        # variant A: refined endpoints (bisect to the eps boundary) -- precise
        # for sharp feature ends.
        t0r = float(ts[i0]) if i0 == 0 else refine(float(ts[i0]), float(ts[i0] - step))
        t1r = float(ts[i1]) if i1 == len(ts) - 1 else refine(float(ts[i1]), float(ts[i1] + step))
        emit(t0r, t1r, i0, i1)
        # variant B: half-cell extension -- better at chamfered/filleted ends
        # where the ring distance grows gradually past eps (the true feature
        # boundary is the SLOPE SPIKE, slightly beyond the eps crossing).
        t0h, t1h = float(ts[i0] - 0.5 * step), float(ts[i1] + 0.5 * step)
        if abs(t0h - t0r) > 0.5 or abs(t1h - t1r) > 0.5:
            emit(t0h, t1h, i0, i1)
    return cands


# --------------------------------------------------------------------------
# 3. Voxel-IoU assembly: union of all + backward marginal pruning
# --------------------------------------------------------------------------

def _occupancy(mesh: trimesh.Trimesh, P: np.ndarray, pitch: float) -> Optional[np.ndarray]:
    # EXACT point-inside test (consistent with the analytic extrude occupancy used
    # for candidates).  contains() is bounded (ray casting) -- unlike the default
    # voxelized().fill() which explodes in subdivide_to_size on thin/large meshes.
    # Both GT and candidates now use the same exact "inside the solid" definition,
    # so the IoU metric stays consistent (a voxel-vs-analytic mix regressed quality).
    try:
        return np.asarray(mesh.contains(P), dtype=bool)
    except Exception:
        try:
            return voxelize_lattice(mesh, pitch).fill().is_filled(P)
        except Exception:
            return None


# fast vectorized point-in-polygon (holes handled natively) -- shapely 2.x / 1.x
try:
    from shapely import contains_xy as _contains_xy
    def _pip(poly, x, y):
        return np.asarray(_contains_xy(poly, x, y), dtype=bool)
except Exception:
    try:
        from shapely.vectorized import contains as _vcontains
        def _pip(poly, x, y):
            return np.asarray(_vcontains(poly, x, y), dtype=bool)
    except Exception:
        _pip = None


# Homogeneous grid coordinates: `_occ_extrude` is called for every candidate with the same
# grid, and `column_stack` on it was a quarter of its time. The cache holds the last grid,
# keyed by array identity; grids in det are never modified in place.
_HOMOGENEOUS: list = [None, None]


def _homogeneous(P: np.ndarray) -> np.ndarray:
    if _HOMOGENEOUS[0] is not P:
        _HOMOGENEOUS[0], _HOMOGENEOUS[1] = P, np.column_stack([P, np.ones(len(P))])
    return _HOMOGENEOUS[1]


def _occ_extrude(c: "Candidate", P: np.ndarray) -> Optional[np.ndarray]:
    """Exact occupancy of an extrude candidate on grid P: map P into the profile
    plane, then (t in [t0,t1]) AND (2D point-in-profile).  Replaces the per-
    candidate mesh.voxelized().fill() that dominates assemble runtime (~110x/call).
    Returns None on any problem -> caller falls back to the voxel path."""
    if _pip is None:
        return None
    try:
        Tinv = np.linalg.inv(c.profile.to_3d)
        Ph = _homogeneous(P)
        Pp = (Tinv @ Ph.T).T[:, :3]                      # plane coords (x, y, t)
        t = Pp[:, 2]
        lo, hi = (c.t0, c.t1) if c.t0 <= c.t1 else (c.t1, c.t0)
        m = (t >= lo) & (t <= hi)
        # and the profile's extent: a point outside it is certainly outside, so `_pip` runs half as often
        x0, y0, x1, y1 = c.profile.polygon.bounds
        m &= (Pp[:, 0] >= x0) & (Pp[:, 0] <= x1) & (Pp[:, 1] >= y0) & (Pp[:, 1] <= y1)
        out = np.zeros(len(P), dtype=bool)
        if m.any():
            xy = Pp[m]
            out[m] = _pip(c.profile.polygon, xy[:, 0], xy[:, 1])
        return out
    except Exception:
        return None


def _adaptive_grid_pts(lo, hi, pitch: float, K: int = 5) -> np.ndarray:
    """Regular occupancy grid, but each axis step is refined so EVERY axis gets
    >= K samples ACROSS the part.  A uniform pitch-2.0 grid never lands a point
    inside a <2u-thick sheet/plate -- with the exact contains() occupancy both GT
    and candidate occupancies come back empty -> IoU undefined -> 0 ops.  Normal
    parts (extent >> pitch on all axes) are UNCHANGED: step==pitch, margin==3*pitch
    (==6 at pitch 2), so this only adds samples on genuinely thin axes."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ext = np.maximum(hi - lo, 1e-9)
    step = np.minimum(float(pitch), ext / K)          # finer only on thin axes
    m = 3.0 * step                                    # per-axis margin (6 at step 2)
    lo2 = lo - m; hi2 = hi + m
    n = np.maximum(((hi2 - lo2) / step).astype(int) + 1, 2)
    xs = [lo2[k] + (np.arange(n[k]) + 0.5) * step[k] for k in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def assemble(gt: trimesh.Trimesh, cands: list[Candidate], pitch: float = 2.0,
             margin: float = 6.0, min_gain: float = 0.004, verbose: bool = True,
             fixed_occ: Optional[np.ndarray] = None, grid_pts: Optional[np.ndarray] = None,
             gt_occ: Optional[np.ndarray] = None, deadline: Optional[float] = None):
    """Union all candidate occupancies, then backward marginal-IoU pruning
    WITH COMPACTNESS PRESSURE: an op is kept only if removing it would cost
    at least ``min_gain`` IoU (each op must EXPLAIN a feature, not fill a few
    voxels).  Near-duplicate candidates (occupancy overlap) are deduped first.
    ``gt_occ`` (paired with ``grid_pts``) skips the expensive GT contains();
    past ``deadline`` (epoch s) the prune loop stops early with what it has.
    Returns (kept candidates, final IoU, diagnostics)."""
    import time as _tm
    if grid_pts is None:
        P = _adaptive_grid_pts(gt.bounds[0], gt.bounds[1], pitch)
    else:
        P = grid_pts

    G = gt_occ if gt_occ is not None else _occupancy(gt, P, pitch)
    if G is None:
        return [], 0.0, {"error": "gt voxelization failed"}

    occ, keep = [], []
    for c in cands:
        o = _occ_extrude(c, P)                 # fast EXACT analytic extrude occupancy
        if o is None:
            o = _occupancy(c.mesh, P, pitch)   # fallback: exact contains()
        if o is not None and o.any():
            occ.append(o); keep.append(c)
    if not occ:
        return [], 0.0, {"error": "no candidate occupancies"}
    occ = np.stack(occ)                       # (m, V) bool

    # --- structural dedup: collapse MUTUAL near-duplicates only (the
    # both-variant twins, repeated slices of the same slab).  A small precise
    # candidate CONTAINED in a coarse big one is NOT a duplicate -- the pruner
    # decides between them (and naturally drops the coarse one, since its
    # excess volume overshoots the target and costs IoU).  Keep the candidate
    # with the better boundary fit (lower ring_err) from each duplicate group.
    # Intersections are counted on packed bits (`_popcounts`): an integer count identical to
    # `(occ[j] & occ[k]).sum()`, with 8x less data.
    bits = np.packbits(occ, axis=1)
    sizes = occ.sum(axis=1)
    order = np.argsort([keep[j].ring_err for j in range(len(keep))])  # best fit first
    alive = []
    for j in order:
        dup = False
        for k in alive:
            inter_jk = int(np.bitwise_count(bits[j] & bits[k]).sum())
            if inter_jk >= 0.85 * sizes[j] and inter_jk >= 0.85 * sizes[k]:
                dup = True; break             # near-identical volumes
        if not dup:
            alive.append(int(j))
    if verbose and len(alive) < len(keep):
        print(f"[dedup] {len(keep)} -> {len(alive)} candidates (mutual near-duplicates)")
    occ = occ[alive]; keep = [keep[j] for j in alive]; bits = bits[alive]

    fixed = fixed_occ if fixed_occ is not None else np.zeros(P.shape[0], dtype=bool)

    def _lost_counts(active):
        """(a, b) for each of `active`: how many voxels owned only by this candidate
        (`counts == 1`, outside `fixed`) fall inside GT and outside it: the same as
        `(occ[j] & (counts == 1) & ~fixed & G).sum()` and `... & ~G`, in one pass."""
        uniq = (counts == 1) & ~fixed
        in_g = np.packbits(uniq & G)
        out_g = np.packbits(uniq & ~G)
        sub = bits[active]
        a = np.bitwise_count(sub & in_g).sum(axis=1)
        b = np.bitwise_count(sub & out_g).sum(axis=1)
        return {j: (int(a[i]), int(b[i])) for i, j in enumerate(active)}
    counts = occ.sum(axis=0).astype(np.int32)
    U = (counts > 0) | fixed
    inter = int((U & G).sum()); uni = int((U | G).sum())
    iou = inter / max(uni, 1)
    if verbose:
        print(f"[assemble] {len(keep)} candidates, union IoU={iou:.3f}"
              + (" (with protected primitives)" if fixed_occ is not None else ""))

    # --- backward pruning with min-gain: repeatedly remove the op whose
    # removal hurts LEAST, while that hurt is < min_gain.
    active = list(range(len(keep)))
    while len(active) > 1:
        if deadline is not None and _tm.time() > deadline:
            # deadline: ONE bulk approximate prune instead of the remaining
            # rounds -- drop every candidate whose unique-voxel loss is below
            # its bar in a single pass.  Returning the raw UNPRUNED set here
            # would hand downstream stages (merge, footprint subtraction,
            # emit, stage-B CadQuery) a ~100-op program and cost MORE time
            # than the loop it skipped.
            drop = []
            lost_ab = _lost_counts(active)
            for j in active:
                a, b = lost_ab[j]
                iou_j = (inter - a) / max(uni - b, 1)
                bar = (min_gain * 0.15 if getattr(keep[j].profile, "source", "") == "oriented"
                       else min_gain)
                if (iou - iou_j) < bar:
                    drop.append(j)
            for j in drop:
                if len(active) <= 1:
                    break
                counts -= occ[j].astype(np.int32)
                active.remove(j)
            U = (counts > 0) | fixed
            inter = int((U & G).sum()); uni = int((U | G).sum())
            iou = inter / max(uni, 1)
            if verbose:
                print(f"[budget] bulk prune -> {len(active)} ops, IoU={iou:.3f}")
            break
        # a candidate is REMOVABLE if its unique-voxel loss costs < its gain bar;
        # remove the least-costly one each pass.  Oriented (off-axis thin frame
        # members) get a softer bar -- one member is below min_gain at pitch 2 yet
        # collectively they ARE the part, so the default bar would prune them all.
        best_j, best_iou, best_ab = None, -1.0, None
        # voxels ONLY this candidate covers: a — losing target coverage, b — shedding overshoot
        lost_ab = _lost_counts(active)
        for j in active:
            a, b = lost_ab[j]
            iou_j = (inter - a) / max(uni - b, 1)
            bar = (min_gain * 0.15 if getattr(keep[j].profile, "source", "") == "oriented"
                   else min_gain)
            if (iou - iou_j) < bar and iou_j > best_iou:
                best_j, best_iou, best_ab = j, iou_j, (a, b)
        if best_j is None:
            break                              # nothing removable: every op explains a feature
        a, b = best_ab
        counts -= occ[best_j].astype(np.int32)
        inter -= a; uni -= b; iou = inter / max(uni, 1)
        active.remove(best_j)
        if verbose:
            print(f"[prune] drop #{best_j} ({keep[best_j].profile.source}, "
                  f"span {keep[best_j].t1-keep[best_j].t0:.0f}) -> IoU={iou:.3f}  "
                  f"({len(active)} left)")
    kept = [keep[j] for j in active]
    kept_occ = occ[active].any(axis=0) | fixed if len(active) else fixed
    return kept, iou, {"n_initial": len(keep), "n_kept": len(kept), "pitch": pitch,
                       "min_gain": min_gain, "union_occ": kept_occ, "grid_pts": P}


# --------------------------------------------------------------------------
# Top level: single pass
# --------------------------------------------------------------------------

def _merge_coaxial(kept: list, gt: trimesh.Trimesh, pitch: float,
                   verbose: bool = True) -> list:
    """Merge kept ops that are segments of ONE feature: same plane normal,
    same profile footprint (area + lateral centroid), intervals adjacent or
    overlapping (gap <= 6).  A chamfered waist splits a column's stable run
    into 2-3 segments -- in CAD terms it is one extrude."""
    used = [False] * len(kept)
    merged = []
    for i, c in enumerate(kept):
        if used[i]:
            continue
        group = [c]; used[i] = True
        ni = c.profile.to_3d[:3, 2]
        c2 = np.array(c.profile.polygon.centroid.coords[0])
        ci3 = (c.profile.to_3d @ np.array([c2[0], c2[1], 0.0, 1.0]))[:3]
        ext_s = sorted(float(np.dot((c.profile.to_3d @ np.array([c2[0], c2[1], t, 1.0]))[:3], ni))
                       for t in (c.t0, c.t1))   # group's axial extent (world s)
        for j in range(i + 1, len(kept)):
            if used[j]:
                continue
            d = kept[j]
            nj = d.profile.to_3d[:3, 2]
            if abs(float(np.dot(ni, nj))) < 0.99:
                continue
            ar = max(c.profile.area, d.profile.area) / max(min(c.profile.area, d.profile.area), 1e-9)
            if ar > 1.10:
                continue   # 1.10: chamfer-split segments of ONE feature have
                           # ratio ~1.0; a real diameter STEP (tip lip) is >1.15
                           # and must stay a separate op
            d2 = np.array(d.profile.polygon.centroid.coords[0])
            dj3 = (d.profile.to_3d @ np.array([d2[0], d2[1], 0.0, 1.0]))[:3]
            lat = (dj3 - ci3) - np.dot(dj3 - ci3, ni) * ni     # lateral offset
            if float(np.linalg.norm(lat)) > 6.0:
                continue
            # AXIAL GAP CHECK: merge only adjacent/overlapping segments (one
            # chamfer-split feature).  Two tip lips at OPPOSITE ends of a rod
            # must NOT be bridged into one full-length op.  Computed as scalar
            # projections of each segment's axis end-points onto the SHARED
            # axis direction, all in world space (no frame inversions).
            dj2w = d.profile.to_3d
            d2 = np.array(d.profile.polygon.representative_point().coords[0])
            sj = sorted(float(np.dot((dj2w @ np.array([d2[0], d2[1], t, 1.0]))[:3], ni))
                        for t in (d.t0, d.t1))
            # CHAIN-AWARE gap: measured vs the GROUP'S accumulated extent
            # (band3 can be far from the seed yet adjacent to band2).
            # Chamfer/hub transition zones are 10-45u; a tip-tip bridge
            # across a rod is >100u from any group extent.  60 splits them.
            gap = max(ext_s[0], sj[0]) - min(ext_s[1], sj[1])
            if gap > 60.0:
                continue
            group.append(d); used[j] = True
            ext_s = [min(ext_s[0], sj[0]), max(ext_s[1], sj[1])]
        if len(group) == 1:
            merged.append(c); continue
        # merge: spans in the FIRST op's plane coords (project interval ends)
        base = group[0]
        Tin = np.linalg.inv(base.profile.to_3d)
        t_lo, t_hi = [], []
        for g in group:
            for t in (g.t0, g.t1):
                p3 = (g.profile.to_3d @ np.array([0.0, 0.0, t, 1.0]))[:3]
                t_lo.append(float((Tin @ np.append(p3, 1.0))[2]))
        t0, t1 = min(t_lo), max(t_lo)
        try:
            m = trimesh.creation.extrude_polygon(base.profile.polygon, t1 - t0)
            m.apply_translation([0.0, 0.0, t0]); m.apply_transform(base.profile.to_3d)
            merged.append(Candidate(mesh=m, profile=base.profile, t0=t0, t1=t1,
                                    ring_err=min(g.ring_err for g in group)))
            if verbose:
                print(f"[merge] {len(group)} coaxial segments -> one op "
                      f"(span {t1-t0:.0f}, area {base.profile.area:.0f})")
        except Exception:
            merged.extend(group)
    return merged



def _subtract_primitive_footprints(c: Candidate, prims: list[Candidate],
                                   buffer: float = 0.8) -> Optional[Candidate]:
    """Feature isolation: remove protected primitives' 2D footprints from a
    polygon candidate's profile.  A slice plane that crosses a primitive (e.g.
    the flange slice crossing the arm rod) drags the primitive's silhouette
    along as a strip -- extruded, that strip wraps the round feature in a box
    sheath.  Sectioning each crossing primitive with the candidate's mid-plane
    and subtracting the (buffered convex-hull) footprint yields the candidate's
    OWN feature only.  Returns a LIST of rebuilt Candidates (a bisected\n    profile yields one per remaining part); empty if nothing remains.
    """
    from shapely.geometry import MultiPoint
    T = c.profile.to_3d
    Tin = np.linalg.inv(T)
    nrm = T[:3, 2]
    t_mid = 0.5 * (c.t0 + c.t1)
    o_mid = (T @ np.array([0.0, 0.0, t_mid, 1.0]))[:3]
    poly = c.profile.polygon
    # collect radii of primitives CROSSING this profile's plane (their section
    # exists at the candidate's mid-plane) -- they leave silhouette strips of
    # width ~2r in the section polygon.
    w_max = 0.0
    for p in prims:
        pn = p.profile.to_3d[:3, 2]
        coaxial = abs(float(np.dot(nrm, pn))) > 0.9
        if coaxial and p.profile.area > 0.5 * c.profile.area:
            continue   # same-axis STACKING (collar on column) -- keep
        try:
            sec = p.mesh.section(plane_origin=o_mid, plane_normal=nrm)
            if sec is None or len(sec.vertices) < 3:
                continue
            r_p = math.sqrt(max(p.profile.area, 1.0) / math.pi)
            w_max = max(w_max, r_p + 1.5)
        except Exception:
            continue
    if w_max <= 0.0:
        return [c]
    # MORPHOLOGICAL OPENING: removes protruding strips narrower than 2*w_max
    # (the crossing primitives' silhouettes) while leaving the profile's
    # INTERIOR untouched -- so stacked levels of one block keep IDENTICAL
    # cross-sections and the coaxial merge can unify them into ONE op.
    try:
        opened = poly.buffer(-w_max).buffer(w_max)
    except Exception:
        return [c]
    if opened.is_empty or opened.area < 0.3 * poly.area:
        return []      # strip-dominated bundle: its true geometry IS the crossing
                       # primitives (already protected) -- keep nothing
    poly = opened
    changed = True
    # the footprint strip may BISECT the profile (rod through the flange disc
    # -> two half-discs): keep EVERY remaining part as its own candidate.
    parts = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    out = []
    for part in parts:
        if part.is_empty or part.area < max(0.08 * c.profile.area, 40.0):
            continue
        try:
            m = trimesh.creation.extrude_polygon(part, c.t1 - c.t0)
            m.apply_translation([0.0, 0.0, c.t0]); m.apply_transform(T)
            if len(m.faces) == 0:
                continue
            prof = Profile(polygon=part, to_3d=T, source=c.profile.source,
                           area=float(part.area))
            out.append(Candidate(mesh=m, profile=prof, t0=c.t0, t1=c.t1,
                                 ring_err=c.ring_err))
        except Exception:
            continue
    return out


def _oriented_sweep_candidates(gt_mesh: trimesh.Trimesh, gtree: cKDTree, ext: float,
                               max_dirs: int = 4, max_cands: int = 60) -> list:
    """OFF-AXIS extrude support: members/features running along NON-cardinal
    directions (a frame's diagonal braces, an angled boss) are missed by the
    cardinal + cluster-normal sweeps.  Detect the dominant non-cardinal EDGE
    directions and sweep profiles sectioned perpendicular to each.  Returns []
    for axis-aligned parts (cheap edge scan), so it never perturbs them."""
    try:
        E = gt_mesh.vertices[gt_mesh.edges_unique]
    except Exception:
        return []
    vec = E[:, 1] - E[:, 0]; Ln = np.linalg.norm(vec, axis=1)
    keep = Ln > 0.05 * ext
    if int(keep.sum()) < 8:
        return []
    d = vec[keep] / Ln[keep, None]
    d = d * np.sign(d[:, 2:3] + 1e-9)                      # fold to a hemisphere
    Lk = Ln[keep]; cl = []
    for i in np.argsort(-Lk):
        for c in cl:
            if abs(d[i] @ c[0]) > 0.985:
                c[1] += Lk[i]; break
        else:
            cl.append([d[i].copy(), float(Lk[i])])
    tot = sum(c[1] for c in cl) + 1e-9
    cl.sort(key=lambda c: -c[1])
    dirs = [c[0] for c in cl if max(abs(c[0])) < 0.985 and c[1] > 0.06 * tot][:max_dirs]
    if not dirs:
        return []
    center = gt_mesh.centroid; out = []
    for n in dirs:
        n = n / np.linalg.norm(n)
        proj = (gt_mesh.vertices - center) @ n; lo, hi = float(proj.min()), float(proj.max())
        for q in np.linspace(0.12, 0.88, 6):
            o = center + (lo + q * (hi - lo)) * n
            try:
                profs = _section_profiles(gt_mesh, o, n, "oriented", 5e-4 * ext * ext)
            except Exception:
                profs = []
            for p in profs:
                out += sweep_candidates(p, gtree, ext, gt_mesh=gt_mesh)
                if len(out) >= max_cands:
                    return out
    return out


def cadfit_single_pass(gt_mesh: trimesh.Trimesh,
                       n_gt_points: int = 20000,
                       max_candidates: int = 110,
                       pitch: float = 2.0,
                       min_gain: float = 0.004,
                       verbose: bool = True,
                       deadline: Optional[float] = None,
                       grid_pts: Optional[np.ndarray] = None,
                       gt_occ: Optional[np.ndarray] = None):
    """Profiles -> sweep candidates -> union+prune.  Returns (kept, iou, info).
    ``grid_pts``/``gt_occ``: caller-provided occupancy grid + GT occupancy on it
    (must be _grid(gt_mesh, pitch) / _occupancy of the SAME mesh) -- avoids
    recomputing the GT contains() in every stage.  ``deadline`` (epoch s):
    soft wall-clock cap; loops stop early and assemble what exists."""
    import time as _tm
    ext = float(np.max(gt_mesh.extents))
    gp, _ = trimesh.sample.sample_surface(gt_mesh, n_gt_points,
                                          seed=42)   # deterministic: stable sweeps
    gtree = cKDTree(np.asarray(gp))

    profiles = extract_profiles(gt_mesh)
    if verbose:
        print(f"[profiles] {len(profiles)} unique "
              f"({sum(1 for p in profiles if p.source=='cluster')} cluster, "
              f"{sum(1 for p in profiles if p.source=='axis')} axis)")

    # gt_vox for sweep_candidates' trim_end inside-test.  voxelized(2.0).fill()
    # HANGS on big solid meshes (subdivide_to_size blowup -- 84s on Coupling, a
    # 345k-voxel box) so we only build it when the voxel grid is CHEAP.  The
    # dilated fill helps the gear's collapse (its thin 7u disc trims cleanly to a
    # 2-op result, 0.984 vs 0.952).  Above the cap, sweep falls back to the exact
    # embree gt_mesh.contains() -- fast and accurate, just not dilated.
    _ext3 = gt_mesh.extents
    _approx_vox = float(_ext3[0] * _ext3[1] * _ext3[2]) / 8.0     # voxels at pitch 2
    _gvox = None
    if _approx_vox < 1.2e5:
        try:
            _gvox = voxelize_lattice(gt_mesh, 2.0).fill()
        except Exception:
            _gvox = None
    cands: list[Candidate] = []
    for p in profiles:
        if deadline is not None and _tm.time() > deadline:
            if verbose:
                print("[budget] sweep loop stopped early")
            break
        cands += sweep_candidates(p, gtree, ext, gt_mesh=gt_mesh, gt_vox=_gvox)
    cands.sort(key=lambda c: c.ring_err)
    cands = cands[:max_candidates]
    # off-axis extrudes (diagonal frame members, angled bosses); [] for aligned
    # parts.  Added AFTER the ring_err cap -- oriented sections of a busy frame have
    # noisy boundaries (high ring_err) and would be sorted out of the top set.
    # OPT-IN (DET_ORIENTED): helps angled/housing parts (cad23 0.46->0.53) but slightly
    # perturbs aligned parts (tdn mean -0.003), so OFF by default; frames also need
    # spurious-circle suppression to benefit (00441507 still routes through prim-first).
    import os as _oso
    _ori = _oriented_sweep_candidates(gt_mesh, gtree, ext) if _oso.environ.get("DET_ORIENTED") else []
    if _ori:
        cands += _ori
        if verbose:
            print(f"[oriented] +{len(_ori)} off-axis candidates")
    if verbose:
        print(f"[sweep] {len(cands)} extrusion candidates")
    if not cands:
        return [], 0.0, {"error": "no candidates"}

    # PRIMITIVE-FIRST two-stage assembly: cylinders (outer ring fits a circle)
    # are assembled and PROTECTED first -- bundled cross-section polygons (a
    # Z-slice through the flange also drags the arms' silhouettes along as
    # strips) otherwise cover the arm volume as boxes and make the true round
    # features look "redundant" to the pruner.
    def _is_circle(c):
        # OUTER ring must fit a circle; interior holes are WELCOME (annulus =
        # hollow cylinder, e.g. the top collar with its bore) -- but each hole
        # must itself be circular so the feature stays a clean turned part.
        # RADIUS CAP: a huge "circle" (r > 25% of the part) is almost always
        # the side SILHOUETTE of the whole body (flange+collars happen to be
        # round), not a cylindrical feature -- protecting it lets an unprunable
        # bundle squat inside the central block.
        f = _fit_circle_2d(_ring_coords(c.profile.polygon.exterior))
        if f is None or f[3] >= 1.0:
            return False
        if f[2] > 0.25 * ext:
            return False
        for hole in c.profile.polygon.interiors:
            fh = _fit_circle_2d(_ring_coords(hole))
            if fh is None or fh[3] >= 1.5:
                return False
        return True
    prim = [c for c in cands if _is_circle(c)]
    poly = [c for c in cands if not _is_circle(c)]
    if verbose:
        print(f"[stage] {len(prim)} primitive (circular) candidates, {len(poly)} polygon")
    if prim:
        kept_p, iou_p, info_p = assemble(gt_mesh, prim, pitch=pitch,
                                         min_gain=min_gain, verbose=verbose,
                                         grid_pts=grid_pts, gt_occ=gt_occ,
                                         deadline=deadline)
        if verbose:
            print(f"[stage1] primitives kept {len(kept_p)} IoU={iou_p:.3f}")
        # feature isolation: strip protected primitives' silhouettes out of
        # the polygon candidates' profiles (kills the box sheaths)
        poly_iso = []
        for c in poly:
            poly_iso += _subtract_primitive_footprints(c, kept_p)
        if verbose:
            print(f"[isolate] {len(poly)} polygon candidates -> {len(poly_iso)} after footprint subtraction")
        # GT occupancy is recomputed here although the grid is the same as in the first stage:
        # `contains` settles ambiguous points with an `np.random` ray, and reusing the result
        # shifts the `warm` output (worse overall).
        kept_q, iou, info = assemble(gt_mesh, poly_iso, pitch=pitch, min_gain=min_gain,
                                     verbose=verbose, fixed_occ=info_p.get("union_occ"),
                                     grid_pts=info_p.get("grid_pts"), gt_occ=gt_occ,
                                     deadline=deadline)
        kept = kept_p + kept_q
    else:
        kept, iou, info = assemble(gt_mesh, cands, pitch=pitch, min_gain=min_gain, verbose=verbose,
                                   grid_pts=grid_pts, gt_occ=gt_occ, deadline=deadline)
    kept = _merge_coaxial(kept, gt_mesh, pitch, verbose=verbose)

    # DOMINANT-BASE COLLAPSE: single-extrude parts (spur gears, flat levers,
    # plates) have a toothed/complex OUTER profile that _is_circle() rejects, so
    # the primitive-first + polygon-isolation path shreds that one profile into a
    # box grid.  If a single high-volume extrude candidate matches the GT about as
    # well as the whole assembly, prefer it -- the residual pass then refines it.
    try:
        import os as _os
        if _os.environ.get("DET_NO_COLLAPSE"):
            raise RuntimeError("collapse disabled")
        _P = grid_pts if grid_pts is not None else _grid(gt_mesh, pitch)
        _G = gt_occ if gt_occ is not None else _occupancy(gt_mesh, _P, pitch)
        if _G is None or not _G.any():
            raise RuntimeError("gt occupancy failed")
        def _cand_occ(c):
            # analytic extrude occupancy (same fast path assemble uses);
            # contains() fallback for candidates without a usable profile.
            o = _occ_extrude(c, _P)
            if o is None:
                o = _occupancy(c.mesh, _P, pitch)
            return o
        def _iou_of_occ(occ):
            if occ is None:
                return 0.0
            return float((occ & _G).sum() / max((occ | _G).sum(), 1))
        _vgt = float(abs(gt_mesh.volume))
        # base pool = big capped cands + a CLEAN re-sweep of the largest profiles.
        # (the gt_vox sweep overshoots thin single-extrude parts -- a gear's 7u
        # disc becomes an 11u block -- so re-sweep big profiles WITHOUT gt_vox to
        # get a tight extrude.)  Filter to big meshes BEFORE any occupancy call.
        def _big(c):
            # Extrusion volume = profile area * height (matches the mesh volume to 1e-9):
            # clearly small candidates are dropped without `mesh.volume`; at the threshold and
            # above the mesh volume decides, as before.
            poly = getattr(getattr(c, "profile", None), "polygon", None)
            if poly is not None and poly.area * abs(c.t1 - c.t0) < 0.45 * _vgt:
                return False
            return float(abs(c.mesh.volume)) >= 0.5 * _vgt
        base_pool = [c for c in cands if _big(c)]
        # relax min_span HERE ONLY (not globally) so a thin flat part's single
        # extrude survives -- a 1.3u ring / 2.2u plate is < the global min_span
        # (0.02*ext=4) and would be rejected, yet it IS the whole part.  Keeping
        # this local avoids spawning spurious thin candidates in the gear's
        # assembly (which broke it when min_span was relaxed globally).
        for _p in sorted(profiles, key=lambda q: -q.area)[:3]:
            if deadline is not None and _tm.time() > deadline:
                break
            for c in sweep_candidates(_p, gtree, ext, min_span_frac=0.002, gt_mesh=gt_mesh):
                if _big(c):
                    base_pool.append(c)
        base, base_iou = None, 0.0
        for c in base_pool:
            if deadline is not None and _tm.time() > deadline:
                break
            bi = _iou_of_occ(_cand_occ(c))
            if bi > base_iou:
                base_iou, base = bi, c
        if base is not None and base_iou >= 0.80 and len(kept) > 2:
            _occ_asm = np.zeros(len(_P), dtype=bool)
            for c in kept:
                o = _cand_occ(c)
                if o is not None:
                    _occ_asm |= o
            asm_iou = _iou_of_occ(_occ_asm)
            if base_iou >= asm_iou - 0.03:               # assembly isn't buying accuracy
                if verbose:
                    print(f"[collapse] dominant base IoU={base_iou:.3f} replaces "
                          f"{len(kept)} fragments (assembly IoU={asm_iou:.3f})")
                kept, iou = [base], base_iou
    except Exception as _e:
        if verbose:
            print(f"[collapse] skipped: {_e}")

    info["n_profiles"] = len(profiles)
    info["n_kept"] = len(kept)
    return kept, iou, info


# --------------------------------------------------------------------------
# Residual iteration (Alg. 3): R+ = GT \ S -> union;  R- = S \ GT -> cut
# --------------------------------------------------------------------------

def _grid(gt: trimesh.Trimesh, pitch: float, margin: float = 6.0):
    # margin kept for call-site compat; _adaptive_grid_pts derives a per-axis
    # margin so thin parts (<2u) get sampled across their thickness.
    return _adaptive_grid_pts(gt.bounds[0], gt.bounds[1], pitch)



def tip_extensions(kept, gt_mesh: trimesh.Trimesh, gtree, ext: float,
                   probe: float = 3.0, verbose: bool = True) -> list:
    """Multi-diameter feature continuation: for each kept CYLINDRICAL op, probe
    just past each end of its span.  If the GT section there contains a loop
    centered on the same axis (a tip segment at a new radius -- a turned-part
    step), sweep that loop's profile and emit it as a candidate.  This is how
    the small end extensions of long rods are recovered: the main sweep stops
    at the radius step, and the boolean residual is too noisy to find them.
    """
    out = []
    for c in kept:
        f = _fit_circle_2d(_ring_coords(c.profile.polygon.exterior))
        if f is None or f[3] >= 1.0:
            continue                              # only continue cylinders
        T = c.profile.to_3d
        cen = c.profile.polygon.representative_point()
        for t_end, direction in ((c.t1, +1.0), (c.t0, -1.0)):
            t_probe = t_end + direction * probe
            o3 = (T @ np.array([0.0, 0.0, t_probe, 1.0]))[:3]
            n3 = T[:3, 2] * 1.0
            for prof in _section_profiles(gt_mesh, o3, n3, "tip", 30.0):
                # the loop must sit on the SAME axis (lateral offset small)
                p2 = prof.polygon.representative_point()
                Tin = np.linalg.inv(prof.to_3d)
                ax3 = (T @ np.array([float(cen.x), float(cen.y), t_probe, 1.0]))[:3]
                ax2 = (Tin @ np.append(ax3, 1.0))[:2]
                if not prof.polygon.contains(
                        type(p2)(ax2[0], ax2[1])):
                    continue
                fp = _fit_circle_2d(_ring_coords(prof.polygon.exterior))
                if fp is None or fp[3] >= 1.5:
                    continue                       # tip must be circular too
                cands = sweep_candidates(prof, gtree, ext, gt_mesh=gt_mesh)
                for tc in cands:
                    if (tc.t1 - tc.t0) <= 0.6 * (c.t1 - c.t0):
                        out.append(tc)
                        if verbose:
                            print(f"[tips] extension r={fp[2]:.1f} span="
                                  f"{tc.t1-tc.t0:.1f} past end of "
                                  f"r={f[2]:.1f} op")
                break                              # one loop per end is enough
    return out


# --------------------------------------------------------------------------
# Non-extrude primitives: revolve (axisymmetric turned parts) as whole-part ops
# --------------------------------------------------------------------------

def _best_primitive(gt_mesh: trimesh.Trimesh, P: np.ndarray, G: np.ndarray,
                    pitch: float, kinds=("revolve", "sweep", "loft")) -> Optional[tuple]:
    """Detect the best whole-part PRIMITIVE (revolve / sweep / loft) and score it
    on the SAME occupancy grid as the extrude assembly.  Returns (PrimitiveCandidate,
    iou) or None.  A turned part is ONE revolve (not ~15 discs), a bent round rod is
    ONE sweep (not a straight box), an organic taper is ONE loft."""
    detmap = {}
    for kind, modname, fnname in (("revolve", "revolve", "detect_revolve"),
                                  ("sweep", "sweep", "detect_sweep"),
                                  ("loft", "loft", "detect_loft")):
        if kind not in kinds:
            continue
        try:
            mod = importlib.import_module(f"{__package__}.detectors.{modname}")
            detmap[kind] = getattr(mod, fnname)
        except Exception:
            pass
    best = None
    for kind in kinds:
        fn = detmap.get(kind)
        if fn is None:
            continue
        try:
            outs = fn(gt_mesh)
        except Exception:
            continue
        for r in outs:
            m = getattr(r, "mesh", None)
            if m is None or len(m.faces) == 0:
                continue
            o = _occupancy(m, P, pitch)
            if o is None:
                continue
            iou = float((o & G).sum() / max((o | G).sum(), 1))
            if best is None or iou > best[1]:
                best = (PrimitiveCandidate(mesh=m, program=getattr(r, "program", "") or "",
                                           kind=kind), iou)
    return best


def _revolve_envelope(gt_mesh: trimesh.Trimesh, axis: int, center: np.ndarray,
                      nb: int = 80):
    """Max-radius meridian revolve: outer R(z)=max-azimuth radius, inner bore
    r_in(z)=min-azimuth radius at each axial height.  Builds a surface of
    revolution that CONTAINS a turned body even when windows / holes / flats
    break strict axisymmetry (so detect_revolve rejects it) -- the residual pass
    then CUTS those features out.  Returns (mesh, rz) or (None, None)."""
    try:
        V, _ = trimesh.sample.sample_surface(gt_mesh, 60000, seed=1)
    except Exception:
        return None, None
    others = [i for i in range(3) if i != axis]
    z = V[:, axis] - center[axis]
    r = np.sqrt((V[:, others[0]] - center[others[0]]) ** 2 +
                (V[:, others[1]] - center[others[1]]) ** 2)
    edges = np.linspace(z.min(), z.max(), nb + 1)
    zc = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(z, edges) - 1, 0, nb - 1)
    Ro = np.zeros(nb); Ri = np.zeros(nb); ok = np.zeros(nb, bool)
    for b in range(nb):
        rr = r[idx == b]
        if len(rr) < 5:
            continue
        ok[b] = True
        Ro[b] = np.percentile(rr, 99.5)     # outer envelope (robust max)
        Ri[b] = np.percentile(rr, 0.5)      # bore (robust min)
    if ok.sum() < 4:
        return None, None
    zc, Ro, Ri = zc[ok], Ro[ok], Ri[ok]
    bore = np.median(Ri) > 0.15 * max(np.median(Ro), 1e-9)
    if bore:
        rz = np.vstack([np.column_stack([Ro, zc]),
                        np.column_stack([Ri[::-1], zc[::-1]])])
    else:
        rz = np.vstack([np.column_stack([Ro, zc]), [[0.0, zc[-1]], [0.0, zc[0]]]])
    rz[:, 0] = np.maximum(rz[:, 0], 1e-3)
    try:
        from .detectors.revolve import _build_revolve_mesh
        mesh = _build_revolve_mesh(rz, axis, center)
    except Exception:
        mesh = None
    return mesh, rz


def _best_revolve_envelope(gt_mesh: trimesh.Trimesh, P: np.ndarray, G: np.ndarray,
                           pitch: float):
    """Choose the axis whose max-radius revolve envelope best CONTAINS the part
    (covers_GT high) with the TIGHTEST volume.  Returns (PrimitiveCandidate, iou,
    covers_GT) or None.  Use it as a base for a turned body with cuts."""
    center = (gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0
    gtn = max(int(G.sum()), 1)
    best = None
    for ax in (0, 1, 2):
        mesh, rz = _revolve_envelope(gt_mesh, ax, center)
        if mesh is None or len(mesh.faces) == 0:
            continue
        o = _occupancy(mesh, P, pitch)
        if o is None:
            continue
        inter = int((o & G).sum()); rvol = int(o.sum())
        covers = inter / gtn
        if covers < 0.90:
            continue
        iou = inter / max((o | G).sum(), 1)
        if best is None or rvol < best[3]:        # tightest covering envelope
            prog = ""
            try:
                from .detectors.revolve import _emit_revolve_program
                prog = _emit_revolve_program(rz, ax, center)
            except Exception:
                pass
            best = (PrimitiveCandidate(mesh=mesh, program=prog, kind="revolve"),
                    iou, covers, rvol)
    if best is None:
        return None
    return best[0], best[1], best[2]


def _extend_extrude(c: "Candidate", margin: float) -> "Candidate":
    """Over-pierce an extrude candidate: extend its depth by ``margin`` on BOTH
    ends so, when used as a CUT, it clears the parent surface instead of leaving
    coincident-face slivers.  Stays an extrude (with a .profile) -> emittable."""
    try:
        t0, t1 = (c.t0, c.t1) if c.t0 <= c.t1 else (c.t1, c.t0)
        t0 -= margin; t1 += margin
        mm = trimesh.creation.extrude_polygon(c.profile.polygon, t1 - t0)
        mm.apply_translation([0.0, 0.0, t0])
        mm.apply_transform(c.profile.to_3d)
        return Candidate(mesh=mm, profile=c.profile, t0=t0, t1=t1, ring_err=c.ring_err)
    except Exception:
        return c


def _halfspace_box(c0: np.ndarray, n: np.ndarray, ext: float) -> trimesh.Trimesh:
    """A huge box covering the -n half-space, its near face on the plane (c0, n)."""
    box = trimesh.creation.box(extents=[3.0 * ext] * 3)
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, n); cth = float(z @ n)
    if np.linalg.norm(v) > 1e-8:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + cth))
    else:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    T = np.eye(4); T[:3, :3] = R
    box.apply_transform(T)
    box.apply_translation(c0 - n * (1.5 * ext))
    return box


def _revolve_planar_cuts(base: "PrimitiveCandidate", gt_mesh: trimesh.Trimesh,
                         P: np.ndarray, G: np.ndarray, pitch: float, min_gain: float):
    """Revolve base (op 1) + ONE clean PLANAR cut per flat end/face that the revolve
    over-fills.  A turned part with an angled/flat end (Stopper's diagonal bottom)
    has a planar GT cut face; fit its plane and cut the revolve with a half-space ->
    a single clean `r.cut(<plane>)` instead of a stack of IoU-greedy boxes.
    Returns (ops, iou)."""
    occ = _occupancy(base.mesh, P, pitch)
    if occ is None:
        return [("union", base)], 0.0
    ops = [("union", base)]
    iou = float((occ & G).sum() / max((occ | G).sum(), 1))
    ext = float(np.max(gt_mesh.extents)); gc = gt_mesh.centroid
    try:
        facets = gt_mesh.facets
        fareas = gt_mesh.facets_area
        fnorms = gt_mesh.facets_normal
    except Exception:
        return ops, iou
    if len(facets) == 0:
        return ops, iou
    # A real cut FACE is one of the largest planar facets; a revolve's lateral
    # surface tessellates into MANY large strip facets (Bushing: 255), so cap the
    # candidates and skip facets whose plane passes THROUGH the part (a side-wall
    # half-space would slice the body in half) -- only end/oblique caps qualify.
    fareas = np.asarray(fareas)
    rmax = float(np.max(np.linalg.norm(gt_mesh.vertices - gc, axis=1)))
    for fi in np.argsort(-fareas)[:10]:
        if fareas[fi] < 0.004 * ext * ext:
            break
        faces = facets[fi]
        pts = gt_mesh.vertices[np.unique(gt_mesh.faces[faces].ravel())]
        c0 = pts.mean(axis=0)
        n = np.asarray(fnorms[fi], float)
        if n @ (gc - c0) < 0:
            n = -n                                         # point INTO kept material
        if (gc - c0) @ n < 0.15 * rmax:                    # plane near/through centre
            continue                                       # -> would bisect the part
        # ANALYTIC half-space occupancy: removed side is (P - c0)·n < 0.  (contains()
        # on a huge box over ~900k grid points costs 22s; the dot product is instant.)
        removed = ((P - c0) @ n) < 0.0
        occ2 = occ & ~removed
        iou2 = float((occ2 & G).sum() / max((occ2 | G).sum(), 1))
        if iou2 >= iou + min_gain:                         # this plane cuts real overfill
            occ, iou = occ2, iou2
            pc = PrimitiveCandidate(mesh=_halfspace_box(c0, n, ext), program="",
                                    kind="planar_cut")
            pc.plane = (c0.tolist(), n.tolist(), ext)      # for emit + render
            ops.append(("cut", pc))
    return ops, iou


def _revolve_with_residual_cuts(base: "PrimitiveCandidate", gt_mesh: trimesh.Trimesh,
                                P: np.ndarray, G: np.ndarray, pitch: float,
                                min_gain: float):
    """Revolve base (op 1) + residual CUTS.  An axisymmetric revolve usually
    OVER-fills (predict\\GT non-empty: a recessed/flat/notched end, a window),
    while GT\\predict is ~empty.  Approximate that overfill WITH DETECTORS
    (single_pass -> clean extrude pieces), over-pierce them, and cut -- so the
    program is `r = <revolve>; r = r.cut(<extrude>) ...`, emittable and multi-op.
    Returns (ops, iou)."""
    occ = _occupancy(base.mesh, P, pitch)
    if occ is None:
        return [("union", base)], 0.0
    ops = [("union", base)]
    iou = float((occ & G).sum() / max((occ | G).sum(), 1))
    try:
        from .residual import compute_residuals
        cm = compute_residuals(base.mesh, gt_mesh).cut_mesh        # predict \ GT
    except Exception:
        cm = None
    if cm is None or len(cm.faces) == 0:
        return ops, iou
    try:
        r_kept, _, _ = cadfit_single_pass(cm, pitch=pitch, min_gain=min_gain,
                                          verbose=False)
    except Exception:
        r_kept = []
    margin = 0.02 * float(np.max(gt_mesh.extents))
    for c in sorted(r_kept, key=lambda c: -abs(c.mesh.volume)):
        cc = _extend_extrude(c, margin)                           # over-pierce
        o = _occupancy(cc.mesh, P, pitch)
        if o is None:
            continue
        occ2 = occ & ~o
        iou2 = float((occ2 & G).sum() / max((occ2 | G).sum(), 1))
        if iou2 >= iou + min_gain:
            occ, iou = occ2, iou2
            ops.append(("cut", cc))
    return ops, iou


def _emit_primitive_program(ops: list) -> str:
    """Emit a CadQuery program for a reconstruction made of primitive ops
    (revolve/sweep/loft).  Each carries a standalone program; a single op
    returns its program verbatim, multiple are unioned by their expressions."""
    progs = [c.program for _, c in ops if getattr(c, "program", "")]
    if not progs:
        return "import cadquery as cq\nresult = cq.Workplane('XY')"
    if len(progs) == 1:
        return progs[0]
    exprs = [p.split("result = ", 1)[1].strip() for p in progs if "result = " in p]
    if not exprs:
        return progs[0]
    body = exprs[0]
    for e in exprs[1:]:
        body = f"({body}).union({e})"
    return "import cadquery as cq\nresult = " + body


def cadfit_reconstruct(gt_mesh: trimesh.Trimesh,
                       n_residual_iters: int = 1,
                       pitch: float = 2.0,
                       min_gain: float = 0.004,
                       max_residual_ops: int = 10,
                       verbose: bool = True):
    """Single pass + bounded residual iterations.  Composition is done in
    occupancy space:  occ = (base | R+ recon) & ~(R- recon).
    Returns (ops, iou, info): ops = list of (op, Candidate), op in {union, cut}.
    """
    from .residual import compute_residuals
    import os as _os, time as _time
    _t0 = _time.time()
    # wall-clock budget: the residual pass (boolean.union + compute_residuals +
    # a full single_pass on the residual mesh) can run for >60s and push a part
    # past the harness timeout -- which returns NOTHING (0 IoU).  When the budget
    # would be blown, we instead KEEP the strong base result.  fillet_hole's base
    # is 0.956, Housing 0.729: a timeout threw both away.  Env-tunable.
    _budget = float(_os.environ.get("DET_TIME_BUDGET", "95"))
    _huge = len(gt_mesh.faces) > 130000          # huge GT -> residual booleans too slow
    # soft deadline for the inner loops (sweeps / prune / collapse): stop
    # collecting and assemble what exists, leaving headroom for the emit.
    # The pre-existing SIGALRM in the harness cannot interrupt long C calls,
    # so parts used to overshoot the budget 5-6x; this cap is cooperative.
    _deadline = _t0 + 0.92 * _budget

    P = _grid(gt_mesh, pitch)
    G = _occupancy(gt_mesh, P, pitch)
    if G is None:
        return [], 0.0, {"error": "gt voxelization failed"}

    def _op_occ(c):
        """Occupancy of a kept op on the shared grid: analytic for extrudes
        (same fast path assemble uses), contains() for primitive meshes."""
        o = _occ_extrude(c, P) if getattr(c, "profile", None) is not None else None
        if o is None:
            o = _occupancy(c.mesh, P, pitch)
        return o

    # PRIMITIVE-FIRST (revolve): a turned part is ONE revolve, not a stack of ~15
    # extrude discs.  Detect the best whole-part revolve up front; if it already
    # explains the part (>=0.95), return it directly (clean + fast, skips the
    # disc-stack).  Revolve is unambiguous (axisymmetry-gated), so early-exit is
    # safe here; sweep/loft are compared only at the end (below), where the
    # extrude result is known, to avoid grabbing a part an extrude does cleaner.
    # detect_revolve does 12 mesh.section() calls (4 angles x 3 axes); on a large
    # mesh that costs ~6s and tips slow parts (Coupling 19k, Rack 23k, Hook 14k faces)
    # over the harness timeout.  Turned parts that benefit are all small (<4k faces),
    # so gate detection by size -- cheap sections only.
    _rev = None
    if len(gt_mesh.faces) <= 10000:
        _rev = _best_primitive(gt_mesh, P, G, pitch, kinds=("revolve",))
    if _rev is not None and _rev[1] >= 0.95:
        # revolve base (op 1) + residual cuts (op 2..): cut the overfill (predict\GT)
        # approximated by detectors.  Stopper 0.971 -> ~0.99 as revolve + cut.
        e_ops, e_iou = _revolve_planar_cuts(_rev[0], gt_mesh, P, G, pitch, min_gain)
        if verbose:
            print(f"[revolve] revolve + {len(e_ops)-1} residual cut(s) IoU={e_iou:.3f}")
        return e_ops, e_iou, {"primitive": "revolve",
                              "iou_final": e_iou, "n_ops": len(e_ops)}

    kept, iou0, info = cadfit_single_pass(gt_mesh, pitch=pitch, min_gain=min_gain, verbose=verbose,
                                          deadline=_deadline, grid_pts=P, gt_occ=G)
    ops = [("union", c) for c in kept]
    occ = np.zeros(len(P), dtype=bool)
    for c in kept:
        o = _op_occ(c)
        if o is not None:
            occ |= o
    iou = float((occ & G).sum() / max((occ | G).sum(), 1))
    if verbose:
        print(f"[base] IoU={iou:.3f} ({len(kept)} union ops)")

    # multi-diameter tip continuation (small end segments of rods/bosses).
    try:
        gp, _ = trimesh.sample.sample_surface(gt_mesh, 20000, seed=42)
        gtree = cKDTree(np.asarray(gp))
        ext = float(np.max(gt_mesh.extents))
        for tc in tip_extensions([o[1] for o in ops], gt_mesh, gtree, ext,
                                 verbose=verbose):
            o = _op_occ(tc)
            if o is None:
                continue
            occ2 = occ | o
            iou2 = float((occ2 & G).sum() / max((occ2 | G).sum(), 1))
            if iou2 >= iou + min_gain * 0.25:
                occ, iou = occ2, iou2
                ops.append(("union", tc))
                if verbose:
                    print(f"    + tip extension -> IoU={iou:.3f}")
    except Exception as e:
        if verbose:
            print(f"[tips] skipped: {e}")

    for it in range(n_residual_iters):
        # BUDGET GUARD: skip the (expensive) residual pass rather than risk a hard
        # timeout that discards the base result.  Three triggers, all returning the
        # current best-so-far (>= base, since residual only ADDS gain-positive ops):
        #   - base already near-perfect (residual gain would be marginal anyway)
        #   - huge GT mesh (compute_residuals booleans dominate, e.g. 168k-face Housing)
        #   - too little budget left for one residual iter (its single_pass on the
        #     residual mesh costs ~ the base single_pass).
        _elapsed = _time.time() - _t0
        if iou >= 0.95 or _huge or _elapsed > 0.55 * _budget:
            if verbose:
                print(f"[budget] skip residual: iou={iou:.3f} huge={_huge} "
                      f"elapsed={_elapsed:.0f}s/{_budget:.0f}s")
            break
        # current solid mesh (boolean union of kept pieces; fall back: skip)
        try:
            S = trimesh.boolean.union([o[1].mesh for o in ops if o[0] == "union"])
            if isinstance(S, list):
                S = trimesh.util.concatenate(S)
        except Exception:
            S = None
        if S is None or len(S.faces) == 0:
            break
        res = compute_residuals(S, gt_mesh)
        changed = False
        for side, rmesh in (("union", res.add_mesh), ("cut", res.cut_mesh)):
            if rmesh is None or len(rmesh.faces) == 0:
                continue
            try:
                rvol = float(abs(rmesh.volume))
            except Exception:
                rvol = 0.0
            if rvol < 1e-4 * float(abs(gt_mesh.volume)):
                continue
            r_kept, r_iou, _ = cadfit_single_pass(rmesh, pitch=pitch, min_gain=min_gain, verbose=False,
                                                  deadline=_deadline)
            if verbose:
                print(f"[residual {it+1}] {side}: recon IoU={r_iou:.3f} ({len(r_kept)} ops)")
            n_acc = 0
            for c in r_kept:
                if n_acc >= max_residual_ops:
                    break
                # ROUND small pieces are real features (tip lips, pins) -- only
                # POLYGON slivers re-box clean geometry.  Circle-fit pieces get
                # a lower area floor and a lower gain bar.
                fcirc = _fit_circle_2d(_ring_coords(c.profile.polygon.exterior))
                is_round = fcirc is not None and fcirc[3] < 1.0
                if c.profile.area < (60.0 if is_round else 120.0):
                    continue              # sliver patches re-box clean features
                gain_bar = (min_gain * 0.25) if is_round else min_gain
                # OBLIQUE polygon sections (residual cluster planes) emit
                # CadQuery polygons that OCCT renders unreliably (near-
                # collinear point runs on a skew plane) -- a marginal voxel
                # gain is not worth the render risk, so demand a big one.
                # (GT-scored A/B: marginal oblique residual ops on parts
                # already at IoU>=0.8 were net NEGATIVE on true-GT GMS.)
                _n3 = c.profile.to_3d[:3, 2]
                if not is_round and float(np.max(np.abs(_n3))) < 0.99:
                    gain_bar = max(gain_bar, 0.02)
                o = _op_occ(c)
                if o is None:
                    continue
                # accept only ops that EXPLAIN a feature: global IoU gain >= bar
                occ2 = (occ | o) if side == "union" else (occ & ~o)
                iou2 = float((occ2 & G).sum() / max((occ2 | G).sum(), 1))
                if iou2 >= iou + gain_bar:
                    occ, iou = occ2, iou2
                    ops.append((side, c)); changed = True; n_acc += 1
                    if verbose:
                        print(f"    + {side} piece -> IoU={iou:.3f}")
        if not changed:
            break

    try:
        ops, occ, iou, n_pat = _complete_circular_patterns(ops, occ, iou, G, P, pitch)
        if verbose and n_pat:
            print(f"[pattern] +{n_pat} completed instances -> IoU={iou:.3f}")
    except Exception as e:
        if verbose:
            print(f"[pattern] skipped: {e}")

    # LATE primitive compare: if a single revolve/sweep/loft fits the part about
    # as well as the whole extrude assembly, prefer it -- one clean primitive beats
    # a disc stack or a straight box (Vase revolve vs 17 discs; Spike round sweep
    # vs a square box).  Sweep/loft are only tried when the extrude left room
    # (<0.95), so parts an extrude already nails (sheets/brackets) skip the cost.
    # LATE revolve compare (sweep/loft are NOT tried: they always score below the
    # extrude baseline yet their section analysis is expensive -- running it on every
    # sub-0.95 part re-introduced timeouts.  Revolve only.).  Carry the residual cuts
    # so it wins AND keeps its overfill-cut features.
    if _rev is not None and _rev[1] >= iou - 0.02 and _rev[1] >= 0.80:
        e_ops, e_iou = _revolve_planar_cuts(_rev[0], gt_mesh, P, G, pitch, min_gain)
        if e_iou >= iou - 0.02:
            if verbose:
                print(f"[primitive] revolve + {len(e_ops)-1} cut(s) IoU={e_iou:.3f} "
                      f"replaces {len(ops)} extrude ops (assembly IoU={iou:.3f})")
            ops, iou = e_ops, e_iou
            info["primitive"] = "revolve"

    # REVOLVE ENVELOPE + CUTS (prototype, DISABLED): a turned body with windows/holes
    # (housing/cup/tank) is shattered into a box cage by the extrude pipeline.  A
    # max-radius revolve envelope + wholesale cut of the (envelope\GT) residual hits
    # cad23 0.46->0.98 LOCALLY -- but the boolean cut is environment-flaky (server:
    # 0.69, cut doesn't fire) and roughly doubles runtime on low-IoU parts.  Helpers
    # (_revolve_envelope / _best_revolve_envelope / _envelope_with_cuts) are kept for
    # future work; re-enable once the cut is deterministic (window detection instead
    # of a raw boolean) and gated for speed.  See [[det_revolve_loft_sweep]].
    if _os.environ.get("DET_REVOLVE_ENVELOPE") and iou < 0.85:
        _env = _best_revolve_envelope(gt_mesh, P, G, pitch)
        if _env is not None and _env[2] >= 0.92 and _env[1] >= 0.50:
            e_ops, e_iou = _envelope_with_cuts(_env[0], gt_mesh, P, G, pitch, min_gain)
            if e_iou > iou:
                if verbose:
                    print(f"[revolve-env] envelope+cuts IoU={e_iou:.3f} "
                          f"({len(e_ops)} ops) beats extrude {iou:.3f}")
                ops, iou = e_ops, e_iou
                info["primitive_base"] = "revolve_envelope"

    info["iou_final"] = iou
    info["n_ops"] = len(ops)
    # surfaced so the harness can refuse to CACHE a truncated (esp. empty)
    # result -- a resume would otherwise skip the part forever with no marker.
    info["deadline_hit"] = bool(_time.time() > _deadline)
    return ops, iou, info


def _complete_circular_patterns(ops, occ, iou, G, P, pitch):
    """Rotational-array completion.  Teeth / bolt-circles / pin rings come out
    of the residual pass as INDEPENDENT small ops with instances missing
    (Toothed coupling 0.74: several teeth merged or skipped).  Find groups of
    >=3 small coaxial ops on a common circle, infer the instance count M from
    the angular gaps, and add the missing rotated copies -- each accepted
    only if it lies on target material (>=60%) and does not hurt IoU."""
    from collections import defaultdict
    AXIJ = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    mem = []
    for k, (side, c) in enumerate(ops):
        if side != "union" or c.profile.area > 2000.0:
            continue
        n = c.profile.to_3d[:3, 2]
        ax = int(np.argmax(np.abs(n)))
        if abs(n[ax]) < 0.99:
            continue
        cen = c.profile.polygon.representative_point()
        w0 = (c.profile.to_3d @ np.array([cen.x, cen.y,
                                          0.5 * (c.t0 + c.t1), 1.0]))[:3]
        mem.append((k, ax, w0, c))
    groups = defaultdict(list)
    for k, ax, w0, c in mem:
        key = (ax, round(c.profile.area / 30.0),
               round(abs(c.t1 - c.t0) / 5.0))
        groups[key].append((ax, w0, c))
    added = 0
    for key, g in groups.items():
        if len(g) < 3:
            continue
        ax = g[0][0]
        i, j = AXIJ[ax]
        pts = np.array([m[1] for m in g])
        cx, cy = float(pts[:, i].mean()), float(pts[:, j].mean())
        rr = np.hypot(pts[:, i] - cx, pts[:, j] - cy)
        if rr.mean() < 5.0 or rr.std() > 0.15 * rr.mean():
            continue                       # members not on a common circle
        angs = np.arctan2(pts[:, j] - cy, pts[:, i] - cx)
        srt = np.sort(angs)
        gaps = np.diff(np.concatenate([srt, [srt[0] + 2 * np.pi]]))
        base = float(np.median(gaps[gaps > 1e-3]))
        M = int(round(2 * np.pi / base))
        if M < 4 or M > 64 or abs(2 * np.pi / M - base) > 0.25 * base:
            continue
        step = 2 * np.pi / M
        a0 = float(srt[0])
        have = set(int(round((a - a0) / step)) % M for a in angs)
        proto = g[0][2]
        a_p = float(np.arctan2(g[0][1][j] - cy, g[0][1][i] - cx))
        for s in range(M):
            if s in have:
                continue
            dth = (a0 + s * step) - a_p
            ca, sa = math.cos(dth), math.sin(dth)
            R4 = np.eye(4)
            R4[i, i] = ca; R4[i, j] = -sa; R4[j, i] = sa; R4[j, j] = ca
            T4 = np.eye(4); T4[i, 3] = cx; T4[j, 3] = cy
            T4i = np.eye(4); T4i[i, 3] = -cx; T4i[j, 3] = -cy
            M4 = T4 @ R4 @ T4i
            nm = proto.mesh.copy()
            nm.apply_transform(M4)
            npr = Profile(polygon=proto.profile.polygon,
                          to_3d=M4 @ proto.profile.to_3d,
                          source=proto.profile.source,
                          area=proto.profile.area)
            ncand = Candidate(mesh=nm, profile=npr, t0=proto.t0, t1=proto.t1,
                              ring_err=proto.ring_err)
            # analytic occupancy of the rotated extrude (contains() on the
            # full grid per candidate instance dominated pattern completion)
            o = _occ_extrude(ncand, P)
            if o is None:
                o = _occupancy(nm, P, pitch)
            if o is None:
                continue
            on_mat = float((o & G).sum()) / max(float(o.sum()), 1.0)
            occ2 = occ | o
            iou2 = float((occ2 & G).sum() / max((occ2 | G).sum(), 1))
            if on_mat < 0.6 or iou2 < iou:
                continue
            ops.append(("union", ncand))
            occ, iou = occ2, iou2
            added += 1
    return ops, occ, iou, added


# --------------------------------------------------------------------------
# CadQuery code emission: ops -> runnable program (CADRecode-style sketches)
# --------------------------------------------------------------------------

def _fit_circle_2d(coords: np.ndarray):
    """Algebraic circle fit; returns (cx, cy, r, rms) or None."""
    if len(coords) < 8:
        return None
    x, y = coords[:, 0], coords[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return cx, cy, r, rms


def _ring_coords(ring, max_pts: int = 200) -> np.ndarray:
    c = np.asarray(ring.coords)[:-1]
    if len(c) > max_pts:
        idx = np.linspace(0, len(c) - 1, max_pts).astype(int)
        c = c[idx]
    return c


def _ring_coords_uniform(ring, n: int = 160) -> np.ndarray:
    """Arc-length-uniform boundary samples for shape tests.  Raw polygon
    VERTICES cluster on curved spans (an arc gets dozens of vertices, a
    straight edge two), which biases circle fits: a stadium's end-arcs
    dominate and it 'fits' a circle at rms<0.3 (Disc ovals emitted as
    discs, self-IoU 0.53; Vase crescents at 0.03).  Uniform sampling makes
    the fit see the true shape."""
    pts = [ring.interpolate(ring.length * i / n) for i in range(n)]
    return np.asarray([[p.x, p.y] for p in pts], float)


def _valid_ring_pts(ring, tol: float):
    """Points for a segment-chain ring that cadquery can actually build:
    simplified at ``tol`` but VALIDATED -- non-self-intersecting (gear teeth
    at coarse tol produce bowties -> assemble() yields no face ->
    'extrudeLinear: 0 methods'), no duplicate consecutive points after
    4-decimal rounding (-> 'inner wire is not closed').  Falls back to finer
    tolerance, then to a decimated raw ring; None if nothing valid."""
    from shapely.geometry import LinearRing
    # FIDELITY: the recon mesh is built from the FULL-resolution profile
    # (trimesh.creation.extrude_polygon(c.profile.polygon)), so a coarse
    # Douglas-Peucker simplify here (tol=circle_tol~1.2) decimated rounded
    # outlines by up to ~1.8u -> the cadquery rebuild of the emitted program
    # diverged from the recon (GMS 0.997->0.88 on a flat plate).  Try a FINE
    # tolerance first (faithful boundary); fall back to coarser / raw 240-pt
    # only if a finer ring self-intersects (bowtie -> invalid wire).
    candidates = [0.04, 0.15, tol, tol / 3.0, -1.0, -2.0]
    for t in candidates:
        try:
            if t == -2.0:
                # last resort: shapely buffer(0) self-intersection repair,
                # keep the largest piece (frame outlines that stay invalid
                # even raw -- part 06's silently dropped op)
                from shapely.geometry import Polygon as _Pg
                rep = _Pg(np.asarray(ring.coords)).buffer(0)
                if rep.is_empty:
                    continue
                if rep.geom_type == "MultiPolygon":
                    rep = max(rep.geoms, key=lambda g: g.area)
                c = _ring_coords(rep.exterior, 240)
            elif t > 0:
                c = np.asarray(ring.simplify(t).coords)[:-1]
            else:
                c = _ring_coords(ring, 240)
        except Exception:
            continue
        pts = [(round(float(x), 4), round(float(y), 4)) for x, y in c]
        ded = [p for i, p in enumerate(pts) if i == 0 or
               abs(p[0] - pts[i - 1][0]) > 2e-3 or abs(p[1] - pts[i - 1][1]) > 2e-3]
        if len(ded) >= 3 and abs(ded[0][0] - ded[-1][0]) <= 2e-3 \
                and abs(ded[0][1] - ded[-1][1]) <= 2e-3:
            ded = ded[:-1]
        if len(ded) < 3:
            continue
        try:
            lr = LinearRing(ded)
            if lr.is_valid and lr.is_simple:
                return ded
        except Exception:
            continue
    return None


def _sketch_frag(poly, circle_tol: float, jit: float = 0.0) -> str:
    """Sketch fragment for a polygon WITH holes.  Outer ring: circle if it fits
    (rms < circle_tol), else simplified polygon.  Holes: circles when circular,
    else simplified polygon subtracted (mode='s' on assemble).

    cadquery Sketch GOTCHA: ``push([(x,y)])`` locations stay ACTIVE for every
    later face op -- and ``assemble()`` distributes its face over the current
    locations.  A segment-chain hole emitted after a push-circle hole would be
    subtracted TRANSLATED by the circle's center (shredded-walls bug,
    hole_Housing_hole 0.79->0.33).  Reset with ``.push([(0.0,0.0)])`` before
    any segment chain that follows a push."""
    from . import cadrecode_emit as cre
    out = ""
    pushed = False
    ext = _ring_coords_uniform(poly.exterior)
    fit = _fit_circle_2d(ext)
    def _poly_call(pts, sub=False):
        # Sketch.polygon: ONE self-contained face op.  Segment chains are a
        # trap: Sketch.assemble() builds its face from ALL accumulated
        # _edges (never cleared), so a second chain in the same sketch
        # silently bundles the outer ring's edges with its own -- works or
        # fails depending on geometry (Flange-12 / gear-36 exec failures).
        body = ",".join(f"({x:.4f},{y:.4f})" for x, y in pts)
        closed = body + f",({pts[0][0]:.4f},{pts[0][1]:.4f})"
        m = ", mode='s'" if sub else ""
        return f".polygon([{closed}]{m})"

    if fit is not None and fit[3] < circle_tol:
        out += cre.circle_frag(fit[0], fit[1], fit[2])
        pushed = True
    else:
        pts = _valid_ring_pts(poly.exterior, circle_tol)
        if pts is None:
            return ""
        if pushed:
            out += ".push([(0.0,0.0)])"
        out += _poly_call(pts)
    for hole in poly.interiors:
        hc = _ring_coords_uniform(hole)
        hf = _fit_circle_2d(hc)
        if hf is not None and hf[3] < circle_tol:
            out += cre.circle_frag(hf[0], hf[1], hf[2] + jit, sub=True)
            pushed = True
        else:
            pts = _valid_ring_pts(hole, circle_tol)
            if pts is not None:
                if pushed:
                    out += ".push([(0.0,0.0)])"
                out += _poly_call(pts, sub=True)
                pushed = False
    return out


def emit_cadquery(ops: list, circle_tol: float = 1.2) -> str:
    """Full CadQuery program for [(op, Candidate), ...]:
    r = <eq0>; r = r.union(<eq1>) / r.cut(<eqk>) ..."""
    lines = ["import cadquery as cq"]
    expr_n = 0
    for side, c in ops:
        T = c.profile.to_3d
        o3 = (T @ np.array([0.0, 0.0, c.t0, 1.0]))[:3]
        xd = T[:3, 0]; nrm = T[:3, 2]
        plane = (f"cq.Plane(origin=cq.Vector({o3[0]:.4f},{o3[1]:.4f},{o3[2]:.4f}),"
                 f" xDir=cq.Vector({xd[0]:.4f},{xd[1]:.4f},{xd[2]:.4f}),"
                 f" normal=cq.Vector({nrm[0]:.4f},{nrm[1]:.4f},{nrm[2]:.4f}))")
        frag = _sketch_frag(c.profile.polygon, circle_tol)
        if not frag:
            continue
        expr = f"cq.Workplane({plane}).sketch(){frag}.finalize().extrude({c.t1 - c.t0:.4f})"
        if expr_n == 0:
            if side == "cut":
                continue                    # a cut cannot be the first op
            lines.append(f"r = {expr}")
        else:
            verb = "union" if side == "union" else "cut"
            lines.append(f"r = r.{verb}({expr})")
        expr_n += 1
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Design-intent emission: anchored workplanes (partial constraint recovery)
# --------------------------------------------------------------------------

def _op_axis_info(c):
    """(axis_id 0/1/2 or None, lo, hi, area) for an axis-aligned op, in world
    coords along its axis (lo/hi sorted)."""
    n = c.profile.to_3d[:3, 2]
    axis = int(np.argmax(np.abs(n)))
    if abs(n[axis]) < 0.99:
        return None, 0.0, 0.0, c.profile.area
    cen = c.profile.polygon.representative_point()
    pts = [(c.profile.to_3d @ np.array([cen.x, cen.y, t, 1.0]))[:3][axis]
           for t in (c.t0, c.t1)]
    lo, hi = sorted(float(v) for v in pts)
    return axis, lo, hi, c.profile.area


def infer_relations(ops, tol: float = 3.0):
    """Attachment graph for design-intent emission.
    Returns {child_i: (parent_j, kind, anchor)} with kind in
    {"on_top", "under", "on_face+"," on_face-"}; anchor = world coord.
    Only UNION ops can be parents: a cut op's mouth has no material, so
    attaching there makes PointOnFaceSelector snap to the hole wall and
    the feature extrudes sideways (valve bottom-collar bug)."""
    info = [_op_axis_info(c) for _, c in ops]
    rel = {}
    for i, (ai, lo_i, hi_i, area_i) in enumerate(info):
        if ai is None:
            continue
        best = None
        for j, (aj, lo_j, hi_j, area_j) in enumerate(info):
            if i == j or aj is None or ops[j][0] != "union":
                continue
            if ai == aj:                       # same axis: cap attachments
                # 2x tol: a boss may be partially EMBEDDED in its parent; the
                # emitter re-bases it on the parent face (h_eff correction),
                # which yields the same final solid for union children.
                if abs(lo_i - hi_j) < 2 * tol and area_j > 0.5 * area_i:
                    cand = (j, "on_top", hi_j, area_j)      # child base on parent top cap
                elif abs(hi_i - lo_j) < 2 * tol and area_j > 0.5 * area_i:
                    cand = (j, "under", lo_j, area_j)       # child top under parent base cap
                else:
                    continue
            else:                              # perpendicular: lateral face
                # parent's lateral extent along the CHILD's axis
                ring = _ring_coords(ops[j][1].profile.polygon.exterior, 64)
                Tj = ops[j][1].profile.to_3d
                w = (Tj @ np.concatenate([ring, np.full((len(ring), 1), ops[j][1].t0),
                                          np.ones((len(ring), 1))], axis=1).T).T[:, :3]
                pj = w[:, ai]
                plo, phi = float(pj.min()), float(pj.max())
                if abs(lo_i - phi) < tol:
                    cand = (j, "on_face+", phi, area_j)
                elif abs(hi_i - plo) < tol:
                    cand = (j, "on_face-", plo, area_j)
                else:
                    continue
                # the parent's BODY must actually be there: project the
                # child's BASE-end center (the end that touches the face --
                # the mid/far end protrudes outside by construction) into
                # the parent's local frame: axially inside the parent's
                # span AND on/near its footprint.  (extent-only matching
                # attached bracket ears to a plate 60u below them; the
                # attach then survived only by luck.)
                from shapely.geometry import Point as _P
                cen_i = ops[i][1].profile.polygon.representative_point()
                Ti = ops[i][1].profile.to_3d
                ends = []
                for t in (ops[i][1].t0, ops[i][1].t1):
                    e = (Ti @ np.array([cen_i.x, cen_i.y, t, 1.0]))[:3]
                    ends.append(e)
                # base end = the one at the attachment side along child axis
                base = min(ends, key=lambda e: e[ai]) if cand[1] == "on_face+" \
                    else max(ends, key=lambda e: e[ai])
                lj = (np.linalg.inv(Tj) @ np.array([base[0], base[1], base[2], 1.0]))[:3]
                tj0, tj1 = sorted((ops[j][1].t0, ops[j][1].t1))
                if not (tj0 - 1.0 < lj[2] < tj1 + 1.0):
                    continue
                if not ops[j][1].profile.polygon.buffer(1.5).contains(_P(lj[0], lj[1])):
                    continue
            if best is None or cand[3] > best[3]:
                best = cand
        if best is not None:
            rel[i] = best[:3]
    return rel


def _anchor_on_material(w, ops, pj, margin: float = 1.2, lateral: bool = False,
                        ci: int = -1):
    """True if world point ``w`` lies on actual material of parent pj's
    surface.  Cap anchors (lateral=False): inside the parent's profile
    (holes excluded, ``margin`` from any rim).  Lateral anchors: ON the
    profile's outer rim (within 0.8) and axially inside the parent's span.
    Either way the point must not fall inside a cut op crossing it."""
    from shapely.geometry import Point
    _, cp = ops[pj]
    Ti = np.linalg.inv(cp.profile.to_3d)
    l = (Ti @ np.array([w[0], w[1], w[2], 1.0]))[:3]
    if lateral:
        t0, t1 = sorted((cp.t0, cp.t1))
        if not (t0 + margin < l[2] < t1 - margin):
            return False
        if cp.profile.polygon.exterior.distance(Point(l[0], l[1])) > 0.8:
            return False
    else:
        core = cp.profile.polygon.buffer(-margin)
        if core.is_empty or not core.contains(Point(l[0], l[1])):
            return False
    for k, (side_k, ck) in enumerate(ops):
        Tk = np.linalg.inv(ck.profile.to_3d)
        lk = (Tk @ np.array([w[0], w[1], w[2], 1.0]))[:3]
        t0, t1 = sorted((ck.t0, ck.t1))
        if not (t0 + 0.2 < lk[2] < t1 - 0.2):
            continue
        if side_k == "cut":
            # cut crossing the plane removes face material around the anchor
            if ck.profile.polygon.buffer(margin).contains(Point(lk[0], lk[1])):
                return False
        elif k != pj and k != ci:
            # another union op BURIES the anchor: by the time this feature is
            # attached the point may be interior, and PointOnFaceSelector
            # would grab an arbitrary nearby face (zero-norm xDir crash)
            if ck.profile.polygon.buffer(-0.2).contains(Point(lk[0], lk[1])):
                return False
        # WALL-TIE guard (cap anchors): a point lying ON any op's lateral wall
        # (its profile boundary at this height) ties with that wall in the
        # face selector -- e.g. an anchor at exactly a coaxial bore's radius
        # grabs the bore cylinder, radial normal, zero-norm crash (Valve-24).
        if not lateral and k != ci:
            pt = Point(lk[0], lk[1])
            if ck.profile.polygon.exterior.distance(pt) < 1.0:
                return False
            for hole in ck.profile.polygon.interiors:
                if hole.distance(pt) < 1.0:
                    return False
    return True


def _lateral_edge_slide(w, ops, pj, n3, ci: int = -1,
                        clear: float = 2.5):
    """Anchor for a LATERAL attach must stay ``clear`` away from profile
    corners: at a corner the nearest-face tie is broken arbitrarily and
    PointOnFaceSelector may grab the perpendicular face (wheel-orientation
    bug, part 08).  Slide the anchor along the rim SEGMENT whose outward
    normal matches the child's axis.  Returns adjusted world point, or
    None if no safe spot exists on that segment."""
    from shapely.geometry import Point
    _, cp = ops[pj]
    Tj = cp.profile.to_3d
    Ti = np.linalg.inv(Tj)
    l = (Ti @ np.array([w[0], w[1], w[2], 1.0]))[:3]
    # child axis direction in the parent's 2D profile frame
    n2 = (Ti[:3, :3] @ np.asarray(n3, float))[:2]
    if np.linalg.norm(n2) < 0.5:
        return w                      # degenerate; keep as-is
    n2 = n2 / np.linalg.norm(n2)
    ext = cp.profile.polygon.exterior
    ccw = 1.0 if ext.is_ccw else -1.0
    co = np.asarray(ext.coords)       # closed ring
    best = None
    for i in range(len(co) - 1):
        p, q = co[i], co[i + 1]
        t = q - p
        L = float(np.hypot(t[0], t[1]))
        if L < 2 * clear:
            continue
        th = t / L
        seg_n = ccw * np.array([th[1], -th[0]])
        if float(seg_n @ n2) < 0.95:
            continue                  # wrong face direction
        s = float(np.clip((np.array([l[0], l[1]]) - p) @ th, clear, L - clear))
        pt = p + s * th
        d = float(np.hypot(pt[0] - l[0], pt[1] - l[1]))
        if best is None or d < best[0]:
            best = (d, pt)
    if best is None:
        return None
    wn = (Tj @ np.array([best[1][0], best[1][1], l[2], 1.0]))[:3]
    if not _anchor_on_material(wn, ops, pj, lateral=True, ci=ci):
        return None
    return wn


def _find_material_anchor(w, fit_r, xd, yd, ops, pj, ci: int = -1):
    """Search a face-material point near ``w`` (the child circle's center on
    the parent plane): ring of offsets inside the child's own footprint.
    Returns world point or None."""
    for fr in (0.55, 0.7, 0.4, 0.85, 0.25):
        for ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            a = w + (fr * fit_r) * (math.cos(ang) * xd + math.sin(ang) * yd)
            if _anchor_on_material(a, ops, pj, ci=ci):
                return a
    return None


def emit_cadquery_vlm(ops: list, circle_tol: float = 1.2) -> str:
    """CadQuery program in the VLM's OWN format (same imports, same chain
    grammar) with PARTIAL DESIGN INTENT: circular children that sit on a
    parent's cap or lateral face are emitted via cadgen's ``_wp_attach_at``
    -- the workplane is DERIVED from the face of the solid built so far
    (PointOnFaceSelector + tangent plane), so their locations are constrained
    to the parent surface instead of hanging at absolute coordinates.
    Polygon ops and unattached features stay as plain absolute-plane blocks,
    exactly like the VLM's outputs."""
    # primitive ops (revolve/sweep/loft) carry a ready program in the CADRecode
    # `result = <expr>` form; the extrude anchoring grammar below assumes .profile
    # on every op.  Emit any reconstruction that contains a primitive in the SAME
    # composable VLM chain as the extrude ops -- `r = <expr>` for the first op then
    # `r = r.union(<expr>)` / `r = r.cut(<expr>)` -- so a revolve is the base OR a
    # 2nd/3rd union/cut op, never a standalone `result = ...`.
    if any(getattr(c, "kind", None) in ("revolve", "sweep", "loft") for _, c in ops):
        has_ext = any(getattr(c, "profile", None) is not None for _, c in ops)

        def _expr_of(c):
            if getattr(c, "kind", None) in ("revolve", "sweep", "loft"):
                p = getattr(c, "program", "") or ""
                return p.split("result = ", 1)[1].strip() if "result = " in p else ""
            if getattr(c, "kind", None) == "planar_cut" and getattr(c, "plane", None):
                # half-space cut: a big rect on the plane, extruded into the removed side
                (cx, cy, cz), (nx, ny, nz), e3 = c.plane
                n = np.array([nx, ny, nz], float); n = n / (np.linalg.norm(n) + 1e-9)
                xd = np.cross(n, [0.0, 0.0, 1.0])
                if np.linalg.norm(xd) < 1e-3:
                    xd = np.cross(n, [1.0, 0.0, 0.0])
                xd = xd / (np.linalg.norm(xd) + 1e-9)
                L = 3.0 * float(e3)
                return (f"cq.Workplane(cq.Plane(origin=cq.Vector({cx:.4f},{cy:.4f},{cz:.4f}), "
                        f"xDir=cq.Vector({xd[0]:.4f},{xd[1]:.4f},{xd[2]:.4f}), "
                        f"normal=cq.Vector({n[0]:.4f},{n[1]:.4f},{n[2]:.4f})))"
                        f".rect({L:.2f},{L:.2f}).extrude({-L:.2f})")
            if getattr(c, "profile", None) is None:
                return ""                       # mesh-only op (no emittable program)
            try:                                # extrude op: lift its bare expression
                te = emit_cadquery_vlm([("union", c)], circle_tol)
            except Exception:
                return ""
            for ln in te.splitlines():
                if ln.startswith("r = "):
                    return ln[4:].strip()
            return ""

        lines = []
        for side, c in ops:
            e = _expr_of(c)
            if not e:
                continue
            if not lines:
                lines.append(f"r = {e}")
            else:
                lines.append(f"r = r.{'cut' if side == 'cut' else 'union'}({e})")
        if lines:
            header = "import cadquery as cq\n"
            if has_ext:                         # extrude exprs may use the cadgen helpers
                header += ("from cadgen.attach_array import _wp_attach_at\n"
                           "from cadgen.sselectors import PointOnFaceSelector, PointOnEdgeSelector\n"
                           "from cadgen.spherical_coords import SphericalAnglesDirection, Plane\n")
            return header + "\n".join(lines) + "\n"
    AXV = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}
    # Emit in ACCEPTANCE order: the assembler scores ops incrementally
    # (occ |= union / occ &= ~cut, in sequence), so a union after a cut
    # legitimately refills -- re-sorting across a cut changes the composed
    # solid (Vase/fillet_hole divergences).  CUT positions are the only
    # order-sensitive boundaries; unions BETWEEN cuts commute, so each
    # union run is size-sorted (attach parents emit before children).
    order = []
    _run = []

    def _flush():
        _run.sort(key=lambda k: -(ops[k][1].profile.area *
                                  abs(ops[k][1].t1 - ops[k][1].t0)))
        order.extend(_run)
        _run.clear()

    for k in range(len(ops)):
        if ops[k][0] == "union":
            _run.append(k)
        else:
            _flush()
            order.append(k)
    _flush()
    pos = {k: r for r, k in enumerate(order)}
    rel = infer_relations(ops)
    info = {k: _op_axis_info(ops[k][1]) for k in range(len(ops))}

    # v7.3: exactly-coincident union-union faces crack OCC's tessellation
    # (part 04 non-watertight in v6/v7.1/v7.2 alike; fast 0.993 scored
    # 0.676).  Collect every union's world-axis boundary planes -- its two
    # caps AND its axis-aligned polygon-edge walls (exterior + interior
    # rings: part 04's plate cap meets the I-beam's CHANNEL wall, an
    # interior edge) -- then extend any union end that lands on another
    # union's plane by 0.3 INTO it.  A healthy 0.3 overlap fuses cleanly;
    # the 0.017 jitter of v7 only produced slivers.  Cuts over-pierce 0.6
    # > 0.3, so no membranes can appear.
    union_planes = {0: [], 1: [], 2: []}      # world axis -> [(k, coord)]
    for k in range(len(ops)):
        if ops[k][0] != "union":
            continue
        axk, lok, hik, _ = info[k]
        if axk is None:                 # sweep/rotated op: no world axis
            continue
        union_planes[axk].append((k, float(lok)))
        union_planes[axk].append((k, float(hik)))
        Tk = ops[k][1].profile.to_3d
        poly_k = ops[k][1].profile.polygon
        for d in (0, 1, 2):
            if d == axk:
                continue
            for ring in [poly_k.exterior, *poly_k.interiors]:
                co = np.asarray(ring.coords)
                wd = co[:, 0] * Tk[d, 0] + co[:, 1] * Tk[d, 1] + Tk[d, 3]
                for i in range(len(co) - 1):
                    if abs(wd[i] - wd[i + 1]) < 1e-6:
                        union_planes[d].append((k, float(wd[i])))

    lines = ["import cadquery as cq",
             "from cadgen.attach_array import _wp_attach_at",
             "from cadgen.sselectors import PointOnFaceSelector, PointOnEdgeSelector",
             "from cadgen.spherical_coords import SphericalAnglesDirection, Plane",
             ""]
    emitted_any = False
    for r, k in enumerate(order):
        side, c = ops[k]
        axis, lo, hi, _a = info[k]
        h = hi - lo
        fit = _fit_circle_2d(_ring_coords_uniform(c.profile.polygon.exterior))
        is_circ = fit is not None and fit[3] < circle_tol
        attach = None
        if (emitted_any and is_circ and axis is not None and k in rel
                and pos[rel[k][0]] < r):
            attach = rel[k]                  # (parent, kind, anchor)

        if attach is not None:
            pj, kind, anchor = attach
            # world center of the child's circle at its BASE, snapped onto the
            # parent's face plane (the point must lie ON the built solid)
            T = c.profile.to_3d
            t_base = c.t0 if abs((T @ np.array([0,0,c.t0,1.0]))[:3][axis] - anchor) <= \
                     abs((T @ np.array([0,0,c.t1,1.0]))[:3][axis] - anchor) else c.t1
            w = (T @ np.array([fit[0], fit[1], t_base, 1.0]))[:3]
            w[axis] = anchor
            # height measured FROM the parent's face plane to the child's far
            # end (the child base may sit slightly inside the parent)
            h_eff = (hi - anchor) if kind in ("on_top", "on_face+") else (anchor - lo)
            xh = (1.0, 0.0, 0.0) if axis == 2 else (0.0, 0.0, 1.0)
            # local frame of the tangent plane _wp_attach_at will build there
            n3 = np.zeros(3); n3[axis] = 1.0 if kind in ("on_top", "on_face+") else -1.0
            xd = np.array(xh, float); yd = np.cross(n3, xd)
            # the anchor must lie on face MATERIAL -- if the circle center sits
            # in a hole (e.g. a bore), offset it onto the annulus and re-center
            # the sketch with .center(u,v)
            lat = kind in ("on_face+", "on_face-")
            a_pt, ctr = w, ""
            if h_eff < 0.5 or (not _anchor_on_material(w, ops, pj, lateral=lat, ci=k)):
                # cap anchors may be rescued by an offset onto the annulus;
                # lateral anchors have no safe offset -> absolute fallback
                a_pt = _find_material_anchor(w, fit[2], xd, yd, ops, pj, ci=k) \
                    if (h_eff >= 0.5 and not lat) else None
                if a_pt is None:
                    attach = None       # no safe anchor: absolute block below
                else:
                    u = float(np.dot(w - a_pt, xd)); v = float(np.dot(w - a_pt, yd))
                    ctr = f".center({u:.4f},{v:.4f})"
            elif lat:
                # corner-tie guard: keep the anchor 2.5u away from profile
                # corners so the face selector cannot grab the wrong face
                a_pt = _lateral_edge_slide(w, ops, pj, n3, ci=k)
                if a_pt is None:
                    attach = None
                elif not np.allclose(a_pt, w, atol=1e-6):
                    u = float(np.dot(w - a_pt, xd)); v = float(np.dot(w - a_pt, yd))
                    ctr = f".center({u:.4f},{v:.4f})"
        if attach is not None:
            # annular child -> ONE sketch-based attach (annulus in a single
            # sketch, VLM's own grammar).  A separate bore attach at the same
            # anchor would run AFTER the collar union buried the point and
            # PointOnFaceSelector would grab the bore wall (zero-norm xDir).
            holes = []
            for hole in c.profile.polygon.interiors:
                hf = _fit_circle_2d(_ring_coords_uniform(hole))
                if hf is None or hf[3] >= 1.5:
                    holes = None        # non-circular hole: absolute fallback
                    break
                wh = (T @ np.array([hf[0], hf[1], t_base, 1.0]))[:3]
                wh[axis] = anchor
                holes.append((float(np.dot(wh - a_pt, xd)),
                              float(np.dot(wh - a_pt, yd)),
                              hf[2] + 0.02 + 0.011 * (r % 5)))
            if holes is None:
                attach = None
        if attach is not None:
            comb = "'s'" if side == "cut" else "'a'"
            if holes:
                u0 = float(np.dot(w - a_pt, xd)); v0 = float(np.dot(w - a_pt, yd))
                frag = f".sketch().push([({u0:.4f},{v0:.4f})]).circle({fit[2]:.4f})"
                for uh, vh, rh in holes:
                    frag += f".push([({uh:.4f},{vh:.4f})]).circle({rh:.4f},mode='s')"
                chain = frag + f".finalize().extrude({h_eff:.4f})"
            else:
                chain = f"{ctr}.circle({fit[2]:.4f}).extrude({h_eff:.4f})"
            lines.append(
                f"r = _wp_attach_at(r, [({a_pt[0]:.4f},{a_pt[1]:.4f},{a_pt[2]:.4f})], "
                f"\"{chain}\", x_hint={xh}, combine={comb})"
                f"  # on op{pos[pj]}'s {'cap' if kind in ('on_top','under') else 'face'}")
            continue

        # plain absolute-plane block (VLM grammar)
        T = c.profile.to_3d
        # v7.2: over-pierce cuts 0.6 past BOTH ends so they always fully
        # pierce.  v7's union height jitter (+0.017*(r%3)) made formerly
        # exact-through cuts stop a hairline short -> membrane caps the
        # hole -> voxel fill floods the hole interior (Disc 0.841->0.728);
        # it also turned exactly-coincident union caps into 0.02-unit
        # slivers that crack the OCC tessellation (part 04 0.994->0.676,
        # non-watertight).  Over-piercing cuts is free (cutting air) and
        # removes the union/cut coincidence class without touching union
        # geometry at all.  Blind cuts deepen by 0.6 on a ~200-unit part:
        # negligible.
        pierce = 0.6 if side == "cut" else 0.0
        # v7.3: anti-coincidence extension for unions (see union_planes)
        ext_base = ext_cap = 0.0
        if side == "union" and axis is not None and abs(float(T[axis, 2])) > 0.999:
            h0 = abs(c.t1 - c.t0)
            base_d = float(T[axis, 2] * c.t0 + T[axis, 3])
            cap_d = base_d + h0 * float(T[axis, 2])
            # tol 0.3, NOT exact: recon spans land 0.1-0.25 past neighbour
            # walls (part 04: op1 cap -92.857 vs op0 wall -92.754), leaving
            # sliver-deep penetrations that crack tessellation just like
            # exact coincidence.  Pushing 0.3 further makes the overlap
            # healthy (~0.4); extending into material is volume-free.
            for m, coord in union_planes[axis]:
                if m == k:
                    continue
                if abs(coord - base_d) < 0.3:
                    ext_base = 0.3
                if abs(coord - cap_d) < 0.3:
                    ext_cap = 0.3
        o3 = (T @ np.array([0.0, 0.0, c.t0 - pierce - ext_base, 1.0]))[:3]
        xd = T[:3, 0]; n3 = T[:3, 2]
        plane = (f"cq.Plane(origin=cq.Vector({o3[0]:.4f},{o3[1]:.4f},{o3[2]:.4f}),"
                 f" xDir=cq.Vector({xd[0]:.4f},{xd[1]:.4f},{xd[2]:.4f}),"
                 f" normal=cq.Vector({n3[0]:.4f},{n3[1]:.4f},{n3[2]:.4f}))")
        jit = 0.02 + 0.011 * (r % 5)
        frag = _sketch_frag(c.profile.polygon, circle_tol, jit=jit)
        if not frag:
            continue
        h_abs = abs(c.t1 - c.t0) + 2.0 * pierce + ext_base + ext_cap
        expr = f"cq.Workplane({plane}).sketch(){frag}.finalize().extrude({h_abs:.4f})"
        if not emitted_any:
            lines.append(f"r = {expr}")
            emitted_any = True
        elif side == "cut":
            lines.append(f"r = r.cut({expr})")
        else:
            lines.append(f"r = r.union({expr})")
    out = "\n".join(lines) + "\n"
    # OPT-IN array re-rolling (DET_REROLL): collapse repeated same-radius sketch
    # cutouts into one generative push (grid/lattice/polar).  Text-only, exact,
    # idempotent -- never affects the reconstruction mesh / IoU.
    import os as _os2
    if _os2.environ.get("DET_REROLL"):
        try:
            from .reroll import reroll_program
            out = reroll_program(out)[0]
        except Exception:
            pass
    return out
