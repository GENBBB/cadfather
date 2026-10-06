import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Literal, Union
import os


# single-thread behavior for pykdtree, NumPy/BLAS
# can be overridden by environment variables, inherited in multiprocessing mode
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import trimesh
from pykdtree.kdtree import KDTree

LOGGER = logging.getLogger(__name__)

DECIMALS = 12
MAX_SIZE_FOR_TRUE_MEDIAN = 500000


_ORIGINAL_OS_SYSCONF = os.sysconf


def _safe_sysconf(name):
    try:
        return _ORIGINAL_OS_SYSCONF(name)
    except PermissionError:
        # Some sandboxed environments deny this specific lookup, which breaks
        # ProcessPoolExecutor initialization before tests can run.
        if name == "SC_SEM_NSEMS_MAX":
            return -1
        raise


try:
    _ORIGINAL_OS_SYSCONF("SC_SEM_NSEMS_MAX")
except PermissionError:
    os.sysconf = _safe_sysconf

AllowedRelativeUnits = Literal["Lmax", "Lmid", "Laverage", "diag_bbox"]
MeshInput = Union[trimesh.Trimesh, str, os.PathLike]


def round_down(value, decimals=DECIMALS):
    factor = 1 / (10**decimals)
    return (value // factor) * factor


def rescale_and_center(
    mesh: trimesh.Trimesh,
    method: Literal["diag_bbox", "Lmax", None] = "Lmax",
    center=True,
    downscale_factor=None,
):
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        raise ValueError(
            f"Mesh has invalid extents for normalization: extents={mesh.extents!r}"
        )
    # Clamp degenerate (zero or negative) dimensions so we can normalize without division by zero.
    # Scale factor is determined by max(extents), so clamping does not change it.
    max_ext = float(np.max(extents))
    if max_ext <= 0:
        raise ValueError(
            f"Mesh has invalid extents for normalization: extents={mesh.extents!r}"
        )
    eps = max(max_ext * 1e-10, np.finfo(extents.dtype).tiny)
    extents = np.maximum(extents, eps)

    if center:
        mesh.vertices = mesh.vertices - mesh.centroid

    if downscale_factor is None:
        if method == "diag_bbox":
            downscale_factor = np.sqrt(sum(extents**2))
        elif method == "Lmax":
            downscale_factor = np.max(extents)
        else:
            raise NotImplementedError(f"{method}. method must be 'diag_bbox' or 'Lmax'")

    if not np.isfinite(downscale_factor) or downscale_factor <= 0:
        raise ValueError(
            "Mesh has zero or invalid extent; cannot rescale for normalization."
        )

    mesh.vertices = mesh.vertices / downscale_factor
    return mesh


def make_same_scale(
    zero_mesh: trimesh.Trimesh,
    current_mesh: trimesh.Trimesh,
    coscale_method: Literal["diag_bbox", "Lmax", "cadrecode"] = "Lmax",
    center=True,
):
    if coscale_method == "cadrecode":
        current_mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 100 / 2))
        current_mesh.apply_transform(
            trimesh.transformations.translation_matrix([0.5, 0.5, 0.5])
        )
        zero_mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 2))
        zero_mesh.apply_transform(
            trimesh.transformations.translation_matrix([0.5, 0.5, 0.5])
        )
    elif coscale_method in {"Lmax", "diag_bbox"}:
        zero_mesh = rescale_and_center(zero_mesh, method=coscale_method, center=center)
        current_mesh = rescale_and_center(
            current_mesh, method=coscale_method, center=center
        )

    return zero_mesh, current_mesh


class TrimeshHandler:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        Np=10000,
        use_cube_trick=False,
        how_many_medians=5.0,
        max_possible_N=1000000,
        compute_median=True,
        sample_seed=None,
    ):
        self.mesh = mesh
        self._N_initial = Np
        self.use_cube_trick = use_cube_trick
        self.how_many_medians = how_many_medians
        self.max_possible_N = max_possible_N
        self._compute_median_flag = compute_median
        self._sample_seed = sample_seed
        self._median = None
        self.sample_pc_and_normals()

    @property
    def N(self):
        return len(self.pc)

    @property
    def median(self):
        if self._median is None:
            self._compute_median()
        return self._median

    def sample_pc_and_normals(self, N=None):
        if N is None:
            N = self._N_initial
        self._median = None
        if self._sample_seed is not None:
            state = np.random.get_state()
            np.random.seed((int(self._sample_seed) + int(N)) % (2**32))

        try:
            if not self.use_cube_trick:
                self.pc, face_index = self.mesh.sample(N, return_index=True)
                self.normals = self.mesh.face_normals[face_index]
                self.tree = KDTree(self.pc, leafsize=400)
                if self._compute_median_flag:
                    self._compute_median()
            else:
                Ls = self.mesh.extents
                if not np.all(np.isfinite(Ls)) or Ls.shape[0] != 3 or np.any(Ls <= 0):
                    # Fallback: cube trick is unsafe for degenerate extents, use plain sampling.
                    self.pc, face_index = self.mesh.sample(N, return_index=True)
                    self.normals = self.mesh.face_normals[face_index]
                    self.tree = KDTree(self.pc, leafsize=400)
                    if self._compute_median_flag:
                        self._compute_median()
                else:
                    mesh_copy = self.mesh.copy()
                    mesh_copy.apply_scale([1 / Ls[0], 1 / Ls[1], 1 / Ls[2]])
                    self.pc, face_index = mesh_copy.sample(N, return_index=True)
                    self.pc = self.pc * Ls
                    self.normals = self.mesh.face_normals[face_index]
                    self.tree = KDTree(self.pc, leafsize=400)
                    if self._compute_median_flag:
                        self._compute_median()
        finally:
            if self._sample_seed is not None:
                np.random.set_state(state)

    def _compute_median(self):
        if len(self.pc) < MAX_SIZE_FOR_TRUE_MEDIAN:
            distances_to_myself = self.tree.query(self.pc, k=2)[0][:, 1]
            self._median = np.median(distances_to_myself)
        else:
            COEF = 0.55
            S_total = self.mesh.area
            N_triangles = len(self.pc) - 3
            S = S_total / N_triangles
            self._median = np.sqrt(4 * S / np.sqrt(3)) * COEF

    def query(self, points):
        distances, indices = self.tree.query(points, k=1)
        return distances, indices, self.normals[indices]

    def compute_new_N(self, distance_threshold, wanted_coef=3.5):
        current_coef = distance_threshold / self.median
        new_N = int(np.ceil(self.N * (wanted_coef / current_coef) ** 2))
        return new_N

    def relative_unit_length(self, relative_unit: AllowedRelativeUnits = "Lmax"):
        if relative_unit == "Lmax":
            return max(self.mesh.extents)
        elif relative_unit == "Lmid":
            return sorted(self.mesh.extents)[1]
        elif relative_unit == "Laverage":
            return np.mean(self.mesh.extents)
        elif relative_unit == "diag_bbox":
            return np.linalg.norm(self.mesh.bounds[1] - self.mesh.bounds[0])
        raise ValueError(f"Relative unit {relative_unit} not allowed")

    def distance_threshold(self, tol, relative_unit: AllowedRelativeUnits = "Lmax"):
        return self.relative_unit_length(relative_unit) * tol

    def autofix_sampling(self, distance_threshold, max_iterations=100):
        i = 0
        do_not_update = False
        coef = self.how_many_medians
        while not self.sampling_is_ok(distance_threshold) and i < max_iterations:
            i += 1
            coef += 0.1 * self.how_many_medians
            new_N = self.compute_new_N(distance_threshold, coef)
            if new_N > self.max_possible_N:
                do_not_update = True
            if not do_not_update:
                self.sample_pc_and_normals(new_N)
                LOGGER.info(
                    "New N: %s for %s. Iteration %s. The ratio is %.2f",
                    new_N,
                    self,
                    i,
                    distance_threshold / self.median,
                )
            else:
                break

    def sampling_is_ok(self, distance_threshold):
        return distance_threshold >= self.how_many_medians * self.median

    def __str__(self):
        return (
            f"TrimeshHandler(N={self.N}, median={self.median}, "
            f"use_cube_trick={self.use_cube_trick} \\n mesh: {self.mesh})"
        )


def _pykdtree_query_ball_tree(
    tree_A,
    n_points_A,
    points_B,
    distance_threshold,
    eps,
    initial_k=256,
    max_k=4096,
):
    if not isinstance(points_B, np.ndarray):
        points_B = np.asarray(points_B, dtype=np.float64, order="C")
    elif points_B.dtype != np.float64 or not points_B.flags["C_CONTIGUOUS"]:
        points_B = np.asarray(points_B, dtype=np.float64, order="C")

    n_points_B = len(points_B)
    if n_points_B == 0 or n_points_A == 0:
        offsets = np.zeros(n_points_B + 1, dtype=np.int64)
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.zeros(n_points_B, dtype=np.int64),
            offsets,
        )

    k_cap = min(max(1, max_k), n_points_A)
    k_current = min(max(1, initial_k), k_cap)

    if initial_k == max_k:
        distances, indices = tree_A.query(
            points_B,
            k=k_current,
            distance_upper_bound=distance_threshold,
            eps=eps,
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        mask = (indices < n_points_A) & (distances < np.inf)
        counts = mask.sum(axis=1, dtype=np.int64)
        offsets = np.empty(n_points_B + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        cand_A_idx = indices[mask]
        cand_B_idx = np.repeat(np.arange(n_points_B, dtype=np.int32), counts)
        return cand_A_idx, cand_B_idx, counts, offsets

    indices_2d = np.full((n_points_B, k_cap), n_points_A, dtype=np.int32)
    distances_2d = np.full((n_points_B, k_cap), np.inf, dtype=np.float64)
    k_used = np.zeros(n_points_B, dtype=np.int32)

    def _fill_rows(row_indices, k):
        if row_indices.size == 0:
            return
        distances, indices = tree_A.query(
            points_B[row_indices],
            k=k,
            distance_upper_bound=distance_threshold,
            eps=eps,
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        indices = indices.astype(np.int32, copy=False)
        indices_2d[row_indices, :k] = indices
        distances_2d[row_indices, :k] = distances
        k_used[row_indices] = k

    all_rows = np.arange(n_points_B)
    _fill_rows(all_rows, k_current)

    while True:
        last_idx = k_used - 1
        finite_last = np.isfinite(distances_2d[np.arange(n_points_B), last_idx])
        needs_more = finite_last & (k_used < k_cap)
        if not needs_more.any():
            break
        next_k = int(min(k_cap, k_used[needs_more].max() * 2))
        if next_k == k_used[needs_more].max():
            break
        rows_to_refine = np.nonzero(needs_more)[0]
        _fill_rows(rows_to_refine, next_k)

    mask = (indices_2d < n_points_A) & np.isfinite(distances_2d)
    counts = np.sum(mask, axis=1, dtype=np.int64)
    offsets = np.empty(n_points_B + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    cand_A_idx = indices_2d[mask].astype(np.int32, copy=False)
    cand_B_idx = np.repeat(np.arange(n_points_B, dtype=np.int32), counts)
    return cand_A_idx, cand_B_idx, counts, offsets

def A_in_B_ball_matching_multiangle_v3(
    mh_A: TrimeshHandler,
    mh_B: TrimeshHandler,
    distance_threshold=0.05,
    angle_tolerances=None,
    use_abs_in_cos=False,
    query_k=256,
    max_query_k=4096,
    ball_query_eps_relative=1e-1,
):
    n_points_B = len(mh_B.pc)
    angle_tolerances = np.asarray(angle_tolerances, dtype=np.float64)
    cos_tolerances = round_down(np.cos(np.deg2rad(angle_tolerances)))
    n_angles = len(angle_tolerances)

    cand_A_idx, cand_B_idx, counts_B, offsets_B = _pykdtree_query_ball_tree(
        tree_A=mh_A.tree,
        n_points_A=len(mh_A.pc),
        points_B=mh_B.pc,
        distance_threshold=distance_threshold,
        eps=ball_query_eps_relative * distance_threshold,
        initial_k=query_k,
        max_k=max_query_k,
    )

    total_candidates = offsets_B[-1]
    if total_candidates == 0 or n_points_B == 0:
        return np.zeros((n_angles, n_points_B), dtype=np.float64)

    cosines_all = np.empty(total_candidates, dtype=mh_A.normals.dtype)
    np.einsum(
        "ij,ij->i",
        mh_A.normals[cand_A_idx],
        mh_B.normals[cand_B_idx],
        out=cosines_all,
    )
    np.clip(cosines_all, -1.0, 1.0, out=cosines_all)
    if use_abs_in_cos:
        np.abs(cosines_all, out=cosines_all)

    max_cos_per_B = np.empty(n_points_B, dtype=np.float64)
    max_cos_per_B.fill(-np.inf)
    nonempty = counts_B > 0
    if nonempty.any():
        starts = offsets_B[:-1][nonempty]
        seg_max = np.maximum.reduceat(cosines_all, starts)
        max_cos_per_B[nonempty] = seg_max

    match_scores = np.where(
        max_cos_per_B[None, :] >= cos_tolerances[:, None],
        1.0,
        0.0,
    )
    return match_scores


class LRUCache:
    def __init__(self, maxsize=512):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value


_handler_cache = LRUCache(maxsize=512)
_handler_cache_lock = threading.Lock()
_handler_cache_pid = os.getpid()


def _reset_handler_cache_after_fork():
    global _handler_cache, _handler_cache_lock, _handler_cache_pid
    _handler_cache = LRUCache(maxsize=512)
    _handler_cache_lock = threading.Lock()
    _handler_cache_pid = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_handler_cache_after_fork)


def _get_handler_cache_key(mesh, Np, use_cube_trick, how_many_medians, max_possible_N):
    hasher = hashlib.blake2b()
    hasher.update(mesh.vertices.tobytes())
    hasher.update(mesh.faces.tobytes())
    hasher.update(str(Np).encode())
    hasher.update(str(use_cube_trick).encode())
    hasher.update(str(how_many_medians).encode())
    hasher.update(str(max_possible_N).encode())
    return hasher.hexdigest()


def _get_cached_handler(
    mesh,
    Np,
    use_cube_trick,
    how_many_medians,
    max_possible_N=100000,
    pc_cache_enable=False,
    compute_median=True,
):
    global _handler_cache_pid
    current_pid = os.getpid()
    if current_pid != _handler_cache_pid:
        _reset_handler_cache_after_fork()

    cached_handler = None
    cache_key = None
    sample_seed = None
    if pc_cache_enable:
        cache_key = _get_handler_cache_key(
            mesh, Np, use_cube_trick, how_many_medians, max_possible_N
        )
        sample_seed = int(cache_key[:16], 16)
        with _handler_cache_lock:
            cached_handler = _handler_cache.get(cache_key)

    if cached_handler is not None:
        return cached_handler

    handler = TrimeshHandler(
        mesh,
        Np,
        use_cube_trick,
        how_many_medians,
        max_possible_N,
        compute_median=compute_median,
        sample_seed=sample_seed,
    )

    if pc_cache_enable:
        with _handler_cache_lock:
            _handler_cache.set(cache_key, handler)
        return handler

    return handler


def _coerce_mesh(mesh_or_path: MeshInput) -> trimesh.Trimesh:
    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path.copy()
    mesh = trimesh.load(mesh_or_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        meshes = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                meshes.append(geom)
        if not meshes:
            raise ValueError("Loaded scene has no valid mesh geometry")
        if len(meshes) == 1:
            return meshes[0].copy()
        return trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not coerce input to trimesh.Trimesh")
    return mesh.copy()


def aoc_gms(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    n_points=8192,
    n_angles=125,
    rel_dist_tol=0.05,
    cube_trick=True,
    pc_cache_enable=False,
    upper_bound_tol_rt=25,
    autofix_sampling=False,
    add_auc=False,
    logger=None,
):
    gms = _aoc_gms(
        gt_mesh=gt_mesh,
        pred_mesh=pred_mesh,
        n_points=n_points,
        n_angles=n_angles,
        rel_dist_tol=rel_dist_tol,
        cube_trick=cube_trick,
        pc_cache_enable=pc_cache_enable,
        upper_bound_tol_rt=upper_bound_tol_rt,
        autofix_sampling=autofix_sampling,
        add_auc=add_auc,
        logger=logger,
    )

    return gms[0], 1 - gms[0]/25


def _aoc_gms(
    gt_mesh: MeshInput,
    pred_mesh: MeshInput,
    n_points=8192,
    n_angles=125,
    rel_dist_tol=0.05,
    cube_trick=True,
    pc_cache_enable=False,
    upper_bound_tol_rt=25,
    autofix_sampling=False,
    add_auc=False,
    logger=None,
):
    if logger is None:
        logger = LOGGER
    gt_mesh = _coerce_mesh(gt_mesh)
    pred_mesh = _coerce_mesh(pred_mesh)

    tol_angles = np.linspace(0, upper_bound_tol_rt, n_angles)

    max_possible_n = 100000
    relative_unit = "Lmax"
    how_many_medians = 5.0

    gt_handler = _get_cached_handler(
        gt_mesh,
        Np=n_points,
        use_cube_trick=cube_trick,
        how_many_medians=how_many_medians,
        max_possible_N=max_possible_n,
        pc_cache_enable=pc_cache_enable,
        compute_median=autofix_sampling,
    )
    pred_handler = _get_cached_handler(
        pred_mesh,
        Np=n_points,
        use_cube_trick=cube_trick,
        how_many_medians=how_many_medians,
        max_possible_N=max_possible_n,
        pc_cache_enable=pc_cache_enable,
        compute_median=autofix_sampling,
    )

    absolute_distance_threshold = rel_dist_tol * gt_handler.relative_unit_length(
        relative_unit
    )
    if autofix_sampling:
        gt_handler.autofix_sampling(absolute_distance_threshold)
        pred_handler.autofix_sampling(absolute_distance_threshold)

    def _match_scores(mh_A, mh_B):
        query_k = max(1, len(mh_B.pc) // 100)
        return A_in_B_ball_matching_multiangle_v3(
            mh_A=mh_A,
            mh_B=mh_B,
            distance_threshold=absolute_distance_threshold,
            angle_tolerances=tol_angles,
            use_abs_in_cos=False,
            query_k=query_k,
            max_query_k=query_k,
        )

    gt_in_pred_scores = _match_scores(gt_handler, pred_handler)
    pred_in_gt_scores = _match_scores(pred_handler, gt_handler)

    recall = gt_in_pred_scores.sum(axis=1) / gt_in_pred_scores.shape[1]
    precision = pred_in_gt_scores.sum(axis=1) / pred_in_gt_scores.shape[1]

    inv_recall = np.divide(1.0, recall, out=np.zeros_like(recall), where=recall > 0)
    inv_precision = np.divide(
        1.0, precision, out=np.zeros_like(precision), where=precision > 0
    )
    denom = inv_recall + inv_precision - 1.0
    valid = (recall > 0) & (precision > 0) & np.isfinite(denom) & (denom != 0)
    gms_scores_all = np.divide(1.0, denom, out=np.zeros_like(denom), where=valid)

    stop_index = next(
        (i for i, score in enumerate(gms_scores_all) if i > 0 and score >= 0.99),
        None,
    )
    if stop_index is None:
        gms_scores = gms_scores_all
    else:
        gms_scores = gms_scores_all[: stop_index + 1]

    number_of_angles = len(gms_scores)
    tol_angles = tol_angles[:number_of_angles]
    if number_of_angles < 2:
        max_possible_area = 0.0
    else:
        max_possible_area = tol_angles[-1] - tol_angles[0]
    raw_auc = np.trapezoid(gms_scores, tol_angles)
    aoc = max_possible_area - raw_auc

    if add_auc:
        auc = raw_auc / max_possible_area if max_possible_area > 0 else 0.0
        logger.info(f"{aoc=} {raw_auc=} {max_possible_area=} {auc=}")
        return aoc, tol_angles, gms_scores.tolist(), auc

    logger.info(f"{aoc=} {raw_auc=} {max_possible_area=}")
    return aoc, tol_angles, gms_scores.tolist()