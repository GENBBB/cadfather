"""Unified metrics layer: IoU, GMS, CD and contract scoring.

One implementation per project. Metrics used to live in two places and drift
apart: one computed IoU without the watertight/volume filter and without
clipping, so numbers were not comparable and the runtime selected candidates by
CD alone.

This is a **verbatim** port of the reference implementation (`cicada`
`utils/metrics.py`), whose numbers must match; the GMS machinery is a snapshot in
`vendor/cicada_metrics/gms.py`. Any deviation would make results incomparable, so
formulas, constants and operation order are untouched, including the `1000`
factor for CD, the `iou > 1.05` threshold and the GMS normalisation
`1 - gms/25`.

IoU and GMS drive optimisation; CD stays observational: it is not normalised and
depends on point sampling, so part of its variance belongs to the measurement
procedure.

Coordinate system. GT and prediction are normalised differently
(`transform_mesh_0_1` vs `/200 + 0.5`), the common cube is centred at
`(0.5, 0.5, 0.5)`, and the prediction regularly leaves the cube. There are
deliberately no checks like "the pair lies in a unit cube centred at zero": they
would reject every pair.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from cad_agent import dsl_runtime
from cad_agent.capabilities.objective import FIELD_IOU_UNAVAILABLE

logger = logging.getLogger(__name__)

# Reference constants. Kept here so that a deviation from the reference shows up
# as a changed value rather than being hidden inside an expression.
CD_N_POINTS = 8192
CD_SCALE = 1000.0
IOU_REJECT_ABOVE = 1.05
# Tolerance of the guard V(G∩P) <= min(V(G), V(P)), as a fraction of the smaller
# volume: above the boolean-operation noise, below genuine boolean failures.
INTERSECTION_EXCESS_TOL = 1e-3
GMS_N_POINTS = 8192
GMS_N_ANGLES = 125
GMS_REL_DIST_TOL = 0.05
GMS_UPPER_BOUND_TOL_RT = 25
# Constructor parameters of the snapshot handler. Shared because the handler is
# also built outside (from the per-part cache), and both places must call the
# snapshot with identical values.
GMS_HOW_MANY_MEDIANS = 5.0
GMS_MAX_POSSIBLE_N = 100000


# --- failures ---------------------------------------------------------------

FAILURE_EXECUTION = "execution_error"
FAILURE_NOT_WATERTIGHT = "not_watertight"
# The part returned no measurement at all: the worker died with it, or the run
# was interrupted before it finished. Per contract this is a zero in the score
# vector, not a skip -- otherwise a hard part drops out of the denominator and
# silently improves both scores. A separate reason from `execution_error`:
# there the geometry failed, here the machine did.
FAILURE_NO_RESULT = "no_result"
# GT and prediction are valid and a boolean engine exists, but IoU could not be
# computed (degenerate volumes, negative intersection, IoU above
# `IOU_REJECT_ABOVE`, engine exception). Most likely an invalid mesh, so it must
# not compete with candidates on GMS alone in the IoU+GMS scale. A separate
# reason from `not_watertight`: the prediction is closed here.
FAILURE_IOU_UNAVAILABLE = "iou_unavailable"

# Strata by GT watertight status. The third one is not a dataset stratum but an
# honest "not looked at": the status belongs to GT alone, and when unknown the
# part cannot be silently assigned to either real stratum.
# Seed for point-cloud sampling in metrics. A constant, not the run seed: the
# measurement is a property of the (mesh, point count) pair and must not be a
# source of run-to-run differences. Before seeding, the score still jittered on
# parts with bit-identical IoU.
METRICS_SAMPLE_SEED = 42

STRATUM_WATERTIGHT = "watertight_gt"
STRATUM_NON_WATERTIGHT = "non_watertight_gt"
STRATUM_UNKNOWN = "unknown_gt"


# --- raw metrics (verbatim port of the reference metrics) -------------------


def _cd_components(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    n_points: int,
) -> tuple[float, float]:
    """Unnormalised ``(precision, recall)``: the common basis of both CD scales."""
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points, seed=METRICS_SAMPLE_SEED)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points, seed=METRICS_SAMPLE_SEED)

    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)

    return float(np.mean(np.square(gt_distance))), float(np.mean(np.square(pred_distance)))


def compute_cd(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    n_points: int = CD_N_POINTS,
) -> tuple[float, float, float]:
    """Symmetric Chamfer distance in the reference scale. ``(cd, precision, recall)``.

    The factor 1000 comes from the reference; without it numbers are not
    comparable. The metric is **observational**: it does not take part in
    candidate selection.
    """
    precision, recall = _cd_components(gt_mesh, pred_mesh, n_points)
    return (precision + recall) * CD_SCALE, precision * CD_SCALE, recall * CD_SCALE


# Alias for `evaluate_pair`, where a parameter named `compute_cd` (matching
# `compute_extended`) shadows the function. Renaming the parameter would make the
# config key and the code name of the same thing diverge.
compute_cd_metric = compute_cd


def compute_cd_runtime(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    n_points: int = CD_N_POINTS,
) -> float:
    """CD in the **runtime scale**: the same number without the factor 1000.

    Careful: this is not a different metric but a different scale of the same
    CD. Runtime thresholds (``SUCCESS_CD_THRESHOLD`` = 1e-4,
    ``planner_stepwise_cd_threshold``) are calibrated to it; x1000 is applied only
    when reporting (``aggregate_best_candidate_metrics``). Swapping one scale for
    the other shifts every threshold a thousandfold and shows up only as changed
    search behaviour, so the two scales are separated by explicit names rather
    than a defaulted parameter.
    """
    precision, recall = _cd_components(gt_mesh, pred_mesh, n_points)
    return precision + recall


def iou_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Mesh parts suitable for volume operations: closed and with volume.

    Factored out of :func:`compute_iou` because for GT this is a property of the
    part, not of the candidate, and is cached per part. trimesh `split()`
    **repairs** components, so it matters that this is exactly the call that was
    used inline, not an "equivalent" one.

    Deviation from the reference: a mesh with negative total volume (flipped
    normals) is inverted before splitting. Otherwise `is_valid_gt` (by absolute
    volume) admits the part to IoU while it has no usable components. The whole
    mesh is inverted, not the negative-volume parts: such a part inside another
    is a cavity wall, not an inside-out body. The condition matches
    `is_valid_gt`: the volume sign of an open mesh means nothing, and such GT
    does not go to IoU.

    A closed mesh with inconsistent winding (some faces flipped) is first passed
    through `fix_winding`: `is_valid_gt` ignores winding and admits such GT to
    IoU, while `is_volume` drops its parts, leaving every candidate without IoU
    (`iou_unavailable`).
    """
    if mesh.is_watertight and not mesh.is_winding_consistent:
        mesh = mesh.copy()
        trimesh.repair.fix_winding(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh = mesh.copy()
        mesh.invert()
    return merge_bodies([m for m in mesh.split() if m.is_watertight and m.is_volume])


def merge_bodies(parts: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Merge mesh parts into one body by boolean union.

    Deviation from the reference. CAD STL files often store a part as separate,
    non-unioned bodies (a blade with its own copy of the hub, overlapping bars).
    A pairwise sum over parts counts shared volume once per covering part, so IoU
    rises to 1.0 and beyond the rejection threshold. If merging fails, the parts
    stay as they were (the previous computation).
    """
    if len(parts) < 2:
        return parts
    try:
        merged = trimesh.boolean.union(parts)
    except Exception:
        logger.debug("Merging %d bodies failed", len(parts), exc_info=True)
        return parts
    if merged is None or not merged.is_volume:
        return parts
    return [merged]


def compute_iou(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    gt_components: list[trimesh.Trimesh] | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Volumetric IoU. Returns ``(iou, iog, iop)`` or a triple of ``None``.

    ``None`` means "metric unavailable on this pair" (no watertight components,
    degenerate volumes, negative intersection, IoU above the plausibility
    threshold). This is **not** a per-figure failure: failure is determined by
    the prediction's watertight status, see :func:`evaluate_pair`.

    Boolean operations use the default trimesh engine; in the server environment
    that is ``manifold``, so the fast path is already active.

    ``gt_components`` is an already split GT (:func:`iou_components`). It depends
    only on GT but used to be paid for on every candidate; a caller that has a
    per-part cache must pass it, otherwise everything is computed as before.
    """
    try:
        gt_meshes = iou_components(gt_mesh) if gt_components is None else gt_components
        # Prediction bodies are merged the same way as GT: overlapping bodies in a
        # candidate inflate IoU through the same mechanism.
        pred_meshes = merge_bodies([m for m in pred_mesh.split() if m.is_watertight and m.is_volume])

        if not gt_meshes or not pred_meshes:
            return None, None, None

        intersection_volume = 0.0
        for gt_i in gt_meshes:
            for pred_i in pred_meshes:
                inter = gt_i.intersection(pred_i)
                if inter is None or inter.is_empty or len(inter.faces) == 0:
                    continue
                # Raw volume, without the reference's `fix_normals`: on a
                # multi-body intersection (merged body with a cavity) it flips
                # the cavity and IoU goes above 1.
                vol = inter.volume
                intersection_volume += vol if np.isfinite(vol) else 0.0

        gt_volume = sum(m.volume for m in gt_meshes)
        pred_volume = sum(m.volume for m in pred_meshes)
        union_volume = gt_volume + pred_volume - intersection_volume

        if union_volume <= 0 or gt_volume <= 0 or pred_volume <= 0:
            logger.debug(
                "IoU unavailable: union=%s gt=%s pred=%s",
                union_volume,
                gt_volume,
                pred_volume,
            )
            return None, None, None

        if intersection_volume < 0:
            return None, None, None

        # An intersection cannot exceed its operand. If it does, the boolean
        # operation disagrees with the geometry, and between 1 and
        # `IOU_REJECT_ABOVE` it would be counted as 1.0.
        if intersection_volume > min(gt_volume, pred_volume) * (1.0 + INTERSECTION_EXCESS_TOL):
            return None, None, None

        iou = intersection_volume / union_volume
        iog = intersection_volume / gt_volume
        iop = intersection_volume / pred_volume

        # Above the plausibility threshold the result is treated as a boolean
        # artefact, not a good match.
        if iou > IOU_REJECT_ABOVE:
            return None, None, None

        return (
            float(np.clip(iou, 0.0, 1.0)),
            float(np.clip(iog, 0.0, 1.0)),
            float(np.clip(iop, 0.0, 1.0)),
        )
    except Exception:
        logger.exception("Failed to compute IoU")
        return None, None, None


def gms_handler(
    mesh: trimesh.Trimesh,
    n_points: int = GMS_N_POINTS,
    cube_trick: bool = True,
    autofix_sampling: bool = False,
):
    """GMS snapshot handler: a cloud of ``n_points`` points, normals and a KD-tree.

    For GT this is a property of the part, yet it used to be rebuilt for every
    candidate, together with `mesh.copy()` and the tree. Seeding
    (``sample_seed``) makes construction deterministic, so a cached handler is
    pointwise equal to a fresh one; ``ball_matching`` only reads
    ``pc``/``normals``/``tree``, so candidates can share it.
    """
    trimesh_handler, _ = dsl_runtime.import_gms()
    return trimesh_handler(
        mesh,
        Np=n_points,
        use_cube_trick=cube_trick,
        how_many_medians=GMS_HOW_MANY_MEDIANS,
        max_possible_N=GMS_MAX_POSSIBLE_N,
        compute_median=autofix_sampling,
        # `sample_seed` is a constructor kwarg of the snapshot and the only way to
        # seed GMS without editing `vendor/`. Inside, the snapshot saves and
        # restores the global numpy RNG state, so neighbours in the process do not
        # see it. Without it `mesh.sample()` was unseeded: the snapshot seeds only
        # on the `pc_cache_enable=True` path, which is off here.
        sample_seed=METRICS_SAMPLE_SEED,
    )


def compute_gms(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    n_points: int = GMS_N_POINTS,
    n_angles: int = GMS_N_ANGLES,
    rel_dist_tol: float = GMS_REL_DIST_TOL,
    cube_trick: bool = True,
    upper_bound_tol_rt: int = GMS_UPPER_BOUND_TOL_RT,
    autofix_sampling: bool = False,
    gt_handler=None,
) -> tuple[float, float]:
    """GMS. Returns ``(gms_raw, gms_norm)`` where ``gms_norm = 1 - gms_raw/25``.

    Raw GMS is an AOC, "lower is better"; the normalised value is "higher is
    better" and goes into the score. It is available for non-watertight GT too,
    so on that stratum the whole signal rests on it.
    """
    aoc = _aoc_gms(
        gt_mesh=gt_mesh,
        pred_mesh=pred_mesh,
        n_points=n_points,
        n_angles=n_angles,
        rel_dist_tol=rel_dist_tol,
        cube_trick=cube_trick,
        upper_bound_tol_rt=upper_bound_tol_rt,
        autofix_sampling=autofix_sampling,
        gt_handler=gt_handler,
    )[0]

    return float(aoc), float(1 - aoc / upper_bound_tol_rt)


def _aoc_gms(
    gt_mesh: trimesh.Trimesh,
    pred_mesh: trimesh.Trimesh,
    n_points: int,
    n_angles: int,
    rel_dist_tol: float,
    cube_trick: bool,
    upper_bound_tol_rt: int,
    autofix_sampling: bool,
    gt_handler=None,
) -> tuple[float, np.ndarray, list[float]]:
    """Verbatim port of the reference ``_aoc_gms``.

    One difference: the GT handler may be passed ready-made, since it depends
    only on GT and is cached per part (see :func:`gms_handler`).
    """
    _, ball_matching = dsl_runtime.import_gms()

    tol_angles = np.linspace(0, upper_bound_tol_rt, n_angles)

    relative_unit = "Lmax"

    if gt_handler is None:
        gt_handler = gms_handler(
            gt_mesh, n_points=n_points, cube_trick=cube_trick, autofix_sampling=autofix_sampling
        )
    pred_handler = gms_handler(
        pred_mesh, n_points=n_points, cube_trick=cube_trick, autofix_sampling=autofix_sampling
    )

    absolute_distance_threshold = rel_dist_tol * gt_handler.relative_unit_length(relative_unit)
    if autofix_sampling:
        gt_handler.autofix_sampling(absolute_distance_threshold)
        pred_handler.autofix_sampling(absolute_distance_threshold)

    def _match_scores(mh_a, mh_b):
        query_k = max(1, len(mh_b.pc) // 100)
        return ball_matching(
            mh_A=mh_a,
            mh_B=mh_b,
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
    inv_precision = np.divide(1.0, precision, out=np.zeros_like(precision), where=precision > 0)
    denom = inv_recall + inv_precision - 1.0
    valid = (recall > 0) & (precision > 0) & np.isfinite(denom) & (denom != 0)
    gms_scores_all = np.divide(1.0, denom, out=np.zeros_like(denom), where=valid)

    stop_index = next(
        (i for i, score in enumerate(gms_scores_all) if i > 0 and score >= 0.99),
        None,
    )
    gms_scores = gms_scores_all if stop_index is None else gms_scores_all[: stop_index + 1]

    number_of_angles = len(gms_scores)
    tol_angles = tol_angles[:number_of_angles]
    max_possible_area = 0.0 if number_of_angles < 2 else tol_angles[-1] - tol_angles[0]
    aoc = max_possible_area - np.trapezoid(gms_scores, tol_angles)

    return aoc, tol_angles, gms_scores.tolist()


# --- watertight status ------------------------------------------------------


def is_watertight(mesh: trimesh.Trimesh) -> bool:
    """Whether the PREDICTION is valid: the whole mesh is closed and a volume.

    An open part now fails the whole prediction (previously one closed component
    with volume sufficed, so an open shell next to a positive-volume fragment
    passed and only the fragment went to IoU). ``is_volume`` is kept: for the
    whole mesh it adds consistent normals, a finite centre of mass and volume > 0.
    """
    try:
        return bool(mesh.is_watertight and mesh.is_volume)
    except Exception:
        logger.exception("Could not determine the watertight status")
        return False


def is_valid_gt(mesh: trimesh.Trimesh) -> bool:
    """Whether GT goes to IoU: the whole mesh is closed and |volume| > 0.

    Same rule as the run-table validity check, verbatim except for loading: a
    multi-mesh scene has already been concatenated by the caller. The earlier
    criterion ("at least one closed body") admitted GT with open bodies, so the
    prediction of a whole part was compared with a fragment of it. Invalid GT is
    judged by GMS alone.
    """
    try:
        return bool(mesh.is_watertight and abs(float(mesh.volume)) > 0)
    except Exception:
        logger.exception("Failed to determine GT validity")
        return False


@lru_cache(maxsize=1)
def iou_engine_available() -> bool:
    """Whether a boolean engine exists, without which IoU cannot be computed.

    The `iou_unavailable` failure only makes sense where IoU CAN be computed: in
    an environment without `manifold3d` an empty IoU is a property of the machine,
    not of the prediction.
    """
    return importlib.util.find_spec("manifold3d") is not None


# --- per-figure result ------------------------------------------------------


@dataclass
class FigureMetrics:
    """Measurement result for one figure.

    ``failure`` separates a **per-figure failure** (score is floored) from the
    mere absence of a metric: IoU is unavailable for non-watertight GT, which is
    normal for that stratum, not a breakage.
    """

    figure_id: str
    # None means "GT status was not determined", which is not the same as
    # `False`. The field used to be two-valued, and any part that failed at
    # execution returned before the line computing the status, so it landed in
    # `non_watertight_gt` by the field default, not by its geometry.
    gt_watertight: bool | None = None
    pred_watertight: bool = False
    failure: str | None = None

    iou: float | None = None
    iog: float | None = None
    iop: float | None = None
    gms_raw: float | None = None
    gms_norm: float | None = None
    cd: float | None = None
    cd_precision: float | None = None
    cd_recall: float | None = None

    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def stratum(self) -> str:
        """Figure stratum: the GT watertight status sets the composition of the score signal."""
        if self.gt_watertight is None:
            return STRATUM_UNKNOWN
        return STRATUM_WATERTIGHT if self.gt_watertight else STRATUM_NON_WATERTIGHT

    @property
    def is_failure(self) -> bool:
        return self.failure is not None

    def score(self) -> float:
        """``score_i`` per contract: 0.0 on failure, otherwise the mean of the normalised metrics.

        Only IoU and GMS enter the score. CD does not: it is not normalised and
        depends on the sampling procedure.
        """
        if self.is_failure:
            return 0.0

        available = [v for v in (self.iou, self.gms_norm) if v is not None]
        if not available:
            # No metric at all: this cannot count as a good result.
            return 0.0
        return float(np.clip(float(np.mean(available)), 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stratum"] = self.stratum
        data["score"] = self.score()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FigureMetrics":
        """Rebuild from a `per_figure.json` record.

        `stratum` and `score` in the dict are derived: they are recomputed, not
        read, so an aggregate cannot diverge from the fields it is computed from.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def evaluate_pair(
    figure_id: str,
    gt_mesh: trimesh.Trimesh | None,
    pred_mesh: trimesh.Trimesh | None,
    *,
    execution_error: str | None = None,
    compute_extended: bool = False,
    compute_cd: bool = False,
) -> FigureMetrics:
    """Compute the metrics of a GT/prediction pair and determine the failure.

    The two failure levels differ as follows: ``execution_error`` means the code
    did not execute; a non-watertight prediction means the construction ended
    unfinished. Both floor ``score_i``, but the IR decomposition must tell them
    apart.

    ``compute_extended`` enables the full observational set (``iog``/``iop``, CD
    with its decomposition), ``compute_cd`` only CD (`experiment.metrics.cd`).
    Neither enters ``score_i``, which uses IoU and GMS only, so mass runs can
    skip paying for the observational ones.
    """
    result = FigureMetrics(figure_id=figure_id)

    # The GT status depends on GT alone and is always known when the mesh is
    # given, including on failure. Computed **before** the failure branches: the
    # stratum is a property of the dataset, not of the rollout outcome, and must
    # not be lost to failing code.
    if gt_mesh is not None:
        result.gt_watertight = is_valid_gt(gt_mesh)

    if execution_error is not None:
        result.failure = FAILURE_EXECUTION
        result.error = execution_error
        return result

    if gt_mesh is None or pred_mesh is None:
        result.failure = FAILURE_EXECUTION
        result.error = "no GT or prediction geometry"
        return result

    result.pred_watertight = is_watertight(pred_mesh)

    # A non-watertight prediction is a per-figure failure, not a bad result, and
    # has no metrics.
    if not result.pred_watertight:
        result.failure = FAILURE_NOT_WATERTIGHT
        return result

    try:
        gms_raw, gms_norm = compute_gms(gt_mesh, pred_mesh)
        result.gms_raw, result.gms_norm = gms_raw, gms_norm
    except Exception as exc:
        logger.exception("Failed to compute GMS for %s", figure_id)
        result.error = f"gms: {exc}"

    # IoU is defined only for watertight GT: a stratum attribute, not a breakage.
    if result.gt_watertight:
        try:
            iou, iog, iop = compute_iou(gt_mesh, pred_mesh)
            result.iou = iou
            if compute_extended:
                result.iog, result.iop = iog, iop
        except Exception as exc:
            # The boolean engine fails on degenerate geometry and is absent
            # without `manifold3d`. That is no reason to lose the whole figure:
            # a missing IoU is visible in the field, the error is in `error`.
            logger.exception("Failed to compute IoU for %s", figure_id)
            result.error = f"{result.error or ''} iou: {exc}".strip()
        if result.failure is None and result.iou is None and iou_engine_available():
            result.failure = FAILURE_IOU_UNAVAILABLE

    # CD is a requested metric, like IoU and GMS in the runtime. It is not in
    # `score_i` (IoU and GMS only), so disabling it changes no contract number,
    # only observability. Enabled by `experiment.metrics.cd`; `extended_metrics`
    # pulls it in as part of the full analysis set.
    if compute_cd or compute_extended:
        try:
            cd, precision, recall = compute_cd_metric(gt_mesh, pred_mesh)
            result.cd = cd
            # The precision/recall decomposition is only in the extended set.
            if compute_extended:
                result.cd_precision, result.cd_recall = precision, recall
        except Exception as exc:
            logger.exception("Failed to compute CD for %s", figure_id)
            result.error = f"{result.error or ''} cd: {exc}".strip()

    return result


# --- aggregates over a subset -----------------------------------------------


@dataclass
class RunAggregate:
    """Aggregates over a set of figures.

    ``score_with_zeros`` is primary: only it is protected from gaming through
    failures. ``score_without_invalid`` improves by itself if construction is made
    to fail on hard figures, since they drop out of the denominator.
    """

    n_figures: int
    n_failures: int
    ir: float
    ir_execution: float
    ir_not_watertight: float
    # The IR components must sum to IR. The third reason appeared with parts lost
    # together with a worker: without its own field it would enter `ir` but not
    # the decomposition, and the sum would silently stop adding up.
    ir_no_result: float
    ir_iou_unavailable: float
    score_with_zeros: float
    score_without_invalid: float
    identity_holds: bool
    identity_gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(metrics: Sequence[FigureMetrics], *, tol: float = 1e-9) -> RunAggregate:
    """Combine per-figure results and check the contract identity.

    ``score_with_zeros = (1 - IR) * score_without_invalid``: "how often it worked"
    times "how good it is when it worked". The identity is checked here: a gap
    means an error in failure accounting, not numerical noise.
    """
    n = len(metrics)
    if n == 0:
        return RunAggregate(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, 0.0)

    failures = [m for m in metrics if m.is_failure]
    valid = [m for m in metrics if not m.is_failure]

    ir = len(failures) / n
    ir_execution = sum(1 for m in failures if m.failure == FAILURE_EXECUTION) / n
    ir_not_watertight = sum(1 for m in failures if m.failure == FAILURE_NOT_WATERTIGHT) / n
    ir_no_result = sum(1 for m in failures if m.failure == FAILURE_NO_RESULT) / n
    ir_iou_unavailable = sum(1 for m in failures if m.failure == FAILURE_IOU_UNAVAILABLE) / n

    score_with_zeros = float(np.mean([m.score() for m in metrics]))
    score_without_invalid = float(np.mean([m.score() for m in valid])) if valid else 0.0

    expected = (1.0 - ir) * score_without_invalid
    gap = abs(score_with_zeros - expected)

    if gap > tol:
        logger.error(
            "Contract identity violated: score_with_zeros=%.12f, (1-IR)*score_without=%.12f, "
            "gap=%.3g: failure accounting is wrong",
            score_with_zeros,
            expected,
            gap,
        )

    # The decomposition must sum to IR. A new failure reason added without its
    # own field would otherwise vanish from the decomposition while staying in IR,
    # and the report would quietly contradict itself.
    decomposed = ir_execution + ir_not_watertight + ir_no_result + ir_iou_unavailable
    if abs(ir - decomposed) > tol:
        logger.error(
            "IR decomposition does not add up: IR=%.12f, sum of reasons=%.12f - "
            "there is a failure with a reason that has no share of its own",
            ir,
            decomposed,
        )

    return RunAggregate(
        n_figures=n,
        n_failures=len(failures),
        ir=ir,
        ir_execution=ir_execution,
        ir_not_watertight=ir_not_watertight,
        ir_no_result=ir_no_result,
        ir_iou_unavailable=ir_iou_unavailable,
        score_with_zeros=score_with_zeros,
        score_without_invalid=score_without_invalid,
        identity_holds=gap <= tol,
        identity_gap=gap,
    )


def aggregate_by_stratum(metrics: Sequence[FigureMetrics]) -> dict[str, RunAggregate]:
    """The same aggregates computed separately per stratum.

    The strata levels may differ systematically (on the second one GMS is the
    whole signal), so they must be looked at separately.
    """
    strata: dict[str, list[FigureMetrics]] = {}
    for m in metrics:
        strata.setdefault(m.stratum, []).append(m)
    return {name: aggregate(items) for name, items in sorted(strata.items())}


# --- runtime measurement inside the executor process ------------------------
#
# A layer separate from `evaluate_pair`: that computes the contract metrics of a
# figure from ready meshes, while this is called directly in the child process
# right after construction, while the mesh is still in memory. Otherwise the
# prediction is written to shared storage by one process and immediately re-read
# by another, for every candidate.
#
# Normalisation repeats the former CD helper (box/box, sym, 8192 points): the
# runtime thresholds are calibrated to this scale.

RUNTIME_CD_N_POINTS = 8192


def load_gt_cached(
    gt_mesh_path: str,
    cache: dict[str, Any],
    n_points: int = RUNTIME_CD_N_POINTS,
) -> dict[str, Any]:
    """Load GT, normalise and sample it, once per figure.

    The cache outlives a task only where it is inherited by fork (the
    ``proxy_pool`` backend): the shim warms the GT of its part and every
    grandchild gets the ready points for free.

    A useful side effect: all candidates of one figure are compared with **the
    same** GT point set. GT used to be resampled per candidate, and sampling
    spread leaked straight into the comparison of siblings.
    """
    cached = cache.get(gt_mesh_path)
    if cached is not None:
        # The counter lives in the cache dict itself: it is the only thing that
        # survives a fork together with its contents, while a separate object
        # would have to be carried across the process boundary by hand.
        _count(cache, "hits")
        return cached

    _count(cache, "misses")
    gt_mesh = _norm_box_half(_as_trimesh(trimesh.load_mesh(gt_mesh_path)))
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points, seed=METRICS_SAMPLE_SEED)
    entry = {"mesh": gt_mesh, "points": np.asarray(gt_points)}
    cache[gt_mesh_path] = entry
    return entry


# Key of the counters in the cache dict. Contains a colon because all other keys
# are GT file paths and cannot collide with it.
GT_CACHE_STATS_KEY = "::stats"


def _count(cache: dict[str, Any], field: str) -> None:
    stats = cache.setdefault(GT_CACHE_STATS_KEY, {"hits": 0, "misses": 0})
    stats[field] = stats.get(field, 0) + 1


def gt_cache_stats(cache: dict[str, Any]) -> dict[str, int]:
    """Hits and misses of this process's GT cache."""
    return dict(cache.get(GT_CACHE_STATS_KEY) or {"hits": 0, "misses": 0})


def measure_pair(
    gt_mesh_path: str,
    pred_mesh_path: str,
    gt_cache: dict[str, Any] | None = None,
    n_points: int = RUNTIME_CD_N_POINTS,
    extended: bool = False,
    needs: Sequence[str] = (),
) -> dict[str, Any]:
    """Runtime metrics of a pair: everything on request, except the prediction's watertight status.

    ``needs`` is the minimal set without which the harness cannot decide
    (`objective.Objective.needs`): ``"cd"``, ``"iou"`` and/or ``"gms"``.
    ``extended`` adds the observational metrics for analysis: the same plus
    ``iog``/``iop``. They are separate because IoU and GMS cost an order of
    magnitude more than CD and are paid for on **every** candidate, so the
    `objective: iou` mode must not pay for GMS, nor `objective: cd` for anything
    beyond what it always cost.

    Coordinate system. CD is computed in the runtime scale (both sides in a unit
    box), while the watertight status, IoU and GMS are computed **in the contract
    scale**: GT is normalised by its own extents, the prediction is divided by
    200. This is a comparability condition, not an implementation detail:
    otherwise the runtime would select by one number while the report showed
    another. The GT frame is the same in both scales: ``_norm_box_half`` and
    ``normalize_mesh(gt_flag=True)`` are the same three operations.

    The prediction's watertight status is taken in the same frame as IoU and the
    final recomputation (`figure_run._score_final`). In the runtime frame a
    degenerate zero-thickness body passed ``is_volume`` on rounding noise but
    not in the contract frame: the search led with and returned an open mesh the
    contract rejected, and its IoU silently dropped to ``None``.
    """
    cache = gt_cache if gt_cache is not None else {}
    gt = load_gt_cached(gt_mesh_path, cache, n_points=n_points)

    pred_raw = _as_trimesh(trimesh.load_mesh(pred_mesh_path))
    pred_mesh = _norm_box_half(pred_raw)
    pred_contract = pred_raw.copy()
    normalize_mesh(pred_contract, gt_flag=False)

    # `pred_watertight` is always computed: the failure decision rests on it, not
    # observability. Everything else is on request.
    out: dict[str, Any] = {"pred_watertight": bool(is_watertight(pred_contract))}
    # An open prediction is a failure, not a candidate with a number: it has no
    # metrics.
    if not out["pred_watertight"]:
        return out

    want_cd = extended or "cd" in needs
    want_iou = extended or "iou" in needs
    want_gms = extended or "gms" in needs

    if want_cd:
        # Prediction sampling and the two trees are needed for CD only; with CD
        # off they are not paid for. The GT cloud comes from the part cache.
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points, seed=METRICS_SAMPLE_SEED)
        gt_points = gt["points"]
        pred_to_gt, _ = cKDTree(gt_points).query(pred_points, k=1)
        gt_to_pred, _ = cKDTree(pred_points).query(gt_points, k=1)
        cd_runtime = float(np.mean(np.square(gt_to_pred)) + np.mean(np.square(pred_to_gt)))
        out["cd_runtime"] = cd_runtime if np.isfinite(cd_runtime) else None

    if not (want_iou or want_gms):
        return out

    gt_mesh = gt["mesh"]
    out["gt_watertight"] = _gt_watertight_cached(gt, gt_mesh)

    if want_iou and out["gt_watertight"]:
        iou, iog, iop = compute_iou(gt_mesh, pred_contract, gt_components=_gt_components_cached(gt))
        out["iou"] = iou
        if extended:
            out["iog"], out["iop"] = iog, iop
        # The same failure as in the contract (`evaluate_pair`), computed here in
        # the measurement process: only it knows whether a boolean engine exists.
        # The harness reads the flag in `search._classify` and ranks such a
        # candidate below any valid one.
        if iou is None and out["pred_watertight"] and iou_engine_available():
            out[FIELD_IOU_UNAVAILABLE] = True

    if want_gms:
        try:
            gms_raw, gms_norm = compute_gms(gt_mesh, pred_contract, gt_handler=_gms_gt_handler_cached(gt))
            out["gms_raw"], out["gms_norm"] = gms_raw, gms_norm
        except Exception as exc:  # GMS requires pykdtree; it is absent locally
            out["gms_error"] = str(exc)

    return out


def warm_gt_iou(gt_entry: dict[str, Any]) -> None:
    """Precompute in a warmed GT everything IoU needs: validity and parts.

    Called when warming the shim (`execute._warm_gt`), the only place where the
    cache outlives a candidate: the execution grandchild gets it by fork. Lazily,
    the parts were recomputed in every grandchild and lost with it. While this was
    a single `split()` the cost was tolerable, but body merging (`merge_bodies`)
    on multi-body GT can take tens of seconds, more than the execution timeout.
    """
    if _gt_watertight_cached(gt_entry, gt_entry["mesh"]):
        _gt_components_cached(gt_entry)


def _gt_components_cached(gt_entry: dict[str, Any]) -> list[trimesh.Trimesh]:
    """GT components for IoU, computed once per part.

    Where the cache outlives a candidate (`proxy_pool` shims), the warm-up computes
    them (:func:`warm_gt_iou`); this is the fallback for other backends.
    """
    parts = gt_entry.get("iou_components")
    if parts is None:
        parts = gt_entry["iou_components"] = iou_components(gt_entry["mesh"])
    return parts


def _gms_gt_handler_cached(gt_entry: dict[str, Any]):
    """GMS handler for GT: also once per part, for the same reason."""
    handler = gt_entry.get("gms_handler")
    if handler is None:
        handler = gt_entry["gms_handler"] = gms_handler(gt_entry["mesh"])
    return handler


def _gt_watertight_cached(gt_entry: dict[str, Any], gt_mesh: trimesh.Trimesh) -> bool:
    """GT validity for IoU (:func:`is_valid_gt`), once per figure.

    The status depends only on GT, and the mesh volume of a part with thousands of
    faces is noticeable if paid on every execution. The field is still called
    `gt_watertight`: strata and reports read it.
    """
    if "watertight" not in gt_entry:
        gt_entry["watertight"] = bool(is_valid_gt(gt_mesh))
    return bool(gt_entry["watertight"])


def _as_trimesh(mesh_obj: Any) -> trimesh.Trimesh:
    if isinstance(mesh_obj, trimesh.Scene):
        return trimesh.util.concatenate(list(mesh_obj.geometry.values()))
    return mesh_obj


def _norm_box_half(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Unit-size cube centred at (0.5, 0.5, 0.5): the runtime scale."""
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = float(np.max(mesh.extents))
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, gt_flag: bool = False) -> None:
    """Bring a mesh to the common coordinate system. Moved from `utils` verbatim.

    GT and prediction are normalised **differently**: GT by its own extents, the
    prediction is divided by 200 (the scale the model writes it in). The centre of
    the common cube is (0.5, 0.5, 0.5), and the prediction regularly leaves its
    bounds; there are deliberately no "pair lies in a unit cube" checks.
    """
    if gt_flag:
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-center)
        extent = np.max(mesh.extents)
        if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    else:
        mesh.apply_scale(1.0 / 200)
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))


def normalize_for_metrics(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh) -> None:
    """Normalise a pair before the contract metrics (in place, both at once)."""
    normalize_mesh(gt_mesh, gt_flag=True)
    normalize_mesh(pred_mesh, gt_flag=False)
