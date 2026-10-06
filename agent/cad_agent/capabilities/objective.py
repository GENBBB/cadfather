"""Selection objective: what the runtime uses to compare candidates with each other.

Why a separate layer. A runtime that decides by CD alone (whom to keep in the beam,
what counts as an improvement, when to stop) is cheap but **blind to the contract
metric**: two parts can have the same CD to the fifth digit while their contract
scores differ widely, the whole difference being in GMS. Worse, a part can stop on
the rule "success threshold reached by CD" while sitting at a poor contract score.

Hence the requirement: runtime scoring and contract scoring must be one scale, and
which one is a config choice, not a code constant. Four modes:

``cd``
    Symmetric Chamfer in the runtime scale (lower is better). Cheap but **not
    free**: it is computed only on request, like the others. Computing it always
    only made sense while it was the selection objective; after the switch to IoU it
    was paid for nothing.
``iou``
    Volumetric IoU (higher is better). Defined **only for a watertight GT**; on the
    second stratum it is unavailable, so the objective falls back to GMS there.
``gms``
    Normalized GMS ``1 - aoc/25`` (higher is better). Available on both strata; on
    a non-watertight GT it is the only contract signal.
``hmean``
    Harmonic mean of the available IoU and GMS (higher is better). Closest to the
    contract's ``score_i`` but stricter: the contract takes the arithmetic mean,
    while the harmonic one drops to zero if either metric fails, so one metric
    cannot carry the score.

Cost. CD is two trees on 8192 points. IoU is boolean operations on every pair of
components, GMS is two ball matchings over 125 angles. Both are an order of
magnitude more expensive and are paid **on every candidate**, not once per part.
So an objective declares :attr:`Objective.needs`, the minimal set that
:func:`capabilities.metrics.measure_pair` must compute, and ``iou`` mode does not
pay for GMS.

Fallback. A metric can be unavailable not because of a bad candidate but because of
a property of the shape: IoU is undefined for a non-watertight GT. The harness finds
this out before the first execution (`res.difficulty()`) and switches the objective
to :attr:`Objective.fallback` once per part, recording it in the log. Scales must
not be mixed within one part: "best of the rollout" would stop being a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# Fields `measure_pair` puts into the result. Same names as in `FigureMetrics`,
# so the runtime and the contract do not diverge in naming.
FIELD_CD = "cd_runtime"
FIELD_IOU = "iou"
FIELD_GMS = "gms_norm"

# What to ask `measure_pair` for. No metric is computed "by itself": CD used to be
# computed always as a cheap observable, but after the switch to IoU it became pure
# overhead (two trees on 8192 points for EVERY candidate, for a number nobody decides
# by). Now it is requested like the others: the `cd` objective pulls it through
# `needs`, analysis through `extended`, the report through `experiment.metrics.cd`.
NEED_IOU = "iou"
NEED_GMS = "gms"
NEED_CD = "cd"


@dataclass(frozen=True)
class Objective:
    """The scale in which the runtime compares candidates of one part.

    Immutable: the objective is chosen by the config and lives for the whole rollout
    (except for one fallback, see :func:`resolve`). All comparisons go through this
    object's methods; the harness has no direct metric comparisons left, otherwise
    changing the objective would mean edits in a dozen places.
    """

    name: str
    needs: tuple[str, ...]
    higher_is_better: bool
    default_success: float
    default_stepwise: float
    fallback: str | None

    # --- value -------------------------------------------------------------

    def value(self, metrics: dict[str, Any] | None) -> float | None:
        """Objective value from a candidate's measurement. ``None`` means not applicable here.

        ``None`` means exactly one thing: this candidate cannot be compared on this
        scale (not computed, not defined). Such a candidate does not take part in
        selection but stays in the log with what is known about it.
        """
        if not metrics:
            return None
        if self.name == "cd":
            return _finite(metrics.get(FIELD_CD))
        if self.name == "iou":
            return _finite(metrics.get(FIELD_IOU))
        if self.name == "gms":
            return _finite(metrics.get(FIELD_GMS))
        if self.name == "hmean":
            return _harmonic(_finite(metrics.get(FIELD_IOU)), _finite(metrics.get(FIELD_GMS)))
        raise ValueError(f"Unknown objective {self.name!r}")

    # --- comparisons -------------------------------------------------------

    def sort_key(self, value: float) -> float:
        """Sort key in which **lower is always better**.

        One key for all modes: otherwise every `sorted`/`min` in the harness would grow
        its own branch for the metric direction.
        """
        return value if not self.higher_is_better else -value

    def better(self, new: float | None, old: float | None, eps: float = 0.0) -> bool:
        """Whether ``new`` is strictly better than ``old`` by a margin ``eps``. ``old is None`` means yes."""
        if new is None:
            return False
        if old is None:
            return True
        return self.sort_key(new) < self.sort_key(old) - eps

    def best(self, values: Iterable[Any], key: Any = None) -> Any:
        """Best element by the objective. ``key`` extracts the value from an element."""
        pick = (lambda item: item) if key is None else key
        return min(values, key=lambda item: self.sort_key(pick(item)))

    def best_value(self, values: Iterable[float | None]) -> float | None:
        """Best of the values; ``None`` if there are none.

        Separate from :meth:`best` because the caller usually needs to compare against
        "was there anything at all", and `min()` over an empty list is not an error
        there but the normal course of events (the first iteration of a part).
        """
        rows = [value for value in values if value is not None]
        return min(rows, key=self.sort_key) if rows else None

    def success_threshold(self, configured: float | None) -> float:
        """The "already good enough" threshold in the scale of **this** objective.

        Resolved at check time, not in the harness constructor. The reason: a part's
        objective can differ from the config's (`for_detail` changes it by the GT's
        watertight status, `resolve` by unmeasurability). Binding the threshold once
        from the configured objective would not follow a fallback: after `iou -> cd`
        the CD threshold would be 0.98 instead of 1e-4, declaring any built candidate
        a success.

        A value set explicitly by the config stays as is, but it is written in the
        scale of the configured objective, so a fallback does not carry it over and
        must be noticed in the log.
        """
        return self.default_success if configured is None else float(configured)

    def stepwise_threshold(self, configured: float | None) -> float:
        """Per-step threshold ("the step is good enough") in the scale of this objective."""
        return self.default_stepwise if configured is None else float(configured)

    def reached(self, value: float | None, threshold: float | None) -> bool:
        """Whether the "already good enough" threshold is reached."""
        if value is None or threshold is None:
            return False
        return self.sort_key(value) <= self.sort_key(threshold)

    def ambiguity(self, best: float, second: float) -> float:
        """How much the objective did **not** distinguish the two best: 1.0 means not at all.

        Always "smaller over larger", a value in (0, 1]. For CD the smaller is the
        best (``best/second``), for quality metrics it is the second (``second/best``);
        in both cases one means "the candidates are indistinguishable", and the same
        `confidence_ratio` threshold works in all modes without caveats.
        """
        top, low = max(best, second), min(best, second)
        return low / top if top > 1e-12 else 1.0

    # --- description -------------------------------------------------------

    @property
    def direction(self) -> str:
        return "higher is better" if self.higher_is_better else "lower is better"

    def describe(self, value: float | None) -> str:
        return "none" if value is None else f"{value:.8g}"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _harmonic(iou: float | None, gms: float | None) -> float | None:
    """Harmonic mean of the available metrics.

    If only one is available, it is the result (the contract's `score_i` does the
    same with its arithmetic mean). If either is zero, the harmonic mean is zero: a
    failure on one axis is not compensated by the other, a deliberate difference from
    the contract's mean.
    """
    available = [v for v in (iou, gms) if v is not None]
    if not available:
        return None
    if len(available) == 1:
        return available[0]
    total = sum(available)
    if total <= 0:
        return 0.0
    return float(len(available) * (available[0] * available[1]) / total)


# Default thresholds. For quality metrics they are **not calibrated by measurement**
# and are set conservatively: too low a success threshold stops the rollout early,
# which is exactly the error that made the CD rule unfit.
OBJECTIVES: dict[str, Objective] = {
    "cd": Objective(
        name="cd",
        needs=(NEED_CD,),
        higher_is_better=False,
        default_success=1e-4,
        default_stepwise=0.03,
        fallback=None,
    ),
    "iou": Objective(
        name="iou",
        needs=(NEED_IOU,),
        higher_is_better=True,
        default_success=0.98,
        default_stepwise=0.80,
        fallback="gms",
    ),
    "gms": Objective(
        name="gms",
        needs=(NEED_GMS,),
        higher_is_better=True,
        default_success=0.98,
        default_stepwise=0.80,
        fallback="cd",
    ),
    "hmean": Objective(
        name="hmean",
        needs=(NEED_IOU, NEED_GMS),
        higher_is_better=True,
        default_success=0.98,
        default_stepwise=0.80,
        fallback="gms",
    ),
}

OBJECTIVE_NAMES = tuple(OBJECTIVES)


def get_objective(name: str | Objective) -> Objective:
    """Objective by its config name. An unknown name is an error, not a fallback to CD."""
    if isinstance(name, Objective):
        return name
    try:
        return OBJECTIVES[str(name)]
    except KeyError:
        raise ValueError(
            f"Unknown objective {name!r}; expected one of {list(OBJECTIVE_NAMES)}"
        ) from None


def for_detail(objective: Objective, difficulty: Any) -> tuple[Objective, str]:
    """The objective applicable to this part, and the reason for a fallback (empty if unchanged).

    IoU is undefined for a non-watertight GT; this is a property of the stratum, not
    a broken candidate. It is determined from the GT alone, **before** the first
    execution: otherwise the first step would be spent on measurements that cannot
    yield a single value.

    ``difficulty`` is the `res.difficulty` capability; it is called only where the
    objective actually depends on IoU, so `cd` and `gms` modes do not pay for an extra
    mesh load.
    """
    if NEED_IOU not in objective.needs:
        return objective, ""
    features = difficulty() or {}
    status = features.get("gt_watertight")
    if status is None:
        # The features were not computed (unreadable mesh, trimesh failure). Defaulting
        # with `.get("gt_watertight", True)` would read a failed measurement as "GT is
        # closed": one field for two events. The objective is left unchanged (a
        # fallback on unknown would be a guess), but the fact is recorded: otherwise
        # the part silently goes down a branch whose grounds were never checked.
        return objective, (
            f"GT watertight status is unknown ({features.get('error', 'reason not recorded')}): "
            f"objective {objective.name} left as is"
        )
    if bool(status):
        return objective, ""
    fallen = get_objective(objective.fallback or "cd")
    return fallen, f"GT is not watertight, IoU is undefined: objective {objective.name} -> {fallen.name}"


def chain_needs(objective: Objective) -> tuple[str, ...]:
    """Metrics of the whole fallback chain, not just the current objective.

    Requested exactly where the guard :func:`resolve` lives, i.e. while nothing has
    been selected on the part yet. Without this the guard is blind: it asks the next
    objective "do you have values", but nobody computed its metrics, so "no" only
    means "not requested". That made the chain run straight through to `cd`, and
    fixing it with a one-step fallback killed the beam at the very step where it
    changed objective.

    The cost is bounded: the guard closes as soon as the first selected candidate
    appears, which is usually the first step of a part and only it.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    current: Objective | None = objective
    while current is not None:
        for need in current.needs:
            if need not in seen:
                seen.add(need)
                ordered.append(need)
        current = get_objective(current.fallback) if current.fallback else None
    return tuple(ordered)


def resolve(objective: Objective, measured: Iterable[dict[str, Any] | None]) -> Objective:
    """Guard for the case where the objective is not computed at all.

    Needed where the GT is not the cause: GMS requires `pykdtree`, IoU a boolean
    engine, and in an environment without them the objective silently yields no
    values. The whole rollout would then look like "nothing was built". The fallback
    happens once per part and is recorded in the log.
    """
    if any(objective.value(metrics) is not None for metrics in measured):
        return objective
    if objective.fallback is None:
        return objective
    # Walk to the first objective that HAS values. This is correct precisely because
    # the caller requested the metrics of the whole chain (`chain_needs`): "no values"
    # then means "the objective is not computed", not "it was not requested". Without
    # this condition the guard ran the chain through and always landed on `cd` by
    # exhaustion.
    return resolve(get_objective(objective.fallback), measured)


# Key under which the policy keeps the chosen scale in the part memory.
SCALE_KEY = "scale"


def scale_for(
    declared: Objective,
    memory: dict[str, Any],
    measured: Iterable[dict[str, Any] | None],
) -> Objective:
    """This part's scale for the policy: the first in the chain that yields at least one value.

    One rule instead of two separate harness fallbacks. "GT is not watertight" and
    "the metric is not computed in this environment" look the same from the policy's
    side: no measured candidate has a value, so there is no reason to tell them apart
    by mechanism. The first fallback no longer has to consult the GT in advance: the
    harness fixes the metric set (IoU where possible, and GMS), so missing values can
    no longer be confused with "nobody requested them".

    The choice is **frozen** in the part memory as soon as a computable scale is
    found: comparing candidates measured with different rulers is meaningless. While
    none is computable, nothing is frozen and the declared one is returned: values may
    appear later (a prediction can be non-watertight for the first candidates and
    watertight for later ones).

    ``memory`` is `state.memory`, the per-part dict created and discarded by the
    harness. The policy needs no field of its own.

    The function lives in capabilities, not in the seam: the selection scale belongs
    to the policy as a whole, and the harness knows nothing about it. The harness has
    its own scale (`search._fitness`), which decides something else: which candidate
    the part returns outward.
    """
    name = memory.get(SCALE_KEY)
    if name:
        return get_objective(name)
    rows = [metrics for metrics in measured if metrics]
    if not rows:
        return declared
    seen: set[str] = set()
    current: Objective | None = declared
    while current is not None and current.name not in seen:
        seen.add(current.name)
        if any(current.value(metrics) is not None for metrics in rows):
            memory[SCALE_KEY] = current.name
            return current
        current = get_objective(current.fallback) if current.fallback else None
    return declared


# --- why a step has no usable candidates ----------------------------------
#
# There are two reasons, and they are about different things:
#
# - **not built**: the code crashed or the executor died; this is the contract's
#   `ir_execution`, and the model or the machine is to blame;
# - **built, but the objective is not computed**: the geometry exists but the value
#   does not. Usually the prediction is non-watertight, so volumetric IoU is
#   undefined for it.
#
# Writing both to the log as one line "no variant was built" while collecting the
# error list only for the first would leave the list empty for the second, and the
# report would contradict itself (the part stopped on "keeps failing to build" while
# execution reported everything built). The two must be told apart, and it starts
# here, in what is written to the step log.

# Measurement field showing whether the prediction is usable for volumetric metrics.
FIELD_PRED_WATERTIGHT = "pred_watertight"
# Measurement flag: GT and prediction are valid, a boolean engine exists, yet IoU was
# not computed. Set by `metrics.measure_pair` (only the measuring process knows about
# the engine), read by `search._classify`. It lives here rather than in `metrics`:
# the harness must not import trimesh just for a field name.
FIELD_IOU_UNAVAILABLE = "iou_unavailable"

# What each objective need reads from the measurement.
NEED_FIELDS: dict[str, str] = {
    NEED_CD: FIELD_CD,
    NEED_IOU: FIELD_IOU,
    NEED_GMS: FIELD_GMS,
}


@dataclass(frozen=True)
class StepDiagnosis:
    """Diagnosis of a step on which no comparable candidate is left.

    The stop reason comes in two forms because harnesses tolerate failure
    differently: some count consecutive failures and stop once `max_failed_steps` have
    accumulated, others stop at the first such step. One text for both would say
    "consecutive ... maximum allowed steps" where nothing was consecutive.
    """

    n_broken: int
    n_unscored: int
    event: str
    # For harnesses that count consecutive failures.
    stop_reason: str
    # For harnesses that stop at the first unusable step.
    stop_reason_single: str
    errors: list[str]
    unscored: list[str]


def why_unscored(objective: Objective, metrics: dict[str, Any] | None) -> str:
    """Why a candidate that **was built** has no objective value."""
    if not metrics:
        return "the measurement did not return"
    if metrics.get(FIELD_PRED_WATERTIGHT) is False:
        return "prediction not watertight"
    missing = [
        NEED_FIELDS[need]
        for need in objective.needs
        if need in NEED_FIELDS and _finite(metrics.get(NEED_FIELDS[need])) is None
    ]
    if missing:
        return "not computed: " + ", ".join(missing)
    return "the objective gave no value"


def diagnose(
    objective: Objective,
    outcomes: Iterable[tuple[bool, str | None, dict[str, Any] | None]],
    limit: int = 3,
) -> StepDiagnosis:
    """What to record about a step with no usable candidates.

    ``outcomes`` is, per candidate, a triple "was it built, error text, measurement".
    The harness assembles it from whatever it has at hand (`EvalResult` or its own
    branch), so there is no dependency here on the capability layer's types.
    """
    broken: list[str] = []
    unscored: list[str] = []
    for built, error, metrics in outcomes:
        if built:
            unscored.append(why_unscored(objective, metrics))
        else:
            broken.append(error or "no traceback recorded")

    n_broken, n_unscored = len(broken), len(unscored)
    if n_broken == 0 and n_unscored == 0:
        # There was nothing to execute: the generator gave no step and the deterministic
        # branch no operation. This says nothing about geometry, and such a step does
        # not belong in the IR.
        event = "nothing to expand with: no candidates"
        stop_reason = stop_reason_single = "nothing to expand with"
    elif n_unscored == 0:
        event = "no variant built"
        stop_reason = "nothing built for the maximum allowed number of consecutive steps"
        stop_reason_single = "no candidate built"
    elif n_broken == 0:
        event = f"built, but objective {objective.name} is not computed"
        stop_reason = f"objective {objective.name} not computed for the maximum allowed number of consecutive steps"
        stop_reason_single = f"objective {objective.name} is not computed on any candidate"
    else:
        event = (
            f"no usable variants: {n_broken} did not build, "
            f"objective {objective.name} not computed on {n_unscored}"
        )
        stop_reason = "no usable variants for the maximum allowed number of consecutive steps"
        stop_reason_single = "no usable candidate"

    counts: dict[str, int] = {}
    for reason in unscored:
        counts[reason] = counts.get(reason, 0) + 1

    return StepDiagnosis(
        n_broken=n_broken,
        n_unscored=n_unscored,
        event=event,
        stop_reason=stop_reason,
        stop_reason_single=stop_reason_single,
        errors=broken[:limit],
        unscored=[
            reason if count == 1 else f"{reason} ×{count}"
            for reason, count in list(counts.items())[:limit]
        ],
    )
