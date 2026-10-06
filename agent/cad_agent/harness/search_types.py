"""Search-loop types: the vocabulary shared by the harness and the policy.

This module holds data only: no type here
computes anything or calls anyone. This is so that a policy cannot reach the
machinery through an object it was handed to read.

Three things to keep in mind:

- **The unit of the loop is a search iteration, not a DSL step.** The step became a
  property of the candidate (`Candidate.depth`), and the whole pool, all depths at
  once, is live. Hence rollback and re-sampling need no mechanisms of their own:
  both are just a choice of parent from the pool.
- **There is no selection scale in the seam, but the scale of the ANSWER is visible.**
  A candidate carries the full measurement (`metrics`), and what to compare by is
  the policy's business: a metric, an image, or a model. With its own scale
  (`harness_fitness`, the mean of IoU and normalized GMS) the harness decides only
  one thing: which candidate the part returns.

  The number is handed to the policy by QUERY (`SearchState.fitness_of`), not as a
  candidate field: storing it in `Candidate` is still forbidden, because next to
  `metrics` it would read as one more measurement, while it is a harness fold.
  The selection scale remains the policy's responsibility: `scored` still takes
  `value` from it.
- **The failure cause is typed.** "Raise the temperature" and "change the tool"
  cure different failures; if the policy sees only "failed", a retry policy cannot
  be learned. A merged field has already given us a plausible report with the
  wrong diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

# Why a candidate did not come about. The breakdown is the same as in the report:
# the contract's IR is computed from these causes, and a second outcome vocabulary
# would give two inconsistent versions of one number.
Failure = Literal[
    "exec_error",             # the code failed at execution
    "not_watertight",         # it built, but the prediction is not watertight
    "iou_unavailable",        # GT and prediction are valid, yet IoU was not computed
    "empty_generation",       # the tool produced no code at all
    # Two names split off from `empty_generation`. A shared name merged outcomes
    # that are cured differently, and a scaffold reading only `failure` repeated
    # the same choice.
    "unusable_answer",        # there is an answer, but it is not code (reasoning, a fragment)
    "optimizer_failed",       # the optimizer ran and returned no usable code
    "objective_unavailable",  # it built, but the objective is not computed in this scale
    "tool_timeout",           # the tool did not finish within its ceiling
    "tool_budget",            # the tool's per-part ceiling is exhausted
]

# Cost kinds. `genome_cpu` is not a call but seconds: policy blocks may write their
# own scoring functions over code, mesh and images, and under the model
# `cost_i = Σ W_type · n_type` they are free yet eat wall time. It is measured by
# a wrapper around the block call (`harness/search.py`).
COST_KINDS = ("vlm", "agent_text", "agent_visual", "agent_repair", "det", "opt", "exec", "genome_cpu")

# Sampling defaults of the generator: what applies when the policy did not name a
# knob. They live in the seam, not in `capabilities/llm.py`, for the same reason as
# `PromptTooLarge` below: that module pulls in `openai`, while the tool registry and
# the policy registry are read on every config check, where there are no models.
#
# `temperature = 0` is greedy decoding. It is a PARTICULAR VALUE of the knob, not a
# separate mode: a mode switch in the config would silently override `n`,
# temperature and `top_p` of any policy.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 1000


# The measurement field the harness falls back to when the contract scale gives
# values for NO candidate of the part. This happens not because of geometry but
# because of the environment: IoU needs a boolean engine, GMS needs `pykdtree`.
# Without the fallback such a part would return "no valid candidate" with a full
# pool of built ones, i.e. the report would explain the environment by model quality.
FALLBACK_FIELD = "cd_runtime"


# A number in candidate code: `-102.0000`, `15.2141`, `3`, but not digits inside a
# name and not the fractional part of a number already taken. This expression
# defines the legality threshold of `optimize` (`ToolSpec.max_code_numbers`), and
# both the harness and the policy must count with the same expression: a different
# one would give a different count at the same threshold. It lives in the seam
# because the policy does not see `tools.py`.
_CODE_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+\.\d+|(?<![\w.])-?\d+(?![\w.])")


@lru_cache(maxsize=4096)
def code_numbers(code: str) -> int:
    """Number of numeric literals in the code: what `optimize` fits.

    Cached by text: legality is computed on every iteration over the whole pool,
    and det code can be tens of kilobytes.
    """
    return len(_CODE_NUMBER_RE.findall(code or ""))


def harness_fitness(metrics: dict[str, Any] | None) -> float | None:
    """Harness scale: the same fold as the contract's `score_i`.

    The mean of the available IoU and normalized GMS, exactly as
    `metrics.FigureMetrics.score()` computes it.

    **It lives in the seam, not in the loop.** The reason is a second reader: the
    policy asks for this number through `SearchState.fitness_of` to see what the part
    is returned by. Were the definition left in `search.py`, the seam query would
    either import the loop (the seam must not know the machinery) or compute the fold
    a second time. Two versions of one number have already diverged silently.

    Note when reading comparisons: WHICH metrics are computed is decided by the
    harness itself and always the same way (`search._needs`): IoU where possible,
    and GMS. This choice does not belong to the policy, otherwise one variant's
    fitness would be a mean of two metrics and another's would equal IoU, and a gate
    would compare them as homogeneous numbers. The policy may read the number, not set it.

    `None` means NO half was computed: GT is not watertight and `pykdtree` did not
    come up, or there is no boolean engine. This is about the part and the
    environment, not about something the policy did not order.
    """
    if not metrics:
        return None
    values = [
        float(metrics[key])
        for key in ("iou", "gms_norm")
        if metrics.get(key) is not None
    ]
    if not values:
        return None
    return max(0.0, min(1.0, sum(values) / len(values)))


# Marker for the place of an image in the question text. `capabilities/llm.py`
# parses it; it lives in the seam so that the policy does not import `llm`. Without
# markers, images go before the text. With markers, each image takes the place of
# its marker, in order, so the unchanging part of the question can precede the
# turn's images and hit the server's prefix cache. The number of markers must equal
# the number of images; otherwise the markers are stripped and images go before the
# text, so a caller that miscounted gets the old request rather than an image shifted
# under someone else's caption.
IMAGE_SLOT = "<<image>>"


class PromptTooLarge(RuntimeError):
    """The prompt does not fit the model context.

    A separate type rather than a `RuntimeError` with text, because the policy must
    tell "the model did not answer" (fall back to selection by metric) from "we asked
    for too much" (the same question can be asked shorter). The first is cured by a
    retry, the second only by shortening the prompt.

    The policy catches it on purpose, unlike `BudgetExceeded`, which the policy must
    not touch: the budget ceiling is imposed by the harness, while the prompt length
    is a property of the question the policy composed itself.

    **It lives here, not where it is raised** (`capabilities/llm.py`). That module
    pulls in `openai`, while the policy registry is imported on every config check,
    including where there are no models at all (`preflight`, log analysis).
    """


@dataclass(frozen=True)
class Origin:
    """Where a candidate came from: the tool, its parameters and the draw."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    # Attempt number from THIS parent with THIS tool. Kept by the harness. Without
    # it, choosing the same parent again would return the same seed, the same point
    # and almost surely the same answer, i.e. re-sampling would silently become
    # buying a copy for a full call.
    attempt: int = 0

    def key(self) -> tuple:
        """Dedup and cache key: tool, parameters, draw."""
        return (self.tool, tuple(sorted(self.params.items())), self.seed, self.attempt)


@dataclass
class Attempt:
    """One attempt to extend a candidate: what was called, when, and how it ended."""

    origin: Origin
    produced: int = 0            # how many candidates the tool returned
    built: int = 0               # how many of them built
    failure: Failure | None = None
    # Pool candidates the call returned AGAIN (the same code was already on the part,
    # `candidate_dedupe`). Without them "the call ran and gave nothing new" is
    # indistinguishable from "the call returned nothing", and the dialogue would
    # answer the assistant with a bare "no new candidates".
    reused: list[str] = field(default_factory=list)
    # At which iteration this happened. Without a number, "failure share over the
    # last N iterations" is indistinguishable from the share over the whole part, and
    # the policy would have to keep its own call history, a second version of a number
    # the harness already recorded here.
    iteration: int = 0


@dataclass
class Candidate:
    """One reconstruction: code and/or mesh, measurement, origin."""

    id: str
    parent_id: str | None
    depth: int
    origin: Origin
    code: str | None = None
    mesh_path: str | None = None
    # The full measurement as the executor returned it. The objective is computed
    # from it rather than stored as a number: the scale belongs to the policy and may fall back.
    metrics: dict[str, Any] | None = None
    built: bool = False
    failure: Failure | None = None
    # Whether a continuation can be built on this candidate. The property comes from
    # the tool registry (`ToolSpec.extendable`), not from the presence of code: a tool
    # that returns a mesh without code cannot be continued, and the registry, not the
    # loop, must know that.
    extendable: bool = True
    error: str | None = None

    @property
    def alive(self) -> bool:
        """Fit to be a parent and the best: built, measured and VALID.

        The third condition matters. A non-watertight prediction built and has a
        measurement, but it is an invalid model, not a candidate with a bad number:
        building on it is not allowed (the same rule is held by `ToolSpec.accepts`),
        and returning it as the answer while a valid one exists is not either. It stays
        available to repair, which looks for its target by `failure`, not `alive`.
        """
        return bool(self.built and self.metrics is not None and self.failure is None)


@dataclass(frozen=True)
class ToolCost:
    """Price list of one tool: a constant part plus a scalable one.

    Visible to the policy on purpose: `n` costs differently for different tools, and
    there is no linear "n -> price" scale. K variants of `stepwise` is ONE request to
    the model; `n = 5` of det is five operations from the top of an ALREADY paid-for
    pool. A policy that does not see this turns `n` blindly.

    Why two scales rather than one dict: a form that knew only `calls` was read by the
    loop as a price per call, while the budget moved `n` linearly along all axes of
    `scales`. That is not merely inexact but inverted: a generator sample costs a
    fraction of one execution of a small candidate against 1.0 for the request itself,
    so `n` is nearly free for the model and paid for in executions. Yet `n > 1` is the
    most productive knob. A single scale penalized exactly what should be encouraged.

    The weight unit is one execution of a small candidate, not a second: a second
    depends on pod throttling, which differs severalfold between runs.

    `calls` is what is charged per call independent of `n`; `per_n` is the extra for
    each unit of `n`. `wall_sec` and `wall_sec_per_n` are the same in seconds, for a
    policy that counts the part's wall time rather than calls.

    `wall_sec_hi` is the highest observed wall time of ONE call when a large code
    goes in (a det candidate is several times bigger than a stepwise one). It exists
    because the price of `repair` and `optimize` differs two- to three-fold between
    lines, which is not scatter but dependence on input size: one number cannot
    express such a price. `wall_at` uses the TYPICAL value (`wall_sec`): planning by
    the upper one would always understate `n`; the upper one is for whoever estimates
    the worst case.

    `measured` says whether the weights were measured in a run, and it is a property
    of the TOOL, not of the whole registry. It must not be misrepresented: the policy
    reads the price list identically in both cases, and honesty of the number is a
    matter for the report and `task_description.txt`.
    """

    calls: dict[str, float] = field(default_factory=dict)
    per_n: dict[str, float] = field(default_factory=dict)
    wall_sec: float = 0.0
    wall_sec_per_n: float = 0.0
    wall_sec_hi: float = 0.0
    measured: bool = False

    def at(self, n: int = 1) -> dict[str, float]:
        """What a call with the given `n` costs, by call kind.

        Computed here rather than at each reader: two scales are easy to add up
        wrongly, and diverging copies of this arithmetic would mean the policy and
        the report name different things as the price.
        """
        total = dict(self.calls)
        for kind, weight in self.per_n.items():
            total[kind] = total.get(kind, 0.0) + weight * n
        return total

    def wall_at(self, n: int = 1) -> float:
        """Estimated wall time of a call with the given `n`, seconds."""
        return self.wall_sec + self.wall_sec_per_n * n


@dataclass(frozen=True)
class ParamSpec:
    """One tool knob: what it means, its type and its bounds.

    `params_schema` used to be a dict of hint strings, and action parameters went to
    the tool as they were. That was enough for a human reading the registry, but not
    for a mutant. A policy is written by a model, and a wrong key name or a wrong
    value scale (`temperature: 12`) is not a typo there but a normal outcome; it
    should cost a rejected action with a reason in the log, not a strange run.

    Validation lives here next to the description, not in each tool: the tool
    describes the knob as data, and the harness validates it on that data.
    """

    doc: str
    type: str = "float"                      # float | int | str | bool
    lo: float | None = None
    hi: float | None = None
    # The lower bound is STRICT: the value must be greater than `lo`, not "at least".
    # Introduced for `top_p`, where the endpoint requires exactly `(0, 1]`: a
    # non-strict bound let `top_p = 0.0` through and the request was rejected by the
    # server, i.e. schema validation said "legal" about something that cannot be run.
    # A scale bound must be written the same as in whoever really checks it.
    lo_exclusive: bool = False
    choices: tuple[Any, ...] | None = None
    # The value in effect when the policy did NOT name the knob. `None` means "no
    # default", and the tool decides itself.
    #
    # The default lives here, not in the run config: a knob must not have two sources
    # of truth, and the generation mode is as much a policy knob as everything else in
    # `params_schema`.
    default: Any = None

    def check(self, value: Any) -> tuple[Any, str]:
        """Coerce a value to the knob's type. Returns (value, rejection reason)."""
        try:
            if self.type == "int":
                coerced: Any = int(value)
            elif self.type == "float":
                coerced = float(value)
            elif self.type == "bool":
                coerced = bool(value)
            else:
                coerced = str(value)
        except (TypeError, ValueError):
            return None, f"expected {self.type}, got {type(value).__name__}"
        if self.choices is not None and coerced not in self.choices:
            return None, f"allowed {list(self.choices)}, got {coerced!r}"
        if self.lo is not None and isinstance(coerced, (int, float)):
            if self.lo_exclusive and coerced <= self.lo:
                return None, f"must be strictly greater than {self.lo}"
            if not self.lo_exclusive and coerced < self.lo:
                return None, f"below the lower bound {self.lo}"
        if self.hi is not None and isinstance(coerced, (int, float)) and coerced > self.hi:
            return None, f"above the upper bound {self.hi}"
        return coerced, ""


@dataclass(frozen=True)
class ToolInfo:
    """Tool card for the policy: what it is, what can be tuned, what it costs.

    Without it the policy would see only the price (`state.price`), and neither the
    semantics of `n`, nor the allowed knobs, nor which counters a call moves. The
    practical result would be a policy turning `n` blindly and unable to tell "costly"
    from "cannot afford right now".

    There are two DIFFERENT scales here, and they must not be confused:

    - `forecast(n)` is the consumption of COUNTERS (`n_vlm`, `n_det`, `n_exec`, ...),
      i.e. what the `budget.*` ceilings and the contract's `cost_i` are computed from;
    - `price(n)` / `wall(n)` is the price in units of one small-candidate execution
      and in seconds, i.e. what profitability is computed from.

    One `stepwise` call with `n = 3` is three samples by the `vlm` counter but ONE
    request to the model by price. Adding the two together would give a plausible
    number that neither plans consumption nor compares tools.
    """

    name: str
    requires: str
    produces: str
    n_semantics: str
    extendable: bool
    terminal: bool
    deterministic: bool
    params: dict[str, ParamSpec] = field(default_factory=dict)
    cost: ToolCost = field(default_factory=lambda: ToolCost())
    spends: tuple[str, ...] = ()
    scales: tuple[str, ...] = ()
    # Legality threshold on the parent's code (`ToolSpec.max_code_numbers`): from this
    # many numbers (`code_numbers`) the tool is not offered on the candidate. It is in
    # the card so the policy can explain a refusal rather than guess it.
    max_code_numbers: int | None = None

    def forecast(self, n: int = 1) -> dict[str, int]:
        """How many COUNTERS a call with this `n` charges.

        Exactly the same arithmetic the harness uses to judge the legality of an
        action: kinds in `spends` are charged one per call, and those also in
        `scales` by `n`. There must be no second copy in the policy: it would drift
        from the first silently, and the policy would plan a consumption that will
        not happen.
        """
        count = max(1, int(n))
        return {kind: (count if kind in self.scales else 1) for kind in self.spends}

    def price(self, n: int = 1) -> dict[str, float]:
        """Price of a call in price-list units (one small-candidate execution)."""
        return self.cost.at(n)

    def wall(self, n: int = 1) -> float:
        """Estimated wall time of a call, seconds."""
        return self.cost.wall_at(n)


@dataclass(frozen=True)
class LegalAction:
    """What the harness considers permissible right now.

    The list is computed on every iteration from the registry, the pool, the budget
    and the attempt history. The policy may ask only for what is here; everything else
    is rejected by name and with a reason.
    """

    tool: str
    parent_id: str | None
    max_n: int
    # Why `n` is limited this way; goes to the log together with rejections.
    limited_by: str = ""


@dataclass
class Remaining:
    """Remaining amount per ceiling. Read by the policy, kept by the harness."""

    iterations: int | None = None
    calls: dict[str, int | None] = field(default_factory=dict)
    executions: int | None = None
    depth: int | None = None
    wall_sec: float | None = None


@dataclass
class SearchState:
    """Everything the policy knows about the part at decision time.

    Everything is readable and nothing is writable: every field here is kept by the
    harness, and a policy-side edit is either overwritten on the next iteration or
    (for `active`) means exactly what `select` returned.
    """

    figure_id: str
    gt_mesh_path: Path
    difficulty: dict[str, Any] = field(default_factory=dict)
    pool: list[Candidate] = field(default_factory=list)
    # Candidates born on the LAST iteration that ran. In `select` these are its own
    # offspring, in `plan` those of the previous one (the harness keeps the list and
    # clears it before launching actions).
    #
    # Introduced so the policy need not keep a set of "already seen": that is harness
    # bookkeeping, and every policy used to keep its own copy. A candidate the tool
    # returned again (the same code) is not included: it is not new but found again.
    fresh: list[Candidate] = field(default_factory=list)
    # The working set held by `select`. Pruning is SOFT: `plan` may address any pool
    # candidate, while `active` is only what the policy keeps at hand (for the
    # assistant variant it is also the prompt size). Hard pruning would close off a
    # class of strategies that return to earlier candidates.
    active: set[str] = field(default_factory=set)
    attempts: dict[str, list[Attempt]] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)
    # Cards of the tools in the RUN'S SET: the semantics of `n`, allowed knobs, price,
    # which counters a call moves. A tool absent from the set is absent here too,
    # otherwise the policy would read the price of something it will not be given
    # legally, and "costly" would differ from "unavailable" only by an empty legal list.
    tools: dict[str, ToolInfo] = field(default_factory=dict)
    remaining: Remaining = field(default_factory=Remaining)
    legal: list[LegalAction] = field(default_factory=list)
    # The policy's scratch memory for ONE part. Created with the state and dies with
    # it, so the policy need not keep its own field or reset it at iteration zero.
    #
    # This removes a trap: a policy is built once per run and outlives a part. While
    # memory lived in the policy itself, every variant had to remember to reset it,
    # and one that forgot leaked state between parts, silently and biasing the
    # measurement rather than crashing.
    memory: dict[str, Any] = field(default_factory=dict)
    # How many operations remain in the deterministic branch's pool for each parent.
    # The pool is finite, which is the natural bound of re-sampling: at zero the
    # action stops being legal.
    det_remaining: dict[str, int] = field(default_factory=dict)
    iteration: int = 0
    # Best-so-far on the harness scale. Kept by the harness and updated BEFORE pruning,
    # so the policy cannot lose a find by dropping a branch.
    best_id: str | None = None
    # At which iteration the current best became the best. Kept by the harness: it
    # alone knows when `best` changed, and a policy counting this itself would keep a
    # copy of foreign bookkeeping and drift from it silently. The difference
    # `iteration - best_set_iteration` is "how many turns the best has not changed",
    # i.e. observable stagnation: without it neither the policy nor the assistant sees
    # anything in the question except an unchanging line about the best.
    best_set_iteration: int = 0
    # Ask the assistant. Provided by the harness because the call must be charged to a
    # counter and logged; the question text is the policy's business.
    ask: Callable[..., str] | None = None
    # Whether the LAST assistant answer was cut off at the answer ceiling
    # (`max_tokens`). Ask right after `ask`. Needed because the text cannot tell:
    # a thinking model spends the ceiling on reasoning, the cut falls in the middle of
    # it, and a truncated answer looks like an answer in the wrong format, i.e. a reason
    # to fix the prompt rather than the ceiling. The flag comes from the endpoint
    # (`finish_reason == "length"`), not from the text.
    answer_truncated: Callable[[], bool] | None = None
    # Function calls of the LAST assistant answer, parsed by the server, when the
    # policy passed `ask(..., tools=...)`. Ask right after `ask`, like
    # `answer_truncated`. `None` means the harness does not provide the field (old or
    # hand-built): there are then no calls, and the policy parses the text.
    answer_tool_calls: Callable[[], list[dict[str, str]]] | None = None
    # The assistant's reasoning mode as set by the RUN (`experiment.agent.thinking`):
    # `True`/`False` if set, `None` if not set (the server template decides). Read, not
    # set: it is a condition of the run under which measurements were taken, not a
    # policy knob.
    #
    # The policy needs it because "cut off mid-reasoning" differs from "cut off in the
    # answer" by the presence of `</think>`, and with reasoning off the closing tag
    # never appears. Without the mode, the policy would read EVERY truncation as
    # unfinished thinking: discard the whole answer, advise "answer in the format, not
    # with reasoning" about a mechanism that did not exist, and re-ask with half the
    # ceiling, i.e. treat the wrong thing and do worse.
    agent_thinking: bool | None = None
    # Ceiling of the assistant's answer on a decision turn as set by the RUN
    # (`experiment.agent.answer_max_tokens`). Read, not set: a run condition. `None`
    # means the run did not set it and the policy uses its own constant.
    agent_answer_max_tokens: int | None = None
    # Whether the harness warms up the assistant's prefix cache (`experiment.agent.prewarm`).
    # Read, not set: when `True` the policy passes the unchanging question start to `ask`
    # under the `prefix` key; otherwise it sends no such key.
    agent_prewarm: bool = False
    # How many images the assistant endpoint accepts in ONE request
    # (the server's `--limit-mm-per-prompt`). Read, not set: a run condition.
    #
    # The policy needs it because images travel as a LIST and the server rejects a
    # request over the ceiling AS A WHOLE, not just the extra image. A scaffold that asks
    # for one panel more gets not "showed less" but "the assistant did not answer", on
    # every turn in a row: the diagnosis points at a dead channel instead of its own
    # constant. `None` means no ceiling is set.
    agent_max_images: int | None = None
    # Show candidates to the assistant: data blocks and a collage with the same labels
    # (`harness/state_render.py`). Also the harness, for the same reason state fields
    # are read rather than retold: otherwise an algorithmic block would judge by the
    # fields and an assistant block by its own string, and they would drift silently.
    # The prompt around this data is the policy's.
    render: Callable[..., Any] | None = None
    # Show the assistant the TARGET, without candidates. Separate from `render`
    # because the target is not a candidate: it has no pool id and no collage label,
    # and it must be viewable before the first candidate with a mesh appears.
    # Returns an image or `None`.
    render_target: Callable[[], Any] | None = None

    @property
    def price(self) -> dict[str, ToolCost]:
        """Price list of the run's tool set. Derived from the cards, not stored beside them.

        A second copy of the same dict would be exactly the thing that has repeatedly
        happened in this project: two versions of one number drifting apart silently.
        """
        return {name: info.cost for name, info in self.tools.items()}

    def cost_of(self, tool: str, n: int = 1) -> dict[str, float]:
        """Price of a call in price-list units. An empty dict means the tool is not in the set."""
        info = self.tools.get(tool)
        return {} if info is None else info.price(n)

    def wall_of(self, tool: str, n: int = 1) -> float:
        """Estimated wall time of a call, seconds. Zero means the tool is not in the set."""
        info = self.tools.get(tool)
        return 0.0 if info is None else info.wall(n)

    def affordable_n(self, tool: str, want: int, *, pace: bool = False, spread: int = 1) -> int:
        """The largest `n` <= `want` that the remainder pays for. 0 means none.

        Computed from COUNTERS (`ToolInfo.forecast`), not the price list: the `budget.*`
        ceilings are set in calls, and comparing them with a price in small-candidate
        executions would add up different scales. Wall time is checked separately and by
        the price list, as there is no other source of second estimates.

        `pace=True` additionally divides the remainder by the number of remaining
        iterations: "how much can I afford HERE so as to last to the end". The share
        never drops below one unit, otherwise a policy with twelve iterations and ten
        calls in the ceiling would make no call at all.

        `spread` is how many actions of ONE iteration this share is divided among.
        The remainder does not change while an iteration is being planned (it is
        recomputed after actions launch), so a beam of width three asking three times in
        a row would get three full shares and spend triple. Only the policy knows the
        number of parallel actions, hence a parameter rather than a harness guess.

        It lives here, not in each policy: it is arithmetic over data the harness keeps,
        and several copies would drift apart at the first change to the cost model.
        """
        info = self.tools.get(tool)
        want = int(want)
        if info is None or want <= 0:
            return 0
        left = self.remaining.calls
        share = max(1, int(spread))
        if pace and self.remaining.iterations:
            share *= max(1, int(self.remaining.iterations))
        wall_left = self.remaining.wall_sec

        for n in range(want, 0, -1):
            forecast = info.forecast(n)
            if any(
                _allowance(left.get(kind), share) is not None
                and need > _allowance(left.get(kind), share)
                for kind, need in forecast.items()
            ):
                continue
            total = _allowance(left.get("total"), share)
            if total is not None and sum(forecast.values()) > total:
                continue
            if wall_left is not None and info.wall(n) > wall_left:
                continue
            return n
        return 0

    def legal_for(self, tool: str, parent_id: str | None) -> LegalAction | None:
        """Whether calling `tool` from this parent is legal now, and with what `max_n`.

        It lives here, not in each policy: it is a query over data the harness keeps,
        not a decision. Several policies kept a verbatim identical `_legal`, and every
        new variant would derive it anew.
        """
        for item in self.legal:
            if item.tool == tool and item.parent_id == parent_id:
                return item
        return None

    def get(self, candidate_id: str | None) -> Candidate | None:
        if candidate_id is None:
            return None
        for candidate in self.pool:
            if candidate.id == candidate_id:
                return candidate
        return None

    def lineage_tools(self, candidate: Candidate | None) -> frozenset[str]:
        """What built the candidate's code: its tool and the tools of all ancestors.

        It lives here for the same reason as `legal_for`: a query over the pool the
        harness keeps, not a policy decision. The lineage is needed because
        `origin.tool` answers only for the LAST step: a `stepwise` that continued a det
        candidate is marked `stepwise` while the code is entirely det's, and a policy
        that fences itself off by `origin.tool` is fenced only from direct det proposals.

        The chain reaches the root: the pool only grows, candidates are never removed
        from it (`search.py`, `self.pool.append`).
        """
        tools: set[str] = set()
        seen: set[str] = set()
        while candidate is not None and candidate.id not in seen:
            seen.add(candidate.id)
            tools.add(candidate.origin.tool)
            candidate = self.get(candidate.parent_id)
        return frozenset(tools)

    def fitness_of(self, candidate: Candidate | None) -> float | None:
        """The harness scale for this candidate: the scale the part is returned by.

        A query, not a candidate field, and that matters. Next to `metrics` the number
        would read as one more measured quantity, while it is a harness fold over a
        measurement; asked for explicitly, it names itself.

        Why the policy should see it although the selection scale is its own: so as not
        to rank blindly. A policy that ranks its menu by a different scale (`iou`) while
        the part is returned by the contract scale would honestly improve a number other
        than the one it is scored by.

        `None` means the fold is not computed on this candidate (see `harness_fitness`).
        """
        return None if candidate is None else harness_fitness(candidate.metrics)

    def lineage(self, candidate: Candidate | None) -> list[Candidate]:
        """Chain of the candidate's ancestors, from the root to the candidate itself.

        It lives here for the same reason as `lineage_tools`: a walk over the pool the
        harness keeps. It is returned as a list, not a folded number, because there are
        several readers with different questions: "what did the branch start with",
        "how many operations gave nothing", "which tool was at which step". Folding it
        into one number would choose the question for the policy.

        The pool root (`c0`) is not in the chain: it has neither a measurement nor an
        operation; it is an empty reference point, not a step.
        """
        chain: list[Candidate] = []
        seen: set[str] = set()
        while candidate is not None and candidate.id not in seen:
            seen.add(candidate.id)
            if candidate.depth > 0:
                chain.append(candidate)
            candidate = self.get(candidate.parent_id)
        chain.reverse()
        return chain

    @property
    def best(self) -> Candidate | None:
        return self.get(self.best_id)

    def alive(self) -> list[Candidate]:
        """Candidates with a measurement: what it makes sense to choose from."""
        return [candidate for candidate in self.pool if candidate.alive]

    def attempts_on(self, candidate_id: str | None, tool: str | None = None) -> list[Attempt]:
        rows = self.attempts.get(candidate_id or "", [])
        return [row for row in rows if tool is None or row.origin.tool == tool]

    def tool_stats(self, tool: str, window: int | None = None) -> tuple[int, int]:
        """How much the tool returned and how much of it built.

        `window` is how many recent iterations to count (None means the whole part). It
        lives here because it is computed from `attempts`, which the harness keeps: a
        policy that kept its own call history would hold a second version of the same
        number, and two versions of one number have drifted apart silently before.
        """
        produced = built = 0
        for rows in self.attempts.values():
            for row in rows:
                if row.origin.tool != tool:
                    continue
                if window is not None and row.iteration <= self.iteration - window:
                    continue
                produced += row.produced
                built += row.built
        return produced, built

    def settled(self) -> list[Candidate]:
        """The pool minus the fresh: what was known before the last iteration.

        Answers "is what we just bought better than what we already had". Each policy
        used to answer it itself by accumulating the best value in part memory; now it
        is derived from the pool and there is nothing to accumulate.
        """
        born = {candidate.id for candidate in self.fresh}
        return [candidate for candidate in self.pool if candidate.id not in born]

    def scored(
        self, candidates: list[Candidate], value: Callable[[dict[str, Any] | None], float | None]
    ) -> list[tuple[Candidate, float]]:
        """Candidates with a value on the policy's scale; unmeasurable ones dropped.

        The scale is set by the policy itself (`value`); the harness neither knows nor
        imposes it. Here is only what was verbatim identical in all policies: compute
        and drop `None`. A forgotten `None` filter is the cheapest way for a mutant to
        compare a number with nothing and crash.
        """
        rows = [(candidate, value(candidate.metrics)) for candidate in candidates]
        return [(candidate, item) for candidate, item in rows if item is not None]

    def frontier(self) -> list[Candidate]:
        """Whom to extend by default: the working set, else the best, else the root.

        A query over harness data, not a decision: a policy may address anyone in the
        pool. It is here because several policies kept this verbatim identical, and
        every new variant would derive it anew.
        """
        if self.active:
            return [c for c in (self.get(cid) for cid in sorted(self.active)) if c is not None]
        best = self.best
        if best is not None:
            return [best]
        return self.pool[:1]


def _allowance(left: int | None, share: int) -> int | None:
    """How many units of a cost kind may be spent here. `None` means no ceiling.

    The floor of one unit lets a policy with twelve iterations and ten calls in the
    ceiling make at least one call rather than none. It does not apply to an EXHAUSTED
    ceiling: `max(1, 0 // 12)` would return one, i.e. a zero remainder would read as
    "enough for one more call". This very error made "the remainder is not enough for
    anything" a condition that never fires, and it was found by a smoke run, not by reading.
    """
    if left is None:
        return None
    left = int(left)
    return 0 if left <= 0 else max(1, left // max(1, share))


@dataclass
class Action:
    """One action of a plan: whom to extend, with what, how and how many."""

    tool: str
    parent_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    n: int = 1
    # Goes to the log and affects nothing. Needed so the run directory shows not only
    # what the scaffold did but why.
    reason: str = ""


@dataclass
class Plan:
    """What the policy asks to launch on this iteration, and whether it is time to stop.

    `done` exists because the decision "the part is ready" is often visible exactly
    where the policy looks at the pool and chooses an action, not a move later.
    Otherwise a planning policy has to buy this decision with a SEPARATE question in
    `select()`, which has neither the fresh report of the move (it reaches the policy
    memory only at the start of the next `plan()`) nor this move's images.

    `done` must still not be merged with an EMPTY plan (see `STOP_DONE`): an empty plan
    without the flag is `plan_empty`, a broken variant, and it must stay distinguishable
    from a scaffold that finished deliberately. Hence the flag is explicit rather than
    derived from the absence of actions.

    The flag together with actions in one plan is legal and means "this move is the
    last": the actions run, and the part ends after them.

    `stalled` is the second legal reason to end: "attempts stopped improving", not
    "ready". Without it a stall left through `done` with a false justification about
    quality, and the outcome merged convergence with surrender. It is read only together
    with `done`; the harness records it in the `done_by` field (`DONE_BY_STALL`), and
    `stop_reason` does not change because of it.
    """

    actions: list[Action] = field(default_factory=list)
    done: bool = False
    reason: str = ""
    stalled: bool = False


@dataclass
class Verdict:
    """Whether we are finished and what to keep at hand."""

    # `None` means "do not touch the working set". An empty set means exactly the
    # opposite: clear it. The default used to be the empty set, and a verdict built
    # without `keep` silently extinguished the beam.
    keep: set[str] | None = None
    done: bool = False
    reason: str = ""


# Stop reasons.
# `done` belongs to the policy, the others to the harness. An empty plan must not be
# merged with `done`: a broken variant would look like a scaffold that quickly and
# cheaply decides it is ready. The policy says `done` in two ways, `Verdict.done` from
# `select()` and `Plan.done` from `plan()`, and both lead here: the part's outcome is
# one, only the moment the policy realized it differs.
STOP_DONE = "done"
# The policy said `done`, but it had no choice left: the remainder did not cover any
# legal action. A separate outcome rather than `done`, because these are different
# questions about a run: `done` is a judgment of QUALITY ("good enough"), `done:exhausted`
# is hitting the PRICE. A planning policy (`affordable_n`) stops itself a step before the
# harness ceiling, so `wall`/`limit:*` never occur with it, and without this line the
# price would silently read as quality. It is determined by the FACT of the remainder,
# not by the reason text: the wording is written by the policy, and a mutant writes anything.
STOP_DONE_EXHAUSTED = "done:exhausted"
# Through which door the policy ended the part: the `done_by` field of the log line,
# the `search_stop` event and the per-figure record. One field for three cases:
# `quality` is `Plan.done`, a judgment of quality; `stall` is `Plan.done` with
# `stalled`, "attempts stopped improving"; `select` is `Verdict.done`. The `select`
# door is distinguished by FACT, not by words: the wording is written by the policy.
# A separate field rather than new `stop_reason` values: reports read `done`, and a
# rename would cut their comparison by outcome.
DONE_BY_QUALITY = "quality"
DONE_BY_STALL = "stall"
DONE_BY_SELECT = "select"
STOP_NO_LEGAL = "no_legal_action"
STOP_PLAN_EMPTY = "plan_empty"
STOP_PLAN_INVALID = "plan_invalid"
STOP_LIMIT_ITERATIONS = "limit:iterations"
STOP_LIMIT_CALLS = "limit:calls"
STOP_LIMIT_EXEC = "limit:exec"
STOP_LIMIT_DEPTH = "limit:depth"
STOP_WALL = "wall"
STOP_GENOME_ERROR = "genome_error"
