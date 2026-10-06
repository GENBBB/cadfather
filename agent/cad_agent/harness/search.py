"""Search loop: the unit is an iteration, not a DSL step.

The `scaffold.run(obs, res)` contract seam is implemented here: callers
(`run_eval`) see nothing changed, while the loop itself lives in the harness,
not in the policy.

Why. Call and wall-time ceilings used to rely on the policy politely letting
`BudgetExceeded` propagate. A policy written with `except Exception: pass`
would silently disable both ceilings, yet the ceilings must hold regardless of
what the policy does. So ceilings, counters, seeds, dedup, best-so-far and
action legality are owned by the loop, and the policy keeps two decisions:

    plan(state)   -> what to run next
    select(state) -> whether to finish and what to keep at hand (OPTIONAL)

The names `plan` and `select` are fixed. `select` may be omitted: a policy that
finishes a part with the `Plan.done` flag and does not use the working front
gets the default "change nothing, not done".

Three consequences worth remembering:

- **The live pool is all candidates of all depths.** Rolling back to an old
  ancestor, returning to a pruned candidate and re-sampling are the same
  action: choosing a parent from the pool. There are no separate mechanisms.
- **Pruning is soft.** `plan` may address any candidate; `active` is only the
  policy's working set. Hard pruning would rule out backtracking strategies.
- **Stopping belongs to the policy, through two doors.** `done` comes either as
  a `select` verdict or as the `Plan.done` flag, and both lead to the same
  outcome; an empty plan and an illegal plan are separate harness reasons.
  Merging them would make a broken policy indistinguishable from a frugal one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Iterable, Sequence

from cad_agent.capabilities import objective as objective_mod
from cad_agent.harness import config as config_mod
from cad_agent.harness import state_render
from cad_agent.harness import tools as tools_mod
from cad_agent.harness.budget import (
    WALL_STOP_REASON,
    BudgetExceeded,
    DeadlineExceeded,
)
from cad_agent.harness.search_types import (
    DONE_BY_QUALITY,
    DONE_BY_SELECT,
    DONE_BY_STALL,
    FALLBACK_FIELD,
    STOP_DONE,
    STOP_DONE_EXHAUSTED,
    STOP_GENOME_ERROR,
    STOP_LIMIT_CALLS,
    STOP_LIMIT_DEPTH,
    STOP_LIMIT_EXEC,
    STOP_LIMIT_ITERATIONS,
    STOP_NO_LEGAL,
    STOP_PLAN_EMPTY,
    STOP_PLAN_INVALID,
    STOP_WALL,
    Action,
    Attempt,
    Candidate,
    LegalAction,
    Origin,
    Plan,
    Remaining,
    SearchState,
    Verdict,
    harness_fitness as _fitness,
)
from cad_agent.harness.seeding import DEFAULT_RUN_SEED, attempt_tag, derive_seed
from cad_agent.scaffold.base import Observation, Reconstruction

logger = logging.getLogger(__name__)

# Default ceilings. They live in the harness and are set by the config
# (`experiment.limits`), not by the policy: run liveness must not depend on the
# policy.
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_MAX_DEPTH = 30
# Hard ceiling on ONE action. Guards against a policy asking for `n = 10**6`:
# the other ceilings derive from the remaining budget, which may be unlimited
# (`null` in the config).
MAX_N_PER_ACTION = 32


def _harness_rank(metrics: dict[str, Any] | None) -> tuple[int, float] | None:
    """Key by which the harness picks the best candidate. Larger is better.

    Two tiers with a strict order: a candidate measured on the contract scale
    always beats one measured only on the fallback scale. Otherwise different
    scales would be compared and "best" would depend on which metrics happened
    to be computed.
    """
    value = _fitness(metrics)
    if value is not None:
        return (1, value)
    if not metrics:
        return None
    fallback = metrics.get(FALLBACK_FIELD)
    try:
        fallback = float(fallback)
    except (TypeError, ValueError):
        return None
    if fallback != fallback:  # NaN
        return None
    # CD: lower is better, so it enters the "larger is better" scale negated.
    return (0, -fallback)


class SearchLoop:
    """Harness part of a rollout. The policy is an object with `plan` and `select`.

    The object outlives a part (the config builds it once per run), so all
    search state lives in `run`, not in fields.
    """

    def __init__(
        self,
        policy: Any,
        *,
        max_iterations: int | None = None,
        max_depth: int | None = None,
        tools: Sequence[str] | None = None,
    ):
        self.policy = policy
        # None means "take from the run config": ceilings are a harness knob,
        # and keeping the value in two places would create a second source of
        # truth.
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.tools = tuple(tools) if tools else None

    # --- contract entry ----------------------------------------------------

    def run(self, obs: Observation, res: Any) -> Reconstruction:
        run = _FigureSearch(self, obs, res)
        return run.execute()


class _FigureSearch:
    """Search state for one part. Lives for exactly one `run` call."""

    def __init__(self, loop: SearchLoop, obs: Observation, res: Any):
        self.loop = loop
        self.policy = loop.policy
        self.obs = obs
        self.res = res
        self.journal = getattr(res, "journal", None)
        config = dict(getattr(res, "config", {}) or {})
        limits = dict(config.get("limits") or {})
        self.run_seed = int(config.get("seed", DEFAULT_RUN_SEED))
        self.max_iterations = _first_not_none(
            loop.max_iterations, limits.get("iterations"), DEFAULT_MAX_ITERATIONS
        )
        self.max_depth = _first_not_none(loop.max_depth, limits.get("depth"), DEFAULT_MAX_DEPTH)

        # Run toolset: explicitly passed (tests), otherwise `experiment.tools`.
        # A tool missing from the set is removed HERE rather than refusing from
        # inside. Otherwise the policy would burn iterations on an action that
        # always returns nothing, which the journal would show as a failing
        # tool rather than a run setting.
        asked = tuple(loop.tools) if loop.tools else config_mod.run_tools(config)
        # Separate from the config and applied after it: this is an environment
        # fact, not a setting. The config may legitimately ask for repair while
        # the assistant server is down, and then the run's toolset differs from
        # the requested one. The difference is recorded so that a run without
        # the assistant and a run that did not ask for it do not both show the
        # same zeros for different reasons.
        available = asked
        if getattr(res, "ask_agent", None) is None:
            available = tuple(name for name in available if name != "repair")
        self.specs = [tools_mod.get(name) for name in available]
        self.tool_names = available
        self.tools_withheld = tuple(name for name in asked if name not in available)

        self.pool: list[Candidate] = []
        # The same list as `state.fresh`: the loop fills it on materialization,
        # and `_run_actions` creates a new one every iteration.
        self.fresh: list[Candidate] = []
        self.by_code: dict[str, Candidate] = {}
        self.attempts: dict[str, list[Attempt]] = {}
        self.next_id = 0
        self.best: Candidate | None = None
        # Best key (scale tier and value) and its value on the contract scale.
        # The latter may be `None`: then the best was chosen on the fallback
        # scale, which the part record should show rather than look like zero.
        # (validity, scale tier, value) - see `_update_best`.
        self.best_rank: tuple[int, int, float] | None = None
        # Who was best at the previous update and at which iteration it became
        # so: stagnation is observable only as a difference from the current
        # iteration.
        self._best_id_seen: str | None = None
        self.best_set_iteration: int = 0
        self.best_fitness: float | None = None
        self.genome_cpu_sec = 0.0
        self.journal_rows: list[dict[str, Any]] = []

    # --- main loop ---------------------------------------------------------

    def execute(self) -> Reconstruction:
        result = Reconstruction(figure_id=self.obs.figure_id)
        if self.tools_withheld:
            self._event("tools_withheld", tools=list(self.tools_withheld),
                        reason="capability missing from the run environment",
                        available=list(self.tool_names))
        state = self._init_state()
        stop_reason = ""
        stop_detail = ""
        done_by: str | None = None

        while True:
            limit_stop = self._limit_reached(state)
            if limit_stop:
                stop_reason = limit_stop
                break

            state.legal, blocked_by_depth = self._legal_actions(state)
            self._log_legal(state)
            if not state.legal:
                # An exact reason beats a generic one: "all live candidates hit
                # the depth ceiling" and "nothing to continue from" are fixed
                # differently.
                stop_reason = STOP_LIMIT_DEPTH if blocked_by_depth else STOP_NO_LEGAL
                break

            try:
                plan = self._genome(self.policy.plan, state)
            except (DeadlineExceeded, BudgetExceeded) as exc:
                stop_reason, stop_detail = self._stop_from_budget(exc)
                break
            except Exception as exc:
                # A broken policy loses its part, not the run. The reason is
                # distinct so that the summary shows the policy crashed rather
                # than stopped frugally.
                logger.exception("policy plan() crashed on part %s", self.obs.figure_id)
                stop_reason, stop_detail = STOP_GENOME_ERROR, repr(exc)
                result.error = repr(exc)
                break

            actions, rejected = self._validate(plan, state)
            self._log_rejected(rejected, state)
            if not actions:
                # An empty plan and a fully rejected one are different
                # diagnoses: the first means "the policy wanted nothing while
                # actions were live", the second "it asked only for the
                # impossible". If ALL rejections hit the same harness ceiling,
                # that is the part's outcome and calling it a policy fault
                # would be wrong.
                #
                # An explicit `Plan.done` is checked first: the policy did not
                # "want nothing", it finished, and only the flag tells the two
                # cases apart. It must not be inferred from an empty plan (see
                # `STOP_DONE`).
                if getattr(plan, "done", False):
                    stop_reason, stop_detail = self._done_reason(state), getattr(plan, "reason", "")
                    done_by = self._done_by(plan, verdict_done=False)
                    # The finish record must be written on THIS exit too:
                    # otherwise a policy that finishes a part with no actions
                    # leaves the loop before the journal write and its
                    # rationale survives only in `events.jsonl`.
                    self.journal_rows.append(
                        self._journal_row(state, [], done=True, reason=stop_detail,
                                          done_by=done_by)
                    )
                    break
                # `or ""` is load-bearing. Only two of the six rejection
                # branches set `stop_hint`; for the rest `.get` returns `None`,
                # and a set holding a single `None` passed the "no empty
                # hints" check, so `stop_reason` became `None` and the
                # `STOP_PLAN_INVALID` branch never fired.
                hints = {row.get("stop_hint") or "" for row in rejected}
                if rejected and len(hints) == 1 and "" not in hints:
                    stop_reason = hints.pop()
                else:
                    stop_reason = STOP_PLAN_EMPTY if not rejected else STOP_PLAN_INVALID
                break

            # Taken BEFORE execution: `plan` does not change later, but reading
            # the flag after the branches risks reading it on the wrong path.
            plan_done = bool(getattr(plan, "done", False))
            try:
                self._run_actions(actions, state)
            except DeadlineExceeded as exc:
                # The wall ceiling is not a failure: what was found stays, the
                # rollout stops.
                stop_reason, stop_detail = STOP_WALL, str(exc)
                break
            except BudgetExceeded as exc:
                stop_reason, stop_detail = self._stop_from_budget(exc)
                break

            self._refresh(state)

            # `select` is OPTIONAL. A policy that decides to finish in `plan()`
            # via `Plan.done` and does not use the working front would declare
            # a function with no outputs. The default `keep=None` means "leave
            # the working set as is" and differs from the empty set, with which
            # a policy deliberately drops branches.
            selector = getattr(self.policy, "select", None)
            if selector is None:
                verdict = Verdict(keep=None, done=False, reason=None)
            else:
                try:
                    verdict = self._genome(selector, state)
                except (DeadlineExceeded, BudgetExceeded) as exc:
                    stop_reason, stop_detail = self._stop_from_budget(exc)
                    break
                except Exception as exc:
                    logger.exception("policy select() crashed on part %s", self.obs.figure_id)
                    stop_reason, stop_detail = STOP_GENOME_ERROR, repr(exc)
                    result.error = repr(exc)
                    break

            if verdict.keep is not None:
                # `None` means "leave as is"; an empty set means "reset". They
                # differ: the first is written by a policy with nothing to
                # change, the second by one that deliberately drops all
                # branches.
                state.active = self._accept_keep(verdict.keep, state)
            if verdict.done or plan_done:
                done_by = self._done_by(plan, verdict_done=bool(verdict.done))
            self.journal_rows.append(self._journal_row(
                state, actions,
                done=bool(verdict.done or plan_done),
                reason=verdict.reason or getattr(plan, "reason", "") or None,
                done_by=done_by,
            ))
            if verdict.done or plan_done:
                # `Plan.done` together with actions means "this move is the
                # last": the actions are already executed and the part ends
                # after them. Silently ignoring the flag is wrong: a policy
                # that said "done" and got ten more iterations reads it as a
                # broken seam, and the journal would not distinguish it from
                # the policy changing its mind.
                stop_reason = self._done_reason(state)
                stop_detail = verdict.reason or getattr(plan, "reason", "")
                break

            state.iteration += 1
            self._refresh(state)

        return self._finish(result, state, stop_reason, stop_detail, done_by)

    # --- initialization ----------------------------------------------------

    def _init_state(self) -> SearchState:
        """The part root: a code prefix with no mesh and no measurement.

        It sits in the pool like the others, so `det_cold` and `stepwise` get a
        parent without knowing it is special, and `plan` addresses it with an
        ordinary `parent_id`.
        """
        root = Candidate(
            id=self._new_id(),
            parent_id=None,
            depth=0,
            origin=Origin(tool="root"),
            code=self.obs.prefix_code,
            built=True,        # the prefix is executable by construction
            metrics=None,      # but not measured: nothing to measure in it
            extendable=True,
        )
        self.pool.append(root)
        state = SearchState(
            figure_id=self.obs.figure_id,
            gt_mesh_path=self.obs.gt_mesh_path,
            difficulty=dict(self.obs.difficulty or {}),
            pool=self.pool,
            active={root.id},
            attempts=self.attempts,
            tools=tools_mod.cards(self.tool_names),
            ask=getattr(self.res, "ask_agent", None),
            answer_truncated=getattr(self.res, "answer_truncated", None),
            answer_tool_calls=getattr(self.res, "answer_tool_calls", None),
            agent_thinking=getattr(self.res, "agent_thinking", None),
            agent_answer_max_tokens=getattr(self.res, "agent_answer_max_tokens", None),
            agent_prewarm=bool(getattr(self.res, "agent_prewarm", False)),
            agent_max_images=getattr(self.res, "agent_image_limit", None),
            render=state_render.build_renderer(self.res, self.obs, self._lookup),
            render_target=state_render.build_target_renderer(self.res, self.obs),
        )
        self._refresh(state)
        # Target difficulty features are computed HERE, before the first
        # execution. They tell the policy what it would otherwise learn only by
        # measuring: with a non-watertight GT no candidate has a volumetric
        # IoU, and waiting for it wastes an iteration. The cost is loading the
        # GT, which is loaded (cached) on the first measurement anyway.
        if not state.difficulty:
            probe = getattr(self.res, "difficulty", None)
            if probe is not None:
                try:
                    state.difficulty = dict(probe() or {})
                except Exception as exc:
                    # Does not crash the rollout: `for_detail` reads empty
                    # features as "status undefined" and leaves the goal alone.
                    logger.warning("Complexity features could not be computed: %s", exc)
                    state.difficulty = {"error": repr(exc)[:200]}

        return state

    def _lookup(self, candidate_id: str) -> Candidate | None:
        """Candidate by id - from the live pool, not from a copy.

        The state renderer calls this: the policy addresses candidates with the
        same ids the loop uses to check action legality, so the same table must
        answer it.
        """
        for candidate in self.pool:
            if candidate.id == candidate_id:
                return candidate
        return None

    def _new_id(self) -> str:
        candidate_id = f"c{self.next_id}"
        self.next_id += 1
        return candidate_id

    # --- ceilings and remainders -------------------------------------------

    def _budget(self):
        return getattr(self.res, "budget", None)

    def _remaining_calls(self) -> dict[str, int | None]:
        budget = self._budget()
        if budget is None:
            return {}
        remaining: dict[str, int | None] = {}
        for kind, spent in budget.counts.items():
            limit = budget.limits.get(kind)
            remaining[kind] = None if limit is None else max(0, int(limit) - int(spent))
        total_limit = budget.limits.get("total")
        if total_limit is not None:
            remaining["total"] = max(0, int(total_limit) - budget.total)
        return remaining

    def _refresh(self, state: SearchState) -> None:
        """Recompute everything the policy reads: remainders, spend, det pool, best."""
        budget = self._budget()
        remaining_calls = self._remaining_calls()
        depth_reached = max((c.depth for c in self.pool), default=0)
        state.remaining = Remaining(
            iterations=None if self.max_iterations is None
            else max(0, int(self.max_iterations) - state.iteration),
            calls=remaining_calls,
            executions=remaining_calls.get("exec"),
            depth=None if self.max_depth is None else max(0, int(self.max_depth) - depth_reached),
            wall_sec=None if budget is None or budget.wall_sec is None
            else max(0.0, float(budget.wall_sec) - budget.elapsed_sec),
        )
        spent: dict[str, float] = dict(budget.counts) if budget is not None else {}
        # The policy's own time is caught by no call counter: blocks may write
        # scoring functions over code, mesh and images that cost nothing by
        # `cost_i` but eat wall time. A separate cost type.
        spent["genome_cpu"] = round(self.genome_cpu_sec, 4)
        state.spent = spent
        state.best_id = None if self.best is None else self.best.id
        # A change of best is noted HERE, not where it is chosen
        # (`_track_best`): the iteration would not reach there, and a second
        # iteration counter in the loop would be a second owner of the turn
        # number.
        if self.best is not None and self.best.id != self._best_id_seen:
            self._best_id_seen = self.best.id
            self.best_set_iteration = state.iteration
        state.best_set_iteration = self.best_set_iteration
        state.det_remaining = self._det_remaining()

    def _pool_left(self, spec: tools_mod.ToolSpec, parent_id: str, state: SearchState) -> int | None:
        """Remainder of THIS tool's pool for this parent.

        The registry (`ToolSpec.pool_field`), not the loop, knows which state
        field to ask. Reading `det_remaining` for any rank-style tool would let
        a second such tool be removed from the legal set by another tool's
        drained pool and look broken.
        """
        if not spec.pool_field:
            return None
        return getattr(state, spec.pool_field, {}).get(parent_id)

    def _det_remaining(self) -> dict[str, int]:
        """Remainder of the deterministic branch's pool per parent.

        The key is a candidate id, not a mesh path: only our ids cross into the
        policy. `None` (pool not computed) is omitted entirely - "unknown" and
        "empty" differ by key presence.
        """
        ask = getattr(self.res, "det_remaining", None)
        if ask is None:
            return {}
        out: dict[str, int] = {}
        for candidate in self.pool:
            probe = None if candidate.depth == 0 and not candidate.mesh_path else candidate.mesh_path
            left = ask(probe)
            if left is not None:
                out[candidate.id] = int(left)
        return out

    def _limit_reached(self, state: SearchState) -> str:
        remaining = state.remaining
        if remaining.wall_sec is not None and remaining.wall_sec <= 0:
            return STOP_WALL
        if remaining.iterations is not None and remaining.iterations <= 0:
            return STOP_LIMIT_ITERATIONS
        if remaining.executions is not None and remaining.executions <= 0:
            return STOP_LIMIT_EXEC
        total = remaining.calls.get("total")
        if total is not None and total <= 0:
            return STOP_LIMIT_CALLS
        return ""

    # --- action legality ---------------------------------------------------

    def _legal_actions(self, state: SearchState) -> tuple[list[LegalAction], bool]:
        legal: list[LegalAction] = []
        blocked_by_depth = False
        remaining = state.remaining

        for spec in self.specs:
            # A ceiling exhausted for a call type removes the tool entirely: the
            # policy must not learn of it by an exception after the charge.
            if any(_zero(remaining.calls.get(kind)) for kind in spec.spends):
                continue
            for parent in self.pool:
                if not spec.accepts(parent):
                    continue
                if spec.grows_depth and self.max_depth is not None and parent.depth >= self.max_depth:
                    blocked_by_depth = True
                    continue
                if spec.deterministic and state.attempts_on(parent.id, spec.name):
                    # Same input, same answer: a repeat would return a copy for
                    # the price of a full call.
                    continue
                left = self._pool_left(spec, parent.id, state)
                if spec.n_semantics == "rank_depth" and left is not None and left <= 0:
                    # Pool drained: nothing more to ask of this parent.
                    continue
                max_n, limited_by = self._max_n(spec, parent, state)
                if max_n <= 0:
                    continue
                legal.append(
                    LegalAction(tool=spec.name, parent_id=parent.id, max_n=max_n, limited_by=limited_by)
                )
        return legal, blocked_by_depth and not legal

    def _max_n(self, spec: tools_mod.ToolSpec, parent: Candidate, state: SearchState) -> tuple[int, str]:
        """How much can be asked of this tool right now, and what limits it."""
        cap, reason = MAX_N_PER_ACTION, "per-action cap"
        for kind in spec.scales:
            left = state.remaining.calls.get(kind)
            if left is not None and left < cap:
                cap, reason = int(left), f"remaining calls of {kind}"
        if spec.n_semantics == "rank_depth":
            left = self._pool_left(spec, parent.id, state)
            if left is not None and left < cap:
                cap, reason = int(left), f"remaining pool of {spec.name}"
        if spec.n_semantics == "single":
            # Exactly one result: asking such a tool for five copies is
            # meaningless. Decided by registry semantics, not by tool name, so
            # a second such tool does not require changing the loop.
            cap, reason = min(cap, 1), "the tool returns one result"
        return max(0, cap), reason

    # --- plan validation ---------------------------------------------------

    def _validate(self, plan: Any, state: SearchState) -> tuple[list[Action], list[dict[str, Any]]]:
        """Keep the legal actions, return the rest with a reason for each.

        Rejections are logged by name on purpose. Silently cleaning the list
        would make a policy that always asks for the impossible
        indistinguishable from one that stops at the first step.
        """
        actions = list(getattr(plan, "actions", None) or [])
        allowed = {(item.tool, item.parent_id): item for item in state.legal}
        accepted: list[Action] = []
        rejected: list[dict[str, Any]] = []
        # Remainder ALREADY claimed by earlier actions of this plan. Legality
        # is computed once per iteration while a plan holds several actions;
        # without this counter each would get the whole remainder, so a plan of
        # three det calls with one call left would run in full - the ceiling
        # would be soft for the whole plan instead of per call.
        left = {
            kind: value for kind, value in state.remaining.calls.items() if value is not None
        }

        for action in actions:
            if not isinstance(action, Action):
                rejected.append({"action": repr(action)[:200], "reason": "not an Action"})
                continue
            legal = allowed.get((action.tool, action.parent_id))
            if legal is None:
                reason, hint = self._why_illegal(action, state)
                rejected.append({
                    "tool": action.tool,
                    "parent": action.parent_id,
                    "reason": reason,
                    "stop_hint": hint,
                })
                continue
            spec = tools_mod.get(action.tool)
            params, why = tools_mod.validate_params(spec, action.params)
            if why:
                # A parameter of the wrong scale or name makes this a rejected
                # action, not a cleaned one. Otherwise a policy asking for
                # `temperature: 12` would run at the run's temperature and
                # conclude "the knob changes nothing".
                rejected.append({
                    "tool": action.tool,
                    "parent": action.parent_id,
                    "reason": "; ".join(why),
                })
                continue
            n = int(action.n or 0)
            if n <= 0:
                rejected.append({"tool": action.tool, "parent": action.parent_id,
                                 "reason": f"n={action.n}"})
                continue
            if n > legal.max_n:
                # Clip and say so, rather than reject: a policy asking for more
                # than the remainder is workable, it just does not know what is
                # left. Silently running less than requested would make the
                # journal lie.
                self._event("plan_clipped", tool=action.tool, parent=action.parent_id,
                            requested=n, allowed=legal.max_n, limited_by=legal.limited_by)
                n = legal.max_n

            fitted = self._fit_to_plan(spec, n, left)
            if fitted <= 0:
                rejected.append({
                    "tool": action.tool,
                    "parent": action.parent_id,
                    "reason": "the remaining cap was taken by earlier plan actions",
                    "stop_hint": STOP_LIMIT_CALLS,
                })
                continue
            if fitted < n:
                self._event("plan_clipped", tool=action.tool, parent=action.parent_id,
                            requested=n, allowed=fitted, limited_by="remaining iteration plan")
                n = fitted

            # The knob combination is checked here, not in `validate_params`:
            # the PAIR (knob, `n`) is illegal, and `n` is final only now - above
            # it was still being cut by the action ceiling and the plan
            # remainder. Checking before truncation would reject a request for
            # 13 greedy samples even where the remainder would cut it to one,
            # a legal call.
            why_pair = tools_mod.check_combination(spec, params, n)
            if why_pair:
                rejected.append({
                    "tool": action.tool,
                    "parent": action.parent_id,
                    "reason": why_pair,
                })
                continue

            for kind, need in tools_mod.card(spec).forecast(n).items():
                if kind in left:
                    left[kind] -= need
                # The total ceiling is not a call type but a sum, and is charged
                # as a sum. Forgetting it here would hold the plan per counter
                # separately and breach `budget.total`.
                if "total" in left:
                    left["total"] -= need
            accepted.append(replace(action, n=n, params=params))
        return accepted, rejected

    @staticmethod
    def _fit_to_plan(spec: tools_mod.ToolSpec, n: int, left: dict[str, int]) -> int:
        """Largest `n` <= requested that fits the PLAN's remainder.

        Computed from the same cards the policy reads (`ToolInfo.forecast`):
        one arithmetic on both sides of the seam, otherwise "how much I may
        ask" and "how much I will get" would silently diverge.
        """
        card = tools_mod.card(spec)
        for size in range(int(n), 0, -1):
            forecast = card.forecast(size)
            if not all(left.get(kind, need) >= need for kind, need in forecast.items()):
                continue
            if left.get("total", sum(forecast.values())) < sum(forecast.values()):
                continue
            return size
        return 0

    def _why_illegal(self, action: Action, state: SearchState) -> tuple[str, str]:
        """Why the action is not in the legal list, and whose outcome that is.

        The difference is not cosmetic. "The policy asks for the impossible"
        marks a broken policy, and the share of such parts is kept observable.
        "The tool ran out under a ceiling" is a normal end of work, and
        recording it as a fault would pollute the one cheap fault indicator
        with noise from healthy policies.
        """
        try:
            spec = tools_mod.get(action.tool)
        except KeyError:
            return "tool not in the registry", ""
        if spec not in self.specs:
            return "tool disabled in the run config", ""

        for kind in spec.spends:
            if _zero(state.remaining.calls.get(kind)):
                return (
                    f"call cap exhausted for {kind}",
                    STOP_LIMIT_EXEC if kind == "exec" else STOP_LIMIT_CALLS,
                )
        parent = state.get(action.parent_id)
        if parent is None:
            return "parent not in the pool", ""
        if spec.grows_depth and self.max_depth is not None and parent.depth >= self.max_depth:
            return "parent at maximum depth", STOP_LIMIT_DEPTH
        left = self._pool_left(spec, parent.id, state)
        if spec.n_semantics == "rank_depth" and left is not None and left <= 0:
            return f"tool pool {spec.name} exhausted", STOP_NO_LEGAL
        if spec.too_many_numbers(parent):
            return f"parent code has at least {spec.max_code_numbers} numbers", ""
        if not spec.accepts(parent):
            return "parent does not meet the tool registry requirements", ""
        if spec.deterministic and state.attempts_on(parent.id, spec.name):
            return "deterministic tool already tried on this parent", ""
        return "action is not in the legal list", ""

    # --- running tools -----------------------------------------------------

    def _run_actions(self, actions: Iterable[Action], state: SearchState) -> None:
        needs = self._needs(state)
        # This iteration's offspring are collected anew: the policy reads them
        # as "what just appeared", and a leftover from the previous iteration
        # would count a candidate as fresh twice.
        state.fresh = self.fresh = []
        for action in actions:
            parent = state.get(action.parent_id)
            if parent is None:
                continue
            spec = tools_mod.get(action.tool)
            attempt = len(state.attempts_on(parent.id, spec.name))
            tag = f"{parent.id}_{spec.name}_a{attempt}"
            depth = parent.depth + 1 if spec.grows_depth else parent.depth
            params = dict(action.params or {})
            params.setdefault("tag", tag)
            origin = Origin(
                tool=spec.name,
                params={k: v for k, v in params.items() if k != "prompt"},
                # The seed is computed by the same expression as inside
                # `propose_steps`: one formula for two places, otherwise the
                # journal and the draw diverge silently. Deterministic tools
                # have no seed - the pool cursor tells them apart.
                seed=None if spec.n_semantics == "rank_depth" else derive_seed(
                    self.run_seed,
                    self.obs.figure_id,
                    step=depth,
                    tag=attempt_tag(tag, attempt),
                    variant=int(params.get("variant", 0)),
                ),
                attempt=attempt,
            )

            ctx = tools_mod.ToolContext(obs=self.obs, res=self.res, journal=self.journal)
            try:
                outcome = spec.run(ctx, parent, params, action.n, origin)
            except (DeadlineExceeded, BudgetExceeded):
                raise
            except Exception as exc:
                # A tool crash is a failed action, not a failed part: what was
                # found lives on, other actions of the iteration proceed.
                logger.warning("Tool %s crashed on part %s: %s",
                               spec.name, self.obs.figure_id, exc, exc_info=True)
                outcome = tools_mod.ToolOutcome(products=[], failure="exec_error",
                                                info={"error": repr(exc)[:200]})

            self._event("tool_call", tool=spec.name, parent=parent.id, n=action.n,
                        attempt=attempt, reason=action.reason or None, **outcome.info)

            born = len(self.fresh)
            children = self._materialize(parent, spec, origin, outcome, needs, tag, depth)
            new_ids = {candidate.id for candidate in self.fresh[born:]}
            self.attempts.setdefault(parent.id, []).append(
                Attempt(
                    origin=origin,
                    produced=len(outcome.products),
                    built=sum(1 for child in children if child.built),
                    failure=outcome.failure or _first_failure(children),
                    reused=[child.id for child in children if child.id not in new_ids],
                    iteration=state.iteration,
                )
            )
            self._refresh(state)

    def _materialize(
        self,
        parent: Candidate,
        spec: tools_mod.ToolSpec,
        origin: Origin,
        outcome: tools_mod.ToolOutcome,
        needs: Sequence[str],
        tag: str,
        depth: int,
    ) -> list[Candidate]:
        """Execute the tool products, measure them and put them in the pool."""
        if not outcome.products:
            return []

        fresh: list[tools_mod.Product] = []
        children: list[Candidate] = []
        for product in outcome.products:
            if product.code is None:
                # No registered tool returns a mesh without code yet. When one
                # appears it needs a measurement path without execution, and
                # silently dropping such a product would be the worst option:
                # the loop would pretend the tool returned nothing.
                raise NotImplementedError(
                    f"Tool {spec.name} returned a product without code; "
                    "the measurement path for a ready-made mesh is not implemented yet"
                )
            known = self.by_code.get(product.code)
            if known is not None:
                # The same code was already executed on this part. A second
                # time it would cost an execution and metrics - a quarter of the
                # run's wall time lives there - and return the same thing.
                self._event("candidate_dedupe", tool=spec.name, parent=parent.id,
                            reused=known.id)
                children.append(known)
                continue
            fresh.append(product)

        if not fresh:
            return children

        evaluations = self.res.evaluate_codes(
            codes=[product.code for product in fresh],
            gt_mesh_path=self.obs.gt_mesh_path,
            name_prefix=f"step{depth:03d}_{tag}",
            step=depth,
            needs=tuple(needs),
        )
        codes = self._try_fallbacks(spec, fresh, evaluations, needs, tag, depth)
        for product, code, evaluation in zip(fresh, codes, evaluations):
            candidate = Candidate(
                id=self._new_id(),
                parent_id=parent.id,
                depth=depth,
                origin=origin,
                code=code,
                mesh_path=evaluation.mesh_path if evaluation.success else None,
                metrics=evaluation.metrics,
                built=bool(evaluation.success),
                failure=_classify(evaluation),
                extendable=spec.extendable and not spec.terminal,
                error=(evaluation.error or None),
            )
            self.pool.append(candidate)
            self.by_code[product.code] = candidate
            self.by_code[code] = candidate
            children.append(candidate)
            self.fresh.append(candidate)
            # Best is updated on EVERY new candidate and BEFORE `select` prunes
            # anything: the policy must not be able to lose a found result by
            # dropping a branch.
            self._update_best(candidate)
        return children

    def _try_fallbacks(
        self,
        spec: tools_mod.ToolSpec,
        products: list[tools_mod.Product],
        evaluations: list[Any],
        needs: Sequence[str],
        tag: str,
        depth: int,
    ) -> list[str]:
        """Execute fallback code where the primary code came out non-watertight.

        Returns the code of each product that was finally taken. ``evaluations``
        is modified in place: an accepted fallback replaces the primary's
        execution. Only a watertight fallback is accepted, and a watertight
        primary never gets here, so the fallback cannot break anything.
        """
        codes = [str(product.code) for product in products]
        tried = [
            i for i, (product, evaluation) in enumerate(zip(products, evaluations))
            if product.fallback_code is not None and _classify(evaluation) == "not_watertight"
        ]
        if not tried:
            return codes
        retries = self.res.evaluate_codes(
            codes=[products[i].fallback_code for i in tried],
            gt_mesh_path=self.obs.gt_mesh_path,
            name_prefix=f"step{depth:03d}_{tag}_fallback",
            step=depth,
            needs=tuple(needs),
        )
        accepted = 0
        for i, retry in zip(tried, retries):
            if retry.success and (retry.metrics or {}).get(objective_mod.FIELD_PRED_WATERTIGHT) is True:
                codes[i], evaluations[i] = products[i].fallback_code, retry
                accepted += 1
        self._event("fallback_code", tool=spec.name, tried=len(tried), accepted=accepted)
        return codes

    def _update_best(self, candidate: Candidate) -> None:
        rank = _harness_rank(candidate.metrics) if candidate.built else None
        if rank is None and candidate.built and candidate.failure is not None:
            # A non-watertight candidate has no metrics at all
            # (`metrics.measure_pair`), but it must be kept as best while there
            # are no valid ones, for the diagnosis below: lowest rank, first
            # one wins.
            rank = (0, float("-inf"))
        if rank is None:
            return
        # The MAJOR key is model validity, then scale and value. A
        # non-watertight prediction is an invalid model, not "a candidate with
        # a different number": it must not be returned as the part's answer
        # when a valid one exists.
        #
        # Why it matters: such a candidate used to compete with valid ones in
        # the FIRST tier. `gms_norm` is ordered for every candidate (`_needs`)
        # and computed on the surface, so also for an open mesh, while `iou` is
        # None. Then `_fitness` equals GMS alone - an ordinary contract-scale
        # number. A candidate with a good GMS beat the valid ones and the part
        # was rejected (`ir_notwt`) although a valid candidate was in the
        # pool. The CD fallback is irrelevant here - it is only the third
        # line, when neither half of the contract scale was computed.
        #
        # The rule is "below any valid one", not "forbidden": if no valid
        # candidate exists, the best invalid one is still returned. The part
        # is a failure either way, but the diagnosis stays exact -
        # `ir_notwt` is not replaced by `ir_no_result`.
        ranked = (0 if candidate.failure is not None else 1, *rank)
        if self.best_rank is None or ranked > self.best_rank:
            self.best, self.best_rank = candidate, ranked
            self.best_fitness = _fitness(candidate.metrics)

    def _needs(self, state: SearchState | None = None) -> tuple[str, ...]:
        """Which metrics to order from the executor. NOT a policy decision.

        Both halves of the contract scale are always computed: IoU (where it is
        possible at all: needs a watertight GT and the boolean engine) and
        GMS. The policy used to order the set via `metric_needs`, which was
        wrong on two counts:

        - **fitness stopped being comparable across policies.** A policy that
          ordered only IoU got fitness equal to IoU by construction, one that
          ordered both got the mean of the two, and the gate would compare such
          numbers as homogeneous;
        - **saving on measurement is a knob the policy pays for with someone
          else's currency.** A policy could make itself cheaper by blinding the
          report.

        The cost is stated openly: GMS takes a few percent of wall time and
        every run now pays it equally - which is the point: branches are
        compared, not metric sets.
        """
        return (objective_mod.NEED_IOU, objective_mod.NEED_GMS)

    # --- policy call -------------------------------------------------------

    def _genome(self, call: Any, state: SearchState) -> Any:
        """Call a policy block and record its own time.

        The measurement matters: blocks may write their own scoring functions
        over code, mesh and images. In the cost model `cost_i = sum W_type * n_type`
        such functions are free yet eat wall time, so the cost axis would not
        see them. `genome_cpu` is the only place where they become visible.
        """
        started = time.monotonic()
        try:
            return call(state)
        finally:
            self.genome_cpu_sec += time.monotonic() - started

    # --- journal -----------------------------------------------------------

    def _event(self, kind: str, **fields: Any) -> None:
        if self.journal is not None:
            self.journal.event(kind, **fields)

    def _done_reason(self, state: SearchState) -> str:
        """Which outcome to record for "the policy finished": quality or cost.

        `done` is the policy's judgement of QUALITY. But a planning policy also
        says `done` when it has no choice left: `affordable_n` returned zero
        everywhere, i.e. the part hit COST rather than being declared ready.
        They are told apart by the actual remainder, not by the reason text:
        the wording is written by the policy.

        One function for both doors (`Verdict.done` and `Plan.done`) on
        purpose: if they diverged they would give the same outcome under two
        names, and the share of `done:exhausted` would depend on the moment the
        policy said "ready" rather than on whether its budget ran out.
        """
        return STOP_DONE_EXHAUSTED if self._stopped_on_a_ceiling(state) else STOP_DONE

    @staticmethod
    def _done_by(plan: Any, *, verdict_done: bool) -> str:
        """Which door the policy used to finish the part.

        `Plan.done` outranks `Verdict.done`: `plan()` decides earlier and with
        the fresh report of the move, while `select()` of the same move judges
        afterwards. If both said it, the finish belongs to the plan. `stalled`
        is the policy's word, the `select` door is a harness fact.

        Orthogonal to `_done_reason`: that one says whether the budget sufficed,
        this one which decision the policy took. `done:exhausted` with `stall`
        is legal.
        """
        if getattr(plan, "done", False):
            return DONE_BY_STALL if getattr(plan, "stalled", False) else DONE_BY_QUALITY
        return DONE_BY_SELECT if verdict_done else DONE_BY_QUALITY

    def _stopped_on_a_ceiling(self, state: SearchState) -> bool:
        """Whether the policy's stop coincided with an exhausted ceiling it SPENT.

        The question: attribute this part to cost or to quality. The answer
        rests only on harness facts - what was spent and what is left - because
        the reason text is written by the policy and may be anything.

        Three signs, all about a ceiling the policy actually used:

        - **the wall ran out.** It is shared and always consumed, so it is
          checked unconditionally;
        - **a channel is closed and the policy used it**: `spent > 0` with a
          zero remainder. The `spent` condition is load-bearing. Without it a
          config with `opt: 0` - which is legal, it is a way to switch a tool
          off - would declare EVERY stop exhaustion, including an honest "good
          enough";
        - **the remainder is not zero yet but buys nothing**: no currently
          legal action is affordable (`affordable_n(tool, 1)`).

        The third sign is load-bearing. A planning policy ASKS the price before
        spending (`affordable_n`) and therefore stops one step BEFORE the
        ceiling - the wall never reaches zero with it. By the first two signs
        such a part would get `done`, "the policy decided it is ready", while
        it wrote exactly the opposite: there was nothing to continue with. The
        policy's politeness would be what distinguished outcomes: the same
        cost limit would read as quality for a careful policy and as cost for
        one that runs until refusal.

        The three conditions are joined with "or". An earlier version used only
        the third sign INSTEAD of the second and never fired: a policy
        exhausted `vlm` while `det` and `repair` stayed free, so "something is
        still affordable" held with a closed channel. The second sign covers
        that case.

        What the function does NOT claim: that the policy wanted to continue.
        It may have stopped on quality exactly when the remainder stopped
        buying anything, and such a part will be attributed to cost. The
        reverse error is costlier: a policy hitting a ceiling would look as if
        it decided the part was ready, and cost would read as quality across
        the whole line.
        """
        wall = state.remaining.wall_sec
        if wall is not None and wall <= 0:
            return True
        if any(
            left is not None and int(left) <= 0 and float(state.spent.get(kind, 0)) > 0
            for kind, left in state.remaining.calls.items()
        ):
            return True
        # The set comes from `state.legal` - the actions available to the
        # harness, not the subset a particular policy uses: the harness does not
        # know that and must not. The error is in the conservative direction:
        # while any legal tool is affordable, the stop stays `done`.
        # `done`.
        return bool(state.legal) and all(
            state.affordable_n(item.tool, 1) == 0 for item in state.legal
        )

    def _log_legal(self, state: SearchState) -> None:
        self._event(
            "legal_actions",
            iteration=state.iteration,
            pool=len(self.pool),
            n=len(state.legal),
            # Not the whole list: on a wide pool it is long and repetitive.
            # A per-tool summary answers the question usually asked of the
            # journal - what was available at this iteration at all.
            by_tool={
                tool: sum(1 for item in state.legal if item.tool == tool)
                for tool in sorted({item.tool for item in state.legal})
            },
        )

    def _accept_keep(self, keep: Iterable[str], state: SearchState) -> set[str]:
        """Which of the policy's requested candidates may be kept in the working set.

        Two kinds are dropped: a nonexistent candidate and an INVALID model.
        The latter is the same rule as `ToolSpec.accepts`: a non-watertight
        prediction is an error, not a candidate with a bad number, and cannot
        be chosen for continuation. Keeping the ban only in `accepts` is not
        enough: the invalid candidate still landed in `active`, the front hit
        it, and the part stalled with `plan_empty` instead of continuing from
        a valid ancestor - the ban would read as "the harness gave up".

        The rejection is recorded as an event, not silent: a policy trying to
        select an invalid model must be visible in the journal. Repair is
        unaffected - it finds its target by `failure` among the fresh
        candidates, not in `active`.
        """
        accepted: set[str] = set()
        for candidate_id in keep:
            candidate = state.get(candidate_id)
            if candidate is None:
                continue
            if candidate.failure is not None:
                self._event(
                    "keep_rejected", iteration=state.iteration, candidate=candidate_id,
                    reason=f"invalid model ({candidate.failure}): cannot be chosen as a continuation",
                )
                continue
            accepted.add(candidate_id)
        return accepted

    def _journal_row(
        self,
        state: SearchState,
        actions: Sequence[Action],
        *,
        done: bool,
        reason: str | None,
        done_by: str | None = None,
    ) -> dict[str, Any]:
        """Journal row of an iteration. One assembly for all loop exits.

        A separate method rather than an inline literal because there are two
        exits: the ordinary one (after `select`) and the early one - `Plan.done`
        with an empty plan. While the literal sat only on the first, the second
        silently wrote nothing and the journal could not reconstruct how most
        finished parts ended.

        `reason` is the rationale, not just the outcome: the run directory must
        reconstruct why the harness decided, not only what. For an agent policy
        it says whether the metric or the assistant decided - otherwise "the
        assistant chose" is indistinguishable from "the metric was confident".
        """
        return {
            "iteration": state.iteration,
            "actions": [
                {"tool": a.tool, "parent": a.parent_id, "n": a.n, "reason": a.reason}
                for a in actions
            ],
            "pool": len(self.pool),
            "active": sorted(state.active),
            "best": None if self.best is None else self.best.id,
            "best_fitness": self.best_fitness,
            "done": done,
            "done_by": done_by if done else None,
            "select_reason": reason or None,
        }

    def _log_rejected(self, rejected: list[dict[str, Any]], state: SearchState) -> None:
        for row in rejected:
            self._event("plan_rejected", iteration=state.iteration, **row)

    def _stop_from_budget(self, exc: BudgetExceeded) -> tuple[str, str]:
        if isinstance(exc, DeadlineExceeded):
            return STOP_WALL, str(exc)
        text = str(exc)
        return (STOP_LIMIT_EXEC if "exec" in text else STOP_LIMIT_CALLS), text

    # --- result ------------------------------------------------------------

    def _finish(
        self, result: Reconstruction, state: SearchState, stop_reason: str, stop_detail: str,
        done_by: str | None = None,
    ) -> Reconstruction:
        self._refresh(state)
        self._event(
            "search_stop",
            reason=stop_reason,
            detail=stop_detail or None,
            done_by=done_by,
            iterations=state.iteration,
            pool=len(self.pool),
            best=None if self.best is None else self.best.id,
            best_fitness=self.best_fitness,
            genome_cpu_sec=round(self.genome_cpu_sec, 3),
        )
        # The wall-ceiling string is the same as in earlier harnesses: the
        # report uses it to tell "the part took too long" from a failure, and
        # diverging spellings would make that breakdown impossible.
        result.stop_reason = WALL_STOP_REASON if stop_reason == STOP_WALL else stop_reason
        result.done_by = done_by
        result.journal = self.journal_rows
        if self.best is not None:
            result.code = self.best.code
            result.mesh_path = self.best.mesh_path
            result.metrics = dict(self.best.metrics or {})
            result.metrics["fitness"] = self.best_fitness
            # The scale on which the best was chosen. Without this field a
            # fallback would read as "fitness was not computed", which is
            # different: in that case there was nothing to choose from.
            result.metrics["fitness_scale"] = (
                "contract" if self.best_rank and self.best_rank[1] == 1 else FALLBACK_FIELD
            )
            result.metrics["depth"] = self.best.depth
            result.n_steps = self.best.depth
        elif result.error is None:
            result.error = "no valid candidate"
        return result


def _zero(value: int | None) -> bool:
    return value is not None and value <= 0


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_failure(children: list[Candidate]):
    for child in children:
        if child.failure is not None:
            return child.failure
    return None


def _classify(evaluation: Any):
    """Why a candidate failed - a typed reason, not a string.

    Same breakdown as the contract IR: an execution failure and a non-watertight
    prediction are different events, and one field for both has already given
    a plausible report with the wrong diagnosis.
    """
    if not evaluation.success:
        return "tool_timeout" if getattr(evaluation, "timed_out", False) else "exec_error"
    metrics = evaluation.metrics or {}
    if metrics.get(objective_mod.FIELD_PRED_WATERTIGHT) is False:
        return "not_watertight"
    if metrics.get(objective_mod.FIELD_IOU_UNAVAILABLE):
        # A candidate without IoU where IoU is computed is most likely an
        # invalid mesh. The failure ranks it below any valid one in
        # `_update_best`, like an open one, instead of competing on GMS alone
        # in the IoU+GMS scale.
        return "iou_unavailable"
    if _harness_rank(metrics) is None:
        # Neither the contract scale nor the fallback gave a value: the
        # candidate built but there is nothing to compare it by. This is not
        # an execution failure and they must not be confused - the IR breakdown
        # requires telling "the code crashed" from "the code built but there is
        # no value".
        return "objective_unavailable"
    return None
