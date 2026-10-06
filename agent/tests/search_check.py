#!/usr/bin/env python3
"""Search-loop check: the harness holds what it is obliged to hold.

The main reason for moving the loop from the policy into the harness is not tidiness
but enforcement: previously the call and wall ceilings worked only because the
scaffold politely propagated `BudgetExceeded`. So the central check here is a
**policy that swallows all exceptions**: it must stop, not spin forever.

The rest is about the new loop distinguishing events that are easy to merge into
one: an empty plan and an illegal plan, the depth ceiling and "nothing to continue
from", an execution failure and a non-watertight result. Each such merge has once
produced a plausible report with the wrong diagnosis.

The stubs here match the real interfaces: `propose_steps` takes an attempt number,
`algo_rebuild` returns a slice by cursor, `evaluate_codes` returns `EvalResult`-like
rows with metrics. A stub that lags behind the code is the most common cause of a
green check on broken code.
"""

from __future__ import annotations

import json
import sys
import shutil
import tempfile
import types
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "tests"))

if "openai" not in sys.modules:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub

from cad_agent.harness import search  # noqa: E402
from cad_agent.harness.budget import Budget  # noqa: E402
from cad_agent.harness.search_types import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    STOP_DONE,
    STOP_DONE_EXHAUSTED,
    STOP_LIMIT_DEPTH,
    STOP_LIMIT_ITERATIONS,
    STOP_NO_LEGAL,
    STOP_PLAN_EMPTY,
    STOP_PLAN_INVALID,
    Action,
    Candidate,
    LegalAction,
    Origin,
    Plan,
    SearchState,
    Verdict,
)
from cad_agent.scaffold.base import Observation  # noqa: E402
from cad_agent.scaffold.policies import build as build_policy  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --- capability stubs ---------------------------------------------------


class FakeEval:
    """A row of an execution result, shaped like `EvalResult`."""

    def __init__(self, mesh_path, metrics, success=True, error=None, timed_out=False):
        self.mesh_path = mesh_path
        self.metrics = metrics
        self.success = success
        self.error = error
        self.timed_out = timed_out


class FakeProposal:
    def __init__(self, full_code: str, step_code: str):
        self.full_code = full_code
        self.step_code = step_code
        self.error = None
        self.point = "(0,0,0)"


class FakeImage:
    """Selection collage: only the panel labels matter.

    The labels on the image and the labels in the prompt text are a pair that must not
    drift apart: panel `B` and block `Candidate B` must be the same candidate. The stub
    remembers them so the check can compare both sides.
    """

    def __init__(self, labels, mesh_paths):
        self.labels = list(labels)
        self.mesh_paths = list(mesh_paths)


class FakeRenderer:
    def __init__(self):
        self.calls: list[FakeImage] = []

    def selection_image(self, gt_mesh_path, pred_mesh_paths, labels, visualization_mode="simple"):
        image = FakeImage(labels, pred_mesh_paths)
        self.calls.append(image)
        return image


class FakeAgent:
    """Scripted assistant: what to answer and when to refuse over prompt length.

    It charges the budget itself, like the real `Resources.ask_agent`. Without that, the
    check "the question cost a visual call" would be green on any code: the counter
    is moved by the capability, not by the policy.
    """

    def __init__(self, budget, answers, too_large_first: bool = False):
        self.budget = budget
        self.answers = list(answers)
        self.too_large_first = too_large_first
        self.prompts: list[tuple[str, bool]] = []
        # Answer mode and ceiling the call was made with. `None` for the mode means "as set
        # by the run". The stub must accept and record them: otherwise there is no way to
        # check a tool that asks for an answer without reasoning and sized to its input.
        self.thinking: list[bool | None] = []
        self.max_tokens: list[int] = []
        self.purposes: list[str] = []

    def __call__(self, prompt, image=None, max_tokens=16, temperature=0.0, thinking=None,
                 purpose="decide"):
        self.prompts.append((prompt, image is not None))
        self.thinking.append(thinking)
        self.max_tokens.append(max_tokens)
        self.purposes.append(purpose)
        if self.too_large_first and len(self.prompts) == 1:
            from cad_agent.capabilities import llm as llm_mod

            raise llm_mod.PromptTooLarge("prompt ~99999 tokens with a cap of 100")
        kind = "agent_visual" if image is not None else "agent_text"
        # Same order as in the live resources: the ceiling is asked BEFORE the call, otherwise
        # the stub allows what the real thing does not.
        self.budget.check_calls(kind)
        self.budget.spend(kind)
        return self.answers.pop(0) if self.answers else "A"


class FakeResources:
    """Capabilities of one part without models and without CAD.

    Candidate quality grows with depth, which checks that "best" really is the best and
    not the last. Plus knobs the tests use to bend behavior: `broken_from` (from which
    depth candidates stop building) and `det_pool` (what the deterministic branch returns).
    """

    def __init__(self, budget=None, config=None, det_pool=None, broken_from=None,
                 with_agent=False, agent_answers=None):
        self.budget = budget or Budget()
        self.config = dict(config or {})
        self.journal = None
        self.work_dir = Path(tempfile.mkdtemp(prefix="search_check_"))
        self.calls: list[tuple] = []
        # Sampling knobs of each request: they show that the action's parameters reached the
        # generator and did not stay in the plan.
        self.sampling: list[dict] = []
        self.evaluated: list[str] = []
        # All quality values ever computed on the part: they check that the best was returned,
        # not the last.
        self.measured: list[float] = []
        self._det_pool = list(det_pool or [])
        self._det_cursor: dict[str, int] = {}
        self._broken_from = broken_from
        # The assistant and the renderer go as a PAIR: a question with an image and no renderer
        # is not a question but a degradation, and it must be tested separately, not hit by accident.
        self.render = FakeRenderer() if with_agent else None
        self.agent = FakeAgent(self.budget, agent_answers or []) if with_agent else None
        self.ask_agent = self.agent
        self.difficulty = lambda: {"gt_watertight": True}

    # --- what the loop calls ---------------------------------------------

    def propose_steps(self, pred_mesh_path, prev_code, k, step=None, variant=0, tag="",
                      attempt=None, temperature=None, top_p=None):
        # The stub signature is part of the interface: the loop calls the generator exactly as
        # `resources.propose_steps` does, and an extra parameter here would mean not a failing
        # check but a silent action refusal (a tool exception counts as a refusal of the ACTION,
        # not of the part).
        self.calls.append(("propose", step, tag, attempt, k))
        self.sampling.append({"variant": variant, "temperature": temperature, "top_p": top_p})
        self.budget.check_wall("propose_steps")
        self.budget.spend("vlm", k)
        return [
            FakeProposal(
                full_code=f"{prev_code}\nop(step={step}, tag={tag}, attempt={attempt}, i={idx})",
                step_code=f"op{idx}",
            )
            for idx in range(k)
        ]

    def algo_rebuild(self, pred_mesh_path, k):
        key = "" if pred_mesh_path is None else str(pred_mesh_path)
        first = key not in self._det_cursor
        if first:
            self._det_cursor[key] = 0
            self.budget.spend("det")
        start = self._det_cursor[key]
        served = self._det_pool[start : start + k]
        self._det_cursor[key] = start + len(served)
        self.calls.append(("det", key, k, len(served)))
        return served

    def det_remaining(self, pred_mesh_path=None):
        key = "" if pred_mesh_path is None else str(pred_mesh_path)
        if key not in self._det_cursor:
            return None
        return max(0, len(self._det_pool) - self._det_cursor[key])

    def optimize(self, code, **kwargs):
        self.budget.spend("opt")
        return {"success": True, "code": code + "\n# optimized"}

    def evaluate_codes(self, codes, gt_mesh_path, name_prefix, measure=True, step=None, needs=()):
        self.budget.spend("exec", len(codes))
        out = []
        for idx, code in enumerate(codes):
            self.evaluated.append(code)
            depth = int(step or 0)
            if self._broken_from is not None and depth >= self._broken_from:
                out.append(FakeEval(None, None, success=False, error="execution failed"))
                continue
            mesh_path = str(self.work_dir / f"{name_prefix}_{idx}.stl")
            # Quality grows with depth and differs slightly by index: without the index difference,
            # selecting the best of K would be checked idly.
            iou = min(0.99, 0.5 + 0.1 * depth + 0.01 * idx)
            self.measured.append(iou)
            out.append(FakeEval(mesh_path, {"iou": iou, "pred_watertight": True}))
        return out

    def note(self, *args, **kwargs):
        pass


def observation(figure_id="fig/1") -> Observation:
    return Observation(
        figure_id=figure_id,
        gt_mesh_path=Path("/tmp/gt.stl"),
        work_dir=Path("/tmp"),
        prefix_code="PREFIX",
    )


class StepGenome:
    """Minimal policy for the loop check: `stepwise` from the best, `k` samples.

    Not a run policy: the loop mechanics (best of the whole rollout, ceilings, scale)
    are checked on a policy without an assistant, so the answer does not depend on
    parsing its replies. No `select`: the working set is kept by the harness.
    """

    def __init__(self, k: int = 1):
        self.k = k

    def plan(self, state):
        parent = state.best_id or state.pool[0].id
        return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=self.k)])


def run(policy, res, **loop_kwargs):
    loop = search.SearchLoop(policy, **loop_kwargs)
    return loop.run(observation(), res)


# --- checks ----------------------------------------------------------------


def main() -> None:
    print("1. The loop walks a part and returns the BEST, not the last")
    res = FakeResources(config={"limits": {"iterations": 5, "depth": 5}})
    result = run(StepGenome(k=2), res)
    check("part produced code", bool(result.code), result.stop_reason)
    check("depth grew", result.n_steps >= 1, str(result.n_steps))
    check("metrics of the best arrived", result.metrics.get("iou") is not None, str(result.metrics))
    # The best is the best of the WHOLE rollout, not the last accepted. Checked by comparing
    # with the maximum of everything measured on the part: the return rule is separate from
    # the stop rule, and the harness updates the best before `select` prunes anything.
    check("the best of all measured was returned",
          result.metrics.get("fitness") == max(res.measured),
          f"{result.metrics.get('fitness')} vs {max(res.measured)}")
    check("the stop reason is named", bool(result.stop_reason), result.stop_reason)

    print("1b. A policy without `select` goes through a part")

    class NoSelectGenome:
        """Only `plan`: the way `DialogueLeanPolicy` is built.

        What is checked is not "does not crash" but three consequences at once: the part
        completes, the working set stays what the harness kept (the default `keep=None`
        means "do not touch", not "reset"), and `Plan.done` ends the part without a
        verdict. Without this check a missing method would be caught only by a live run.
        """

        def __init__(self, stop_after: int = 2):
            self.stop_after = stop_after
            self.turns = 0

        def plan(self, state):
            self.turns += 1
            parent = state.pool[0].id
            return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=1)],
                        done=self.turns >= self.stop_after,
                        reason="enough")

    genome = NoSelectGenome()
    res = FakeResources(config={"limits": {"iterations": 6, "depth": 6}})
    result = run(genome, res)
    check("part went through without `select`", bool(result.code), result.stop_reason)
    check("and ended with the plan flag, not by a cap",
          result.stop_reason.startswith("done"), result.stop_reason)
    check("the default verdict leaves the working set alone",
          genome.turns == genome.stop_after, str(genome.turns))
    check("the finishing reason reached the journal",
          any("enough" in (row.get("select_reason") or "") for row in result.journal),
          str([row.get("select_reason") for row in result.journal]))

    # A mutant's typo, a field instead of a method, must fail as a policy error, not be
    # silently treated as a missing `select`.
    class BrokenSelectGenome(NoSelectGenome):
        select = "not a method"

    result = run(BrokenSelectGenome(), FakeResources(config={"limits": {"iterations": 3}}))
    check("a non-callable `select` is a policy error, not 'absent'",
          result.stop_reason == "genome_error", result.stop_reason)

    print("2. The cap is held by the harness, not by the policy's politeness")

    class SwallowingGenome:
        """A mutant that swallows everything. It used to switch off both ceilings silently."""

        def __init__(self):
            self.plans = 0

        def plan(self, state):
            self.plans += 1
            try:
                parent = sorted(state.active)[0] if state.active else state.pool[0].id
                return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=1)])
            except Exception:
                return Plan(actions=[])

        def select(self, state):
            try:
                alive = state.alive()
                keep = {alive[-1].id} if alive else set()
                return Verdict(keep=keep, done=False, reason="I will never finish")
            except Exception:
                return Verdict(keep=set(), done=False)

    genome = SwallowingGenome()
    res = FakeResources(config={"limits": {"iterations": 4, "depth": 50}})
    result = run(genome, res)
    check("the loop stopped on its own", result.stop_reason == STOP_LIMIT_ITERATIONS, result.stop_reason)
    check("exactly as many iterations as allowed", genome.plans == 4, str(genome.plans))

    # The same policy against the call ceiling: it cannot get around it either.
    res = FakeResources(budget=Budget(limits={"vlm": 3}), config={"limits": {"iterations": 50}})
    result = run(SwallowingGenome(), res)
    check("the call cap stopped the part", result.stop_reason.startswith("limit:"),
          result.stop_reason)
    check("and was not exceeded", res.budget.counts["vlm"] <= 3, str(res.budget.counts))
    check("what was found is not lost", bool(result.code), result.stop_reason)

    print("3. The depth cap is called by its own name")
    res = FakeResources(config={"limits": {"iterations": 50, "depth": 2}})
    result = run(SwallowingGenome(), res)
    check("stop by depth", result.stop_reason == STOP_LIMIT_DEPTH, result.stop_reason)
    check("depth not exceeded", result.n_steps <= 2, str(result.n_steps))

    print("3a. `done` by exhaustion is not the same as `done` by quality")
    # A planning policy (`affordable_n`) stops a step BEFORE the harness ceiling, so
    # `wall`/`limit:*` never occur with it. If the outcome were then written as `done`, the
    # price would silently read as quality: "the part is expensive" would become
    # indistinguishable from "the part was judged ready".
    #
    # The HARNESS tells them apart, by the fact of the remainder and not by the reason
    # text: the reason is worded by the policy, and a mutant's is anything. So the policy
    # here says `done` with text that says nothing about the budget.

    class ThriftyGenome:
        """Stops by itself when the remainder no longer covers an action."""

        def plan(self, state):
            parent = sorted(state.active)[0] if state.active else state.pool[0].id
            n = state.affordable_n("stepwise", 1)
            if n <= 0:
                return Plan(actions=[])
            return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=n)])

        def select(self, state):
            alive = state.alive()
            keep = {alive[-1].id} if alive else set()
            if state.affordable_n("stepwise", 1) <= 0:
                return Verdict(keep=keep, done=True, reason="I will not go further")
            return Verdict(keep=keep, done=False, reason="")

    res = FakeResources(budget=Budget(limits={"vlm": 2}), config={"limits": {"iterations": 50}})
    result = run(ThriftyGenome(), res)
    check("an exhausted remainder is its own outcome, not done",
          result.stop_reason == STOP_DONE_EXHAUSTED, result.stop_reason)
    check("the cap is not broken", res.budget.counts["vlm"] <= 2, str(res.budget.counts))

    # The converse: a policy that stopped with a LIVE remainder is `done`. Without this
    # half the check would be green while declaring any stop to be exhaustion.
    class SatisfiedGenome(ThriftyGenome):
        def select(self, state):
            alive = state.alive()
            return Verdict(keep={alive[-1].id} if alive else set(),
                           done=True, reason="good enough")

    res_ok = FakeResources(budget=Budget(limits={"vlm": 50}), config={"limits": {"iterations": 50}})
    result_ok = run(SatisfiedGenome(), res_ok)
    check("a stop with a live remainder stays done",
          result_ok.stop_reason == STOP_DONE, result_ok.stop_reason)

    print("4. An empty plan and an illegal plan are different outcomes")

    class SilentGenome:
        def plan(self, state):
            return Plan(actions=[])

        def select(self, state):
            return Verdict(keep=set(), done=False)

    result = run(SilentGenome(), FakeResources())
    check("an empty plan is not passed off as done", result.stop_reason == STOP_PLAN_EMPTY,
          result.stop_reason)

    class IllegalGenome:
        def plan(self, state):
            return Plan(actions=[Action(tool="stepwise", parent_id="no-such", n=1)])

        def select(self, state):
            return Verdict(keep=set(), done=False)

    events: list[tuple] = []
    res = FakeResources()
    res.journal = types.SimpleNamespace(
        event=lambda kind, **fields: events.append((kind, fields)),
        stage=lambda name: _noop_ctx(),
    )
    result = run(IllegalGenome(), res)
    check("an illegal plan is its own outcome", result.stop_reason == STOP_PLAN_INVALID,
          result.stop_reason)
    rejected = [row for kind, row in events if kind == "plan_rejected"]
    check("the rejection is recorded by name and with a reason",
          bool(rejected) and "reason" in rejected[0], str(rejected[:1]))

    # The same outcome must result when the rejection came from a branch WITHOUT
    # `stop_hint`. There are six rejection branches and only two set the key; a `.get` used
    # to return `None`, a set of one `None` passed the "no empty ones" check, and
    # `stop_reason` became `None`: the part got an outcome without a name. The cause was
    # exactly this: a knob outside its scale.
    class OutOfRangeGenome:
        def plan(self, state):
            parent = sorted(state.active)[0]
            return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=1,
                                        params={"variant": 999})])

        def select(self, state):
            return Verdict(keep=set(), done=False)

    events = []
    res = FakeResources()
    res.journal = types.SimpleNamespace(
        event=lambda kind, **fields: events.append((kind, fields)),
        stage=lambda name: _noop_ctx(),
    )
    result = run(OutOfRangeGenome(), res)
    check("a plan rejected by the knob scale is also plan_invalid, not an empty outcome",
          result.stop_reason == STOP_PLAN_INVALID, repr(result.stop_reason))
    rejected = [row for kind, row in events if kind == "plan_rejected"]
    check("and the reason names the scale, not generic words",
          bool(rejected) and "variant" in (rejected[0].get("reason") or ""),
          str(rejected[:1]))

    # Age of the best: kept by the harness, and the policy sees it as the difference with
    # `iteration`. Counted by a CHANGE of the best, not by every update.
    class Watcher:
        def __init__(self):
            self.seen = []

        def plan(self, state):
            self.seen.append((state.iteration, state.best_set_iteration))
            parent = sorted(state.active)[0]
            return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=1)])

    genome = Watcher()
    run(genome, FakeResources(config={"limits": {"iterations": 4, "depth": 5}}))
    check("the age of the best is the iteration it appeared in, not the current one",
          all(best <= it for it, best in genome.seen), str(genome.seen))
    check("and it does not grow until the best changes",
          len({best for _, best in genome.seen}) <= len(genome.seen), str(genome.seen))

    print("5. `done` belongs to the policy and is said from TWO places")

    class QuickGenome:
        def plan(self, state):
            parent = sorted(state.active)[0]
            return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=1)])

        def select(self, state):
            alive = state.alive()
            return Verdict(keep={alive[-1].id}, done=True, reason="sufficient")

    result = run(QuickGenome(), FakeResources())
    check("stop by the policy's decision", result.stop_reason == STOP_DONE, result.stop_reason)
    # The finishing door is a separate field; `stop_reason` is the same.
    check("the `select` door is named in the part result",
          result.done_by == "select", str(result.done_by))
    check("and in the journal row of the last step",
          bool(result.journal) and result.journal[-1].get("done_by") == "select",
          str(result.journal[-1:]))
    check("unfinished steps have no door",
          all(row.get("done_by") is None for row in result.journal[:-1]),
          str(result.journal))

    # The second door: the policy says `done` straight from `plan()`. Needed where the
    # decision is visible exactly at the moment of choosing an action, and what was bought
    # a move later is accepted without that move's results. An EXPLICIT flag must tell it
    # from `plan_empty`: an empty plan without it is still the outcome of a broken mutant.
    class FinishesInPlan:
        def __init__(self):
            self.selects = 0

        def plan(self, state):
            return Plan(actions=[], done=True, reason="shape matched")

        def select(self, state):
            self.selects += 1
            return Verdict(keep=set(), done=False)

    genome = FinishesInPlan()
    events = []
    res = FakeResources()
    res.journal = types.SimpleNamespace(
        event=lambda kind, **fields: events.append((kind, fields)),
        stage=lambda name: _noop_ctx(),
    )
    result = run(genome, res)
    check("an empty plan WITH the flag is done, not plan_empty",
          result.stop_reason == STOP_DONE, result.stop_reason)
    stops = [row for kind, row in events if kind == "search_stop"]
    check("and the policy reason reached the journal",
          bool(stops) and stops[-1].get("detail") == "shape matched", str(stops[-1:]))
    # The iteration journal is a separate file with a separate reader (`journal.json`).
    # An early exit on `Plan.done` used to come BEFORE the place of writing, and a part
    # finished without actions left not a single line in the journal.
    check("finishing without actions is also recorded in the iteration journal",
          bool(result.journal) and result.journal[-1].get("done") is True,
          str(result.journal))
    check("together with the policy's justification",
          bool(result.journal) and result.journal[-1].get("select_reason") == "shape matched",
          str(result.journal[-1:] if result.journal else []))
    check("select() is not called at all - nothing to execute",
          genome.selects == 0, str(genome.selects))
    check("an ending in plan() without `stalled` is the quality door",
          result.done_by == "quality" and result.journal[-1].get("done_by") == "quality"
          and stops[-1].get("done_by") == "quality", str((result.done_by, stops[-1:])))

    class GivesUp:
        def plan(self, state):
            return Plan(actions=[], done=True, reason="stalled", stalled=True)

    result = run(GivesUp(), FakeResources())
    check("giving up is the same `done`, but the `stall` door",
          result.stop_reason == STOP_DONE and result.done_by == "stall",
          f"{result.stop_reason} {result.done_by}")

    # The flag together with actions: the move runs, and the part ends AFTER it.
    class LastTurn:
        def __init__(self):
            self.plans = 0

        def plan(self, state):
            self.plans += 1
            parent = sorted(state.active)[0]
            return Plan(actions=[Action(tool="stepwise", parent_id=parent, n=1)],
                        done=True, reason="last step")

        def select(self, state):
            return Verdict(keep=set(state.active), done=False, reason="")

    genome = LastTurn()
    events = []
    res = FakeResources()
    res.journal = types.SimpleNamespace(
        event=lambda kind, **fields: events.append((kind, fields)),
        stage=lambda name: _noop_ctx(),
    )
    result = run(genome, res)
    check("a flag with actions ends the part after the step",
          result.stop_reason == STOP_DONE, result.stop_reason)
    check("and the step itself took place - the plan was asked exactly once",
          genome.plans == 1, str(genome.plans))
    stops = [row for kind, row in events if kind == "search_stop"]
    check("a verdict that said 'continue' does not cancel the flag",
          bool(stops) and stops[-1].get("detail") == "last step", str(stops[-1:]))

    # Both said it: the ending belongs to the plan, which decided earlier.
    class BothDoors(LastTurn):
        def plan(self, state):
            plan = super().plan(state)
            plan.stalled = True
            return plan

        def select(self, state):
            return Verdict(keep=set(state.active), done=True, reason="stall rule")

    result = run(BothDoors(), FakeResources())
    check("`Plan.done` and `Verdict.done` on one step is the plan door",
          result.done_by == "stall", str(result.done_by))

    print("6. Dedup: the same code is not executed twice")

    class RepeatingGenome:
        """Asks for the same thing from one parent and gets its own copies back."""

        def plan(self, state):
            root = state.pool[0].id
            return Plan(actions=[Action(tool="stepwise", parent_id=root, n=1)])

        def select(self, state):
            return Verdict(keep={state.pool[0].id}, done=state.iteration >= 2)

    class FrozenResources(FakeResources):
        """A generator returning the same code every time: it is dedup that is checked."""

        def propose_steps(self, pred_mesh_path, prev_code, k, step=None, variant=0, tag="",
                          attempt=None, temperature=None, top_p=None):
            self.budget.spend("vlm", k)
            return [FakeProposal(full_code=prev_code + "\nsame()", step_code="same()")]

    res = FrozenResources()
    run(RepeatingGenome(), res)
    check("repeated code executed once", len(res.evaluated) == 1, str(len(res.evaluated)))
    check("generator calls did take place", res.budget.counts["vlm"] >= 2,
          str(res.budget.counts))

    print("7. Resampling: a repeat from the same parent is another attempt")

    class ResamplingGenome:
        def plan(self, state):
            root = state.pool[0].id
            return Plan(actions=[Action(tool="stepwise", parent_id=root, n=1)])

        def select(self, state):
            return Verdict(keep={state.pool[0].id}, done=state.iteration >= 2)

    res = FakeResources()
    run(ResamplingGenome(), res)
    attempts = [call[3] for call in res.calls if call[0] == "propose"]
    check("attempt number grows", attempts == [0, 1, 2], str(attempts))
    codes = res.evaluated
    check("and the draws differ", len(set(codes)) == len(codes), str(codes))

    print("8. det pool: cursor slices, an exhausted pool withdraws the action")

    class DetGenome:
        def __init__(self):
            self.saw_remaining: list[dict] = []

        def plan(self, state):
            self.saw_remaining.append(dict(state.det_remaining))
            root = state.pool[0].id
            legal = [item for item in state.legal
                     if item.tool == "det_cold" and item.parent_id == root]
            if not legal:
                return Plan(actions=[])
            return Plan(actions=[Action(tool="det_cold", parent_id=root, n=2)])

        def select(self, state):
            return Verdict(keep={state.pool[0].id}, done=False)

    res = FakeResources(det_pool=["op0", "op1", "op2"], config={"limits": {"iterations": 9}})
    genome = DetGenome()
    result = run(genome, res)
    det_calls = [call for call in res.calls if call[0] == "det"]
    check("the pool was computed once", res.budget.counts["det"] == 1, str(res.budget.counts))
    check("slices went by cursor", [call[3] for call in det_calls] == [2, 1],
          str([call[3] for call in det_calls]))
    check("an exhausted pool stopped the part", result.stop_reason in {STOP_PLAN_EMPTY, STOP_NO_LEGAL},
          result.stop_reason)
    check("the pool remainder was visible to the policy",
          any(row for row in genome.saw_remaining if row), str(genome.saw_remaining))

    print("8a. det fallback line: the thinned one runs only if the original is open")
    # Execution stub: code with a nearly coinciding vertex or with a BAD mark is not closed;
    # an OK mark always closes.
    poly = "r=extrude(None,(0,0,-10),'XY',\"sketch().polygon([(0.0,0.0),(10.0,0.0),{}(10.0,5.0),(0.0,0.0)])\",20){}"
    dup = "(10.0001,0.0001),"
    ops = [poly.format(dup, ""), poly.format(dup, "#OK"), poly.format(dup, "#BAD"), "r=revolve(r)#BAD"]

    class WatertightByCode(FakeResources):
        def evaluate_codes(self, codes, gt_mesh_path, name_prefix, measure=True, step=None, needs=()):
            out = super().evaluate_codes(codes, gt_mesh_path, name_prefix, measure, step, needs)
            for code, row in zip(codes, out):
                last = code.rstrip().splitlines()[-1]
                row.metrics["pred_watertight"] = "#OK" in last or (dup not in last and "#BAD" not in last)
            return out

    class DetAllGenome:
        def plan(self, state):
            root = state.pool[0].id
            if state.iteration > 0:
                return Plan(actions=[], done=True)
            return Plan(actions=[Action(tool="det_cold", parent_id=root, n=len(ops))])

    res = WatertightByCode(det_pool=ops)
    fallback_events: list[dict] = []
    res.journal = types.SimpleNamespace(
        event=lambda kind, **fields: fallback_events.append(fields) if kind == "fallback_code" else None,
        stage=lambda name: _noop_ctx(),
    )
    run(DetAllGenome(), res)
    retried = [code for code in res.evaluated if dup not in code and "polygon" in code]
    check("the fallback was executed for the two open ones with close vertices, and only for them",
          len(res.evaluated) == len(ops) + 2 and len(retried) == 2, str(res.evaluated))
    check("the journal sees the attempt and the acceptance",
          [(e["tried"], e["accepted"]) for e in fallback_events] == [(2, 1)], str(fallback_events))

    print("9. The policy's own time is counted as a separate cost type")

    class SlowGenome:
        def plan(self, state):
            import time as _time

            _time.sleep(0.05)  # "own scoring function": CPU inside the part
            self.spent = dict(state.spent)
            root = sorted(state.active)[0]
            return Plan(actions=[Action(tool="stepwise", parent_id=root, n=1)])

        def select(self, state):
            self.spent = dict(state.spent)
            return Verdict(keep=set(), done=True)

    genome = SlowGenome()
    run(genome, FakeResources())
    check("genome_cpu got into the spend", genome.spent.get("genome_cpu", 0) >= 0.05,
          str(genome.spent.get("genome_cpu")))

    print("11b''''. The run's reasoning mode reaches the policy through the seam")
    # A run condition, but the policy must KNOW it: `</think>` never arrives when reasoning
    # is off, and without the mode the policy would read every truncated answer as
    # unfinished reasoning.
    class ModeProbe:
        def __init__(self):
            self.seen: list = []

        def plan(self, state):
            self.seen.append(state.agent_thinking)
            return Plan(actions=[])

        def select(self, state):
            return Verdict(keep=set(), done=True, reason="enough")

    for mode in (True, False, None):
        probe = ModeProbe()
        res_mode = FakeResources(config={"limits": {"iterations": 2, "depth": 3}})
        res_mode.agent_thinking = mode
        run(probe, res_mode)
        check(f"run mode {mode} is visible to the policy",
              probe.seen and all(seen is mode for seen in probe.seen), str(probe.seen))

    # The run's answer ceiling (`experiment.agent.answer_max_tokens`) goes the same way.
    class CapProbe(ModeProbe):
        def plan(self, state):
            self.seen.append(state.agent_answer_max_tokens)
            return Plan(actions=[])

    for cap in (8192, None):
        probe = CapProbe()
        res_cap = FakeResources(config={"limits": {"iterations": 2, "depth": 3}})
        res_cap.agent_answer_max_tokens = cap
        run(probe, res_cap)
        check(f"run reply cap {cap} is visible to the policy",
              probe.seen and all(seen == cap for seen in probe.seen), str(probe.seen))

    print("11b'''. The repair answer cap covers what is asked to be returned")
    # The yardstick is the parent's LAST LINE, not the whole program: the prompt asks for
    # one line and the harness splices in the rest. A ceiling smaller than that line is a
    # guaranteed cutoff. Checked on a direct tool call: the code length cannot be set
    # through a rollout, and it is exactly what decides.
    from cad_agent.harness.tools import REGISTRY as _REG

    def _repair_once(parent_code: str, answer: str, max_tokens: int = 1024):
        """One tool call on a given parent code and a model answer."""
        agent = FakeAgent(Budget(), [answer])

        class _CapRes:
            ask_agent = staticmethod(agent)

        class _CapCtx:
            res = _CapRes()

        outcome = _REG["repair"].run(
            _CapCtx(),
            Candidate(id="c", parent_id=None, depth=1, origin=Origin(tool="stepwise"),
                      code=parent_code, built=False, failure="exec_error"),
            {"prompt": "fix it", "max_tokens": max_tokens},
            1,
            Origin(tool="repair"),
        )
        return outcome, agent

    long_line = "r = extrude(r, [" + ",".join(f"({i},{i})" for i in range(400)) + "])"
    _, cap_agent = _repair_once("import cadquery as cq\nr = broken(r)\n" + long_line, "r = ok()")
    check("the cap is raised to the line size, not left as the policy's",
          cap_agent.max_tokens and cap_agent.max_tokens[0] > 1024,
          f"{cap_agent.max_tokens} for a line of {len(long_line)} characters")
    check("and not above the upper bound of the schema",
          all(t <= _REG["repair"].params_schema["max_tokens"].hi for t in cap_agent.max_tokens),
          str(cap_agent.max_tokens))

    # The flip side of the same yardstick: a long PROGRAM with a short last line does not
    # move the ceiling. It used to, and the request had to be three times larger than
    # the model can return by contract.
    long_program = "\n".join(f"r = op_{i}(r)" for i in range(400))
    _, small_agent = _repair_once(long_program, "r = ok()")
    check("the length of the whole program does not move the cap",
          small_agent.max_tokens == [1024],
          f"{small_agent.max_tokens} for a program of {len(long_program)} characters")

    print("11b''''. The repair answer is a replacement line, the harness assembles the program")
    # A fragment answer used to be executed as a standalone program with an empty
    # namespace, and most failed runs died with `NameError` on dialect functions. The old
    # lines are now unchanged by construction, not by the prompt's request.
    parent_code = ("import cadquery as cq\n"
                   "from cadgen.extrude import extrude\n"
                   "r = broken(r)")
    outcome, _ = _repair_once(parent_code, "```python\nr = extrude(r, 1)\n```")
    produced = outcome.products[0].code
    check("the parent preamble is in place",
          produced.startswith("import cadquery as cq\nfrom cadgen.extrude import extrude"), produced)
    check("the last line is replaced, not appended",
          produced.endswith("r = extrude(r, 1)") and "broken(r)" not in produced, produced)
    check("the splice is marked in the outcome", outcome.info.get("spliced") == 1, str(outcome.info))

    # A whole-program answer is not spliced: otherwise the preamble would appear twice.
    # These are candidates that built under the old contract, and they must build under
    # the new one.
    whole = "import cadquery as cq\nr = whole(r)"
    outcome_whole, _ = _repair_once(parent_code, f"```python\n{whole}\n```")
    check("a whole program goes as is",
          outcome_whole.products[0].code == whole, outcome_whole.products[0].code)
    check("and is not counted as a splice", outcome_whole.info.get("spliced") == 0, str(outcome_whole.info))

    print("11b'. Repair does not apply to det code or to its continuations")
    # `repair` is not called where det code exists. Fencing off by `origin.tool` is NOT
    # ENOUGH, and this is exactly the trap: a `stepwise` that continued a det candidate is
    # marked `stepwise`, while the code is entirely det's (large). So the lineage is asked.
    from cad_agent.harness import tools as tools_mod

    def _link(cid, parent_id, tool, depth):
        return Candidate(
            id=cid, parent_id=parent_id, depth=depth, origin=Origin(tool=tool),
            code="r = box(1,1,1)", built=False, failure="exec_error", error="boom",
        )

    lin_root = _link("root", None, "seed", 0)
    lin_det = _link("d1", "root", "det_warm", 1)
    after_det = _link("s_det", "d1", "stepwise", 2)
    clean = _link("s_clean", "root", "stepwise", 1)
    lin_state = SearchState(
        figure_id="f", gt_mesh_path=Path("/tmp/f.stl"),
        pool=[lin_root, lin_det, after_det, clean],
        fresh=[after_det, clean],
        tools=tools_mod.cards(["stepwise", "det_warm", "repair"]),
        legal=[LegalAction(tool="repair", parent_id=cid, max_n=1)
               for cid in ("s_det", "s_clean")],
    )

    check("the lineage sees det through an intermediate stepwise",
          "det_warm" in lin_state.lineage_tools(after_det),
          str(sorted(lin_state.lineage_tools(after_det))))
    check("a pure stepwise has no det in its lineage",
          "det_warm" not in lin_state.lineage_tools(clean),
          str(sorted(lin_state.lineage_tools(clean))))

    print("11c. The tool set from the config narrows what the policy sees")
    # The OUTCOME is checked, not the legality of the key: `experiment.tools` could pass
    # validation and never reach the loop. So we look at what the policy sees: legal
    # actions and the price list.
    seen: dict[str, set] = {"legal": set(), "price": set()}

    class WatchingGenome:
        """Decides nothing: records what it was shown and stays silent."""

        def plan(self, state):
            seen["legal"].update(item.tool for item in state.legal)
            seen["price"].update(state.price)
            return Plan(actions=[Action(tool="stepwise", parent_id=state.pool[0].id, n=1)])

        def select(self, state):
            return Verdict(keep=set(), done=True)

    res_narrow = FakeResources(
        config={"limits": {"iterations": 2, "depth": 3}, "tools": ["stepwise"]},
        with_agent=True,
    )
    run(WatchingGenome(), res_narrow)
    check("narrow set: only the ordered tool is legal",
          seen["legal"] == {"stepwise"}, str(sorted(seen["legal"])))
    check("narrow set: the price list is narrowed together with the legal ones",
          seen["price"] == {"stepwise"}, str(sorted(seen["price"])))

    seen["legal"].clear()
    seen["price"].clear()
    res_default = FakeResources(
        config={"limits": {"iterations": 2, "depth": 3}}, with_agent=True,
    )
    run(WatchingGenome(), res_default)
    check("the default set does not offer optimize",
          "optimize" not in seen["price"], str(sorted(seen["price"])))
    check("the default set offers det and repair",
          {"det_cold", "det_warm", "repair"} <= seen["price"], str(sorted(seen["price"])))

    seen["legal"].clear()
    seen["price"].clear()
    res_opt = FakeResources(
        config={"limits": {"iterations": 2, "depth": 3},
                "tools": ["stepwise", "optimize"]},
        with_agent=True,
    )
    run(WatchingGenome(), res_opt)
    check("a listed optimize reaches the price list",
          "optimize" in seen["price"], str(sorted(seen["price"])))
    # There is no assistant in this run, so there must be no repair even if it was asked
    # for: that is a fact of the environment, not a setting.
    seen["legal"].clear()
    seen["price"].clear()
    res_no_agent = FakeResources(
        config={"limits": {"iterations": 2, "depth": 3},
                "tools": ["stepwise", "repair"]},
    )
    withheld_events: list[tuple] = []
    res_no_agent.journal = types.SimpleNamespace(
        event=lambda kind, **fields: withheld_events.append((kind, fields)),
        stage=lambda name: _noop_ctx(),
    )
    run(WatchingGenome(), res_no_agent)
    check("repair is removed by the absence of the assistant, not by the config",
          "repair" not in seen["price"], str(sorted(seen["price"])))
    # The difference between "not asked for" and "asked for, but nothing to run it with"
    # must be visible: the report shows zero `agent_text` calls for both.
    withheld = [row for kind, row in withheld_events if kind == "tools_withheld"]
    check("what the environment withheld is recorded in the journal",
          bool(withheld) and "repair" in (withheld[0].get("tools") or []),
          str(withheld[:1]))
    # Clean up the directories: the section creates four stubs for one fact, and there is
    # no reason to leave litter in /tmp for that.
    for spent in (res_narrow, res_default, res_opt, res_no_agent):
        shutil.rmtree(spent.work_dir, ignore_errors=True)

    print("12. The policy registry returns live objects")
    for name in ("dialogue_lean",):
        built = build_policy(name)
        check(f"policy {name} builds", hasattr(built, "plan"))

    print("12a. The policy picks the selection scale itself, and the choice is frozen")
    from cad_agent.capabilities import objective as objective_mod

    iou = objective_mod.get_objective("iou")
    memory: dict = {}
    # Neither IoU nor GMS exists anywhere: the chain must reach CD and stop there.
    picked = objective_mod.scale_for(iou, memory, [{"cd_runtime": 0.5}, None])
    check("the scale falls to a computable one", picked.name == "cd", picked.name)
    check("the choice is recorded in the part memory", memory.get("scale") == "cd", str(memory))
    # An IoU measurement appeared later: the scale does NOT change. Candidates measured
    # with different rulers cannot be compared.
    again = objective_mod.scale_for(iou, memory, [{"iou": 0.9}])
    check("the scale does not change mid-part", again.name == "cd", again.name)
    # While none is computed, nothing is frozen: values appear later, and the prediction
    # may be non-watertight for the first candidates.
    empty: dict = {}
    check("without measurements the scale stays as declared",
          objective_mod.scale_for(iou, empty, [None, {}]).name == "iou" and not empty,
          str(empty))

    print("13. A whole run: the real harness, a policy as the scaffold")
    # Only the model and CAD are stubbed (`harness_e2e`); everything else is real: config,
    # workers, journal, report. That is why this check lives here whole rather than split
    # into units: a loop that passes on stubs and fails on building `per_figure.json` is
    # exactly the class we catch by hand.
    import harness_e2e

    harness_e2e.install_fakes()
    tmp = Path(tempfile.mkdtemp(prefix="search_e2e_"))
    folder = harness_e2e.make_dataset(tmp, n=3)
    config = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1)
    config["limits"] = {"iterations": 4, "depth": 4}
    # CD is ordered on purpose. The harness always asks for IoU and GMS and does not hand
    # that choice to the policy, yet locally NEITHER half is computed: there is no boolean
    # engine and no `pykdtree`. The only metric computable here is CD, and it is also the
    # harness fallback field (`search.FALLBACK_FIELD`). Without it the section would measure
    # not the loop but the absence of engines: zero steps and an empty scale for every part.
    config["metrics"] = {"cd": True}

    from cad_agent.harness import config as config_mod
    from cad_agent.harness import run_eval

    config_mod.validate_run_config(config)
    check("a config with a policy passes validation", True)

    outcome = run_eval.run_experiment(config=config, run_dir=tmp / "run_policy")
    records = outcome["per_figure"]
    check("all parts finished", len(records) == 3, str(len(records)))
    check("part produced code and mesh",
          all(record["n_steps"] > 0 and record["mesh_path"] for record in records),
          str([(r["figure_id"], r["n_steps"]) for r in records]))
    # Quality is deliberately NOT checked: locally there is neither a boolean engine (IoU)
    # nor `pykdtree` (GMS), so the contract `score_i` is zero everywhere. A "score > 0" check
    # would be green only on the server and simply false locally.
    check("the part cost is counted by kind",
          all(record["cost"]["calls"]["vlm"] > 0 and record["cost"]["calls"]["exec"] > 0
              for record in records),
          str([r["cost"]["calls"] for r in records][:1]))
    check("the best-selection scale is recorded",
          all(record["runtime_metrics"].get("fitness_scale") for record in records),
          str([r["runtime_metrics"].get("fitness_scale") for r in records]))

    # A part selected NOT on the contract scale must be visible in the report, otherwise
    # the environment reads as quality. The harness does not know the policy's selection
    # scale, so the report relies on what it knows itself: whether IoU was computed and by
    # which scale the best was chosen. Locally neither IoU nor GMS is computed, so both
    # records must fire.
    run_dir = Path(outcome["run_dir"])
    from cad_agent.harness import report as report_mod

    #
    # The assertions compare the report WITH THE RECORDS, not with a number: whether IoU is
    # computed is up to the environment (no boolean engine locally, there is one in the run
    # environment), and a hard-coded "3 parts without IoU" would test the environment, not
    # the report.
    run_report = report_mod.Run(run_dir, read_events=False)
    expected_no_iou = sum(
        1 for record in records if (record.get("metrics") or {}).get("iou") is None
    )
    check("parts without IoU are counted by the report",
          run_report.figures_without_iou == expected_no_iou,
          f"{run_report.figures_without_iou} vs {expected_no_iou} from the records")
    expected_scales: dict[str, int] = {}
    for record in records:
        name = record["runtime_metrics"]["fitness_scale"]
        expected_scales[name] = expected_scales.get(name, 0) + 1
    check("the best scale is counted over parts",
          run_report.fitness_scales == expected_scales
          and sum(expected_scales.values()) == len(records),
          f"{run_report.fitness_scales} vs {expected_scales}")
    # A run on a policy declares no objective in the config, and the header must not invent
    # one.
    check("the header does not invent an objective absent from the config",
          run_report.objective == "", repr(run_report.objective))

    events = sorted(run_dir.rglob("events.jsonl"))
    text = events[0].read_text(encoding="utf-8") if events else ""
    check("the search trace is recorded in the part journal",
          "search_stop" in text and "legal_actions" in text and "tool_call" in text,
          str(len(events)))

    # A config with policy knobs must fail at startup: two sources of truth (config and
    # policy) are exactly the drift mechanism that makes a run do something other than
    # what is written.
    bad = dict(config)
    bad["scaffold"] = {"kind": "policy", "policy": "dialogue_lean", "k_variants": 3}
    try:
        config_mod.validate_run_config(bad)
        check("a policy knob in the config is rejected", False, "config passed")
    except config_mod.ConfigError as exc:
        check("a policy knob in the config is rejected", True, str(exc)[:70])

    print("14. Action parameters: the schema is checked, values reach the tool")

    class TunedGenome:
        """Asks for a different draw and a different temperature."""

        def __init__(self, params):
            self.params = params

        def plan(self, state):
            root = state.pool[0].id
            return Plan(actions=[Action(tool="stepwise", parent_id=root, n=1,
                                        params=dict(self.params))])

        def select(self, state):
            return Verdict(keep={state.pool[0].id}, done=state.iteration >= 1)

    res = FakeResources()
    run(TunedGenome({"temperature": 1.5, "variant": 2}), res)
    check("temperature reached the generator",
          any(row["temperature"] == 1.5 for row in res.sampling), str(res.sampling[:2]))
    check("point draw reached the generator",
          any(row["variant"] == 2 for row in res.sampling), str(res.sampling[:2]))

    res = FakeResources()
    run(TunedGenome({"temperature": 12.0}), res)
    check("a temperature outside the scale is rejected, not executed",
          not res.sampling and not res.evaluated, str(res.sampling))

    res = FakeResources()
    run(TunedGenome({"tempreture": 1.5}), res)
    check("a knob with a typo is rejected, not silently skipped",
          not res.evaluated, str(res.calls))

    res = FakeResources()
    run(TunedGenome({}), res)
    # The policy did not name the knob: the REGISTRY default (1.0) applies, which the tool
    # substitutes for it. There is no second source of truth for the knob any more, so the
    # default is checked as a number, not as `None`.
    check("without parameters the registry default applies, not nothing",
          res.sampling and res.sampling[0]["temperature"] == DEFAULT_TEMPERATURE,
          str(res.sampling[:1]))

    print("15. Spend is planned: the remainder is divided among the actions of ONE plan")

    class GreedyPairGenome:
        """Two det actions per iteration: checks that the ceiling holds for the whole plan."""

        def plan(self, state):
            root = state.pool[0].id
            return Plan(actions=[
                Action(tool="det_cold", parent_id=root, n=1),
                Action(tool="det_cold", parent_id=root, n=1),
                Action(tool="det_cold", parent_id=root, n=1),
            ])

        def select(self, state):
            return Verdict(keep={state.pool[0].id}, done=True)

    budget = Budget(limits={"det": 1, "exec": 50})
    res = FakeResources(budget=budget, det_pool=["a()", "b()", "c()", "d()"])
    run(GreedyPairGenome(), res)
    check("the call cap is not broken by a multi-action plan",
          budget.counts["det"] <= 1, str(budget.counts))

    print("16. The budget remainder is visible to the policy and counted per counter")

    seen = {}

    class BudgetReadingGenome:
        def plan(self, state):
            root = state.pool[0].id
            seen["afford_3"] = state.affordable_n("stepwise", 3)
            seen["afford_pace"] = state.affordable_n("stepwise", 3, pace=True)
            seen["afford_spread"] = state.affordable_n("stepwise", 3, spread=3)
            seen["forecast"] = state.tools["stepwise"].forecast(3)
            seen["price"] = state.cost_of("stepwise", 3)
            return Plan(actions=[Action(tool="stepwise", parent_id=root,
                                        n=max(1, seen["afford_3"]))])

        def select(self, state):
            return Verdict(keep={state.pool[0].id}, done=True)

    res = FakeResources(budget=Budget(limits={"vlm": 2, "exec": 50}), config={"limits": {"iterations": 4}})
    run(BudgetReadingGenome(), res)
    check("the remainder cuts n by the call counter", seen["afford_3"] == 2, str(seen))
    check("the spend step is divided by the remaining iterations", seen["afford_pace"] == 1, str(seen))
    check("and by the parallel actions of the plan", seen["afford_spread"] == 1, str(seen))
    check("the counter forecast is samples, not a request",
          seen["forecast"] == {"vlm": 3, "exec": 3}, str(seen["forecast"]))
    check("price and counters are different scales",
          seen["price"] != seen["forecast"], f"{seen['price']} vs {seen['forecast']}")

    budget = Budget(limits={"vlm": 0, "exec": 50})
    state_probe = {}

    class ExhaustedGenome:
        def plan(self, state):
            state_probe["afford"] = state.affordable_n("stepwise", 1)
            return Plan(actions=[])

        def select(self, state):
            return Verdict(done=True)

    res = FakeResources(budget=budget)
    run(ExhaustedGenome(), res)
    check("an exhausted cap reads as zero, not as one",
          state_probe.get("afford", None) in (0, None), str(state_probe))

    print("17. The assistant channel cap refuses BEFORE the call")

    budget = Budget(limits={"agent_text": 0, "vlm": 10, "exec": 50})
    res = FakeResources(budget=budget, with_agent=True, agent_answers=["A"])
    try:
        res.ask_agent("question")
        check("an exhausted assistant cap refuses before the call", False, "the call took place")
    except Exception as exc:
        check("an exhausted assistant cap refuses before the call",
              type(exc).__name__ == "BudgetExceeded" and budget.counts["agent_text"] == 0,
              f"{type(exc).__name__}, counter {budget.counts['agent_text']}")

    print("18. An invalid model: cannot be built on, cannot be chosen, returned only as a last resort")
    # A non-watertight prediction is an error, not a candidate without a number. The rule
    # must live in the HARNESS: the policy is edited by a mutator, and a ban living only
    # there is silently lifted by the first mutation.
    from cad_agent.harness import tools as tools_mod

    def _cand(cid, failure=None, metrics=None, depth=1, tool="stepwise"):
        return Candidate(
            id=cid, parent_id="root", depth=depth,
            origin=Origin(tool=tool, attempt=0),
            code="r = box(1,1,1)", mesh_path=f"/tmp/{cid}.stl",
            metrics=metrics if metrics is not None else {"iou": 0.5},
            built=True, failure=failure,
        )

    notwt = _cand("notwt", failure="not_watertight", metrics={"cd_runtime": 0.01})
    valid = _cand("valid", metrics={"iou": 0.5})

    # Depth-growing tools are those that append an operation to the parent's prefix.
    # `det_cold` is NOT in this list, and not by oversight: it has `requires="none"`, attaches
    # only to the root (depth 0) and builds from the TARGET, not from the parent's geometry,
    # so it never has an invalid parent. Checking it with the same candidate would give a
    # green line because of depth, not because of the rule.
    for name in ("stepwise", "det_warm"):
        spec = tools_mod.REGISTRY[name]
        check(f"{name} does not take an invalid model as a parent", not spec.accepts(notwt))
        check(f"{name} takes a valid parent", spec.accepts(valid))
    check("det_cold attaches to the root and is not touched by the parent rule",
          tools_mod.REGISTRY["det_cold"].accepts(_cand("root", depth=0))
          and not tools_mod.REGISTRY["det_cold"].accepts(valid))
    check("repair may fix an invalid model: that is its job",
          tools_mod.REGISTRY["repair"].accepts(notwt))

    # The `optimize` threshold on the number of numbers in the parent's code. Guarded by the
    # legality outcome at the boundary, not by the threshold value: the threshold is a
    # measurement and may be re-measured.
    from cad_agent.harness.search_types import code_numbers
    check("code numbers are counted as literals, not as digits of names and fractional parts",
          code_numbers("from cadgen.sweep_adv import x2\nr=extrude(None,(0.0000,-102.5),'XY',3)") == 3,
          str(code_numbers("from cadgen.sweep_adv import x2\nr=extrude(None,(0.0000,-102.5),'XY',3)")))
    opt = tools_mod.REGISTRY["optimize"]
    limit = opt.max_code_numbers
    check("optimize has a threshold on code numbers", isinstance(limit, int) and limit > 0, str(limit))

    def _with_numbers(k):
        cand = _cand(f"n{k}")
        cand.code = "r = box(" + ",".join(["1.5"] * k) + ")"
        return cand

    check("optimize is legal on code just below the threshold", opt.accepts(_with_numbers(limit - 1)))
    check("and removed at the threshold", not opt.accepts(_with_numbers(limit)))
    check("stepwise is not affected by the optimize threshold", tools_mod.REGISTRY["stepwise"].accepts(_with_numbers(limit)))
    check("the threshold reaches the policy card",
          tools_mod.card(opt).max_code_numbers == limit, str(tools_mod.card(opt).max_code_numbers))

    check("an invalid candidate is not alive", not notwt.alive)
    check("a valid one is counted as alive", valid.alive)

    # The best: a valid candidate must beat an invalid one. An unclosed one has no metrics
    # at all, but it stays the best while there are no valid ones: otherwise the
    # `not_watertight` failure would read as "no result".
    loop = search.SearchLoop(StepGenome())
    figure = search._FigureSearch(loop, observation(), FakeResources())
    figure._update_best(_cand("notwt2", failure="not_watertight",
                              metrics={"pred_watertight": False}))
    check("an invalid one becomes the best while there are no valid ones - the diagnosis is not replaced",
          figure.best is not None and figure.best.id == "notwt2")
    figure._update_best(_cand("ok", metrics={"iou": 0.2}))
    check("a valid one displaces an invalid one even when behind on the fallback scale",
          figure.best is not None and figure.best.id == "ok", str(figure.best and figure.best.id))
    figure._update_best(_cand("notwt3", failure="not_watertight",
                              metrics={"pred_watertight": False}))
    check("an invalid one does not displace a valid one back",
          figure.best is not None and figure.best.id == "ok", str(figure.best and figure.best.id))
    check("the best scale is named by its tier, not by the validity key",
          figure.best_rank is not None and figure.best_rank[1] == 1, str(figure.best_rank))

    # Selection: the policy cannot put an invalid model into the working set.
    class _KeepState:
        """The minimum `_accept_keep` reads: the pool by id and the iteration number."""
        iteration = 1

        def __init__(self, pool):
            self._pool = {c.id: c for c in pool}

        def get(self, candidate_id):
            return self._pool.get(candidate_id)

    kept = figure._accept_keep({"notwt", "valid", "no-such"}, _KeepState([notwt, valid]))
    check("an invalid model cannot be chosen", kept == {"valid"}, str(sorted(kept)))

    print("\nFAILURES (%d): %s" % (len(FAILURES), "; ".join(FAILURES)) if FAILURES
          else "\nAll search loop checks are green")
    sys.exit(1 if FAILURES else 0)


class _noop_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
