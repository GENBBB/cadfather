#!/usr/bin/env python3
"""Check of the `dialogue_lean` policy: answer parsing, the candidate table, degradations.

The end-to-end run on stubs (`harness_e2e.py`) verifies that the policy runs at all
and stays within its caps, but a well-behaved stub always answers it. This file
checks exactly what a smoke test cannot show: what happens when the assistant is
silent, answers off-format, gets cut off in its reasoning, or asks for something not
in the table. Each of these outcomes is a normal event in a live run, and all of
them must end in a move, not in a lost part.

Most stubs answer in the plain line format (`ACTION:`, `SHOW:`, `DONE:`), which the
policy reads when an answer has no function calls; `_ToolAnswers` answers with calls.

`SearchState` is real here: it is plain data and needs no harness. The tool cards
(`tools.cards`) are real too: the policy reads the semantics of `n`, knobs and price
from them, and a stub of the cards would test an interface that does not exist in
production.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

if "openai" not in sys.modules:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub

from cad_agent.harness import tools as tools_mod  # noqa: E402
from cad_agent.harness.budget import BudgetExceeded  # noqa: E402
from cad_agent.harness.search_types import (  # noqa: E402
    Attempt,
    Candidate,
    LegalAction,
    Origin,
    PromptTooLarge,
    Remaining,
    SearchState,
)
from cad_agent.scaffold import dialogue_io  # noqa: E402
from cad_agent.scaffold.policies import build  # noqa: E402
from cad_agent.scaffold.policies.dialogue_lean import DialogueLeanPolicy  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class _Answers:
    """An assistant that answers from a list and remembers what it was asked."""

    def __init__(self, *answers, raises: Exception | None = None):
        self.answers = list(answers)
        self.raises = raises
        self.prompts: list[str] = []
        self.images: list[object] = []
        # The knobs of every call are part of the stub's interface, not a detail: a
        # re-ask differs from the first question ONLY by them (answer mode and cap),
        # and a stub that did not record them would stay green with the re-ask removed.
        self.calls: list[dict] = []

    def __call__(self, prompt, image=None, max_tokens=16, **kwargs):
        self.prompts.append(prompt)
        self.images.append(image)
        self.calls.append({"max_tokens": max_tokens, "image": image, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.answers.pop(0) if self.answers else ""


class _Image:
    """What `state.render` returns: a set of panels and refusals.

    There is one more panel than candidates: the target goes first, exactly as the
    live `panel_images` returns them.
    """

    def __init__(self, ids=("root0000",), dropped=(), drawn=True):
        self.ids = list(ids)
        self.labels = list("ABCDEFGH")[: len(self.ids)]
        self.image = None
        self.images = [f"panel {index}" for index in range(len(self.ids) + 1)] if drawn else []
        self.dropped = list(dropped)

    def __bool__(self) -> bool:
        return bool(self.ids)


class _Renderer:
    """A renderer that remembers whom it was asked to draw.

    `refuse` reproduces the real refusal of `state_render`: a candidate without a
    mesh does not enter the set, and the reason goes into `dropped` (this is how any
    `exec_error` looks, the very one one wants to inspect before repairing).
    """

    def __init__(self, refuse: bool = False):
        self.calls: list[list[str]] = []
        self.labels: list[list[str] | None] = []
        self.refuse = refuse

    def __call__(self, ids, **kwargs):
        self.calls.append(list(ids))
        self.labels.append(kwargs.get("labels"))
        if self.refuse:
            return _Image(ids=(), drawn=False,
                          dropped=[f"{cid}: no mesh for the image" for cid in ids])
        return _Image(ids)


def _candidate(cid: str, *, metrics=None, built=True, failure=None, tool="stepwise", depth=1,
               mesh=True):
    return Candidate(
        id=cid,
        parent_id=None if depth == 0 else "root0000",
        depth=depth,
        origin=Origin(tool=tool),
        code="r = box(1,1,1)",
        # Everything that was built has a mesh: the policy uses it to decide whether
        # there is anything to draw before calling the renderer.
        mesh_path=f"/dev/null/{cid}.stl" if mesh and built else None,
        metrics=metrics,
        built=built,
        failure=failure,
    )


class _TargetRenderer:
    "Target renderer: remembers how many times it was called."

    def __init__(self, image="target"):
        self.calls = 0
        self.image = image

    def __call__(self):
        self.calls += 1
        return self.image


def _state(*, pool=None, legal=None, ask=None, render=None, fresh=None,
           remaining=None, render_target=None, answer_truncated=None,
           agent_thinking=None, agent_max_images=None, agent_answer_max_tokens=None,
           tools=("stepwise", "det_cold", "optimize")) -> SearchState:
    root = _candidate("root0000", metrics={"iou": 0.1}, tool="root", depth=0)
    root.parent_id = None
    pool = [root] + list(pool or [])
    return SearchState(
        figure_id="check/figure",
        gt_mesh_path=Path("/dev/null"),
        pool=pool,
        fresh=list(fresh or []),
        tools=tools_mod.cards(tools),
        remaining=remaining or Remaining(),
        legal=list(legal if legal is not None else [
            LegalAction(tool="stepwise", parent_id="root0000", max_n=4),
            LegalAction(tool="det_cold", parent_id="root0000", max_n=3),
        ]),
        best_id="root0000",
        ask=ask,
        render=render,
        render_target=render_target,
        answer_truncated=answer_truncated,
        agent_thinking=agent_thinking,
        agent_max_images=agent_max_images,
        agent_answer_max_tokens=agent_answer_max_tokens,
    )


def _broken_state(ask, **kwargs) -> SearchState:
    """A pool with an invalid candidate.

    Broken candidates do not enter the table but do not vanish from the pool: the
    turn report names them, and the assistant may ask to see them (`SHOW:` resolves
    against the pool, not the menu). Both are checked on this state.
    """
    broken = _candidate("bbbb2222", failure="not_watertight", metrics={"iou": 0.3})
    state = _state(
        ask=ask,
        pool=[_candidate("aaaa1111", metrics={"iou": 0.5}), broken],
        legal=[LegalAction(tool="stepwise", parent_id="aaaa1111", max_n=4)],
        **kwargs,
    )
    # The best is not the root: the root has nothing to do here (no legal actions),
    # and only candidates that can be acted on enter the table.
    state.best_id = "aaaa1111"
    return state


def trace_fixes(policy: DialogueLeanPolicy) -> None:
    """Fixes found by analysing traces of live runs.

    Each closes a defect observed in live runs, and each is guarded here by the
    OUTCOME, not by the text: the prompt text belongs to the policy and may be
    rewritten.
    """
    print("\nA refusal names the reason and asks again")
    ask = _Answers("WHY: again\nACTION: tool=optimize candidate=aaaa1111 n=1",
                   "WHY: from the table\nACTION: tool=stepwise candidate=aaaa1111 n=2")
    state = _broken_state(ask)
    state.attempts["aaaa1111"] = [Attempt(origin=Origin(tool="optimize"))]
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("the re-ask happened in the same turn", len(ask.prompts) == 2, str(len(ask.prompts)))
    check("and moved with what it named on the re-ask",
          [(a.tool, a.parent_id) for a in plan.actions] == [("stepwise", "aaaa1111")],
          str([(a.tool, a.parent_id) for a in plan.actions]))
    check("the outcome is a re-ask, not 'assistant's choice' and not a fallback",
          plan.actions and plan.actions[0].reason == policy.OUTCOMES["reasked"][0],
          str(plan.actions and plan.actions[0].reason))
    check("the reason is named: the one-shot tool was already executed",
          "already ran on aaaa1111" in said, said)
    check("and the candidate's legal set is named", "you can call: stepwise" in said, said)
    check("the refusal line starts as before, so old and new runs are parsed the same way",
          "not on the list, ignored — tool=optimize candidate=aaaa1111" in said, said)

    ask = _Answers("ACTION: tool=optimize candidate=aaaa1111 n=1",
                   "ACTION: tool=optimize candidate=aaaa1111 n=1")
    state = _broken_state(ask)
    state.attempts["aaaa1111"] = [Attempt(origin=Origin(tool="optimize"))]
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("no more re-asks than the slot: the `budget.agent_*` caps are not shifted",
          len(ask.prompts) == 1 + policy.MAX_RETRIES_PER_TURN, str(len(ask.prompts)))
    check("everything rejected twice is a separate outcome, not 'assistant's choice'",
          plan.actions and plan.actions[0].reason == policy.OUTCOMES["rejected"][0],
          str(plan.actions and plan.actions[0].reason))
    check("and the assistant is told the action was chosen for it",
          "none of the actions you named were available" in said, said)

    print("\nAn answer with only a WHY is asked again, not sent to the fallback")
    ask = _Answers("WHY: the target is a flanged hub with a bolt circle",
                   "WHY: from the table\nACTION: tool=stepwise candidate=aaaa1111 n=3")
    state = _broken_state(ask)
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("the re-ask happened in the same turn", len(ask.prompts) == 2, str(len(ask.prompts)))
    check("the question changed: it says what the answer lacked",
          "called no function" in ask.prompts[1]
          and "called no function" not in ask.prompts[0], said[-300:])
    check("and moved with what it named on the re-ask, with its n",
          [(a.tool, a.parent_id, a.n) for a in plan.actions] == [("stepwise", "aaaa1111", 3)],
          str([(a.tool, a.parent_id, a.n) for a in plan.actions]))
    check("the outcome is its own, not `reasked` and not 'assistant's choice'",
          plan.actions and plan.actions[0].reason == policy.OUTCOMES["reasked_unparsed"][0],
          str(plan.actions and plan.actions[0].reason))

    ask = _Answers("WHY: the target is a flanged hub",
                   "WHY: from the table\nACTION: tool=optimize candidate=aaaa1111 n=1")
    state = _broken_state(ask)
    state.attempts["aaaa1111"] = [Attempt(origin=Origin(tool="optimize"))]
    plan = policy.plan(state)
    check("one slot for both causes: after a WHY re-ask a rejection is not re-asked",
          len(ask.prompts) == 1 + policy.MAX_RETRIES_PER_TURN, str(len(ask.prompts)))
    check("and everything discarded stays a `rejected` fallback",
          plan.actions and plan.actions[0].reason == policy.OUTCOMES["rejected"][0],
          str(plan.actions and plan.actions[0].reason))

    print("\nThe optimize threshold, in code numbers, is visible to the assistant")
    limit = tools_mod.REGISTRY["optimize"].max_code_numbers
    ask = _Answers("WHY: tune it\nACTION: tool=optimize candidate=aaaa1111 n=1",
                   "WHY: from the table\nACTION: tool=stepwise candidate=aaaa1111 n=2")
    state = _broken_state(ask)
    state.pool[1].code = "r = box(" + ",".join(["1.5"] * limit) + ")"
    check("the optimize legend names the threshold with the number from the card",
          f"{limit} or more numbers" in policy._legend(state), policy._legend(state))
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("a refusal on long code names the number and the threshold, not \"not available now\"",
          f"its code has {limit} numbers, the limit is {limit}" in said, said)

    print("\nThe fallback acts on the best candidate, not the root")
    state = _state(
        ask=_Answers("no format at all"),
        pool=[_candidate("aaaa1111", metrics={"iou": 0.5, "gms_norm": 0.5})],
        legal=[LegalAction(tool="det_cold", parent_id="root0000", max_n=3),
               LegalAction(tool="stepwise", parent_id="root0000", max_n=4),
               LegalAction(tool="stepwise", parent_id="aaaa1111", max_n=4)],
    )
    state.best_id = "aaaa1111"
    plan = policy.plan(state)
    check("an unparsed answer yields an action on the best candidate",
          [a.parent_id for a in plan.actions] == ["aaaa1111"],
          str([(a.tool, a.parent_id) for a in plan.actions]))

    print("\nInspection within a turn")
    render = _Renderer()
    ask = _Answers("SHOW: aaaa1111", "SHOW: aaaa1111",
                   "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render)
    plan = policy.plan(state)
    check("the same ids are not drawn a second time", render.calls == [["aaaa1111"]], str(render.calls))
    check("panels are captioned with the same ids as in the text, not with menu letters",
          render.labels == [["aaaa1111"]], str(render.labels))
    check("and the image stays attached",
          ask.images[2] is not None, str(ask.images))
    check("the turn was settled by the assistant's choice",
          plan.actions and plan.actions[0].reason == policy.OUTCOMES["chosen"][0],
          str(plan.actions and plan.actions[0].reason))

    print("\nDuplicates are named in the turn report")
    ask = _Answers("ACTION: tool=stepwise candidate=aaaa1111 n=2",
                   "ACTION: tool=stepwise candidate=aaaa1111 n=2")
    state = _broken_state(ask)
    state.pool[1].metrics = {"iou": 0.5, "gms_norm": 0.5}
    policy.plan(state)
    state.fresh = []
    state.attempts["aaaa1111"] = [Attempt(origin=Origin(tool="stepwise"), produced=2, built=2,
                                          reused=["aaaa1111", "root0000"])]
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("'no new ones' explains that everything came back as repeats",
          "already on the list as aaaa1111, root0000" in said, said)
    check("the turn report names the current best", "Best so far: aaaa1111" in said, said)

    # Age of the best: stagnation is a harness fact in the same line. By itself it
    # need not advise anything, so it is also checked that a fresh best shows no age.
    line = policy._best_line(state)
    check("a just-changed best has no age in the line",
          "unchanged" not in line, line)
    state.best_set_iteration, state.iteration = 2, 3
    check("the best's age is printed in turns, singular",
          "unchanged for 1 turn." in policy._best_line(state), policy._best_line(state))
    state.iteration = 9
    check("and in the plural", "unchanged for 7 turns." in policy._best_line(state),
          policy._best_line(state))
    state.iteration, state.best_set_iteration = 3, 9
    check("a negative age is not printed",
          "unchanged" not in policy._best_line(state), policy._best_line(state))


def auto_look_checks() -> None:
    """The harness shows images on every turn: panel selection, captions, the model's own request."""
    print("\nInspection without a request")
    policy = DialogueLeanPolicy(objective="iou")

    # Turn 1: no candidates, so the target alone, in the FIRST question, without `SHOW: target`.
    target = _TargetRenderer()
    ask = _Answers("WHY: start\nACTION: tool=stepwise candidate=root0000 n=4")
    state = _state(ask=ask, render=_Renderer(), render_target=target)
    plan = policy.plan(state)
    check("before the first candidate the target is shown unasked",
          target.calls == 1 and ask.images[0] == ["target"], f"{target.calls} {ask.images}")
    check("and the turn happened on the very first question", len(ask.prompts) == 1 and len(plan.actions) == 1,
          f"{len(ask.prompts)} questions, {plan.actions}")
    check("the question says only the target is shown",
          policy.IMAGE_NOTE_TARGET in ask.prompts[0], ask.prompts[0][-900:])

    def pool_state(ask, render, *, fresh_ids):
        parent = _candidate("pppp0000", metrics={"iou": 0.4})
        best = _candidate("aaaa1111", metrics={"iou": 0.8}, depth=2)
        best.parent_id = "pppp0000"
        head = _candidate("nnnn2222", metrics={"iou": 0.3})
        broken = _candidate("bbbb3333", failure="not_watertight", metrics={"iou": 0.9})
        pool = [parent, best, head, broken]
        state = _state(
            ask=ask, render=render, pool=pool,
            fresh=[c for c in pool if c.id in fresh_ids],
            legal=[LegalAction(tool="stepwise", parent_id=c.id, max_n=4)
                   for c in (parent, best, head)],
        )
        state.best_id = "aaaa1111"
        return state

    # The last turn started a new branch: its head next to the best, captioned by roles.
    render = _Renderer()
    ask = _Answers("ACTION: tool=stepwise candidate=nnnn2222 n=4")
    state = pool_state(ask, render, fresh_ids={"nnnn2222", "bbbb3333"})
    policy.plan(state)
    check("the best and the best of the last turn are shown (an invalid one does not count)",
          render.calls == [["aaaa1111", "nnnn2222"]], str(render.calls))
    check("panels are captioned with role, operation count and score",
          "aaaa1111 — best so far (2 ops, score 0.8000)" in ask.prompts[0]
          and "nnnn2222 — best made by your last turn (1 ops, score 0.3000)" in ask.prompts[0],
          ask.prompts[0][-1200:])

    # The last turn produced the best: its parent instead of a duplicate, "before/after".
    render = _Renderer()
    ask = _Answers("ACTION: tool=stepwise candidate=aaaa1111 n=4")
    state = pool_state(ask, render, fresh_ids={"aaaa1111", "nnnn2222"})
    policy.plan(state)
    check("a new best is shown with its parent, not twice",
          render.calls == [["aaaa1111", "pppp0000"]], str(render.calls))
    check("the parent's role is named",
          "pppp0000 — what the best was built on" in ask.prompts[0], ask.prompts[0][-1200:])

    # The last turn gave nothing valid: no third panel, stated in words.
    render = _Renderer()
    ask = _Answers("ACTION: tool=stepwise candidate=aaaa1111 n=4")
    state = pool_state(ask, render, fresh_ids={"bbbb3333"})
    policy.plan(state)
    check("without a new valid candidate a single best is shown", render.calls == [["aaaa1111"]],
          str(render.calls))
    check("and it says there is no panel for the last turn",
          policy.NOTE_NOTHING_NEW in ask.prompts[0], ask.prompts[0][-1200:])

    # The model's request from the last turn is drawn TOGETHER with the automatic ones, after them.
    render = _Renderer()
    ask = _Answers("ACTION: tool=stepwise candidate=aaaa1111 n=4")
    state = pool_state(ask, render, fresh_ids={"nnnn2222"})
    dialogue_io.session_of(state, policy.MEMORY_KEY).want_image = ["pppp0000", "aaaa1111"]
    policy.plan(state)
    check("what the model ordered is added after the automatic ones, without repeats",
          render.calls == [["aaaa1111", "nnnn2222", "pppp0000"]], str(render.calls))

    # A look the model ordered within the turn does not repeat the "no panel" note.
    render = _Renderer()
    ask = _Answers("SHOW: pppp0000", "ACTION: tool=stepwise candidate=aaaa1111 n=4")
    state = pool_state(ask, render, fresh_ids=set())
    policy.plan(state)
    check("the automatic-show note applies only to it",
          policy.NOTE_NOTHING_NEW in ask.prompts[0]
          and policy.NOTE_NOTHING_NEW not in ask.prompts[1].split("This turn's images")[-1],
          ask.prompts[1][-900:])


class _ToolAnswers:
    """An assistant with function calling: answers with (text, calls) pairs and remembers requests.

    Calls are returned the way the harness returns them, separately after the answer
    (`state.answer_tool_calls`), not in the text. `raises_first` makes the first
    question not fit the context (`PromptTooLarge`), like `_OnceTooLarge`.
    """

    def __init__(self, *answers, truncated_at=(), raises_first: bool = False):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.last: list[dict] = []
        self.truncated_at = set(truncated_at)
        self.raises_first = raises_first
        self.cut = False

    def __call__(self, prompt, image=None, max_tokens=16, **kwargs):
        self.calls.append({"prompt": prompt, "image": image, **kwargs})
        if self.raises_first and len(self.calls) == 1:
            raise PromptTooLarge("did not fit")
        self.cut = len(self.calls) in self.truncated_at
        text, calls = self.answers.pop(0) if self.answers else ("", [])
        self.last = [{"id": f"srv_{len(self.calls)}_{index}", "name": name,
                      "arguments": __import__("json").dumps(args)}
                     for index, (name, args) in enumerate(calls)]
        return text

    def tool_calls(self):
        return list(self.last)

    def truncated(self):
        return self.cut


def _tool_state(answers: "_ToolAnswers", **kwargs) -> SearchState:
    state = _state(ask=answers, render=_Renderer(), render_target=_TargetRenderer(), **kwargs)
    state.answer_tool_calls = answers.tool_calls
    state.answer_truncated = answers.truncated
    return state


def function_call_checks() -> None:
    """A turn is function calls: the request carries them, and the history is text.

    The main thing here is the request order: the first turn's prefix must be a prefix
    of the second's, the target image must be the same as on the first turn, and there
    must be exactly as many image markers as images. Otherwise the policy loses the
    prefix cache while still working.
    """
    print("\nFunction calls and the prefix cache")
    from cad_agent.harness.search_types import IMAGE_SLOT
    policy = build("dialogue_lean")
    check("the policy builds by its registry name", type(policy).__name__ == "DialogueLeanPolicy")
    check("a server with a call parser is required", policy.TOOL_CHOICE == "auto")

    # Turn 1: the target only; act with its own n and knobs.
    ask = _ToolAnswers((
        "Start from scratch both ways.",
        [("act", {"tool": "stepwise", "candidate": "root0000", "n": 2,
                  "params": {"temperature": 1.7}}),
         ("act", {"tool": "det_cold", "candidate": "root0000", "n": 16})],
    ))
    state = _tool_state(ask, legal=[LegalAction(tool="stepwise", parent_id="root0000", max_n=32),
                                    LegalAction(tool="det_cold", parent_id="root0000", max_n=32)])
    plan = policy.plan(state)
    first = ask.calls[0]
    specs = {spec["function"]["name"]: spec["function"] for spec in first.get("tools", [])}
    check("the functions are only act and finish", sorted(specs) == ["act", "finish"], str(sorted(specs)))
    check("act has no knobs, n is optional",
          "params" not in specs["act"]["parameters"]["properties"]
          and specs["act"]["parameters"]["required"] == ["tool", "candidate"])
    prompt = first["prompt"]
    check("the question has no look, no knobs and no policy objective",
          "call look" not in prompt and "LOOK" not in prompt and "params" not in prompt
          and "temperature" not in prompt and "Tracked objective" not in prompt, prompt)
    check("one marker, one target image, at the end of the unchanged part",
          prompt.count(IMAGE_SLOT) == 1 and first["image"] == ["target"]
          and prompt.index(IMAGE_SLOT) < prompt.index("(you were shown the target"), prompt[:300])
    by_tool = {action.tool: action for action in plan.actions}
    check("n for stepwise is the model's choice, for det_cold it is fixed",
          by_tool["stepwise"].n == 2 and by_tool["det_cold"].n == policy.FIXED_N["rank_depth"],
          str([(a.tool, a.n) for a in plan.actions]))
    check("the model's knobs do not reach the action",
          by_tool["stepwise"].params.get("temperature") != 1.7, str(by_tool["stepwise"].params))
    check("the unchanged part is the registry legend, not the turn menu",
          "- optimize —" in prompt and "- det_cold —" in prompt, prompt[:2500])
    check("the det_cold card compares it with the generator rather than calling it an algorithm",
          "Fitted is not more accurate than generated" in " ".join(prompt.split()), prompt[:2500])

    # Turn 2: the best and the head of a new branch. The first turn's prefix is the second's prefix.
    best = _candidate("aaaa1111", metrics={"iou": 0.8, "gms_norm": 0.7})
    head = _candidate("nnnn2222", metrics={"iou": 0.3})
    state.pool.extend([best, head])
    # The best is from earlier turns, only the new branch head is fresh: two panels.
    state.fresh = [head]
    state.best_id = "aaaa1111"
    state.legal = [LegalAction(tool="stepwise", parent_id=c.id, max_n=8) for c in state.pool]
    ask.answers.append(("Done.", [("finish", {"reason": "shape", "why": "all features in place"}),
                                  ("look", {"ids": ["aaaa1111"]})]))
    plan = policy.plan(state)
    second = ask.calls[1]["prompt"]
    head1 = prompt[: prompt.index(policy.IMAGE_NOTE_TARGET)].rstrip()
    check("the first turn's prefix is the second's prefix (the cache holds)",
          second.startswith(head1), second[:len(head1) + 200])
    images = ask.calls[1]["image"]
    check("exactly as many markers as images",
          isinstance(images, list) and second.count(IMAGE_SLOT) == len(images) == 3,
          f"{second.count(IMAGE_SLOT)} {images!r}")
    check("the target on the second turn is the same image as on the first", images[0] == "target", repr(images))
    check("turn markers come after the history, before the table",
          second.index("Turn 1 executed") < second.rindex(IMAGE_SLOT)
          < second.index("Candidates you can act on"), second)
    executed = next(line for line in second.splitlines() if line.startswith("Turn 1 executed"))
    check("the executed action is one line, without knobs",
          executed == "Turn 1 executed: stepwise on root0000 n=2; det_cold on root0000 n=4",
          executed)
    table = second[second.index("Candidates you can act on"):]
    check("n≤ is printed only for stepwise", "stepwise(n≤8)" in table, table)
    check("its own earlier answer is in the transcript as words and a CALL line",
          "Start from scratch both ways." in second and "CALL act {" in second, second[:2500])
    check("functions are constant from turn to turn (prefix cache)",
          ask.calls[0]["tools"] == ask.calls[1]["tools"])
    check("finish ended the part, look was ignored",
          plan.done and not plan.actions
          and not dialogue_io.session_of(state, policy.MEMORY_KEY).want_image, str(plan))
    check("without prewarm the question start is not passed to the harness",
          all("prefix" not in call for call in ask.calls), str([list(c) for c in ask.calls]))

    # Prewarm (`experiment.agent.prewarm`): the same, but the beginning of the question goes to the harness.
    from PIL import Image
    from cad_agent.capabilities import llm
    ask = _ToolAnswers(
        ("Start.", [("act", {"tool": "stepwise", "candidate": "root0000", "n": 2})]),
        ("Go on.", [("act", {"tool": "stepwise", "candidate": "aaaa1111", "n": 2})]),
    )
    state = _tool_state(ask, legal=[LegalAction(tool="stepwise", parent_id="root0000", max_n=8)])
    state.agent_prewarm = True
    policy = build("dialogue_lean")
    policy.plan(state)
    state.pool.extend([best, head])
    state.fresh = [head]
    state.best_id = "aaaa1111"
    state.legal = [LegalAction(tool="stepwise", parent_id=c.id, max_n=8) for c in state.pool]
    policy.plan(state)
    first, second = ask.calls
    check("the question start is a prefix of the question and ends with a separator",
          all(c["prompt"].startswith(c["prefix"]) and c["prefix"].endswith("\n\n")
              for c in (first, second)), repr(first.get("prefix", ""))[-80:])
    check("the start of the first turn is a prefix of the start of the second (the save point carries over)",
          second["prefix"].startswith(first["prefix"])
          and "Turn 1 executed" in second["prefix"][len(first["prefix"]):], second["prefix"][-300:])
    check("the start has one marker, the target image",
          first["prefix"].count(IMAGE_SLOT) == second["prefix"].count(IMAGE_SLOT) == 1)
    # Client: the same questions with real images yield the prewarm content.
    pics: dict = {}

    def real(images):
        return [pics.setdefault(name, Image.new("RGB", (2, 2))) for name in images]

    for call in (first, second):
        parts = llm.prewarm_content(call["prompt"], real(call["image"]), call["prefix"])
        check("the client accepts the start as a prefix: text, target, transcript",
              parts is not None and [p["type"] for p in parts] == ["text", "image_url", "text"],
              str(parts and [p["type"] for p in parts]))

    # An answer without a call is re-asked, and the re-ask speaks the language of functions.
    ask = _ToolAnswers(("I think stepwise is best.", []),
                       ("Now calling.", [("act", {"tool": "stepwise", "candidate": "root0000",
                                                  "n": 1})]))
    plan = build("dialogue_lean").plan(_tool_state(ask))
    check("an answer without a call is asked again, the turn took place",
          len(ask.calls) == 2 and len(plan.actions) == 1, f"{len(ask.calls)} {plan.actions}")
    check("the re-ask speaks the function language, not format lines",
          "called no function" in ask.calls[1]["prompt"]
          and "ACTION lines" not in ask.calls[1]["prompt"], ask.calls[1]["prompt"][-2500:])

    # finish with a valid best: the part is done, the stagnation door is recognized.
    ask = _ToolAnswers(("Nothing helps.", [("finish", {"reason": "stalled",
                                                       "why": "last three turns gave nothing"})]))
    state = _broken_state(ask)
    state.answer_tool_calls = ask.tool_calls
    state.answer_truncated = ask.truncated
    plan = build("dialogue_lean").plan(state)
    check("finish ended the part through the stall door",
          plan.done and plan.stalled and not plan.actions, f"{plan}")

    # The server parser failed: the call stayed as text in the template format.
    xml = ("Extend.\n<tool_call>\n<function=act>\n<parameter=tool>\nstepwise\n</parameter>\n"
           "<parameter=candidate>\nroot0000\n</parameter>\n<parameter=n>\n2\n</parameter>\n"
           "</function>\n</tool_call>")
    ask = _ToolAnswers((xml, []))
    plan = build("dialogue_lean").plan(_tool_state(ask))
    check("a call left as template text is still parsed",
          len(ask.calls) == 1 and plan.actions and plan.actions[0].n == 2, f"{plan.actions}")


def llm_slot_checks() -> None:
    """Images at marker positions (`llm._request`); without markers or on mismatch, before the text."""
    print("\nImage markers in the request")
    from PIL import Image
    from cad_agent.capabilities import llm
    from cad_agent.harness.search_types import IMAGE_SLOT

    class _Client:
        def __init__(self):
            self.messages = None
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, model, messages, **kwargs):
            self.messages = messages
            return None

    def kinds(client):
        return [part["type"] for part in client.messages[-1]["content"]]

    pics = [Image.new("RGB", (2, 2), color) for color in ("red", "green", "blue")]
    client = _Client()
    llm._request(client, "m", f"rules{IMAGE_SLOT}history{IMAGE_SLOT}{IMAGE_SLOT}table", pics, {},
                 None)
    check("images land on their markers",
          kinds(client) == ["text", "image_url", "text", "image_url", "image_url", "text"],
          str(kinds(client)))
    client = _Client()
    llm._request(client, "m", f"rules{IMAGE_SLOT}history", pics[:2], {}, None)
    check("fewer markers than images: the old order, markers removed",
          kinds(client) == ["image_url", "image_url", "text"]
          and IMAGE_SLOT not in client.messages[-1]["content"][-1]["text"], str(client.messages))
    client = _Client()
    llm._request(client, "m", f"rules{IMAGE_SLOT}history", None, {}, None)
    check("no images: text without markers",
          client.messages[-1]["content"] == "ruleshistory", str(client.messages))
    client = _Client()
    llm._request(client, "m", "plain", pics[:1], {}, None)
    check("no markers: images before the text, as before",
          kinds(client) == ["image_url", "text"], str(kinds(client)))

    # Prewarm: the beginning of the question in the same parts, the last one a beginning of a question part.
    S = IMAGE_SLOT

    def warm(text, images, prefix):
        parts = llm.prewarm_content(text, images, prefix)
        return parts if parts is None or isinstance(parts, str) else [
            part.get("text", "<img>") for part in parts]

    got = warm(f"rules{S}hist.\n\n{S}{S}table", pics, f"rules{S}hist.\n\n")
    check("warm-up runs to the end of the transcript, with the separator",
          got == ["rules", "<img>", "hist.\n\n"], str(got))
    got = warm(f"rules{S}\n\nnote", pics[:1], f"rules{S}\n\n")
    check("first turn: the separator after the target stays, the target is in the prewarm",
          got == ["rules", "<img>", "\n\n"], str(got))
    got = warm(f"rules{S}\n\n{S}table", pics[:2], f"rules{S}\n\n")
    check("a whitespace piece dropped by the question: prewarm up to the last common text",
          got == ["rules"], str(got))
    check("not the question start: no prewarm",
          warm(f"rules{S}hist\n\n{S}table", pics[:2], f"other{S}") is None)
    check("markers do not match images: no warm-up",
          warm(f"rules{S}hist\n\n{S}table", pics, f"rules{S}hist\n\n") is None)
    check("no images: the start is a string",
          warm("rules\n\nhist\n\ntable", None, "rules\n\nhist\n\n") == "rules\n\nhist\n\n")


def main() -> None:
    print("Answer parsing")
    policy = DialogueLeanPolicy()
    parsed = policy._parse_answer(
        "ACTION: tool=stepwise candidate=aaaa1111 n=3 params=temperature=1.4, bogus=1\n"
        "ACTION: tool=repair candidate=bbbb2222 n=2\n"
        "SHOW: aaaa1111, bbbb2222"
    )
    check("several actions per turn are parsed", len(parsed.picks) == 2,
          str([(p.tool, p.candidate, p.n) for p in parsed.picks]))
    check("knobs named in the line are not taken",
          parsed.picks[0].params == {} and parsed.picks[0].n == 3,
          str(parsed.picks[0]))
    check("an image request is parsed as a list",
          parsed.show == ["aaaa1111", "bbbb2222"], str(parsed.show))
    check("SHOW: none is not a request",
          policy._parse_answer("SHOW: none").show == [])
    check("a line without an address does not become an action",
          policy._parse_answer("ACTION: do something clever").picks == [])
    check("free text gives no actions", policy._parse_answer("let's sample more").picks == [])
    check("the finish line is parsed with its reason",
          policy._parse_answer("DONE: shape matches, only rounding left").done
          == "shape matches, only rounding left")
    check("a finish without a reason is still a finish",
          policy._parse_answer("DONE:").done == "")
    check("a missing line differs from an empty reason",
          policy._parse_answer("WHY: thinking").done is None)
    check("the format placeholder does not end the part",
          policy._parse_answer("DONE: <one short sentence saying why nothing more would help>")
          .done is None)
    check("a finish is parsed together with actions",
          policy._parse_answer(
              "ACTION: tool=stepwise candidate=aaaa1111 n=2\nDONE: last turn").done == "last turn")

    state = _state(ask=_Answers(""))
    question = policy.CHOOSE_QUESTION.format(
        candidates=policy._candidates_text(state, policy._menu(state)),
        max_actions=policy.MAX_ACTIONS_PER_TURN)
    choose_tail = question.rsplit("\n\n", 1)[-1]
    # The smoke stub recognizes the question by its last paragraph.
    check("the turn request is the last paragraph of the question",
          "call act" in choose_tail and "finish" in choose_tail, choose_tail)
    # Stagnation has a legitimate door, and the ban is on the word `shape`, not on
    # finishing in general; otherwise giving up again goes out through "the shape matches".
    check("the finish names two reasons",
          "- shape:" in policy.INTRO and "- stalled:" in policy.INTRO)
    check("the ban on a missing element applies to `shape`, not to the finish",
          "Do NOT say shape while" in policy.INTRO and "Do NOT finish while" not in policy.INTRO)
    check("the policy has no stop question",
          not hasattr(policy, "STOP_QUESTION") and not hasattr(policy, "STOP_MAX_TOKENS"))
    # The assistant did not apply "a repeat takes new samples" to the root.
    check("the rules say that repeating stepwise on the empty start gives a new first operation",
          "repeating stepwise on it draws new ones" in " ".join(policy.INTRO.split()),
          policy.INTRO)

    print("\nPlan line")
    parsed = policy._parse_answer(
        "WHY: c1 is the only watertight branch, extending it before repairing anything\n"
        "ACTION: tool=stepwise candidate=aaaa1111 n=3"
    )
    check("the plan line does not become an action and does not disturb parsing",
          len(parsed.picks) == 1 and parsed.show == [], str(parsed.picks))
    state = _state(ask=_Answers(
        "<think>weighing c1 against c2</think>\n"
        "WHY: extending the only watertight branch\n"
        "ACTION: tool=stepwise candidate=root0000 n=1"))
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    # This line is why the format changed: reasoning does not go into the transcript,
    # and without it the next turn would see its actions without a word about the plan.
    check("the plan went into the transcript, the reasoning did not",
          "WHY: extending" in said and "weighing c1" not in said, said)

    print("\nThinking mode")
    check("the reasoning block is stripped",
          dialogue_io.visible("<think>hmm, c1 looks better</think>\nACTION: tool=stepwise candidate=c1")
          == "ACTION: tool=stepwise candidate=c1")
    check("a tail after </think> without an opening tag",
          dialogue_io.visible("reasoning...</think>\nSTOP") == "STOP")
    check("an unclosed block leaves no visible answer",
          dialogue_io.visible("<think>I need to compare c1 and") == "")
    check("an answer without reasoning is left intact", dialogue_io.visible("STOP") == "STOP")
    state = _state(ask=_Answers("<think>long reasoning that never ends"))
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("truncated reasoning is its own outcome, not \"not parsed\"",
          len(plan.actions) == 1 and "cut off before the named action" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))
    check("and the reasoning did not reach the transcript", "long reasoning" not in said, said)

    # The main truncation case does NOT look like an unclosed tag: the server returns
    # the reasoning without an opening `<think>`, the cut does not deliver the closing
    # one either, and the text is indistinguishable from an off-format answer. The
    # signal comes from the harness.
    cut = "we should extend the strongest branch, but first let me"
    state = _state(ask=_Answers(cut), answer_truncated=lambda: True)
    plan = policy.plan(state)
    check("a cut-off without any tag is recognised by the harness signal",
          len(plan.actions) == 1 and "cut off before the named action" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))
    # The same form is why reasoning leaked into the prompt. `_visible` cannot strip
    # it (there are no tags), so a truncated answer is not put into the transcript at
    # all. The check guards the transcript specifically: the `truncated` outcome
    # occurred before too, yet the text still went into the history and from there
    # into every following question.
    said = "\n".join(state.memory["dialogue"].lines)
    check("cut-off reasoning without tags did not reach the transcript",
          cut not in said, said[-300:])
    check("but the cut-off itself is reported, otherwise the turn looks as if it did not happen",
          "answer cap" in said, said[-300:])
    # The opposite case of the same pair: the answer arrived whole and the parser did
    # not understand it. Merged into one outcome, they would call for fixing the prompt
    # where the cap is too small. The answer is given twice: an understood-not answer
    # that fit is re-asked, and the "not recognized" outcome occurs only after the re-ask.
    state = _state(ask=_Answers(cut, cut), answer_truncated=lambda: False)
    plan = policy.plan(state)
    check("a complete but misunderstood answer is still 'not recognized'",
          len(plan.actions) == 1 and "not recognized" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))

    # The form on which the truncation guard was bypassed: the answer is cut, but the
    # thinking model managed to write a VALID format draft in the middle of its
    # reasoning and kept thinking. The parser found it, the action was executed, and
    # the whole reasoning went into the transcript with it. A draft is not executed at
    # all: it is the middle of a thought, not its conclusion (after the draft the
    # assistant often reached a DIFFERENT decision). The turn is cured by a re-ask,
    # like any truncation.
    draft = (
        "we should extend the strongest branch. draft:\n"
        "WHY: extending the only watertight branch\n"
        "ACTION: tool=stepwise candidate=root0000 n=1\n"
        "but wait, maybe det_warm is better here, let me reconsider that once more"
    )
    good = "ACTION: tool=det_cold candidate=root0000 n=2"
    ask = _Answers(draft, good)
    # The signal refers to the LAST answer, not the turn: a constant would mean
    # "truncated, and the re-ask too", which is a different case (checked below).
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1)
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("a draft in the middle of cut-off reasoning is not executed",
          len(plan.actions) == 1 and plan.actions[0].tool == "det_cold",
          str([(a.tool, a.n) for a in plan.actions]))
    check("instead, a re-ask without reasoning",
          len(ask.calls) == 2 and ask.calls[-1].get("thinking") is False, str(ask.calls))
    check("and the turn is recorded as a re-ask decision",
          "re-ask" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))
    check("neither the reasoning nor the draft reached the transcript",
          "reconsider" not in said and "WHY: extending" not in said
          and "strongest branch" not in said, said[-400:])
    check("but the cut-off is reported", "answer cap" in said, said[-400:])

    # The re-ask runs with reasoning off: it has no tag by nature, and a cut in it is a
    # cut of the ANSWER, not of a thought. Discarding it as a draft would mean the
    # mechanism destroys its own result.
    ask = _Answers(draft, "ACTION: tool=det_cold candidate=root0000 n=2\nACTION: tool=stepw")
    state = _state(ask=ask, answer_truncated=lambda: True)
    plan = policy.plan(state)
    check("a cut-off re-ask is not discarded as a draft",
          len(plan.actions) == 1 and plan.actions[0].tool == "det_cold",
          str([(a.tool, a.n) for a in plan.actions]))

    # Parsing RUNS but yields nothing, in both forms where it runs. A fragment cut
    # mid-word is not put into the transcript: the "your answer" caption would declare
    # a turn what did not become one, and the fragment would travel into every next
    # prompt, the same class as the leaking reasoning, only smaller.
    closed = "<think>weighing c1 against c2</think>\n"
    ask = _Answers(closed + "hmm, the best move here would be to", good)
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1)
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("an answer cut off after reasoning with nothing to take goes to a re-ask",
          len(ask.calls) == 2 and plan.actions[0].tool == "det_cold",
          str([(a.tool, a.n) for a in plan.actions]))
    check("and the fragment did not reach the transcript",
          "best move here" not in said and "your answer hit the answer cap" in said, said[:200])
    check("and it gets no advice about reasoning, its cut-off is different",
          "not with reasoning" not in said, said[:200])

    # There is no visible part at all: the cut fell exactly on the closing tag.
    # Nothing used to be said about such a turn, and the assistant read it as a turn
    # that never happened.
    ask = _Answers(closed, good)
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1)
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("an empty visible part is also a cut-off, and it is reported",
          "your answer hit the answer cap" in said and plan.actions[0].tool == "det_cold", said[:160])

    # The other side of the same distinction: a cut AFTER closed reasoning is a
    # different event. The visible part is the answer, though incomplete, the reasoning
    # can be stripped, and there is nothing to re-ask. Merged into one case, they
    # would make us pay for a re-ask for an answer already received.
    late = ("<think>c1 is the only watertight branch</think>\n"
            "WHY: extending the only watertight branch\n"
            "ACTION: tool=stepwise candidate=root0000 n=1\n"
            "ACTION: tool=det_cold candi")
    ask = _Answers(late)
    state = _state(ask=ask, answer_truncated=lambda: True)
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("a cut-off after closed reasoning is parsed as an ordinary answer",
          len(plan.actions) == 1 and plan.actions[0].tool == "stepwise",
          str([(a.tool, a.n) for a in plan.actions]))
    check("and there is no re-ask", len(ask.calls) == 1, str(len(ask.calls)))
    check("the plan went into the transcript, the reasoning did not",
          "WHY: extending" in said and "c1 is the only" not in said, said[-300:])
    check("an unfinished action did not become an action",
          "det_cold" not in str([a.tool for a in plan.actions]),
          str([a.tool for a in plan.actions]))
    state = _state(ask=_Answers(cut, cut))
    plan = policy.plan(state)
    check("a harness without the signal does not break parsing",
          len(plan.actions) == 1 and "not recognized" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))

    # A re-ask without reasoning. A noticeable share of answers hit the cap without
    # closing the reasoning: they cost a full cap and gave no decision at all, and the
    # turn was made by the harness fallback. Cured by the answer MODE, not by the cap:
    # lowering `CHOOSE_MAX_TOKENS` was examined and rejected.
    good = "ACTION: tool=stepwise candidate=root0000 n=2\nSHOW: none"
    ask = _Answers(cut, good)
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1)
    plan = policy.plan(state)
    check("a cut-off ends in a re-ask, not a fallback", len(ask.calls) == 2, str(len(ask.calls)))
    check("the re-ask goes WITHOUT reasoning",
          ask.calls[-1].get("thinking") is False, str(ask.calls[-1]))
    check("the re-ask has its own cap, not the decision cap",
          ask.calls[-1]["max_tokens"] == policy.RETRY_MAX_TOKENS
          and ask.calls[0]["max_tokens"] == policy.CHOOSE_MAX_TOKENS,
          str([c["max_tokens"] for c in ask.calls]))
    check("the first question does NOT override the mode: it is a run condition",
          "thinking" not in ask.calls[0], str(ask.calls[0]))
    # A decision cap set by the run (`experiment.agent.answer_max_tokens`) replaces
    # the constant for the question, but not for the re-ask without reasoning.
    ask = _Answers(cut, good)
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1,
                   agent_answer_max_tokens=8192)
    policy.plan(state)
    check("the decision cap comes from the run, the re-ask has its own",
          [c["max_tokens"] for c in ask.calls] == [8192, policy.RETRY_MAX_TOKENS],
          str([c["max_tokens"] for c in ask.calls]))
    check("the assistant made the turn, and that is a separate outcome",
          len(plan.actions) == 1 and "re-ask" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))
    check("the one named by the assistant is chosen, not the first in the list",
          plan.actions[0].tool == "stepwise" and plan.actions[0].n == 2,
          str((plan.actions[0].tool, plan.actions[0].n)))
    said = "\n".join(state.memory["dialogue"].lines)
    check("cut-off reasoning still did not reach the transcript", cut not in said, said[-200:])

    # A run WITHOUT reasoning (`experiment.agent.thinking: false`). The mode reaches
    # the policy through the seam (`SearchState.agent_thinking`), and without it the
    # whole truncation behaviour would be inverted: there is never a closing tag
    # there by nature, so EVERY cut would read as unfinished reasoning.
    #
    # Here a truncated answer is a cut of the ANSWER itself, and must be parsed as
    # usual: the action in it is real, not a draft in the middle of a thought.
    cut_with_action = (
        "WHY: extending the only watertight branch\n"
        "ACTION: tool=det_cold candidate=root0000 n=2\n"
        "ACTION: tool=stepw"
    )
    ask = _Answers(cut_with_action)
    state = _state(ask=ask, answer_truncated=lambda: True, agent_thinking=False)
    plan = policy.plan(state)
    # `n` of det_cold is the policy's (`FIXED_N`, cut to the row's max_n), not the answer's.
    check("on a non-thinking run a cut-off answer is parsed, not discarded",
          len(ask.calls) == 1 and len(plan.actions) == 1
          and plan.actions[0].tool == "det_cold",
          str([(a.tool, a.n) for a in plan.actions]) + str(len(ask.calls)))
    said = "\n".join(state.memory["dialogue"].lines)
    check("and the assistant gets no advice about reasoning that did not happen",
          "not with reasoning" not in said, said[-300:])

    # On a thinking run the same answer is a draft in the middle of a thought and is
    # not executed. The same string, two readings: the MODE tells them apart, not the text.
    ask = _Answers(cut_with_action, good)
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1,
                   agent_thinking=True)
    plan = policy.plan(state)
    check("the same answer on a thinking run is a draft and a re-ask",
          len(ask.calls) == 2 and plan.actions[0].tool == "stepwise",
          str([(a.tool, a.n) for a in plan.actions]) + str(len(ask.calls)))

    # A re-ask on a non-thinking run does not change the mode, so it has no right to
    # cut the cap: the same question in the same mode with a halved cap is certainly
    # worse than the first attempt.
    nothing = "let me think about which branch deserves the next step, hmm"
    ask = _Answers(nothing, good)
    state = _state(ask=ask, answer_truncated=lambda: len(ask.calls) == 1,
                   agent_thinking=False)
    plan = policy.plan(state)
    check("a re-ask on a non-thinking run goes with the FULL cap",
          len(ask.calls) == 2 and ask.calls[-1]["max_tokens"] == policy.CHOOSE_MAX_TOKENS,
          str([c["max_tokens"] for c in ask.calls]))
    check("and it does not override the mode, nothing to change",
          ask.calls[-1].get("thinking") is None, str(ask.calls[-1]))
    said = "\n".join(state.memory["dialogue"].lines)
    check("the mechanism that fired is named: the cap, not the reasoning",
          "hit the answer cap" in said and "without reasoning" not in said, said[-300:])

    # There is one re-ask: a cut that repeats even without reasoning is no longer
    # "the cap is too small", and a third question would be a payment without hope.
    ask = _Answers(cut, cut)
    state = _state(ask=ask, answer_truncated=lambda: True)
    plan = policy.plan(state)
    check("exactly one re-ask", len(ask.calls) == 1 + policy.MAX_RETRIES_PER_TURN,
          str(len(ask.calls)))
    check("after a failed re-ask, the previous cut-off outcome",
          len(plan.actions) == 1 and "cut off before the named action" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason if plan.actions else None))

    # An answer that fit but was not understood is re-asked with a CHANGED question,
    # and the mode is not touched: there was no cut, so nothing to change. The slot is
    # the same: there is no second re-ask per turn.
    ask = _Answers(cut, cut)
    state = _state(ask=ask, answer_truncated=lambda: False)
    plan = policy.plan(state)
    check("a complete but not understood answer is asked again exactly once",
          len(ask.calls) == 1 + policy.MAX_RETRIES_PER_TURN, str(len(ask.calls)))
    check("and the re-ask does not change the mode or the cap",
          ask.calls[-1]["max_tokens"] == policy.CHOOSE_MAX_TOKENS
          and "thinking" not in ask.calls[-1], str(ask.calls[-1]))
    check("misunderstood on the re-ask too: the previous 'not recognized' outcome",
          plan.actions and plan.actions[0].reason == policy.OUTCOMES["unparsed"][0],
          str(plan.actions and plan.actions[0].reason))

    print("\nCandidate table")
    state = _broken_state(_Answers("ACTION: tool=stepwise candidate=aaaa1111 n=2\nSHOW: none"))
    menu = policy._menu(state)
    text = policy._candidates_text(state, menu)
    best_row = next(line for line in text.splitlines() if line.startswith("aaaa1111"))
    check("an invalid candidate is NOT in the table", "bbbb2222" not in text, text)
    check("both metrics are shown", "0.500" in best_row and " - " in best_row, text)
    check("the best is marked", "best" in best_row.split(), text)

    # Broken candidates are not shown. A consequence that matters more than the rule
    # itself: `repair` becomes UNREACHABLE by construction, since it is legal only on
    # invalid candidates and the menu is built from what is shown. Checked on a state
    # where the repair action is LEGAL: otherwise the check would be green just
    # because the tool is absent from the run's set, and whoever returned it to
    # `experiment.tools` would get `empty_generation` on every choice (the policy no
    # longer has a repair prompt).
    repairable = _broken_state(_Answers("SHOW: none"), tools=("stepwise", "repair"))
    repairable.legal = list(repairable.legal) + [
        LegalAction(tool="repair", parent_id="bbbb2222", max_n=2)]
    check("a legal repair is not in the menu, since broken candidates are not shown",
          all(a.tool != "repair" for a in policy._menu(repairable)),
          str([(a.tool, a.parent_id) for a in policy._menu(repairable)]))
    check("the columns are in place",
          all(col in text for col in ("ops", "score", "iou", "gms", "first", "tools")), text)
    check("the answer scale is printed as a number, not only as its halves",
          "0.5000" in text, text)

    # The menu is ranked by the HARNESS scale. Here `gms_norm` inverts the order
    # relative to `iou`: by the policy's objective `cccc3333` would be first, but the
    # part is returned as `aaaa1111`, and the table must show the latter.
    ranked = _state(
        ask=_Answers("SHOW: none"),
        pool=[_candidate("aaaa1111", metrics={"iou": 0.50, "gms_norm": 0.90}),
              _candidate("cccc3333", metrics={"iou": 0.60, "gms_norm": 0.10})],
        legal=[LegalAction(tool="stepwise", parent_id="aaaa1111", max_n=4),
               LegalAction(tool="stepwise", parent_id="cccc3333", max_n=4)],
    )
    ranked.best_id = "aaaa1111"
    order = [c.id for c in policy._shown_candidates(ranked)]
    check("the menu is ordered by what the part is returned by, not by the policy objective",
          order[0] == "aaaa1111", str(order))

    # The root is a reservation: by quality it is always last, yet it is the only
    # action able to change the first operation.
    deep = _state(
        ask=_Answers("SHOW: none"),
        pool=[_candidate("c%04d" % i, metrics={"iou": 0.5 + i / 1000}) for i in range(14)],
        legal=([LegalAction(tool="stepwise", parent_id="root0000", max_n=4)]
               + [LegalAction(tool="stepwise", parent_id="c%04d" % i, max_n=4)
                  for i in range(14)]),
    )
    deep.best_id = "c0013"
    shown = {c.id for c in policy._shown_candidates(deep)}
    check("the root is not pushed out of the table by a growing pool", "root0000" in shown,
          str(sorted(shown)))
    deep_text = policy._candidates_text(deep, policy._menu(deep))
    check("and it is named for what it is, a fresh start",
          any(line.startswith("root0000") and "start" in line.split()
              for line in deep_text.splitlines()), deep_text)

    # The best is the second reservation: without a legal action it used to drop out
    # of the list together with its mark.
    stuck = _state(
        ask=_Answers("SHOW: none"),
        pool=[_candidate("aaaa1111", metrics={"iou": 0.9}),
              _candidate("cccc3333", metrics={"iou": 0.2})],
        legal=[LegalAction(tool="stepwise", parent_id="cccc3333", max_n=4)],
    )
    stuck.best_id = "aaaa1111"
    stuck_text = policy._candidates_text(stuck, policy._menu(stuck))
    check("the best is shown even without a legal action on it",
          any(line.startswith("aaaa1111") and "best" in line.split()
              for line in stuck_text.splitlines()), stuck_text)
    check("and it is visible that it has no actions",
          any(line.startswith("aaaa1111") and line.rstrip().endswith("-")
              for line in stuck_text.splitlines()), stuck_text)

    # The price of a sample tool has two points: a single point at `n=1` described the
    # price wrongly where the assistant decides `n`.
    legend = policy._legend(state)
    stepwise_row = next(line for line in legend.splitlines() if line.startswith("- stepwise"))
    point = policy.COST_POINT_N
    check("the stepwise legend has the meaning of `n` and the cost at two points",
          "n = how many samples" in stepwise_row and "at n=1" in stepwise_row
          and f"≈{state.tools['stepwise'].wall(point):.1f}s at n={point}" in stepwise_row,
          stepwise_row)
    check("the legend has no knobs", "temperature" not in legend, legend)
    action = policy.plan(state).actions[0]
    check("the assistant's choice arrived as an action",
          action.tool == "stepwise" and action.parent_id == "aaaa1111",
          str((action.tool, action.parent_id)))

    many = [LegalAction(tool="stepwise", parent_id=f"cand{index:04d}", max_n=4)
            for index in range(30)]
    state = _state(legal=many,
                   pool=[_candidate(f"cand{index:04d}", metrics={"iou": index / 100})
                         for index in range(30)],
                   ask=_Answers(""))
    menu = policy._menu(state)
    shown = {action.parent_id for action in menu}
    text = policy._candidates_text(state, menu)
    rows = re.findall(r"^(cand\d{4})\s", text, re.M)
    check("the table is truncated", len(rows) <= policy.MAX_VALID_SHOWN, str(len(rows)))
    check("what is shown and what is accepted are one list", set(rows) == {c[:8] for c in shown},
          f"{len(rows)} rows vs {len(shown)} actions")
    check("it says that not all are shown", "more candidates are not shown" in text, text)
    check("the best are shown, not the first that come to hand", "cand0029" in text and "cand0000" not in text)

    print("\nWhat is named but not offered")
    state = _broken_state(_Answers(
        "ACTION: tool=stepwise candidate=aaaa1111 n=2\n"
        "ACTION: tool=optimize candidate=zzzz9999 n=1\nSHOW: none"))
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("the legal one is executed", [a.tool for a in plan.actions] == ["stepwise"],
          str([a.tool for a in plan.actions]))
    check("the illegal one is named to the assistant", "not on the list" in said and "zzzz9999" in said, said)

    state = _broken_state(_Answers(
        "\n".join([f"ACTION: tool=stepwise candidate=aaaa1111 n=1"] * 6) + "\nSHOW: none"))
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("no more actions per turn than the cap",
          len(plan.actions) <= policy.MAX_ACTIONS_PER_TURN, str(len(plan.actions)))
    check("the extras are reported", "over the limit" in said, said)

    print("\nA turn without an assistant answer")
    for name, ask in (("channel not connected", None),
                      ("call failed", _Answers(raises=RuntimeError("server down"))),
                      ("answer outside the format", _Answers("I do not know"))):
        state = _state(ask=ask)
        plan = policy.plan(state)
        session = state.memory["dialogue"]
        check(f"{name}: the turn took place", len(plan.actions) == 1,
              str([a.tool for a in plan.actions]))
        check(f"{name}: the reason is recorded in the journal",
              bool(plan.actions and plan.actions[0].reason), str(plan.actions))
        check(f"{name}: and in the dialogue transcript",
              any("executed" in line for line in session.lines), str(session.lines))

    print("\nThe harness cap cannot be bypassed")
    # Only `plan`: `select()` does not call the assistant, so the cap has nowhere to
    # come from there. Left in the list, it would test not the policy but the fact
    # that the stub can raise an exception.
    for block in ("plan",):
        state = _state(ask=_Answers(raises=BudgetExceeded("agent_text")),
                       fresh=[_candidate("aaaa1111", metrics={"iou": 0.5})])
        try:
            getattr(policy, block)(state)
            raised = False
        except BudgetExceeded:
            raised = True
        check(f"{block}: BudgetExceeded is propagated outward", raised)

    print("\nA choice the remaining budget cannot cover")
    # The generator is exhausted by its counter, the deterministic branch is not.
    state = _state(ask=_Answers("ACTION: tool=stepwise candidate=root0000 n=4\nSHOW: none"),
                   remaining=Remaining(calls={"vlm": 0}))
    plan = policy.plan(state)
    check("the first affordable action is taken",
          len(plan.actions) == 1 and plan.actions[0].tool == "det_cold",
          str([(a.tool, a.n) for a in plan.actions]))
    check("the substitution is named to the assistant",
          any("not affordable" in line for line in state.memory["dialogue"].lines),
          str(state.memory["dialogue"].lines))

    state = _state(ask=_Answers("ACTION: tool=stepwise candidate=root0000 n=4\nSHOW: none"),
                   remaining=Remaining(calls={"vlm": 0, "det": 0}))
    check("nothing is affordable: an empty plan", not policy.plan(state).actions)

    state = _state(ask=_Answers(
        "ACTION: tool=stepwise candidate=root0000 n=4\n"
        "ACTION: tool=det_cold candidate=root0000 n=3\nSHOW: none"),
        remaining=Remaining(calls={"vlm": 4, "det": 4}))
    plan = policy.plan(state)
    check("the remainder is shared among the turn's actions, not given in full to each",
          sum(a.n for a in plan.actions if a.tool == "stepwise") <= 2,
          str([(a.tool, a.n) for a in plan.actions]))

    print("\nAction knobs")
    state = _state(ask=_Answers(
        "ACTION: tool=stepwise candidate=root0000 n=2 params=temperature=1.4,bogus=7\nSHOW: none"))
    action = policy.plan(state).actions[0]
    check("named knobs do not reach the action, it goes with the defaults",
          action.params == {} and action.n == 2, str(action))

    print("\nTurn outcome report")
    ask = _Answers("ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none",
                   "CONTINUE",
                   "ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none")
    state = _state(ask=ask)
    policy.plan(state)
    state.fresh = []
    state.attempts = {"root0000": [Attempt(origin=Origin(tool="stepwise"),
                                           failure="tool_timeout")]}
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("a fruitless turn is not silent", "no new candidates" in said, said[-200:])
    check("and names the reason by name", "stepwise on root0000: tool_timeout" in said, said[-200:])

    state = _state(ask=_Answers("ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none",
                                "ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none"))
    policy.plan(state)
    state.fresh = [
        _candidate("aaaa1111", metrics={"iou": 0.42, "gms_norm": 0.55}),
        _candidate("dddd4444", metrics={"iou": 0.40, "gms_norm": 0.51}),
        _candidate("bbbb2222", failure="not_watertight", metrics={"iou": 0.9}),
        _candidate("cccc3333", built=False, failure="exec_error"),
    ]
    state.pool.extend(state.fresh)
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("the report has both metrics", "iou=0.420 gms=0.550" in said, said)
    check("an invalid candidate is called invalid and has no numbers",
          "built, but invalid (not_watertight)" in said
          and "built, but invalid (not_watertight);" not in said, said)
    check("one that did not build is named separately", "did not build (exec_error)" in said)
    check("the best of the turn is marked exactly once", said.count("best so far") == 1, said)

    # No valid candidates: the harness keeps an open (non-watertight) one as the best
    # for the failure diagnosis. It does not become a selection candidate because of
    # that. It has numbers here on purpose: the gate is by the failure, not by the
    # absence of a measurement.
    invalid = _candidate("eeee5555", failure="not_watertight",
                         metrics={"iou": 0.99, "gms_norm": 0.99})
    state.pool.append(invalid)
    state.best_id = invalid.id
    check("an invalid best has no \"Best so far\" line", policy._best_line(state) == "",
          policy._best_line(state))
    check("and is not in the choice table",
          invalid.id not in {c.id for c in policy._shown_candidates(state)})

    print("\nThe part scale falls back")
    state = _state(ask=_Answers("ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none",
                                "ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none"))
    # The root has no measurement and the candidate has only GMS: exactly how a part
    # with a non-watertight GT looks, where nobody will have a volumetric IoU.
    state.pool[0].metrics = None
    policy.plan(state)
    state.fresh = [_candidate("aaaa1111", metrics={"gms_norm": 0.71})]
    state.pool.extend(state.fresh)
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("without IoU the report uses the fallback scale, not 'no metric'",
          "gms=0.710" in said, said)

    print("\nInspection is a separate answer, not an addendum to the turn")
    render = _Renderer()
    ask = _Answers(
        "SHOW: bbbb2222",                                        # look only
        "ACTION: tool=stepwise candidate=aaaa1111 n=1\nSHOW: none",
        "CONTINUE")
    state = _broken_state(ask, render=render)
    plan = policy.plan(state)
    check("inspection does not spend an iteration: the turn still took place",
          len(plan.actions) == 1 and plan.actions[0].tool == "stepwise",
          str([a.tool for a in plan.actions]))
    check("the first question carries only the automatic panels",
          ask.images[0] is not None and render.calls[0] == ["aaaa1111"], str(render.calls))
    check("the requested one is drawn", render.calls[1:] == [["bbbb2222"]], str(render.calls))
    check("the same question is asked again, now with an image",
          len(ask.prompts) == 2 and ask.images[1] is not None)
    check("images are sent as a list, not as a collage",
          isinstance(ask.images[1], list) and len(ask.images[1]) == 2, str(ask.images[1]))
    check("the image order is named in the prompt",
          "image 1 = the TARGET shape" in ask.prompts[1]
          and "image 2 = bbbb2222" in ask.prompts[1], ask.prompts[1][-300:])
    check("the show is marked in the transcript",
          any("shown rendered images of bbbb2222" in line
              for line in state.memory["dialogue"].lines))

    render = _Renderer()
    ask = _Answers(
        "ACTION: tool=stepwise candidate=aaaa1111 n=1\nSHOW: bbbb2222",
        "ACTION: tool=stepwise candidate=aaaa1111 n=1\nSHOW: none")
    state = _broken_state(ask, render=render)
    policy.plan(state)
    check("a request made together with a turn waits for the next choice instead of being wasted",
          len(ask.images) == 1 and render.calls == [["aaaa1111"]], str(render.calls))
    policy.plan(state)
    check("and arrives at the next choice, after the automatic panels",
          render.calls[-1] == ["aaaa1111", "bbbb2222"], str(render.calls))

    ask = _Answers("SHOW: bbbb2222", "SHOW: aaaa1111", "SHOW: bbbb2222")
    state = _broken_state(ask, render=_Renderer())
    plan = policy.plan(state)
    check("one cannot look forever: the turn took place",
          len(plan.actions) == 1, str([a.tool for a in plan.actions]))
    check("and that is a separate outcome in the journal",
          "named no action" in (plan.actions[0].reason or ""), str(plan.actions[0].reason))
    check("no more consecutive inspections than the cap",
          len(ask.prompts) == policy.MAX_LOOKS_PER_TURN + 1, str(len(ask.prompts)))

    print("\nThe target can be viewed separately")
    target = _TargetRenderer()
    render = _Renderer()
    ask = _Answers("SHOW: target", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render, render_target=target)
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("the target is drawn", target.calls == 1 and ask.images[1] is not None,
          f"target render calls: {target.calls}")
    check("no candidate collage was built for it, only the automatic one",
          render.calls == [["aaaa1111"]], str(render.calls))
    check("the prompt says this is the target",
          policy.IMAGE_NOTE_TARGET in ask.prompts[1], ask.prompts[1][-300:])
    check("the target show is marked in the transcript", "you were shown the target shape" in said, said)

    # First turn of a part: there are no candidates with a mesh at all; there used to
    # be nothing to look at, and the first tool was chosen blindly.
    target = _TargetRenderer()
    ask = _Answers("SHOW: target", "ACTION: tool=stepwise candidate=root0000 n=1")
    state = _state(ask=ask, render=_Renderer(), render_target=target,
                   pool=[_candidate("aaaa1111", built=False, failure="exec_error")])
    state.pool[0].mesh_path = None
    policy.plan(state)
    check("the target is visible even when there is nothing more to draw", target.calls == 1,
          f"target render calls: {target.calls}")

    target = _TargetRenderer()
    render = _Renderer()
    ask = _Answers("SHOW: target, aaaa1111", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render, render_target=target)
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("with a candidate the target is not drawn as a separate image",
          target.calls == 0 and render.calls[-1] == ["aaaa1111"],
          f"{target.calls}, {render.calls}")
    check("and that is reported", "always shown as a panel next to the candidates" in said, said)

    ask = _Answers("SHOW: target", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=_Renderer(), render_target=None)
    policy.plan(state)
    check("no target render: reported, not silent",
          "target: could not be drawn" in "\n".join(state.memory["dialogue"].lines))

    target = _TargetRenderer()
    ask = _Answers("SHOW: target", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=_Renderer(), render_target=target,
                          remaining=Remaining(calls={"agent_visual": 0}))
    policy.plan(state)
    check("the target is not drawn when the visual call is not paid for",
          target.calls == 0 and all(image is None for image in ask.images),
          f"{target.calls}, {ask.images}")

    print("\nEndpoint image cap")
    # The panel count is a policy constant, while the `--limit-mm-per-prompt` cap is a
    # run condition. The server rejects a request over the cap WHOLE, so a policy that
    # raised the constant would get "the assistant did not answer" on every turn with
    # an image: a diagnosis of a dead channel instead of its own knob.
    render = _Renderer()
    ask = _Answers("SHOW: aaaa1111, bbbb2222, cccc3333",
                   "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render, agent_max_images=3)
    state.pool.append(_candidate("cccc3333", metrics={"iou": 0.4}))
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("panels are cut by the endpoint cap, not all sent",
          render.calls[-1] == ["aaaa1111", "bbbb2222"], str(render.calls))
    check("exactly the reason that triggered is named",
          "the assistant endpoint takes 3 images per request" in said, said)
    check("and the turn still took place", len(ask.prompts) == 2, str(len(ask.prompts)))

    # The policy's own constant and a foreign cap are different causes and must not be
    # confused: the first is cured by editing the policy, the second only by the run config.
    render = _Renderer()
    ask = _Answers("SHOW: aaaa1111, bbbb2222, cccc3333, dddd4444, eeee5555",
                   "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render)
    for cid in ("cccc3333", "dddd4444", "eeee5555"):
        state.pool.append(_candidate(cid, metrics={"iou": 0.4}))
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("without an endpoint cap the own constant cuts",
          len(render.calls[-1]) == policy.MAX_IMAGE_PANELS, str(render.calls))
    check("and the endpoint is not mentioned at all",
          "endpoint takes" not in said, said)

    # A cap of one image: no candidate panels, but the TARGET can still be shown as a
    # separate image; these are different cases.
    target = _TargetRenderer()
    render = _Renderer()
    ask = _Answers("SHOW: target, aaaa1111",
                   "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render, render_target=target, agent_max_images=1)
    policy.plan(state)
    check("with a cap of one image the target is still shown",
          target.calls == 1 and render.calls == [], f"{target.calls}, {render.calls}")

    target = _TargetRenderer()
    render = _Renderer()
    ask = _Answers("SHOW: target, aaaa1111",
                   "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=render, render_target=target, agent_max_images=0)
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("with a zero cap nothing is drawn and the reason is given",
          target.calls == 0 and render.calls == []
          and "takes no images in this run" in said, f"{target.calls}, {render.calls}")

    print("\nWhen there is nothing to show")
    ask = _Answers("SHOW: zzzz9999", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    render = _Renderer()
    state = _broken_state(ask, render=render)
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("a nonexistent candidate is not drawn", render.calls == [["aaaa1111"]], str(render.calls))
    check("and we say why", "zzzz9999: no such candidate" in said, said)

    ask = _Answers("SHOW: cccc3333", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    render = _Renderer()
    state = _broken_state(ask, render=render)
    state.pool.append(_candidate("cccc3333", built=False, failure="exec_error"))
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("one that did not build is not given to the renderer at all",
          render.calls == [["aaaa1111"]], str(render.calls))
    check("and we say why, in English, in its own words",
          "cccc3333: nothing to draw, it did not build" in said, said)
    check("and the question is asked anyway", len(ask.prompts) == 2, str(len(ask.prompts)))

    ask = _Answers("SHOW: bbbb2222", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=_Renderer(refuse=True))
    policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("a failure of the renderer itself is explained too",
          "bbbb2222: could not be drawn" in said, said)

    ask = _Answers("SHOW: bbbb2222", "ACTION: tool=stepwise candidate=aaaa1111 n=1")
    state = _broken_state(ask, render=_Renderer(),
                          remaining=Remaining(calls={"agent_visual": 0}))
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("an exhausted visual channel does not end the part",
          len(plan.actions) == 1, str(plan.actions))
    check("the image is not requested from the server when it is not affordable",
          all(image is None for image in ask.images), str(ask.images))
    check("and we tell the assistant", "no visual calls left" in said, said)

    state = _broken_state(
        _Answers("SHOW: bbbb2222", "ACTION: tool=stepwise candidate=aaaa1111 n=1"),
        render=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no renderer")))
    plan = policy.plan(state)
    check("a failed render does not fail the turn", len(plan.actions) == 1, str(plan.actions))
    check("and is named as the reason",
          "rendering failed" in "\n".join(state.memory["dialogue"].lines))

    print("\nFinishing a part is decided in plan(), and the policy does not need `select`")
    fresh = [_candidate("aaaa1111", metrics={"iou": 0.5})]

    # The main claim: `select` is NOT DECLARED on the policy. It did not ask the
    # assistant, and `keep` only filled a log column: the menu is built from
    # `state.legal`, and this policy does not call `frontier()`. Left in place, it
    # would be dead code.
    check("the dialogue has no `select` at all", not hasattr(policy, "select"),
          str(getattr(policy, "select", None)))
    check("`entrypoint()` still returns a live object with `plan`",
          callable(getattr(policy, "plan", None)))

    state = _state(ask=_Answers("ACTION: tool=stepwise candidate=aaaa1111 n=1"),
                   fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("and without DONE the part continues", not plan.done, str(plan))

    # DONE together with actions: the turn is executed, the part ends after it. The
    # candidate is from the MENU (`root0000`): one named outside the menu is discarded,
    # and the check would silently degrade into the previous case, "DONE alone".
    state = _state(ask=_Answers(
        "ACTION: tool=stepwise candidate=root0000 n=1\nDONE: only rounding left"),
        fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("DONE with actions does not cancel the turn", len(plan.actions) == 1, str(plan.actions))
    check("and marks the plan as the last", plan.done and "rounding" in plan.reason, plan.reason)

    # DONE alone: the plan is empty, but this is NOT `plan_empty`; the flag is explicit.
    state = _state(ask=_Answers("DONE: the shape matches the target"), fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("a lone DONE ends the part with a flag, not an empty plan",
          not plan.actions and plan.done and "matches" in plan.reason, str(plan))
    check("the finish is recorded in the transcript",
          any("finished" in line for line in state.memory["dialogue"].lines),
          str(state.memory["dialogue"].lines))
    check("the assistant is asked exactly once about a finish",
          len(state.ask.prompts) == 1, str(len(state.ask.prompts)))
    check("a finish without the word `stalled` is a quality judgement", not plan.stalled, str(plan))

    # The second door: the word `stalled` at the start of the reason.
    for answer in ("DONE: stalled — last attempts did not improve the best",
                   "DONE: STALLED: nothing new", "DONE: **stall** no better move"):
        state = _state(ask=_Answers(answer), fresh=fresh, pool=fresh)
        plan = policy.plan(state)
        check(f"giving up is flagged ({answer[6:20]!r})",
              not plan.actions and plan.done and plan.stalled, str(plan))
    for answer in ("DONE: shape — only rounding left", "DONE: not stalled, shape is right"):
        state = _state(ask=_Answers(answer), fresh=fresh, pool=fresh)
        plan = policy.plan(state)
        check(f"`stalled` not at the start of the reason is not giving up ({answer[6:20]!r})",
              plan.done and not plan.stalled, str(plan))
    state = _state(ask=_Answers(
        "ACTION: tool=stepwise candidate=root0000 n=1\nDONE: stalled, one last try"),
        fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("giving up with actions: the turn executes, the flag travels",
          len(plan.actions) == 1 and plan.done and plan.stalled, str(plan))

    # An empty reason does not cancel the decision.
    state = _state(ask=_Answers("DONE:"), fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("a finish without a reason still finishes",
          plan.done and plan.reason == policy.DONE_WITHOUT_REASON, plan.reason)

    # No valid candidate: there is nothing to finish with, but the decision is not lost.
    #
    # The guard lives in `plan()`, and this is not a reshuffle: in `select()` it never
    # fired. The harness reads `Plan.done` itself (`plan_done`) and finishes the part
    # after the turn, and on the no-action path it leaves the loop before any verdict,
    # so the check would stand AFTER the one who decides.
    state = _state(ask=_Answers("DONE: looks right",
                                "ACTION: tool=stepwise candidate=root0000 n=1"),
                   fresh=fresh, pool=fresh)
    state.best_id = None
    plan = policy.plan(state)
    said = "\n".join(state.memory["dialogue"].lines)
    check("without a valid candidate the part does not end", not plan.done, str(plan))
    check("and the turn still took place instead of becoming an empty plan",
          len(plan.actions) == 1, str(plan.actions))
    check("and this is a separate outcome, not 'finished' and not 'not recognized'",
          "before the first valid" in (plan.actions[0].reason or ""),
          str(plan.actions[0].reason))
    check("the assistant is told why finishing failed",
          "nothing to finish with" in said, said[-300:])

    # The decision is not lost: it belongs to the PART, not the turn, and the first
    # valid candidate carries it out without re-asking the assistant.
    state.best_id = "aaaa1111"
    state.fresh = []
    plan = policy.plan(state)
    check("and the decision is not lost: it fires on the first valid one",
          plan.done and "looks right" in (plan.reason or ""), str(plan))

    # DONE plus a candidate outside the menu: the action is discarded, the part is
    # finished anyway, and the discarded action is mentioned.
    state = _state(ask=_Answers(
        "ACTION: tool=stepwise candidate=nosuchid n=1\nDONE: nothing left to add"),
        fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("a candidate that was not offered does not prevent finishing",
          not plan.actions and plan.done, str(plan))
    check("and stays named in the transcript",
          any("not on the list" in line for line in state.memory["dialogue"].lines),
          str(state.memory["dialogue"].lines))

    # There must be no fallback to the first action on DONE.
    state = _state(ask=_Answers("DONE: done here"), fresh=fresh, pool=fresh)
    plan = policy.plan(state)
    check("a finish is not read as an unparsed answer",
          not plan.actions, str([a.tool for a in plan.actions]))

    print("\nContext overflow")
    class _OnceTooLarge:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompt, image=None, max_tokens=16, **kwargs):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise PromptTooLarge("did not fit")
            return "ACTION: tool=stepwise candidate=root0000 n=1\nSHOW: none"

    ask = _OnceTooLarge()
    state = _state(ask=ask)
    policy.plan(state)
    for _ in range(3):
        state.fresh = []
        policy.plan(state)
    check("after an overflow the question is asked again", len(ask.prompts) >= 2)
    check("the history is trimmed", any("were dropped" in line
                                 for line in state.memory["dialogue"].lines))

    trace_fixes(policy)
    auto_look_checks()
    function_call_checks()
    llm_slot_checks()

    print("\nFAILURES (%d): %s" % (len(FAILURES), "; ".join(FAILURES)) if FAILURES
          else "\nAll dialogue policy checks passed")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
