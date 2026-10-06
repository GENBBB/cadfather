"""`dialogue_lean`: the assistant drives the search by function calls, one turn at a time.

EVERY turn is a decision of the assistant: `plan()` asks what to continue with AND whether
it is time to finish. The conversation session grows in `state.memory`. The conversation
mechanics (calling the assistant, parsing reasoning tags, assembling images, number
formatting) live in `scaffold/dialogue_io.py`; function-call parsing in
`scaffold/tool_calls_io.py`. What stays here is everything with a real choice: prompt texts,
what the table shows and in which order, which images to show, how to parse the answer,
turn caps and what to do when there is no answer.

`state.ask` (`Resources.ask_agent`) is a one-shot flat call: one text plus the images of
this question. The "session" therefore lives entirely in the prompt text, which is rebuilt
on every call: history is text, not server state.

**A turn.** The assistant gets a table of candidates and calls two functions: `act` (one
tool on one candidate, up to `MAX_ACTIONS_PER_TURN` per turn) and `finish` (end the part,
with the reason `shape` or `stalled`). The functions are constant for the run: tool names
in the schema are all tools of the run, not the turn's menu, and the candidate is a string,
not an enum. The chat template puts the functions at the VERY START of the prompt, and a
schema that changed from turn to turn would break the prefix cache from the first token.
Whether an action is legal is checked against the menu (`dialogue_io.legal_picks`).
`tool_choice="auto"`, not `required`: the server enforces `required` with constrained JSON
decoding rather than the model's native format; an answer without a call is caught by a
re-ask.

`n` is chosen by the assistant only for sample tools (`stepwise`); for a ranked list it is
fixed (`FIXED_N`), and tool knobs (`params`) are not offered: actions go with the registry's
default knobs.

**Images.** Before the first question of every turn the harness shows, without being asked:

- until the first valid candidate with a mesh, the target alone;
- afterwards, the target, the best candidate, and the best of those born on the previous
  turn (`state.fresh`): feedback on the assistant's own move;
- if the previous turn produced the new best, two identical panels carry nothing, so the
  parent of the best is shown instead: the "before/after" of the last operation;
- if the previous turn produced no valid candidate, there is no third panel, and this is
  said in words.

Each panel is labeled with its role, the number of operations and its score; otherwise
the picture cannot be tied to a table row.

**Order for the assistant's prefix cache.** The server reuses what it computed for a shared
request prefix. The request is assembled as:

    function schema (chat template) -> rules, legend, tools ->
    target image -> turn history -> turn images -> table -> question

Everything before the turn images stays the same from turn to turn or only grows, so it is
cached. Images take their places via `IMAGE_SLOT` markers (`harness/search_types.py`); on a
mismatch between markers and images (an image failed to build, the visual budget is spent)
the request goes out with images first: more expensive but correct. The target image on
every turn is THE SAME as on the first (kept in the part's memory): a re-render that
differs by one pixel breaks the prefix before the history. Trimming the history on context
overflow (`TRIM_KEEP_LAST`) breaks the prefix too; this is rare.

Rules the turn logic rests on:

- **The dialogue must be coherent.** The transcript holds not only what the assistant said
  but what CAME OF it: which actions actually went to the harness (after truncating `n` to
  the remainder), what came back, which requests were dropped and why.
- **What is shown and what is accepted is one list.** The candidate table, answer
  validation and the fallback all come from ONE set of legal actions (`_menu`).
- **Addressing by names, not numbers.** A call names the tool and the candidate id, not a
  menu row: the menu is recomputed every turn.
- **No answer is not a turn failure.** The assistant is unavailable, the prompt did not
  fit, the answer was cut inside the reasoning: the part continues with a fallback action
  and records it in the log and the transcript. Only `BudgetExceeded` propagates. The causes
  are told apart in the log (`OUTCOMES`), and truncation among them is the harness's answer
  (`state.answer_truncated`), not a guess from the text.

What the policy does not have: **`select()`** (the menu is built from `state.legal`, and the
decision "the part is finished" travels in the `Plan.done` flag; the harness accepts the
absence of the method), and **`repair`** (broken candidates are not shown, so actions on them
never get into the menu by construction).

Thinking mode. Only the visible part of an answer goes into the transcript
(`dialogue_io.visible`): reasoning is a draft, and carrying it through all later prompts
would pay for it on every turn. Instead the assistant writes one sentence before its calls,
saying what it is going after; the reasoning itself is saved by the journal
(`FigureJournal.save_agent_call`) for a human. An answer cut by the cap INSIDE the reasoning
is not parsed at all (the draft is the middle of a thought, not its conclusion); the turn is
cured by re-asking without reasoning. A truncation AFTER a closed reasoning
(`dialogue_io.reasoning_closed`) is parsed as usual.

An answer without function calls is also read in the plain line format (`ACTION:`, `SHOW:`,
`DONE:`; `_parse_text`): the model sometimes answers that way, and it is a valid move.
"""

from __future__ import annotations

import re
from typing import Any

from cad_agent.capabilities import objective as objective_mod
from cad_agent.harness.search_types import (
    IMAGE_SLOT,
    Action,
    Candidate,
    LegalAction,
    Plan,
    SearchState,
    code_numbers,
)
from cad_agent.scaffold import dialogue_io, tool_calls_io


class DialogueLeanPolicy:
    """Function calls on a candidate table; the harness shows images on every turn."""

    # The policy needs a running assistant: it does nothing at all without one. Read by
    # the config validator (`cost_check`).
    needs_assistant = True
    # Read by the config check (`config._check_server`): the server must be started
    # with a tool-call parser, otherwise `auto` is rejected outright.
    TOOL_CHOICE = "auto"

    MEMORY_KEY = "dialogue"
    # Key of this turn's panel roles in `state.memory`: there is one policy per run and it
    # outlives a part, so a field of its own would leak between parts (the same argument as
    # for `dialogue_io.Session`).
    ROLES_KEY = "dialogue_autolook_roles"
    # Key of the stored target image in `state.memory` (see the module docstring).
    TARGET_KEY = "dialogue_lean_target"

    # --- prompts ---------------------------------------------------------------

    INTRO = """
You are reconstructing a 3D CAD target shape step by step. A candidate is a program of DSL operations; a tool either appends operations to a candidate or builds a new one from the empty start. You choose the calls, the harness runs them and reports what came out. Every candidate stays available for the whole part, so you may go back to any earlier one. Only candidates that built and are watertight are listed.

Score: the mean of iou and gms, whichever could be computed; higher is better. The candidate with the best score at the end is the answer for this part. iou and gms fail differently — a candidate can match the volume and miss the shape.

Each turn you get a table of candidates: id; from — the tool that made it; ops — operations in its code; score, iou, gms; first — the score of the first operation of its chain; tools — the only calls you can make on it now. A tool missing from a row is used up on that candidate or does not apply to it. A weak first operation caps what the whole chain can reach, and later operations rarely recover it; only the empty start gives a different first operation, and repeating stepwise on it draws new ones.

Tools:
{tools}

How to answer. First write one short sentence saying what you are going after this turn and why it is worth the calls: your reasoning is not carried over to later turns, this sentence is. Then call act once per call — up to {max_actions} calls in a turn, each on a candidate from the table with a tool from its tools column; n matters only for stepwise. Every turn uses up one of the turns this part has.

To end the part, call finish with one of two reasons:
- shape: the best candidate reproduces every geometric feature of the target and the only difference left is numerical — tolerance, rounding, tessellation — which no further operation can remove. Do NOT say shape while a whole feature is still missing or extra, however small the metric gap looks; do say it once the shape is right, even if the metric is short of 1.0.
- stalled: a feature is still wrong, but your recent attempts have stopped improving the best candidate and you see no different move worth its calls. This is an honest way to end the part, not a failure.
finish may come together with act calls; then this turn is your last one.

Each turn you are shown the target, the best candidate so far and the best candidate your last turn made. This is the TARGET shape:
""".strip()

    # The question must be the LAST paragraph of the prompt. It is recognized from outside by
    # it (the smoke stub looks at `prompt.rsplit("\n\n", 1)[-1]`), and a paragraph added after
    # it silently turns the question into an unrecognized one. Guarded by `dialogue_check`.
    CHOOSE_QUESTION = """
Candidates you can act on:

{candidates}

Your move: call act (up to {max_actions}) and/or finish.
""".strip()

    # What a tool does and what NOT to expect from it: only firm facts, not advice. Without
    # these lines the assistant did not know that `optimize` is deterministic and one-shot per
    # candidate, and repeated already executed pairs.
    TOOL_NOTES = {
        "stepwise": "appends one generated operation to the candidate; repeating the same call "
                    "draws fresh samples by itself",
        # "Fitted is not more accurate" is a measured fact: an earlier wording ("an algorithm,
        # not the model, reads ...") made a thinking assistant call `det_cold` first on almost
        # every part.
        "det_cold": "fits ONE first operation to the target mesh; the ranked list holds "
                    "alternatives for that operation, not the whole part. Fitted is not more "
                    "accurate than generated: on average its first operation scores below a "
                    "stepwise first operation, most of all on targets whose iou cannot be "
                    "computed, so it is a different start to try alongside stepwise on the empty "
                    "start, not a replacement for it. Its profiles are written out point by "
                    "point, so its code, and the code of every candidate built on it, is long; "
                    "only on the empty start",
        "det_warm": "algorithmic reconstruction starting from this candidate, returned as a "
                    "ranked list; the list is finite, and once used up the tool leaves that "
                    "candidate's row",
        "optimize": "tunes the numbers in the candidate's code to the target; deterministic, "
                    "so it runs once per candidate",
    }

    # A list of images, not a collage: the server fits EACH image to its pixel budget, so in
    # a collage the panel resolution dropped the more candidates were shown.
    IMAGE_NOTE = "This turn's images, after the target at the top: {panels}."
    # The target is not a candidate: it has no id in the pool, and it must be shown before the
    # first candidate with a mesh appears.
    TARGET_TOKEN = "target"
    IMAGE_NOTE_TARGET = "No candidate has been built yet; the target is the image at the top."

    ROLE_BEST = "best so far"
    ROLE_BEST_NEW = "best so far, made by your last turn"
    ROLE_LAST = "best made by your last turn"
    ROLE_PARENT = "what the best was built on (before your last turn's operation)"
    NOTE_NOTHING_NEW = "Your last turn made no new valid candidate, so there is no panel for it."

    # The reason substituted when `finish` came without one. The decision does not change:
    # "said to finish without an explanation" is still "said".
    DONE_WITHOUT_REASON = "no explanation"

    # The second door of `finish`: the first word of the reason. Without a legitimate "it does
    # not get better" door, stalling exited through "the shape matched" with a false
    # justification.
    _STALLED_RE = re.compile(r"^\W*stall", re.IGNORECASE)

    @classmethod
    def _stalled(cls, reason: str | None) -> bool:
        return bool(reason) and bool(cls._STALLED_RE.match(reason))

    # Metrics in the turn report. Not only the objective: GMS tells "the shape is about right"
    # from "the volume matched by chance". Title on the left, metric key on the right.
    METRIC_COLUMNS = (("iou", "iou"), ("gms", "gms_norm"))

    # Why the turn turned out this way. Two readers: the action reason goes to the run log
    # (read by a human and by `tools/dialogue_trace_stats.py`), the note goes into the
    # dialogue text (read by the assistant). One table for both so the wording of one event
    # does not drift apart.
    OUTCOMES: dict[str, tuple[str, str]] = {
        "chosen": ("assistant's choice", ""),
        "finished": ("assistant finished the part", ""),
        "done_too_early": (
            # The assistant answered, and in the right format, but has nothing to finish with.
            "assistant finished the part before the first valid candidate, "
            "took the default action",
            "you cannot finish yet: no candidate has built and passed validation, so an "
            "action was taken for you. Your DONE still stands and will end the part "
            "as soon as there is something valid to answer with",
        ),
        "unparsed": (
            "answer not recognized, took the default action",
            "your answer called no function, so an action was taken for you",
        ),
        "silent": (
            "assistant did not answer, took the default action",
            "you did not answer, so an action was taken for you",
        ),
        "truncated": (
            "answer cut off before the named action, even after a re-ask, "
            "took the default action",
            "your answer was cut off before you named an action, so an action was "
            "taken for you",
        ),
        "retried": (
            "answer cut off, decided on a re-ask without reasoning",
            "",
        ),
        "reasked": (
            "named action not on the list, decided on a re-ask",
            "",
        ),
        "reasked_unparsed": (
            "answer without an action, decided on a re-ask",
            "",
        ),
        "looked_only": (
            "assistant asked for images and named no action, took the default action",
            "you kept asking to look without choosing an action, so an action was "
            "taken for you",
        ),
        "rejected": (
            "no named action is on the list, even after a re-ask, "
            "took the default action",
            "none of the actions you named were available, so an action was taken for you",
        ),
        "unaffordable": (
            "assistant's choice is not affordable with the remainder, took the first affordable action",
            "your choices were not affordable any more, so the first affordable action was used",
        ),
    }

    # The action cap per turn. It does not replace the budget (the harness truncates the plan
    # by the remainder anyway) but keeps one turn from eating the whole remainder of a part.
    MAX_ACTIONS_PER_TURN = 4
    # How many times in a row the assistant may ask for an image without acting (a `SHOW:`
    # line). A turn must happen: an empty plan ends the part.
    MAX_LOOKS_PER_TURN = 2
    # How many times per turn to re-ask. A constant, not a literal in `_decide`: the
    # `budget.agent_text`/`agent_visual` caps are set against the maximum computed from it
    # (`cost_check`).
    MAX_RETRIES_PER_TURN = 1
    # Below which objective gain an operation counts as giving nothing
    # (`dialogue_io.prefix_facts`).
    DEAD_STEP_EPS = 0.001
    # How many candidates to show in the table.
    MAX_VALID_SHOWN = 12
    # How many candidates to draw in one show.
    MAX_IMAGE_PANELS = 4
    # The answer cap is shared between the reasoning and the visible part, and the first eats
    # it whole. The visible part is tiny: raising the cap is paid in decoding time, not in
    # transcript length.
    CHOOSE_MAX_TOKENS = 3072
    # The cap of a re-ask without reasoning (`_decide`): the answer is just the calls.
    RETRY_MAX_TOKENS = 512
    VISUALIZATION_MODE = "simple"
    # How many entries of the growing session to keep on context overflow.
    TRIM_KEEP_LAST = 6
    # `n` by tool semantics: for `samples` the assistant chooses it, for the others the
    # harness does. 4 for a ranked list is the mode of the model's own choice.
    FIXED_N = {"rank_depth": 4, "single": 1}
    # Second price point of `stepwise` in the legend. The legend is fixed for the run, while
    # the largest `n` of a menu row depends on the remaining budget; a table row carries it.
    COST_POINT_N = 8

    def __init__(self, objective: str = "iou"):
        self.objective = objective_mod.get_objective(objective)
        # The unchanging start of every question (rules and legend), built in `plan`.
        self._intro = ""

    def _choose_max_tokens(self, state: SearchState) -> int:
        """The answer cap on the decision turn: set by the run, otherwise the constant.

        The run sets it with the `experiment.agent.answer_max_tokens` key.
        """
        return getattr(state, "agent_answer_max_tokens", None) or self.CHOOSE_MAX_TOKENS

    # --- policy contract -----------------------------------------------------------

    def plan(self, state: SearchState) -> Plan:
        # The legend is built from the run registry and does not change between turns;
        # curly braces are stripped from it because `compose` calls `.format`.
        self._intro = (self.INTRO.format(tools=self._legend(state),
                                         max_actions=self.MAX_ACTIONS_PER_TURN)
                       .replace("{", "(").replace("}", ")") + "\n" + IMAGE_SLOT)

        session = dialogue_io.session_of(state, self.MEMORY_KEY)
        session.turn += 1
        session.say(self._report(state, session))

        menu = self._menu(state)
        if not menu:
            # There really are no actions; we do not invent one for its own sake.
            return Plan(actions=[])

        objective = self._scale(state)
        question = self.CHOOSE_QUESTION.format(
            candidates=self._candidates_text(state, menu),
            max_actions=self.MAX_ACTIONS_PER_TURN,
        )
        answer, parsed, outcome = self._decide(state, session, objective, question, menu)
        if parsed.done is not None:
            # The decision lives in the session rather than being returned straight from here:
            # with actions in the same answer `finish` means "this turn is the last", and then
            # the actions must execute.
            session.done_reason = parsed.done or self.DONE_WITHOUT_REASON

        # **Nothing to finish with while there is no valid candidate.** The part's answer is
        # the best candidate, and there is none. The flag is NOT cleared: it belongs to the
        # part, and the first valid candidate will execute it. The guard stands here and not
        # in `select()`: the harness reads `Plan.done` itself and ends the part before any
        # verdict.
        finishing = session.done_reason is not None and state.best_id is not None
        if session.done_reason is not None and not finishing:
            session.say(f"Turn {session.turn}: nothing has built and passed validation yet, "
                        "so there is nothing to finish with — going on.")
            if outcome == "finished":
                outcome = "done_too_early"

        picks, dropped = self._legal_picks(parsed.picks, menu)
        if dropped:
            # Named by name: silence about a dropped call is read by the assistant as a
            # fulfilled request.
            session.say(self._dropped_text(state, session.turn, parsed.picks, dropped))
        if parsed.picks and not picks and outcome in (
                "chosen", "retried", "reasked", "reasked_unparsed"):
            # Everything named was dropped: this is not the assistant's choice but a fallback.
            # `_decide` has already re-asked if there was anything to re-ask with.
            outcome = "rejected"
        if finishing and not picks:
            # Finished with no actions: return here, otherwise `_actions_from` would substitute
            # the fallback and cost an extra tool call exactly where the decision is to stop.
            session.say(f"Turn {session.turn}: finished — {session.done_reason}")
            return Plan(actions=[], done=True, reason=session.done_reason,
                        stalled=self._stalled(session.done_reason))
        if picks:
            # Which question the action came from only `_decide` knows; overwriting a re-ask
            # outcome with `chosen` would make the share of re-asks in the log zero.
            outcome = (outcome if outcome in ("retried", "reasked", "reasked_unparsed")
                       else "chosen")
            # A look request that came TOGETHER with actions belongs to the next turn.
            session.want_image = parsed.show

        actions, outcome = self._actions_from(state, picks, menu, outcome)
        if not actions:
            session.say(f"Turn {session.turn}: nothing on the list is affordable any more.")
            return Plan(actions=[])
        reason, note = self.OUTCOMES[outcome]
        for action in actions:
            action.reason = reason
        session.say(self._executed_text(session.turn, actions, note))
        session.last = [(action.tool, action.parent_id or "") for action in actions]
        if finishing:
            # The actions execute and the part ends after them (`search.py`, `plan_done`).
            return Plan(actions=actions, done=True, reason=session.done_reason,
                        stalled=self._stalled(session.done_reason))
        return Plan(actions=actions)

    # This policy has NO `select()`; see the module docstring.

    def _decide(
        self, state: SearchState, session: dialogue_io.Session,
        objective: objective_mod.Objective, question: str, menu: list[LegalAction],
    ) -> tuple[str, dialogue_io.Answer, str]:
        """Ask "what next" with this turn's images, as function calls."""
        auto, roles, note = self._auto_look(state)
        asked = [token for token in session.want_image if token not in auto]
        # Automatic panels go first: when panels are trimmed by count (`dialogue_io.show`),
        # what the assistant ordered on top of them is dropped, not the automatic ones.
        session.want_image = [*auto, *asked]
        state.memory[self.ROLES_KEY] = {"turn": session.turn, "roles": roles, "note": note}

        original = state.ask
        if original is None:
            return self._ask_loop(state, session, objective, question, menu)
        specs = self._tool_specs(state)

        def ask(prompt: str, image: Any = None, max_tokens: int = 16, **kwargs: Any) -> str:
            # The answer comes back as text: what was said in words, then `CALL <name> <JSON>`
            # lines, so the transcript and `_parse_answer` read one form.
            text = original(prompt, image=image, max_tokens=max_tokens,
                            tools=specs, tool_choice=self.TOOL_CHOICE, **kwargs)
            return tool_calls_io.merged_text(text, tool_calls_io.tool_calls(state))

        state.ask = ask
        try:
            return self._ask_loop(state, session, objective, question, menu)
        finally:
            state.ask = original

    def _ask_loop(
        self, state: SearchState, session: dialogue_io.Session,
        objective: objective_mod.Objective, question: str, menu: list[LegalAction],
    ) -> tuple[str, dialogue_io.Answer, str]:
        """The question, re-asked on a truncation or an answer without a move.

        A `SHOW:` line alone (the plain line format) returns the image and the same question
        again; such looks live INSIDE one turn, capped by `MAX_LOOKS_PER_TURN`.
        """
        image, note = None, ""
        # What has been shown THIS turn: a repeated request for the same ids is not drawn again.
        seen: list[str] = []
        pending = session.take_image_request()
        if pending:
            image, note = self._show(state, session, pending)
            if image is not None:
                seen.extend(pending)

        looks = 0
        # One re-ask per turn. On a truncation it is a DIFFERENT answer mode (no reasoning):
        # the same question in the same mode would get the same runaway reasoning. The same
        # slot is spent on an answer in which everything named is off the list, and on one
        # with no move at all, so the `budget.agent_*` caps (`cost_check`) do not move.
        retries = 0
        # The outcome of a re-ask triggered by the answer (not by a truncation).
        reasked: str | None = None
        max_tokens, thinking = self._choose_max_tokens(state), None
        while True:
            raw = dialogue_io.ask(
                state, session, self._intro, objective, question,
                max_tokens=max_tokens, image=image, image_note=note, thinking=thinking,
                trim_keep_last=self.TRIM_KEEP_LAST,
            )
            answer = dialogue_io.visible(raw)
            # `state.answer_truncated` refers to the LAST call.
            cut = dialogue_io.truncated(state)
            # **Not everything visible is parsed.** A cut INSIDE the reasoning (no closing tag)
            # means the "visible part" is the reasoning itself, and what the parser finds in it
            # is a draft, not a conclusion: in live runs the assistant came to a different
            # decision after it. Asked of the MODE, not of the text: there is no closing tag in
            # a re-ask or a run with `thinking: false` either.
            drafting = cut and dialogue_io.reasoning_possible(state, thinking) \
                and not dialogue_io.reasoning_closed(raw)
            parsed = dialogue_io.Answer() if drafting else self._parse_answer(answer)
            understood = bool(parsed.picks or parsed.show or parsed.done is not None)
            if answer and (understood or not cut):
                label = ("you asked to look" if parsed.show and not parsed.picks
                         else "your answer")
                # The visible part goes as is, including what the parser did not understand:
                # the assistant must see its answer as ITS OWN, not in our retelling.
                session.say(f"Turn {session.turn}, {label}:\n{answer}")
            elif cut:
                # A cut from which nothing was taken must be said, or it reads as a move that
                # did not happen. "Did not finish thinking" and "did not finish the answer" are
                # cured differently, so they are told differently.
                session.say(
                    f"Turn {session.turn}: your answer hit the answer cap before you named "
                    + ("an action, so it was dropped. Answer with function calls, not with "
                       "reasoning." if drafting else "an action, so it was dropped.")
                )
            if parsed.picks:
                legal, dropped = self._legal_picks(parsed.picks, menu)
                if (not legal and parsed.done is None
                        and retries < self.MAX_RETRIES_PER_TURN):
                    # Everything named is off the list. A re-ask with the reason and the legal
                    # set is cheaper than a lost turn after which the assistant copies the same
                    # answer.
                    retries += 1
                    reasked = "reasked"
                    session.say(self._dropped_text(state, session.turn, parsed.picks, dropped))
                    session.say(f"Turn {session.turn}: asked again — call act on candidates "
                                "from the table, each with a tool from that candidate's tools "
                                "column.")
                    continue
                if reasked:
                    return answer, parsed, reasked
                return answer, parsed, "retried" if retries else "chosen"
            if parsed.done is not None:
                # Finishing is a FULL answer, not the absence of an action. A look request
                # together with it is deliberately not executed.
                return answer, parsed, "finished"
            if parsed.show and looks < self.MAX_LOOKS_PER_TURN:
                looks += 1
                if image is not None and set(parsed.show) <= set(seen):
                    session.say(f"Turn {session.turn}: {', '.join(parsed.show)} already "
                                "shown this turn — the same images are still attached.")
                    continue
                image, note = self._show(state, session, parsed.show)
                if image is not None:
                    seen = [*seen, *(item for item in parsed.show if item not in seen)]
                max_tokens, thinking = self._choose_max_tokens(state), None
                continue
            # No answer with actions. Four causes, cured differently: a silent channel, a cut,
            # a misunderstood answer, and "looked but never acted".
            if raw is None:
                return answer, parsed, "silent"
            if cut or not answer:
                # A truncation is cured by the MODE, not by the cap: a lower cap was measured
                # to cut a noticeable share of answers that fit. An empty answer is about the
                # channel or the prompt, and re-asking it with the same text is pointless.
                if cut and retries < self.MAX_RETRIES_PER_TURN:
                    retries += 1
                    # The reduced cap is paid for by dropping the reasoning; on a run already
                    # without reasoning the same question goes with the normal cap.
                    was_reasoning = dialogue_io.reasoning_possible(state, thinking)
                    max_tokens, thinking = (
                        (self.RETRY_MAX_TOKENS, False) if was_reasoning
                        else (self._choose_max_tokens(state), thinking)
                    )
                    session.say(
                        f"Turn {session.turn}: "
                        + ("asked again without reasoning, answer with function calls only."
                           if was_reasoning else
                           "your answer hit the answer cap before you named an action; "
                           "asked again, answer with function calls only.")
                    )
                    continue
                return answer, parsed, "truncated"
            if not parsed.show and retries < self.MAX_RETRIES_PER_TURN:
                # A fitting answer without a call: live, almost always the one sentence alone.
                # The question CHANGES: the transcript records what was missing.
                retries += 1
                reasked = "reasked_unparsed"
                session.say(
                    f"Turn {session.turn}: your answer called no function, so nothing was "
                    "done — a sentence alone is not a move. Asked again: call act on "
                    "candidates from the table."
                )
                continue
            return answer, parsed, "looked_only" if parsed.show else "unparsed"

    # --- the unchanging part of the question -------------------------------------------

    def _legend(self, state: SearchState) -> str:
        rows = []
        for tool in sorted(state.tools):
            info = state.tools[tool]
            parts = [self.TOOL_NOTES.get(tool, "")]
            if info.max_code_numbers is not None:
                parts.append(f"not offered on a candidate whose code has "
                             f"{info.max_code_numbers} or more numbers")
            if info.n_semantics == "samples":
                parts.append("n = how many samples to draw, up to the n in the table")
                parts.append(f"≈{info.wall(1):.1f}s per call at n=1, "
                             f"≈{info.wall(self.COST_POINT_N):.1f}s at n={self.COST_POINT_N}")
            else:
                n = self.FIXED_N.get(info.n_semantics, 1)
                if info.n_semantics == "rank_depth":
                    parts.append(f"each call takes the next {n} from its ranked list")
                parts.append(f"≈{info.wall(n):.1f}s per call")
            rows.append(f"- {tool} — " + "; ".join(part for part in parts if part))
        return "\n".join(rows)

    def _tool_specs(self, state: SearchState) -> list[dict[str, Any]]:
        """The turn's two functions, constant for the run (see the module docstring)."""
        tools = sorted(state.tools)
        return [
            _function(
                "act",
                "Call one tool on one candidate from the table. It runs after this answer, "
                "together with the other act calls of this turn; what it made comes in the "
                "next report.",
                {
                    "tool": {"type": "string", "enum": tools,
                             "description": "a tool from that candidate's tools column"},
                    "candidate": {"type": "string",
                                  "description": "candidate id from the id column of the table"},
                    "n": {"type": "integer", "minimum": 1,
                          "description": "stepwise only: how many samples to draw"},
                },
                ["tool", "candidate"],
            ),
            _function(
                "finish",
                "End the part; the best candidate is its answer. Together with act calls, "
                "this turn is your last one.",
                {"reason": {"type": "string", "enum": ["shape", "stalled"],
                            "description": "why you are finishing, as defined in the rules"},
                 "why": {"type": "string",
                         "description": "one short sentence saying why nothing more would help"}},
                ["reason", "why"],
            ),
        ]

    # --- turn report -----------------------------------------------------------

    def _report(self, state: SearchState, session: dialogue_io.Session) -> str:
        """The text about the results of the PREVIOUS turn. Empty only before the first turn.

        Silence about a fruitless turn is not allowed: a missing report reads as "the turn
        went fine", and the assistant repeats the same choice. The reason comes from
        `state.attempts`, which the harness maintains.

        Broken candidates are named here although they are not in the table: the table is a
        menu, the report is the OUTCOME of the turn.
        """
        if not session.last:
            return ""
        turn = session.turn - 1
        objective = self._scale(state)
        if not state.fresh:
            return (f"Turn {turn} result: no new candidates" + self._empty_why(state, session)
                    + self._best_line(state))

        known = objective.best_value(
            value for _, value in state.scored(state.settled(), objective.value)
        )
        # The "best" mark goes to ONE candidate of the turn, not to everyone who beat the
        # previous best: three "best so far" lines in a row read as three improvements.
        lines = [f"Turn {turn} result:"]
        lines.extend(
            f"- {self._candidate_line(candidate, self._leader(state, objective, known))}"
            for candidate in state.fresh
        )
        lines.extend(self._reused_lines(state, session))
        return "\n".join(lines) + self._best_line(state)

    @staticmethod
    def _leader(state: SearchState, objective, known) -> str | None:
        """Which fresh candidate improved the best known one. `None` if none."""
        scored = state.scored([c for c in state.fresh if c.alive], objective.value)
        if not scored:
            return None
        candidate, value = min(scored, key=lambda pair: objective.sort_key(pair[1]))
        return candidate.id if objective.better(value, known, 0.0) else None

    def _empty_why(self, state: SearchState, session: dialogue_io.Session) -> str:
        """Why the turn gave nothing, per action of the turn."""
        reasons = []
        for tool, parent_id in session.last:
            rows = state.attempts_on(parent_id, tool)
            row = rows[-1] if rows else None
            if row is None:
                continue
            if row.failure:
                reasons.append(f"{tool} on {parent_id[:8]}: {row.failure}")
            elif row.reused:
                # The call worked but everything it returned was already in the pool.
                reasons.append(f"{tool} on {parent_id[:8]}: everything it returned was "
                               f"already on the list as {self._ids_text(row.reused)}")
        return f" ({'; '.join(reasons)})." if reasons else "."

    def _reused_lines(self, state: SearchState, session: dialogue_io.Session) -> list[str]:
        """Report lines for candidates that came back AGAIN in a turn where new ones also appeared."""
        lines = []
        for tool, parent_id in session.last:
            rows = state.attempts_on(parent_id, tool)
            if rows and rows[-1].reused:
                lines.append(f"- {tool} on {parent_id[:8]}: also returned code already on the "
                             f"list as {self._ids_text(rows[-1].reused)}")
        return lines

    @staticmethod
    def _ids_text(ids: list[str], limit: int = 4) -> str:
        unique = list(dict.fromkeys(item[:8] for item in ids))
        head = ", ".join(unique[:limit])
        return head + (f" and {len(unique) - limit} more" if len(unique) > limit else "")

    def _best_line(self, state: SearchState) -> str:
        """Who is best right now, and for how many turns: one line in EVERY report.

        The age is taken from the harness (`iteration - best_set_iteration`), not computed by
        the policy. It is a fact, not advice.
        """
        best = state.best
        fitness = state.fitness_of(best) if best is not None else None
        if best is None or best.failure is not None or fitness is None:
            return ""
        line = f"\nBest so far: {best.id[:8]} (score {fitness:.4f})"
        stale = max(0, int(state.iteration) - int(state.best_set_iteration))
        if stale > 0:
            line += f", unchanged for {stale} turn{'s' if stale > 1 else ''}"
        return line + "."

    def _candidate_line(self, candidate: Candidate, leader: str | None) -> str:
        short = candidate.id[:8]
        origin = candidate.origin.tool
        metrics = dialogue_io.metrics_text(candidate.metrics, self.METRIC_COLUMNS)
        if not candidate.built:
            return f"{short} ({origin}): did not build ({candidate.failure or 'unknown reason'})"
        if candidate.failure is not None:
            # Built but invalid: shown as a refusal, without numbers (an IoU of 0.997 on an
            # open mesh read as "the shape matched").
            return f"{short} ({origin}): built, but invalid ({candidate.failure})"
        better = " — best so far" if candidate.id == leader else ""
        return f"{short} ({origin}): {metrics}{better}"

    def _executed_text(self, turn: int, actions: list[Action], note: str) -> str:
        """What actually went to the harness, after truncating `n`."""
        calls = "; ".join(f"{action.tool} on {(action.parent_id or '?')[:8]} n={action.n}"
                          for action in actions)
        return f"Turn {turn} executed: {calls}" + (f" ({note})" if note else "")

    # --- action menu -----------------------------------------------------------

    def _menu(self, state: SearchState) -> list[LegalAction]:
        """What is shown to the assistant and what is accepted from it: ONE list.

        The whole pool of valid candidates, not a working frontier: a candidate of an earlier
        turn is addressable on a par with a fresh one, otherwise there is no backtracking.
        Broken candidates are not shown (`_shown_candidates`), so actions on them do not get
        here by construction.
        """
        shown = {candidate.id for candidate in self._shown_candidates(state)}
        return [action for action in state.legal if action.parent_id in shown]

    def _shown_candidates(self, state: SearchState) -> list[Candidate]:
        """Who to show in the table: the valid ones by quality, plus two reservations.

        Valid ones are ranked by the HARNESS scale (the score the part's answer is chosen
        by), with the policy objective as the tie-breaker.

        - **the best one, always**, even if it has no legal action: a row useless for the
          turn and necessary as a reference;
        - **the root, always, while it has an action.** The root is the worst row of any
          quality ranking, yet starting over is the only action that changes the first
          operation, which caps the branch.
        """
        objective = self._scale(state)
        actionable = {action.parent_id for action in state.legal}
        valid = [c for c in state.pool
                 if c.id in actionable and c.built and c.failure is None]

        def rank(candidate: Candidate) -> tuple[int, float, float]:
            # Unmeasured ones go to the tail of the group, not to its start as a zero.
            fitness = state.fitness_of(candidate)
            value = objective.value(candidate.metrics)
            return (
                0 if fitness is not None else 1,
                -(fitness or 0.0),
                0.0 if value is None else objective.sort_key(value),
            )

        valid.sort(key=rank)
        shown = valid[: self.MAX_VALID_SHOWN]

        seen = {candidate.id for candidate in shown}
        # An invalid best (kept by the harness while there are no valid ones) is not a
        # candidate for choice.
        best = state.best
        if best is not None and best.failure is None and best.id not in seen:
            shown.append(best)
            seen.add(best.id)
        root = self._root(state)
        if root is not None and root.id not in seen and root.id in actionable:
            shown.append(root)
        return shown

    @staticmethod
    def _root(state: SearchState) -> Candidate | None:
        """The pool root, the "start over" point: found by depth, not by the name `c0`."""
        for candidate in state.pool:
            if candidate.depth == 0:
                return candidate
        return None

    def _candidates_text(self, state: SearchState, menu: list[LegalAction]) -> str:
        """The candidate table: what each is, in what state, and what can be done with it."""
        by_parent: dict[str, list[LegalAction]] = {}
        for action in menu:
            by_parent.setdefault(action.parent_id or "", []).append(action)
        rows = [f"{'id':<10}{'from':<11}{'ops':<5}{'status':<7}{'score':<8}"
                f"{'iou':<7}{'gms':<7}{'first':<8}tools"]
        shown = self._shown_candidates(state)
        for candidate in shown:
            tools = []
            for action in by_parent.get(candidate.id, []):
                info = state.tools.get(action.tool)
                chooses_n = info is not None and info.n_semantics == "samples"
                tools.append(f"{action.tool}(n≤{action.max_n})" if chooses_n else action.tool)
            fitness = state.fitness_of(candidate)
            first, _ = dialogue_io.prefix_facts(state, candidate, self.DEAD_STEP_EPS)
            status = ("start" if candidate.depth == 0
                      else "best" if candidate.id == state.best_id else "")
            metrics = candidate.metrics or {}
            rows.append(
                f"{candidate.id[:8]:<10}{candidate.origin.tool[:10]:<11}{candidate.depth:<5}"
                f"{status:<7}{_num(fitness, 4):<8}{_num(metrics.get('iou'), 3):<7}"
                f"{_num(metrics.get('gms_norm'), 3):<7}{_num(first, 4):<8}"
                f"{', '.join(tools) or '-'}"
            )
        hidden = len({a.parent_id for a in state.legal} - {c.id for c in shown})
        if hidden > 0:
            rows.append(f"({hidden} more candidates are not shown)")
        return "\n".join(rows)

    # --- images --------------------------------------------------------------

    def _auto_look(self, state: SearchState) -> tuple[list[str], dict[str, str], str]:
        """Whom to show without being asked: panel ids, their roles, a note in words."""
        best = state.best
        # The root is an empty start, not a figure: it is the best only until the first
        # valid candidate, and then the target must be shown.
        if best is None or best.depth == 0 or not self._drawable(best):
            return [self.TARGET_TOKEN], {}, ""
        roles = {best.id[:8]: self.ROLE_BEST}
        fresh = [c for c in (state.fresh or []) if self._drawable(c)]
        if not fresh:
            return [best.id[:8]], roles, self.NOTE_NOTHING_NEW
        last = max(fresh, key=lambda c: state.fitness_of(c) or float("-inf"))
        if last.id != best.id:
            roles[last.id[:8]] = self.ROLE_LAST
            return [best.id[:8], last.id[:8]], roles, ""
        roles[best.id[:8]] = self.ROLE_BEST_NEW
        parent = state.get(best.parent_id) if best.parent_id else None
        if parent is not None and self._drawable(parent) and parent.depth > 0:
            roles[parent.id[:8]] = self.ROLE_PARENT
            return [best.id[:8], parent.id[:8]], roles, ""
        return [best.id[:8]], roles, ""

    @staticmethod
    def _drawable(candidate: Candidate | None) -> bool:
        return bool(candidate is not None and candidate.alive and candidate.mesh_path)

    def _show(
        self, state: SearchState, session: dialogue_io.Session, wanted: list[str]
    ) -> tuple[Any, str]:
        """Render what was named (the target always first), captioned and slotted for the cache."""
        image, note = dialogue_io.show(
            state, session, wanted,
            max_panels=self.MAX_IMAGE_PANELS,
            target_token=self.TARGET_TOKEN,
            image_note=self.IMAGE_NOTE,
            image_note_target=self.IMAGE_NOTE_TARGET,
            visualization_mode=self.VISUALIZATION_MODE,
        )
        # Panel roles go only with the automatic show, once: a look ordered later in the same
        # turn does not repeat them. The record is cleared also when the show did not come
        # together.
        record = state.memory.pop(self.ROLES_KEY, None) or {}
        if image is not None and note and record.get("turn") == session.turn:
            roles = record.get("roles") or {}
            lines = []
            for token in wanted:
                role = roles.get(token)
                candidate = next((c for c in state.pool if c.id.startswith(token)), None)
                if role is None or candidate is None:
                    continue
                fitness = state.fitness_of(candidate)
                score = "-" if fitness is None else f"{fitness:.4f}"
                lines.append(f"{token} — {role} ({candidate.depth} ops, score {score})")
            extra = "; ".join(lines)
            if extra:
                note = f"{note}\nPanels: {extra}."
            if record.get("note"):
                note = f"{note}\n{record['note']}"
        if image is None:
            return image, note
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        # The target is the first image of any show; the one stored at the part's first show
        # is used (see the module docstring, on the cache).
        images[0] = state.memory.setdefault(self.TARGET_KEY, images[0])
        # The first marker sits at the end of the unchanging part (`plan`); the rest are
        # here, before the caption of the turn's images.
        return images, IMAGE_SLOT * (len(images) - 1) + note

    # --- answer parsing and action assembly ------------------------------------

    def _parse_answer(self, answer: str) -> dialogue_io.Answer:
        calls = tool_calls_io.calls_in_text(answer)
        if calls:
            return self._answer_from_calls(calls)
        # No call, parsed or in text: the model may have answered in the line format.
        return self._parse_text(answer)

    @staticmethod
    def _answer_from_calls(calls: list[tuple[str, dict[str, Any]]]) -> dialogue_io.Answer:
        parsed = dialogue_io.Answer()
        for name, args in calls:
            if name == "act":
                tool, candidate = args.get("tool"), args.get("candidate")
                if not tool or not candidate:
                    # Without an address the action cannot be recovered: guessing the
                    # candidate would execute something other than what was asked.
                    continue
                parsed.picks.append(dialogue_io.Pick(
                    tool=str(tool), candidate=str(candidate), n=_count(args.get("n")),
                ))
            elif name == "finish":
                reason = str(args.get("reason") or "").strip()
                why = str(args.get("why") or "").strip()
                parsed.done = " — ".join(part for part in (reason, why) if part)
        return parsed

    _ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(.+)$", re.IGNORECASE)
    _SHOW_RE = re.compile(r"^\s*SHOW(?:_IMAGE)?\s*:\s*(.*)$", re.IGNORECASE)
    # Start of line: otherwise a `DONE:` quoted in the transcript would end the part.
    _DONE_RE = re.compile(r"^\s*DONE\s*:\s*(.*)$", re.IGNORECASE)
    _FIELD_RE = {
        "tool": re.compile(r"\btool\s*=\s*([A-Za-z_][\w]*)"),
        "candidate": re.compile(r"\bcandidate\s*=\s*([A-Za-z0-9_]+)"),
        "n": re.compile(r"\bn\s*=\s*(\d+)"),
    }

    def _parse_text(self, answer: str) -> dialogue_io.Answer:
        """The line format: `ACTION: tool=... candidate=... n=...`, `SHOW: ids`, `DONE: reason`."""
        parsed = dialogue_io.Answer()
        for line in (answer or "").splitlines():
            action_match = self._ACTION_RE.match(line)
            if action_match:
                body = action_match.group(1)
                tool = self._FIELD_RE["tool"].search(body)
                candidate = self._FIELD_RE["candidate"].search(body)
                if tool and candidate:
                    count = self._FIELD_RE["n"].search(body)
                    parsed.picks.append(dialogue_io.Pick(
                        tool=tool.group(1), candidate=candidate.group(1),
                        n=int(count.group(1)) if count else None,
                    ))
                continue
            show_match = self._SHOW_RE.match(line)
            if show_match:
                parsed.show = [
                    token for token in re.split(r"[,\s]+", show_match.group(1).strip())
                    if token and token.lower() not in {"none", "no", "-"}
                ]
                continue
            done_match = self._DONE_RE.match(line)
            if done_match:
                text = done_match.group(1).strip()
                # A format placeholder repeated verbatim is not a decision.
                if text.startswith("<") and text.endswith(">"):
                    continue
                parsed.done = text
        return parsed

    def _legal_picks(
        self, picks: list[dialogue_io.Pick], menu: list[LegalAction]
    ) -> tuple[list[tuple[dialogue_io.Pick, LegalAction]], list[str]]:
        return dialogue_io.legal_picks(picks, menu, self.MAX_ACTIONS_PER_TURN)

    def _dropped_text(
        self, state: SearchState, turn: int, picks: list[dialogue_io.Pick], dropped: list[str],
    ) -> str:
        """What was dropped and WHY: by name, with the candidate's legal set.

        The start of the line is fixed: log analysis counts refusals by it.
        """
        shown = {candidate.id for candidate in self._shown_candidates(state)}
        parts = []
        for pick in picks[: self.MAX_ACTIONS_PER_TURN]:
            label = f"tool={pick.tool} candidate={pick.candidate}"
            if label in dropped:
                parts.append(f"{label} ({self._why_not_offered(state, pick, shown)})")
        parts.extend(item for item in dropped if not item.startswith("tool="))
        return f"Turn {turn}: not on the list, ignored — " + "; ".join(parts)

    def _why_not_offered(self, state: SearchState, pick: dialogue_io.Pick, shown: set[str]) -> str:
        """The reason in the assistant's words. Computed from harness data, not guessed."""
        candidate = next((c for c in state.pool
                          if c.id.startswith(pick.candidate) and pick.candidate), None)
        if candidate is None:
            return f"there is no candidate {pick.candidate}"
        short = candidate.id[:8]
        if candidate.id not in shown:
            return f"{short} is not in the table this turn"
        info = state.tools.get(pick.tool)
        if info is None:
            return f"{pick.tool} is not a tool of this run"
        allowed = sorted({a.tool for a in state.legal if a.parent_id == candidate.id})
        can = f"; on {short} you can call: {', '.join(allowed)}" if allowed else \
            f"; nothing more can be called on {short}"
        if state.attempts_on(candidate.id, pick.tool):
            if info.deterministic:
                why = f"{pick.tool} already ran on {short}, and it gives the same result again"
            elif info.n_semantics == "rank_depth":
                why = f"{pick.tool} has used up its ranked list on {short}"
            else:
                why = f"{pick.tool} is used up on {short}"
        elif info.requires == "none" and candidate.depth > 0:
            why = f"{pick.tool} only starts from the empty start"
        elif info.max_code_numbers is not None and \
                code_numbers(candidate.code or "") >= info.max_code_numbers:
            why = (f"{pick.tool} is not offered on {short}: its code has "
                   f"{code_numbers(candidate.code or '')} numbers, the limit is "
                   f"{info.max_code_numbers}")
        else:
            why = f"{pick.tool} is not available on {short} now"
        return why + can

    def _fallback(self, state: SearchState, menu: list[LegalAction]) -> LegalAction:
        """The action when the assistant named none: the cheapest one on the BEST candidate.

        Not the first menu row (in practice `stepwise` from the root with `n=1`, which dedup
        collapsed into a known candidate): continuing the best at least moves the branch the
        part is judged by.
        """
        on_best = [action for action in menu if action.parent_id == state.best_id]
        if not on_best:
            return menu[0]
        return min(on_best, key=lambda action: state.wall_of(action.tool, 1))

    def _actions_from(
        self,
        state: SearchState,
        picks: list[tuple[dialogue_io.Pick, LegalAction]],
        menu: list[LegalAction],
        outcome: str,
    ) -> tuple[list[Action], str]:
        """Turn actions: the chosen ones if affordable, otherwise the first available.

        `n` is the assistant's only for sample tools (`FIXED_N` otherwise). The remainder is
        divided between the actions of one turn (`spread`) rather than given whole to each:
        it is recomputed only after launch.
        """
        if picks:
            chosen = [(self._wanted_n(state, pick, legal), legal) for pick, legal in picks]
        else:
            chosen = [(None, self._fallback(state, menu))]
        spread = max(1, len(chosen))
        actions: list[Action] = []
        for want, legal in chosen:
            want = want if want and want > 0 else 1
            n = state.affordable_n(legal.tool, min(want, legal.max_n), spread=spread)
            if n <= 0:
                continue
            actions.append(Action(tool=legal.tool, parent_id=legal.parent_id, n=n, params={}))
        if actions:
            return actions, outcome

        # No choice is affordable from the remainder: take the first list action that is still
        # affordable rather than return an empty plan, which would end the part.
        for option in menu:
            n = state.affordable_n(option.tool, 1)
            if n > 0:
                return [Action(tool=option.tool, parent_id=option.parent_id, n=n,
                               params={})], "unaffordable"
        return [], outcome

    def _wanted_n(self, state: SearchState, pick: dialogue_io.Pick, legal: LegalAction) -> int | None:
        info = state.tools.get(legal.tool)
        semantics = info.n_semantics if info is not None else "samples"
        return pick.n if semantics == "samples" else self.FIXED_N.get(semantics, 1)

    # --- part scale ------------------------------------------------------------

    def _scale(self, state: SearchState) -> objective_mod.Objective:
        """The scale of this part (see `objective.scale_for`).

        Not the declared objective but the fallback one: with a non-watertight GT no candidate
        has a volumetric IoU, and a report in the declared scale would read the environment
        as quality.
        """
        return objective_mod.scale_for(
            self.objective, state.memory, [c.metrics for c in state.pool]
        )


def _function(name: str, description: str, properties: dict[str, Any],
              required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


def _count(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any, digits: int) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def entrypoint() -> DialogueLeanPolicy:
    """Contract entry point: the live policy object."""
    return DialogueLeanPolicy(objective="iou")
