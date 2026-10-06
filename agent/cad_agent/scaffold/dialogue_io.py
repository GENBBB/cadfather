"""Dialogue mechanics: everything the dialogue policy does but does not decide.

This module is the half of `policies/dialogue_lean.py` that has no strategic
choice in it: talking to the server, parsing reasoning tags, assembling the image and
checking the knobs. What stays in the policy is everything with a real choice (prompt
texts, caps, what the table shows, when to ask for a picture, how to parse the answer).
A convenient test for the boundary: "does editing this place give a different STRATEGY
or only a different implementation?"

Boundaries that should NOT be moved here:

- answer parsing (`_parse_answer` and its regexes) is the second half of the prompt
  contract. A change of the answer format must change the parser in the same file;
  separated by an import boundary, they would make any format change fall back to the
  first legal action;
- the decision "what to show" and "how many panels" is strategy. Only the image assembly
  after the decision is made lives here;
- the `menu[0]` fallback and the division of the remainder between the actions of a turn
  belong to the policy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cad_agent.capabilities import objective as objective_mod
from cad_agent.harness.budget import BudgetExceeded
from cad_agent.harness.search_types import (
    Candidate,
    LegalAction,
    PromptTooLarge,
    SearchState,
)

logger = logging.getLogger(__name__)


# --- conversation structures ----------------------------------------------------


@dataclass
class Pick:
    """One action named by the assistant, not yet checked for legality."""

    tool: str
    candidate: str
    n: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """A parsed answer to the question "what next".

    `done` is `None` if there was no `DONE:` line at all, and the reason text if there
    was. An empty reason is STILL "said to finish": "did not say" and "said without an
    explanation" must be told apart here, otherwise an assistant that did not find the
    wording silently continues the part.
    """

    picks: list[Pick] = field(default_factory=list)
    show: list[str] = field(default_factory=list)
    done: str | None = None


@dataclass
class Session:
    """The dialogue for ONE part: growing text, turn number and the image request.

    It lives in `state.memory`, i.e. for exactly one unit of work: the policy is built
    once per run and outlives a part, so a field of its own would silently leak between
    parts.
    """

    turn: int = 0
    lines: list[str] = field(default_factory=list)
    # Whom the assistant asked to show at the NEXT question.
    want_image: list[str] = field(default_factory=list)
    # What was used last time: pairs (tool, parent). The report needs it to name the
    # reason for an empty turn from `state.attempts` instead of guessing.
    last: list[tuple[str, str]] = field(default_factory=list)
    # The assistant said `DONE:`, so the part ends. It lives in the session rather than in a
    # policy field: the policy outlives a part and a field of its own would leak into the
    # next one (same reason as for `turn` and `want_image`).
    done_reason: str | None = None

    def say(self, text: str) -> None:
        if text:
            self.lines.append(text)

    def take_image_request(self) -> list[str]:
        """Consume the image request. It is consumed in any case, even if the image
        then failed to build: the request belongs to the nearest question and does not
        hang until it can be fulfilled."""
        want, self.want_image = self.want_image, []
        return want

    def trim(self, keep_last: int) -> None:
        self.lines = self.lines[-keep_last:]
        self.lines.append("(earlier turns were dropped: the transcript did not fit)")


def session_of(state: SearchState, key: str) -> Session:
    """The session of this part. Created on first access."""
    session = state.memory.get(key)
    if not isinstance(session, Session):
        session = state.memory[key] = Session()
    return session


# --- reasoning and truncated answers ---------------------------------------------

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_TAIL = re.compile(r"^.*</think>", re.DOTALL | re.IGNORECASE)


def visible(answer: str | None) -> str:
    """The visible part of an answer: without the reasoning of a thinking model.

    Three forms, all of which occur on one server depending on whether it runs with
    `--reasoning-parser`: a full block, an unclosed block (the answer was cut by the cap)
    and a "tail" after `</think>` with no opening tag. An empty result means "there is no
    visible answer", not "the answer is empty"; the caller tells these apart.

    **There is a fourth form that cannot be stripped here:** reasoning with NO tag at all.
    The opening tag is put by the server's chat template, and a truncated answer has no
    closing one. Such an answer cannot be told from a non-format answer by its text, and
    should not be tried: the harness has a truncation flag (`SearchState.answer_truncated`)
    that callers ask. What matters here is that on this form the function returns the
    reasoning WHOLE. So its result is neither parsed nor added to the transcript when the
    truncation fell inside the reasoning itself (`reasoning_closed` is false): otherwise a
    draft in the middle of a thought would be executed as a decision, and the reasoning
    would travel into every following prompt.
    """
    text = answer or ""
    text = _THINK_CLOSED.sub("", text)
    if "</think>" in text.lower():
        text = _THINK_TAIL.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def reasoning_possible(state: SearchState, thinking: bool | None) -> bool:
    """Whether the answer could contain reasoning.

    Two sources, in exactly this order: an override for ONE call (`thinking`) beats the
    run mode, and `None` means "not overridden" and leaves the decision to the run
    (`state.agent_thinking`). A run that said nothing about the mode is also `None`: there
    is no key in the request and the server template decides; we treat reasoning as
    possible because it is the default of thinking models, and an error in this direction
    costs an extra re-ask, while in the other it costs a discarded answer.
    """
    if thinking is not None:
        return bool(thinking)
    return state.agent_thinking is not False


def reasoning_closed(raw: str | None) -> bool:
    """Whether a thinking model managed to close its reasoning.

    The opening tag is put by the server's chat template (it may be absent from the
    answer), so the only reliable sign is the closing one. It separates two different
    truncations: "did not finish thinking" (no tag, no answer) and "finished thinking but
    did not finish the answer" (tag present, the visible part is real).
    """
    return "</think>" in (raw or "").lower()


def truncated(state: SearchState) -> bool:
    """Whether the last assistant answer was cut by the answer cap.

    Ask right after `state.ask`: the next question overwrites the flag. A harness without
    this field (old or hand-built) answers "no", and the outcome is recognized as before,
    by an empty visible answer.
    """
    probe = getattr(state, "answer_truncated", None)
    return bool(probe()) if probe is not None else False


# --- calling the assistant --------------------------------------------------------


def compose(
    session: Session, intro_template: str, objective: objective_mod.Objective,
    question: str, image_note: str,
) -> str:
    """The whole prompt: intro, transcript, image note, question.

    The scale direction is spelled out in our own words rather than taken from
    `objective.direction`: that property is worded for the report and the log,
    not for the prompt.
    """
    parts = [image_note] if image_note else []
    parts.append(question)
    return head(session, intro_template, objective) + "\n\n".join(parts)


def head(session: Session, intro_template: str, objective: objective_mod.Objective) -> str:
    """The invariant start of the `compose` prompt: intro and transcript with a separator.

    The next turn appends to the transcript, so this start is also the start of the next
    turn's prompt; the harness warms it (`experiment.agent.prewarm`). The separator
    `"\\n\\n"` is part of the start on purpose: without it a period at the end of a line and
    `.\\n\\n` in the full prompt are different tokens, and the start stops being a
    token-for-token prefix (`llm.prewarm_content`).
    """
    intro = intro_template.format(
        objective_name=objective.name,
        objective_direction=("higher is better" if objective.higher_is_better
                             else "lower is better"),
    )
    return "\n\n".join([intro, *session.lines]) + "\n\n"


def ask(
    state: SearchState, session: Session, intro_template: str,
    objective: objective_mod.Objective, question: str, *,
    max_tokens: int, image: Any = None, image_note: str = "",
    thinking: bool | None = None, trim_keep_last: int,
) -> str | None:
    """Ask a question with the accumulated history. `None` means no answer.

    There are three different reasons for no answer (the channel is not connected, the call
    failed, the prompt does not fit even trimmed), and all three must end in degradation,
    not an exception: a `plan()` that fails outward costs the whole part, while a turn on
    the first action of the list costs one turn. Only `BudgetExceeded` propagates: the
    harness cap exists so that it cannot be bypassed by the policy's politeness.
    """
    if state.ask is None:
        return None
    prompt = compose(session, intro_template, objective, question, image_note)
    try:
        answer = call(state, prompt, image, max_tokens, thinking,
                      _prefix(state, session, intro_template, objective))
    except BudgetExceeded:
        raise
    except PromptTooLarge as exc:
        # The session did not fit as a whole: degrade rather than refuse the turn: trim
        # the history and ask the same thing shorter.
        logger.warning("Dialogue session does not fit the context, truncating history: %s", exc)
        session.trim(trim_keep_last)
        answer = _ask_once(
            state, compose(session, intro_template, objective, question, image_note),
            image, max_tokens, thinking, _prefix(state, session, intro_template, objective))
    except Exception as exc:
        logger.warning("Assistant did not answer: %s", exc)
        return None
    return answer


def _prefix(state: SearchState, session: Session, intro_template: str,
            objective: objective_mod.Objective) -> str | None:
    """The start of the question for warm-up, only when the harness warms up."""
    if not getattr(state, "agent_prewarm", False):
        return None
    return head(session, intro_template, objective)


def _ask_once(
    state: SearchState, prompt: str, image: Any, max_tokens: int,
    thinking: bool | None = None, prefix: str | None = None,
) -> str | None:
    try:
        return call(state, prompt, image, max_tokens, thinking, prefix)
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning("Assistant did not answer even on the truncated history: %s", exc)
        return None


def call(
    state: SearchState, prompt: str, image: Any, max_tokens: int, thinking: bool | None,
    prefix: str | None = None,
) -> str:
    """One assistant call. `thinking=None` means "as set by the run".

    The mode is passed ONLY when it is really overridden: the `thinking` key appeared in
    `ask_agent` together with the repair channel, and an older harness (or one hand-built
    in a check) does not know it. Staying silent about the default keeps such a harness
    working, whereas `ask` catches `Exception` and would return `None`, so the
    incompatibility would arrive here as "the assistant did not answer" and send people to
    fix the endpoint.
    """
    # `prefix` follows the same reasoning: only when present (`_prefix`).
    extra: dict[str, Any] = {"prefix": prefix} if prefix is not None else {}
    if thinking is None:
        return state.ask(prompt, image=image, max_tokens=max_tokens, **extra)
    return state.ask(prompt, image=image, max_tokens=max_tokens, thinking=thinking, **extra)


# --- images ----------------------------------------------------------------------


def visual_affordable(state: SearchState) -> bool:
    """Whether the remainder is enough for a question with an image. Computed BEFORE the call.

    By the same counter that `ask_agent` will charge (`agent_visual`): a question with an
    image is costlier than a text one not figuratively but as a separate cap item.
    """
    left = state.remaining.calls.get("agent_visual")
    return left is None or left > 0


def resolve_ids(state: SearchState, wanted: list[str]) -> list[str]:
    """Abbreviations from the table to pool ids, in the order of the request."""
    ids: list[str] = []
    for token in wanted:
        candidate = next((c for c in state.pool if c.id.startswith(token)), None)
        if candidate is not None and candidate.id not in ids:
            ids.append(candidate.id)
    return ids


def nothing_drawn(session: Session, turn: int, problems: list[str]) -> tuple[Any, str]:
    session.say(f"Turn {turn}: nothing could be drawn"
                + (f" — {'; '.join(problems)}" if problems else "") + ".")
    return None, ""


def _panel_budget(state: SearchState, max_panels: int) -> tuple[int, int | None]:
    """How many candidate panels actually fit into one request.

    Two caps, about different things. `max_panels` is a policy constant: more panels
    means each is smaller, and the question resolution is set by the server config.
    `state.agent_max_images` is the ENDPOINT cap (`--limit-mm-per-prompt`) and it is
    hard: the server rejects a request over it WHOLE instead of trimming it.

    Hence the `-1`: the target is the first panel of a request, so it always takes one of
    the images (`show` builds `image 1 = the TARGET shape`).

    Why trim here instead of refusing. A policy that raised `MAX_IMAGE_PANELS` above the
    cap got not "showed less" but `None` from `ask`, i.e. "the assistant did not answer"
    on EVERY turn with an image. The fitness is honest (bad) while the diagnosis in the log
    points at a dead channel instead of the policy's own constant. Trimming keeps the
    change legal and names the reason aloud (same reasoning as the dropped knob in
    `params_for`).

    The endpoint cap is not verified against the endpoint itself: it is derived from the
    run config (`launch_plan.server_image_limit`), and the server may have been started
    earlier with another config. This guards against a mistake of the POLICY, not a check
    against the live server.
    """
    endpoint = getattr(state, "agent_max_images", None)
    if endpoint is None:
        return max_panels, None
    return max(0, min(max_panels, endpoint - 1)), endpoint


def show(
    state: SearchState, session: Session, wanted: list[str], *,
    max_panels: int, target_token: str, image_note: str, image_note_target: str,
    visualization_mode: str,
) -> tuple[Any, str]:
    """Build an image of the named candidates. If empty, say why.

    What to show is decided by the policy; only the assembly is here. `target` may also be
    named: the target is shown as a separate image, without candidates
    (`state.render_target`). Otherwise it could not be looked at while there is no
    candidate with a mesh, which is exactly the first step of a part.

    A refusal to show must be loud. `Choice.dropped` used to be thrown away, and the
    assistant learned no reason: the candidate has no mesh, the name is not recognized,
    the renderer failed, more were named than fit in the image. From the dialogue side this
    looked like "asked and not shown", and a sensible model asks again.

    The reasons are worded in OUR words instead of retelling `choice.dropped`: that one is
    written in Russian for the run log, its text would mix languages in an English prompt,
    and depending on someone else's wording would break the dialogue when it is edited.
    The other list goes to the log, for a human.
    """
    turn = session.turn
    wants_target = any(token.lower() == target_token for token in wanted)
    tokens = [token for token in wanted if token.lower() != target_token]

    ids = resolve_ids(state, tokens)
    problems = [f"{token}: no such candidate" for token in tokens
                if not any(cid.startswith(token) for cid in ids)]

    allowed, endpoint = _panel_budget(state, max_panels)
    if endpoint is not None and endpoint < 1 and (ids or wants_target):
        # The endpoint takes no images at all. Not a refusal of the turn: ask in text, as
        # with an exhausted visual budget. Separate from `allowed == 0`: with a cap of one
        # image there are no candidate panels, but the TARGET can still be shown as a
        # separate image.
        return nothing_drawn(
            session, turn,
            problems + ["the assistant endpoint takes no images in this run, "
                        "asking without a picture"])

    drawable = []
    for cid in ids:
        candidate = state.get(cid)
        if candidate is None or not candidate.mesh_path:
            # Nothing to draw: the code did not build, there is no mesh. This is our
            # knowledge, not the renderer's, and must be said before the call, not after.
            problems.append(f"{cid[:8]}: nothing to draw, it did not build")
        elif len(drawable) >= allowed:
            # Two DIFFERENT reasons under one number, named separately: our own constant
            # is fixed by editing the policy, the endpoint cap only by the run config.
            # Merged, they send people to fix the wrong knob.
            problems.append(
                f"{cid[:8]}: over the limit of {allowed} panels in one image"
                + ("" if allowed == max_panels else
                   f" (the assistant endpoint takes {endpoint} image"
                   f"{'' if endpoint == 1 else 's'} per request, "
                   f"and the target takes one of them)"))
        else:
            drawable.append(cid)

    if wants_target and drawable:
        # The target panel is already in the candidates collage, so a second image is
        # not needed for it, but silence is not allowed either: the request was named.
        problems.append("target: always shown as a panel next to the candidates")

    if not visual_affordable(state) and (drawable or wants_target):
        # A visual call is not covered by the remainder. We still need to ask, but in
        # text: `ask_agent` refuses BEFORE sending, while `BudgetExceeded` out of `plan()`
        # ends the part. An image request must not cost a part its remaining iterations.
        return nothing_drawn(
            session, turn,
            problems + ["no visual calls left in the budget, asking without a picture"])

    if not drawable and wants_target:
        return _show_target(state, session, problems, image_note_target)

    if drawable and state.render is None:
        return nothing_drawn(session, turn, problems + ["no renderer in this run"])

    choice = None
    if drawable:
        try:
            # The panel is labeled with the same id the text uses ("image 2 = c2"). By
            # default the renderer puts menu letters `A`/`B`, which appear nowhere in the
            # dialogue, and a thinking assistant wrote that it saw only the target.
            choice = state.render(drawable, labels=[cid[:8] for cid in drawable],
                                  with_image=True, include_code=False,
                                  as_list=True, visualization_mode=visualization_mode)
        except Exception as exc:
            logger.warning("Dialogue image could not be built: %s", exc)
            problems.append("rendering failed")
            choice = None
    if choice is not None and choice.dropped:
        # Foreign reasons go to the log for a human, not to the assistant's prompt.
        logger.info("Render dropped candidates: %s", "; ".join(map(str, choice.dropped)))
    shown_ids = list(choice.ids) if choice is not None else []
    problems.extend(f"{cid[:8]}: could not be drawn" for cid in drawable
                    if cid not in shown_ids)

    if not shown_ids or choice is None or not choice.images:
        return nothing_drawn(session, turn, problems)

    # Numbering in the text is the order of images in the request: the target goes first,
    # then the candidates. If they diverged, a number would point at the wrong shape, the
    # same class of bug that made the menu stop being numbered.
    panels = ", ".join(
        ["image 1 = the TARGET shape"]
        + [f"image {index} = {cid[:8]}" for index, cid in enumerate(choice.ids, start=2)]
    )
    shown = ", ".join(cid[:8] for cid in choice.ids)
    # The image is not carried into the history (the endpoint takes one per call), but the
    # fact of showing is: without a note a late turn does not know it already saw the
    # shape and asks for it again.
    session.say(f"(you were shown rendered images of {shown} at turn {turn}"
                + (f"; {'; '.join(problems)}" if problems else "") + ")")
    return choice.images, image_note.format(panels=panels)


def _show_target(
    state: SearchState, session: Session, problems: list[str], image_note_target: str,
) -> tuple[Any, str]:
    """The target as a separate image. The only way to see it on the first turn."""
    image = None if state.render_target is None else state.render_target()
    if image is None:
        return nothing_drawn(session, session.turn,
                             problems + ["target: could not be drawn"])
    session.say(f"(you were shown the target shape at turn {session.turn}"
                + (f"; {'; '.join(problems)}" if problems else "") + ")")
    return image, image_note_target


# --- legality of actions and knobs ---------------------------------------------


def legal_picks(
    picks: list[Pick], menu: list[LegalAction], max_actions: int,
) -> tuple[list[tuple[Pick, LegalAction]], list[str]]:
    """Keep what was named from the shown list; return the rest by name.

    Matching is by the pair (tool, candidate) and by id prefix: the table labels a
    candidate with eight characters, and requiring the full id from the assistant would
    catch it on our own abbreviated notation.
    """
    legal: list[tuple[Pick, LegalAction]] = []
    dropped: list[str] = []
    for pick in picks[:max_actions]:
        match = next(
            (action for action in menu
             if action.tool == pick.tool
             and (action.parent_id or "").startswith(pick.candidate)),
            None,
        )
        if match is None:
            dropped.append(f"tool={pick.tool} candidate={pick.candidate}")
            continue
        legal.append((pick, match))
    if len(picks) > max_actions:
        dropped.append(
            f"{len(picks) - max_actions} extra action(s) over the limit of {max_actions}"
        )
    return legal, dropped


def params_for(
    state: SearchState, picked: LegalAction, asked: dict[str, Any],
    dropped: list[str] | None = None,
) -> dict[str, Any]:
    """Action knobs: those named by the assistant, but only the ones the seam knows.

    An unknown knob name is dropped here instead of the harness rejecting the whole action:
    the assistant, unlike a policy mutant, has no second attempt inside a turn, and losing a
    whole turn to a typo costs more than going with the defaults.

    But dropping SILENTLY is not allowed: `temp=` instead of `temperature=` used to pass
    this way many times and in the log is indistinguishable from "the assistant did not
    name a temperature". The names of dropped knobs go to `dropped` and from there into the
    turn transcript, the same way an unoffered candidate is reported.
    """
    info = state.tools.get(picked.tool)
    known = set(info.params) if info else set()
    params = {key: value for key, value in asked.items() if key in known}
    if dropped is not None:
        dropped.extend(f"{picked.tool}.{key}" for key in asked if key not in known)
    return params


def param_hint(key: str, spec) -> str:
    """A knob with type and bounds: without them the assistant sends `temperature=12`."""
    bounds = ""
    if spec.lo is not None or spec.hi is not None:
        # A strict lower bound is shown with the sign ">", not as a value: `top_p 0.0..1.0`
        # reads as "zero is allowed", the assistant named it, and the endpoint then rejected
        # the request.
        low = "" if spec.lo is None else (f">{spec.lo}" if spec.lo_exclusive else f"{spec.lo}")
        bounds = f" {low}..{'' if spec.hi is None else spec.hi}"
    return f"{key}({spec.type}{bounds})"


def param_value(value: Any) -> str:
    """A knob value for the transcript, not for the harness.

    Long values are truncated: a foreign multi-line value written into the history verbatim
    would be repeated in EVERY following prompt of the part.
    """
    text = str(value).replace("\n", " ")
    return text if len(text) <= 60 else f"{text[:57]}..."


# --- number formatting -------------------------------------------------------------


def metrics_text(metrics: dict[str, Any] | None, columns) -> str:
    """Metrics as a string by the policy's columns. A dash where there is no number.

    Not only the objective: the run measures both IoU and GMS, and showing one meant the
    assistant could not tell "the volume matched, the shape differs" from real progress.
    A dash differs from zero on purpose.
    """
    parts = []
    for title, key in columns:
        value = (metrics or {}).get(key)
        parts.append(f"{title}={float(value):.3f}" if value is not None else f"{title}=-")
    return " ".join(parts)


def status(candidate: Candidate, state: SearchState) -> str:
    """The candidate state in the words of the table."""
    # The root is named separately and first. By other signs it looks like an ordinary
    # candidate ("valid" with an empty measurement), and that an action on it STARTS OVER
    # rather than continues does not follow from the table anywhere.
    if candidate.depth == 0:
        return "empty start: from scratch"
    if not candidate.built:
        return f"failed: {candidate.failure or 'unknown'}"
    if candidate.failure is not None:
        return f"invalid: {candidate.failure}"
    return "valid, best so far" if candidate.id == state.best_id else "valid"


def prefix_facts(
    state: SearchState, candidate: Candidate, dead_step_eps: float,
) -> tuple[float | None, int]:
    """Where the branch started and how many of its operations gave nothing.

    Two numbers per one read lineage, because both answer one question of the assistant
    (is it worth continuing THIS branch) and it cannot reconstruct either now: the table
    has the current number and the operation count, but not where the branch started or
    what was wasted on the way.

    Why exactly these two. The first operation sets a ceiling that is not closed later: a
    branch that starts low ends lower than one that starts high, and the order of the bands
    does not flip. And a share of operations do not move the objective at all and stay in
    the code forever: there is no tool to delete operations.

    Computed on the HARNESS scale, the same as in the `score` column, otherwise the line
    would compare numbers from different scales. The threshold below which an operation
    counts as giving nothing belongs to the policy and comes as an argument.
    """
    chain = state.lineage(candidate)
    if not chain:
        return None, 0
    first = state.fitness_of(chain[0])
    dead = 0
    previous = 0.0
    for step in chain:
        value = state.fitness_of(step)
        if value is None or value - previous < dead_step_eps:
            dead += 1
        if value is not None:
            previous = value
    return first, dead
