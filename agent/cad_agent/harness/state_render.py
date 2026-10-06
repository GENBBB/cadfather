"""Render the search state into something that can be shown to the assistant.

**The harness substitutes the data into the prompt, while the framing and the
instruction belong to the policy.** The reason is not a division of labor but
divergence: the algorithmic variant of a block reads `SearchState` fields, the
assistant variant builds a string about them, and if the policy builds the string
the two variants of one block start talking about different things, silently.

The second pair that must not drift apart also lives here: **labels in the text
and labels on the image**. Panel `B` in the collage and the `Candidate B` block in
the prompt must be the same candidate, so both are built in one call and returned
together with a "label -> identifier" mapping. Letters are the default for a menu
with a letter answer; a caller whose text names candidates differently (the
dialogue uses identifiers) passes its own labels via `labels`, otherwise a panel
is captioned with a letter that is not in the text.

What this module does NOT do:

- **It does not parse the answer.** The prompt dictates the answer format, the
  prompt belongs to the policy, so parsing does too;
- **It does not choose whom to show.** The candidate list comes from the policy:
  how many and in what order is its decision about the price of the question;
- **It does not set the question text.** Only data blocks are returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from cad_agent.capabilities import code as code_utils
from cad_agent.harness.search_types import Candidate

logger = logging.getLogger(__name__)

# Candidate labels. One Latin letter each: a short model answer ("B") is caught by
# the parser more reliably than any number, and the collage panels carry the same letters.
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class Choice:
    """A ready set of data about one group of candidates.

    `ids` and `labels` are parallel lists: `labels[i]` captions `ids[i]` both in the
    text and on the image. Candidates without a mesh are left out when an image was
    requested: there is nothing to draw a panel from, and misaligned captions are
    worse than a missing candidate.
    """

    ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    text: str = ""
    parent_code: str = ""
    image: Any = None
    # The same panels, but EACH as a separate image (`as_list=True`): the target
    # first, then the candidates in `ids` order. Filled instead of `image`, not
    # together with it: two forms of one set would cost two renders, and the
    # reader sends only one of them anyway.
    images: list[Any] = field(default_factory=list)
    # How many coordinate lists were collapsed. A nonzero value obliges the policy
    # to say so in the prompt: a collapse that was not announced changes the
    # assistant's decision (it may consider the code corrupted), an announced one does not.
    elided: int = 0
    # What the set lacks compared with what was requested, and why. Goes to the
    # policy journal, so that "the assistant chose" means the same in all records.
    dropped: list[str] = field(default_factory=list)

    def id_of(self, label: str) -> str | None:
        """Candidate identifier by the label the assistant returned."""
        label = (label or "").strip().upper()
        return self.ids[self.labels.index(label)] if label in self.labels else None

    def __bool__(self) -> bool:
        return bool(self.ids)

    @property
    def has_image(self) -> bool:
        """Whether there is anything to show, as a collage or as a list."""
        return self.image is not None or bool(self.images)


def step_code(candidate: Candidate, parent: Candidate | None) -> str:
    """What this candidate added to the parent's prefix.

    The previous harness stored a step in a separate field (`Branch.history[-1]`);
    in the search loop a candidate has only the accumulated code, and the "step" is
    the difference from the parent. Tools that rewrite the whole prefix (optimizer,
    repair) give no difference; for them it is more honest to show the last line
    than to pass off emptiness as "no changes".
    """
    code = candidate.code or ""
    parent_code = (parent.code if parent is not None else "") or ""
    if parent_code and code.startswith(parent_code):
        tail = code[len(parent_code):].strip()
        if tail:
            return tail
    lines = [line for line in code.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def build_target_renderer(res: Any, obs: Any) -> Callable[[], Any]:
    """Build `state.render_target`: the image of ONE target, with no candidates.

    Separate from `render` rather than a flag on it: that one builds a candidate
    set with labels and text blocks, and a "set without candidates" would be an
    empty `Choice`, which all readers already interpret as "nothing to show".
    Returns the image itself or `None`; labels and data blocks are irrelevant here,
    there is nothing to caption.
    """

    def render_target() -> Any:
        renderer = getattr(res, "render", None)
        if renderer is None or not hasattr(renderer, "target_image"):
            return None
        try:
            return renderer.target_image(obs.gt_mesh_path)
        except Exception as exc:
            # Same degradation as for the collage: a question without an image is
            # cheaper than a lost turn.
            logger.warning("Target image could not be built: %s", exc)
            return None

    return render_target


def build_renderer(res: Any, obs: Any, lookup: Callable[[str], Candidate | None]) -> Callable[..., Choice]:
    """Build `state.render` for one part.

    `lookup` goes to the harness's live pool, not to a copy: the policy addresses
    candidates by identifiers, and the same table the loop uses to check action
    legality must answer it.

    Note on cost: the collage render is paid in wall time and, since it is called
    from the policy block, lands in `genome_cpu`. This is intended: a question to
    the assistant with an image costs more than a text one in more than the call,
    and the cost axis must see it.
    """

    def render(
        ids: Sequence[str],
        *,
        labels: Sequence[str] | None = None,
        include_code: bool = True,
        notes: dict[str, str] | None = None,
        with_image: bool = True,
        as_list: bool = False,
        visualization_mode: str = "simple",
        elide: bool = True,
        elide_min_points: int = code_utils.DEFAULT_ELIDE_MIN_POINTS,
    ) -> Choice:
        choice = Choice()
        # Labels passed by the caller are parallel to `ids` and are bound to the
        # identifier BEFORE filtering: by position after filtering, a candidate
        # dropped from the middle would shift the captions onto its neighbors.
        given = dict(zip(ids, labels)) if labels is not None else None
        candidates: list[Candidate] = []
        for candidate_id in ids:
            candidate = lookup(candidate_id)
            if candidate is None:
                choice.dropped.append(f"{candidate_id}: not in the pool")
                continue
            if with_image and not candidate.mesh_path:
                # Without a mesh there is nothing to draw a panel from. Removing the
                # candidate entirely is more honest than leaving it in the text without
                # an image: the assistant would compare heterogeneous descriptions without knowing it.
                choice.dropped.append(f"{candidate_id}: no mesh for the image")
                continue
            candidates.append(candidate)

        if not candidates:
            return choice

        if given is not None:
            choice.dropped.extend(
                f"{candidate.id}: no label left" for candidate in candidates if candidate.id not in given
            )
            candidates = [candidate for candidate in candidates if candidate.id in given]
            if not candidates:
                return choice
        choice.ids = [candidate.id for candidate in candidates]
        choice.labels = ([given[candidate.id] for candidate in candidates] if given is not None
                         else LABELS[: len(candidates)])
        if len(choice.labels) < len(candidates):
            # Fewer labels than candidates: show exactly as many as can be captioned.
            # Silently collapsing the set to the "first N" is not allowed: the
            # policy must see that it asked about something other than intended.
            choice.dropped.extend(
                f"{candidate.id}: no label left" for candidate in candidates[len(choice.labels):]
            )
            candidates = candidates[: len(choice.labels)]
            choice.ids = [candidate.id for candidate in candidates]

        def shorten(code: str) -> str:
            if not elide:
                return code
            shortened, n = code_utils.elide_point_lists(code, min_points=elide_min_points)
            choice.elided += n
            return shortened

        parent = lookup(candidates[0].parent_id or "")
        # A shared parent is shown only when it really is shared: for candidates
        # with different parents the "previous code" is different code, and one of
        # them in the prompt would be plainly false.
        shared_parent = parent if all(c.parent_id == candidates[0].parent_id for c in candidates) else None
        choice.parent_code = shorten((shared_parent.code if shared_parent else "") or "")

        blocks: list[str] = []
        for label, candidate in zip(choice.labels, candidates):
            lines = [f"Candidate {label}", f"source: {candidate.origin.tool}"]
            note = (notes or {}).get(candidate.id)
            if note:
                lines.append(note)
            if include_code:
                body = shorten(step_code(candidate, lookup(candidate.parent_id or "")))
                lines.append(f"```python\n{body}\n```")
            blocks.append("\n".join(lines))
        choice.text = "\n\n".join(blocks)

        if with_image:
            renderer = getattr(res, "render", None)
            if renderer is None:
                choice.dropped.append("renderer unavailable: question without an image")
            else:
                try:
                    if as_list:
                        # A list of panels instead of a collage: each image has its own
                        # pixel budget on the server, so the panel resolution does not
                        # depend on how many candidates were requested.
                        choice.images = renderer.panel_images(
                            gt_mesh_path=obs.gt_mesh_path,
                            pred_mesh_paths=[candidate.mesh_path for candidate in candidates],
                            labels=list(choice.labels),
                            visualization_mode=visualization_mode,
                        )
                    else:
                        choice.image = renderer.selection_image(
                            gt_mesh_path=obs.gt_mesh_path,
                            pred_mesh_paths=[candidate.mesh_path for candidate in candidates],
                            labels=list(choice.labels),
                            visualization_mode=visualization_mode,
                        )
                except Exception as exc:
                    # A missing image is a degradation of the question, not a failure
                    # of the part: asking from code is cheaper than not asking at all.
                    logger.warning("Selection collage could not be built: %s", exc)
                    choice.dropped.append(f"image could not be built: {exc!r}"[:200])
        return choice

    return render
