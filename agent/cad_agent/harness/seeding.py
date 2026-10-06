"""Run seeds: the single place that decides what seeds the random choices.

Why the module. Runs were not reproducible: two series of the same branch on the
same set coincided bit for bit on only 8 parts out of 1000, and most of the
differences were explained by the **model receiving a different input**: the point
on the target surface was chosen by an unseeded global `random`. Per-part noise was
wider than any plausible gain from the harness, leaving nothing to compare
variants with.

**The seed belongs to the harness, not to the policy.** If the policy sets the draw
base, variants are compared under different inputs, which brings back exactly the
noise this is meant to remove. Same class as the budget and counters: out of the
policy's reach. The policy keeps a `variant` knob: asking for a *different* point
is its right (a search lever); setting the base is not.

The seed is computed **from (run, part, step, branch, variant)** rather than one
per part: the point must change between steps (otherwise the model sees the same
one at every step) and must not change between runs. That is exactly the
difference between "sampled by a local generator" and "sampled by the global one".

Why `hashlib` and not the builtin `hash`. `hash()` of a string is randomized by
`PYTHONHASHSEED` at every interpreter start: the seed would be stable within a run
and different between runs, so the defect would remain but stop being visible.
Checked by a test (`tests/seed_check.py`) that computes the seed in a separate
process with a different `PYTHONHASHSEED`.
"""

from __future__ import annotations

import hashlib

# Default run seed. The default is a number rather than "do not seed": reproducibility
# should be a property of an ordinary run, not of a special mode. Repeat measurements
# for estimating the noise floor use a **different** seed in the config
# (`experiment.seed`) instead of disabling seeding.
DEFAULT_RUN_SEED = 42


def derive_seed(
    run_seed: int,
    figure_id: str,
    step: int | None = None,
    tag: str = "",
    variant: int = 0,
) -> int:
    """Seed of one draw: `H(run_seed, figure_id, step, tag, variant)`.

    `tag` labels a beam branch or a repair circle: two parents of the same step
    must draw differently, otherwise beam branches pick the same point.

    Returns a 32-bit non-negative number, which both `np.random.seed` and the
    vLLM endpoint's `seed` accept.
    """
    blob = f"{int(run_seed)}|{figure_id}|{'' if step is None else int(step)}|{tag}|{int(variant)}"
    digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def attempt_tag(tag: str, attempt: int) -> str:
    """Draw label that accounts for the attempt number.

    One function for two places: `Resources.propose_steps`, which computes the seed,
    and the search loop, which records it in the candidate's provenance. Two
    expressions for one label would silently diverge, and the log would name a seed
    that drew nothing.
    """
    return tag if int(attempt) == 0 else f"{tag}#{int(attempt)}"


def figure_seed(run_seed: int, figure_id: str) -> int:
    """Base for the harness's own draws on this part.

    Needed where the harness itself draws randomness (stochastic branch selection
    under `selection: probs`). The base comes from the harness, so such a draw is
    reproducible even if the policy did not set its own `seed`.
    """
    return derive_seed(run_seed, figure_id, step=None, tag="scaffold", variant=0)
