"""Tool registry: what the search loop can run, and under which conditions.

A tool is described by data (`ToolSpec`) rather than by `if` branches in the loop,
so the loop knows only what it needs for the legality of an action: what the tool
needs as input, what it returns, whether its result can be extended, and what `n`
means.

Consequences: a non-stepwise tool (`terminal=True`) is never offered as a parent,
and a tool that returns a mesh without code (`produces="mesh"`) cannot be a parent
for `stepwise`, which requires `code`.

`n_semantics` exists because det is deterministic: there `n` is not "how many
samples to draw" but "how many to take from the top of the ranked list". The same
number means different things for two tools, and the policy must see that.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from cad_agent.capabilities import code as code_utils
from cad_agent.harness.search_types import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    Candidate,
    Failure,
    Origin,
    ParamSpec,
    ToolCost,
    ToolInfo,
    code_numbers,
)

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """Raw tool result: not yet executed and not yet measured."""

    code: str | None = None
    # Set only by a tool that returns a ready mesh (`produces` = "mesh"). For the
    # others the mesh appears after the code is executed.
    mesh_path: str | None = None
    # Fallback code: executed only if `code` built open (not watertight), and used
    # instead of it only if it is itself closed. A fix that repairs some cases and
    # breaks others therefore cannot break any.
    fallback_code: str | None = None


@dataclass
class ToolOutcome:
    """Outcome of a tool call, before its products are executed."""

    products: list[Product] = field(default_factory=list)
    failure: Failure | None = None
    # Extra information for the journal: how many were requested, what the tool
    # answered, why it refused.
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """Everything a tool executor needs besides the candidate itself."""

    obs: Any
    res: Any
    journal: Any = None


@dataclass(frozen=True)
class ToolSpec:
    """Tool description for the search loop."""

    name: str
    requires: str            # none | code | mesh | candidate
    produces: str            # code+mesh | mesh
    extendable: bool         # whether the result can be extended
    terminal: bool           # "one step and done"
    # samples     - `n` samples from the model, each its own draw;
    # rank_depth  - `n` operations from the top of an ALREADY computed list;
    # single      - exactly one result, `n` is meaningless.
    # A property of the tool, not a name in an `if` of the loop: "ask the optimizer
    # for five copies" is rejected by semantics, and the next single-result tool
    # needs no loop change.
    n_semantics: str
    run: Callable[[ToolContext, Candidate, dict[str, Any], int, Origin], ToolOutcome]
    # What the policy may turn on this tool. A key missing here means the parameter
    # is illegal: the harness rejects the action by name (`validate_params`). The
    # schema is a check, not a hint for a human.
    params_schema: dict[str, ParamSpec] = field(default_factory=dict)
    cost: ToolCost = field(default_factory=ToolCost)
    # Which kinds of cost one call moves. The loop uses it to drop the action from
    # the legal ones when the matching cap is exhausted, rather than finding out
    # from an exception after the charge.
    spends: tuple[str, ...] = ()
    # Which of them grow with `n`. Kept apart from `spends`: det makes one call
    # regardless of `n`, so capping its `n` by the remaining det calls would forbid
    # taking five pool operations when two calls remain. It does spend exactly `n`
    # executions.
    scales: tuple[str, ...] = ()
    # An extra condition on the parent beyond `requires`. `requires` describes the
    # SHAPE of the input, not its state: repair needs a candidate that did NOT
    # build, which the word "candidate" cannot express. It lives in the registry so
    # that adding a tool does not touch the loop.
    parent_predicate: Callable[[Candidate], bool] | None = None
    # A deterministic tool with the same parent and parameters returns the same
    # result. Repeating it is pointless, so the loop drops such an action from the
    # legal ones instead of paying for a copy.
    deterministic: bool = False
    # Name of the `SearchState` field that holds the remaining pool of THIS tool.
    # Empty = no pool, nothing to limit by.
    #
    # It lives in the registry so that a second `rank_depth` tool does not read
    # another tool's remainder and get dropped by someone else's exhausted pool.
    pool_field: str = ""
    # Whether the tool is in the run's tool set when the config says nothing about
    # it. Kept in the registry rather than in a config list, so a new tool needs one
    # edit, not two that could silently diverge.
    #
    # `opt_in=True` means "only on explicit request". It does NOT mean "broken": the
    # tool works, but its cost or applicability is not yet measured, so it must not
    # be silently included.
    opt_in: bool = False
    # Legality threshold on the parent's code: from this many numeric literals
    # (`search_types.code_numbers`) the tool is not offered on the candidate. A
    # separate field rather than `parent_predicate`: the policy (via the card) must
    # see the threshold, and a lambda does not carry over to the card.
    max_code_numbers: int | None = None

    def accepts(self, parent: Candidate | None) -> bool:
        """Whether this candidate is acceptable as a parent for this tool."""
        if self.requires == "none":
            # The tool needs no input, but a parent still exists: the result is
            # appended to it. This is the root of the part.
            return parent is not None and parent.depth == 0
        if parent is None:
            return False
        # The condition is "built", not "measured". These differ: a candidate may
        # have built and have no objective value (no metrics engine), and extending
        # it is legal since it has both code and mesh. The part root passes by the
        # same rule: code exists, no measurement and there cannot be one.
        #
        # A non-watertight candidate is an INVALID MODEL, not a candidate without a
        # number. Appending an operation to an open body builds on knowingly broken
        # geometry. The ban must live in the harness: the policy is rewritable, and a
        # rule that lives only there can be dropped silently.
        if self.requires == "code":
            ok = bool(parent.code) and parent.built
        elif self.requires == "mesh":
            ok = bool(parent.mesh_path) and parent.built
        elif self.requires == "candidate":
            ok = True
        else:
            raise ValueError(f"Unknown requires={self.requires!r} on tool {self.name}")
        if not ok:
            return False
        if self.extendable_parent_required and not parent.extendable:
            return False
        # Depth can grow only from a VALID model. Transforming tools (`repair`,
        # `optimize`) deliberately skip this: they rewrite the same prefix rather
        # than build on it, and repairing an invalid model is exactly their job.
        if self.grows_depth and parent.failure is not None:
            return False
        if self.too_many_numbers(parent):
            return False
        return self.parent_predicate is None or self.parent_predicate(parent)

    def too_many_numbers(self, parent: Candidate) -> bool:
        """Whether the parent's code reaches the `max_code_numbers` threshold."""
        return (
            self.max_code_numbers is not None
            and code_numbers(parent.code or "") >= self.max_code_numbers
        )

    @property
    def extendable_parent_required(self) -> bool:
        """Whether the tool grows depth, and so requires an extendable parent.

        Transforming tools (optimizer, repair) do not grow depth: they rewrite the
        same prefix. The depth cap does not apply to them, and a dead-end parent is
        enough as long as it has code.
        """
        return self.grows_depth

    @property
    def grows_depth(self) -> bool:
        return self.name in _DEPTH_GROWING


# Tools that append an operation to the prefix. The others rewrite existing code,
# so they do not change the candidate's depth and the depth cap does not apply to
# them (otherwise a candidate at the maximum depth could not even be repaired).
_DEPTH_GROWING = {"stepwise", "det_cold", "det_warm"}


def _append_op(parent_code: str, op: str) -> str:
    """Append the deterministic branch's operation to the parent's prefix."""
    return parent_code.rstrip() + "\n" + op + "\n"


def _run_stepwise(
    ctx: ToolContext, parent: Candidate, params: dict[str, Any], n: int, origin: Origin
) -> ToolOutcome:
    """K variants of the next step from the stepwise model, in ONE request for all K.

    The attempt number comes from outside (`origin.attempt`): the harness keeps it,
    and without it a repeated choice of the same parent would return the same draw.
    """
    # Sampling knobs are ALWAYS passed: the policy's value if it named one, the
    # registry default otherwise, so there is no second source of truth.
    effective = effective_params(REGISTRY["stepwise"], params)
    overrides = {key: effective[key] for key in ("temperature", "top_p")}
    proposals = ctx.res.propose_steps(
        parent.mesh_path,
        parent.code,
        n,
        step=parent.depth + 1,
        tag=params.get("tag", ""),
        variant=int(params.get("variant", 0)),
        attempt=origin.attempt,
        **overrides,
    )
    good = [
        proposal
        for proposal in proposals
        if proposal.error is None and proposal.step_code.strip()
    ]
    return ToolOutcome(
        products=[Product(code=proposal.full_code) for proposal in good],
        failure=None if good else "empty_generation",
        info={"requested": n, "returned": len(proposals), "usable": len(good), **overrides},
    )


def _run_det(
    ctx: ToolContext, parent: Candidate, params: dict[str, Any], n: int, origin: Origin, *, warm: bool
) -> ToolOutcome:
    """Slice of the deterministic branch's ranked pool.

    The pool is computed once per parent and then served by a cursor: a repeat
    takes the NEXT operations and does not count as a call (`Resources.algo_rebuild`).
    So there is no seed or attempt number here; the cursor is the differentiator.
    """
    pred_mesh_path = parent.mesh_path if warm else None
    ops = ctx.res.algo_rebuild(pred_mesh_path, n)
    parent_code = parent.code or ""
    products = []
    for op in ops:
        thinned = code_utils.thin_polygons(op)
        products.append(Product(
            code=_append_op(parent_code, op),
            fallback_code=_append_op(parent_code, thinned) if thinned != op else None,
        ))
    return ToolOutcome(
        products=products,
        failure=None if ops else "empty_generation",
        info={"requested": n, "returned": len(ops), "warm": warm},
    )


def _run_det_cold(ctx, parent, params, n, origin) -> ToolOutcome:
    return _run_det(ctx, parent, params, n, origin, warm=False)


def _run_det_warm(ctx, parent, params, n, origin) -> ToolOutcome:
    return _run_det(ctx, parent, params, n, origin, warm=True)


def _run_optimize(
    ctx: ToolContext, parent: Candidate, params: dict[str, Any], n: int, origin: Origin
) -> ToolOutcome:
    """Fit the numbers in a candidate's code. One call, one result.

    A capability disabled in the config answers `success: False`, the same as a
    failed one. There is no need to tell them apart here: the loop gets a candidate
    without products either way, and the reason goes to the journal.
    """
    result = ctx.res.optimize(parent.code)
    if not result.get("success") or not (result.get("code") or "").strip():
        # `optimizer_failed`, not `empty_generation`: this tool has no model that
        # could stay silent, and the generic name merged its refusal with the
        # generator's. The harness sees only `failure`.
        return ToolOutcome(
            products=[],
            failure="optimizer_failed",
            info={"error": str(result.get("error"))[:200], "desugared": result.get("desugared")},
        )
    return ToolOutcome(
        products=[Product(code=result["code"])],
        info={"desugared": result.get("desugared")},
    )


def _extract_repaired_code(raw_response: str) -> str:
    """Extract the code from an assistant answer. Parsing order is as in `agentic.py`.

    An empty string means "no code in the answer", not "the answer is empty".
    """
    code = code_utils.extract_code(raw_response)
    fence = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
    return fence.group(1).strip() if fence is not None else code.strip()


def _is_repaired_code(text: str) -> bool:
    """Whether the text looks like an executable repair, before we execute it.

    A thinking model starts its answer with reasoning, the chat template inserts
    the opening `<think>` itself, and only the closing tag remains in the answer;
    often the answer is cut by the cap mid-word. `extract_code` without a closing
    tag returns the text as is, so the harness would execute prose.

    Two checks, both literally the contract of `REPAIR_PROMPT`:

    - the text parses as Python (catches reasoning and code fragments);
    - the name `r` is bound somewhere (catches an answer that is valid Python but
      useless: `...` parses fine, yet execution fails on `namespace["r"]`).

    The second condition is deliberately broad: any binding, not only a top-level
    assignment, since dialect code writes `r` by unpacking, in loops and inside
    functions it calls at once. An error toward "execute" is cheaper: a candidate
    that did not build is a normal outcome with a diagnosis, while a rejected
    repair looks like a tool refusal.

    This is asked of the ANSWER, not of the spliced code (`_splice_repair`): after
    splicing, the parent's preamble binds `r`, and the second condition would be
    always true.

    The check does NOT claim the repair is successful: code that parses and names
    `r` may still fail to build, and that normal outcome must reach the executor
    and the metrics.
    """
    if not text.strip():
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == "r" and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )


def _looks_like_whole_program(text: str) -> bool:
    """The answer is a whole program, not a replacement line.

    The sign is the dialect preamble: the candidate's code starts with
    `import cadquery as cq` and `from cadgen.* import ...` lines, and no body line
    contains them. The prompt asks for one line, but a model that returned a whole
    program must not get a second copy of the preamble glued on.
    """
    return any(
        line.startswith(("import ", "from ")) for line in text.splitlines()
    )


def _splice_repair(parent_code: str, answer: str) -> str | None:
    """Splice the repair answer into the parent's code in place of its last line.

    Same as `stepwise` does (`propose.py`, `full_code`), for the same reason: the
    useful part of the answer is one step, while a whole program is executed.

    Splicing makes the earlier lines unchanged by construction, not by the
    prompt's request (a bare one-line answer would execute with an empty namespace
    and fail with `NameError`).

    `None` means "nothing to splice into": the parent has no code. A whole-program
    answer (`_looks_like_whole_program`) passes through unchanged.
    """
    if _looks_like_whole_program(answer):
        return answer
    lines = (parent_code or "").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    return "\n".join(lines[:-1] + [answer.strip()])


def _run_repair(
    ctx: ToolContext, parent: Candidate, params: dict[str, Any], n: int, origin: Origin
) -> ToolOutcome:
    """Repair failed code by hand of the assistant.

    The prompt arrives in the action parameters and belongs to the POLICY. The
    harness is responsible for the question being charged to the counter and
    logged, but not for what is written in it.
    """
    prompt = str(params.get("prompt") or "")
    if not prompt.strip():
        return ToolOutcome(products=[], failure="empty_generation", info={"error": "empty prompt"})

    # The repair answer is code, not text for a human, so thinking is turned off
    # for this one call (the run key `experiment.agent.thinking` still governs the
    # dialogue, where an assistant reads the answer and needs it). Here thinking ate
    # the whole answer cap.
    #
    # The answer cap must fit the part of the code that is asked back: the LAST
    # LINE, not the whole program (splicing is done by `_splice_repair`). A cap
    # smaller than the input guarantees a truncation. The policy's value stays the
    # lower bound; raising it to the size of what is asked is not "deciding for the
    # policy" but not sending a request that cannot succeed.
    #
    # The measure is the parent's last line, not the whole program: for det
    # candidates the profile line can be 10-15 KB.
    #
    # The upper bound is the same as in the knob schema: if the line is longer, the
    # candidate is simply too large to repair, and hitting the cap explicitly is
    # more honest than growing it silently.
    from cad_agent.capabilities import llm as llm_mod  # local import: the registry does not pull in `openai`

    floor = int(params.get("max_tokens", 1024))
    last_line = code_utils.get_last_step_code(parent.code) if (parent.code or "").strip() else ""
    need = int(len(last_line) / llm_mod.CHARS_PER_TOKEN * 1.2) + 256
    ceiling = int(REGISTRY["repair"].params_schema["max_tokens"].hi or floor)
    ask_kwargs: dict[str, Any] = {
        "max_tokens": min(max(floor, need), ceiling),
        "thinking": False,
        # Measurement of this channel must be separate from the harness's
        # questions: the prompt differs (template plus candidate code, no
        # transcript or images), the answer differs (a program, not a decision).
        "purpose": "repair",
    }
    if params.get("temperature") is not None:
        # Zero (the `ask_agent` default) means "most probable answer", so n
        # attempts give n copies. Asking repair for more than one attempt makes
        # sense only with a temperature, and the policy must name it itself.
        ask_kwargs["temperature"] = float(params["temperature"])

    products: list[Product] = []
    unusable = 0
    spliced = 0
    for _ in range(max(1, n)):
        answer = ctx.res.ask_agent(prompt, **ask_kwargs)
        code = _extract_repaired_code(answer)
        if not _is_repaired_code(code):
            unusable += 1 if answer.strip() else 0
            continue
        full = _splice_repair(parent.code or "", code)
        # Splicing returns `None` only when there is nothing to splice into (the
        # parent has no code), which `parent_predicate` does not let in. The case
        # is still handled: returning a bare fragment would bring back the failure
        # that splicing exists to prevent.
        if full is None:
            unusable += 1
            continue
        if full is not code:
            spliced += 1
        products.append(Product(code=full))
    # Two different outcomes: "the assistant did not answer" and "the assistant
    # answered with non-code" are cured differently, and the harness sees only
    # `failure`.
    failure = None
    if not products:
        failure = "unusable_answer" if unusable else "empty_generation"
    return ToolOutcome(
        products=products,
        failure=failure,
        info={"requested": n, "returned": len(products), "unusable": unusable,
              # How many answers had to be spliced, i.e. how well the prompt is
              # followed. Zero with non-empty `returned` means the model sends whole
              # programs and the contract should be reverted.
              "spliced": spliced,
              "temperature": ask_kwargs.get("temperature")},
    )


# Price list. The unit is the execution of a small candidate (about 0.7 s), not a
# second: pod throttling differs between runs, and weights in seconds would mean
# different things on different runs. Seconds are still recorded in separate
# fields (`wall_sec`, `wall_sec_per_n`), for a policy that counts wall time rather
# than calls, but they are an order of magnitude.
#
# `measured` is a property of the TOOL: all five are measured on live models, and
# the cost of `repair` and `optimize` depends on the input size, hence two numbers
# (see `ToolCost.wall_sec_hi`).

# Legality threshold of `optimize` on the number of numeric literals in the
# parent's code (`ToolSpec.max_code_numbers`). Above it the call almost always hit
# its cap and did not improve the candidate. A threshold is cheaper than a call
# cap: a refusal pays nothing, while the cap pays its full wait. The tool's gain
# depends on one or two parts, so the boundary between 200 and 300 lies within
# that spread.
OPT_MAX_CODE_NUMBERS = 200

REGISTRY: dict[str, ToolSpec] = {
    "stepwise": ToolSpec(
        name="stepwise",
        requires="code",
        produces="code+mesh",
        extendable=True,
        terminal=False,
        n_semantics="samples",
        run=_run_stepwise,
        params_schema={
            # The draw of the point. With `greedy: true` this is the ONLY source
            # of diversity in the generator: there is no sampling, and a repeat
            # without changing the variant returns the same answer.
            "variant": ParamSpec("a different draw of the step point", type="int", lo=0, hi=64),
            "tag": ParamSpec("branch tag in the journal and in the seed", type="str"),
            # Temperature and top_p. The upper bound 2.0 is the limit of the
            # OpenAI-compatible API; above it the endpoint refuses, and rejecting
            # the action is cheaper than rejecting the call.
            #
            # **The generation mode belongs to the policy**: `temperature = 0` is
            # greedy decoding, a particular value of the knob rather than a
            # separate mode. Defaults live here.
            "temperature": ParamSpec("generator temperature; 0 = greedy decoding",
                                     lo=0.0, hi=2.0, default=DEFAULT_TEMPERATURE),
            # The bound is strict from below: the endpoint requires `(0, 1]` and
            # rejects the whole request at `top_p = 0.0`. At zero temperature the
            # knob is not sent at all (`make_generation_kwargs`), so the check
            # guards only the sampling branch, where it is needed.
            "top_p": ParamSpec("generator nucleus sampling", lo=0.0, hi=1.0,
                               lo_exclusive=True, default=DEFAULT_TOP_P),
        },
        # K variants are ONE request to the model but K executions. The constant
        # part is the request itself (about 1.0 in units of a small candidate's
        # execution); the scaled part is the sample (about 0.4) and its execution
        # (stepwise code is small, hence exec exactly 1.0).
        #
        # The unit of the `vlm` counter stays the sample, not the request, as
        # `propose_steps` counts it, so `scales` below describes the counter, not the
        # price. They may differ: the counter answers "how much budget is spent",
        # the price "what it cost".
        cost=ToolCost(
            calls={"vlm": 1.0},
            per_n={"vlm": 0.4, "exec": 1.0},
            wall_sec=0.70, wall_sec_per_n=0.28,
            measured=True,
        ),
        spends=("vlm", "exec"),
        scales=("vlm", "exec"),
    ),
    "det_cold": ToolSpec(
        name="det_cold",
        requires="none",
        produces="code+mesh",
        extendable=True,
        terminal=False,
        n_semantics="rank_depth",
        run=_run_det_cold,
        pool_field="det_remaining",
        # The most expensive purchase in the registry (about 18 small-candidate
        # executions per call). A repeat is NOT cheaper than the first call: the
        # pool cursor makes only a repeat from the SAME parent cheaper, while
        # policies call det from new ones.
        #
        # `n` is the rank depth, not samples: the call is already paid, and each
        # further operation costs only its execution. That is dearer than
        # stepwise's: det code is larger, and exec grows with code size - hence 3.0.
        cost=ToolCost(
            calls={"det": 18.0},
            per_n={"exec": 3.0},
            wall_sec=12.5, wall_sec_per_n=2.1,
            measured=True,
        ),
        spends=("det", "exec"),
        scales=("exec",),
    ),
    "det_warm": ToolSpec(
        name="det_warm",
        requires="mesh",
        produces="code+mesh",
        extendable=True,
        terminal=False,
        n_semantics="rank_depth",
        run=_run_det_warm,
        pool_field="det_remaining",
        # The same price as the cold branch, and measured: a "warm" call comes from
        # a new parent and is computed anew. A cursor would make it cheap, but it
        # helps only on a repeat from the same parent.
        cost=ToolCost(
            calls={"det": 18.0},
            per_n={"exec": 3.0},
            wall_sec=12.5, wall_sec_per_n=2.1,
            measured=True,
        ),
        spends=("det", "exec"),
        scales=("exec",),
    ),
    "optimize": ToolSpec(
        name="optimize",
        requires="candidate",
        produces="code+mesh",
        extendable=True,
        terminal=False,
        n_semantics="single",
        run=_run_optimize,
        # Not in the default set: the cost is measured, but it is large. The
        # channel multiplies the wall time of a part and does not clear the noise
        # floor, and the direct contribution of its products to the best candidate
        # is negligible. Such a tool is turned on deliberately, by listing it in
        # `experiment.tools`, not by default.
        opt_in=True,
        # Deterministic: the same code in gives the same result. A repeat from
        # the same parent would return a copy for a full call, so the loop does
        # not offer it.
        deterministic=True,
        # Depth above zero: there is nothing to fit in an empty prefix.
        parent_predicate=lambda candidate: (
            bool(candidate.code) and candidate.built and candidate.depth > 0
        ),
        # On long code the call does not fit in `execution.opt_timeout_sec` and
        # almost never improves anything: the optimizer runs central differences
        # over every number, the cost grows with their count, and det profiles are
        # written as hundreds of vertices. Raising the call cap was rejected:
        # calls finished above the threshold do not improve either, so wall time
        # would be paid for nothing.
        max_code_numbers=OPT_MAX_CODE_NUMBERS,
        # Latency measured under a pool of 32 workers, not an isolated call: that
        # is how long a part waits in a run, and that is what wall-time planning
        # needs. `wall_sec_hi` is for the case when the code of a det candidate
        # (11-15 KB) is being fitted; it was measured with an older, larger call cap
        # and is an overestimate now. `n` is meaningless - one result; executing the
        # product costs as executing the parent, i.e. more than 1.0 on det
        # candidates.
        cost=ToolCost(
            calls={"opt": 20.0},
            per_n={"exec": 1.0},
            wall_sec=14.0, wall_sec_hi=29.7,
            measured=True,
        ),
        spends=("opt", "exec"),
        scales=(),
    ),
    "repair": ToolSpec(
        name="repair",
        requires="candidate",
        produces="code+mesh",
        extendable=True,
        terminal=False,
        n_semantics="samples",
        run=_run_repair,
        params_schema={
            "prompt": ParamSpec("question to the assistant; owned by the policy", type="str"),
            "max_tokens": ParamSpec("reply cap", type="int", lo=1, hi=8192),
            # Without a temperature `ask_agent` answers greedily (0.0), and `n > 1`
            # buys n copies of one answer for n calls.
            "temperature": ParamSpec("assistant answer temperature", lo=0.0, hi=2.0),
        },
        # It makes sense to repair what did not succeed, and that is TWO events:
        # the code crashed (`built` false), and the code ran but the prediction is
        # not watertight (`built` TRUE, the failure is visible only in `failure`).
        # The second bucket is larger than the first. The predicate used to be
        # `not candidate.built`, which was a harness prohibition rather than a policy
        # choice: no policy could reach the larger bucket.
        parent_predicate=lambda candidate: bool(candidate.code) and (
            not candidate.built or candidate.failure == "not_watertight"),
        # Latency measured under a pool of 32 workers: shorter for the greedy
        # line, longer where the prompt carries a det candidate's code
        # (11-15 KB).
        #
        # Everything is in `per_n`, nothing in `calls`, deliberately: each repair
        # sample is its OWN request to the assistant (`_run_repair` calls
        # `ask_agent` in a loop), unlike `stepwise`, where K variants go in one
        # request. Such a call has no constant part.
        #
        # The counter is `agent_repair`, its own and not shared with the harness's
        # decisions: a shared cap made repair take calls from decisions and vice
        # versa, and the merged counter shows neither.
        cost=ToolCost(
            per_n={"agent_repair": 17.0, "exec": 1.0},
            wall_sec_per_n=11.9, wall_sec_hi=23.7,
            measured=True,
        ),
        spends=("agent_repair", "exec"),
        scales=("agent_repair", "exec"),
    ),
}


def names() -> tuple[str, ...]:
    """All tools in the registry: what the config may choose from."""
    return tuple(REGISTRY)


def default_names() -> tuple[str, ...]:
    """The run's tool set when the config says nothing about it.

    The default is derived from the registry rather than kept as a list: a list
    would be a second source of truth and lag behind the first new tool.
    """
    return tuple(name for name, spec in REGISTRY.items() if not spec.opt_in)


def get(name: str) -> ToolSpec:
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Tool {name!r} is not in the registry. Available: {sorted(REGISTRY)}")
    return spec


def validate_params(spec: ToolSpec, params: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Check the action parameters against the tool's schema.

    Returns (accepted values, rejection reasons). A non-empty list of reasons is a
    rejected action, not cleaned-up parameters: a policy that asked for
    `temperature: 12` asks for something other than what we would execute, and
    executing something similar would lie in the journal. This is also why the loop
    does not clean the plan silently.

    `tag` is an exception to the "no key = illegal" rule: the loop sets it itself
    (`_run_actions`), after the check. The schema still describes it, since the
    policy may turn it too.
    """
    accepted: dict[str, Any] = {}
    reasons: list[str] = []
    for key, value in (params or {}).items():
        param = spec.params_schema.get(key)
        if param is None:
            allowed = sorted(spec.params_schema) or ["no knobs"]
            reasons.append(f"{key}: tool {spec.name} has no such knob (available: {allowed})")
            continue
        coerced, why = param.check(value)
        if why:
            reasons.append(f"{key}: {why}")
            continue
        accepted[key] = coerced
    return accepted, reasons


def effective_params(spec: ToolSpec, params: dict[str, Any] | None) -> dict[str, Any]:
    """Knob values that will actually go into the call: the named ones plus schema defaults.

    Combinations cannot be checked from `params` alone: an unnamed knob takes the
    REGISTRY default, and "the policy did not touch the temperature" means the
    default temperature, not its absence. `_run_stepwise` substitutes them by the
    same expression: one formula for two places, otherwise the check and the call
    drift apart silently.
    """
    given = params or {}
    return {
        key: (given[key] if given.get(key) is not None else param.default)
        for key, param in spec.params_schema.items()
    }


def check_combination(spec: ToolSpec, params: dict[str, Any] | None, n: int) -> str:
    """Combinations that are illegal for the endpoint although each knob alone is legal.

    An empty string means the combination is acceptable.

    `ParamSpec.check` validates knobs one at a time, which is not enough: vLLM
    rejects `n > 1` under greedy decoding (`n must be 1 when using greedy
    sampling`), while `temperature = 0` and `n = 13` are each in range.

    A mistake costs more than a lost round trip. `Resources.propose_steps` charges
    the budget by the number of ORDERED samples (`budget.spend("vlm",
    len(proposals))`), while a refusal returns n empty proposals, so a rejected
    request is paid for in full. It surfaced as `empty_generation`, which is
    indistinguishable from "the model returned nothing".

    Reject rather than silently fix (for example by cutting `n` to one): a policy
    that asked for 13 greedy samples asks for something other than what we would
    execute, and a substitution would give it a verdict "the knob changes
    nothing". `validate_params` rejects the whole action for the same reason.
    """
    if spec.n_semantics != "samples" or int(n) <= 1:
        return ""
    values = effective_params(spec, params)
    temperature = values.get("temperature")
    if temperature is not None and float(temperature) == 0.0:
        return (
            f"temperature=0 (greedy decoding) is incompatible with n={int(n)}: "
            f"the endpoint serves a greedy request only with n=1. "
            f"Either n=1 or temperature>0"
        )
    return ""


def card(spec: ToolSpec) -> ToolInfo:
    """Tool card for the policy: everything descriptive and nothing executable.

    A card rather than the `ToolSpec` itself: the spec holds `run`, a live function
    leading to the capabilities and the budget. Handing it to the policy would hand
    it a way around the harness.
    """
    return ToolInfo(
        name=spec.name,
        requires=spec.requires,
        produces=spec.produces,
        n_semantics=spec.n_semantics,
        extendable=spec.extendable,
        terminal=spec.terminal,
        deterministic=spec.deterministic,
        params=dict(spec.params_schema),
        cost=spec.cost,
        spends=spec.spends,
        scales=spec.scales,
        max_code_numbers=spec.max_code_numbers,
    )


def cards(names: Iterable[str] | None = None) -> dict[str, ToolInfo]:
    """Cards of the run's tool set: what the loop puts into `SearchState.tools`."""
    chosen = REGISTRY if names is None else {n: REGISTRY[n] for n in names if n in REGISTRY}
    return {name: card(spec) for name, spec in chosen.items()}


def price_list(names: Iterable[str] | None = None) -> dict[str, ToolCost]:
    """Price list for the policy: a copy, so that the policy does not edit the registry.

    By default the whole registry, but the search loop passes the run's tool set
    here. Otherwise the policy would read the price of a tool it will never be
    offered, and "expensive" would differ from "unavailable" only by an empty list
    of legal actions.
    """
    chosen = REGISTRY if names is None else {n: REGISTRY[n] for n in names if n in REGISTRY}
    return {name: spec.cost for name, spec in chosen.items()}
