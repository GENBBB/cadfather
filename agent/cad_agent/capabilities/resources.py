"""Assembly of `Resources`, the set of capabilities the scaffold receives.

The capabilities here are already bound to a specific part (its own renderer,
executor, directory) and wrapped with a budget counter. The scaffold sees only
calls and knows nothing about process pools, caches, or the fact that metrics are
computed in the same process as the build.

Counting wrappers live **here**, not inside the capabilities: a capability must
not know about the budget, and the budget must not depend on whether the author of
a capability remembered to increment it.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

from cad_agent.capabilities import code as code_utils
from cad_agent.capabilities import llm as llm_mod
from cad_agent.capabilities import objective as objective_mod
from cad_agent.capabilities.execute import EvalResult, EvalTask, Executor
from cad_agent.capabilities.propose import StepProposal, StepProposer
# Annotations only (a dataclass field and a parameter), and the module has
# `from __future__ import annotations`, so they are not evaluated. At module level
# this import pulled `pyvista` into everything that touches resources and made half
# of the local checks unrunnable without a graphics stack.
if TYPE_CHECKING:
    from cad_agent.capabilities.render import FigureRenderer
from cad_agent.harness.budget import Budget
from cad_agent.harness.config import run_tools
from cad_agent.harness.seeding import (
    DEFAULT_RUN_SEED,
    attempt_tag,
    derive_seed,
    figure_seed,
)

logger = logging.getLogger(__name__)


def _step_from_name(name_prefix: str) -> int | None:
    """Extract the step number from a name like `step007` if the scaffold did not pass it."""
    match = re.match(r"step(\d+)", name_prefix or "")
    return int(match.group(1)) if match else None


def _tag_from_name(name_prefix: str) -> str:
    """Remainder of the name after `stepNNN`: a beam branch or repair round label.

    Needed so that artifacts of different parents of one step do not overwrite each other.
    """
    return re.sub(r"^step\d+_?", "", name_prefix or "")


@contextmanager
def _noop():
    """Empty context: stages are not timed without a journal."""
    yield


def _clip_error(text: str | None, limit: int = 400) -> str | None:
    """Trace for the summary table: head and TAIL, not the first `limit` characters.

    The exception type and message are on the last line of a traceback, while the
    first lines are always the same `run_in_fork` / `_evaluate` frames. Cutting from
    the head would leave the useless part in `candidates.jsonl`: a fragment like
    `File "<string>", line 14, in <module>\\n  F` cannot distinguish a syntax error
    from a geometry failure, which is exactly what the table is used for. The full
    text lies next to it in `status.json`, but only at level `full`, so the table
    must be self-sufficient.
    """
    text = text or ""
    if len(text) <= limit:
        return text or None
    head = limit // 4
    tail = limit - head - 5
    return text[:head] + "\n...\n" + text[-tail:]


def _last_line(code: str) -> str:
    """The last non-empty line of the code, that is, the step this candidate is."""
    lines = [line for line in (code or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _column(results: list[EvalResult], field_name: str) -> list[Any] | None:
    """Metric column over a step's candidates; `None` if nobody computed it."""
    values = [(result.metrics or {}).get(field_name) for result in results]
    return values if any(value is not None for value in values) else None

# Hard cap on ONE det call. A long cap does not help against anything and only
# lengthens the tail: individual calls can run for many minutes and the part then
# sits idle on them for hours.
DEFAULT_DET_TIMEOUT = 180.0
# Cap on det per PART. Separate from the call timeout, because a call is not the
# unit of spending: the scaffold calls det at every step (`det_candidates`, up to
# `max_steps` times), and a part could legitimately burn hours without violating
# any timeout. This is how a run can "hang": nearly all parts finish while a few
# remaining ones sit in det, an hour each, with idle workers.
#
# Everything spent inside `algo_rebuild` counts, including GT frame warm-up: the
# part waits for that too.
DEFAULT_DET_BUDGET_SEC = 600.0

# Hard cap on ONE optimizer call. The earlier value was the default from
# `capabilities/optimize.py`, set before the channel had ever run. Calls that hit
# it returned nothing and consumed a large share of the `opt` latency and of the
# run wall time.
#
# The next value was the p95 of SUCCESSFUL calls. A cap taken as a "success
# quantile" protects the successes and does not ask what they cost: interrupted
# calls ate a noticeable part of the `optimize` stage and of the run wall time
# without returning a single candidate.
#
# The current value is chosen by the COST OF THE CUTOFF, not by a quantile:
# improvements live in short calls, and the long tail pays for everyone. The
# legality threshold by code size (`harness/tools.py`, `OPT_MAX_CODE_NUMBERS`)
# applies EARLIER and does not pay even these seconds; the timeout catches what
# the threshold let through.
#
# What this cap does NOT fix: an interrupted call kills the fork and everything
# found is lost, although the optimizer is iterative. A cap in steps
# (`optimize.DEFAULT_STEPS`) or returning the best at the deadline would return
# those seconds as candidates.
DEFAULT_OPT_TIMEOUT = 30.0


def _det_pool_unknown(*args: Any, **kwargs: Any) -> None:
    """Default answer: the deterministic branch pool was not counted."""
    return None


def _answer_not_truncated() -> bool:
    """Default answer: the assistant was not asked, there was nothing to cut off."""
    return False


def _no_tool_calls() -> list[dict[str, str]]:
    """Default answer: no functions were passed to the assistant."""
    return []


def _chat_text(history: list[dict[str, Any]] | None) -> str:
    """Chat messages as one text, for length estimation and for the part's log.

    Not for the model: it gets the history as messages. Function calls are printed
    as name and arguments, images as a marker (they do not enter the history).
    """
    parts = []
    for message in history or []:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(item.get("text", "[image]") for item in content
                               if isinstance(item, dict))
        lines = [f"=== {message.get('role')} ===", str(content or "")]
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            lines.append(f"<tool_call {function.get('name')}> {function.get('arguments')}")
        parts.append("\n".join(line for line in lines if line))
    return "\n\n".join(parts)


@dataclass
class Resources:
    """Capabilities of one part. The composition is set by the contract."""

    propose_steps: Callable[..., list[StepProposal]]
    evaluate: Callable[[list[EvalTask]], list[EvalResult]]
    ask_agent: Callable[..., str]
    algo_rebuild: Callable[..., list[str]]
    optimize: Callable[..., dict[str, Any]]
    render: FigureRenderer
    difficulty: Callable[[], dict[str, Any]]
    budget: Budget
    work_dir: Path
    # Where the execution fork writes candidate meshes. Usually tmpfs
    # (`harness/scratch.py`) rather than the run directory: a mesh is needed during
    # the rollout, and the run directory is on NFS. Empty means write next to the
    # part, as before (tests do this, they have no use for scratch).
    mesh_dir: Path | None = None
    # Which candidate meshes land in the part directory: `none`, `best` or `all`.
    # Decided by the harness from `logging.save_meshes`; it also puts the best mesh
    # there after the rollout; only candidate copies under `all` are made here.
    save_meshes: str = "best"
    # Remainder of the deterministic branch pool per parent: how many operations
    # there are not yet issued. Maintained by the harness, read by the search loop:
    # the pool is finite, and this is the natural boundary for resampling
    # (`SearchState.det_remaining`). The default answers "pool not counted": a
    # `Resources` built by hand without the deterministic branch (tests) need not
    # know anything about its pool. `default_factory` rather than a default value:
    # a function in a class attribute is a descriptor, and access through an
    # instance would pass `self` as its first argument. The factory puts it into
    # the INSTANCE attribute, where it stays a plain function.
    det_remaining: Callable[..., int | None] = field(default_factory=lambda: _det_pool_unknown)
    # Whether the assistant's LAST answer was cut off at the `max_tokens` cap.
    # A separate query rather than a field in the answer: `ask_agent` returns text,
    # and changing its type would mean rewriting the call in all policies at once
    # for a flag that one needs. The default answers "not asked".
    answer_truncated: Callable[[], bool] = field(default_factory=lambda: _answer_not_truncated)
    # Function calls of the assistant's last answer (`ask_agent(tools=...)`).
    # Same reasoning as for `answer_truncated`: a separate query, not a change of
    # the answer type. The default answers "no functions were called".
    answer_tool_calls: Callable[[], list[dict[str, str]]] = field(
        default_factory=lambda: _no_tool_calls)
    journal: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    # Reasoning mode as set by the RUN (`experiment.agent.thinking`):
    # `True`/`False` is set, `None` is unset and the server template decides.
    # A run condition, not a policy knob, but the policy must KNOW it: it tells
    # "cut off mid-reasoning" from "cut off in the answer" by the presence of
    # `</think>`, and with the mode off that tag naturally never appears, so every
    # cut-off would read as unfinished thinking.
    agent_thinking: bool | None = None
    # Cap on the answer at a decision turn as set by the run
    # (`experiment.agent.answer_max_tokens`); `None` means the policy's constant.
    agent_answer_max_tokens: int | None = None
    # How many images the assistant endpoint accepts per request
    # (`--limit-mm-per-prompt`, derived by `launch_plan.server_image_limit`).
    # A run condition, not a policy knob, but the policy must KNOW it: it sends
    # panels as a list, and the server rejects a request over the cap entirely, so
    # the turn ends with the outcome "assistant did not answer". `None` means no cap is set.
    agent_image_limit: int | None = None
    # Whether to warm the assistant's prefix cache before a question
    # (`experiment.agent.prewarm`). The policy knows it in order to pass `ask_agent`
    # the unchanging start of the question (`prefix`) only when it is expected.
    agent_prewarm: bool = False
    # Base for the scaffold's own draws on this part (stochastic selection of beam
    # branches). Computed by the harness: the scaffold may draw randomness but may
    # not set the base, otherwise mutants are compared on different inputs, which
    # is exactly the noise that seeding exists to remove.
    seed: int = 0

    def note(self, decision: str, **fields: Any) -> None:
        """Write the scaffold's decision and its rationale to the run journal.

        The contract requires that the log restore not only what happened but why
        the scaffold decided so. This is the only channel through which the policy
        writes to the log; everything else is written by the harness itself.
        """
        if self.journal is not None:
            self.journal.event("decision", decision=decision, **fields)

    def evaluate_codes(
        self,
        codes: list[str],
        gt_mesh_path: str | Path,
        name_prefix: str,
        measure: bool = True,
        step: int | None = None,
        needs: Sequence[str] = (),
    ) -> list[EvalResult]:
        """Convenience wrapper: list of codes -> execution and metrics.

        Meshes go to the part's scratch directory (tmpfs) under predictable names
        (`step007_2.stl`). They are copied to the run directory only with
        `logging.save_meshes: all`, under the same names so that a step directory
        reads well by eye. The accumulated prefix and the execution status are
        written here too: this is the last point where both the code and its
        result are known.

        ``needs`` are the metrics without which the scaffold cannot compare
        candidates (`objective.Objective.needs`). The scaffold orders them itself:
        the selection objective belongs to the policy, and its cost must be charged
        to it, not baked into the capability.
        """
        extended = bool(self.config.get("extended_metrics", False))
        # Observability is the harness's business, not the policy's: the scaffold
        # orders only what it cannot decide without (`objective.needs`). If the
        # config asks for CD for the log, it is added here, not in the scaffold.
        if bool((self.config.get("metrics") or {}).get("cd", False)):
            needs = tuple(needs) + (objective_mod.NEED_CD,)

        # Identical code is executed once. Deduplication is here, not only in the
        # generator, because duplicates also occur across sources: the deterministic
        # branch can propose exactly the same operation as the model, and repair can
        # return the code unchanged. Outwardly the order and length of the list are
        # preserved: scaffolds zip the results with their candidates.
        first_seen: dict[str, int] = {}
        order: list[int] = []  # position in `codes` -> position in `unique`
        unique: list[str] = []
        for code in codes:
            position = first_seen.get(code)
            if position is None:
                position = first_seen[code] = len(unique)
                unique.append(code)
            order.append(position)

        tasks = [
            EvalTask(
                task_id=f"{name_prefix}_{idx}",
                code=code,
                mesh_path=str(Path(self.mesh_dir or self.work_dir) / f"{name_prefix}_{idx}.stl"),
                gt_mesh_path=str(gt_mesh_path),
                measure=measure,
                extended=extended,
                needs=tuple(needs),
            )
            for idx, code in enumerate(unique)
        ]
        if self.journal is not None and len(unique) < len(codes):
            self.journal.event(
                "dedupe",
                step=step,
                name_prefix=name_prefix,
                submitted=len(codes),
                executed=len(unique),
            )
        evaluated = self.evaluate(tasks)
        results = [evaluated[position] for position in order]
        saved = self._save_meshes(evaluated)
        self._log_step_results(name_prefix, step, codes, results, saved)
        return results

    def _save_meshes(self, results: list[EvalResult]) -> dict[str, str]:
        """Copy candidate meshes into the part directory if `all` was requested.

        Returns the mapping "scratch -> copy in the log": the result itself keeps the
        path in scratch, because the scaffold works with it further (rendering the
        next step, choosing a point), and moving it to NFS for the log is pointless.
        """
        if self.save_meshes != "all" or self.mesh_dir is None:
            return {}
        saved: dict[str, str] = {}
        for result in results:
            if not result.mesh_path:
                continue
            source = Path(result.mesh_path)
            if not source.exists():
                continue
            target = Path(self.work_dir) / source.name
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                saved[str(source)] = str(target)
            except OSError:
                logger.warning("Failed to save the candidate mesh %s", source, exc_info=True)
        return saved

    def _log_step_results(
        self,
        name_prefix: str,
        step: int | None,
        codes: list[str],
        results: list[EvalResult],
        saved: dict[str, str] | None = None,
    ) -> None:
        if self.journal is None:
            return
        if step is None:
            step = _step_from_name(name_prefix)
        if step is None:
            return

        tag = _tag_from_name(name_prefix)
        # Candidate table: a line for each, with the full measurement. Written
        # before `save_step_output` and regardless of the log level: at level
        # `metrics` there are no per-step artifacts at all, while candidate
        # metrics are needed exactly where they are at `full`.
        self.journal.record_candidates(
            step,
            tag,
            [
                {
                    "index": idx,
                    # The step itself, not the whole prefix: at level `full` the prefix
                    # lies next to it as a file, and in the table it would drown the line.
                    "step_code": _last_line(code),
                    "code_chars": len(code),
                    "success": result.success,
                    "outcome": result.outcome,
                    "timed_out": result.timed_out,
                    "worker_died": result.worker_died,
                    "error": _clip_error(result.error),
                    "wall_sec": round(result.wall_sec, 4),
                    # The whole measurement as returned by the executor: both what
                    # selection used and what is observed. There is nothing to trim
                    # here: this is what the table exists for.
                    "metrics": result.metrics,
                    "mesh_path": (saved or {}).get(result.mesh_path or ""),
                }
                for idx, (code, result) in enumerate(zip(codes, results))
            ],
        )
        for idx, (code, result) in enumerate(zip(codes, results)):
            self.journal.save_step_output(
                step=step,
                index=idx,
                tag=tag,
                prefix_code=code,
                status={
                    "success": result.success,
                    "timed_out": result.timed_out,
                    "outcome": result.outcome,
                    "worker_died": result.worker_died,
                    "error": result.error,
                    # A path that outlives the run, not a scratch file in tmpfs:
                    # `null` means "the mesh was not saved" (`logging.save_meshes`),
                    # not "there was none".
                    "mesh_path": (saved or {}).get(result.mesh_path or ""),
                    "metrics": result.metrics,
                    "wall_sec": result.wall_sec,
                },
            )


def build_resources(
    *,
    figure_id: str,
    gt_mesh_path: Path,
    work_dir: Path,
    executor: Executor,
    renderer: FigureRenderer,
    proposer: StepProposer | None,
    agent_client: Any = None,
    agent_model: str | None = None,
    budget: Budget | None = None,
    config: dict[str, Any] | None = None,
    journal: Any = None,
    run_seed: int = DEFAULT_RUN_SEED,
    mesh_dir: Path | None = None,
    save_meshes: str = "best",
    agent_context_limit: int | None = None,
    agent_image_tokens: int = llm_mod.AGENT_IMAGE_TOKENS,
    agent_chars_per_token: float = llm_mod.CHARS_PER_TOKEN,
    agent_thinking: bool | None = None,
    agent_answer_max_tokens: int | None = None,
    agent_image_limit: int | None = None,
    agent_prewarm: bool = False,
    agent_dp_size: int | None = None,
) -> Resources:
    budget = budget or Budget()
    config = dict(config or {})
    # Through the same resolver as the search loop: "the optimizer exists" and
    # "the optimizer is in the toolset" are one fact, and two answers would drift apart silently.
    optimize_enabled = "optimize" in run_tools(config)

    # How many times the scaffold has already asked for a draw at this (step, branch, variant).
    attempts: dict[tuple[Any, str, int], int] = {}

    def propose_steps(
        pred_mesh_path: str | Path | None,
        prev_code: str,
        k: int,
        step: int | None = None,
        variant: int = 0,
        tag: str = "",
        attempt: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        """k variants of the next step. The draw seed is computed here.

        `temperature` and `top_p` are POLICY knobs, not the run's: the run has no
        such keys any more. If not named, the registry default (`ParamSpec.default`)
        applies, which the tool substituted on the policy's behalf.
        `temperature = 0` means greedy decoding: it is a particular knob value, not a
        separate mode, and `n` with it is still charged to the policy, although
        dedup collapses all copies into one.

        The scaffold has no `seed` knob and must not have one: the base is set by the
        harness (`harness.seeding.derive_seed`), otherwise mutants are compared at
        different points. The policy keeps `variant`: asking for a **different** draw
        of the same (part, step, branch). At zero temperature the point remains the
        only source of diversity, and a scaffold that resampled it gets different
        candidates without giving up determinism.
        """
        # The part wall-time cap is checked BEFORE the work, not after: the point of
        # the soft deadline is not to start a new unit, not to interrupt a started one.
        budget.check_wall("propose_steps")
        if proposer is None:
            raise RuntimeError("Step generator not connected: no generation endpoint")
        # A repeated call at the same (step, branch, variant) is a request for new
        # samples, not the same ones. The attempt number goes into the seed derivation,
        # otherwise the second request would return the first byte for byte, and
        # resampling would look done. The counter is per figure and deterministic, so
        # reproducibility does not suffer.
        attempt_key = (step, tag, int(variant))
        if attempt is None:
            # Older scaffolds do not track the attempt number, so we count it ourselves.
            attempt = attempts.get(attempt_key, 0)
            attempts[attempt_key] = attempt + 1
        else:
            # The search loop tracks the attempt number itself (`Origin.attempt`): its
            # parent is a candidate from the pool, not "step, branch", and with its own
            # counter it tells repeats apart more precisely. The closure counter stays
            # consistent, otherwise a mixed call would return what was already drawn.
            attempt = int(attempt)
            attempts[attempt_key] = max(attempts.get(attempt_key, 0), attempt + 1)
        seed = derive_seed(
            run_seed,
            figure_id,
            step=step,
            tag=attempt_tag(tag, attempt),
            variant=int(variant),
        )
        proposals = proposer.propose(
            gt_mesh_path=gt_mesh_path,
            pred_mesh_path=pred_mesh_path,
            prev_code=prev_code,
            k=k,
            seed=seed,
            step=step,
            tag=tag,
            temperature=temperature,
            top_p=top_p,
        )
        budget.spend("vlm", len(proposals))
        if journal is not None:
            journal.event(
                "propose_steps",
                step=step,
                requested=k,
                returned=len(proposals),
                failed=sum(1 for proposal in proposals if proposal.error),
                # The seed is written to the log: the part directory must show what exactly
                # the point was drawn with, otherwise "the run is reproducible" can only be
                # checked by rerunning.
                seed=seed,
                variant=int(variant) or None,
                attempt=attempt or None,
                # What the policy named itself. `None` means not named, the registry default
                # applies. The run can no longer override it: the keys
                # `generation.{greedy,temperature,top_p,top_k}` do not exist.
                temperature=temperature,
                top_p=top_p,
            )
        return proposals

    def evaluate(tasks: list[EvalTask]) -> list[EvalResult]:
        budget.spend("exec", len(tasks))
        stage = journal.stage("execute") if journal is not None else _noop()
        with stage:
            results = executor.evaluate(tasks)
        deaths = [r for r in results if r.worker_died]
        if journal is not None:
            journal.record_call("exec", latency_sec=sum(r.wall_sec for r in results), count=len(results))
            # Next to the total, the breakdown: body construction, mesh export and metric
            # computation cost different amounts and are fixed in different ways.
            journal.record_exec_phases(results)
            for result in deaths:
                journal.record_worker_death(result.outcome)
            journal.record_fork_threads("exec", [r.n_threads for r in results if r.n_threads])
            # GT cache inside the fork: hits occur only with `proxy_pool`, where what the
            # shim warmed up is inherited by the grandchild. A zero share with this
            # backend means the warm-up does not arrive, and that is directly the cost of
            # reloading and resampling GT for every candidate.
            journal.record_cache(
                "exec_gt",
                hits=sum((r.gt_cache or {}).get("hits", 0) for r in results),
                misses=sum((r.gt_cache or {}).get("misses", 0) for r in results),
            )
            journal.record_executor_stats(executor.stats())
            journal.event(
                "evaluate",
                n=len(tasks),
                ok=sum(1 for r in results if r.success),
                timeouts=sum(1 for r in results if r.timed_out),
                # Executor deaths are written to the event separately from code failures:
                # `events.jsonl` must show what exactly spoiled the step, geometry or the machine.
                worker_deaths=[result.outcome for result in deaths] or None,
                cd=[(r.metrics or {}).get("cd_runtime") for r in results],
                # Objective metrics are written next to CD and only when requested:
                # `events.jsonl` must show by which numbers the scaffold selected at this
                # step, not only by which it could have.
                iou=_column(results, "iou"),
                gms_norm=_column(results, "gms_norm"),
            )
        return results

    # The number of calls to the agent on this part: the request seed is derived
    # from it. The order of calls is deterministic for deterministic input, so a
    # counter is enough, and it keeps two different questions from sharing one draw.
    agent_calls: dict[str, int] = {"n": 0}

    # How the last assistant answer ended. Lives exactly as long as the part, like
    # the resources themselves: the flag refers to the LAST question, and state that
    # outlives a unit of work would answer about another part.
    last_answer: dict[str, Any] = {"truncated": False, "tool_calls": []}

    def answer_truncated() -> bool:
        """Whether the last assistant answer was cut off at the answer cap.

        Ask right after `ask_agent`: the next question overwrites the flag. If there
        was no question or it did not reach the endpoint, `False`.
        """
        return last_answer["truncated"]

    def answer_tool_calls() -> list[dict[str, str]]:
        """Function calls of the last answer, as parsed by the server.

        A separate query for the same reason as `answer_truncated`: `ask_agent`
        returns text, and changing its type would mean rewriting all callers for a
        field that two policies need. Ask right after `ask_agent`; if there was no
        question or no functions were passed, the result is empty.
        """
        return [dict(item) for item in last_answer["tool_calls"]]

    # The assistant's DP replica the part is pinned to (`X-data-parallel-rank`).
    # Each replica has its own prefix cache, and the balancer does not keep
    # affinity: without pinning, the warm-up and the question go to different
    # replicas. Only when warming up: pinning spoils load balance, and without
    # warm-up there is no reason to pay for it.
    dp_rank = (zlib.crc32(str(figure_id).encode()) % int(agent_dp_size)
               if agent_prewarm and agent_dp_size and int(agent_dp_size) > 1 else None)
    # The last warmed-up prefix: a re-ask within a turn carries the same one, and
    # the state at its end is already in the cache.
    prewarmed: dict[str, Any] = {"prefix": None}

    def prewarm(prefix: str, prompt: str, image: Any, generation_kwargs: dict[str, Any],
                history: list[dict[str, Any]] | None) -> None:
        """A request with the unchanging start of the question and a one-token answer, before the question.

        Not charged to the budget and does not fail the turn: it is the server's
        concern, not the assistant's decision. Its cost is a stage of its own and the
        `agent_prewarm` event, separate from the question.
        """
        if prefix == prewarmed["prefix"]:
            return
        content = llm_mod.prewarm_content(prompt, image, prefix)
        if content is None:
            if journal is not None:
                journal.event("agent_prewarm", skipped="not_a_prefix", prefix_chars=len(prefix))
            return
        stage = journal.stage("agent_prewarm") if journal is not None else _noop()
        with stage:
            call = llm_mod.call_prewarm(agent_client, agent_model, content, generation_kwargs,
                                        history=history)
        if call.ok:
            prewarmed["prefix"] = prefix
        if journal is not None:
            journal.record_call("agent_prewarm", latency_sec=call.latency_sec)
            journal.event("agent_prewarm", prefix_chars=len(prefix),
                          images=prefix.count(llm_mod.IMAGE_SLOT), dp_rank=dp_rank,
                          prompt_tokens=call.prompt_tokens, cached_tokens=call.cached_tokens,
                          latency_sec=round(call.latency_sec, 4), error=call.error)

    def ask_agent(
        prompt: str,
        image: Any = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
        thinking: bool | None = None,
        purpose: str = "decide",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        history: list[dict[str, Any]] | None = None,
        prefix: str | None = None,
    ) -> str:
        """A question to the assistant. `image` is one image, a list of images or nothing.

        `thinking` is the answer mode for ONE call: `None` (default) means "as set by
        the run" (`experiment.agent.thinking`), `False` means "answer without
        reasoning". The config key remains a run condition and still controls the
        dialogue; the override exists for a channel whose useful part of the answer
        is not human-readable text but code (`repair`), where reasoning eats the
        whole cap and nothing comes out.

        `purpose` is **why** we ask, and it is a MEASUREMENT, not a cap. Two quite
        different consumers go to the assistant: the scaffold asks "what to do next"
        (the answer is a decision read by the policy) and the `repair` tool asks to
        fix code (the answer is a program that gets executed). Their prompts differ
        too: for decisions it is the accumulated transcript with the pool table, for
        repair a template plus the candidate's code and nothing else.

        They are charged to one counter for now (`agent_text`), but must be measured
        separately: merged, they gave a plausible summary with a wrong diagnosis (a
        cut-off share per channel hid that it was moderate for decisions and
        near-total for repair, that is, the tool did not work at all).

        A list is NOT a collage: each image goes as a separate block and is
        preprocessed by the server on its own, so panels do not share a pixel budget.
        The price is different: the prompt length grows with the number of images, and
        the server must be started with `limit-mm-per-prompt` no less than their
        number, otherwise it rejects the request entirely.

        `tools`/`tool_choice` are functions the model may call (OpenAI format), and
        `history` is the chat messages before the question. They are needed by the
        dialogue with function calls (`policies/dialogue_lean.py`); other callers do
        not pass them, and the request goes out exactly as before. Parsed calls are
        `answer_tool_calls()`. The server must be started with
        `--enable-auto-tool-choice --tool-call-parser` when `tool_choice="auto"`,
        otherwise it rejects the request entirely (400); the config check
        (`config._check_server`) catches this before startup.

        `prefix` is the unchanging start of `prompt` (image markers in it are the first
        images of `image`). With `experiment.agent.prewarm` a warm-up of this start
        goes out before the question, and both requests go to the pinned replica;
        without the key `prefix` means nothing.
        """
        if agent_client is None or agent_model is None:
            raise RuntimeError("Decision agent not connected: no assistant endpoint")

        images = llm_mod.as_image_list(image)
        # The cap and the measurement go under one name: a separate entry for each
        # consumer. `agent_text`/`agent_visual` are questions of the SCAFFOLD ("what
        # next", "is it time to stop"), `agent_repair` is a request of the repair
        # tool. They had to be split not for tidiness: a shared call cap used to be
        # hit by many parts, and repair took calls from decisions (and vice versa),
        # which a merged counter does not show at all.
        if purpose == "repair":
            kind = "agent_repair"
        else:
            kind = "agent_visual" if images else "agent_text"

        # The call cap is checked BEFORE sending, for the same reason as the context
        # cap below: the policy calls this channel itself, and a post-check does not
        # hold it (see `Budget.check_calls`).
        budget.check_calls(kind)

        # The context cap is checked BEFORE sending and before charging the budget.
        # Not to save a round trip but because a length refusal is not an endpoint
        # failure, it is a question that is too long: the scaffold needs to be able to
        # ask the same thing shorter, and the cost counter must not count a call that
        # did not happen.
        #
        # The estimate is approximate (`estimate_prompt_tokens`), and this is
        # deliberate: only the server's tokenizer knows the exact length, and there
        # is a safety net anyway, since a 400 from the server raises the same
        # exception. An estimate miss costs one extra fallback, a check miss one
        # extra request.
        if agent_context_limit:
            allowance = int(agent_context_limit) - int(max_tokens) - llm_mod.CONTEXT_SAFETY_MARGIN
            # History and function descriptions are prompt too: the chat template puts
            # them into the same context. They are counted as text by the same estimate.
            sized = prompt
            if history or tools:
                sized = "\n".join([_chat_text(history), json.dumps(tools or [], ensure_ascii=False),
                                   prompt])
            estimate = llm_mod.estimate_prompt_tokens(
                sized,
                images=len(images),
                image_tokens=agent_image_tokens,
                chars_per_token=agent_chars_per_token,
            )
            if estimate > allowance:
                if journal is not None:
                    journal.event(
                        "ask_agent_refused",
                        call=kind,
                        prompt_chars=len(prompt),
                        images=len(images),
                        estimate_tokens=estimate,
                        allowance_tokens=allowance,
                        context_limit=int(agent_context_limit),
                    )
                raise llm_mod.PromptTooLarge(
                    f"agent prompt ~{estimate} tokens with a cap of {allowance} "
                    f"(context {int(agent_context_limit)}, reply {int(max_tokens)})"
                )
        budget.spend(kind)
        agent_calls["n"] += 1
        # Reset BEFORE the request: a question that did not reach the endpoint must
        # not answer with the previous flag, since "did not answer" and "answered
        # with a fragment" are cured differently.
        last_answer["truncated"] = False
        last_answer["tool_calls"] = []
        # The agent seed is always sent, not only at `temperature > 0`: the
        # temperature knob belongs to the policy, and leaving reproducibility to its
        # discretion is the same class as leaving it the seed.
        seed = derive_seed(run_seed, figure_id, step=None, tag="agent", variant=agent_calls["n"])
        generation_kwargs: dict[str, Any] = {
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
        }
        # Three states, not two. `None` means "the run said nothing about the mode,
        # the server template decides"; `True`/`False` means the run said it, and the
        # server must be told. If the value collapsed into `bool`, "said nothing" and
        # "turned off" were the same, and `thinking: false` sent NOTHING: the `Qwen3`
        # template reads a missing key as enabled (`enable_thinking is undefined
        # or ... is true`), so the run would reason while marking every call as off.
        want_thinking = agent_thinking if thinking is None else bool(thinking)
        if want_thinking is not None:
            # The reasoning mode is switched on by the chat template on the SERVER, not
            # by the prompt text: the key goes into `chat_template_kwargs` the same way
            # `top_k` goes into `extra_body` for the generator. What comes back, a
            # `<think>…</think>` block in the answer or an answer already parsed by the
            # server (`--reasoning-parser`), is decided by the server launch, and the
            # policy must parse both forms: here we are responsible only for the
            # request getting through.
            #
            # An explicit `False` is sent as a key, not as silence: the template default
            # is not ours, and "do not ask for reasoning" and "ask for it to be turned
            # off" are different requests. Silence is left for exactly one case: the
            # config has no `experiment.agent.thinking` key at all; earlier measurements
            # were taken on such configs, and substituting our value would change their
            # conditions retroactively.
            generation_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": bool(want_thinking)}
            }
        if tools:
            generation_kwargs["tools"] = tools
            if tool_choice is not None:
                generation_kwargs["tool_choice"] = tool_choice
        if dp_rank is not None:
            generation_kwargs["extra_headers"] = {"X-data-parallel-rank": str(dp_rank)}
        if agent_prewarm and prefix:
            prewarm(prefix, prompt, image, generation_kwargs, history)
        # `history` only when present: the `call_vision` stubs in checks repeat the
        # old signature, and an extra key would break them all at once.
        chat = {"history": history} if history else {}
        stage = journal.stage("agent") if journal is not None else _noop()
        with stage:
            call = llm_mod.call_vision(
                client=agent_client,
                model_name=agent_model,
                text=prompt,
                image=image,
                generation_kwargs=generation_kwargs,
                **chat,
            )
        last_answer["truncated"] = bool(call.truncated)
        last_answer["tool_calls"] = list(getattr(call, "tool_calls", None) or [])
        if journal is not None:
            journal.record_llm(kind, call)
            # The field is called `call`, not `kind`: `kind` is the name of the event
            # itself in the `journal.event` signature, and a parameter of that name breaks it.
            # `finish_reason` next to the answer: for a thinking model the first 200
            # characters are the start of the reasoning, and they cannot tell a full
            # answer from one cut off by the cap. The flag is asked from the endpoint.
            # `purpose` next to the channel: `events.jsonl` is read line by line, and
            # without it a policy question is indistinguishable from a repair request,
            # which have different prompts, answer lengths and cut-off shares.
            #
            # Two fields are written about the reasoning mode, and this is not
            # redundancy: `thinking` is what was asked for (exactly what went out as a
            # key; `null` means "there was no key in the request, the server template
            # decided"), `reasoned` is what ACTUALLY came back. When there was a single
            # field taken from the intent, a run marked all calls as off while
            # reasoning in each: the request did not reach the server, and the journal
            # repeated the request.
            #
            # `reasoned` is a fact from the answer, but an INCOMPLETE one: a cut-off in
            # the middle of reasoning leaves no closing tag, and such a call honestly
            # says `false`. Read it as a pair with `finish_reason`: `reasoned: false`
            # with `length` means "was thinking and did not finish", with `stop` it
            # means "did not think at all".
            #
            # `answer` is the first 200 characters, and the cut is not marked in any
            # way; `answer_chars` is the full length. Without it, analysis from the
            # journal classified truncated text: "no `ACTION`" came out many times more
            # often than the real rate, because `ACTION` stood past the 200th character.
            # The full answer is `agent/call_*_answer.txt`. The cut itself is not
            # lengthened: runs before and after are compared by this field.
            journal.event("ask_agent", call=kind, purpose=purpose,
                          prompt_chars=len(prompt),
                          images=len(images), thinking=want_thinking,
                          reasoned=bool(call.reasoning) or "</think>" in call.text,
                          finish_reason=call.finish_reason, truncated=call.truncated,
                          answer=call.text[:200], answer_chars=len(call.text),
                          # How many functions the model called through the SERVER's parsing and
                          # how many history messages went before the question. `null` means
                          # no functions were passed: this is how the event tells "did not
                          # call" from "there was nothing to call".
                          tool_calls=len(last_answer["tool_calls"]) if tools else None,
                          history=len(history) if history else None,
                          # Prefix cache hit for the call (`LLMCall.cached_tokens`): the
                          # share can be read without `/metrics` and per replica.
                          prompt_tokens=call.prompt_tokens, cached_tokens=call.cached_tokens)
            # The whole dialogue goes to disk, at level `full`. The event above keeps
            # the first 200 characters of the answer, and for a thinking model that is
            # the start of the reasoning: `events.jsonl` shows neither what the
            # assistant decided nor why. The reasoning is taken from the answer field
            # (a server with `--reasoning-parser` parsed it itself) or extracted from
            # the text (a server without a parser): the form depends on the server
            # launch, and the part directory must look the same in both.
            # `extract_think` leaves the opening tag in place (the generator's answer
            # has none at all, and there it goes unnoticed); here it is removed so that
            # both forms give the same file and do not differ by a markup line.
            thought = call.reasoning or code_utils.extract_think(call.text)
            # Function calls and history go into the same two files: otherwise the
            # dialogue with functions cannot be restored from the part directory (an
            # answer consisting only of calls would give an empty `_answer.txt`).
            saved_answer = call.text
            if last_answer["tool_calls"]:
                saved_answer = "\n".join(
                    [call.text] + [f"<tool_call {item['name']}> {item['arguments']}"
                                   for item in last_answer["tool_calls"]]).strip()
            saved_prompt = f"{_chat_text(history)}\n\n=== user ===\n{prompt}" if history else prompt
            journal.save_agent_call(
                index=agent_calls["n"],
                prompt=saved_prompt,
                answer=saved_answer,
                reasoning=thought.removeprefix("<think>").strip(),
                images=images,
            )
        if call.context_overflow:
            # The estimate did not catch it, the server did. The exception type is the
            # same so that the scaffold does not parse the error text: the reaction to
            # "too long" is the same whoever noticed the length.
            raise llm_mod.PromptTooLarge(f"the agent rejected the prompt by length: {call.error}")
        if not call.ok:
            raise RuntimeError(f"the decision agent did not answer: {call.error}")
        return call.text

    # det state for this part: whether the GT frame is warmed up and how many seconds
    # are already spent. Lives in the part's resource closure, like the part itself.
    det_state: dict[str, Any] = {"warm": False, "spent_sec": 0.0}

    # Deterministic branch pools per parent: the key is the parent (its mesh; empty =
    # "from scratch"), the value is the whole ranked list and a cursor into it.
    # Lives as long as the part, like everything else here.
    det_pools: dict[str, dict[str, Any]] = {}

    # Parents on which a det call did not fit into the timeout. The algorithm is
    # deterministic: the same GT and the same parent mesh give the same work, so a
    # repeat would hit the same cap and spend it again. The cost of the mistake is
    # not theoretical: parts piled up many timeouts, each at exactly the cap.
    #
    # Only the timeout is remembered. A fork crash and `success: False` still mean
    # "we do not know": a process death is a property of the machine and the moment,
    # not of the input, and the deterministic branch must not be closed forever because of it.
    det_timed_out: set[str] = set()

    def _det_key(pred_mesh_path: str | Path | None) -> str:
        """A parent as a pool key.

        A candidate's mesh is its identity within a part: the path is unique
        (`step007_2.stl`), and the det output depends exactly on this mesh and the
        GT. Hashing the parent's code is neither needed nor possible: `algo_rebuild`
        does not see the code at all.
        """
        return "" if pred_mesh_path is None else str(pred_mesh_path)

    def det_remaining(pred_mesh_path: str | Path | None = None) -> int | None:
        """How many operations are left in this parent's pool.

        `None` means the pool has not been counted yet, i.e. a call would be real and
        paid. Zero means there is nothing more to ask of this parent, and the search
        loop removes such an action from the legal ones (`SearchState.det_remaining`).
        Zero answers two cases: the pool is exhausted by the cursor, and a call on this
        parent hit the timeout. For the policy they mean the same, "nothing to catch
        here"; the journal tells them apart (`algo_rebuild.outcome`).
        """
        key = _det_key(pred_mesh_path)
        if key in det_timed_out:
            return 0
        pool = det_pools.get(key)
        return None if pool is None else max(0, len(pool["ops"]) - pool["cursor"])

    def _serve_det(key: str, k: int, requested_from_cache: bool) -> list[str]:
        """Hand out the next k operations from the pool and advance the cursor."""
        pool = det_pools[key]
        start = pool["cursor"]
        served = pool["ops"][start : start + max(0, int(k))]
        pool["cursor"] = start + len(served)
        if journal is not None and requested_from_cache:
            # A repeated request is a separate event, not a silent copy of
            # `algo_rebuild`: the journal must show that there was no call but candidates
            # appeared. Otherwise run analysis would see executions without the call that
            # produced them and put it down to a defect.
            journal.event(
                "det_pool",
                requested=int(k),
                served=len(served),
                remaining=len(pool["ops"]) - pool["cursor"],
                billed=False,
            )
        return served

    def algo_rebuild(pred_mesh_path: str | Path | None, k: int) -> list[str]:
        """Deterministic reconstruction in a separate process, with a timeout.

        Isolation is not overcaution: the detectors run voxelization, contouring and
        boolean operations, so they can both hang and crash the process. In the part's
        process such a call would stop its whole rollout.

        There are two caps, and they are about different things. `det_timeout_sec`
        limits ONE call: against a hung detector. `det_budget_sec` limits the PART:
        against an honestly working but slow one: a part makes up to `det_candidates`
        calls at every step, and without a part cap their sum is unbounded.

        The GT cache is warmed up once here as well: the fork inherits what is warmed up.

        The part wall-time cap (`budget.check_wall`) is the third and outermost: det
        is expensive but not the only item, and a part can overspend the wall time
        without exceeding either of its two caps.
        """
        # Before the work, not after: see `Budget.check_wall`. The check is needed here
        # alongside `propose_steps`, because an iteration with source `det` does not
        # call the generator at all, so on such an iteration a check only in
        # `propose_steps` would never fire.
        budget.check_wall("algo_rebuild")
        from cad_agent.capabilities import det
        from cad_agent.capabilities.execute import OUTCOME_TIMEOUT, WORKER_DEATHS, run_in_fork

        # This parent's pool is already counted: hand out the next slice and spend
        # nothing. The economics are fair: the deterministic branch's work was done
        # entirely in the first call, and charging for it a second time would inflate
        # the branch cost on the cost axis. But execution and metrics of each issued
        # candidate are still counted, since they are not free, and without that a
        # "free" slice would become a loophole: drain the pool into fifty candidates,
        # execute them all and show a zero price.
        key = _det_key(pred_mesh_path)
        if key in det_pools:
            return _serve_det(key, k, requested_from_cache=True)

        # The same parent has already eaten the whole timeout. There is nothing to
        # repeat: the input is the same, the algorithm is deterministic, and a second
        # call would buy the same empty answer for the same seconds. The refusal is
        # free: no call, no counter, no seconds of the part budget.
        #
        # The check is here, not only in the search loop, for two reasons: policies
        # also call `algo_rebuild` directly (`scaffold/policies`), and
        # `SearchState.det_remaining` is asked by the candidate's mesh while `det_cold`
        # goes with an empty key; for a parent with a mesh these two keys differ, and
        # from outside the refusal would not be visible.
        if key in det_timed_out:
            if journal is not None:
                # A separate outcome name, not `timeout`: that one counts timeouts that
                # HAPPENED, and merging them into one field would inflate their number by
                # every refused repeat.
                journal.event(
                    "algo_rebuild", requested=k, returned=0, timed_out=False,
                    outcome="timeout_cached", ok=False, billed=False,
                )
            return []

        execution = config.get("execution", {}) or {}
        det_budget = execution.get("det_budget_sec", DEFAULT_DET_BUDGET_SEC)
        det_budget = None if det_budget is None else float(det_budget)

        # The part cap is checked BEFORE warm-up and before charging the call: a
        # capability that was not called must not cost, otherwise `n_det` in the report
        # would count what did not happen, and the branch cost would drift with it.
        # We answer as a failed call does, with an empty list: the scaffold already
        # knows how to go on, and introducing a separate outcome in the contract for a
        # run setting is pointless.
        if det_budget is not None and det_state["spent_sec"] >= det_budget:
            if journal is not None:
                # The event name is the same as for a call that happened, otherwise run
                # analysis would lose the refusal entirely. `outcome` tells them apart:
                # merging an exhausted budget with a failed call would give a plausible
                # report with a wrong diagnosis.
                journal.event(
                    "algo_rebuild", requested=k, returned=0, timed_out=False,
                    outcome="budget_exhausted", ok=False,
                    spent_sec=round(float(det_state["spent_sec"]), 1), budget_sec=det_budget,
                )
            return []

        budget.spend("det")
        # The GT frame is an intermediate mesh just like candidates: it is derived
        # from GT, lives for one part and does not go to the log. So it belongs in
        # scratch (tmpfs), not in the run directory on NFS.
        cache_dir = Path(mesh_dir or work_dir) / "_det_gt_frame"
        timeout = float(execution.get("det_timeout_sec", DEFAULT_DET_TIMEOUT))
        # The remainder of the part budget trims the call timeout: otherwise the last
        # call before exhaustion could go out on the full timeout on top of the cap,
        # and "10 minutes per part" would mean 10 minutes plus one more call.
        # A floor of a second: a non-positive timeout is pointless to pass to a fork.
        # Because of it the last call of a part may overshoot the cap by a second,
        # which is nothing against a budget of hundreds of seconds.
        if det_budget is not None:
            timeout = max(1.0, min(timeout, det_budget - float(det_state["spent_sec"])))
        # The remainder of the PART WALL TIME does not trim the timeout, and this is a
        # decision, not an oversight: the wall cap is soft, it answers "may we start"
        # and does not interrupt a unit of work already begun. A part may overshoot it
        # by one call; trimming would turn a full call into a stump that cannot finish.
        max_faces = int(execution.get("det_residual_faces", det.RESIDUAL_MAX_FACES))

        stage = journal.stage("det") if journal is not None else _noop()
        started_figure_call = time.monotonic()
        if not det_state["warm"]:
            # Warm-up is also in a fork, with the same timeout, and this is not
            # overcaution: a part once hung exactly here. Warm-up ran IN THE PART'S
            # PROCESS, without a fork and without a timeout, the only non-isolated call
            # of the deterministic branch, and with it the whole rollout died, and then
            # the run: finished parts sat idle for an hour without a summary.
            #
            # The price of isolation was that nobody inherits the fork's memory: the
            # frame went to disk and survived calls, while the repair of a leaky GT died
            # with the warm-up, and on part of the set the 256^3 voxelization was paid
            # in EVERY det call. Fixed: the repair is also stored as a file
            # (`det._persist_repaired`), and its hits are visible in the `det_gt_repair`
            # cache; a miss there means it is being recomputed again.
            warm_ok, warm_payload, warm_outcome = run_in_fork(
                det.warm_gt_frame, (str(gt_mesh_path), str(cache_dir)), timeout
            )
            det_state["warm"] = True
            if not warm_ok:
                # Must not stay silent: without the GT frame all later det calls on
                # this part will be failures, and without this line they would look
                # like "the deterministic branch found nothing".
                if journal is not None:
                    journal.event("det_warm_failed", outcome=warm_outcome)

        started = time.monotonic()
        with stage:
            ok, payload, outcome = run_in_fork(
                det.det_rebuild_isolated,
                (
                    str(gt_mesh_path),
                    str(pred_mesh_path) if pred_mesh_path is not None else None,
                    str(cache_dir),
                    # The same timeout is passed inside too: the snapshot can wind up
                    # its loops in advance and return a partial result instead of being
                    # killed empty-handed.
                    timeout,
                    max_faces,
                ),
                timeout,
            )
        wall = time.monotonic() - started
        # Everything the part waited inside `algo_rebuild` goes to its account,
        # including GT frame warm-up. Counted by the parent's clock: a fork does not
        # return its time when killed, and exactly such a call eats the budget.
        det_state["spent_sec"] = float(det_state["spent_sec"]) + (
            time.monotonic() - started_figure_call
        )
        timed_out = outcome == OUTCOME_TIMEOUT
        # det lives in its own process and dies like the executor: its death must also
        # reach the statistics, not only the log.
        if journal is not None:
            # Latency must reach the journal: the price of ONE call is read as
            # `latency_sec.det / calls.det`, not `stages_sec.det`, since the stage
            # includes neighboring candidates of the part. Without the argument the
            # fraction was identically zero for the most expensive capability of the run.
            # The fork measures its own time; on death and timeout there is none, and
            # the parent's is taken.
            latency = wall
            if isinstance(payload, dict) and payload.get("wall_sec") is not None:
                latency = float(payload["wall_sec"])
            journal.record_call("det", latency_sec=latency)
            if isinstance(payload, dict):
                journal.record_fork_threads("det", [payload.get("n_threads")])
                det_cache = payload.get("gt_cache") or {}
                journal.record_cache(
                    "det_gt",
                    hits=int(det_cache.get("hits", 0)),
                    misses=int(det_cache.get("misses", 0)),
                )
                # The GT repair is a cache of its own, and it answers the question "did
                # the fork's work survive". A hit means the ready result was taken from
                # disk; a miss means it was repaired again, with 256^3 voxelization.
                repair_cache = payload.get("gt_repair_cache") or {}
                if any(repair_cache.values()):
                    journal.record_cache(
                        "det_gt_repair",
                        hits=int(repair_cache.get("hits", 0)),
                        misses=int(repair_cache.get("misses", 0)),
                        writes=int(repair_cache.get("writes", 0)),
                    )
            if outcome in WORKER_DEATHS:
                journal.record_worker_death(f"det_{outcome}")
            journal.event(
                "algo_rebuild",
                requested=k,
                # How many were found in total is the pool size, not what went out: the
                # slice is handed out by a cursor, and from one number they can no
                # longer be told apart.
                returned=len(payload.get("ops", [])) if ok and isinstance(payload, dict) else 0,
                timed_out=timed_out,
                outcome=outcome,
                ok=bool(ok),
            )
        if timed_out:
            # The parent is remembered, not the pool: there is no pool and will not be,
            # and a repeat from the same parent would buy an already paid empty answer.
            # A timeout trimmed by the budget remainder is remembered like a full one:
            # by this time the part has less than `det_timeout_sec` left for the whole
            # det, and the next call will hit either the same trimmed cap or
            # `budget_exhausted`.
            det_timed_out.add(key)
            return []
        if not ok:
            logger.warning("The deterministic branch crashed: %s", payload)
            return []
        if not payload.get("success"):
            logger.warning("The deterministic branch returned an error: %s", payload.get("error"))
            return []

        # The pool is stored whole, only the requested slice goes out. A failed call
        # does NOT create a pool: an empty list from a crashed fork or one interrupted
        # by timeout means "we do not know", not "nothing to offer", and remembering it
        # as a pool would close the deterministic branch for this parent forever
        # because of a single failure.
        det_pools[key] = {"ops": list(payload.get("ops", [])), "cursor": 0}
        return _serve_det(key, k, requested_from_cache=False)

    def optimize(code: str, **kwargs: Any) -> dict[str, Any]:
        from cad_agent.capabilities import optimize as optimize_mod

        # A disabled capability answers as a failed one does: the scaffold already
        # knows how to read `success: False` and go on. The contract has no separate
        # "no such capability", and adding one would change the interface for a run setting.
        if not optimize_enabled:
            if journal is not None:
                journal.event("optimize", enabled=False, ok=False,
                              reason="the optimize tool is not in experiment.tools")
            # No budget, no counter: a capability that was not called must not cost.
            # Otherwise `n_opt` in the report would count what did not happen, and the
            # branch cost would drift with it.
            return {"success": False, "error": "the optimize tool is not in experiment.tools"}

        # Before the work, not after, as in `algo_rebuild`: the optimizer also produces
        # a candidate and also goes into a fork for minutes. Earlier there was no such
        # check here at all, and the part wall cap was invisible to the optimizer.
        budget.check_wall("optimize")

        # No key = the run cap `DEFAULT_OPT_TIMEOUT`; an explicit `null` = "no run
        # cap", and then a call is held only by the module timeout and the wall
        # remainder. These two cases must be told apart: the default must protect, and
        # protection can be removed only by saying so out loud.
        opt_timeout = (config.get("execution", {}) or {}).get("opt_timeout_sec", DEFAULT_OPT_TIMEOUT)
        timeout = optimize_mod.DEFAULT_TIMEOUT if opt_timeout is None else float(opt_timeout)

        # The part wall remainder does NOT trim this cap; see `algo_rebuild` and
        # `Budget.check_wall`: the part wall is soft, and a call started before it is
        # exhausted is carried to the end. The overshoot is bounded from above by this
        # very cap, so its size is the price of softness.
        budget.spend("opt")
        if journal is not None:
            # The call is counted here, before launch: it is already spent however it
            # ends. Latency is added separately below (`count=0`) when it becomes known.
            journal.record_call("opt")
        # The harness cap cannot be overridden: if the caller named its own timeout,
        # the smaller one is taken. A cap that can be bypassed with an argument is not a cap.
        asked = kwargs.pop("timeout", None)
        if asked is not None:
            timeout = min(timeout, float(asked))
        stage = journal.stage("optimize") if journal is not None else _noop()
        with stage:
            result = optimize_mod.optimize_params(
                code=code, gt_mesh_path=gt_mesh_path, work_dir=Path(work_dir),
                timeout=timeout, **kwargs
            )
        if journal is not None:
            # The counter is already incremented above; only latency is added here.
            journal.record_call("opt", latency_sec=float(result.get("wall_sec") or 0.0), count=0)
            # The event is written on both success and refusal. Before that, optimize
            # calls were absent from `events.jsonl` altogether, and an optimizer failure
            # read only as "the stage took 0.05 s", that is, not at all.
            #
            # `desugared` and `desugar_sec` are about translating wrapped into a method
            # chain, without which the snapshot parser does not read our dialect
            # (`capabilities/desugar.py`). They must be visible separately: a translation
            # failure and an optimizer failure are different events with different
            # causes, and by `ok` alone they would look the same.
            #
            # `timeout_sec` and `timed_out` are always written: without them run analysis
            # cannot tell "the optimizer could not cope" from "it was interrupted at the
            # N-th second", and this very difference is what closed a defect.
            # `changed` tells whether the optimizer moved at least one number. With the
            # guard removed this is the only sign that the call did anything: before,
            # "nothing changed" was read from the guard not running and no refusal
            # occurring. Without it a successful idle run is indistinguishable from a
            # successful fit.
            journal.event("optimize", enabled=True, ok=bool(result.get("success")),
                          desugared=result.get("desugared"),
                          desugar_sec=result.get("desugar_sec"),
                          changed=result.get("changed"),
                          timeout_sec=round(timeout, 1),
                          timed_out=bool(result.get("timed_out")),
                          error=str(result.get("error"))[:200] if not result.get("success") else None)
        return result

    def difficulty() -> dict[str, Any]:
        from cad_agent.capabilities import difficulty as difficulty_mod

        return difficulty_mod.shape_features(gt_mesh_path)

    return Resources(
        propose_steps=propose_steps,
        evaluate=evaluate,
        ask_agent=ask_agent,
        answer_truncated=answer_truncated,
        answer_tool_calls=answer_tool_calls,
        algo_rebuild=algo_rebuild,
        det_remaining=det_remaining,
        optimize=optimize,
        render=renderer,
        difficulty=difficulty,
        budget=budget,
        work_dir=Path(work_dir),
        mesh_dir=Path(mesh_dir) if mesh_dir is not None else None,
        save_meshes=save_meshes,
        journal=journal,
        config=config,
        agent_thinking=agent_thinking,
        agent_answer_max_tokens=agent_answer_max_tokens,
        agent_image_limit=agent_image_limit,
        agent_prewarm=bool(agent_prewarm),
        seed=figure_seed(run_seed, figure_id),
    )
