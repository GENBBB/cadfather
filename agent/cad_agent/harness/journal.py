"""Log of one part's run: files on disk, simple formats.

The log is a directory, not a service: runs go on a cluster with no access to
external trackers, and will be read with scripts and by eye. Hence predictable
paths and `.txt` / `.json` / `.jsonl` / `.stl` / `.jpg`.

Requirement: self-sufficiency. From one part's directory the course of the
reconstruction must be recoverable entirely, without server logs and without
rerunning. So at level `full` it saves **what was actually given to the model**
(the image and the prompt text), not what was supposed to be given.

    figures/<figure_id>/
      events.jsonl          sequence of calls and harness decisions
      candidates.jsonl      one line per candidate: the whole measurement as
                            returned by the executor (`logging.candidates`)
      tech.json             calls, tokens, latency, retries, stages, caches
      journal.json          step-by-step summary from the scaffold
      best.py               best prefix
      agent/                dialogue with the decision assistant, one call per file:
        call_001_prompt.txt     the whole question, with the accumulated history
        call_001_answer.txt     raw answer, before the policy strips the reasoning
        call_001_reasoning.txt  reasoning, if the model returned it
        call_001_input_1.jpg    images actually given with this question
      step_001/
        prompt.txt          text given to the model
        input.jpg           image given to the model
        raw_answer.txt      raw answer
        reasoning.txt       reasoning, if the model returns it
        step.py             the DSL operation produced
        prefix.py           accumulated code after the step
        status.json         execution status, metrics, error trace

Levels (`logging.level` in the config):

``off``      nothing is written; technical metrics are computed in memory, they
             are cheap and without them branches cannot be compared;
``metrics``  `events.jsonl` and `tech.json`, without prompts and images;
``full``     everything listed above.

The candidate table is switched on by a separate knob (`logging.candidates`), not
by the level: it is needed on a mass run too, where `full` is too heavy on IO. A
line per candidate costs tens of bytes against hundreds of kilobytes for its image
and mesh.

The full level is expensive in IO (NFS), so it is enabled deliberately: `metrics`
on mass runs, `full` on subsets and investigations. In comparison runs the level
must be the same for both branches, otherwise the overhead leaks into the measurement.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cad_agent.capabilities.cache import CacheStats, as_dict, merge_stats
from cad_agent.harness.load import LoadSampler

logger = logging.getLogger(__name__)

LEVELS = ("off", "metrics", "full")


class FigureJournal:
    """Log of one part. Lives in the part's process and hands out a summary."""

    def __init__(
        self,
        figure_dir: Path | str,
        level: str = "metrics",
        profile: bool = False,
        image_format: str = "jpeg",
        candidates: bool = False,
    ):
        if level not in LEVELS:
            raise ValueError(f"Unknown logging level {level!r}; expected {list(LEVELS)}")

        self.figure_dir = Path(figure_dir)
        self.level = level
        self.profile = profile
        self.image_format = image_format
        # Per-figure candidate table (`candidates.jsonl`). A separate knob rather
        # than a log level: lines are cheap on disk (tens of bytes per candidate vs
        # hundreds of kilobytes for its image and mesh), but there is no reason to
        # pay for them on a mass run anyway.
        self.log_candidates = bool(candidates) and level != "off"
        self.n_candidates = 0
        self.started = time.monotonic()

        self.calls: dict[str, int] = {}
        self.tokens: dict[str, int] = {"prompt": 0, "completion": 0, "prompt_visual": 0, "completion_visual": 0}
        self.latency: dict[str, float] = {}
        self.attempts: dict[str, int] = {}
        # How many answers the channel got CUT OFF at the `max_tokens` cap.
        # Separate from errors: a cut-off is a successful call with an incomplete
        # answer, and summed with failures it would read as endpoint trouble, while
        # it is cured by the answer cap.
        self.truncated: dict[str, int] = {}
        # Tokens PER CHANNEL. `tokens` above are split by "prompt/answer" and by
        # "with image or without": the cost weights `cost_i` are derived from them,
        # and there is no channel there. But the question "how long is the answer
        # for this consumer" is asked precisely per channel: summed over the channel,
        # long repair answers that get cut off look like a moderate cut-off share.
        # Everything else (calls, latency, attempts, cut-offs) already lies per
        # channel, because repair has its own entry (`agent_repair`) rather than
        # sharing one with the harness decisions.
        self.tokens_by_call: dict[str, dict[str, int]] = {}
        self.errors: list[dict[str, Any]] = []
        # Calls are counted both by type (vlm / agent_text / agent_visual) and by
        # model name: types are needed for cost weights, models to see which
        # endpoint the traffic went to.
        self.calls_by_model: dict[str, int] = {}
        self.stages: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        # Executor deaths by reason. Counted always, at any log level: this is not a
        # convenience for reading but a correction to the validity of the measurement:
        # a failure caused by a process death says nothing about geometry.
        self.worker_deaths: dict[str, int] = {}
        self.executor_stats: dict[str, int] = {}
        self._load: LoadSampler | None = None
        # Execution fork threads: measured by the fork itself, because a background
        # sampler almost always misses a process that lives a second.
        self.fork_threads: dict[str, list[int]] = {}
        # Execution broken down by phases. `latency_sec.exec` answers "how much
        # does a candidate cost", and this answers "for what exactly"; without it
        # "execution is 39% of the run" does not hint what to fix: body
        # construction, mesh export or metric computation.
        self.exec_phases: dict[str, float] = {}
        # Cache hits. Counted always, at any log level: it is a pair of integers
        # per part, and without them "the cache works" cannot be told from "there is
        # as if no cache", and in the second case the run pays for what is already computed.
        self.caches: dict[str, CacheStats] = {}

        self._events_path = self.figure_dir / "events.jsonl"
        self._candidates_path = self.figure_dir / "candidates.jsonl"
        if self.level != "off":
            self.figure_dir.mkdir(parents=True, exist_ok=True)

    # --- events ------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        """Write an event: a capability call or a harness decision with its rationale."""
        if self.level == "off":
            return
        record = {"t": round(time.monotonic() - self.started, 4), "kind": kind, **fields}
        try:
            with open(self._events_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.warning("Could not write event %s", kind, exc_info=True)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a stage. Without `profile=True` it costs nothing but the call."""
        if not self.profile:
            yield
            return

        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            self.stage_counts[name] = self.stage_counts.get(name, 0) + 1

    # --- technical metrics --------------------------------------------------

    def watch_load(self, interval_sec: float = 5.0, enabled: bool = True) -> None:
        """Start watching this part's load (its process and descendants)."""
        if not enabled or self._load is not None:
            return
        self._load = LoadSampler(run_dir=None, interval_sec=interval_sec, enabled=True)
        self._load.start()

    def record_fork_threads(self, kind: str, values: list[int | None]) -> None:
        """Remember how many threads there were in forks of kind `kind`.

        There are two kinds, `exec` and `det`, and they are measured the same way:
        the fork reads its own `/proc/self/stat` before returning. The key is by
        kind rather than a separate field for each: a third fork must not require
        edits in four files.
        """
        bucket = self.fork_threads.setdefault(kind, [])
        bucket.extend(int(value) for value in values if value)

    def record_exec_phases(self, results: list[Any]) -> None:
        """Add up the execution breakdown by phases as the forks returned it.

        Counted over tasks that **ran to the end**: a failed candidate's fork
        returns an error text with no phases. So `n` here is not the number of
        `exec` calls but the number of candidates for which the breakdown was
        computed at all; `exec_phases` cannot be compared with `latency_sec.exec`
        directly without dividing by its own `n`.
        """
        for result in results:
            for key, value in (getattr(result, "phases_sec", None) or {}).items():
                self.exec_phases[key] = self.exec_phases.get(key, 0.0) + value

    def record_cache(self, name: str, hits: int = 0, misses: int = 0,
                     writes: int = 0, evictions: int = 0) -> None:
        """Add accesses to cache `name`."""
        stats = self.caches.setdefault(name, CacheStats())
        stats.hit(hits)
        stats.miss(misses)
        stats.write(writes)
        stats.evict(evictions)

    def record_cache_stats(self, stats: dict[str, CacheStats]) -> None:
        """Take counters from a cache that keeps them itself (renderer, generator).

        The cache does not know about the journal, the journal does not know the
        cache's structure; they meet here, in one call at the end of the part.
        """
        merge_stats(self.caches, stats)

    def record_worker_death(self, outcome: str, count: int = 1) -> None:
        """The executor died: timeout, signal, unreadable result, loss."""
        self.worker_deaths[outcome] = self.worker_deaths.get(outcome, 0) + count

    def record_executor_stats(self, stats: dict[str, int]) -> None:
        """What happened to the executor itself (shim restarts etc.)."""
        for key, value in (stats or {}).items():
            if value:
                self.executor_stats[key] = int(value)

    def record_call(self, kind: str, latency_sec: float = 0.0, count: int = 1) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + count
        self.latency[kind] = self.latency.get(kind, 0.0) + latency_sec

    def record_llm(self, kind: str, call: Any) -> None:
        """Account for a model call: tokens, latency, retries, errors.

        Visual tokens are counted as a separate pair of fields, from which the cost
        weights `cost_i` are later derived. Separately from them are the tokens per
        channel (`tokens_by_call`): they answer the question "how long is the answer
        for this consumer", which the "prompt/answer" breakdown does not answer.
        """
        self.record_call(kind, latency_sec=getattr(call, "latency_sec", 0.0))
        row = self.tokens_by_call.setdefault(kind, {"prompt": 0, "completion": 0})
        row["prompt"] += int(getattr(call, "prompt_tokens", None) or 0)
        row["completion"] += int(getattr(call, "completion_tokens", None) or 0)
        attempts = int(getattr(call, "attempts", 1) or 1)
        self.attempts[kind] = self.attempts.get(kind, 0) + attempts
        if attempts > 1:
            self.event("retry", call=kind, attempts=attempts, errors=getattr(call, "errors", [])[:3])

        model = str(getattr(call, "model", "") or "unknown model")
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

        suffix = "_visual" if getattr(call, "has_image", False) else ""
        for field, value in (("prompt", getattr(call, "prompt_tokens", None)),
                             ("completion", getattr(call, "completion_tokens", None))):
            if value:
                self.tokens[f"{field}{suffix}"] = self.tokens.get(f"{field}{suffix}", 0) + int(value)

        if getattr(call, "truncated", False):
            self.truncated[kind] = self.truncated.get(kind, 0) + 1

        if getattr(call, "error", None):
            self.errors.append({"call": kind, "error": str(call.error)})
            self.event("call_failed", call=kind, error=str(call.error))

    def record_candidates(self, step: int | None, tag: str, rows: list[dict[str, Any]]) -> None:
        """Append a line per candidate of the step to `candidates.jsonl`.

        Written by the **harness**, not the scaffold, and it writes what the
        executor returned: whether it built, how it ended, the whole measurement.
        The policy does not touch this table, otherwise a mutant selecting by
        numbers other than the ones it wrote itself would be indistinguishable
        from an honest one.

        The scaffold's decisions (what it expanded with, what it selected) are not
        included: their side of the journal is `journal.json` and `events.jsonl`.
        They are joined by the key (`step`, `tag`, `index`).
        """
        if not self.log_candidates or not rows:
            return
        self.n_candidates += len(rows)
        try:
            with open(self._candidates_path, "a", encoding="utf-8") as stream:
                for row in rows:
                    record = {"step": step, "tag": tag, **row}
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.warning("Could not write the candidates of step %s", step, exc_info=True)

    # --- per-step artifacts -------------------------------------------------

    def step_dir(self, step: int) -> Path:
        path = self.figure_dir / f"step_{step:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_step_input(self, step: int, prompt: str, image: Any = None, tag: str = "") -> None:
        """Save what was **actually given** to the model.

        `tag` distinguishes beam branches: several parents are expanded in one
        step, and without a tag their artifacts would overwrite each other.
        """
        if self.level != "full":
            return
        directory = self.step_dir(step)
        safe = _safe_part(tag)
        suffix = f"_{safe}" if safe else ""
        _write_text(directory / f"prompt{suffix}.txt", prompt)
        if image is not None:
            try:
                image.convert("RGB").save(directory / f"input{suffix}.{self.image_format}", quality=90)
            except Exception:
                logger.warning("Could not save the input image of step %d", step, exc_info=True)

    def save_agent_call(
        self,
        index: int,
        prompt: str,
        answer: str | None = None,
        reasoning: str | None = None,
        images: Any = (),
    ) -> None:
        """Save one question to the decision assistant and its whole answer.

        A file per call, not a shared transcript: the dialogue prompt already CARRIES
        the whole conversation history, so the last `call_NNN_prompt.txt` together
        with its answer is the whole dialogue, and gluing a second file from the
        same lines would give them a second owner.

        The answer is written RAW, with the `<think>` block if the server runs
        without `--reasoning-parser`. The reasoning is also stored as a separate
        file, because in the second form (the server parsed the answer itself) it
        is absent from the text entirely: without that file half of the runs would
        lose the reasoning silently, and the difference would read as a difference
        between models. The reasoning is not returned to the model; see
        `dialogue_io.visible`.
        """
        if self.level != "full":
            return
        directory = self.figure_dir / "agent"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"call_{index:03d}"
        _write_text(directory / f"{name}_prompt.txt", prompt)
        if answer is not None:
            _write_text(directory / f"{name}_answer.txt", answer)
        if reasoning:
            _write_text(directory / f"{name}_reasoning.txt", reasoning)
        for number, image in enumerate(images or (), start=1):
            try:
                image.convert("RGB").save(
                    directory / f"{name}_input_{number}.{self.image_format}", quality=90
                )
            except Exception:
                logger.warning(
                    "Could not save the image of assistant question %d", index, exc_info=True
                )

    def save_step_output(
        self,
        step: int,
        index: int = 0,
        tag: str = "",
        raw_answer: str | None = None,
        reasoning: str | None = None,
        step_code: str | None = None,
        prefix_code: str | None = None,
        status: dict[str, Any] | None = None,
    ) -> None:
        if self.level != "full":
            return
        directory = self.step_dir(step)
        parts = [part for part in (_safe_part(tag), str(index) if index else "") if part]
        suffix = ("_" + "_".join(parts)) if parts else ""
        if raw_answer is not None:
            _write_text(directory / f"raw_answer{suffix}.txt", raw_answer)
        if reasoning:
            _write_text(directory / f"reasoning{suffix}.txt", reasoning)
        if step_code is not None:
            _write_text(directory / f"step{suffix}.py", step_code)
        if prefix_code is not None:
            _write_text(directory / f"prefix{suffix}.py", prefix_code)
        if status is not None:
            _write_text(directory / f"status{suffix}.json", json.dumps(status, ensure_ascii=False, indent=2, default=str))

    # --- summary ------------------------------------------------------------

    def tech_metrics(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "calls_by_model": dict(self.calls_by_model),
            "tokens": {key: value for key, value in self.tokens.items() if value},
            "latency_sec": {key: round(value, 4) for key, value in self.latency.items()},
            "attempts": dict(self.attempts),
            "truncated": dict(self.truncated),
            "tokens_by_call": {
                kind: dict(row) for kind, row in sorted(self.tokens_by_call.items())
                if any(row.values())
            },
            "n_endpoint_errors": len(self.errors),
            "endpoint_errors": self.errors[:10],
            "stages_sec": {key: round(value, 4) for key, value in self.stages.items()},
            "stage_counts": dict(self.stage_counts),
            # Breakdown of `latency_sec.exec` by fork phases. The key `n` is the number
            # of candidates it was computed over (failed ones do not count).
            "exec_phases": {
                key: (int(value) if key == "n" else round(value, 4))
                for key, value in self.exec_phases.items()
            },
            "profile_enabled": self.profile,
            "level": self.level,
            # How many candidates reached `candidates.jsonl`. Zero with the knob
            # on means no step reached execution, which is not the same as "the
            # knob is off".
            "candidates_logged": self.n_candidates if self.log_candidates else None,
            "fork_threads": {
                kind: {
                    "max": max(values),
                    "mean": round(sum(values) / len(values), 2),
                    "n": len(values),
                }
                for kind, values in sorted(self.fork_threads.items())
                if values
            },
            # Cache hits: name -> hits/misses/hit_rate (+writes/evictions when
            # present). `hit_rate: null` means "no accesses", not "no hits": these
            # are different things.
            "caches": as_dict(self.caches),
            "worker_deaths": dict(self.worker_deaths),
            "n_worker_deaths": sum(self.worker_deaths.values()),
            "executor_stats": dict(self.executor_stats),
            # The load created by **this** part: the part's own process plus its
            # execution pool and forks. The **peak over the rollout** is taken, not a
            # snapshot at the end: by `finalize` the pool is already closed, and a
            # snapshot would show a lone process with five threads, that is, nothing.
            "load": self._load.summary() if self._load is not None else {},
        }

    def finalize(self) -> dict[str, Any]:
        """Append to `tech.json` and measure the size of the part's log."""
        if self._load is not None:
            self._load.stop()
        metrics = self.tech_metrics()
        metrics["log_bytes"] = _dir_size(self.figure_dir)
        if self.level != "off":
            _write_text(self.figure_dir / "tech.json", json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
        return metrics


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception:
        logger.warning("Could not write %s", path, exc_info=True)


# The branch tag is written by the POLICY (`ToolSpec.params_schema`, knob `tag`),
# and in dialogue mode directly by the assistant as a string `params=tag=...`. So
# arbitrary text ends up in the file name, and one slash in it breaks the step
# directory: `_write_text` does `mkdir(parents=True)` before writing and creates a
# directory named by a phrase, and then saving the image fails because it lacks
# that directory (`FileNotFoundError`).
#
# File names are the journal's concern, not the policy's: restricting `tag` by the
# schema is not enough, because the schema guards what the policy NAMED, while the
# tag also arrives here from the loop itself (`_run_actions`). So the cleanup is
# here, on the last stretch before the file system.
#
# The tag is cleaned ONLY for the file name. In the seed (`derive_seed`), in
# `candidates.jsonl` and in events it goes as is: there it breaks nothing, and
# substitution would make the journal incomparable with the draw.
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
# Name tail: `input_<tag>.jpeg` with a 205-character `tag` also hits the 255-byte
# `NAME_MAX`, which is a second way to get the same error.
MAX_NAME_PART = 48


def _safe_part(part: str) -> str:
    """A piece of a file name from arbitrary text: no separators and not longer."""
    cleaned = _UNSAFE_NAME_RE.sub("_", str(part or "")).strip("._-")
    return cleaned[:MAX_NAME_PART]


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
