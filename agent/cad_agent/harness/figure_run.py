"""Run one whole part: the body of a worker process.

Everything one part needs is assembled here: its own renderer, its own DSL
executor, its own step generator, its own budget. Control then passes to the
scaffold in exactly one call, `scaffold.run(obs, res)`, which is the only place
where the harness touches the policy.

OpenAI clients are created **inside** the process: an HTTP connection survives
neither fork nor pickling.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from cad_agent import dsl_runtime
from cad_agent.capabilities import llm
from cad_agent.capabilities import metrics as metrics_mod
from cad_agent.capabilities.execute import build_executor
from cad_agent.capabilities.propose import StepProposer
from cad_agent.capabilities.render import DEFAULT_RENDER_CACHE_IMAGES, FigureRenderer
from cad_agent.capabilities.resources import build_resources
from cad_agent.harness import scratch
from cad_agent.harness.artifacts import RunLayout, save_json
from cad_agent.harness.budget import (
    WALL_STOP_REASON,
    Budget,
    BudgetExceeded,
    DeadlineExceeded,
)
from cad_agent.harness.config import DEFAULT_FIGURE_WALL_SEC, resolve_save_meshes
from cad_agent.harness.dataset import FigureSpec
from cad_agent.harness.journal import FigureJournal
from cad_agent.harness.seeding import DEFAULT_RUN_SEED
from cad_agent.scaffold.base import Observation, Reconstruction

logger = logging.getLogger(__name__)

# Clients live across several parts within one worker process.
_CLIENTS: dict[str, Any] = {}


def _get_clients(server_config: dict[str, Any]) -> dict[str, Any]:
    if _CLIENTS:
        return _CLIENTS

    from openai import OpenAI

    _CLIENTS["generation"] = OpenAI(base_url=server_config["generation_base_url"], api_key="EMPTY")
    _CLIENTS["generation_model"] = server_config["generation_served_model_name"]

    # The generator client is created unconditionally, unlike the assistant one:
    # the `OpenAI` constructor does not touch the network (the tracebacks came from
    # `fetch_context_limit`, which is not called here), and `StepProposer` is built
    # for every part regardless of the toolset. The check "the toolset calls the
    # generator but the server is off" runs at startup, in `config._check_server`:
    # there it costs a second, here it would cost the run.

    assistant_url = server_config.get("assistant_base_url")
    assistant_model = server_config.get("assistant_served_model_name")
    # The client is created only for a RUNNING server. The assistant address is
    # always present in the config (it is shared by the baseline and the agent
    # branch), but the server itself is started only where the wrapper asks for it:
    # `launch.servers.assistant.enabled`, which arrives here as
    # `server.assistant_enabled`. Without this check every worker polled an empty
    # port via `/v1/models` and logged a traceback, one per worker in the first
    # seconds of the run. Default True: a config without a `launch` section
    # (tests, older runs) behaves as before.
    assistant_enabled = server_config.get("assistant_enabled", True)
    if assistant_url and assistant_model and assistant_enabled:
        client = OpenAI(base_url=assistant_url, api_key="EMPTY")
        _CLIENTS["assistant"] = client
        _CLIENTS["assistant_model"] = assistant_model
        # The context limit is asked from the server, not taken from the config:
        # it is set by the `--max-model-len` flag in the `launch` section, which only
        # `run_system.sh` reads. A second key with the same number is exactly the
        # divergence mechanism that would make the run talk to the server according
        # to its own idea of it.
        #
        # One request per worker process: clients outlive the part.
        _CLIENTS["assistant_context"] = llm.fetch_context_limit(client, assistant_model)
    return _CLIENTS


def run_figure(
    spec: FigureSpec,
    config: dict[str, Any],
    scaffold: Any,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Run one part and return its per-figure record."""
    started = time.monotonic()
    layout = RunLayout(run_dir)
    work_dir = layout.figure_dir(spec.figure_id)

    execution_config = config.get("execution", {})
    generation_config = config.get("generation", {})
    server_config = config.get("server", {})
    logging_config = config.get("logging", {})
    agent_config = config.get("agent", {}) or {}

    # Candidate meshes live in the scratch directory (tmpfs), not in the run
    # directory: they are needed during the rollout and are read by the same process
    # that writes them. Only what was asked to be saved reaches the part directory.
    save_meshes = resolve_save_meshes(logging_config)
    scratch_dir = scratch.figure_dir(
        scratch.run_dir(Path(run_dir).name, execution_config.get("scratch_dir")),
        spec.figure_id,
    )

    journal = FigureJournal(
        figure_dir=work_dir,
        level=str(logging_config.get("level", "metrics")),
        profile=bool(logging_config.get("profile", False)),
        candidates=bool(logging_config.get("candidates", False)),
    )
    # Load sampling of the part runs for the whole rollout: the execution pool is
    # already closed by the end, and a single snapshot in `finalize` would show emptiness.
    journal.watch_load(
        interval_sec=float(logging_config.get("load_interval_sec", 5.0)),
        enabled=bool(logging_config.get("load", True)),
    )
    journal.event("figure_start", figure_id=spec.figure_id, gt=str(spec.gt_mesh_path))

    renderer = FigureRenderer(
        max_images=int((config.get("cache") or {}).get("render_images", DEFAULT_RENDER_CACHE_IMAGES)),
    )
    executor = build_executor(
        backend=execution_config.get("backend", "proxy_pool"),
        pool_size=int(execution_config.get("pool_size", 4)),
        timeout=float(execution_config.get("timeout_sec", 30)),
    )
    # The part wall-time cap is a harness knob, not the policy's: run liveness must
    # not depend on the scaffold. Scaffolds have their own `scaffold.max_time_sec`,
    # which remains; this cap works independently of it and of whatever the policy
    # did to itself.
    figure_wall_sec = config.get("figure_wall_sec", DEFAULT_FIGURE_WALL_SEC)
    figure_wall_sec = None if figure_wall_sec is None else float(figure_wall_sec)
    if figure_wall_sec is not None and figure_wall_sec <= 0:
        figure_wall_sec = None
    budget = Budget(
        limits=dict(config.get("budget", {}) or {}),
        wall_sec=figure_wall_sec,
        # Counted from the start of the PART, not from the creation of the counter:
        # GT warm-up is already its time, and a part stuck in it must fall under the cap.
        started=started,
    )

    reconstruction = Reconstruction(figure_id=spec.figure_id)
    # Created before `try`: the `finally` block asks it for cache counters, and a
    # failure can come earlier than the generator is created.
    proposer: StepProposer | None = None
    try:
        with journal.stage("warm_gt"):
            executor.warm_gt(str(spec.gt_mesh_path))

        clients = _get_clients(server_config)
        proposer = StepProposer(
            client=clients["generation"],
            model_name=clients["generation_model"],
            renderer=renderer,
            max_tokens=int(generation_config.get("max_tokens", 1024)),
            # The run no longer has sampling knobs: mode and temperature belong to the
            # policy and arrive as action parameters. The registry defaults stand
            # here: what applies when the policy did not name a knob.
            postprocess_code=bool(generation_config.get("postprocess_code", False)),
            dedupe=bool(generation_config.get("dedupe", True)),
            max_attempts=int(generation_config.get("max_attempts", 3)),
            journal=journal,
        )

        resources = build_resources(
            figure_id=spec.figure_id,
            gt_mesh_path=spec.gt_mesh_path,
            work_dir=work_dir,
            executor=executor,
            renderer=renderer,
            proposer=proposer,
            agent_client=clients.get("assistant"),
            agent_model=clients.get("assistant_model"),
            budget=budget,
            config=config,
            journal=journal,
            # The run seed is a measurement condition, not a harness knob: it goes
            # from the config to the harness and on to the capabilities, bypassing the policy.
            run_seed=int(config.get("seed", DEFAULT_RUN_SEED)),
            mesh_dir=scratch_dir,
            save_meshes=save_meshes,
            # An explicit config value beats the server's answer: the server may not be
            # believed (old vLLM does not report `max_model_len`), but silently
            # substituting our own number for its is not allowed.
            agent_context_limit=(
                agent_config.get("context_limit")
                if agent_config.get("context_limit") is not None
                else clients.get("assistant_context")
            ),
            agent_image_tokens=int(agent_config.get("image_tokens", llm.AGENT_IMAGE_TOKENS)),
            agent_chars_per_token=float(agent_config.get("chars_per_token", llm.CHARS_PER_TOKEN)),
            # No `bool()`: the key is tri-state, and "the run said nothing" (`None`)
            # differs from "the run turned it off" (`False`) in that the first leaves
            # the mode to the server template and the second must reach it as a key.
            # Collapsing these two states was the defect: `thinking: false` sent
            # nothing, and the `Qwen3` template reads a missing key as enabled.
            agent_thinking=agent_config.get("thinking"),
            agent_answer_max_tokens=agent_config.get("answer_max_tokens"),
            # Not from `agent_config`: the image cap is set by a server flag, not by an
            # experiment section, and is derived once in `build_run_config`.
            agent_image_limit=server_config.get("assistant_image_limit"),
            # Prefix cache warm-up and pinning the part to a DP replica: there are as
            # many replicas as `launch` started (`launch_plan.server_cache_setup`).
            agent_prewarm=bool(agent_config.get("prewarm", False)),
            agent_dp_size=server_config.get("assistant_data_parallel_size"),
        )

        observation = Observation(
            figure_id=spec.figure_id,
            gt_mesh_path=Path(spec.gt_mesh_path),
            work_dir=work_dir,
            prefix_code=dsl_runtime.code_prefix(),
        )

        reconstruction = scaffold.run(observation, resources)

    except DeadlineExceeded as exc:
        # The exception reaches here only if the scaffold did not stop by itself:
        # all our scaffolds catch the deadline and return what they found. A policy
        # that does not do this loses its rollout, but the run does not stall.
        # The two outcomes must be told apart: stopping with its best prefix and
        # losing the whole rollout are different events.
        reconstruction.stop_reason = WALL_STOP_REASON + " (the scaffold did not stop by itself)"
        reconstruction.error = str(exc)
        logger.warning("Part %s stopped by the wall-time cap: %s", spec.figure_id, exc)
    except BudgetExceeded as exc:
        # The cap is not a failure but a regular outcome: the part is counted as a refusal.
        reconstruction.stop_reason = "budget cap exhausted"
        reconstruction.error = str(exc)
    except Exception:
        reconstruction.error = traceback.format_exc()
        logger.exception("Rollout of part %s crashed", spec.figure_id)
    finally:
        # Cache counters are collected BEFORE closing: the renderer and the generator
        # keep them themselves, and after `close()` there is nothing to ask. They are
        # collected for a failed part too: a part that failed on step five reports
        # about the cache as much as one that ran to the end.
        journal.record_cache_stats(renderer.cache_stats())
        if proposer is not None:
            journal.record_cache_stats(proposer.cache_stats())
        executor.close()
        renderer.close()

    journal.event(
        "figure_end",
        stop_reason=reconstruction.stop_reason,
        n_steps=reconstruction.n_steps,
        error=reconstruction.error,
    )
    # The best mesh moves from the scratch directory to the part directory BEFORE
    # scoring: `_score_final` reads it from there.
    _promote_best_mesh(reconstruction, work_dir, save_meshes)
    # Scoring runs BEFORE `journal.finalize()`, and the part wall time is counted
    # after it. It used to be the other way round, and the final metrics measurement
    # (loading two meshes plus IoU and GMS) fell into neither a stage nor `wall_sec`:
    # the worker was busy with it, but by the numbers the part did not spend it.
    record = _finalize(spec, reconstruction, budget, work_dir, config, journal)
    record["wall_sec"] = time.monotonic() - started
    tech = journal.finalize()
    record["tech"] = tech
    record["log_bytes"] = tech.get("log_bytes", 0)
    if save_meshes == "none":
        # Scoring has already gone over the scratch copy, and the copy is about to
        # disappear: the record must not reference a nonexistent file.
        record["mesh_path"] = None
    save_json(record, work_dir / "figure.json")
    # The scratch directory is removed last: it is still being read until this line.
    scratch.cleanup(scratch_dir)
    return record


def _promote_best_mesh(reconstruction: Reconstruction, work_dir: Path, save_meshes: str) -> None:
    """Move the returned mesh from the scratch directory to the part directory.

    The scaffold hands over a path in the scratch directory, where it lay during the
    whole rollout. Only this mesh must outlive the run, under a clear name:
    `best.stl` next to `best.py`. With `save_meshes: none` it is not saved either,
    and then the path in the per-figure record is cleared, otherwise the record
    would reference a vanished file.
    """
    if not reconstruction.mesh_path or save_meshes == "none":
        return
    source = Path(reconstruction.mesh_path)
    if not source.exists():
        return
    target = work_dir / "best.stl"
    try:
        shutil.copyfile(source, target)
        reconstruction.mesh_path = str(target)
    except OSError:
        logger.warning("Could not save the best part mesh from %s", source, exc_info=True)


def _finalize(
    spec: FigureSpec,
    reconstruction: Reconstruction,
    budget: Budget,
    work_dir: Path,
    config: dict[str, Any],
    journal: FigureJournal,
) -> dict[str, Any]:
    """Contract metrics and cost after the rollout.

    The runtime measured CD with its own scale at every step; that is enough for
    decisions inside the rollout but not for the report: the contract computes
    `score_i` from IoU and GMS. So the final result is measured once, here, on the
    best prefix.
    """
    figure_metrics = None
    if config.get("compute_metrics", True):
        # A stage of its own, not part of `execute`: this is a second, independent
        # metrics computation, per the contract, on the best prefix and in another
        # process. While it was unnamed, its cost hid in the difference between the
        # part wall time and the sum of stages.
        with journal.stage("score_final"):
            figure_metrics = _score_final(spec, reconstruction, config)

    if reconstruction.code:
        (work_dir / "best.py").write_text(reconstruction.code, encoding="utf-8")
    save_json(reconstruction.journal, work_dir / "journal.json")

    return {
        "figure_id": spec.figure_id,
        "group": spec.group,
        "gt_mesh_path": str(spec.gt_mesh_path),
        "mesh_path": reconstruction.mesh_path,
        "n_steps": reconstruction.n_steps,
        # How many search ITERATIONS the part made is not the same as `n_steps`
        # (depth of the best candidate). Counted here from the journal already in hand:
        # the per-figure table is consumed downstream, and collecting this number by
        # walking `figures/*/journal.json` would mean hundreds of small NFS reads for
        # something already computed.
        "iterations": len(reconstruction.journal or ()),
        "stop_reason": reconstruction.stop_reason,
        # The finish door: the `self_stop_ratio` axis reads it from this table, not
        # from hundreds of `journal.json` files.
        "done_by": reconstruction.done_by,
        "error": reconstruction.error,
        "runtime_metrics": reconstruction.metrics,
        "metrics": figure_metrics.to_dict() if figure_metrics is not None else None,
        "score": figure_metrics.score() if figure_metrics is not None else 0.0,
        "cost": budget.to_dict(),
        # The wall time is appended by `run_figure`: it must include this scoring too,
        # and the scoring is computed here.
        "wall_sec": 0.0,
    }


def _score_final(spec: FigureSpec, reconstruction: Reconstruction, config: dict[str, Any]):
    import trimesh

    if not reconstruction.mesh_path or not Path(reconstruction.mesh_path).exists():
        # GT is loaded here too, although there is no prediction: it determines the
        # stratum, and the stratum is a property of the set, not of the rollout
        # outcome. We pay for this only on failed parts, and in return a failure no
        # longer draws a nonexistent `non_watertight_gt` stratum. An unreadable GT
        # leaves the status unknown: a separate `unknown_gt` stratum, not a silent
        # "not watertight".
        gt_mesh = None
        try:
            gt_mesh = trimesh.load_mesh(spec.gt_mesh_path)
        except Exception:
            logger.exception("Failed to load GT %s to determine the stratum", spec.gt_mesh_path)
        return metrics_mod.evaluate_pair(
            figure_id=spec.figure_id,
            gt_mesh=gt_mesh,
            pred_mesh=None,
            execution_error=reconstruction.error or "no valid prediction",
        )

    gt_mesh = trimesh.load_mesh(spec.gt_mesh_path)
    pred_mesh = trimesh.load_mesh(reconstruction.mesh_path)
    metrics_mod.normalize_for_metrics(gt_mesh, pred_mesh)
    return metrics_mod.evaluate_pair(
        figure_id=spec.figure_id,
        gt_mesh=gt_mesh,
        pred_mesh=pred_mesh,
        compute_extended=bool(config.get("extended_metrics", False)),
        compute_cd=bool((config.get("metrics") or {}).get("cd", False)),
    )
