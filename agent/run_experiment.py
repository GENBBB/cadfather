#!/usr/bin/env python3
"""Run entry point: a config in, a run directory out.

A thin entry point. The only thing done before importing the heavy modules is
fixing the DSL dialect: `utils`/`cadgen` take `CODE_PREFIX` from here, and the
dialect cannot be changed after they are imported.

It is run as a script (`python agent/run_experiment.py`), so Python puts the
`agent/` directory on `sys.path` itself and no `sys.path` edits are needed here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from cad_agent import dsl_runtime
from cad_agent.harness.seeding import DEFAULT_RUN_SEED
# Only the split names. At top level this module pulls in just the standard
# library (trimesh is imported lazily, inside the dataset walk), so importing it
# here does not break the order "the dialect is fixed before importing modules
# that compute CODE_PREFIX".
from cad_agent.harness.subsample import SPLITS
# The `launch` section is read by `run_system.sh`, not by the run. But the run needs
# one answer from it, whether the assistant server is started, and takes it from the
# same function as the shell does. The module is light (yaml and the standard
# library) and does not break the "dialect before heavy imports" order.
#
# The package name matters here: when run from a foreign root that has its own
# `tools` package on `sys.path[0]`, the former `tools.launch_plan` would be shadowed.
from cad_agent.launch_plan import (
    server_cache_setup, server_enabled, server_image_limit, server_parsers,
)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the run's YAML config.")
    parser.add_argument("--output-dir", type=str, required=True, help="Run directory.")
    parser.add_argument(
        "--scaffold",
        type=str,
        default=None,
        help=(
            "Override the scaffold with a policy name (`greedy`, `agent`, ...). "
            "The default comes from the config."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=list(SPLITS),
        help="Override the manifest split: evo | control | test. Taken from the config by default.",
    )
    return parser.parse_args()


# `experiment` sections intentionally not carried over. `details` and `subsample`
# are carried under their own names; nobody reads `final_output`.
NOT_CARRIED = {"final_output"}


def _check_sections(experiment: dict[str, Any], run_config: dict[str, Any]) -> None:
    """No section set in the config may get lost on the way.

    A guard, not decoration: the section list here and the schema in
    `harness/config.py` are two copies of the same knowledge, and they drift apart
    silently. That is how a key could pass the strict config check yet never reach
    the run: the key was legal, the value meaningful, and the runtime took the
    default. The same trap awaits every new key. It cannot be noticed from a run;
    one only sees that the setting changed nothing.
    """
    from cad_agent.harness.config import EXPERIMENT_KEYS

    lost = sorted(
        key for key in EXPERIMENT_KEYS
        if key in experiment and key not in run_config and key not in NOT_CARRIED
    )
    if lost:
        raise ValueError(
            f"Config sections are set but not carried into the run config: {lost}. "
            "Add them to build_run_config, otherwise they will silently have no effect."
        )


def _server_section(config: dict[str, Any]) -> dict[str, Any]:
    """Endpoint addresses plus two facts from `launch`: whether the servers are started.

    The assistant address is always present in the config, but the server behind it
    is started only where it is needed (`launch.servers.assistant.enabled`). Without
    that fact a run would create a client for a port with nobody listening, and each
    worker would greet it with a `/v1/models` poll and a traceback.

    The flag is derived from the same section by the same function that
    `run_system.sh` reads (`tools/launch_plan.py`). It must not be a separate config
    key: two places with one meaning drift apart silently, and the run would judge
    the servers by its own idea of them.
    """
    server = dict(config["server"])
    server["assistant_enabled"] = server_enabled("assistant", config)
    server["generation_enabled"] = server_enabled("generation", config)
    # How many images the assistant endpoint takes per request. Same way and same
    # reason: the setting lives in `launch`, `run_system.sh` reads it, and a second
    # key with the same number would drift from it silently.
    server["assistant_image_limit"] = server_image_limit("assistant", config)
    # Which parsers the assistant was started with: function calls and reasoning.
    # The config check needs them: a function-calling policy on a server without a
    # parser would get a 400 on every question (`config._check_server`).
    parsers = server_parsers("assistant", config)
    server["assistant_tool_call_parser"] = parsers["tool_call_parser"]
    server["assistant_reasoning_parser"] = parsers["reasoning_parser"]
    # Assistant replicas and prefix cache are for warm-up (`experiment.agent.prewarm`):
    # pinning a part to a replica and checking that the cache has somewhere to store.
    cache = server_cache_setup("assistant", config)
    server["assistant_data_parallel_size"] = cache["data_parallel_size"]
    server["assistant_prefix_caching"] = cache["prefix_caching"]
    return server


def build_run_config(
    config: dict[str, Any],
    scaffold_override: str | None,
    split_override: str | None = None,
) -> dict[str, Any]:
    """Reduce the YAML to a flat run config.

    Config sections match layer boundaries and are passed on as they are, without
    unrolling nesting into long `generation_*` keys.
    """
    # Imported here, not at the top: `harness.config` is light, but the constant
    # cannot be taken from `harness.pool`, which pulls in `figure_run` and the whole
    # capability tree, while the DSL dialect is fixed only in `main()`, below.
    from cad_agent.harness.config import (
        DEFAULT_FIGURE_WALL_SEC,
        check_removed_experiment_keys,
        resolve_config_paths,
    )

    experiment = dict(config["experiment"])
    # Before assembly, not after: the config is assembled below by listing known
    # keys, and a removed knob would otherwise vanish without a trace.
    check_removed_experiment_keys(experiment)
    run_config: dict[str, Any] = {
        "details": experiment.get("details"),
        "subsample": dict(experiment.get("subsample", {}) or {}),
        "compute_metrics": experiment.get("compute_metrics", True),
        "extended_metrics": experiment.get("extended_metrics", False),
        "n_workers": experiment.get("n_workers", 8),
        "limit": experiment.get("limit"),
        # Per-part wall-clock ceiling. Like the seed, it is set here rather than
        # read in place: the value lands in the run's `config.json`, which shows
        # what limit the run used.
        "figure_wall_sec": experiment.get("figure_wall_sec", DEFAULT_FIGURE_WALL_SEC),
        # The seed is set here rather than read in place with a default: it lands in
        # the run directory's `config.json`, which later shows what the run was
        # seeded with. A default computed inside a worker would not be recorded.
        "seed": int(experiment.get("seed", DEFAULT_RUN_SEED)),
        "server": _server_section(config),
        # Paths to the weights. The run does not read them (`run_system.sh` does), but
        # `preflight` compares them with `root` of the live endpoint, and it has
        # nothing else to compare with: it receives the run config, not the source one.
        "model": config.get("model", {}) or {},
        "generation": experiment.get("generation", {}),
        "execution": experiment.get("execution", {}),
        "cache": experiment.get("cache", {}),
        "logging": experiment.get("logging", {}),
        "budget": experiment.get("budget", {}),
        # Search-loop ceilings: iterations and candidate depth. Calls and
        # executions stay in `budget`: it is the same ceiling, and giving it a
        # second name would produce two numbers for one thing.
        "limits": dict(experiment.get("limits", {}) or {}),
        "scaffold": dict(experiment.get("scaffold", {}) or {}),
        # The sections below are read at runtime (`figure_run`, `resources`), and
        # each of them used to get lost here: the config passed validation,
        # `metrics: {cd: true}` looked effective, but it never reached the run and
        # silently took the default. Keeping the list by hand is the same drift
        # mechanism, hence the `_check_sections` guard below.
        # The search-loop tool set. `None` means "the default set": it is resolved
        # once in `config.run_tools`, so the run config keeps what the author wrote,
        # not the result of substitution.
        "tools": experiment.get("tools"),
        "metrics": dict(experiment.get("metrics", {}) or {}),
        "agent": dict(experiment.get("agent", {}) or {}),
        "dsl": config.get("dsl", dsl_runtime.DEFAULT_DIALECT),
    }
    _check_sections(experiment, run_config)
    if scaffold_override:
        # The scaffold section is replaced whole, not edited by key: it is the pair
        # `kind: policy` + name, and a key left over from the old value would mean
        # running a different policy than the one requested.
        from cad_agent.scaffold.policies import POLICY_NAMES

        if scaffold_override not in POLICY_NAMES:
            raise SystemExit(
                f"--scaffold {scaffold_override!r}: no such policy. "
                f"Available: {list(POLICY_NAMES)}"
            )
        run_config["scaffold"] = {"kind": "policy", "policy": scaffold_override}
    # Runs being compared must use identical conditions, differing only in the
    # scaffold, and the split is a condition, not the subject of comparison. It is
    # overridden by a flag rather than a second config: a long copy that must match
    # in everything but one word drifts apart silently. A copy is edited, so the
    # run directory's `config.json` records the effective value, not what the file says.
    if split_override:
        run_config["subsample"]["split"] = split_override
    # Paths are made absolute here, once. A relative path in the config used to mean
    # "from the repository root" via cwd, i.e. it depended on where the run was
    # launched from; launched from another root, the subsample manifest is not found.
    # Resolving at read time instead would mean fixing every reader separately.
    # This comes after `split_override`, so `config.json` records the effective value.
    resolve_config_paths(run_config)
    return run_config


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs" / f"{output_dir.name}.log")

    # The dialect is fixed before importing modules that compute CODE_PREFIX, and
    # verified at once: cadgen must resolve to the dialect directory.
    dsl_runtime.configure(config.get("dsl", dsl_runtime.DEFAULT_DIALECT))
    dsl_runtime.verify_dialect()

    # Native thread limits are set here and only here: the variables are read when
    # numpy/BLAS are imported, and forked processes inherit them for free. Heavy
    # imports begin a line below, and after them this would no longer work.
    dsl_runtime.apply_thread_limits(
        int((config.get("experiment", {}).get("execution", {}) or {}).get("native_threads", 1))
    )
    # Here for the same reason: pyvista/vtk read theirs when a window is created, and
    # a window is created in every part worker. This moves rendering to llvmpipe
    # (no CUDA context) and keeps llvmpipe from spawning a pool sized by core count.
    dsl_runtime.apply_graphics_limits()

    # Parent warm-up is here, after the limiters and before the part pool starts.
    # OCP, the dialect preamble and the metrics stack (numpy, scipy, trimesh,
    # pykdtree) are loaded ONCE per run, and the N workers inherit them through fork.
    # Warming each worker separately on its first part would mean N cold imports
    # instead of one. The order matters twice over: the thread limiters are read at
    # import time and warm-up imports numpy, so it must come after them.
    from cad_agent.capabilities.execute import preload_cad

    preload_cad()

    from cad_agent.harness.run_eval import run_experiment

    run_config = build_run_config(config, args.scaffold, args.split)
    result = run_experiment(config=run_config, run_dir=output_dir, source_config=args.config)

    print(f"Run finished. Directory: {result['run_dir']}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
