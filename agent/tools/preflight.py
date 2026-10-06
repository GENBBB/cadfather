#!/usr/bin/env python3
"""Readiness check of a node for a run, before the models are started.

Why. `run_system.sh` first starts two vLLM servers (minutes and all four cards) and
only then launches `run_experiment.py`, where the config is validated, `vendor/` is put
on `sys.path` and execution forks processes. Any trouble from this list (an incomplete
snapshot upload, a missing dependency, an unreachable dataset directory) surfaces only
after the models are up. Here the same things are checked in seconds and without a GPU.

Run in the server environment, from the repository root:

    python agent/tools/preflight.py --config agent/configs/dialogue_lean.yaml

Run locally, this checks the WORKING MACHINE, and only against its own config: the run
configs name the dataset, weights and vLLM environment by cluster paths, and a red
preflight on them means "you are not on the node", not "the code is broken".

Exit code 0 means `run_system.sh` can be started. Non-zero means the run will not start,
and the output says exactly what to fix. Checks marked as optional (the det branch, the
optimizer, endpoints) spoil the output but not the exit code: the baseline branch does
not need them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_ROOT = REPO_ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

FAILED: list[str] = []
WARNED: list[str] = []


def _configured_objective(run_config: dict) -> str:
    """The selection objective the run will use.

    For earlier harnesses it is a key in the config; for a policy it lives inside the
    policy itself. We ask the one that will actually run: otherwise preflight would check
    that CD is available while the run computed IoU and failed on the part.
    """
    scaffold = run_config.get("scaffold") or {}
    if scaffold.get("kind") == "policy":
        from cad_agent.scaffold.policies import build as build_policy

        objective = getattr(build_policy(scaffold.get("policy")), "objective", None)
        return getattr(objective, "name", None) or "cd"
    return scaffold.get("objective", "cd")


def ok(what: str, detail: str = "") -> None:
    print(f"  OK   {what}{(' — ' + detail) if detail else ''}")


def bad(what: str, detail: str) -> None:
    FAILED.append(what)
    print(f"  BAD {what} - {detail}")


def warn(what: str, detail: str) -> None:
    WARNED.append(what)
    print(f"  ~    {what} — {detail}")


# --- 1. config ----------------------------------------------------------------
def check_config(config_path: Path, split_override: str | None = None) -> dict[str, Any] | None:
    print(f"1. Config: {config_path}")
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - present in the server environment
        bad("PyYAML", f"{type(exc).__name__}: {exc}")
        return None

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        bad("config is readable", f"{type(exc).__name__}: {exc}")
        return None

    from run_experiment import build_run_config
    from cad_agent.harness.config import ConfigError, validate_run_config

    run_config = build_run_config(raw, None, split_override)
    try:
        validate_run_config(run_config)
    except ConfigError as exc:
        bad("config passes validation", str(exc))
        return None

    scaffold = run_config.get("scaffold") or {}
    execution = run_config.get("execution") or {}
    ok(
        "config passes validation",
        f"scaffold {scaffold.get('policy') or scaffold.get('kind', 'baseline')}, backend "
        f"{execution.get('backend', 'proxy_pool')}, n_workers {run_config.get('n_workers')}"
        # The seed is visible before the run: a series of comparison runs is matched by
        # it, while a repeat for estimating noise requires a different one; both are
        # decided before launch, not when analyzing results.
        + f", seed {run_config.get('seed')}"
        + (f", limit {run_config['limit']}" if run_config.get("limit") else ""),
    )
    if run_config.get("limit"):
        warn("set truncated by limit", "this is a smoke mode, not a measurement")
    return run_config


# --- 2. vendor/ snapshots --------------------------------------------------------
def check_vendor() -> None:
    print("2. Snapshots of vendor/")
    manifests = sorted(REPO_ROOT.glob("vendor/*/MANIFEST.sha256"))
    if not manifests:
        bad("vendor/ in place", f"no MANIFEST.sha256 in {REPO_ROOT / 'vendor'}")
        return

    for manifest in manifests:
        root = manifest.parent
        missing: list[str] = []
        changed: list[str] = []
        total = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, rel = line.partition("  ")
            total += 1
            target = root / rel.strip()
            if not target.is_file():
                missing.append(rel.strip())
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != digest.strip():
                changed.append(rel.strip())

        name = root.relative_to(REPO_ROOT)
        if missing:
            bad(f"snapshot {name}", f"missing {len(missing)} of {total} files, first: {missing[0]}")
        elif changed:
            bad(f"snapshot {name}", f"{len(changed)} files do not match the signature, first: {changed[0]}")
        else:
            ok(f"snapshot {name}", f"files: {total}, signatures match")


# --- 3. DSL dialect -----------------------------------------------------------
def check_dialect(run_config: dict[str, Any]) -> None:
    print("3. DSL dialect")
    from cad_agent import dsl_runtime

    dialect = run_config.get("dsl", dsl_runtime.DEFAULT_DIALECT)
    try:
        dsl_runtime.configure(dialect)
        dsl_runtime.verify_dialect()
    except Exception as exc:
        bad(f"dialect {dialect}", f"{type(exc).__name__}: {exc}")
        return

    import importlib.util

    spec = importlib.util.find_spec("cadgen")
    origin = getattr(spec, "origin", None) or (spec.submodule_search_locations[0] if spec else "?")
    ok(f"dialect {dialect}", f"cadgen resolves to {origin}")


# --- 4. runtime dependencies --------------------------------------------------
def check_deps() -> None:
    print("4. Runtime dependencies")

    required = {
        "cadquery": "DSL execution",
        "OCP": "OCC kernel",
        "trimesh": "meshes and metrics",
        "numpy": "everywhere",
        "openai": "vLLM client",
        "pyvista": "view rendering",
    }
    for module, why in required.items():
        try:
            __import__(module)
            ok(module, why)
        except Exception as exc:
            bad(module, f"{why}: {type(exc).__name__}: {exc}")

    # Trimesh boolean engine: without it IoU is not computed, and on a slow engine it
    # costs so much that decisions cannot be made on it.
    try:
        import trimesh

        engines = [name for name in ("manifold", "blender") if _engine_available(trimesh, name)]
        if "manifold" in engines:
            ok("trimesh boolean engine", "manifold (fast IoU path)")
        elif engines:
            warn("trimesh boolean engine", f"only {engines} available - IoU will be slow")
        else:
            bad("trimesh boolean engine", "neither manifold nor blender - IoU cannot be computed")
    except Exception as exc:
        bad("trimesh boolean engine", f"{type(exc).__name__}: {exc}")

    try:
        import pykdtree  # noqa: F401

        ok("pykdtree", "GMS is computed")
    except Exception as exc:
        bad("pykdtree", f"without it GMS is not computed: {type(exc).__name__}: {exc}")

    # Optional: the det branch and the optimizer. The baseline works without them.
    from cad_agent import dsl_runtime

    try:
        dsl_runtime.import_det_deps()
        ok("det branch dependencies", "importable")
    except Exception as exc:
        warn("det branch dependencies", f"{type(exc).__name__}: {exc}")

    try:
        import rtree  # noqa: F401

        ok("rtree", "point selection from the second step")
    except Exception as exc:
        warn("rtree", f"without it point selection is limited: {type(exc).__name__}: {exc}")

    # tqdm is decoration, not a capability: without it the run goes exactly the same,
    # only without the per-part progress bar. It must still be reported here, otherwise
    # "no progress bar" reads as a breakage rather than a missing package.
    try:
        import tqdm  # noqa: F401

        ok("tqdm", f"progress bar over parts, version {tqdm.__version__}")
    except Exception as exc:
        warn("tqdm", f"without it the run has no progress bar: {type(exc).__name__}: {exc}")

    # Parameter optimizer: check exactly the bring-up path the runtime uses, and also
    # that the built `_cad_grad` is actually picked up; without it the snapshot cannot
    # be imported.
    try:
        from cad_agent import dsl_runtime as _dsl

        try:
            optimizer = _dsl.import_optimizer()
        except _dsl.OptimizerUnavailable as exc:
            if "_cad_grad" not in str(exc):
                raise
            optimizer = None
        backend = getattr(optimizer, "_cad_grad", None)
        if backend is None:
            warn("parameter optimizer", _cad_grad_hint(_dsl))
        else:
            ok("parameter optimizer", f"_cad_grad picked up: {getattr(backend, '__file__', '?')}")
    except Exception as exc:
        warn("parameter optimizer", f"{type(exc).__name__}: {exc}")

    _check_desugar()


def _check_desugar() -> None:
    """Translation wrapped -> method chain: checked by working, not by importing.

    The point of the check. The snapshot's parser reads only a CadQuery method chain, and
    without translation an optimizer call under our dialect was a guaranteed failure. So
    what must be checked is not "the module imports" but the whole chain: a small wrapped
    script is translated and the result is **parsed by the very parser** that will later
    tune the numbers. Local checks are not enough for this: translation executes cadgen
    operations, so it needs cadquery and lives only in the run environment.

    `_cad_grad` is not needed for this: parsing the code and fitting the numbers are
    different steps, and knowing that the translation works is useful even when the C++
    backend is not built.
    """
    from cad_agent import dsl_runtime

    dialect = dsl_runtime.active_dialect()
    if dialect != "wrapped":
        ok("wrapped->chain translation", f"not needed: dialect {dialect}")
        return

    # Two operations, not one: `extrude` and `hole` take different branches in the
    # translator (the second also adds a 10-unit probe to the chain).
    sketch = "sketch().rect(20,10).finalize()"
    code = (
        f"{dsl_runtime.code_prefix()}\n"
        f"r=extrude(r,(0,0,0),'XY',\"{sketch}\",5)\n"
        f"r=hole(r,(0,0,5),'XY',\"sketch().circle(3).finalize()\",-3)"
    )

    try:
        from cad_agent.capabilities import desugar as desugar_mod

        translated = desugar_mod.to_chain(code)
    except Exception as exc:
        warn("wrapped->chain translation", f"{type(exc).__name__}: {exc}")
        return

    try:
        import importlib

        parser = importlib.import_module(f"{dsl_runtime.OPTIMIZER_PACKAGE}.cq_parser")
        parsed = parser.parse_cadquery(translated["code"])
    except Exception as exc:
        bad(
            "wrapped->chain translation",
            f"translated in {translated['wall_sec']:.2f} s, but the snapshot parser cannot read it "
            f"({type(exc).__name__}: {exc})",
        )
        return

    ok(
        "wrapped->chain translation",
        f"translated in {translated['wall_sec']:.2f} s, the snapshot parser found "
        f"{len(parsed.params)} tunable numbers",
    )


def _cad_grad_hint(dsl_runtime: Any) -> str:
    """Why the C++ backend is missing and what to do about it, in one line.

    The module binary is tied to the Python version, and the snapshot ships no prebuilt
    binaries. So "not picked up" is not enough: show which name is searched for and
    which builds are present.
    """
    import sysconfig

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    found: list[str] = []
    directory = dsl_runtime.NATIVE_BUILD_ROOT / "cad_grad"
    if directory.is_dir():
        found.extend(item.name for item in sorted(directory.glob("_cad_grad*.so")))

    return (
        f"no _cad_grad. Looking for _cad_grad{suffix}, "
        f"present: {found or 'nothing'}. Build for this interpreter: "
        f"./agent/tools/build_cad_grad.sh {sys.executable}"
    )


def _engine_available(trimesh: Any, name: str) -> bool:
    """Whether a boolean engine is available. `engines_available` is a set of names, not a function."""
    available = getattr(trimesh.boolean, "engines_available", None)
    if available is None:
        return False
    if callable(available):
        try:
            return bool(available(name))
        except Exception:
            return False
    return name in available


# --- 5. part set ---------------------------------------------------------
def check_dataset(run_config: dict[str, Any]) -> list[Any]:
    print("5. Part set")
    from cad_agent.harness.dataset import load_figures, load_figures_from_manifest

    subsample = run_config.get("subsample") or {}
    try:
        if subsample.get("manifest"):
            figures = load_figures_from_manifest(
                subsample["manifest"],
                split=subsample.get("split", "evo"),
                verify=subsample.get("verify", True),
            )
            source = f"manifest {subsample['manifest']} ({subsample.get('split', 'evo')})"
        else:
            figures = load_figures(run_config["details"])
            source = ", ".join(
                f"{group}: {folder}" for detail in run_config["details"] for group, folder in detail.items()
            )
    except Exception as exc:
        bad("set is readable", f"{type(exc).__name__}: {exc}")
        return []

    limit = run_config.get("limit")
    if limit:
        figures = figures[:limit]
    ok("set is readable", f"parts: {len(figures)}; source: {source}")
    for figure in figures[:3]:
        size_mb = figure.gt_mesh_path.stat().st_size / 1e6
        print(f"       {figure.figure_id}  ({size_mb:.2f} MB)")
    return figures


# --- 6. execution ------------------------------------------------------------
# Candidate code is executed in an empty namespace: everything available to it comes
# with the dialect prefix (`import cadquery as cq`, cadgen operations, `r`). So the trial
# cube must go with the same prefix as a real prediction, otherwise the check fails on
# `cq is not defined` and says nothing about what is actually broken.
SMOKE_STEP = "r = cq.Workplane('XY').box(10, 10, 10)"


def check_execution(run_config: dict[str, Any], figures: list[Any], tmp_dir: Path) -> None:
    print("6. Code execution and metrics")
    from cad_agent.capabilities.execute import EvalTask, build_executor

    execution = run_config.get("execution") or {}
    backend = execution.get("backend", "proxy_pool")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    from cad_agent import dsl_runtime

    # The objective's metrics are requested the same as the harness will request them:
    # if the chosen objective cannot be computed in this environment (no `pykdtree` for
    # GMS, no boolean engine for IoU), this should be learned here, not from a fallback
    # in the log of every part in the middle of a run.
    from cad_agent.capabilities.objective import get_objective

    objective = get_objective(_configured_objective(run_config))

    gt_path = str(figures[0].gt_mesh_path) if figures else None
    task = EvalTask(
        task_id="preflight",
        code=f"{dsl_runtime.code_prefix()}\n{SMOKE_STEP}",
        mesh_path=str(tmp_dir / "preflight.stl"),
        gt_mesh_path=gt_path,
        measure=bool(gt_path) and bool(run_config.get("compute_metrics", True)),
        extended=bool(run_config.get("extended_metrics", False)),
        needs=objective.needs,
    )

    started = time.time()
    try:
        with build_executor(
            backend=backend,
            pool_size=int(execution.get("pool_size", 4)),
            timeout=float(execution.get("timeout_sec", 30)),
        ) as executor:
            if gt_path:
                executor.warm_gt(gt_path)
            result = executor.evaluate([task])[0]
    except Exception as exc:
        bad(f"backend {backend}", f"{type(exc).__name__}: {exc}")
        return

    wall = time.time() - started
    if not result.success:
        bad(f"backend {backend}", f"cube failed to build: {result.error}")
        return

    ok(f"backend {backend}", f"cube built and exported in {wall:.2f} s")
    if result.metrics:
        shown = {
            key: round(value, 6)
            for key, value in result.metrics.items()
            if isinstance(value, (int, float))
        }
        ok("metrics computed", f"on the cube/{figures[0].figure_id} pair: {shown}")
        _check_objective_available(objective, result.metrics, figures[0].figure_id)
    elif task.measure:
        warn("metrics computed", "a measurement was requested, but the response has no metrics")


def _check_objective_available(objective: Any, metrics: dict[str, Any], figure_id: str) -> None:
    """Whether the chosen objective is computed on a real pair in this environment."""
    if objective.name == "cd":
        ok("objective", "cd: always computed")
        return
    if metrics.get("gms_error"):
        bad(f"objective {objective.name}", f"GMS is not computed: {metrics['gms_error']}")
        return
    value = objective.value(metrics)
    if value is not None:
        ok(f"objective {objective.name}", f"on the cube/{figure_id} pair: {value:.6g}")
    elif "iou" in objective.needs and not metrics.get("gt_watertight", True):
        # Not a breakage but a property of the part: on such parts the objective falls
        # back to GMS. Worth knowing before the run, not worth stopping for.
        warn(f"objective {objective.name}", f"GT of part {figure_id} is not watertight: will fall back to "
             f"{objective.fallback}")
    else:
        bad(f"objective {objective.name}", f"gave no value on the probe pair: {metrics}")


# --- 7. endpoints (by flag) --------------------------------------------------
def check_servers(run_config: dict[str, Any]) -> None:
    print("7. vLLM endpoints")
    import json
    import urllib.request

    server = run_config.get("server") or {}
    endpoints = [
        ("generation", server.get("generation_base_url"), server.get("generation_served_model_name")),
        ("assistant", server.get("assistant_base_url"), server.get("assistant_served_model_name")),
    ]
    for name, base_url, served in endpoints:
        if not base_url:
            continue
        # A disabled server is a config decision, not an environment problem. The probe
        # would honestly fail to reach the port and write "server not up yet?", a warning
        # that is false for every baseline: it does not need the assistant and does not
        # start it.
        if name == "assistant" and server.get("assistant_enabled") is False:
            ok("endpoint assistant", "disabled in launch.servers: not started, the run will not call it")
            continue
        url = base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.load(response)
            entries = payload.get("data", [])
            names = [item.get("id") for item in entries]
            if served and served not in names:
                warn(f"endpoint {name}", f"up, but serves {names} while the config expects {served!r}")
            else:
                ok(f"endpoint {name}", f"{url} serves {names}")

            entry = next((item for item in entries if item.get("id") == served), None) or (entries or [None])[0]
            if entry is None:
                continue
            # `root` and `max_model_len` are printed together for a reason. `root` answers
            # whether these are the weights from the config, the only probe that can patch
            # `wait_for_servers`. `max_model_len` is the cap the run asks the server for so
            # as not to overflow it with the agent's prompt; seeing it before the run is
            # cheaper than seeing 400 errors in the log.
            root = entry.get("root")
            context = entry.get("max_model_len")
            ok(f"endpoint {name}: context",
               f"max_model_len={context if context is not None else 'not reported'}")
            if name == "assistant" and context is None:
                warn(f"endpoint {name}: context",
                     "the server does not report max_model_len: the agent prompt length check "
                     "will not work; set experiment.agent.context_limit")
            expected_path = (run_config.get("model") or {}).get(f"{name}_model_path")
            if root and expected_path:
                if str(root) == str(expected_path):
                    ok(f"endpoint {name}: weights", f"root matches the config ({root})")
                else:
                    warn(f"endpoint {name}: weights",
                         f"root={root!r}, but the config has model.{name}_model_path={expected_path!r}")
        except Exception as exc:
            warn(f"endpoint {name}", f"{url} does not answer ({type(exc).__name__}): server not up yet?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Run config that is about to be launched.")
    # The same flag as in `run_experiment.py` and `run_system.sh`. Without it the
    # readiness check would look at the sample from the config while the run went over
    # another one: for `test` these are different manifests, i.e. different sets.
    from cad_agent.harness.subsample import SPLITS

    parser.add_argument("--split", default=None, choices=list(SPLITS),
                        help="Check this manifest split instead of the one in the config.")
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Where to write the probe .stl. By default next to the config, in work_dirs/preflight.",
    )
    parser.add_argument("--skip-exec", action="store_true", help="Do not run the probe execution.")
    parser.add_argument("--ping-servers", action="store_true", help="Check whether the vLLM endpoints answer.")
    return parser.parse_args()


# Printed at the end of a successful check. Without it "Ready" reads as "the run
# passed": that is exactly how a passed preflight once got recorded as a successful
# smoke on live models.
WHAT_WAS_NOT_CHECKED = """
This is a readiness check, NOT a run. Nothing that needs the models was
checked: step generation, rollout, candidate selection, the agent branch, the
run journal. There is no run directory either: only a probe .stl is written here.

The run itself and its log:  ./run_system.sh agent/configs/<config>.yaml
The log will be in:          <launch.runs_root>/<launch.run_name>/ (on a repeat, with a suffix)
"""


# --- 8. launch section ------------------------------------------------------
def check_launch(config_path: Path) -> None:
    """Whether the config parses into a launch plan and whether what it intends to launch exists.

    This section is read by `run_system.sh`, not by the run, so the config validator
    does not see it at all. Without the check an error in it surfaces only when the models
    are being started.
    """
    print("8. Launch (the launch section is read by run_system.sh)")
    import subprocess

    plan = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("launch_plan.py")), "--config", str(config_path)],
        capture_output=True, text=True,
    )
    if plan.returncode != 0:
        bad("launch plan", (plan.stderr or plan.stdout).strip().splitlines()[-1])
        return
    ok("config parses into a launch plan")

    values: dict[str, str] = {}
    arrays: dict[str, str] = {}
    for line in plan.stdout.splitlines():
        name, _, value = line.partition("=")
        if value.startswith("("):
            arrays[name] = value.strip("()")
        else:
            values[name] = value.strip("'")

    for role in ("GENERATION", "ASSISTANT"):
        if not values.get(f"{role}_ENABLED"):
            warn(f"server {role.lower()}", "not started (absent from the config or enabled: false)")
            continue
        env_path = values.get(f"{role}_ENV") or ""
        where = f"{env_path}/bin/vllm" if env_path else "vllm from PATH"
        if env_path and not Path(env_path, "bin", "vllm").exists():
            bad(f"environment {role.lower()}", f"no {where}")
        else:
            ok(f"server {role.lower()}", f"{values.get(f'{role}_URL')}, devices {values.get(f'{role}_DEVICES') or '<all>'}, {where}")
        # The server's own variables are shown separately: only it sees them, and a mixed-up
        # role means silently the wrong setting on the wrong server.
        own_vars = arrays.get(f"{role}_ENV_VARS", "").strip()
        if own_vars:
            ok(f"variables {role.lower()}", own_vars)

    # Where the run will land. Shown here rather than derived by the reader from the
    # config: the retry suffix is appended by `run_system.sh`, and the printed name is
    # what the directory will start with, not necessarily what it will end with.
    ok("run directory", f"{values.get('RUNS_ROOT')}/{values.get('RUN_NAME')}"
                          " (on a repeat - with the suffix _2, _3, ...)")

    run_env = values.get("RUN_ENV") or ""
    if run_env and not Path(run_env, "bin", "python").exists():
        bad("run environment", f"no {run_env}/bin/python")
    else:
        ok("run environment", run_env or "<active>")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"No such config: {config_path}")
        return 2

    print(f"Readiness check before a run. Repository root: {REPO_ROOT}\n")

    run_config = check_config(config_path, args.split)
    print()
    check_vendor()
    print()
    if run_config is None:
        print("The config did not pass - the remaining checks are meaningless without it.")
        return 1

    check_dialect(run_config)
    print()
    check_deps()
    print()
    figures = check_dataset(run_config)
    print()

    if not args.skip_exec and not FAILED:
        tmp_dir = Path(args.tmp_dir) if args.tmp_dir else REPO_ROOT / "work_dirs" / "preflight"
        check_execution(run_config, figures, tmp_dir)
        print()
    elif not args.skip_exec:
        print("6. Code execution and metrics\n  ~    skipped: not reached, there are errors above\n")

    if args.ping_servers:
        check_servers(run_config)
        print()

    check_launch(config_path)
    print()

    if FAILED:
        print(f"NOT READY: {len(FAILED)} checks failed: {', '.join(FAILED)}")
        return 1
    if WARNED:
        print(f"Ready with caveats ({len(WARNED)}): {', '.join(WARNED)}")
    else:
        print("Ready to launch.")
    print(WHAT_WAS_NOT_CHECKED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
