"""Expand a YAML config into variables for `run_system.sh`.

The config is the single source of truth for model paths, ports and vLLM flags:
the shell receives ready-made values, so the servers and the run cannot end up
on different models.

Output: `KEY=value` lines and `KEY=(...)` arrays suitable for bash `eval`.
Everything goes through `shlex.quote`, so JSON inside vLLM flags
(`--mm-processor-kwargs '{"min_pixels": ...}'`) survives.

    eval "$(python agent/tools/launch_plan.py --config configs/dialogue_lean.yaml)"

The module lives in the package (not next to the CLI) because one of its
answers, `server_enabled`, is needed by the run itself and imported as a module.
A generic top-level name such as `tools` can be shadowed by another package on
`sys.path[0]`; `cad_agent` collides with nothing. The CLI wrapper stays in
place because shell scripts call it by path.

The `launch` config section (entirely optional):

    launch:
      vllm_env: /path/to/env           # default vLLM server environment
      run_env:  /path/to/env           # environment the run itself uses
      runs_root: work_dirs
      run_name: my_run                 # run directory name inside runs_root
      env:                             # environment variables for everything
        HF_HOME: /path/to/cache
      servers:
        generation:
          enabled: true
          vllm_env: /path              # OWN environment, overrides launch.vllm_env
          env:                         # OWN variables, on top of launch.env
            VLLM_ATTENTION_BACKEND: FLASHINFER
          cuda_visible_devices: "0,1"
          args:                        # `vllm serve` flags, as is
            dtype: bfloat16
            trust-remote-code: true    # true -> flag without a value
            max-model-len: 2000

Two environments for two servers is a normal case: the generator and the
assistant may be built for different vLLM versions. Each server therefore has
its own `vllm_env` and `env`: the interpreter comes from its own environment and
the variables are visible only to that server.

Naming rule: `*_env` is always a **path to an environment**, `env` is always
**environment variables**.

Model, host, port and `--served-model-name` come from the `model` and `server`
sections; do not duplicate them in `launch`.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

# Mapping "server in the config" -> "keys that describe it".
SERVERS = {
    "generation": {
        "model": "generation_model_path",
        "url": "generation_base_url",
        "name": "generation_served_model_name",
    },
    "assistant": {
        "model": "assistant_model_path",
        "url": "assistant_base_url",
        "name": "assistant_served_model_name",
    },
}


# Keys understood here. Anything else is a typo: silently taking a default
# instead of the intended environment would start the server on the wrong vLLM.
LAUNCH_KEYS = {"vllm_env", "run_env", "runs_root", "run_name", "server_timeout_sec",
               "env", "servers"}
SERVER_SPEC_KEYS = {"enabled", "vllm_env", "env", "cuda_visible_devices", "args"}

# Default native-library thread pools for vLLM servers, used when the config does
# not set them. Without them torch in the API process and in EngineCore creates a
# pool per core of the MACHINE rather than per pod quota; the idle threads
# together throttle the CPU stages of the run, which share the pod quota. The
# servers' CPU work is one hot thread, so 1 thread is enough. A value set in
# `launch.env` or `launch.servers.<role>.env` takes precedence.
SERVER_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

# Run directory name when the config does not give one. A numbered default
# would not tell which branch or objective a directory belongs to.
DEFAULT_RUN_NAME = "experiment"

# The name becomes a directory name, so not any string is accepted: a path
# separator would lead the run into another directory, a leading dot would hide
# it from a plain `ls`. Any letters are fine, including non-ASCII (`\w` is Unicode).
RUN_NAME_RE = re.compile(r"^\w[\w.-]{0,63}$")


def run_name(launch: dict[str, Any], override: str | None = None) -> str:
    """Run directory name: the launch flag, else `launch.run_name`, else the default.

    A repeat-run suffix is not added here: only the creator of the directory
    (`run_system.sh`) can claim a name, with one atomic `mkdir`. Computing a free
    name here would reopen the window in which two consecutive launches get the
    same directory.
    """
    name = override if override is not None else launch.get("run_name")
    if name is None or str(name).strip() == "":
        name = DEFAULT_RUN_NAME
    name = str(name).strip()
    if not RUN_NAME_RE.match(name):
        fail(f"run name {name!r} is not a valid directory name: expected letters, "
             f"digits, dot, hyphen and underscore (up to 64 characters); a dot or hyphen "
             f"cannot be the first character")
    return name


def _looks_like_path(value: str) -> bool:
    """Whether the value is a filesystem path (as opposed to an HF Hub model id)."""
    return value.startswith(("/", "./", "../", "~"))


def fail(message: str) -> None:
    print(f"launch_plan: {message}", file=sys.stderr)
    raise SystemExit(2)


def emit(name: str, value: Any) -> None:
    print(f"{name}={shlex.quote('' if value is None else str(value))}")


def emit_array(name: str, values: list[str]) -> None:
    print(f"{name}=({' '.join(shlex.quote(item) for item in values)})")


def _env_pairs(where: str, env: Any) -> list[str]:
    """Mapping name -> value into a list of `NAME=value`.

    A string is rejected on purpose: `env` here means variables, while the
    environment path is called `vllm_env`. Mixing them up would silently start a
    server from the wrong environment.
    """
    if env is None:
        return []
    if isinstance(env, str):
        fail(f"{where}={env!r}: these are environment variables (name -> value); "
             f"the environment path is set by the vllm_env key")
    if not isinstance(env, dict):
        fail(f"{where} must be a mapping name -> value")
    return [f"{name}={value}" for name, value in env.items()]


def flag_argv(args: Any) -> list[str]:
    """`vllm serve` flags from the config into argv.

    A mapping (rather than a list of strings) keeps the YAML readable and lets a
    flag be overridden by name. Order is preserved.
    """
    if not args:
        return []
    if isinstance(args, list):
        # A list of strings is accepted too: sometimes a whole flag is easier to write.
        argv: list[str] = []
        for item in args:
            argv.extend(shlex.split(str(item)))
        return argv
    if not isinstance(args, dict):
        fail(f"launch.servers.*.args must be a mapping or a list, got {type(args).__name__}")

    argv = []
    for key, value in args.items():
        flag = f"--{str(key).lstrip('-')}"
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue  # a disabled flag is simply not passed
        else:
            argv.extend([flag, str(value)])
    return argv


def server_enabled(role: str, config: dict[str, Any], launch: dict[str, Any] | None = None) -> bool:
    """Whether the server of this role is started under this config.

    A server starts only if it is fully described. The assistant is not needed by
    the fast branch, and starting a 32B model for it would waste two GPUs.

    The answer is needed by more than the shell: the run creates a client to the
    endpoint and queries its context limit, and `preflight` probes the endpoint;
    neither makes sense for a server that does not start. The decision therefore
    lives in one function; a second config key with the same meaning would
    silently diverge.
    """
    if role not in SERVERS:
        fail(f"unknown server role: {role!r}; expected one of {sorted(SERVERS)}")
    keys = SERVERS[role]
    if launch is None:
        launch = config.get("launch") or {}
    spec = ((launch.get("servers") or {}).get(role)) or {}
    model = (config.get("model") or {}).get(keys["model"])
    url = (config.get("server") or {}).get(keys["url"])
    return bool(spec.get("enabled", True)) and bool(model) and bool(url)


def server_image_limit(
    role: str, config: dict[str, Any], launch: dict[str, Any] | None = None
) -> int | None:
    """How many images the endpoint of this role accepts PER REQUEST.

    This is `--limit-mm-per-prompt` in `launch.servers.<role>.args`, the same flag
    `run_system.sh` starts the server with, read by the same function for the
    same reason as `server_enabled`.

    The dialogue sends panels as a LIST (target plus candidates). The number of
    panels is a policy constant (`MAX_IMAGE_PANELS`) while the cap is a run
    condition. A policy that raises the constant above the cap gets a rejection
    of the WHOLE request, not of the extra image, and every following turn ends
    in a fallback; the diagnosis would point at a dead channel instead of "asked
    for too many images".

    The number comes from the run config, while the server may have been started
    earlier from another config. So the cap is an intent, not a fact about the
    endpoint, and it guards against a policy slip, not against a mismatch with
    the live server. The server cannot be asked either: `/v1/models` does not
    report multimodal limits.

    `None` means no cap is set, so nothing is limited. A cap that is set but not
    parseable is a config error, not "unset": such `args` would go to vLLM as is
    and fail at startup.
    """
    if role not in SERVERS:
        fail(f"unknown server role: {role!r}; expected one of {sorted(SERVERS)}")
    if launch is None:
        launch = config.get("launch") or {}
    spec = ((launch.get("servers") or {}).get(role)) or {}
    args = spec.get("args")
    raw = args.get("limit-mm-per-prompt") if isinstance(args, dict) else None
    if raw is None:
        return None
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(
                f"launch.servers.{role}.args.limit-mm-per-prompt is not valid JSON: {raw!r}"
            ) from None
    if not isinstance(value, dict) or "image" not in value:
        raise ValueError(
            f"launch.servers.{role}.args.limit-mm-per-prompt: expected "
            f'a mapping with the key "image", got {raw!r}'
        )
    try:
        return int(value["image"])
    except (TypeError, ValueError):
        raise ValueError(
            f"launch.servers.{role}.args.limit-mm-per-prompt: "
            f'the "image" value is not an integer: {raw!r}'
        ) from None


def server_parsers(
    role: str, config: dict[str, Any], launch: dict[str, Any] | None = None
) -> dict[str, str | None]:
    """Response parsers the endpoint of this role runs with: function calls and reasoning.

    Reads the same `launch.servers.<role>.args` that `run_system.sh` starts the
    server with, for the same reason as `server_image_limit`. `tool_call_parser`
    is the parser name only if `enable-auto-tool-choice` is also set (without it
    vLLM rejects `tool_choice="auto"` entirely); `reasoning_parser` is a name or
    `None`.

    Like `server_image_limit`, this is the config's intent, not the live server.
    A reused (`--keep-servers`) server is checked only by its weights
    (`reuse_server`), so one started from another config without these flags
    would pass.
    """
    if role not in SERVERS:
        fail(f"unknown server role: {role!r}; expected one of {sorted(SERVERS)}")
    if launch is None:
        launch = config.get("launch") or {}
    spec = ((launch.get("servers") or {}).get(role)) or {}
    args = spec.get("args") if isinstance(spec.get("args"), dict) else {}

    def named(key: str) -> str | None:
        value = args.get(key)
        return str(value) if value not in (None, False, True, "") else None

    auto = args.get("enable-auto-tool-choice") is True
    return {
        "tool_call_parser": named("tool-call-parser") if auto else None,
        "reasoning_parser": named("reasoning-parser"),
    }


def server_cache_setup(
    role: str, config: dict[str, Any], launch: dict[str, Any] | None = None
) -> dict[str, Any]:
    """DP replica count of this role's endpoint and whether prefix caching is on.

    Needed by cache prewarming (`experiment.agent.prewarm`): a part is pinned to a
    replica by `data_parallel_size`, and without `enable-prefix-caching` the
    prewarm stores nothing (for the hybrid model the cache is off without the
    flag). Reads the same `launch.servers.<role>.args`. `prefix_caching`: `True`
    means the flag is on, `False` means off (`no-enable-prefix-caching`), `None`
    means unspecified (the vLLM default decides).
    """
    if role not in SERVERS:
        fail(f"unknown server role: {role!r}; expected one of {sorted(SERVERS)}")
    if launch is None:
        launch = config.get("launch") or {}
    spec = ((launch.get("servers") or {}).get(role)) or {}
    args = spec.get("args") if isinstance(spec.get("args"), dict) else {}
    size = args.get("data-parallel-size")
    caching = None
    if args.get("enable-prefix-caching") is True:
        caching = True
    elif args.get("enable-prefix-caching") is False or args.get("no-enable-prefix-caching") is True:
        caching = False
    return {"data_parallel_size": int(size) if size is not None else None,
            "prefix_caching": caching}


def server_plan(role: str, keys: dict[str, str], config: dict[str, Any], launch: dict[str, Any]) -> None:
    prefix = role.upper()
    server_config = config.get("server") or {}
    spec = ((launch.get("servers") or {}).get(role)) or {}
    unknown = sorted(set(spec) - SERVER_SPEC_KEYS)
    if unknown:
        fail(f"launch.servers.{role}: unknown keys {unknown}; expected {sorted(SERVER_SPEC_KEYS)}")

    model = (config.get("model") or {}).get(keys["model"])
    url = server_config.get(keys["url"])
    served_name = server_config.get(keys["name"]) or role

    # "Start or not" is decided by `server_enabled`: the run asks it too, so it
    # does not create a client to a disabled server.
    enabled = server_enabled(role, config, launch)
    emit(f"{prefix}_ENABLED", "1" if enabled else "")
    if not enabled:
        return

    # The model is either a path on disk or an HF Hub id
    # (`Qwen/Qwen3-VL-32B-Instruct`). Only the former can be checked for
    # existence; for the latter "no such path" would be a false alarm that
    # prevented startup altogether.
    if _looks_like_path(str(model)) and not Path(str(model)).exists():
        fail(f"model.{keys['model']}: no such path: {model}")

    parsed = urlparse(str(url))
    if not parsed.hostname or not parsed.port:
        fail(f"server.{keys['url']}={url!r}: cannot parse host and port")

    argv = ["--served-model-name", str(served_name),
            "--host", parsed.hostname, "--port", str(parsed.port)]
    argv.extend(flag_argv(spec.get("args")))

    emit(f"{prefix}_MODEL", model)
    emit(f"{prefix}_URL", f"{parsed.scheme}://{parsed.hostname}:{parsed.port}{parsed.path}")
    emit(f"{prefix}_DEVICES", spec.get("cuda_visible_devices", ""))
    # The server's own environment overrides the common one: the generator and the
    # assistant may be built for different vLLM versions (a normal case).
    emit(f"{prefix}_ENV", spec.get("vllm_env") or launch.get("vllm_env") or "")
    # Own environment variables, for this server only, on top of launch.env.
    # Different vLLM builds want different things (attention backend, HF cache,
    # the environment's LD_LIBRARY_PATH), which a shared variable cannot express.
    # Thread defaults come first and only for names the config set in neither
    # place: otherwise `env` on top of `launch.env` would override the shared
    # value with the default.
    named = set(launch.get("env") or {}) | set(spec.get("env") or {})
    defaults = [f"{name}={value}" for name, value in SERVER_THREAD_DEFAULTS.items()
                if name not in named]
    emit_array(f"{prefix}_ENV_VARS",
               defaults + _env_pairs(f"launch.servers.{role}.env", spec.get("env")))
    emit_array(f"{prefix}_ARGV", argv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    # Run name override via the `run_system.sh --name` flag. Parsed here, not in
    # the shell, so there is a single name validator.
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        fail(f"no such config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    launch = config.get("launch") or {}
    unknown = sorted(set(launch) - LAUNCH_KEYS)
    if unknown:
        fail(f"launch: unknown keys {unknown}; expected {sorted(LAUNCH_KEYS)}")

    # Launch environment variables: HF_HOME and the like.
    emit_array("LAUNCH_ENV", _env_pairs("launch.env", launch.get("env")))

    emit("RUN_ENV", launch.get("run_env") or "")
    emit("RUNS_ROOT", launch.get("runs_root") or "work_dirs")
    # Run directory name. The repeat-run suffix is added by `run_system.sh`, which
    # creates the directory.
    emit("RUN_NAME", run_name(launch, args.run_name))
    emit("SERVER_TIMEOUT_SECONDS", launch.get("server_timeout_sec") or 300)

    # Target directories: needed not for starting models but for measurements that
    # walk the same set as the run. Empty if the set is given by a subsample
    # manifest rather than a directory.
    experiment = config.get("experiment") or {}
    detail_groups: list[str] = []
    detail_paths: list[str] = []
    for item in experiment.get("details") or []:
        if not isinstance(item, dict):
            fail(f"experiment.details: expected a list of mappings 'group: path', got {item!r}")
        for group, path in item.items():
            detail_groups.append(str(group))
            detail_paths.append(str(path))
    emit_array("DETAIL_GROUPS", detail_groups)
    emit_array("DETAIL_PATHS", detail_paths)

    for role, keys in SERVERS.items():
        server_plan(role, keys, config, launch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
