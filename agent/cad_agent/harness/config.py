"""Strict validation of the run config.

A typo in a key name must cost a second at startup, not hours of a run: a config
read through `.get(..., default)` runs on defaults and comes back as something other
than intended. On long runs that is an expensive silence.

What is checked:

- unknown keys are an error, not "never mind": most often it is a typo or a field
  left over from an earlier config schema;
- types and allowed enum values (execution backend, logging level, DSL dialect,
  scaffold kind);
- mutual consistency: `details` and `subsample` cannot both be set, drawings cannot
  be requested without STEP export, the worker count cannot be non-positive;
- the scaffold section names the policy and nothing else: knobs belong to the
  policy, and an extra key next to it is a second source of truth, not a trifle.

The check deliberately lives in the harness: it must run before models and
processes come up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """The config will not get past startup."""


EXECUTION_BACKENDS = ("serial_fork", "ephemeral_pool", "proxy_pool")
LOGGING_LEVELS = ("off", "metrics", "full")
# There is one scaffold kind: the harness holds the search loop, and the policy
# arrives by name, with its knobs living in it rather than in the config.
SCAFFOLD_KINDS = ("policy",)
DEFAULT_SCAFFOLD_KIND = "policy"

# Split names by role. The list is the same as in the manifest: two copies would
# drift apart, and the config would accept a name the manifest does not have.
from cad_agent.harness.subsample import SPLIT_EVO, SPLITS  # noqa: E402


# Repository root is derived from `__file__`, not from cwd: `agent/cad_agent/harness/`.
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_config_path(path: str) -> str:
    """Make a config path absolute: a relative one is relative to the repository root.

    A relative path in the config always meant "from the repository root" (runs
    start there) but meant it SILENTLY, via cwd. While only our own launcher read the
    config this made no difference; a run started from another directory would look
    for `manifests/subsamples/...` in the wrong tree. Here the implied rule is
    written explicitly and does not depend on cwd at all.
    """
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else REPO_ROOT / candidate)


def resolve_config_paths(run_config: dict[str, Any]) -> None:
    """Resolve the run config's paths in place, before any disk read.

    The keys are listed by name, deliberately: `model.*` is excluded because it holds
    either an absolute path to weights or an HF Hub identifier (`Qwen/Qwen3.8-27B`),
    and treating the latter as a path would turn a model name into a nonexistent
    directory inside the repository.

    Paths in the `launch` section are not ours either: the shell reads them through
    `cad_agent.launch_plan`, and they never reach the run config.
    """
    subsample = run_config.get("subsample")
    if isinstance(subsample, dict) and subsample.get("manifest"):
        subsample["manifest"] = resolve_config_path(str(subsample["manifest"]))

    details = run_config.get("details")
    if details:
        # The shape is a list of single-key mappings "group: directory"; this is how
        # both `dataset.load_figures` and `cad_agent.launch_plan` read it.
        run_config["details"] = [
            {group: resolve_config_path(str(where)) for group, where in item.items()}
            if isinstance(item, dict) else item
            for item in details
        ]

# Knobs that disappeared together with the mechanism that read them. They must not
# be silently ignored: a config with `request_workers: 8` looks meaningful but means
# its author counts on parallel requests that no longer exist.
_SAMPLING_MOVED = (
    "generation mode moved into the policy: temperature is a parameter "
    "of the `stepwise` action (`temperature`, 0 = greedy decoding), the number "
    "of variants is its `n`. Defaults (n=1, temperature=1.0, top_p=0.95) live in "
    "harness/search_types.py. The old key silently overrode `n` and the temperature of "
    "ANY policy, so runs could go through without sampling"
)

REMOVED_GENERATION_KEYS = {
    "greedy": _SAMPLING_MOVED,
    "temperature": _SAMPLING_MOVED,
    "top_p": _SAMPLING_MOVED,
    "top_k": _SAMPLING_MOVED,
    "request_workers": (
        "k variants are taken in one request with parameter n, the thread pool is gone; "
        "sampling width is set by experiment.scaffold.k_variants"
    ),
}

# Top-level knobs that disappeared together with their mechanism. The old name must
# fail the config: `figure_stall_sec: 900` looks like a working guard but would mean
# a run with no per-part cap at all.
REMOVED_EXPERIMENT_KEYS = {
    "capabilities": (
        "the tool set is given by the list experiment.tools "
        "(names come from the registry in harness/tools.py). The old "
        "capabilities.optimize: true now means adding 'optimize' "
        "to this list, and false means not listing it"
    ),
    "figure_stall_sec": (
        "the pool guard was removed (it did not work: orphaned proxies held the executor "
        "sentinels). The cap is now per part and soft: "
        "experiment.figure_wall_sec"
    ),
}

# The same, per section. A knob that lost its mechanism must fail the config, not
# quietly do nothing.
REMOVED_SECTION_KEYS: dict[str, dict[str, str]] = {
    "cache": {
        "max_size_gb": (
            "prediction images are cached in the part process memory, not on disk; "
            "the cache size is a number of images: experiment.cache.render_images"
        ),
    },
}

# Which candidate meshes remain in the part directory after the rollout.
SAVE_MESHES_MODES = ("none", "best", "all")


def resolve_save_meshes(logging_config: dict[str, Any] | None) -> str:
    """Mesh-saving policy: explicit, or derived from the log level.

    By default (`null`) it follows the level: the full log promises an `.stl` for every
    step, other levels only the returned mesh. They are separate because the cost
    differs: there are an order of magnitude more candidates.
    """
    config = logging_config or {}
    mode = config.get("save_meshes")
    if mode is None:
        return "all" if str(config.get("level", "metrics")) == "full" else "best"
    return str(mode)

# Wall-time cap of ONE part (`Budget.check_wall`, checked in the part's process
# before spawning candidates). Fifteen minutes means "longer than the longest
# legitimate part, with margin", not "let's wait a bit more".
#
# The cap is soft: it does not interrupt a started unit of work but prevents the next
# one from starting. So the actual overrun is one unit, which is bounded from above
# by its own caps (`execution.timeout_sec`, `det_timeout_sec`, `det_budget_sec`).
DEFAULT_FIGURE_WALL_SEC = 900.0

# Top-level schema of the `experiment` section: name -> type.
EXPERIMENT_KEYS: dict[str, type | tuple[type, ...]] = {
    "details": (list, type(None)),
    "subsample": dict,
    "compute_metrics": bool,
    "extended_metrics": bool,
    "n_workers": int,
    "limit": (int, type(None)),
    # Wall-time cap of ONE part. Soft: checked inside the part's process before
    # spawning candidates, so it stops the continuation rather than the work. Zero or
    # null means no cap (then a single part can hold the run indefinitely).
    #
    # A harness knob, not a policy one: the run's liveness must not depend on what a
    # policy does.
    "figure_wall_sec": (int, float, type(None)),
    # Run seed. `int` only: "do not seed" is not a supported mode; repeated runs to
    # estimate the noise floor use a different seed.
    "seed": int,
    "scaffold": dict,
    "execution": dict,
    "generation": dict,
    "cache": dict,
    "logging": dict,
    "agent": dict,
    "budget": dict,
    # Caps of the search loop. Separate from `budget`, which counts CALLS, while loop
    # turns and candidate depth are quantities of a different nature; putting them in
    # one dict would make "budget exceeded" mean three different events.
    "limits": dict,
    "final_output": dict,
    # Tool set of the search loop: names from the `harness/tools.py` registry.
    # A list, not a dict of flags: it shows the set of measurement channels at a
    # glance, and a new registry tool needs no config schema edit.
    #
    # This is a HARNESS knob, not a policy one, although it looks like a behaviour
    # setting. The set is determined by the run environment (whether `_cad_grad` is
    # built, whether the assistant is up, whether call prices were taken), and the
    # policy learns it from `state.legal` and `state.price`, not from its own beliefs.
    "tools": (list, type(None)),
    "metrics": dict,
}

SECTION_KEYS: dict[str, dict[str, type | tuple[type, ...]]] = {
    # Observable metrics to compute beyond what the selection objective needs.
    # Empty means do not compute: the objective pulls its own via `Objective.needs`.
    "metrics": {"cd": bool},
    # `null` in either means "no cap". Zero means exactly zero, i.e. the part makes no
    # iteration at all: the two cases must be kept apart, otherwise a disabled cap and
    # a forbidden search would read the same.
    "limits": {"iterations": (int, type(None)), "depth": (int, type(None))},
    "subsample": {"manifest": str, "split": str, "verify": bool},
    "execution": {
        "backend": str, "pool_size": int, "timeout_sec": (int, float),
        "det_timeout_sec": (int, float), "native_threads": int,
        # det cap per PART, in seconds. Separate from `det_timeout_sec`: that one limits a
        # single call, there are many calls per part, and without this cap their sum is
        # unbounded. `null` means no limit.
        "det_budget_sec": (int, float, type(None)),
        # Cap of ONE optimizer call, in seconds. A separate key rather than shared with
        # det: they have different time distributions and different pathologies. With no
        # key the harness default applies; an explicit `null` removes the run's cap and
        # leaves the call to the timeout of `capabilities/optimize.py`.
        "opt_timeout_sec": (int, float, type(None)),
        # Face cap of the residual mesh in det. It sets both the call cost and the length
        # of the DSL string det appends to the prefix, and the next call's cost grows
        # with prefix length (see capabilities/det.py).
        "det_residual_faces": int,
        # Root of scratch directories (candidate meshes). Empty means tmpfs, see harness/scratch.py.
        "scratch_dir": (str, type(None)),
    },
    # There are NO sampling knobs here: generation mode, temperature and nucleus belong
    # to the policy and arrive as action parameters (`ToolSpec.params_schema` of
    # `stepwise`), while defaults live in the seam (`harness/search_types.py`). What
    # remains is what is not the policy's choice: answer length, retries,
    # postprocessing, dedup.
    "generation": {
        "max_tokens": int, "postprocess_code": bool, "max_attempts": int, "dedupe": bool,
    },
    "cache": {"render_images": int},
    "logging": {
        "level": str, "profile": bool, "report": bool,
        # Per-part progress bar. `null` follows the terminal: the run is in the foreground
        # and inherits it, while output redirected to a file must not get a bar. An
        # explicit value beats the terminal. Decoration, not an observable: it affects
        # neither results nor measurement, and a missing `tqdm` is not an error
        # (`harness/progress.py`).
        "progress": (bool, type(None)),
        "load": bool, "load_interval_sec": (int, float),
        "save_meshes": (str, type(None)),
        # Candidate table (`candidates.jsonl`): a row per candidate with its full
        # measurement. Separate from the level because the cost differs: tens of bytes
        # per candidate against hundreds of kilobytes for its image and mesh.
        "candidates": bool,
    },
    # Calls to the decision agent. Everything except the cap and the reasoning mode is
    # prompt-size estimation; the cap itself is taken from the server and set here only
    # when the server does not report it.
    "agent": {
        "context_limit": (int, type(None)),
        "image_tokens": int,
        "chars_per_token": (int, float),
        # Whether to ask the assistant for reasoning mode (`chat_template_kwargs.
        # enable_thinking`). A run condition, not a policy knob: the same policy on a
        # thinking and a non-thinking model gives different costs and answers, and such
        # runs cannot be compared.
        "thinking": (bool, type(None)),
        # Cap of the assistant's answer on a decision turn (`max_tokens` of the policy's
        # question). `None` means the policy constant (`DialogueLeanPolicy.
        # CHOOSE_MAX_TOKENS`). A run condition like `thinking`: for a thinking model the
        # cap is shared with the reasoning.
        "answer_max_tokens": (int, type(None)),
        # Whether to prewarm the assistant's prefix cache before the policy's question: a
        # request with the unchanged beginning of the question and a one-token answer,
        # with the part pinned to a DP replica (`resources.ask_agent`). A run condition:
        # it changes server speed and load, not the model's prompt.
        "prewarm": bool,
    },
    "server": {
        "generation_base_url": str, "assistant_base_url": (str, type(None)),
        "generation_served_model_name": str, "assistant_served_model_name": (str, type(None)),
        # Whether the assistant server is brought up. This key is not in the YAML: it is
        # derived from `launch.servers.assistant.enabled` in
        # `run_experiment.build_run_config` and lands in the run directory's
        # `config.json`, showing which set of endpoints the run had. There is no reason
        # to write it by hand.
        "assistant_enabled": (bool, type(None)),
        # How many images the assistant endpoint accepts per request
        # (`--limit-mm-per-prompt`). Also not in the YAML: derived by
        # `launch_plan.server_image_limit` from the `launch` section. The policy needs
        # it: it sends panels as a list, and a request over the cap is rejected WHOLE,
        # not truncated by the server to what fits.
        "assistant_image_limit": (int, type(None)),
        # Assistant answer parsers (`--tool-call-parser` with `--enable-auto-tool-choice`,
        # `--reasoning-parser`). Not in the YAML either: derived by
        # `launch_plan.server_parsers` from the `launch` section. Needed by the check of
        # function-calling policies (`_check_server`).
        "assistant_tool_call_parser": (str, type(None)),
        "assistant_reasoning_parser": (str, type(None)),
        # The assistant's DP replicas and its prefix cache. Not in the YAML: derived by
        # `launch_plan.server_cache_setup` from the `launch` section. Needed by the prewarm.
        "assistant_data_parallel_size": (int, type(None)),
        "assistant_prefix_caching": (bool, type(None)),
        # The same for the generator. The key appeared after the assistant's one and for
        # the opposite reason: the assistant was guarded, while the harness knew nothing
        # about the generator, so a run created a client for a port with nobody on it and
        # answered with a failure on EVERY part instead of an error at startup.
        "generation_enabled": (bool, type(None)),
    },
}


def run_tools(config: dict[str, Any]) -> tuple[str, ...]:
    """Which tools this run provides. The single answer to this question.

    Both the search loop (what to offer the policy as legal) and the capability layer
    (whether to create the optimizer) ask here. Having two places answer the same
    question from one key lets such pairs drift apart silently.

    An empty or absent `experiment.tools` means the registry's default set, NOT an
    empty set: a config that said nothing about tools asks for an ordinary run, not a
    ban on search.
    """
    from cad_agent.harness import tools as tools_mod

    chosen = (config or {}).get("tools")
    if not chosen:
        return tools_mod.default_names()
    # Order and repeats do not belong to the config: the registry defines both,
    # otherwise two configs with the same set would differ by notation.
    picked = {str(name) for name in chosen}
    return tuple(name for name in tools_mod.names() if name in picked)


def check_removed_experiment_keys(experiment: dict[str, Any]) -> None:
    """A knob that lost its mechanism must fail the config.

    Checked on the RAW `experiment` section, not on the run config:
    `build_run_config` assembles the config by listing known keys, so a removed knob
    never reaches the validator and silently does nothing, exactly the outcome the
    table exists to prevent.
    """
    for old, hint in REMOVED_EXPERIMENT_KEYS.items():
        if old in experiment:
            raise ConfigError(f"experiment.{old} is no longer read: {hint}")


def validate_run_config(config: dict[str, Any]) -> None:
    """Validate the flat run config. Raises `ConfigError` on the first problem."""
    _check_unknown("experiment", config, set(EXPERIMENT_KEYS) | {"server", "dsl", "model"})

    for key, expected in EXPERIMENT_KEYS.items():
        _check_type(f"experiment.{key}", config.get(key), expected, optional=True)

    check_removed_experiment_keys(config)

    for old, hint in REMOVED_GENERATION_KEYS.items():
        if old in (config.get("generation") or {}):
            raise ConfigError(f"experiment.generation.{old} is no longer read: {hint}")

    for section, removed in REMOVED_SECTION_KEYS.items():
        for old, hint in removed.items():
            if old in (config.get(section) or {}):
                raise ConfigError(f"experiment.{section}.{old} is no longer read: {hint}")

    for section, keys in SECTION_KEYS.items():
        section_value = config.get(section)
        if section_value is None:
            continue
        _check_type(f"experiment.{section}", section_value, dict)
        _check_unknown(f"experiment.{section}", section_value, set(keys))
        for key, expected in keys.items():
            _check_type(f"experiment.{section}.{key}", section_value.get(key), expected, optional=True)

    _check_dataset(config)
    _check_execution(config.get("execution") or {})
    _check_cache(config.get("cache") or {})
    _check_logging(config.get("logging") or {})
    _check_agent(config.get("agent") or {}, config)
    _check_budget(config.get("budget") or {})
    _check_limit(config)
    _check_workers(config)
    _check_scaffold(config.get("scaffold") or {})
    _check_tools(config)
    _check_sampling(config)
    _check_server(config.get("server") or {}, config.get("scaffold") or {}, config)


def _check_tools(config: dict[str, Any]) -> None:
    """A name not in the registry is a typo, and it must cost a second at startup.

    What is checked is the typo, not the sensibleness of the set: `tools: [stepwise]`
    is a legitimate measurement of a single channel, and there is no ground to forbid
    it. But `tools: [det]` when the tools are `det_cold`/`det_warm` looks working and
    means a run with no deterministic branch at all.

    An empty list differs from an absent key and is forbidden separately: absence means
    "the default set", while emptiness would read as "search forbidden": the part would
    take no action and return an empty prefix, which the report would show as a model
    failure.
    """
    from cad_agent.harness import tools as tools_mod

    chosen = config.get("tools")
    if chosen is None:
        return
    if not chosen:
        raise ConfigError(
            "experiment.tools is empty: the search loop could not take a single "
            "action. Remove the key to use the default set "
            f"({', '.join(tools_mod.default_names())})"
        )
    known = set(tools_mod.names())
    unknown = [str(name) for name in chosen if str(name) not in known]
    if unknown:
        raise ConfigError(
            f"experiment.tools: tools not in the registry: {sorted(unknown)}. "
            f"Available: {', '.join(sorted(known))}"
        )


def _check_agent(agent: dict[str, Any], config: dict[str, Any]) -> None:
    """Validate the constraints on calls to the decision agent.

    The keys here set not behaviour but **guards**: if the prompt size estimate turns
    out wrong, the run survives it (the server returns 400 and the harness asks
    shorter), but nonsensical values are not worth surviving: a zero cap silently
    disables the check, a negative one disables common sense along with it.
    """
    limit = agent.get("context_limit")
    if limit is not None and int(limit) <= 0:
        raise ConfigError(
            "experiment.agent.context_limit must be positive; "
            "to take the cap from the server, remove the key or set null"
        )
    image_tokens = agent.get("image_tokens")
    if image_tokens is not None and int(image_tokens) < 0:
        raise ConfigError("experiment.agent.image_tokens must not be negative")
    chars = agent.get("chars_per_token")
    if chars is not None and float(chars) <= 0:
        raise ConfigError("experiment.agent.chars_per_token must be greater than zero")
    answer = agent.get("answer_max_tokens")
    if answer is not None and int(answer) <= 0:
        # The policy would silently replace zero with its own constant (`or`), and the
        # key would look working.
        raise ConfigError(
            "experiment.agent.answer_max_tokens must be positive; "
            "to use the policy constant, remove the key or set null"
        )

    if not agent:
        return
    # No assistant means nothing to constrain, yet the keys in the config look
    # working. Same reasoning as in `_check_tools`.
    server = config.get("server") or {}
    if not server.get("assistant_base_url"):
        raise ConfigError(
            "experiment.agent is set, but there is no assistant endpoint "
            "(server.assistant_base_url is empty): nothing to constrain. "
            "Remove the section or connect the assistant"
        )
    if server.get("assistant_enabled") is False:
        raise ConfigError(
            "experiment.agent is set, but the assistant server is disabled "
            "(launch.servers.assistant.enabled: false): nothing to constrain, "
            "the run will not create a client for it. Remove the section or enable the server"
        )
    # A prewarm without a prefix cache is an extra request per turn and nothing more.
    # The key is checked for explicit presence: a config not built from YAML lacks it.
    if agent.get("prewarm") and "assistant_prefix_caching" in server \
            and server["assistant_prefix_caching"] is not True:
        raise ConfigError(
            "experiment.agent.prewarm: true, but the assistant server starts without a prefix "
            "cache: the prewarm has nowhere to store. Set launch.servers.assistant.args "
            "`enable-prefix-caching: true`"
        )


def _check_dataset(config: dict[str, Any]) -> None:
    details = config.get("details")
    subsample = config.get("subsample") or {}

    if not details and not subsample.get("manifest"):
        raise ConfigError(
            "No dataset: set either experiment.details (a directory of .stl) "
            "or experiment.subsample.manifest (a fixed subsample)."
        )
    if details and subsample.get("manifest"):
        raise ConfigError(
            "Both experiment.details and experiment.subsample.manifest are set. "
            "The manifest takes priority, so details would silently do nothing; remove one."
        )
    split = subsample.get("split", SPLIT_EVO)
    if subsample.get("manifest") and split not in SPLITS:
        raise ConfigError(f"experiment.subsample.split={split!r}; expected one of {list(SPLITS)}")


def _check_execution(execution: dict[str, Any]) -> None:
    backend = execution.get("backend", "proxy_pool")
    if backend not in EXECUTION_BACKENDS:
        raise ConfigError(f"experiment.execution.backend={backend!r}; expected one of {list(EXECUTION_BACKENDS)}")

    pool_size = execution.get("pool_size", 4)
    if int(pool_size) <= 0:
        raise ConfigError("experiment.execution.pool_size must be positive")
    if backend == "serial_fork" and int(pool_size) > 1:
        logger.warning(
            "experiment.execution.pool_size=%s is not used with backend=serial_fork", pool_size
        )
    for key in ("timeout_sec", "det_timeout_sec", "det_budget_sec", "opt_timeout_sec"):
        value = execution.get(key)
        if value is not None and float(value) <= 0:
            raise ConfigError(f"experiment.execution.{key} must be positive")

    residual_faces = execution.get("det_residual_faces")
    if residual_faces is not None and int(residual_faces) <= 0:
        raise ConfigError("experiment.execution.det_residual_faces must be positive")

    # A part cap below the call cap is not an error but a working setting: the part
    # then gets exactly one truncated call. The opposite combination deserves a
    # warning: a call cap above the part cap means the call is limited by nothing but
    # the part.
    budget_sec = execution.get("det_budget_sec")
    call_sec = execution.get("det_timeout_sec")
    if budget_sec is not None and call_sec is not None and float(call_sec) > float(budget_sec):
        logger.warning(
            "experiment.execution.det_timeout_sec=%s exceeds det_budget_sec=%s - "
            "the call timeout will never fire, the part is limited by the budget only",
            call_sec, budget_sec,
        )

    native_threads = execution.get("native_threads")
    if native_threads is not None and int(native_threads) <= 0:
        raise ConfigError("experiment.execution.native_threads must be positive")


def _check_cache(cache_config: dict[str, Any]) -> None:
    render_images = cache_config.get("render_images")
    if render_images is not None and int(render_images) < 0:
        raise ConfigError("experiment.cache.render_images must not be negative")


def _check_logging(logging_config: dict[str, Any]) -> None:
    level = logging_config.get("level", "metrics")
    if level not in LOGGING_LEVELS:
        raise ConfigError(f"experiment.logging.level={level!r}; expected one of {list(LOGGING_LEVELS)}")

    interval = logging_config.get("load_interval_sec")
    if interval is not None and float(interval) <= 0:
        raise ConfigError("experiment.logging.load_interval_sec must be positive")

    save_meshes = logging_config.get("save_meshes")
    if save_meshes is not None and save_meshes not in SAVE_MESHES_MODES:
        raise ConfigError(
            f"experiment.logging.save_meshes={save_meshes!r}; "
            f"expected one of {list(SAVE_MESHES_MODES)} or null (follows the log level)"
        )

    # `level: off` writes nothing at all, including the candidate table. This pair
    # must not be swallowed silently: the config looks like "candidates are written"
    # while the run directory is empty, and the explanation is sought in harness code.
    if logging_config.get("candidates") and level == "off":
        raise ConfigError(
            "experiment.logging.candidates: true with level: off: the candidate "
            "table will not be written. Raise the level to metrics or "
            "remove the knob"
        )


def _check_budget(budget: dict[str, Any]) -> None:
    from cad_agent.harness.budget import CALL_KINDS

    allowed = set(CALL_KINDS) | {"total"}
    _check_unknown("experiment.budget", budget, allowed)
    for key, value in budget.items():
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"experiment.budget.{key} must be a positive integer or null, got {value!r}")


def _check_limit(config: dict[str, Any]) -> None:
    limit = config.get("limit")
    if limit is None:
        return
    if isinstance(limit, bool) or int(limit) <= 0:
        raise ConfigError("experiment.limit must be a positive integer or absent")
    logger.warning(
        "experiment.limit=%s: the set will be truncated. This is a smoke mode, not a measurement.", limit
    )


def _check_workers(config: dict[str, Any]) -> None:
    n_workers = int(config.get("n_workers", 8))
    if n_workers <= 0:
        raise ConfigError("experiment.n_workers must be positive")

    from . import load as load_mod

    # Cores by cgroup quota, not by machine: under a CFS quota work proceeds at the
    # quota's speed, which can be several times fewer than the machine's cores. By
    # the machine count the threshold never fired, and heavy oversubscription looked
    # like headroom.
    cores = load_mod.n_cpu()
    pool_size = int((config.get("execution") or {}).get("pool_size", 4))
    backend = (config.get("execution") or {}).get("backend", "proxy_pool")
    processes = n_workers * (1 + (pool_size if backend != "serial_fork" else 1))
    if cores and processes > 2 * cores:
        machine = os.cpu_count() or 0
        quota = load_mod.cpu_quota()
        # Mention the quota only when it is the limiter: otherwise the line would name a
        # number that affects nothing.
        limited = quota is not None and machine and quota < machine
        logger.warning(
            "Up to %d processes on %g available cores (n_workers=%d, pool_size=%d)%s: "
            "oversubscription will eat the parallelism gain",
            processes, cores, n_workers, pool_size,
            f" (this is the cgroup quota, {machine} on the machine)" if limited else "",
        )


def _check_scaffold(scaffold: dict[str, Any]) -> None:
    kind = scaffold.get("kind", DEFAULT_SCAFFOLD_KIND)
    if kind not in SCAFFOLD_KINDS:
        raise ConfigError(f"experiment.scaffold.kind={kind!r}; expected one of {list(SCAFFOLD_KINDS)}")
    _check_policy(scaffold)


def _check_policy(scaffold: dict[str, Any]) -> None:
    """For a policy the config names it and nothing else.

    Policy parameters belong to the policy. An `objective` or `k_variants` key next to
    `kind: policy` is not a harmless extra line but a second source of truth: the
    config says one thing, the policy does another. So this is an error, not a
    warning.
    """
    from cad_agent.scaffold.policies import POLICY_NAMES

    name = scaffold.get("policy")
    if not name:
        raise ConfigError(
            "experiment.scaffold.kind=policy requires experiment.scaffold.policy - "
            f"a policy name from {list(POLICY_NAMES)}"
        )
    if name not in POLICY_NAMES:
        raise ConfigError(
            f"experiment.scaffold.policy={name!r} not found. Available: {list(POLICY_NAMES)}"
        )
    extra = sorted(set(scaffold) - {"kind", "policy"})
    if extra:
        raise ConfigError(
            f"experiment.scaffold: extra keys {extra} with kind=policy. Policy knobs "
            "live in the policy itself (measurement variants are separate policy names), "
            "and the config carries only harness settings: caps, log level, dataset"
        )


def _check_sampling(config: dict[str, Any]) -> None:
    """Asking for many variants at zero temperature means buying copies.

    The temperature is named by the policy itself (`temperature` in the `stepwise`
    action parameters), so the question is put to it: a policy with `temperature = 0`
    and `k_variants > 1` orders k identical samples, decodes them all and ends up with
    one candidate after dedup.

    Only NAMED registry policies are checked, the ones we run ourselves. A foreign or
    substituted policy object is not covered: its wastefulness is a matter of the cost
    axis, not of run startup.
    """
    scaffold = config.get("scaffold") or {}
    if not scaffold.get("policy"):
        return
    from cad_agent.scaffold.policies import build as _build_policy

    policy = _build_policy(scaffold.get("policy"))
    temperature = getattr(policy, "temperature", None)
    width = getattr(policy, "k_variants", 1)
    if temperature is None or float(temperature) != 0.0:
        return
    if isinstance(width, int) and width > 1:
        raise ConfigError(
            f"policy {scaffold.get('policy')!r} asks for k_variants={width} at "
            "temperature=0: there is no sampling, the k replies will be copies, and after "
            "dedup one candidate remains. Either k_variants=1 or a nonzero "
            "policy temperature"
        )


def _check_server(
    server: dict[str, Any], scaffold: dict[str, Any], config: dict[str, Any] | None = None
) -> None:
    if not server.get("generation_base_url") or not server.get("generation_served_model_name"):
        raise ConfigError("server.generation_base_url and server.generation_served_model_name are required")

    has_url = bool(server.get("assistant_base_url"))
    has_name = bool(server.get("assistant_served_model_name"))
    if has_url != has_name:
        raise ConfigError(
            "assistant_base_url and assistant_served_model_name must be set together: "
            "otherwise the decision agent looks connected but does not work"
        )

    # The assistant address may be set even when no agent is used, so the address
    # alone does not tell whether an agent will be asked. What does is whether the
    # server comes up, and that is a run condition: a policy that calls the agent with
    # the assistant disabled would get a failure on EVERY part. An error here costs a
    # second at startup, silence costs the whole run.
    #
    # The key is checked for an explicit False: configs not built from YAML (tests,
    # earlier runs) lack it entirely, which means "as before".
    if server.get("assistant_enabled") is False and _wants_assistant(scaffold):
        raise ConfigError(
            f"experiment.scaffold (policy={scaffold.get('policy')!r}) calls the decision agent, "
            "but the assistant server is disabled (launch.servers.assistant.enabled: false): "
            "the agent would get a failure on every part. Enable the assistant server "
            "or run a branch without the agent"
        )

    # A function-calling policy (`policies/dialogue_lean.py`) and a server without
    # a call parser: vLLM rejects `tool_choice="auto"` whole (400), and every turn
    # would fall back with the outcome "the assistant did not answer". The second
    # condition is reasoning: without `--reasoning-parser` the server looks for calls
    # in the reasoning too, i.e. would execute a draft in the middle of a thought
    # (exactly what the policy forbids parsing). The keys are checked for
    # explicit presence: a config not built from YAML lacks them, which means "as
    # before".
    needs = _tool_calls_needed(scaffold)
    if needs and "assistant_tool_call_parser" in server:
        if needs == "auto" and not server.get("assistant_tool_call_parser"):
            raise ConfigError(
                f"experiment.scaffold (policy={scaffold.get('policy')!r}) calls the assistant with "
                "function calling (tool_choice=auto), but the assistant server starts without "
                "a call parser: set launch.servers.assistant.args "
                "`enable-auto-tool-choice: true` and `tool-call-parser` (qwen3_coder for Qwen3.8)"
            )
        thinking = ((config or {}).get("agent") or {}).get("thinking")
        if thinking is not False and not server.get("assistant_reasoning_parser"):
            raise ConfigError(
                f"experiment.scaffold (policy={scaffold.get('policy')!r}) calls the assistant with "
                "function calling with possible reasoning (experiment.agent.thinking is not false), "
                "and the server has no `reasoning-parser`: calls would also be parsed from the reasoning. "
                "Either thinking: false or launch.servers.assistant.args.reasoning-parser"
            )

    # The same reasoning, for the GENERATOR. A disabled assistant used to be caught
    # while a disabled generator was not, and a run with `stepwise` in the set went to
    # the end answering with a failure on every part.
    #
    # What is asked is not the tool name but its counter: the generator is needed
    # exactly by those that spend `vlm`. A list of names would be a second copy of the
    # registry and would drift from it silently; a new tool would not enable the check.
    if server.get("generation_enabled") is False:
        from cad_agent.harness import tools as tools_mod

        needs_generator = tuple(
            name for name in run_tools(config or {})
            if "vlm" in tools_mod.get(name).spends
        )
        if needs_generator:
            raise ConfigError(
                f"The run tool set calls the generator ({', '.join(needs_generator)}), but its "
                "server is disabled (launch.servers.generation.enabled: false, or "
                "model.generation_model_path is not set): every part would get a failure. "
                "Enable the generator server or remove these tools from "
                "experiment.tools"
            )


def _wants_assistant(scaffold: dict[str, Any]) -> bool:
    """Whether this policy needs a running assistant; asked of the policy itself.

    Knobs moved into the policy while the trap stayed: silently skipping the policy
    would mean the move of the knobs disabled the check, the same class as with the
    sampling guard (`_check_sampling`).
    """
    from cad_agent.scaffold.policies import build as _build_policy

    name = scaffold.get("policy")
    if not name:
        return False
    return bool(getattr(_build_policy(name), "needs_assistant", False))


def _tool_calls_needed(scaffold: dict[str, Any]) -> str | None:
    """Whether the policy calls the assistant with functions, and with which `tool_choice`.

    Asked of the policy itself (`TOOL_CHOICE`), by the reasoning of `_wants_assistant`.
    `None` means the policy passes no functions.
    """
    from cad_agent.scaffold.policies import build as _build_policy

    name = scaffold.get("policy")
    if not name:
        return None
    choice = getattr(_build_policy(name), "TOOL_CHOICE", None)
    return None if choice is None else str(choice)


def _check_unknown(where: str, mapping: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}. Expected: {sorted(allowed)}")


def _check_type(
    where: str,
    value: Any,
    expected: type | tuple[type, ...],
    optional: bool = False,
) -> None:
    if value is None and optional:
        return
    expected_tuple = expected if isinstance(expected, tuple) else (expected,)
    # bool is a subclass of int, and "n_workers: true" must not pass as a number.
    if bool in expected_tuple or not isinstance(value, bool):
        if isinstance(value, expected_tuple):
            return
    names = ", ".join(getattr(item, "__name__", str(item)) for item in expected_tuple)
    raise ConfigError(f"{where}: expected {names}, got {type(value).__name__} ({value!r})")
