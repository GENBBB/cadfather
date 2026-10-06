"""vLLM engine counters for a run: a `/metrics` snapshot before the rollouts and after.

Why: the engine log with the prefix-cache hit rate is silent when DP > 1:
`vllm serve` sets `api_server_count` equal to `data_parallel_size`, and each API
process disables the text stats log (`v1/metrics/loggers.py`, warning
"api_server_count more than 1"). Prometheus counters are still aggregated across
all processes, so the run's hit rate is the difference of two `/metrics` snapshots,
not a parse of the log.

Caveat: the counters are **server-wide**. If another run used the server during the
same window (a pair of runs on shared servers), the difference includes its requests.

A snapshot is cheap and never fails a run: no server or no `/metrics` gives an empty
dict, and the role simply does not appear in the summary.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

# Counter names as Prometheus exposes them (the client appends `_total`).
# Cache queries and hits are counted in tokens, not requests. `mm_cache_*` and
# `prompt_tokens_cached` are not present in every version (older vLLM lacks them),
# and then they are simply absent from the difference.
COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:mm_cache_queries_total",
    "vllm:mm_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)

# Server role -> address key in the `server` config section.
ROLE_URL_KEYS = {"generation": "generation_base_url", "assistant": "assistant_base_url"}


def _root(url: str) -> str:
    root = url.rstrip("/")
    return root[:-len("/v1")] if root.endswith("/v1") else root


def _get(url: str, timeout: float) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode(errors="ignore")
    except Exception:  # noqa: BLE001 — no response, that is all
        return None


def scrape(url: str, timeout: float = 10.0) -> dict[str, float]:
    """Engine counters summed over label sets (with DP, over replicas).

    Empty if the server does not expose them: that is not a measurement failure.
    """
    text = _get(f"{_root(url)}/metrics", timeout)
    if text is None:
        return {}
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in COUNTERS:
            try:
                totals[name] = totals.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return totals


def identity(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Identify which server answers at an address: weights and process start times.

    The process start time (`process_start_time_seconds` of the Prometheus client)
    changes on any restart, and `root` from `/v1/models` is the path to the weights,
    which `--served-model-name` hides behind an alias. With DP there are several
    processes and whichever one answers the scrape is arbitrary, so the values are a set.

    Empty if the server answered neither request.
    """
    root = _root(url)
    out: dict[str, Any] = {}
    models_text = _get(f"{root}/v1/models", timeout)
    if models_text is not None:
        try:
            data = json.loads(models_text).get("data") or []
        except (ValueError, AttributeError):
            data = []
        out["models"] = [
            {key: item.get(key) for key in ("id", "root", "max_model_len") if item.get(key) is not None}
            for item in data if isinstance(item, dict)
        ]
    metrics_text = _get(f"{root}/metrics", timeout)
    if metrics_text is not None:
        starts = set()
        for line in metrics_text.splitlines():
            if line.startswith("process_start_time_seconds"):
                try:
                    starts.add(float(line.rsplit(" ", 1)[1]))
                except (IndexError, ValueError):
                    continue
        if starts:
            out["process_start_time"] = sorted(starts)
    return out


def identities(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Identify all servers of a run: role -> `identity`."""
    server = config.get("server") or {}
    return {role: identity(str(server[key])) for role, key in ROLE_URL_KEYS.items() if server.get(key)}


def snapshot(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Counter snapshot of all servers of a run: role -> counters."""
    server = config.get("server") or {}
    out: dict[str, dict[str, float]] = {}
    for role, key in ROLE_URL_KEYS.items():
        url = server.get(key)
        if url:
            out[role] = scrape(str(url))
    return out


def delta(before: dict[str, float], after: dict[str, float], window_sec: float) -> dict[str, Any]:
    """Difference of two snapshots of one server, and the shares derived from it.

    Empty if either snapshot is missing. A counter that decreased between snapshots
    means the server restarted mid-window: shares are then not computed and the
    output carries `restarted`.
    """
    if not before or not after:
        return {}
    names = [name for name in COUNTERS if name in before and name in after]
    d = {name: after[name] - before[name] for name in names}
    out: dict[str, Any] = {"window_sec": round(window_sec, 1),
                           "counters": {name.removeprefix("vllm:"): round(value, 1) for name, value in d.items()}}
    if any(value < 0 for value in d.values()):
        out["restarted"] = True
        return out

    def share(hits: str, queries: str) -> float | None:
        q = d.get(queries) or 0.0
        return round(d.get(hits, 0.0) / q, 4) if q > 0 else None

    out["prefix_cache_hit_share"] = share("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
    out["mm_cache_hit_share"] = share("vllm:mm_cache_hits_total", "vllm:mm_cache_queries_total")
    out["prompt_tokens_cached_share"] = share("vllm:prompt_tokens_cached_total", "vllm:prompt_tokens_total")
    if window_sec > 0 and d.get("vllm:prompt_tokens_total"):
        out["prompt_tok_per_sec"] = round(d["vllm:prompt_tokens_total"] / window_sec, 1)
        out["generation_tok_per_sec"] = round(d.get("vllm:generation_tokens_total", 0.0) / window_sec, 1)
    return out


def summarize(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    window_sec: float,
) -> dict[str, dict[str, Any]]:
    """Role -> difference; roles without both snapshots are left out of the summary."""
    out: dict[str, dict[str, Any]] = {}
    for role in before:
        d = delta(before.get(role) or {}, after.get(role) or {}, window_sec)
        if d:
            out[role] = d
    return out
