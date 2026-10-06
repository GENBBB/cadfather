#!/usr/bin/env python3
"""Analyse a pair of runs in one call: conditions check, paired quality, cost.

Collects the steps that used to be repeated by hand for every pair: bootstrap,
outcome transitions, tokens per call, the cgroup counter.

Standard library only, read-only. Run it from the repository root on the machine
where the runs were executed, so that the code check looks at the tree the runs
used:

    python3 agent/tools/compare_runs.py work_dirs/<baseline> work_dirs/<new>

On downloaded run directories the code and server checks are silent
(`--no-code` disables the code check explicitly).

What it does NOT do: it knows no noise floor. The interval is a bootstrap over
parts, so it describes the spread across parts, not the spread across runs.
Transcripts are not read; turn and answer counters come from
`dialogue_trace_stats.py`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_trace_stats import _bootstrap  # noqa: E402  one bootstrap shared by all tools

# "Part did not move" threshold, the same as in `dialogue_trace_stats.compare`:
# two tools counting ties differently would give two better/worse tallies.
SAME_EPS = 1e-4

_VLLM_TS = re.compile(r"(?:INFO|WARNING) (\d\d-\d\d \d\d:\d\d:\d\d)")
# A single server logs `(APIServer pid=…)`; with several API processes
# (`data_parallel_size` > 1) it logs `(ApiServer_N pid=…)`, one per process.
_VLLM_PID = re.compile(r"\((?:APIServer|ApiServer_\d+) pid=(\d+)\)")
_VLLM_ARG = {
    "model": re.compile(r"'model': '([^']+)'"),
    "port": re.compile(r"'port': (\d+)"),
    "data_parallel_size": re.compile(r"'data_parallel_size': (\d+)"),
}
_VLLM_QUANT = re.compile(r"quantization=(\w+)")
_VLLM_POST = "POST /v1/chat/completions"


# ---- reading a run directory ------------------------------------------------

def _json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _flat(data, prefix: str = "") -> dict:
    out = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(_flat(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = data
    return out


def _load_edges(path: Path) -> dict:
    """First and last `load.jsonl` samples: run window and the cgroup counter."""
    if not path.is_file():
        return {}
    first: dict | None = None
    last: dict = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line of a live run
            first = first or row
            last = row
    if not first:
        return {}
    return {"start": first.get("ts"), "end": last.get("ts"),
            "periods_start": first.get("cpu_nr_periods"),
            "periods_end": last.get("cpu_nr_periods")}


def _vllm_log(path: Path) -> dict:
    info: dict = {"log": str(path), "requests": 0}
    pids: set[int] = set()
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if _VLLM_POST in line:
                info["requests"] += 1
                continue
            if "started" not in info:
                match = _VLLM_TS.search(line)
                if match:
                    info["started"] = match.group(1)
            match = _VLLM_PID.search(line)
            if match:
                pids.add(int(match.group(1)))
            if "non-default args" in line:
                for key, pattern in _VLLM_ARG.items():
                    match = pattern.search(line)
                    if match:
                        info[key] = match.group(1)
            if "quantization" not in info:
                match = _VLLM_QUANT.search(line)
                if match:
                    info["quantization"] = match.group(1)
    info["pid"] = ",".join(str(pid) for pid in sorted(pids)) or None
    return info


def read_run(root: Path) -> dict:
    run = {"name": root.name, "root": root}
    run["config"] = _json(root / "config.json")
    run["summary"] = _json(root / "summary.json")
    with open(root / "per_figure.csv", encoding="utf-8", newline="") as handle:
        run["figures"] = {row["figure_id"]: row for row in csv.DictReader(handle)}
    run["load"] = _load_edges(root / "load.jsonl")
    # Run start is the first load sample; without it, the config write time
    # (written at start, before the first part).
    run["start"] = run["load"].get("start") or (root / "config.json").stat().st_mtime
    # Server logs are in the run directory only if the run started the servers
    # itself. A reused server (`--keep-servers`) logs where it was started, and
    # its request count is the sum over all runs that used it.
    run["servers"] = {}
    for log in sorted((root / "logs").glob("*_vllm.log")):
        run["servers"][log.name[: -len("_vllm.log")]] = _vllm_log(log)
    return run


# ---- conditions check -----------------------------------------------------------

def _code_digest(repo: Path) -> tuple[str | None, list[str]]:
    """Tree digest and the list of paths it covers.

    `code_digest.py` is loaded by file, not through the package: importing a
    submodule would run `cad_agent/__init__` with everything it pulls in.
    """
    path = repo / "agent/cad_agent/harness/code_digest.py"
    if not path.is_file():
        return None, []
    spec = importlib.util.spec_from_file_location("_code_digest", path)
    if spec is None or spec.loader is None:
        return None, []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = []
    for tree in module.CODE_TREES:
        files += [str(p) for p in sorted((repo / tree).rglob("*.py"))
                  if "__pycache__" not in p.parts]
    files += [str(repo / name) for name in module.CODE_FILES]
    return module.code_digest(repo), files


def _fmt_ts(ts: float | None) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"


def conditions(a: dict, b: dict, repo: Path | None) -> dict:
    out: dict = {}
    sig_a = a["summary"].get("dataset_signature")
    sig_b = b["summary"].get("dataset_signature")
    out["dataset_same"] = sig_a == sig_b
    out["dataset_signature"] = (sig_a or "")[:12]

    fa, fb = _flat(a["config"]), _flat(b["config"])
    out["config_diff"] = {key: (fa.get(key), fb.get(key))
                          for key in sorted(set(fa) | set(fb)) if fa.get(key) != fb.get(key)}

    out["window"] = {run["name"]: (run["load"].get("start"), run["load"].get("end"))
                     for run in (a, b)}
    # The CFS period counter is monotonic within a pod. A later run whose counter
    # is LOWER than the end of the earlier one is certainly a different pod.
    # The converse does not prove the same pod: counters of different pods can be anything.
    first, second = sorted((a, b), key=lambda run: run["start"])
    p_end, p_start = first["load"].get("periods_end"), second["load"].get("periods_start")
    ea, sb = first["load"].get("end"), second["load"].get("start")
    if None in (p_end, p_start, ea, sb):
        out["pod"] = "not checked: no load.jsonl"
    elif sb < ea:
        out["pod"] = "runs overlapped: the pod cannot be checked by the counter"
    elif p_start < p_end:
        out["pod"] = "different pods (the cgroup counter went backwards)"
    else:
        out["pod"] = "the cgroup counter does not contradict one pod (does not prove it)"

    out["servers"] = {}
    for run in (a, b):
        calls = (run["summary"].get("tech") or {}).get("calls_by_model") or {}
        rows = {}
        for role in sorted(set(run["servers"]) | set(calls)):
            info = dict(run["servers"].get(role) or {})
            info["own_calls"] = calls.get(role)
            rows[role] = info
        out["servers"][run["name"]] = rows

    if repo is not None:
        digest, files = _code_digest(repo)
        out["code_digest_now"] = digest
        # An edit between the two starts means the halves ran on different code
        # (the pair is broken); after the second start it does not matter if the
        # run imported its code at start; after both ended it does not affect the
        # pair at all, only the comparison with the current tree.
        #
        # The time is `ctime`, not `mtime`: files are synced with `rsync -a`, so
        # `mtime` on the node is the edit time on the workstation, not the arrival
        # time. A file edited locally before the start and synced during the run
        # would look old by `mtime` (and `find -newer` misses it). `ctime` is set
        # by the write on this machine and cannot be carried over.
        second_end = second["load"].get("end") or second["start"]
        changed: dict[str, list[str]] = {"between_starts": [], "during_second": [], "after_both": []}
        for path in files:
            if not os.path.exists(path):
                continue
            arrived = os.stat(path).st_ctime
            if arrived <= first["start"]:
                continue
            kind = ("between_starts" if arrived <= second["start"]
                    else "during_second" if arrived <= second_end else "after_both")
            changed[kind].append(os.path.relpath(path, repo))
        out["code_changed"] = changed
    return out


# ---- quality -------------------------------------------------------------

def _score(row: dict) -> float:
    return float(row.get("score") or 0.0)


def _paired(deltas: list[float], rounds: int) -> dict:
    if not deltas:
        return {"n": 0}
    low, high = _bootstrap(deltas, rounds)
    return {"n": len(deltas), "mean": st.mean(deltas),
            "se": st.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else None,
            "sd_d": st.stdev(deltas) if len(deltas) > 1 else None,
            "ci95": [low, high],
            "better": sum(d > SAME_EPS for d in deltas),
            "same": sum(abs(d) <= SAME_EPS for d in deltas),
            "worse": sum(d < -SAME_EPS for d in deltas)}


def quality(a: dict, b: dict, rounds: int) -> dict:
    common = sorted(set(a["figures"]) & set(b["figures"]))
    fa, fb = a["figures"], b["figures"]
    delta = {fid: _score(fb[fid]) - _score(fa[fid]) for fid in common}
    out: dict = {"only_a": len(fa) - len(common), "only_b": len(fb) - len(common)}
    out["runs"] = {}
    for run in (a, b):
        q = run["summary"].get("quality") or {}
        out["runs"][run["name"]] = {key: q.get(key) for key in
                                    ("score_with_zeros", "ir", "ir_execution",
                                     "ir_not_watertight", "ir_iou_unavailable", "ir_no_result")
                                    if key in q or key == "score_with_zeros"}
    out["paired"] = _paired(list(delta.values()), rounds)
    strata = sorted({fa[fid].get("stratum") or "—" for fid in common})
    out["by_stratum"] = {s: _paired([delta[f] for f in common
                                     if (fa[f].get("stratum") or "—") == s], rounds)
                         for s in strata}
    out["stop_a"] = dict(Counter(fa[f]["stop_reason"] for f in common).most_common())
    out["stop_b"] = dict(Counter(fb[f]["stop_reason"] for f in common).most_common())
    moved = Counter((fa[f]["stop_reason"], fb[f]["stop_reason"]) for f in common
                    if fa[f]["stop_reason"] != fb[f]["stop_reason"])
    out["stop_moved"] = {f"{x} -> {y}": n for (x, y), n in moved.most_common(10)}
    # Self-stop of one half and its cost: the same part in the other half.
    for label, run, other in (("done_b", b, a), ("done_a", a, b)):
        ids = [f for f in common if str(run["figures"][f]["stop_reason"]).startswith("done")]
        sign = 1 if run is b else -1
        out[label] = {"n": len(ids),
                      "delta_mean": st.mean(sign * delta[f] for f in ids) if ids else None}
    return out


# ---- cost -------------------------------------------------------------------

def _div(x, y):
    return x / y if x is not None and y else None


def cost(run: dict) -> dict:
    s = run["summary"]
    tech = s.get("tech") or {}
    calls = s.get("cost_calls_total") or {}
    latency = tech.get("latency_sec") or {}
    stages = tech.get("stages_sec") or {}
    tokens = tech.get("tokens_by_call") or {}
    phases = tech.get("exec_phases") or {}
    load = s.get("load") or {}
    n = s.get("num_figures") or len(run["figures"])
    out = {"figures": n,
           "wall_rollout_sec": s.get("wall_sec_rollout"),
           "effective_workers": s.get("effective_workers"),
           "pod_cores_mean": _div(load.get("cpu_usage_sec"), s.get("wall_sec_run")),
           "cfs_throttled_share": load.get("cpu_periods_throttled_share")}
    for kind in ("agent_text", "agent_visual", "vlm", "exec", "det", "opt"):
        out[f"calls.{kind}"] = calls.get(kind)
        out[f"per_call_sec.{kind}"] = _div(latency.get(kind), calls.get(kind))
    for kind, tok in tokens.items():
        out[f"prompt_per_call.{kind}"] = _div(tok.get("prompt"), calls.get(kind))
        out[f"answer_per_call.{kind}"] = _div(tok.get("completion"), calls.get(kind))
    for stage, sec in sorted(stages.items(), key=lambda kv: -kv[1]):
        out[f"stage_sec.{stage}"] = sec
    built = phases.get("n")
    for phase in ("build_sec", "metrics_sec", "export_sec"):
        ms = _div(phases.get(phase), built)
        out[f"exec_ms.{phase[:-4]}"] = ms * 1000 if ms is not None else None
    for key in ("n_worker_deaths",):
        out[key] = tech.get(key)
    return out


# ---- printing -------------------------------------------------------------------

def _num(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}".replace(",", " ")
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    return str(v)


def _pair_line(p: dict) -> str:
    if not p.get("n"):
        return "no common parts"
    lo, hi = p["ci95"]
    return (f"{p['mean']:+.4f}  SE {_num(p['se'])}  95% [{lo:+.4f}, {hi:+.4f}]  "
            f"better/equal/worse {p['better']}/{p['same']}/{p['worse']}  sd_d {_num(p['sd_d'])}  n {p['n']}")


def render(a: dict, b: dict, cond: dict, qual: dict, costs: tuple[dict, dict]) -> str:
    na, nb = a["name"], b["name"]
    lines = [f"A = {na}", f"B = {nb}", "", "Pair check", "-" * 40]
    lines.append(f"  dataset          {'same' if cond['dataset_same'] else 'DIFFERENT'}"
                 f" (signature {cond['dataset_signature']})")
    for name, (start, end) in cond["window"].items():
        lines.append(f"  window {name}: {_fmt_ts(start)} – {_fmt_ts(end)}")
    lines.append(f"  pod              {cond['pod']}")
    diff = cond["config_diff"]
    lines.append(f"  config.json      {'identical' if not diff else f'differs in {len(diff)} keys:'}")
    for key, (va, vb) in diff.items():
        lines.append(f"    {key}: {va!r} -> {vb!r}")
    for name, roles in cond["servers"].items():
        for role, info in roles.items():
            if "log" not in info:
                lines.append(f"  server {role} [{name}]: no log in the directory (reused?),"
                             f" own calls {_num(info.get('own_calls'))}")
                continue
            own, req = info.get("own_calls"), info.get("requests")
            verdict = "no foreign calls" if own == req else "MISMATCH - foreign clients or a different count"
            lines.append(f"  server {role} [{name}]: pid {info.get('pid')}, started {info.get('started')},"
                         f" port {info.get('port')}, dp {info.get('data_parallel_size') or 1},"
                         f" quant {info.get('quantization') or '—'}")
            lines.append(f"      model {info.get('model')}")
            lines.append(f"      requests in the log {_num(req)}, own calls {_num(own)} - {verdict}")
    if "code_digest_now" in cond:
        changed = cond["code_changed"]
        lines.append(f"  code_digest      {cond['code_digest_now']} (tree now)")
        if not any(changed.values()):
            lines.append("  code unchanged since the first half started - both halves are on this digest")
        labels = {"between_starts": "between starts - HALVES ON DIFFERENT CODE",
                  "during_second": "during the second - check that it was imported before the edit",
                  "after_both": "after both - does not affect the pair, the current digest is not theirs"}
        for kind, paths in changed.items():
            if paths:
                lines.append(f"  code edited {labels[kind]}: {', '.join(paths)}")

    lines += ["", "Quality", "-" * 40]
    for name, q in qual["runs"].items():
        parts = [f"score {_num(q.get('score_with_zeros'))}"]
        parts += [f"{k} {_num(v)}" for k, v in q.items() if k != "score_with_zeros"]
        lines.append(f"  {name}: " + ", ".join(parts))
    if qual["only_a"] or qual["only_b"]:
        lines.append(f"  NON-COMMON parts: only in A {qual['only_a']}, only in B {qual['only_b']}")
    lines.append(f"  B − A, paired:   {_pair_line(qual['paired'])}")
    for stratum, p in qual["by_stratum"].items():
        lines.append(f"    {stratum:20} {_pair_line(p)}")
    lines.append("  outcomes A: " + ", ".join(f"{k} {v}" for k, v in qual["stop_a"].items()))
    lines.append("  outcomes B: " + ", ".join(f"{k} {v}" for k, v in qual["stop_b"].items()))
    if qual["stop_moved"]:
        lines.append("  outcome changed: " + "; ".join(f"{k} {v}" for k, v in qual["stop_moved"].items()))
    for label, side in (("done_b", "B"), ("done_a", "A")):
        d = qual[label]
        lines.append(f"  self-stop {side}: {d['n']} parts, difference with the other half on them"
                     f" {_num(d['delta_mean'])} (sign: in favour of the one that stopped)")

    ca, cb = costs
    lines += ["", "Cost", "-" * 40, f"  {'':34}{'A':>14}{'B':>14}{'B/A':>8}"]
    for key in dict.fromkeys(list(ca) + list(cb)):
        va, vb = ca.get(key), cb.get(key)
        ratio = _div(vb, va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        lines.append(f"  {key:34}{_num(va):>14}{_num(vb):>14}{(f'{ratio:.2f}' if ratio else ''):>8}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("a", help="baseline run directory")
    parser.add_argument("b", help="directory of the run being compared")
    parser.add_argument("--repo", default=".", help="root of the tree the runs used (code check)")
    parser.add_argument("--no-code", action="store_true", help="skip the code check (directories were downloaded)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json", help="write everything computed to a file")
    args = parser.parse_args()

    a, b = read_run(Path(args.a)), read_run(Path(args.b))
    cond = conditions(a, b, None if args.no_code else Path(args.repo).resolve())
    qual = quality(a, b, args.bootstrap)
    costs = (cost(a), cost(b))
    print(render(a, b, cond, qual, costs))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"a": a["name"], "b": b["name"], "conditions": cond, "quality": qual,
                       "cost": {a["name"]: costs[0], b["name"]: costs[1]}},
                      handle, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
