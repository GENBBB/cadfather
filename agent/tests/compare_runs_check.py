#!/usr/bin/env python3
"""Check `tools/compare_runs.py` on two synthetic run directories.

Run: ``python agent/tests/compare_runs_check.py``.

The directories are built here with known answers: the per-part score difference
is fixed, outcomes are permuted in a known way, the vLLM logs carry a known number
of requests, and the cgroup counter goes backward or forward. A pair analysis
that stitched parts or logs wrongly would still give plausible numbers, so every
number is compared against what was built, not eyeballed.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR / "tools"))

import compare_runs  # noqa: E402

FAILURES: list[str] = []

COLUMNS = ["figure_id", "score", "stop_reason", "stratum"]
N = 40


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_run(root: Path, *, scores: dict, stops: dict, port: int, model: str,
             periods: tuple[int, int], ts: tuple[float, float], requests: int | None,
             calls: int, signature: str = "sig0") -> None:
    (root / "logs").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(
        {"seed": 42, "server": {"assistant_base_url": f"http://127.0.0.1:{port}/v1"},
         "model": {"assistant_model_path": model}}), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({
        "dataset_signature": signature, "num_figures": N,
        "quality": {"score_with_zeros": sum(scores.values()) / N, "ir": 0.0},
        "cost_calls_total": {"agent_text": calls, "exec": 2 * calls},
        "wall_sec_rollout": 100.0, "wall_sec_run": 100.0,
        "load": {"cpu_usage_sec": 3000.0},
        "tech": {"calls_by_model": {"assistant": calls},
                 "latency_sec": {"agent_text": 2.0 * calls},
                 "tokens_by_call": {"agent_text": {"prompt": 100 * calls, "completion": 10 * calls}},
                 "stages_sec": {"agent": 50.0}, "exec_phases": {"n": 10, "build_sec": 5.0}},
    }), encoding="utf-8")
    with open(root / "per_figure.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(N):
            fid = f"f{i:03d}"
            writer.writerow({"figure_id": fid, "score": scores[fid], "stop_reason": stops[fid],
                             "stratum": "watertight_gt" if i % 4 else "non_watertight_gt"})
    # Unfinished last line, as in a live run.
    (root / "load.jsonl").write_text(
        json.dumps({"ts": ts[0], "cpu_nr_periods": periods[0]}) + "\n"
        + json.dumps({"ts": ts[1], "cpu_nr_periods": periods[1]}) + "\n{\"ts\": 1", encoding="utf-8")
    if requests is not None:
        lines = [
            "INFO 09-22 15:00:09 [__init__.py:241] Automatically detected platform cuda.",
            f"(APIServer pid=4242) INFO 09-22 15:00:13 [utils.py:326] non-default args: "
            f"{{'model': '{model}', 'port': {port}, 'data_parallel_size': 3}}",
            "(ApiServer_1 pid=4243) INFO 09-22 15:00:15 started",
            "(EngineCore_DP0 pid=4300) INFO 09-22 15:00:20 Initializing engine with config: quantization=fp8, x",
        ]
        lines += ['INFO:     127.0.0.1:5 - "POST /v1/chat/completions HTTP/1.1" 200 OK'] * requests
        (root / "logs" / "assistant_vllm.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ids = [f"f{i:03d}" for i in range(N)]
    # Per-part difference is fixed: +0.02 for the first 10, -0.01 for the next 10, zero for the rest.
    base = {fid: 0.5 + 0.005 * i for i, fid in enumerate(ids)}
    delta = {fid: (0.02 if i < 10 else -0.01 if i < 20 else 0.0) for i, fid in enumerate(ids)}
    new = {fid: base[fid] + delta[fid] for fid in ids}
    stops_a = {fid: "limit:iterations" for fid in ids}
    # Five parts (among those that grew by 0.02) changed their outcome to `done`.
    stops_b = {fid: ("done" if i < 5 else "limit:iterations") for i, fid in enumerate(ids)}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.time() - 7200
        make_run(tmp / "A", scores=base, stops=stops_a, port=8013, model="Qwen/X",
                 periods=(1000, 2000), ts=(t0, t0 + 1000), requests=None, calls=300)
        # Second run is later, with a cgroup counter LOWER than the end of the first: a different pod.
        make_run(tmp / "B", scores=new, stops=stops_b, port=8014, model="/snap/X-FP8",
                 periods=(500, 900), ts=(t0 + 2000, t0 + 3000), requests=250, calls=250)
        a, b = compare_runs.read_run(tmp / "A"), compare_runs.read_run(tmp / "B")

        q = compare_runs.quality(a, b, 500)
        p = q["paired"]
        expected = sum(delta.values()) / N
        check("paired difference equals the constructed one", abs(p["mean"] - expected) < 1e-12,
              f"{p['mean']} vs {expected}")
        check("better / same / worse by construction", (p["better"], p["same"], p["worse"]) == (10, 20, 10),
              str((p["better"], p["same"], p["worse"])))
        check("the bootstrap interval covers the mean", p["ci95"][0] <= p["mean"] <= p["ci95"][1],
              str(p["ci95"]))
        check("strata come from the reference and sum to all parts",
              sum(s["n"] for s in q["by_stratum"].values()) == N, str(q["by_stratum"].keys()))
        check("outcome transitions are counted", q["stop_moved"] == {"limit:iterations -> done": 5},
              str(q["stop_moved"]))
        check("the cost of B's self-stop is taken over the same parts in A",
              q["done_b"]["n"] == 5 and abs(q["done_b"]["delta_mean"] - 0.02) < 1e-12, str(q["done_b"]))

        cond = compare_runs.conditions(a, b, None)
        check("pod: counter went back means different pods", cond["pod"].startswith("different pods"), cond["pod"])
        check("config.json: exactly two differences",
              set(cond["config_diff"]) == {"server.assistant_base_url", "model.assistant_model_path"},
              str(list(cond["config_diff"])))
        srv_b = cond["servers"]["B"]["assistant"]
        check("the vLLM log is parsed", (srv_b.get("pid"), srv_b.get("port"), srv_b.get("data_parallel_size"),
                                        srv_b.get("quantization"), srv_b.get("requests"))
              == ("4242,4243", "8014", "3", "fp8", 250), str(srv_b))
        check("without a log: own calls, but no invented count",
              "log" not in cond["servers"]["A"]["assistant"]
              and cond["servers"]["A"]["assistant"]["own_calls"] == 300, str(cond["servers"]["A"]))
        check("a partially written load.jsonl line does not break the window", b["load"]["end"] == t0 + 3000)

        # Overlap: the pod is not judged.
        make_run(tmp / "C", scores=new, stops=stops_b, port=8014, model="/snap/X-FP8",
                 periods=(500, 900), ts=(t0 + 500, t0 + 1500), requests=10, calls=250)
        c = compare_runs.read_run(tmp / "C")
        check("overlap: the pod is not judged", "overlapped" in compare_runs.conditions(a, c, None)["pod"])

        cost_b = compare_runs.cost(b)
        check("cost per call and tokens per call",
              (cost_b["per_call_sec.agent_text"], cost_b["prompt_per_call.agent_text"],
               cost_b["pod_cores_mean"], cost_b["exec_ms.build"]) == (2.0, 100.0, 30.0, 500.0), str(cost_b))

        # The code check goes by file arrival time (`ctime`), which `os.utime`
        # cannot fake; so the run windows are moved here, not the files.
        repo = tmp / "repo"
        for tree in ("agent/cad_agent/harness", "agent/cad_agent/scaffold", "agent/cad_agent/capabilities"):
            (repo / tree).mkdir(parents=True)
        # `code_digest.py` lives in the key tree and arrives together with an edit,
        # so it is listed among the arrived files alongside it.
        (repo / "agent/cad_agent/harness/code_digest.py").write_text(
            (AGENT_DIR / "cad_agent/harness/code_digest.py").read_text(encoding="utf-8"), encoding="utf-8")
        arrived = ["agent/cad_agent/harness/code_digest.py", "agent/cad_agent/harness/edited.py"]
        edited = repo / "agent/cad_agent/harness/edited.py"
        edited.write_text("x = 1\n", encoding="utf-8")
        # `mtime` in the past, as for a file edited before the start and uploaded with `rsync -a`.
        os.utime(edited, (t0 - 100, t0 - 100))
        cond = compare_runs.conditions(a, b, repo)
        check("arrived after both, even though mtime is old",
              sorted(cond["code_changed"]["after_both"]) == arrived,
              str(cond["code_changed"]))
        check("the digest is computed", bool(cond["code_digest_now"]))
        now = time.time()
        make_run(tmp / "D", scores=new, stops=stops_b, port=8014, model="/snap/X-FP8",
                 periods=(2500, 2900), ts=(now + 600, now + 1600), requests=10, calls=250)
        d = compare_runs.read_run(tmp / "D")
        cond = compare_runs.conditions(a, d, repo)
        check("arrived between the half starts: a separate line",
              sorted(cond["code_changed"]["between_starts"]) == arrived,
              str(cond["code_changed"]))
        make_run(tmp / "E", scores=base, stops=stops_a, port=8013, model="Qwen/X",
                 periods=(1000, 2000), ts=(now + 100, now + 200), requests=None, calls=300)
        cond = compare_runs.conditions(compare_runs.read_run(tmp / "E"), d, repo)
        check("arrived before the first start: not named", not any(cond["code_changed"].values()),
              str(cond["code_changed"]))

        run = subprocess.run([sys.executable, str(AGENT_DIR / "tools/compare_runs.py"),
                              str(tmp / "A"), str(tmp / "B"), "--no-code", "--bootstrap", "200"],
                             capture_output=True, text=True, check=False)
        check("the command line works", run.returncode == 0 and "B − A, paired" in run.stdout,
              run.stderr[-300:])

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Pair analysis is fine.")


if __name__ == "__main__":
    main()
