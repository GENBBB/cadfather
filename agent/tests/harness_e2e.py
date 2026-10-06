#!/usr/bin/env python3
"""End-to-end harness run without CAD, models or a server.

Run: ``python agent/tests/harness_e2e.py`` from the ``agent`` directory.

Exactly three things are replaced, all at the boundary with the outside world:

- the request to the generation endpoint: a canonical step text instead of a model;
- the request to the assistant: a call of the function `act` instead of a model:
  `stepwise` on the best candidate of the table (`fake_assistant`), as
  `dialogue_lean` does;
- DSL code execution (`execute._evaluate`): a `trimesh` primitive approaching the
  target with the step number is built instead of CadQuery.

Everything else is real: worker processes, executor forks, rendering, point choice,
runtime metrics, the scaffold, the budget, the run directory layout, aggregates.
This layer (`figure_run`, `pool`, `run_eval`, `artifacts`) is otherwise checked by
nothing until the first server run.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import tempfile
from pathlib import Path

import trimesh

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

# `openai` is absent outside the server environment, and it is needed only as a
# client type: requests in this test are replaced anyway.
if "openai" not in sys.modules:
    import types

    stub = types.ModuleType("openai")
    stub.OpenAI = object
    sys.modules["openai"] = stub

from cad_agent import dsl_runtime  # noqa: E402

dsl_runtime.configure("wrapped")

from PIL import Image, ImageDraw  # noqa: E402

from cad_agent.capabilities import execute, llm, metrics as metrics_mod  # noqa: E402
from cad_agent.capabilities import render as render_mod  # noqa: E402
from cad_agent.harness import figure_run, run_eval, scratch  # noqa: E402
from cad_agent.harness.dataset import load_figures  # noqa: E402

# Locally there is neither a boolean engine nor pykdtree, so IoU and GMS honestly
# fail and print a traceback. This is expected and irrelevant, so it is silenced to
# keep the test output readable.
logging.disable(logging.ERROR)

FAILURES: list[str] = []
# Step number from the task name. The tail after `stepNNN` is a branch label, and it
# can be anything: empty for the fast branch, a branch name for a beam, a parent
# identifier with an attempt number for the search loop. Requiring digits in it
# would mean the stub silently stops seeing the step as soon as the label changes,
# the prediction stops converging to the target, and the test does not notice.
STEP_RE = re.compile(r"step(\d+)")

# A part on which execution always fails: checks that a failure of one figure does
# not bring the run down and honestly reaches the report.
BROKEN_FIGURE = "figure_03"


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --- stubs of the outside world ----------------------------------------------


# The assistant model name in `fake_clients`: `fake_call_vision` uses it to tell a
# question to the assistant from a request to the generator, both go through `call_vision`.
ASSISTANT_MODEL = "fake-assistant"
# How many samples the fake assistant asks `stepwise` for. Two, not one: dedup, the
# candidate table and best-of selection are checked only where more than one
# candidate is born per iteration.
ASSISTANT_N = 2


def fake_assistant(text: str) -> list[dict[str, str]]:
    """A `dialogue_lean` turn: `act(stepwise)` on the best from the table, else on the start.

    The candidate is taken from a table row where `stepwise` is legal: the first
    column is the id, `best`/`start` is the status. The answer is empty if `stepwise`
    is legal for nobody: the policy itself decides what to do with a turn without an action.
    """
    rows = [line.split() for line in text.splitlines() if "stepwise(n≤" in line]
    if not rows:
        return []
    row = next((r for r in rows if "best" in r), None) or next((r for r in rows if "start" in r), rows[0])
    args = {"tool": "stepwise", "candidate": row[0], "n": ASSISTANT_N}
    return [{"id": "call_0", "name": "act", "arguments": json.dumps(args)}]


def fake_call_vision(client, model_name, text, image=None, generation_kwargs=None,
                     system_prompt=None, max_attempts=3, n=1, history=None):
    """The "model" answer together with the call cost: tokens and latency are needed by tech.json.

    `n` answers are returned in one call, as a real endpoint does. The answers are
    deliberately **different**: identical ones would be collapsed by dedup, and the
    check "k variants reached execution" would pass vacuously.
    """
    if model_name == ASSISTANT_MODEL:
        calls = fake_assistant(text)
        return llm.LLMCall(
            text="", texts=[""], model=model_name, has_image=image is not None,
            latency_sec=0.01, prompt_tokens=256, completion_tokens=16, attempts=1,
            tool_calls=calls, tool_calls_all=[calls], finish_reason="tool_calls",
            finish_reasons=["tool_calls"],
        )
    texts = [f"r = r.box({10 + idx})  # point {text}" for idx in range(max(1, n))]
    return llm.LLMCall(
        text=texts[0],
        texts=texts,
        n_requested=max(1, n),
        model=model_name,
        has_image=image is not None,
        latency_sec=0.01,
        prompt_tokens=128,
        completion_tokens=16 * max(1, n),
        attempts=1,
    )


def fake_evaluate(task):
    """Instead of CadQuery: a primitive that converges to the target as the step number grows."""
    if BROKEN_FIGURE in str(task.gt_mesh_path):
        raise RuntimeError("execution failed (intended by the test)")

    match = STEP_RE.search(task.task_id or "")
    step = int(match.group(1)) if match else 1

    gt = trimesh.load_mesh(task.gt_mesh_path)
    extents = gt.extents
    # The prediction's proportions approach the target's: CD decreases over steps.
    drift = 1.0 + 1.0 / (step + 1)
    pred = trimesh.creation.box((extents[0], extents[1], extents[2] * drift))
    pred.apply_scale(200.0 / max(pred.extents))

    Path(task.mesh_path).parent.mkdir(parents=True, exist_ok=True)
    pred.export(task.mesh_path)

    # A field as in the real `_evaluate`: the fork measures its own threads.
    # The stub must repeat the interface, otherwise the test checks the wrong thing.
    from cad_agent.capabilities.execute import own_threads

    result = {
        "mesh_path": task.mesh_path, "metrics": None, "wall_sec": 0.0,
        "n_threads": own_threads(),
        # The breakdown by fork phases is also part of the `_evaluate` interface.
        # Without it the check "phases reach the summary" would pass vacuously: the
        # field would simply not be filled, `exec_phases` would stay empty and the
        # report would silently skip the whole section.
        "phases": {"build_sec": 0.01, "export_sec": 0.02, "metrics_sec": 0.0,
                   "overhead_sec": 0.0, "n": 1},
    }
    if task.measure:
        # The GT cache counter increment is passed out exactly as the real
        # `_evaluate` does. Without these four lines the check "GT warm-up reaches the
        # fork" would pass vacuously: the field would simply not be filled, and
        # `exec_gt` would always be empty.
        before = metrics_mod.gt_cache_stats(execute._GT_CACHE)
        # `extended` and `needs` must arrive exactly as in the real `_evaluate`
        # (`execute.py`). While CD was computed unconditionally, their absence here
        # broke nothing and so was not visible; once CD became requestable, the stub
        # stopped computing anything at all, and the rollout was left without
        # objective values. A stub is part of the interface.
        result["metrics"] = metrics_mod.measure_pair(
            gt_mesh_path=task.gt_mesh_path,
            pred_mesh_path=task.mesh_path,
            gt_cache=execute._GT_CACHE,
            extended=task.extended,
            needs=task.needs,
        )
        after = metrics_mod.gt_cache_stats(execute._GT_CACHE)
        result["gt_cache"] = {
            field: after.get(field, 0) - before.get(field, 0) for field in ("hits", "misses")
        }
    return result


def fake_clients(server_config):
    return {"generation": object(), "generation_model": "fake",
            "assistant": object(), "assistant_model": ASSISTANT_MODEL}


def fake_choose_point(gt_mesh_path, pred_mesh_path, point_seed, gt_points=None):
    """Local replacement for point choice.

    The real `choose_point` from the second step computes distances to the
    prediction: through `point_cloud_utils`, and without it through
    `mesh.nearest.signed_distance`, which needs `rtree`. Locally there is neither.
    On the server the branch works, so only it is replaced, and only here.

    The signature repeats the real one entirely, including the returned cloud: a
    stub that lags behind the interface is a recurring problem, and each time the
    symptom pointed anywhere but at the stub.

    The point is computed **from the seed**, not a constant: a stub that ignores
    `point_seed` would hide both that the seed did not reach it and that the model
    input is the same at all steps. This is exactly the class of error seeding was
    introduced to fix.
    """
    import numpy as np

    rng = random.Random(point_seed)
    point = tuple(rng.randrange(-100, 101) for _ in range(3))
    return str(point).replace(" ", ""), (np.zeros((4, 3)) if gt_points is None else gt_points)


def fake_get_img_stepwise(self, mesh_path, cmap, apply_augs=False, color=None,
                          scale=True, apply_noise=False, noise_scale=0.25):
    """A collage of views WITHOUT VTK: the same size, mode and channel as the real one.

    Why. Eight views through `pv.Plotter.screenshot` take most of the time of the
    harness checks. Meanwhile the picture is looked at by `fake_call_vision`, which
    does not care what is drawn on it: checks at this layer measure caches, prompts
    and artifacts, not pixels. Pixels are measured by `render_check`, which keeps the
    real render.

    What the stub must preserve, because the assertions above stand on it: the size
    and mode (`gt_image` splits the collage into channels, `step_image` merges the
    prediction's red with green and blue of GT), the channel from `color`, and **the
    dependence of the picture on geometry**, otherwise "different meshes gave
    different model input" would be green on a constant. A missing file fails the
    same way as `pv.read`.
    """
    import hashlib

    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(path)

    size = (self.cols * self.view_img_size, self.rows * self.view_img_size)
    collage = Image.new("RGB", size, "white")
    # A figure on the collage is a rectangle whose size and tint come from the mesh
    # content: the same geometry gives the same picture, different geometry a
    # different one, just as with the real render.
    digest = hashlib.sha1(path.read_bytes()).digest()
    shade = 1 + digest[0] % 255
    box = (
        size[0] // 8 + digest[1] % (size[0] // 4),
        size[1] // 8 + digest[2] % (size[1] // 4),
    )
    tint = tuple(shade if channel else 0 for channel in (color or (255, 255, 255)))
    ImageDraw.Draw(collage).rectangle(
        [(size[0] - box[0]) // 2, (size[1] - box[1]) // 2,
         (size[0] + box[0]) // 2, (size[1] + box[1]) // 2],
        fill=tint,
    )
    return collage


def install_fakes(render: bool = True) -> None:
    """Local run stubs. `render=False` keeps the real VTK."""
    llm.call_vision = fake_call_vision
    execute._evaluate = fake_evaluate
    execute._preload_cad = lambda: None
    figure_run._get_clients = fake_clients
    # propose imported the function by name, so it is replaced there too
    from cad_agent.capabilities import propose

    propose.llm.call_vision = fake_call_vision
    propose.choose_point = fake_choose_point

    # The render is replaced on the class, not on an instance: `Plotter` is created
    # lazily in the part's process, which cannot be reached from here. One method is
    # patched, the only one through which both GT and the prediction go into VTK
    # (`FigureRenderer.gt_image` and `.render` call it directly).
    if render:
        render_mod.Plotter._get_img_stepwise = fake_get_img_stepwise


def make_dataset(tmp: Path, n: int) -> Path:
    folder = tmp / "targets"
    folder.mkdir(parents=True, exist_ok=True)
    for idx in range(n):
        mesh = trimesh.creation.box((20 + idx, 15 + 2 * idx, 10 + idx))
        mesh.export(folder / f"figure_{idx:02d}.stl")
    return folder


def base_config(folder: Path, backend: str, n_workers: int, log_level: str = "metrics", profile: bool = False) -> dict:
    return {
        "logging": {"level": log_level, "profile": profile},
        "details": [{"test": str(folder)}],
        "compute_metrics": True,
        "extended_metrics": False,
        "n_workers": n_workers,
        "server": {"generation_base_url": "http://fake", "generation_served_model_name": "fake"},
        "generation": {"max_tokens": 64},
        "execution": {"backend": backend, "pool_size": 2, "timeout_sec": 20},
        "cache": {"render_images": 4},
        "budget": {},
        # The scaffold is the harness search loop with a named policy; the loop caps
        # live next to it, in `limits`, because they are harness caps, not the policy's.
        "scaffold": {"kind": "policy", "policy": "dialogue_lean"},
        # Only `stepwise`: det needs a built `cadfit._native`, and `optimize` needs
        # `_cad_grad`; neither is present in the checks.
        "tools": ["stepwise"],
        "limits": {"iterations": 4, "depth": 4},
        # CD is ordered deliberately: policies go for the `iou` objective, which is not
        # available in every environment: it is computed only where a boolean engine and
        # `pykdtree` are installed. CD is both a metric computed everywhere and the
        # harness fallback field (`search.FALLBACK_FIELD`). Without it the check would
        # measure not the loop but the environment composition: zero steps for all parts
        # and an empty scale.
        "metrics": {"cd": True},
    }


def main() -> None:
    install_fakes()
    tmp = Path(tempfile.mkdtemp(prefix="harness_e2e_"))
    folder = make_dataset(tmp, n=6)

    print("1. Run in a process pool (serial_fork, 3 workers)")
    config = base_config(folder, backend="serial_fork", n_workers=3)
    # The progress bar is replaced with a counting one: this checks that it sees EACH
    # part exactly once, including the failed one. There is nothing to count from logs,
    # since the bar writes nothing, and the error here is silent: an extra or lost
    # `advance` gives a plausible bar that lies about the remainder.
    from cad_agent.harness import progress as progress_mod

    advanced: list[object] = []

    class CountingProgress(progress_mod.Progress):
        def advance(self, record=None):
            advanced.append(record)

    real_build = progress_mod.build
    progress_mod.build = lambda total, logging_config=None: CountingProgress()
    try:
        result = run_eval.run_experiment(config=config, run_dir=tmp / "run1")
    finally:
        progress_mod.build = real_build
    records = result["per_figure"]
    check("the progress bar saw each part exactly once",
          len(advanced) == len(records), f"{len(advanced)} vs {len(records)}")

    figures = load_figures(config["details"])
    check("as many records as parts", len(records) == len(figures), f"{len(records)} vs {len(figures)}")
    check(
        "result order matches the dataset order",
        [r["figure_id"] for r in records] == [f.figure_id for f in figures],
    )

    ok_records = [r for r in records if BROKEN_FIGURE not in r["figure_id"]]
    broken = [r for r in records if BROKEN_FIGURE in r["figure_id"]][0]
    check("working parts produced code", all(r["n_steps"] > 0 for r in ok_records))
    check("workers have a mesh on disk", all(r["mesh_path"] and Path(r["mesh_path"]).exists() for r in ok_records))
    check("a broken part did not crash the run", broken["n_steps"] == 0 and broken["error"] is not None)
    check("a broken part got a zero score", broken["score"] == 0.0)

    print("2. Run artifacts")
    run_dir = Path(result["run_dir"])
    for name in ("config.json", "dataset.json", "per_figure.json", "summary.json"):
        check(f"{name} is present", (run_dir / name).exists())

    figure_dir = run_dir / "figures" / ok_records[0]["figure_id"]
    check("the part directory is created", figure_dir.is_dir())
    check("the best prefix is saved", (figure_dir / "best.py").exists())
    check("the step journal is saved", (figure_dir / "journal.json").exists())
    check("per-figure record is saved", (figure_dir / "figure.json").exists())
    # The render cache moved into memory: there must be no directory on disk at all,
    # and the cache work is visible in the counters in tech.json.
    check("the render cache does not write to disk", not (figure_dir / "_renders").exists())
    check("no images on disk", not any(figure_dir.rglob("*.png")))
    # Candidate meshes live in scratch (tmpfs) and do not reach the run directory:
    # at level metrics only the returned one remains.
    check("the best mesh is saved", (figure_dir / "best.stl").exists())
    check("candidate meshes are not saved",
          [p.name for p in figure_dir.glob("*.stl")] == ["best.stl"],
          str(sorted(p.name for p in figure_dir.glob("*.stl"))))
    check("the part record points to the saved mesh",
          ok_records[0]["mesh_path"] == str(figure_dir / "best.stl"), str(ok_records[0]["mesh_path"]))

    journal = json.loads((figure_dir / "journal.json").read_text(encoding="utf-8"))
    check("the journal has iterations", len(journal) > 0, f"iterations: {len(journal)}")
    # The journal must restore WHAT the scaffold did on an iteration and how it
    # ended: without actions it is a list of numbers, without an outcome a list of
    # actions with no answer to why they were taken.
    check("the journal has the iteration actions",
          all(isinstance(entry.get("actions"), list) for entry in journal)
          and any(entry["actions"] for entry in journal), str(journal[:1]))
    check("the action names tool, parent and reason",
          all(action.get("tool") and action.get("parent") and action.get("reason")
              for entry in journal for action in entry["actions"]), str(journal[:1]))
    check("the journal records the best at the time of the iteration",
          any(entry.get("best") for entry in journal), str(journal[-1:]))
    # The reason is explained where the policy STOPPED: on a working iteration the
    # decision "continue" is self-evident and an empty reason is legal. Requiring a
    # string at every iteration would mean requiring text for the sake of text.
    check("the policy stop is explained, not just named",
          all(entry.get("select_reason") for entry in journal if entry.get("done")),
          str([e.get("select_reason") for e in journal if e.get("done")]))
    # The scale in which the best was chosen is not a journal but a part record: one
    # per part, frozen by the first selection. The scale name is not hardcoded:
    # whether IoU is computed is decided by the environment, locally it is
    # `cd_runtime`, and in the run environment `contract`. A hardcoded name would
    # check the environment, not the harness.
    check("the best-selection scale is recorded in the part record",
          all(r["runtime_metrics"].get("fitness_scale") for r in ok_records),
          str([r["runtime_metrics"].get("fitness_scale") for r in ok_records]))
    # The knob is off: there is no candidate table, neither as a file nor in time.
    check("without the knob the candidate table is not written", not (figure_dir / "candidates.jsonl").exists())

    print("2a. Candidate table (logging.candidates)")
    folder_c = make_dataset(tmp, 2)
    config_c = base_config(folder_c, "serial_fork", 2)
    config_c["logging"]["candidates"] = True
    # Several candidates per iteration is exactly the case the table exists for: with
    # one candidate "candidate metrics" and "step metrics" coincide, and the check
    # would check nothing. The width is set by the fake assistant (`ASSISTANT_N`).
    result_c = run_eval.run_experiment(config=config_c, run_dir=tmp / "run_candidates")
    run_dir_c = Path(result_c["run_dir"])
    record_c = next(
        r for r in result_c["per_figure"]
        if r["error"] is None and BROKEN_FIGURE not in r["figure_id"]
    )
    figure_dir_c = run_dir_c / "figures" / record_c["figure_id"]
    candidates_path = figure_dir_c / "candidates.jsonl"
    check("the candidate table is created", candidates_path.exists())

    rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    journal_c = json.loads((figure_dir_c / "journal.json").read_text(encoding="utf-8"))

    check("more candidates than iterations", len(rows) > len(journal_c),
          f"{len(rows)} rows for {len(journal_c)} iterations")
    # What the table exists for: the measurement of every candidate, not only of the
    # one that survived selection.
    check("the candidate has its full measurement recorded",
          all(isinstance(row.get("metrics"), dict) for row in rows),
          str(rows[:1]))
    check("the measurement contains the selection metric",
          all("cd_runtime" in (row.get("metrics") or {}) for row in rows), str(rows[:1]))
    check("the candidate has its execution outcome recorded",
          all("success" in row and "outcome" in row for row in rows), str(rows[:1]))
    check("the candidate has its code step recorded",
          all(row.get("step_code") for row in rows), str(rows[:1]))
    check("step and candidate number are recorded",
          all(row.get("step") is not None and row.get("index") is not None for row in rows), str(rows[:1]))

    # The halves are stitched: the measurement is written by the harness, the decisions
    # by the policy, and without a shared key the table would answer "how many were
    # measured" but not "whose decision measured it". The key is the candidate label:
    # `<parent>_<tool>_a<N>`.
    planned = {
        (entry["iteration"] + 1, action["parent"], action["tool"])
        for entry in journal_c for action in entry["actions"]
    }
    measured = {
        (row["step"], row["tag"].split("_")[0], row["tag"].split("_")[1])
        for row in rows if row.get("tag")
    }
    check("every measured candidate comes from an action in the journal",
          measured and measured <= planned,
          f"unpaired: {sorted(measured - planned)[:3]}")
    check("the selected best is named in the journal and lives in the pool",
          all(entry["best"] is None or entry["best"].startswith("c") for entry in journal_c),
          str([entry["best"] for entry in journal_c][:3]))
    # What goes out is the best OVER THE WHOLE search, not the last measured. The
    # candidate table is the only place where all measured ones are visible at once,
    # so the assertion is here, not in the journal section. The scale locally is the
    # fallback CD, where smaller is better.
    measured_cds = [
        row["metrics"]["cd_runtime"] for row in rows
        if row.get("success") and (row.get("metrics") or {}).get("cd_runtime") is not None
    ]
    returned_cd = record_c["runtime_metrics"].get("cd_runtime")
    check("the best of all measured is returned",
          measured_cds and returned_cd is not None
          and abs(returned_cd - min(measured_cds)) < 1e-12,
          f"returned {returned_cd}, best of {len(measured_cds)} measured {min(measured_cds) if measured_cds else None}")
    tech_c = json.loads((figure_dir_c / "tech.json").read_text(encoding="utf-8"))
    check("the number of recorded candidates reached tech.json",
          tech_c.get("candidates_logged") == len(rows), str(tech_c.get("candidates_logged")))

    print("3. Deepening a part and returning the best")
    runtime = ok_records[0]["runtime_metrics"]
    # Depth grows: the loop does not mark time at the root. The number itself is the
    # `limits` cap or less, if the policy stopped earlier.
    check("the part went deeper than the root", (runtime.get("depth") or 0) > 0, str(runtime.get("depth")))
    # The runtime record must carry what the report relies on: the value of the
    # metric used for measuring and the scale name. There is NO selection objective
    # here and there must not be: it is a policy knob, the config does not name it,
    # and the report has nothing to invent it from.
    check("the runtime record carries the metric and the scale",
          runtime.get("cd_runtime") is not None and runtime.get("fitness_scale"), str(runtime))
    check("the objective is not invented out of nowhere", "objective" not in runtime, str(runtime))

    # CD is a requestable metric, and `base_config` asks for it (without it no
    # objective metric is computed locally AT ALL, and the part fails). What is
    # checked here is that the order reaches the report; the reverse side, that
    # without the key CD does not appear, is checked below by a direct call, without a run.
    contract = ok_records[0]["metrics"]
    check("the requested CD reached the report",
          contract.get("cd") is not None, f"cd={contract.get('cd')}")
    # score_i does not depend on CD: only IoU and GMS enter it.
    check("score is computed without CD", isinstance(ok_records[0].get("score"), float),
          str(ok_records[0].get("score")))

    # And the reverse: when CD is requested, it appears in both layers. Checked by a
    # direct call on real meshes: there is no need to run the whole pipeline for
    # this, and there is exactly one fork.
    gt_box = trimesh.creation.box((20.0, 15.0, 10.0))
    pred_box = trimesh.creation.box((20.0, 15.0, 12.0))
    metrics_mod.normalize_for_metrics(gt_box, pred_box)
    asked = metrics_mod.evaluate_pair(figure_id="t", gt_mesh=gt_box, pred_mesh=pred_box,
                                      compute_cd=True)
    silent = metrics_mod.evaluate_pair(figure_id="t", gt_mesh=gt_box, pred_mesh=pred_box)
    check("metrics.cd: true gives CD in the report", asked.cd is not None, str(asked.cd))
    check("without the key there is no CD in the report", silent.cd is None, str(silent.cd))
    check("score does not depend on the key", abs(asked.score() - silent.score()) < 1e-12,
          f"{asked.score()} vs {silent.score()}")

    gt_path = str(folder / "figure_00.stl")
    pred_path = str(tmp / "cd_probe.stl")
    trimesh.creation.box((20.0, 15.0, 12.0)).export(pred_path)
    with_cd = metrics_mod.measure_pair(gt_mesh_path=gt_path, pred_mesh_path=pred_path, needs=("cd",))
    without_cd = metrics_mod.measure_pair(gt_mesh_path=gt_path, pred_mesh_path=pred_path, needs=("iou",))
    check("needs=('cd',) gives cd_runtime", with_cd.get("cd_runtime") is not None, str(with_cd))
    check("needs without cd does not compute cd_runtime", "cd_runtime" not in without_cd, str(without_cd))
    check("predicted watertight is computed in both cases",
          "pred_watertight" in with_cd and "pred_watertight" in without_cd, str(without_cd))

    print("4. Cost and set signature")
    summary = result["summary"]
    check("the set signature is recorded", bool(summary["dataset_signature"]))
    check("generator calls are counted", summary["cost_calls_total"].get("vlm", 0) > 0, str(summary["cost_calls_total"]))
    check("executions are counted", summary["cost_calls_total"].get("exec", 0) > 0)
    check("the failure is counted in the summary", summary["num_failed"] >= 1)
    check(
        "cost per part is derived from the total",
        abs(summary["cost_calls_per_figure"]["vlm"] * len(records) - summary["cost_calls_total"]["vlm"]) < 1e-9,
    )

    print("5. Contract run_eval and reproducibility")
    scores = run_eval.run_eval(scaffold=None, shapes=figures, config=config, run_dir=tmp / "run2")
    check("as many scores as figures", len(scores) == len(figures))
    check("all scores are numbers", all(isinstance(score, float) for score in scores))
    signature_2 = json.loads((tmp / "run2" / "dataset.json").read_text(encoding="utf-8"))["signature"]
    check("the set signature is reproducible", signature_2 == summary["dataset_signature"])

    print("6. The same run with the proxy_pool backend")
    config_proxy = base_config(folder, backend="proxy_pool", n_workers=2)
    result_proxy = run_eval.run_experiment(config=config_proxy, run_dir=tmp / "run3")
    proxy_records = result_proxy["per_figure"]
    proxy_ok = [r for r in proxy_records if BROKEN_FIGURE not in r["figure_id"]]
    check("proxy_pool processed all parts", len(proxy_records) == len(figures))
    check("proxy_pool left no failures beyond the expected", len(proxy_ok) == len(ok_records))
    # Exact equality is impossible here: `measure_pair` resamples the prediction at
    # every measurement (only GT points are cached), so CD is stochastic between
    # runs. Compared with a tolerance.
    def close(a, b, rel=0.05):
        left = (a.get("runtime_metrics") or {}).get("cd_runtime")
        right = (b.get("runtime_metrics") or {}).get("cd_runtime")
        if left is None or right is None:
            return False
        return abs(left - right) <= rel * max(abs(left), abs(right), 1e-12)

    check("proxy_pool gave the same best CD (with a sampling tolerance)",
          all(close(a, b) for a, b in zip(proxy_ok, ok_records)),
          str([( (a.get("runtime_metrics") or {}).get("cd_runtime"),
                 (b.get("runtime_metrics") or {}).get("cd_runtime")) for a, b in zip(proxy_ok, ok_records)][:2]))
    check("proxy_pool went through the same number of steps", [a["n_steps"] for a in proxy_ok] == [b["n_steps"] for b in ok_records])

    # The main reason `proxy_pool` exists: the GT warmed up by the shim is inherited
    # by the forked grandchild. `serial_fork` has nobody to inherit from, and the hit
    # share there must be zero, which is not a defect but the difference the
    # backends are chosen by.
    proxy_gt = ((result_proxy["summary"].get("tech") or {}).get("caches") or {}).get("exec_gt") or {}
    serial_gt = ((result["summary"].get("tech") or {}).get("caches") or {}).get("exec_gt") or {}
    check("proxy_pool: the warmed GT reaches the fork",
          proxy_gt.get("hit_rate") == 1.0, str(proxy_gt))
    check("serial_fork: nothing to inherit, no hits",
          serial_gt.get("hit_rate") == 0.0 and serial_gt.get("misses", 0) > 0, str(serial_gt))

    print("7. Quality aggregates on stub metrics")
    fabricated = [
        {"metrics": metrics_mod.FigureMetrics(
            figure_id=f"f{i}", gt_watertight=i % 2 == 0, pred_watertight=True,
            iou=0.8 if i % 2 == 0 else None, gms_norm=0.6,
        ).to_dict(), "cost": {"calls": {"vlm": 2}}, "wall_sec": 1.0}
        for i in range(4)
    ]
    # A failure with a KNOWN GT status must stay in its own stratum. Before, `gt_watertight`
    # was two-valued and defaulted to `False`, so every failure was attributed to
    # `non_watertight_gt`, a fictitious stratum of exactly the number of failures.
    fabricated.append({"metrics": metrics_mod.FigureMetrics(
        figure_id="f4", gt_watertight=True,
        failure=metrics_mod.FAILURE_EXECUTION).to_dict(),
        "cost": {"calls": {"vlm": 1}}, "wall_sec": 1.0, "error": "failed"})
    # A failure where GT could not be read: the status is unknown, and this is a
    # separate bucket, not a silent "not watertight".
    fabricated.append({"metrics": metrics_mod.FigureMetrics(
        figure_id="f5", failure=metrics_mod.FAILURE_EXECUTION).to_dict(),
        "cost": {"calls": {"vlm": 1}}, "wall_sec": 1.0, "error": "GT is unreadable"})
    aggregated = run_eval.summarize(fabricated, signature="test")
    quality = aggregated["quality"]
    strata = quality.get("by_stratum", {})
    check("aggregate is computed", bool(quality), str(list(quality)[:5]))
    check("strata are separated",
          set(strata) == {"watertight_gt", "non_watertight_gt", "unknown_gt"}, str(set(strata)))
    check("a failure with watertight GT did not go to another stratum",
          strata.get("watertight_gt", {}).get("n_figures") == 3,
          str(strata.get("watertight_gt")))
    check("no fictitious stratum: non_watertight is exactly as configured",
          strata.get("non_watertight_gt", {}).get("n_figures") == 2,
          str(strata.get("non_watertight_gt")))
    check("unknown GT status gets its own bucket",
          strata.get("unknown_gt", {}).get("n_figures") == 1, str(strata.get("unknown_gt")))
    check("the failure counted in IR", quality.get("ir", 0) > 0, str(quality.get("ir")))
    check("IR is split into reasons", "ir_execution" in quality and "ir_not_watertight" in quality)

    print("8. Run log: technical metrics and CSV")
    check("per_figure.csv is written", (run_dir / "per_figure.csv").exists())
    csv_head = (run_dir / "per_figure.csv").read_text(encoding="utf-8").splitlines()
    check("the CSV has a row per part", len(csv_head) == len(figures) + 1, f"rows: {len(csv_head)}")
    check("the CSV has cost columns", "n_vlm" in csv_head[0] and "n_exec" in csv_head[0])

    figure_tech = json.loads((figure_dir / "tech.json").read_text(encoding="utf-8"))
    check("tokens are counted", figure_tech["tokens"].get("prompt_visual", 0) > 0, str(figure_tech["tokens"]))
    check("latency is computed", figure_tech["latency_sec"].get("vlm", 0) > 0)
    # Separate breakdown of execution: the total (`latency_sec.exec`) does not go
    # anywhere, but next to it "what exactly for" must lie.
    exec_phases = figure_tech.get("exec_phases") or {}
    check("execution is split by phases",
          {"build_sec", "export_sec", "metrics_sec", "overhead_sec", "n"} <= set(exec_phases),
          str(sorted(exec_phases)))
    check("as many phases counted as candidates finished",
          0 < int(exec_phases.get("n", 0)) <= int(figure_tech["calls"].get("exec", 0)),
          f"n={exec_phases.get('n')} at exec={figure_tech['calls'].get('exec')}")
    check("attempts are counted", figure_tech["attempts"].get("vlm", 0) > 0)
    check("log size is measured", figure_tech["log_bytes"] > 0)
    check("the summary has a token aggregate", summary["tech"]["tokens"].get("prompt_visual", 0) > 0)
    check("execution phases are aggregated over the run",
          float((summary["tech"].get("exec_phases") or {}).get("build_sec", 0)) > 0,
          str(summary["tech"].get("exec_phases")))

    print("8b. Real run time with parallelism")
    # The sum of per-figure times is work, not time. The real wall time is known only
    # to whoever holds the whole pool, so `run_experiment` measures it, and without it
    # "the run took an hour" has nowhere to come from except thin air.
    check("the real rollout wall time is recorded", float(summary.get("wall_sec_rollout") or 0) > 0,
          str(summary.get("wall_sec_rollout")))
    check("run wall time is not less than the rollouts' wall time",
          float(summary.get("wall_sec_run") or 0) >= float(summary.get("wall_sec_rollout") or 0),
          f"run={summary.get('wall_sec_run')} rollout={summary.get('wall_sec_rollout')}")
    check("the sum of per-figure times is not passed off as wall time",
          float(summary["wall_sec_total"]) >= float(summary["wall_sec_rollout"]) * 0.5,
          f"sum={summary['wall_sec_total']:.3f} wall={summary['wall_sec_rollout']:.3f}")
    check("effective parallelism is computed and not above the configured one",
          0 < float(summary.get("effective_workers") or 0) <= float(summary.get("n_workers") or 0) + 0.05,
          f"{summary.get('effective_workers')} of {summary.get('n_workers')}")

    events = [json.loads(line) for line in (figure_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {event["kind"] for event in events}
    check("call events are recorded", {"propose_steps", "evaluate"} <= kinds, str(sorted(kinds)))
    # The search course must be restorable from the journal: what was available
    # (`legal_actions`), what the policy ordered (`tool_call`) and how the part ended
    # (`search_stop`). Without the first one cannot tell "the policy did not want to"
    # from "the tool was unavailable", and these are different diagnoses.
    check("the search trace is recorded", {"legal_actions", "tool_call", "search_stop"} <= kinds,
          str(sorted(kinds)))
    check("the requested action names tool and parent",
          all(event.get("tool") and event.get("parent")
              for event in events if event["kind"] == "tool_call"),
          str([e for e in events if e["kind"] == "tool_call"][:1]))
    check("the part stop is explained",
          all(event.get("reason") for event in events if event["kind"] == "search_stop"),
          str([e for e in events if e["kind"] == "search_stop"][:1]))

    print("8b. Caches: hits are counted and reach the summary")
    figure_caches = figure_tech.get("caches") or {}
    check("caches are recorded in the part tech.json",
          {"gt_image", "gt_points", "render_pred"} <= set(figure_caches),
          str(sorted(figure_caches)))
    # Exactly one miss per part: GT is rendered and sampled once, after that there
    # must be hits. A two would mean the cache does not hold.
    for name in ("gt_image", "gt_points"):
        check(f"{name}: one miss per part",
              figure_caches.get(name, {}).get("misses") == 1,
              str(figure_caches.get(name)))
        check(f"{name}: hits are present",
              (figure_caches.get(name, {}).get("hits") or 0) > 0,
              str(figure_caches.get(name)))
    check("hit rate is computed",
          isinstance(figure_caches.get("gt_image", {}).get("hit_rate"), float),
          str(figure_caches.get("gt_image")))

    run_caches = (summary.get("tech") or {}).get("caches") or {}
    check("caches are aggregated over the run", "gt_image" in run_caches, str(sorted(run_caches)))
    check("the run hit total is not below the per-part one",
          run_caches["gt_image"]["hits"] >= figure_caches["gt_image"]["hits"],
          f"{run_caches['gt_image']} vs {figure_caches['gt_image']}")

    print("9. At the metrics level intermediate meshes do not pile up")
    leftover = list(figure_dir.glob("*.stl"))
    best_mesh = ok_records[0]["mesh_path"]
    check("only the best mesh is left", len(leftover) <= 1, f"meshes: {len(leftover)}")
    check("the best mesh is not deleted", best_mesh and Path(best_mesh).exists())

    print("10. Full level: the log is self-contained")
    config_full = base_config(folder, backend="serial_fork", n_workers=2, log_level="full", profile=True)
    result_full = run_eval.run_experiment(config=config_full, run_dir=tmp / "run4")
    full_dir = Path(result_full["run_dir"]) / "figures" / ok_records[0]["figure_id"]
    step_dirs = sorted(full_dir.glob("step_*"))
    check("step directories are created", len(step_dirs) > 0, f"steps: {len(step_dirs)}")

    first_step = step_dirs[0]
    # Names carry the candidate label (`prompt_<parent>_<tool>_a<N>.txt`): there are
    # several candidates per iteration, and a bare `prompt.txt` would be overwritten
    # between them. So the lookup is by pattern, not by exact name.
    saved = {}
    for name in ("prompt", "raw_answer", "step", "prefix", "status"):
        suffix = ".txt" if name in ("prompt", "raw_answer") else (".py" if name in ("step", "prefix") else ".json")
        found = sorted(first_step.glob(f"{name}_*{suffix}")) or sorted(first_step.glob(f"{name}{suffix}"))
        saved[name] = found[0] if found else None
        check(f"{name}{suffix} is saved", saved[name] is not None,
              str(sorted(p.name for p in first_step.iterdir())))
    status = json.loads(saved["status"].read_text(encoding="utf-8"))
    check("the step status has metrics", (status.get("metrics") or {}).get("cd_runtime") is not None, str(status)[:120])
    check("step prefix is executable accumulated code",
          "PREFIX" not in saved["prefix"].read_text(encoding="utf-8")
          and len(saved["prefix"].read_text(encoding="utf-8")) > 20)
    check("the image given to the model is saved", any(first_step.glob("input*")))
    check("the prompt is not empty", len(saved["prompt"].read_text(encoding="utf-8")) > 0)
    check("raw answer is non-empty", len(saved["raw_answer"].read_text(encoding="utf-8")) > 0)
    check("intermediate meshes are saved at full", len(list(full_dir.glob("*.stl"))) > 1)
    check("at full, exactly the step candidates are saved",
          any(p.name.startswith("step") for p in full_dir.glob("*.stl")),
          str(sorted(p.name for p in full_dir.glob("*.stl"))))

    tech_full = json.loads((full_dir / "tech.json").read_text(encoding="utf-8"))
    check("stage profiling is on", bool(tech_full["stages_sec"]), str(tech_full["stages_sec"]))
    check("stages make sense",
          {"render", "generation", "execute", "score_final", "warm_gt"} <= set(tech_full["stages_sec"]),
          str(sorted(tech_full["stages_sec"])))
    # The final scoring must be a separate stage, not hide in the difference between
    # the part wall time and the sum of stages: it loads two meshes and recomputes
    # IoU/GMS, and on a big set this is not minor.
    check("final scoring is in the stages", tech_full["stages_sec"].get("score_final", 0) > 0,
          str(tech_full["stages_sec"].get("score_final")))
    check("the full log is heavier than the trimmed one", tech_full["log_bytes"] > figure_tech["log_bytes"],
          f"{tech_full['log_bytes']} vs {figure_tech['log_bytes']}")

    print("8. save_meshes: none — no .stl at all, but the score is computed")
    # The quietest place of the whole change: part scoring reads the returned mesh,
    # and if it is removed too early, the part gets zero looking like a failure while
    # the run stays green. What is checked is precisely the score.
    config_none = base_config(folder, backend="serial_fork", n_workers=2)
    config_none["logging"]["save_meshes"] = "none"
    result_none = run_eval.run_experiment(config=config_none, run_dir=tmp / "run5")
    records_none = json.loads((Path(result_none["run_dir"]) / "per_figure.json").read_text(encoding="utf-8"))
    ok_none = [r for r in records_none if BROKEN_FIGURE not in r["figure_id"]]
    none_dir = Path(result_none["run_dir"]) / "figures" / ok_none[0]["figure_id"]
    check("no meshes left on disk", not list(none_dir.glob("*.stl")),
          str(sorted(p.name for p in none_dir.glob("*.stl"))))
    # The score is locally zero (no boolean engine, no pykdtree), so what is checked
    # is what does not depend on it: the mesh was **read** before cleanup. If the
    # scratch is removed too early, there will be `failure` and empty fields here.
    check("the mesh was read from scratch during scoring",
          all((r["metrics"] or {}).get("pred_watertight") is True for r in ok_none),
          str([(r["metrics"] or {}).get("pred_watertight") for r in ok_none]))
    check("the part is not counted as a failure meanwhile",
          all((r["metrics"] or {}).get("failure") is None for r in ok_none),
          str([(r["metrics"] or {}).get("failure") for r in ok_none]))
    check("mesh path is cleared, not pointing nowhere",
          all(r["mesh_path"] is None for r in ok_none), str([r["mesh_path"] for r in ok_none]))
    # `run_dir` from scratch would create the directory again, so we look at the path.
    check("run scratch is removed",
          not (scratch.default_root() / Path(result_none["run_dir"]).name).exists(),
          str(scratch.default_root() / Path(result_none["run_dir"]).name))

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("End-to-end harness run is fine.")


if __name__ == "__main__":
    main()
