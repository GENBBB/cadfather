#!/usr/bin/env python3
"""Check of seeding: a run must be reproducible, not merely "the seed is passed".

Why this test is separate and why it looks like this. The seed was already "threaded
through the whole path", and the run was still not reproducible: the parameter reached
`choose_point` and went into sampling the cloud, while the point index was taken from the
global `random`. A "seed is passed" test was green on such code. So this checks the
**outcome**, not the passing:

1. the seed derivation does not depend on `PYTHONHASHSEED`, otherwise the seed is stable
   within a run and different between runs, so the defect remains but stops being visible;
2. the point for the same (run, part, step) is the same, and different across steps;
3. a different run seed gives a different sequence of points;
4. the wrapper has no `seed` knob, and the seed that reached the generator equals the
   one derived by the harness;
5. the metrics (CD and GMS) on the same pair of meshes are bitwise identical;
6. a ray with `seed: null` takes its base from the harness, not from an unseeded generator.

Run: `python agent/tests/seed_check.py`. No CAD, GPU or servers are needed.
"""

from __future__ import annotations

import os
import subprocess
import inspect
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The transport to the models is not needed here, and there is none locally: the stub is
# installed before the capabilities are imported, as in `propose_check.py`.
if "openai" not in sys.modules:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub

import numpy as np
import trimesh

from cad_agent import dsl_runtime

dsl_runtime.configure("wrapped")
from cad_agent.capabilities import propose as propose_mod
from cad_agent.capabilities import metrics as metrics_mod
from cad_agent.capabilities.propose import choose_point
from cad_agent.capabilities.resources import build_resources
from cad_agent.harness.config import ConfigError, validate_run_config
from cad_agent.harness.seeding import (  # noqa: E402
    DEFAULT_RUN_SEED, attempt_tag, derive_seed, figure_seed,
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="seed_check_"))
    gt_path = tmp / "gt.stl"
    trimesh.creation.box((30, 20, 10)).export(gt_path)
    pred_path = tmp / "pred.stl"
    trimesh.creation.box((28, 21, 11)).export(pred_path)

    print("1. Seed derivation does not depend on PYTHONHASHSEED")
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from cad_agent.harness.seeding import derive_seed;"
        "print(derive_seed(42, 'mcb/00000022', 3, 'b1', 0))" % str(Path(__file__).resolve().parents[1])
    )
    outs = []
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, env=env, check=True).stdout.strip())
    check("seed is the same under different PYTHONHASHSEED", len(set(outs)) == 1, str(outs))
    check("matches the value computed in this process",
          outs[0] == str(derive_seed(42, "mcb/00000022", 3, "b1", 0)), outs[0])
    # The value is nailed down: changing the seed derivation changes all runs, and must be
    # deliberate rather than a side effect of editing a line.
    check("seed derivation has not changed", derive_seed(42, "mcb/00000022", 3, "b1", 0) == 3904929835,
          str(derive_seed(42, "mcb/00000022", 3, "b1", 0)))
    check("seed fits in 32 bits",
          all(0 <= derive_seed(s, f"f{i}", i) < 2 ** 32 for s in (1, 42, 10 ** 9) for i in range(5)))
    check("different parts get different seeds",
          derive_seed(42, "a", 1) != derive_seed(42, "b", 1))
    check("different branches give different seeds",
          derive_seed(42, "a", 1, "b0") != derive_seed(42, "a", 1, "b1"))
    check("different variants give different seeds",
          derive_seed(42, "a", 1, "", 0) != derive_seed(42, "a", 1, "", 1))

    print("2. The point is reproducible across runs and differs across steps")
    points = propose_mod.sample_gt_points(gt_path)

    def point_at(run_seed: int, step: int, tag: str = "") -> str:
        seed = derive_seed(run_seed, "fig", step, tag)
        return choose_point(gt_path, None, point_seed=seed, gt_points=points)[0]

    first = [point_at(DEFAULT_RUN_SEED, step) for step in range(1, 21)]
    again = [point_at(DEFAULT_RUN_SEED, step) for step in range(1, 21)]
    check("a repeated draw gave the same points", first == again,
          f"{sum(a != b for a, b in zip(first, again))} mismatches")
    check("points differ across steps", len(set(first)) >= 15, f"unique {len(set(first))} of 20")
    other = [point_at(DEFAULT_RUN_SEED + 1, step) for step in range(1, 21)]
    check("another run seed gives other points", sum(a != b for a, b in zip(first, other)) >= 15,
          f"matched {sum(a == b for a, b in zip(first, other))} of 20")
    check("beam branches get different points",
          point_at(DEFAULT_RUN_SEED, 3, "b0") != point_at(DEFAULT_RUN_SEED, 3, "b1"))

    # The "there is a prediction" branch goes through distances to it: locally there is no
    # point_cloud_utils or rtree, so only the distance is stubbed; the index draw itself
    # stays real, and it is the subject of the check.
    original_distance = propose_mod._signed_distance
    propose_mod._signed_distance = lambda pts, mesh: np.full(len(pts), 5.0)
    try:
        def point_with_pred(run_seed: int, step: int) -> str:
            seed = derive_seed(run_seed, "fig", step)
            return choose_point(gt_path, pred_path, point_seed=seed, gt_points=points)[0]

        with_pred = [point_with_pred(DEFAULT_RUN_SEED, step) for step in range(1, 21)]
        check("with a prediction: the repeat gave the same points",
              with_pred == [point_with_pred(DEFAULT_RUN_SEED, step) for step in range(1, 21)])
        check("with a prediction: points differ across steps", len(set(with_pred)) >= 15,
              f"unique {len(set(with_pred))} of 20")
    finally:
        propose_mod._signed_distance = original_distance

    print("3. A forgotten seed fails instead of seeding globally")
    try:
        choose_point(gt_path, None, gt_points=points)  # type: ignore[call-arg]
        check("call without a seed rejected", False, "passed silently")
    except TypeError:
        check("call without a seed rejected", True)

    print("4. The scaffold has no seed knob, and the derived seed arrives intact")
    seen: list[dict] = []

    class RecordingProposer:
        """A generator that only records which seed it was called with."""

        def propose(self, gt_mesh_path, pred_mesh_path, prev_code, k, seed, step=None, tag="",
                    temperature=None, top_p=None):
            # The signature repeats `StepProposer.propose` in full: the policy passes the
            # sampling knobs through it, and a stub without them breaks the call where the
            # live code works.
            seen.append({"seed": seed, "step": step, "tag": tag})
            return []

        def cache_stats(self):
            return {}

    resources = build_resources(
        figure_id="mcb/00000022",
        gt_mesh_path=gt_path,
        work_dir=tmp,
        executor=types.SimpleNamespace(evaluate=lambda tasks: []),
        renderer=types.SimpleNamespace(),
        proposer=RecordingProposer(),
        run_seed=777,
    )
    params = inspect.signature(resources.propose_steps).parameters
    check("the scaffold has no seed knob", "seed" not in params, str(list(params)))
    check("the variant knob exists", "variant" in params, str(list(params)))

    resources.propose_steps(None, "PREFIX", 1, step=4, tag="b1")
    resources.propose_steps(None, "PREFIX", 1, step=4, tag="b1")
    resources.propose_steps(None, "PREFIX", 1, step=4, tag="b1", variant=1)
    check("seed is computed by the harness", seen[0]["seed"] == derive_seed(777, "mcb/00000022", 4, "b1", 0),
          str(seen[0]))
    # A repeat at the same step is a request for new samples: the seed must differ,
    # otherwise resampling returns byte for byte the same.
    check("a repeated call gets a different draw", seen[1]["seed"] != seen[0]["seed"],
          str([s["seed"] for s in seen]))
    check("variant changes the draw", seen[2]["seed"] != seen[0]["seed"])


    # But a second run must repeat the first: the same call order gives the same sequence
    # of seeds. That is reproducibility.
    replay: list[dict] = []
    seen_backup, seen[:] = list(seen), []
    resources_again = build_resources(
        figure_id="mcb/00000022",
        gt_mesh_path=gt_path,
        work_dir=tmp,
        executor=types.SimpleNamespace(evaluate=lambda tasks: []),
        renderer=types.SimpleNamespace(),
        proposer=RecordingProposer(),
        run_seed=777,
    )
    for _ in range(2):
        resources_again.propose_steps(None, "PREFIX", 1, step=4, tag="b1")
    resources_again.propose_steps(None, "PREFIX", 1, step=4, tag="b1", variant=1)
    replay = list(seen)
    check("the second run repeated the seed sequence",
          [s["seed"] for s in replay] == [s["seed"] for s in seen_backup],
          f"{[s['seed'] for s in replay]} vs {[s['seed'] for s in seen_backup]}")
    check("the scaffold base is derived from the run and the part",
          resources.seed == figure_seed(777, "mcb/00000022"), str(resources.seed))

    # The attempt number is a HARNESS knob, not the wrapper's: in the search loop the parent
    # is a candidate from the pool, not a "step, branch", and the harness itself tells a
    # repeat apart (`Origin.attempt`). This checks that an explicit number fixes the draw
    # uniquely: the same number gives the same seed, another gives another. Without it
    # resampling from one parent would buy a copy for a full call.
    seen[:] = []
    fresh = build_resources(
        figure_id="mcb/00000022",
        gt_mesh_path=gt_path,
        work_dir=tmp,
        executor=types.SimpleNamespace(evaluate=lambda tasks: []),
        renderer=types.SimpleNamespace(),
        proposer=RecordingProposer(),
        run_seed=777,
    )
    check("the attempt knob exists",
          "attempt" in inspect.signature(fresh.propose_steps).parameters,
          str(list(inspect.signature(fresh.propose_steps).parameters)))
    fresh.propose_steps(None, "PREFIX", 1, step=4, tag="c1", attempt=0)
    fresh.propose_steps(None, "PREFIX", 1, step=4, tag="c1", attempt=1)
    fresh.propose_steps(None, "PREFIX", 1, step=4, tag="c1", attempt=1)
    check("attempt zero matches the draw without a number",
          seen[0]["seed"] == derive_seed(777, "mcb/00000022", 4, "c1", 0), str(seen[0]))
    check("another attempt gives another draw", seen[1]["seed"] != seen[0]["seed"],
          str([row["seed"] for row in seen]))
    check("the same attempt number gives the same draw", seen[2]["seed"] == seen[1]["seed"],
          str([row["seed"] for row in seen]))
    # The closure counter must know about explicit numbers: a mixed call must not return what
    # was already drawn.
    fresh.propose_steps(None, "PREFIX", 1, step=4, tag="c1")
    check("a call without a number does not repeat what was already drawn",
          seen[3]["seed"] not in {row["seed"] for row in seen[:3]},
          str([row["seed"] for row in seen]))

    print("5. Metrics are noise-free on the same pair of meshes")
    gt_mesh = trimesh.load_mesh(gt_path)
    pred_mesh = trimesh.load_mesh(pred_path)
    cd_a = metrics_mod.compute_cd(gt_mesh, pred_mesh)
    cd_b = metrics_mod.compute_cd(gt_mesh, pred_mesh)
    check("CD matches bit for bit", cd_a == cd_b, f"{cd_a} vs {cd_b}")
    runtime_a = metrics_mod.compute_cd_runtime(gt_mesh, pred_mesh)
    runtime_b = metrics_mod.compute_cd_runtime(gt_mesh, pred_mesh)
    check("runtime CD matches bit for bit", runtime_a == runtime_b, f"{runtime_a} vs {runtime_b}")

    try:
        dsl_runtime.import_gms()
        gms_available = True
    except Exception as exc:  # pragma: no cover — depends on the environment
        gms_available = False
        print(f"  SKIP GMS: the snapshot is unavailable in this environment ({exc})")
    if gms_available:
        gms_a = metrics_mod.compute_gms(gt_mesh, pred_mesh)
        gms_b = metrics_mod.compute_gms(gt_mesh, pred_mesh)
        check("GMS matches bit for bit", gms_a == gms_b, f"{gms_a} vs {gms_b}")
        # Seeding must be local: neighbors in the process keep getting their own numbers,
        # otherwise one measurement would shift everything else.
        np.random.seed(1)
        before = np.random.random()
        np.random.seed(1)
        metrics_mod.compute_gms(gt_mesh, pred_mesh)
        check("the global numpy RNG is not shifted by the measurement", np.random.random() == before)

    print("6. The draw seed is derived by the harness, not by the policy")
    # The policy has no seed knob of its own: the draw is addressed by the formula
    # `H(run_seed, figure_id, step, tag, variant)`, and the same formula is computed in two
    # places, when launching an action and when recording its origin. That repeats from one
    # parent diverge in their draws is checked by the search-loop check; here are the
    # properties of the formula itself.
    base_seed = derive_seed(777, "fig", step=1, tag="c0_stepwise_a0", variant=0)
    check("seed is reproducible",
          derive_seed(777, "fig", step=1, tag="c0_stepwise_a0", variant=0) == base_seed,
          str(base_seed))
    check("seed is a non-negative 32-bit value", 0 <= base_seed < 2 ** 32, str(base_seed))
    # Every coordinate must move the draw: a match in any of them would mean two different
    # calls draw the same point.
    for name, other in (
        ("another part", derive_seed(777, "fig2", step=1, tag="c0_stepwise_a0", variant=0)),
        ("another depth", derive_seed(777, "fig", step=2, tag="c0_stepwise_a0", variant=0)),
        ("another parent", derive_seed(777, "fig", step=1, tag="c1_stepwise_a0", variant=0)),
        ("another variant", derive_seed(777, "fig", step=1, tag="c0_stepwise_a0", variant=1)),
        ("another run seed", derive_seed(778, "fig", step=1, tag="c0_stepwise_a0", variant=0)),
    ):
        check(f"{name} gives a different draw", other != base_seed, f"{other} == {base_seed}")
    # A repeat from the same parent differs by the attempt mark, otherwise resampling would
    # buy copies.
    check("the attempt is part of the tag and changes the draw",
          derive_seed(777, "fig", step=1, tag=attempt_tag("c0_stepwise_a0", 1), variant=0)
          != base_seed)

    print("7. The seed is a config key, and only an integer")
    base = {
        "details": [{"mcb": str(tmp)}], "subsample": {}, "n_workers": 1,
        "server": {"generation_base_url": "http://x/v1", "generation_served_model_name": "g"},
        "scaffold": {"kind": "policy", "policy": "dialogue_lean"},
    }
    try:
        validate_run_config(dict(base, seed=7))
        check("seed: int is accepted", True)
    except ConfigError as exc:
        check("seed: int is accepted", False, str(exc))
    try:
        validate_run_config(dict(base, seed="forty two"))
        check("seed as a string rejected", False, "passed silently")
    except ConfigError:
        check("seed as a string rejected", True)

    print()
    if failures:
        print(f"FAILURES ({len(failures)}): " + "; ".join(failures))
        sys.exit(1)
    # Not "the run is reproducible": what is checked here is the mechanism: the point, the
    # metrics and the seed derivation. The path through vLLM, execution forks and NFS is
    # checked only by a pair of runs with one seed on the server, which has not been done yet.
    print("Seeding is fine: the reproducibility mechanism is in place.")


if __name__ == "__main__":
    main()
