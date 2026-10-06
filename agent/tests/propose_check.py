#!/usr/bin/env python3
"""Check of the step generator: how many requests, executions and samplings.

Run: ``python agent/tests/propose_check.py`` from the repository root.

It checks exactly what is easy to break unnoticed, because quality does not change,
only the price does:

1. k variants go in **one** request with the `n` parameter, not in k requests;
2. identical samples collapse before execution, the multiplicity is not lost;
3. the GT point cloud is sampled once per part, not at every step;
4. `evaluate_codes` does not execute the same code twice, but returns a result for every
   code submitted, since wrappers zip them with their candidates.

No CAD, model or network is needed: the transport and execution are stubbed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

if "openai" not in sys.modules:
    stub = types.ModuleType("openai")
    stub.OpenAI = object
    sys.modules["openai"] = stub

from cad_agent import dsl_runtime  # noqa: E402

dsl_runtime.configure("wrapped")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from cad_agent.capabilities import llm, propose as propose_mod  # noqa: E402
from cad_agent.capabilities.propose import StepProposer  # noqa: E402
from cad_agent.capabilities.resources import Resources  # noqa: E402
from cad_agent.harness.budget import Budget  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


class FakeRenderer:
    """Rendering is stubbed: this checks the generator, not the image."""

    def __init__(self):
        self.calls = 0

    def step_image(self, gt_mesh_path, pred_mesh_path):
        self.calls += 1
        return "IMAGE"


def make_proposer(answers, tmp: Path, **kwargs) -> tuple[StepProposer, dict]:
    """A generator with a fake transport. `answers` is what the "model" will return."""
    seen: dict = {"requests": 0, "n": [], "kwargs": []}

    def fake_call_vision(client, model_name, text, image=None, generation_kwargs=None,
                         system_prompt=None, max_attempts=3, n=1):
        seen["requests"] += 1
        seen["n"].append(n)
        seen["kwargs"].append(dict(generation_kwargs or {}))
        texts = list(answers[: max(1, n)])
        return llm.LLMCall(
            text=texts[0], texts=texts, n_requested=n, model=model_name,
            has_image=image is not None, latency_sec=0.01,
            prompt_tokens=100, completion_tokens=10 * len(texts),
        )

    propose_mod.llm.call_vision = fake_call_vision
    proposer = StepProposer(
        client=object(), model_name="gen", renderer=FakeRenderer(),
        temperature=1.0, **kwargs,
    )
    return proposer, seen


def main() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="propose_check_"))
    gt_path = tmp / "gt.stl"
    trimesh.creation.box((30, 20, 10)).export(gt_path)

    print("1. k variants — one request with n=k")
    answers = ["r = r.box(1)", "r = r.box(2)", "r = r.box(3)"]
    proposer, seen = make_proposer(answers, tmp)
    proposals = proposer.propose(gt_path, None, "PREFIX", k=3, step=1, seed=7)

    check("exactly one request", seen["requests"] == 1, f"requests: {seen['requests']}")
    check("the request asked for k answers", seen["n"] == [3], f"n: {seen['n']}")
    check("all three variants came back", len(proposals) == 3, f"variants: {len(proposals)}")
    check("variants differ in code", len({p.step_code for p in proposals}) == 3)
    check("the prefix is attached to each", all(p.full_code.startswith("PREFIX") for p in proposals))

    print("2. Duplicates are collapsed, multiplicity is kept")
    proposer, seen = make_proposer(["r = r.box(1)", "r = r.box(1)", "r = r.box(2)"], tmp)
    proposals = proposer.propose(gt_path, None, "PREFIX", k=3, step=1, seed=7)
    check("two unique ones remain", len(proposals) == 2, f"variants: {len(proposals)}")
    check("multiplicity is recorded", sorted(p.n_samples for p in proposals) == [1, 2],
          str([p.n_samples for p in proposals]))
    check("indices of the unique ones are consecutive", [p.index for p in proposals] == [0, 1],
          str([p.index for p in proposals]))

    proposer, seen = make_proposer(["r = r.box(1)", "r = r.box(1)"], tmp, dedupe=False)
    proposals = proposer.propose(gt_path, None, "PREFIX", k=2, step=1, seed=7)
    check("dedupe: false keeps duplicates", len(proposals) == 2, f"variants: {len(proposals)}")

    print("3. A failed request is k failed variants")

    def failing_call(client, model_name, text, image=None, generation_kwargs=None,
                     system_prompt=None, max_attempts=3, n=1):
        return llm.LLMCall(text="", texts=[], n_requested=n, model=model_name,
                           error="connection dropped")

    propose_mod.llm.call_vision = failing_call
    proposer = StepProposer(client=object(), model_name="gen", renderer=FakeRenderer(),
                            temperature=1.0)
    proposals = proposer.propose(gt_path, None, "PREFIX", k=3, step=1, seed=7)
    check("on a failure k variants come back", len(proposals) == 3, f"variants: {len(proposals)}")
    check("each has an error recorded", all(p.error for p in proposals))

    print("4. Zero temperature is a POLICY choice, not a run mode")
    # Zero temperature used to be controlled by the config key `generation.greedy`, and the
    # generator silently took `n_samples = 1` for any k. That cancelled `n`, temperature and
    # `top_p` for any policy: whole runs went without sampling, and the metrics did not show it.
    #
    # Now `n` goes as asked and zero temperature is just a knob value. Whoever ordered the
    # copies pays for them: the request goes for k samples, dedup collapses them into one
    # candidate, and the multiplicity is visible in `n_samples`. This is exactly what is
    # checked, because the price of k copies is an argument against such a pair of knobs, and
    # it must be observable.
    proposer, seen = make_proposer(["r = r.box(1)"] * 5, tmp)
    proposals = proposer.propose(gt_path, None, "PREFIX", k=5, step=1, seed=7, temperature=0.0)
    check("asked for exactly k, as the policy requested", seen["n"] == [5], f"n: {seen['n']}")
    check("temperature went out as zero", seen["kwargs"][0].get("temperature") == 0.0,
          str(seen["kwargs"][0]))
    check("top_p is not passed at zero", "top_p" not in seen["kwargs"][0],
          str(seen["kwargs"][0]))
    check("one candidate left after dedupe", len(proposals) == 1, f"variants: {len(proposals)}")
    check("copy multiplicity is recorded", proposals[0].n_samples == 5,
          f"n_samples: {proposals[0].n_samples}")

    print("4b. A temperature not named by the policy is the registry default, not the run's")
    proposer, seen = make_proposer(["r = r.box(1)", "r = r.box(2)"], tmp)
    proposer.propose(gt_path, None, "PREFIX", k=2, step=1, seed=7)
    check("default 1.0", seen["kwargs"][0].get("temperature") == llm.DEFAULT_TEMPERATURE,
          str(seen["kwargs"][0]))
    check("sampling nucleus is passed", seen["kwargs"][0].get("top_p") == llm.DEFAULT_TOP_P,
          str(seen["kwargs"][0]))

    print("5. The GT point cloud is sampled once per part")
    samples = {"count": 0}
    original_sample = propose_mod.sample_gt_points

    def counting_sample(path, seed=None):
        samples["count"] += 1
        return original_sample(path, seed=seed)

    original_choose = propose_mod.choose_point
    propose_mod.sample_gt_points = counting_sample
    try:
        proposer, seen = make_proposer(["r = r.box(1)"], tmp)
        clouds = []

        def capture_point(gt_mesh_path, pred_mesh_path, point_seed, gt_points=None):
            # The signature repeats the real one: `point_seed` is required with no default,
            # so a forgotten seed must fail here too.
            clouds.append(gt_points)
            return "(1,2,3)", gt_points

        propose_mod.choose_point = capture_point
        for step in range(1, 5):
            proposer.propose(gt_path, None, "PREFIX", k=1, step=step, seed=7)

        check("one sampling for four steps", samples["count"] == 1, f"samplings: {samples['count']}")
        check("the cloud was passed to choose_point", all(c is not None for c in clouds), str([c is None for c in clouds]))
        check("it is the same cloud", all(c is clouds[0] for c in clouds))
        check("the cloud has 10k points", clouds[0].shape == (10_000, 3), str(clouds[0].shape))
    finally:
        # Both must be restored: section 6 checks the real `choose_point`, and with the stub
        # left in place it would pass vacuously.
        propose_mod.sample_gt_points = original_sample
        propose_mod.choose_point = original_choose

    print("6. choose_point returns the cloud and reuses the one it is given")
    points = original_sample(gt_path)
    point_a, back = propose_mod.choose_point(gt_path, None, point_seed=1, gt_points=points)
    check("the cloud is returned to the caller", back is points)
    check("the point is a string of the form (x,y,z)", point_a.startswith("(") and point_a.endswith(")"), point_a)
    check("coordinates are in the model scale", np.abs(points).max() <= 101.0, f"max {np.abs(points).max():.1f}")

    print("7. evaluate_codes does not execute identical codes twice")
    executed: list[list[str]] = []

    def fake_evaluate(tasks):
        executed.append([task.code for task in tasks])
        return [types.SimpleNamespace(
            success=True, error=None, mesh_path=f"/tmp/{task.task_id}.stl",
            metrics={"cd_runtime": 0.1}, timed_out=False, outcome="ok",
            worker_died=False, wall_sec=0.0, n_threads=1,
        ) for task in tasks]

    resources = Resources(
        propose_steps=lambda *a, **k: [], evaluate=fake_evaluate,
        ask_agent=lambda *a, **k: "", algo_rebuild=lambda *a, **k: [],
        optimize=lambda *a, **k: {}, render=None, difficulty=lambda: {},
        budget=Budget(), work_dir=tmp, journal=None, config={},
    )
    codes = ["A", "B", "A", "C", "B"]
    results = resources.evaluate_codes(codes, gt_mesh_path=gt_path, name_prefix="step001")

    check("only unique codes were executed", executed[0] == ["A", "B", "C"], str(executed[0]))
    check("a result is returned for every code", len(results) == len(codes), f"{len(results)} vs {len(codes)}")
    check("duplicates got their twin's result",
          results[0] is results[2] and results[1] is results[4],
          str([r.mesh_path for r in results]))
    check("order is preserved",
          [r.mesh_path for r in results] == [
              "/tmp/step001_0.stl", "/tmp/step001_1.stl", "/tmp/step001_0.stl",
              "/tmp/step001_2.stl", "/tmp/step001_1.stl",
          ],
          str([r.mesh_path for r in results]))

    executed.clear()
    resources.evaluate_codes(["X", "Y"], gt_mesh_path=gt_path, name_prefix="step002")
    check("without duplicates nothing changes", executed[0] == ["X", "Y"], str(executed[0]))

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Step generator is fine.")


if __name__ == "__main__":
    main()
