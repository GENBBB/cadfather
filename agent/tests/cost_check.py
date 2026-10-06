#!/usr/bin/env python3
"""Cost counters cover both branches, and comparison configs are comparable.

The baseline and the agent branch are compared by cost, and the comparison is worth
exactly as much as the counters: a call kind nobody counts makes its channel free,
and a channel that is counted but always zero looks free, which is worse.

What is checked here:

1. every call kind from `CALL_KINDS` increases the budget **and** appears in the
   technical metrics, one call per kind, through the real `build_resources` and not
   around it;
2. the visual channel is counted separately from the text one (the `cost_i` weights
   rest on this);
3. a `dialogue_lean` run spends the generator, execution and the assistant, and
   tools outside the set are not spent;
4. the per-figure record and the summary carry the cost into the report;
5. **the `experiment.tools` set**: a tool not in the set answers with a refusal,
   spends no budget and does not appear in `tech.json`; a listed one is counted
   exactly once;
6. **the stop threshold is in the scale of the chosen objective.** A threshold from
   the CD scale with `objective: iou` would stop the rollout at the first step; this
   is checked both on invented configs and on the one in `configs/`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "tests"))

import harness_e2e  # noqa: E402  — the same stubs as in the end-to-end test
from cad_agent.capabilities import det as det_mod  # noqa: E402
from cad_agent.capabilities import optimize as optimize_mod  # noqa: E402
from cad_agent.harness.config import run_tools  # noqa: E402
from cad_agent.capabilities.execute import EvalResult, EvalTask, Executor  # noqa: E402
from cad_agent.capabilities.propose import StepProposal  # noqa: E402
from cad_agent.capabilities.resources import build_resources  # noqa: E402
from cad_agent.harness import run_eval  # noqa: E402
from cad_agent.harness.budget import CALL_KINDS, Budget  # noqa: E402
from cad_agent.harness.journal import FigureJournal  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --- capability stubs ----------------------------------------------------
class FakeProposer:
    def propose(self, gt_mesh_path, pred_mesh_path, prev_code, k, seed=None, step=None, tag="",
                temperature=None, top_p=None):
        from cad_agent.capabilities import llm as llm_mod

        proposals = []
        for index in range(k):
            call = llm_mod.LLMCall(
                text="r = r.box(1)", model="fake-generation", has_image=True,
                latency_sec=0.5, prompt_tokens=100, completion_tokens=10, attempts=1,
            )
            self.journal.record_llm("vlm", call)
            proposals.append(StepProposal(index=index, raw_text="r = r.box(1)", step_code="r = r.box(1)",
                                          full_code=f"{prev_code}\nr = r.box({index})", point="(0,0,0)"))
        return proposals


class FakeExecutor(Executor):
    # The last set of tasks, to check what reached `EvalTask` (for example, the metrics
    # order via `needs`).
    last_tasks: list[EvalTask] = []

    def evaluate(self, tasks: list[EvalTask]) -> list[EvalResult]:
        FakeExecutor.last_tasks = list(tasks)
        return [
            EvalResult(task_id=task.task_id, success=True, mesh_path="/tmp/x.stl",
                       metrics={"cd_runtime": 0.5}, wall_sec=0.25)
            for task in tasks
        ]

    def warm_gt(self, path):  # pragma: no cover - not called in this test
        pass

    def close(self):
        pass


class FakeAgentClient:
    """Decision-agent client: always answers, counts calls."""

    def __init__(self):
        self.calls = 0


def fake_call_vision(client, model_name, text, image=None, generation_kwargs=None,
                     system_prompt=None, max_attempts=3):
    from cad_agent.capabilities import llm as llm_mod

    client.calls += 1
    return llm_mod.LLMCall(
        text="A", model=model_name, has_image=image is not None, latency_sec=0.1,
        prompt_tokens=200 if image is not None else 50,
        completion_tokens=5, attempts=1,
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cost_check_"))

    print("1. Every call kind increases the budget and reaches tech.json")
    from cad_agent.capabilities import llm as llm_mod

    llm_mod.call_vision = fake_call_vision
    det_mod.warm_gt_frame = lambda gt, cache: None
    # The stub returns `n_threads` exactly like the real fork: it reads its own
    # /proc/self/stat before returning. Without this field in the stub, det thread
    # accounting would silently go unchecked, a hole already found twice on the execution side.
    det_mod.det_rebuild_isolated = lambda gt, pred, cache, timeout=None, max_faces=None: {
        # The pool, not "how many were asked for": the slice by cursor is made by
        # `algo_rebuild`, and a stub returning exactly k would hide that from the check.
        "success": True, "ops": ["r = r.box(1)"] * 8, "n_threads": 7
    }
    # `wall_sec` in the stub is not decoration: the real `optimize_params` always returns
    # it, and call latency rests on it. Without it the stub would lag behind the interface
    # and the check below would be green idly.
    optimize_mod.optimize_params = lambda code, gt_mesh_path, work_dir, **kwargs: {
        "success": True, "code": code + "\n# optimized", "wall_sec": 0.25
    }
    journal = FigureJournal(figure_dir=tmp / "figure", level="metrics", profile=True)
    budget = Budget()
    proposer = FakeProposer()
    proposer.journal = journal
    agent_client = FakeAgentClient()

    resources = build_resources(
        figure_id="test/fig",
        gt_mesh_path=Path("/tmp/gt.stl"),
        work_dir=tmp / "figure",
        executor=FakeExecutor(),
        renderer=None,
        proposer=proposer,
        agent_client=agent_client,
        agent_model="fake-assistant",
        budget=budget,
        # The optimizer is listed here on purpose: the section checks that the call reaches the
        # counters. It is not in the default set, and without this line the section would check
        # the set, not the accounting.
        config={"execution": {"det_timeout_sec": 60},
                "tools": ["stepwise", "det_cold", "det_warm", "repair", "optimize"]},
        journal=journal,
    )

    resources.propose_steps(None, "PREFIX", 2, step=1)
    resources.evaluate([EvalTask(task_id="t0", code="r = r.box(1)")])
    resources.ask_agent("pick a candidate")
    resources.ask_agent("pick a candidate", image="IMAGE")
    # A repair request is its own line item (`agent_repair`), not the same as the scaffold's
    # question. Without this line the section would not check the new counter at all.
    resources.ask_agent("fix the code", purpose="repair")
    resources.algo_rebuild(None, 3)
    resources.optimize("r = r.box(1)")

    counts = budget.to_dict()["calls"]
    for kind in CALL_KINDS:
        check(f"the budget counts {kind}", counts.get(kind, 0) > 0, f"{kind}={counts.get(kind)}")

    tech = journal.tech_metrics()
    for kind in ("vlm", "agent_text", "agent_visual", "det", "opt", "exec"):
        check(f"tech.json counts {kind}", tech["calls"].get(kind, 0) > 0, f"{kind}={tech['calls'].get(kind)}")

    print("\n2. The visual channel is counted separately from the text one")
    tokens = tech["tokens"]
    check("text tokens are present", tokens.get("prompt", 0) > 0, str(tokens))
    check("visual tokens are present", tokens.get("prompt_visual", 0) > 0, str(tokens))
    check("channels are not mixed", tokens.get("prompt") != tokens.get("prompt_visual"), str(tokens))
    # The VALUE is checked, not the presence of the key: `record_call` creates the key even
    # with zero, so a presence check was green exactly where latency was not recorded at all,
    # for det and opt.
    for kind in ("vlm", "agent_text", "agent_visual", "exec", "det", "opt"):
        check(f"latency of {kind} is nonzero",
              float(tech["latency_sec"].get(kind, 0.0)) > 0.0,
              f"{kind}={tech['latency_sec'].get(kind)}")
    check("the opt call is counted exactly once", tech["calls"].get("opt") == 1,
          f"opt={tech['calls'].get('opt')}")
    check("stages were profiled", {"agent", "det", "optimize", "execute"} <= set(tech["stages_sec"]),
          str(sorted(tech["stages_sec"])))

    fork_threads = tech.get("fork_threads") or {}
    check("det fork threads are measured", bool(fork_threads.get("det", {}).get("n")), str(fork_threads))
    check("det fork threads are not mixed with execution",
          fork_threads.get("det", {}).get("max") == 7, str(fork_threads))

    print("\n3. No dead counters in the cost")
    cost = budget.to_dict()
    empty = [key for key, value in cost.items() if isinstance(value, dict) and not any(value.values())]
    check("all cost dictionaries are filled", not empty, f"empty: {empty}")
    check("no tokens in the budget (they are in the journal)", "tokens" not in cost, str(sorted(cost)))

    print("\n4. A `dialogue_lean` run: generator, execution and assistant are spent, nothing else")
    harness_e2e.install_fakes()
    folder = harness_e2e.make_dataset(tmp, n=3)
    config = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1)
    result = run_eval.run_experiment(config=config, run_dir=tmp / "run_baseline")

    total = result["summary"]["cost_calls_total"]
    check("vlm is counted", total.get("vlm", 0) > 0, f"vlm={total.get('vlm')}")
    check("exec is counted", total.get("exec", 0) > 0, f"exec={total.get('exec')}")
    # The assistant is asked on every turn and with images, so the channel is visual.
    check("assistant is counted",
          total.get("agent_visual", 0) + total.get("agent_text", 0) > 0,
          f"agent_visual={total.get('agent_visual')}, agent_text={total.get('agent_text')}")
    # `base_config` gives a set of one `stepwise`: det and the optimizer are outside the set
    # and cannot be spent.
    check("opt was not spent", total.get("opt", 0) == 0, f"opt={total.get('opt')}")
    check("det was not spent", total.get("det", 0) == 0, f"det={total.get('det')}")

    print("\n5. Cost reaches the per-part record and the summary")
    record = result["per_figure"][0]
    check("the per-figure record has calls", bool((record.get("cost") or {}).get("calls")), str(record.get("cost")))
    check("the summary has the cost per part", bool(result["summary"].get("cost_calls_per_figure")))
    check("tokens in the summary are split by channel",
          bool((result["summary"]["tech"].get("tokens") or {}).get("prompt_visual")),
          str(result["summary"]["tech"].get("tokens")))

    import collections
    import collections.abc

    for name in ("Hashable", "Mapping", "MutableMapping", "Sequence"):
        setattr(collections, name, getattr(collections.abc, name))
    import yaml

    configs_dir = AGENT_ROOT / "configs"

    print("\n6. Config validation does not need an installed model client")
    # The trap this check was written for: the policy registry is imported by the config
    # VALIDATOR, and a policy that pulls in `capabilities/llm.py` pulls in `openai`. Tests
    # substitute a stub for it in `sys.modules` and stay green; `preflight` installs no stub
    # and fails on the config. Hence the check runs in a separate process where importing
    # `openai` is forbidden.
    import subprocess

    guard = (
        "import builtins, sys, yaml, collections, collections.abc\n"
        "for _n in ('Hashable','Mapping','MutableMapping','Sequence'):\n"
        "    setattr(collections, _n, getattr(collections.abc, _n))\n"
        "_real = builtins.__import__\n"
        "def _guard(name, *a, **k):\n"
        "    if name == 'openai' or name.startswith('openai.'):\n"
        "        raise ImportError('openai is intentionally unavailable')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _guard\n"
        "sys.path.insert(0, %r)\n"
        "from run_experiment import build_run_config\n"
        "from cad_agent.harness.config import validate_run_config\n"
        "import pathlib\n"
        "for path in sorted(pathlib.Path(%r).rglob('*.yaml')):\n"
        "    validate_run_config(build_run_config(yaml.safe_load(path.read_text()), None, None))\n"
        "print('ok')\n"
    ) % (str(AGENT_ROOT), str(AGENT_ROOT / "configs"))
    proc = subprocess.run([sys.executable, "-c", guard], capture_output=True, text=True)
    check("all configs validate without openai", proc.returncode == 0 and "ok" in proc.stdout,
          (proc.stderr or proc.stdout).strip().splitlines()[-1:] and
          (proc.stderr or proc.stdout).strip().splitlines()[-1])

    print("\n7. The optimizer switch really switches, it does not pretend")
    from cad_agent.harness.journal import FigureJournal as _Journal

    def optimize_probe(enabled: bool) -> tuple[dict, dict, dict]:
        """Call `res.optimize` with the switch set as given."""
        where = tmp / f"switch_{int(enabled)}"
        probe_journal = _Journal(figure_dir=where, level="metrics", profile=False)
        probe_budget = Budget()
        probe = build_resources(
            figure_id="test/switch",
            gt_mesh_path=Path("/tmp/gt.stl"),
            work_dir=where,
            executor=FakeExecutor(),
            renderer=None,
            proposer=None,
            budget=probe_budget,
            config={"tools": ["stepwise", "optimize"] if enabled else ["stepwise"]},
            journal=probe_journal,
        )
        answer = probe.optimize("r = r.box(1)")
        return answer, probe_budget.to_dict()["calls"], probe_journal.tech_metrics()

    off_answer, off_calls, off_tech = optimize_probe(False)
    on_answer, on_calls, on_tech = optimize_probe(True)

    check("a disabled one answers with a refusal, it does not crash",
          off_answer.get("success") is False, str(off_answer))
    check("a disabled one does not spend budget", off_calls.get("opt", 0) == 0, f"opt={off_calls.get('opt', 0)}")
    check("a disabled one does not get into tech.json", off_tech["calls"].get("opt", 0) == 0,
          f"opt={off_tech['calls'].get('opt', 0)}")
    check("an enabled one worked", on_answer.get("success") is True, str(on_answer))
    check("an enabled one spent exactly one call", on_calls.get("opt", 0) == 1, f"opt={on_calls.get('opt', 0)}")
    check("an enabled one got into tech.json", on_tech["calls"].get("opt", 0) == 1,
          f"opt={on_tech['calls'].get('opt', 0)}")

    print("\n8. Ordering observable metrics is the harness's job, not the wiring's")

    def needs_for(metrics_config: dict) -> tuple:
        """What ends up in `EvalTask.needs` for a given `experiment.metrics`."""
        probe = build_resources(
            figure_id="test/needs", gt_mesh_path=Path("/tmp/gt.stl"),
            work_dir=tmp / "needs", executor=FakeExecutor(), renderer=None, proposer=None,
            budget=Budget(), config={"metrics": metrics_config}, journal=None,
        )
        probe.evaluate_codes(codes=["r = r.box(1)"], gt_mesh_path="/tmp/gt.stl",
                             name_prefix="step001", needs=("iou",))
        return tuple(FakeExecutor.last_tasks[0].needs)

    check("without the key the scaffold orders only its own",
          needs_for({}) == ("iou",), str(needs_for({})))
    check("metrics.cd: true adds CD to the order",
          "cd" in needs_for({"cd": True}), str(needs_for({"cd": True})))
    check("the wiring objective is not lost",
          "iou" in needs_for({"cd": True}), str(needs_for({"cd": True})))

    print("\n9. The stop threshold lives in the objective scale")
    from cad_agent.harness.config import validate_run_config, ConfigError
    from cad_agent.capabilities import objective as objective_mod
    from cad_agent.scaffold.policies import POLICIES, POLICY_NAMES, build as build_policy

    # The threshold no longer comes from the config: the objective and the threshold are
    # policy knobs, and the scaffold section carries only the policy name. So the old trap
    # ("a CD-scale threshold with objective=iou") is unreachable from the config: the
    # validator rejects the key itself.
    def rejects_key(key, value):
        cfg = {
            "details": [{"a": "/tmp"}], "n_workers": 1,
            "server": {"generation_base_url": "http://x/v1", "generation_served_model_name": "g"},
            "scaffold": {"kind": "policy", "policy": "dialogue_lean", key: value},
        }
        try:
            validate_run_config(cfg)
            return False
        except ConfigError:
            return True

    check("the threshold in the config is rejected together with the key", rejects_key("success_score", 0.0001))
    check("the objective in the config is rejected together with the key", rejects_key("objective", "cd"))

    # What holds the rule now: the default threshold is taken FROM THE OBJECTIVE, so it
    # physically cannot land in a foreign scale. Checked on all objectives: an objective that
    # forgot its threshold would return a foreign one.
    for name in objective_mod.OBJECTIVE_NAMES:
        obj = objective_mod.get_objective(name)
        default = obj.success_threshold(None)
        in_scale = (0.0 < default <= 1.0) if obj.higher_is_better else (0.0 < default < 0.5)
        check(f"objective threshold {name} is given in its own scale", in_scale, str(default))

    # And what makes the rule hold at all: NO registry policy sets a threshold explicitly.
    # An explicit threshold is written in the scale of its objective and is not converted on
    # fallback (`iou -> cd`); this is a known caveat, harmless exactly as long as everything
    # here is None. The first number here makes it live.
    explicit = sorted(
        name for name in POLICY_NAMES
        if getattr(build_policy(name), "SUCCESS_SCORE", None) is not None
    )
    check("no policy sets the threshold explicitly", not explicit, str(explicit))

    # The same on configs from disk: they are what drifts from the objective.
    shipped_configs = sorted(configs_dir.rglob("*.yaml"))
    check("configs on disk are found", bool(shipped_configs), str(configs_dir))
    # The flat config is built by THE SAME code as in a run (`build_run_config`), not by hand
    # here. The difference is not cosmetic: some run conditions are derived on the way, for
    # example `server.assistant_enabled` from the `launch` section, and a hand-built dict
    # would check a config that never exists.
    from run_experiment import build_run_config

    for path in shipped_configs:
        shipped = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            validate_run_config(build_run_config(shipped, None, None))
            ok, why = True, ""
        except (ConfigError, ValueError) as exc:
            ok, why = False, str(exc)
        check(f"{path.name} passes the validator", ok, why)

        # The assistant must be up wherever the scaffold calls it: an agent branch without it
        # would degrade into a baseline with an agent refusal on every part. The converse is not
        # an error but a waste: a baseline with the assistant up holds two GPUs for nothing, and
        # saying so aloud is enough.
        run_config = build_run_config(shipped, None, None)
        # Whether the scaffold calls the assistant is now known by the policy itself: the knobs
        # moved into the policy, and there is nothing to judge by the scaffold's appearance, since
        # it looks the same for all.
        policy_name = (run_config.get("scaffold") or {}).get("policy", "")
        wants_agent = bool(getattr(build_policy(policy_name), "needs_assistant", False)) if policy_name else False
        enabled = (run_config.get("server") or {}).get("assistant_enabled")
        # The endpoint's image ceiling must REACH the run config, not stay a flag read only by
        # `run_system.sh`. The same class as "key accepted != key works": the policy cuts the
        # number of panels by this number, and a silent key means not "no ceiling" but "cutting
        # by the wrong number". Compared with the source YAML, not with a constant.
        limit = (run_config.get("server") or {}).get("assistant_image_limit")
        declared = (((shipped.get("launch") or {}).get("servers") or {})
                    .get("assistant") or {}).get("args") or {}
        raw = declared.get("limit-mm-per-prompt") if isinstance(declared, dict) else None
        if raw is not None:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            want = int(parsed["image"])
            check(f"{path.name}: the image cap reached the run config",
                  limit == want, f"{limit} vs {want} in launch")
        else:
            check(f"{path.name}: without the flag the cap is not invented",
                  limit is None, str(limit))

        if wants_agent:
            check(f"{path.name}: the assistant is up for the agent branch", enabled is True, f"enabled={enabled}")
        elif enabled:
            print(f"  ~     {path.name}: the scaffold brings up the assistant — two GPUs and minutes wasted "
                  f"(launch.servers.assistant.enabled: false)")


    print("\n10. Config caps reach the part counter")
    # The same class of defect as one already known: a section passes validation and silently
    # never reaches the runtime. Here the ceiling is set ONLY by the config and asked of the
    # run: if `Budget` were built bypassing `experiment.budget`, the part would simply run
    # to the end.
    import harness_e2e as e2e
    e2e.install_fakes()
    capped_dir = Path(tempfile.mkdtemp(prefix="cost_caps_"))
    folder = e2e.make_dataset(capped_dir, n=3)
    capped = e2e.base_config(folder, backend="serial_fork", n_workers=1)
    capped["budget"] = {"vlm": 1}
    result = run_eval.run_experiment(config=capped, run_dir=capped_dir / "run_capped")
    calls = [(r.get("cost") or {}).get("calls", {}).get("vlm", 0) for r in result["per_figure"]]
    # The ceiling itself is SOFT by one call: `Budget.spend` counts first and compares after
    # (`counts > limit`), so a call that breaks the ceiling gets to happen. But a PLANNING
    # policy does not let it come to that: the greedy policy asks for the remainder
    # (`affordable_n`) and does not order what it cannot afford, so on its run there is no
    # overshoot and `budget.vlm: N` means exactly N calls. Confusing N and N+1 when comparing
    # branch costs is easy, so it is recorded as a number, not an assumption.
    #
    # The HARNESS guarantee (the ceiling cannot be bypassed even without planning) is checked
    # separately and on a non-planning policy, in `search_check`.
    check("the vlm cap from the config limited every part",
          calls and all(c <= 2 for c in calls), str(calls))
    check("a planning policy does not break the cap at all",
          calls and max(calls) == 1, str(calls))

    # And the converse: without a ceiling the same parts spend more. Without this half the
    # check would be green even on a run that simply had no time to spend anything, i.e. it
    # would measure the stub, not the ceiling.
    free = e2e.base_config(folder, backend="serial_fork", n_workers=1)
    free_result = run_eval.run_experiment(config=free, run_dir=capped_dir / "run_free")
    free_calls = [(r.get("cost") or {}).get("calls", {}).get("vlm", 0) for r in free_result["per_figure"]]
    check("without a cap the same parts spend more",
          max(free_calls) > max(calls), f"without a cap {free_calls}, with a cap {calls}")

    # The ceilings of measurement configs must not be empty: the set was introduced exactly
    # so that a mutant could not eat the run, and an unfilled section here is a return to
    # "consumption is bounded by nothing".
    measured_configs = sorted(configs_dir.rglob("*.yaml"))
    for path in measured_configs:
        shipped = yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]
        budget_section = shipped.get("budget") or {}
        empty = [k for k, v in budget_section.items() if v is None]
        check(f"{path.name}: call caps are set", bool(budget_section) and not empty,
              f"empty: {empty}" if empty else ("no section" if not budget_section else ""))
        check(f"{path.name}: the part wall-time cap is set",
              bool(shipped.get("figure_wall_sec")), str(shipped.get("figure_wall_sec")))

    print("\n11. Price list: price shape and honesty of `measured`")
    from cad_agent.harness.tools import price_list

    prices = price_list()

    # Shape. For `stepwise`, `n` is samples: the request is paid once, the samples and their
    # executions each. If `n` also moved the constant part, the price would again penalize
    # the most productive knob, which is what the reshaping was for.
    step = prices["stepwise"]
    at1, at3 = step.at(1), step.at(3)
    # Via `isclose`, not `==`: the weights are fractional, and 2.2 - 1.4 in double gives
    # 0.7999999999999998. An exact-equality check would fail on number representation, not
    # on the price.
    import math
    check("stepwise's constant part does not grow with n",
          step.calls.get("vlm") == 1.0
          and math.isclose(at3["vlm"] - at1["vlm"], 2 * step.per_n["vlm"]),
          f"n=1 {at1}, n=3 {at3}")
    check("stepwise executions grow exactly linearly with n",
          at3["exec"] == 3 * at1["exec"], f"{at1['exec']} vs {at3['exec']}")
    check("a sample is cheaper than a request — otherwise the price list is inverted again",
          step.per_n["vlm"] < step.calls["vlm"], f"{step.per_n['vlm']} vs {step.calls['vlm']}")

    # For det, `n` is the rank depth: the call is already paid for, and the next operation
    # costs only its own execution. There is nothing to make it costlier than a call.
    det = prices["det_cold"]
    check("a det call does not get pricier with n",
          det.at(1)["det"] == det.at(8)["det"] == det.calls["det"], str(det.at(8)))
    check("a det call is pricier than a stepwise call — the main fact of the measurement",
          det.calls["det"] > 10 * step.calls["vlm"], f"{det.calls['det']} vs {step.calls['vlm']}")
    check("both det branches cost the same (a cursor repeat does not make it cheaper)",
          prices["det_cold"] == prices["det_warm"])

    # `measured` is a property of the tool. Lying here would pass a guess off as a
    # measurement in `task_description.txt`, which the mutator reads.
    measured = {name for name, cost in prices.items() if cost.measured}
    check("exactly the tools that were run are marked as measured",
          measured == {"stepwise", "det_cold", "det_warm", "optimize", "repair"},
          str(sorted(measured)))
    # The prices of `repair` and `optimize` depend on input size: one number cannot express
    # them. We check not the values (they are re-measured) but that the second figure exists
    # and that it really is the upper one: a price whose "up to" is below the typical is a
    # typo, and the mutator would plan consumption by it.
    for name in ("optimize", "repair"):
        cost = prices[name]
        check(f"{name} carries a second price — on a large input", cost.wall_sec_hi > 0,
              f"wall_sec_hi={cost.wall_sec_hi}")
        check(f"for {name} upper price is not below the typical one",
              cost.wall_sec_hi >= cost.wall_at(1),
              f"{cost.wall_sec_hi} vs {cost.wall_at(1)}")
    # For repair each sample is its own request to the assistant, unlike `stepwise`, where K
    # variants travel in one. So the price must grow with `n` linearly, with no free
    # constant part.
    rep = prices["repair"]
    # A line item of its own (`agent_repair`), not shared with the scaffold's decisions: they
    # ask the same endpoint but have different ceilings, and by the price list the policy must
    # see exactly the item it will spend.
    check("repair gets pricier linearly with n — each sample is a separate request",
          rep.at(3)["agent_repair"] == 3 * rep.at(1)["agent_repair"],
          f"n=1 {rep.at(1)}, n=3 {rep.at(3)}")
    check("and it charges its own item, not the wiring decisions item",
          "agent_text" not in rep.at(1), str(rep.at(1)))

    print("\n12. The dialogue with the assistant is saved to disk in full")
    # The section lives here because this is the only place where the real `ask_agent` is
    # called through `build_resources`: the end-to-end run (`harness_e2e`) replaces
    # `call_vision` itself and does not check dialogue files, and the policy smoke replaces
    # the resources wholesale.
    #
    # BOTH shapes of a thinking model's answer are checked. They come from the same server
    # depending on whether it was started with `--reasoning-parser`, and how the server is
    # started is not part of the run config: if reasoning is kept in only one shape, half
    # the runs lose it silently, and the difference reads as a difference between models.
    from PIL import Image as PIL_Image

    def thinking_call_vision(client, model_name, text, image=None, generation_kwargs=None,
                             system_prompt=None, max_attempts=3):
        client.calls += 1
        parsed = image is not None
        if "turn 3" in text:
            # An answer cut off by the ceiling: it arrived, meaningful, without the format, and by
            # its text it is indistinguishable from an answer not in the format. Only the endpoint's
            # `finish_reason` tells it apart.
            return llm_mod.LLMCall(
                text="we should extend the strongest branch, but first let me",
                model=model_name, latency_sec=0.1, prompt_tokens=50, completion_tokens=1024,
                attempts=1, finish_reason="length", finish_reasons=["length"],
            )
        return llm_mod.LLMCall(
            # A server with a parser returns the visible text and the reasoning separately; without
            # a parser, as one piece with a `<think>` block.
            text="ACTION: tool=stepwise candidate=c1 n=1" if parsed
            else "<think>c1 looks closer</think>\nACTION: tool=stepwise candidate=c1 n=1",
            reasoning="c1 looks closer" if parsed else "",
            model=model_name, has_image=image is not None, latency_sec=0.1,
            prompt_tokens=50, completion_tokens=5, attempts=1,
            finish_reason="stop", finish_reasons=["stop"],
        )

    llm_mod.call_vision = thinking_call_vision
    full_journal = FigureJournal(figure_dir=tmp / "dialogue", level="full", profile=False)
    dialogue_resources = build_resources(
        figure_id="test/dialogue",
        gt_mesh_path=Path("/tmp/gt.stl"),
        work_dir=tmp / "dialogue",
        executor=FakeExecutor(),
        renderer=None,
        proposer=FakeProposer(),
        agent_client=FakeAgentClient(),
        agent_model="fake-assistant",
        budget=Budget(),
        config={"tools": ["stepwise", "repair"]},
        journal=full_journal,
    )
    dialogue_resources.ask_agent("turn 1: what next?")
    dialogue_resources.ask_agent("turn 2: what next?", image=PIL_Image.new("RGB", (8, 8)))

    saved = tmp / "dialogue" / "agent"
    names = sorted(path.name for path in saved.iterdir()) if saved.exists() else []
    check("the question to the assistant is saved", "call_001_prompt.txt" in names, str(names))
    check("and the answer raw, together with the reasoning block",
          "<think>" in (saved / "call_001_answer.txt").read_text(encoding="utf-8"),
          str(names))
    check("the reasoning is saved even without a parser on the server",
          (saved / "call_001_reasoning.txt").read_text(encoding="utf-8").strip() == "c1 looks closer")
    check("and with the parser, when it is no longer in the answer text",
          (saved / "call_002_reasoning.txt").read_text(encoding="utf-8").strip() == "c1 looks closer"
          and "<think>" not in (saved / "call_002_answer.txt").read_text(encoding="utf-8"),
          str(names))
    check("the image given to the assistant is saved",
          any(name.startswith("call_002_input_") for name in names), str(names))

    metrics_journal = FigureJournal(figure_dir=tmp / "dialogue_metrics", level="metrics")
    metrics_journal.save_agent_call(index=1, prompt="q", answer="a", reasoning="r")
    check("at the metrics level the dialogue is not written to disk",
          not (tmp / "dialogue_metrics" / "agent").exists())

    # A cutoff at the ceiling: a successful call with an INCOMPLETE answer. By text it is
    # indistinguishable from an answer not in the format, so the flag must travel from the
    # endpoint to the policy (`state.answer_truncated`) and to the report; otherwise hitting
    # the ceiling reads as "the assistant answers unclearly", and the prompt gets fixed
    # instead of the ceiling.
    check("after a full answer the truncation flag is cleared", not dialogue_resources.answer_truncated())
    dialogue_resources.ask_agent("turn 3: what next?")
    check("a cut-off answer is visible to the policy", dialogue_resources.answer_truncated())
    dialogue_resources.ask_agent("turn 4: what next?")
    check("and the flag belongs to the last answer, it does not stick",
          not dialogue_resources.answer_truncated())
    events = [json.loads(line) for line
              in (tmp / "dialogue" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    asked = [event for event in events if event.get("kind") == "ask_agent"]
    check("the cut-off is recorded in the question event",
          [bool(event.get("truncated")) for event in asked] == [False, False, True, False],
          str([(event.get("finish_reason"), event.get("truncated")) for event in asked]))
    check("and counted in tech.json per channel",
          (full_journal.tech_metrics().get("truncated") or {}).get("agent_text") == 1,
          str(full_journal.tech_metrics().get("truncated")))

    # Two different consumers call the assistant, and a shared counter hides them: for the
    # scaffold's decisions the cutoff share is moderate, while for repair it was nearly total,
    # and by the merged number a tool that never worked looked healthy. The CEILING was also
    # shared, so they took calls from each other.
    before = dict(dialogue_resources.budget.counts)
    dialogue_resources.ask_agent("fix this code", purpose="repair")
    check("repair charges its own item, not the decisions item",
          dialogue_resources.budget.counts["agent_repair"] == before["agent_repair"] + 1
          and dialogue_resources.budget.counts["agent_text"] == before["agent_text"],
          f"repair {before['agent_repair']} -> "
          f"{dialogue_resources.budget.counts['agent_repair']}, "
          f"text {before['agent_text']} -> {dialogue_resources.budget.counts['agent_text']}")
    tech = full_journal.tech_metrics()
    check("and goes to its own channel in tech.json",
          tech["calls"].get("agent_repair") == 1, str(tech["calls"]))
    # Tokens per channel: the "prompt/answer" split does not answer "how long is the answer
    # for THIS consumer", and that is what showed repair answers running very long.
    by_call = tech.get("tokens_by_call") or {}
    check("tokens are split per channel",
          set(by_call) >= {"agent_text", "agent_repair"}, str(sorted(by_call)))

    # Function calling (`policies/dialogue_lean.py`): the functions and history must reach
    # the request, parsed calls must reach the policy, and a question without them must go
    # out exactly as before (the `call_vision` stubs repeat the old signature).
    seen_requests: list[dict] = []

    def tools_call_vision(client, model_name, text, image=None, generation_kwargs=None,
                          system_prompt=None, max_attempts=3, **extra):
        seen_requests.append({"kwargs": dict(generation_kwargs or {}), **extra})
        calls = ([{"id": "call_1", "name": "act",
                   "arguments": '{"tool": "stepwise", "candidate": "c1", "n": 2}'}]
                 if (generation_kwargs or {}).get("tools") else [])
        return llm_mod.LLMCall(text="extend c1", model=model_name, latency_sec=0.1,
                               prompt_tokens=50, completion_tokens=5, attempts=1,
                               finish_reason="tool_calls" if calls else "stop",
                               finish_reasons=["stop"], tool_calls=calls, tool_calls_all=[calls])

    llm_mod.call_vision = tools_call_vision
    specs = [{"type": "function", "function": {"name": "act", "parameters": {}}}]
    history = [{"role": "system", "content": "intro"}]
    dialogue_resources.ask_agent("what next?", tools=specs, tool_choice="auto", history=history)
    check("functions and tool_choice reached the request",
          seen_requests[-1]["kwargs"].get("tools") == specs
          and seen_requests[-1]["kwargs"].get("tool_choice") == "auto", str(seen_requests[-1]))
    check("history reached the request", seen_requests[-1].get("history") == history,
          str(seen_requests[-1]))
    check("parsed calls are visible to the policy",
          [item["name"] for item in dialogue_resources.answer_tool_calls()] == ["act"],
          str(dialogue_resources.answer_tool_calls()))
    dialogue_resources.ask_agent("what next?")
    check("a question without functions goes as before: no tools, no history",
          "tools" not in seen_requests[-1]["kwargs"] and "history" not in seen_requests[-1],
          str(seen_requests[-1]))
    check("and the calls of the previous answer do not stick", dialogue_resources.answer_tool_calls() == [])
    events = [json.loads(line) for line
              in (tmp / "dialogue" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    asked = [event for event in events if event.get("kind") == "ask_agent"][-2:]
    check("the question event counts calls and history, and writes null without functions",
          [(e.get("tool_calls"), e.get("history")) for e in asked] == [(1, 1), (None, None)],
          str([(e.get("tool_calls"), e.get("history")) for e in asked]))
    last_answer = sorted(saved.glob("call_*_answer.txt"))[-2].read_text(encoding="utf-8")
    check("calls are saved in the answer file", "<tool_call act>" in last_answer, last_answer)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Cost counters and comparison configs are fine.")


def _diff(left, right, prefix: str = "") -> list[str]:
    """Paths where two nested structures differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff(left[key], right[key], path))
        return paths
    return [] if left == right else [prefix]


if __name__ == "__main__":
    main()
