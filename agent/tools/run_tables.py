#!/usr/bin/env python3
"""Comparison tables for dialogue runs: quality, calls per part, per-tool breakdown.

Two steps, because collection runs on the node (raw run directories are not
practical to download) while tables are easier to build locally from small JSON files:

    # on the node, in the run environment (--cd needs trimesh and the cad_agent package)
    ./agent/tools/run_tables.py collect work_dirs/<run> --out /tmp/<run>.json --cd --workers 16
    # anywhere
    ./agent/tools/run_tables.py table /tmp/a.json /tmp/b.json --label DeepCAD --label Fusion360
    # where one run's IoU is lost (IoU bins, GMS, outcomes; --vs pairs by part)
    ./agent/tools/run_tables.py loss /tmp/a.json --vs /tmp/a_old.json

Without `--cd` collection uses only the standard library. Runs do not compute CD
(`metrics.cd: false`), so it is computed here from `best.stl` and GT with the same
code the harness would use: `trimesh.load_mesh` -> `metrics.normalize_for_metrics`
-> `metrics.compute_cd(n_points=...)`, the sum of mean squared distances.

Definitions are printed under the tables by `table`. One differs from the earlier
analysis on purpose: "raised the best" is computed from `harness_fitness` (the mean
of the AVAILABLE `iou` and `gms_norm`) rather than `zip(iou, gms_norm)`, which lost
candidates with `iou: null`.
"""


from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_trace_stats import _reason_kind  # noqa: E402

# `DialogueLeanPolicy.OUTCOMES` in which the assistant itself named the action, at once
# or on a re-ask. The others are the harness falling back to a default action.
# Runs from before re-asking have only `chosen`.
ASSISTANT_KINDS = {"chosen", "reasked", "reasked_unparsed", "retried"}
REASKED_KINDS = ASSISTANT_KINDS - {"chosen"}

TOOLS_ORDER = ("stepwise", "det_cold", "det_warm", "optimize", "repair")


# ---------------------------------------------------------------- collection

def _candidate_fitness(event: dict) -> list[float]:
    """`harness_fitness` of each candidate of an evaluation: the mean of the available halves by index."""
    ious = event.get("iou") or []
    gmss = event.get("gms_norm") or []
    out = []
    for i in range(max(len(ious), len(gmss))):
        halves = [v for v in (ious[i] if i < len(ious) else None,
                              gmss[i] if i < len(gmss) else None) if v is not None]
        if halves:
            out.append(sum(halves) / len(halves))
    return out


def _figure_events(path: str) -> dict:
    rec = {"turns": 0, "calls": 0, "assistant": 0, "reasked": 0, "rejected": 0,
           "new_calls": 0, "exec_calls": 0, "raised_calls": 0,
           "new_cands": 0, "exec_cands": 0,
           "reject_reasons": Counter(), "tools": {}}
    if not os.path.exists(path):
        rec["missing_events"] = True
        return rec

    best = None
    current = None

    def close(call: dict | None) -> None:
        nonlocal best
        if call is None:
            return
        tool = rec["tools"].setdefault(call["tool"], Counter())
        rec["new_calls"] += call["n"] > 0
        rec["exec_calls"] += call["ok"] > 0
        tool["exec"] += call["ok"] > 0
        if call["fit"]:
            top = max(call["fit"])
            # The first evaluated call of a part does not count: there is nothing to raise.
            if best is not None and top > best:
                rec["raised_calls"] += 1
                tool["raised"] += 1
            best = top if best is None else max(best, top)

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind = event.get("kind")
            if kind == "ask_agent":
                rec["turns"] += 1
            elif kind == "tool_call":
                close(current)
                name = event.get("tool") or "?"
                reason = _reason_kind(event.get("reason"))
                rec["calls"] += 1
                rec["assistant"] += reason in ASSISTANT_KINDS
                rec["reasked"] += reason in REASKED_KINDS
                rec["tools"].setdefault(name, Counter())["calls"] += 1
                current = {"tool": name, "n": 0, "ok": 0, "fit": []}
            elif kind == "evaluate" and current is not None:
                current["n"] += event.get("n") or 0
                current["ok"] += event.get("ok") or 0
                current["fit"].extend(_candidate_fitness(event))
                rec["new_cands"] += event.get("n") or 0
                rec["exec_cands"] += event.get("ok") or 0
            elif kind == "plan_rejected":
                rec["rejected"] += 1
                rec["tools"].setdefault(event.get("tool") or "?", Counter())["rejected"] += 1
                rec["reject_reasons"][event.get("reason") or ""] += 1
    close(current)
    return rec


def _cd_worker(job: tuple[str, str, str, tuple[int, ...]]) -> tuple[str, dict | None, str | None]:
    figure_id, gt_path, pred_path, points = job
    try:
        import trimesh
        from cad_agent.capabilities import metrics

        gt = trimesh.load_mesh(gt_path)
        pred = trimesh.load_mesh(pred_path)
        metrics.normalize_for_metrics(gt, pred)
        return figure_id, {str(n): metrics.compute_cd(gt, pred, n_points=n)[0] for n in points}, None
    except Exception as exc:  # noqa: BLE001 — a part without CD, not a failed collection
        return figure_id, None, f"{type(exc).__name__}: {exc}"


def is_valid(path) -> bool:
    """Return True if the mesh is watertight and has a non-zero volume.

    Definition of a valid GT: the whole mesh must be watertight, not merely "at
    least one closed body". The run instrument `metrics.is_valid_gt` applies the
    same rule to the mesh.
    """
    import trimesh

    mesh = trimesh.load(path)
    if not isinstance(mesh, trimesh.Trimesh):
        print(type(mesh))
        return False
    if not mesh.is_watertight:
        return False
    vol = abs(mesh.volume)
    if vol <= 0:
        return False
    return True


def _gt_valid_worker(job: tuple[str, str]) -> tuple[str, bool | None, str | None]:
    figure_id, gt_path = job
    try:
        return figure_id, is_valid(gt_path), None
    except Exception as exc:  # noqa: BLE001
        return figure_id, None, f"{type(exc).__name__}: {exc}"


def _init_cd_worker(agent_root: str) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, agent_root)


def _pool_map(worker, jobs: list, workers: int, agent_root: str, what: str) -> dict:
    # spawn, not fork: the parent has loaded nothing native by now, but this also
    # keeps workers from inheriting foreign state.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    out, errors = {}, {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_cd_worker, initargs=(agent_root,)) as pool:
        for done, (fid, value, error) in enumerate(pool.map(worker, jobs, chunksize=8), 1):
            if error:
                errors[fid] = error
            else:
                out[fid] = value
            if done % 500 == 0:
                print(f"  {what} {done}/{len(jobs)}", file=sys.stderr, flush=True)
    # Errors are shown at once: a wholesale failure (wrong `--agent-root`) would
    # otherwise look like "CD for 0 parts" in the table rather than a broken collection.
    for fid, error in list(errors.items())[:3]:
        print(f"  {what} error {fid}: {error}", file=sys.stderr)
    return {"values": out, "errors": errors}


def collect(root: str, cd: bool, points: tuple[int, ...], workers: int, agent_root: str,
            gt_valid: bool = False) -> dict:
    with open(os.path.join(root, "per_figure.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    figures = {}
    for row in rows:
        fid = row["figure_id"]
        contract = row.get("metrics") or {}
        rec = _figure_events(os.path.join(root, "figures", fid, "events.jsonl"))
        rec["reject_reasons"] = dict(rec["reject_reasons"])
        rec["tools"] = {tool: dict(c) for tool, c in rec["tools"].items()}
        rec.update({
            "failure": contract.get("failure"),
            "gt_watertight": contract.get("gt_watertight"),
            "iou": contract.get("iou"),
            "gms_norm": contract.get("gms_norm"),
            "score": row.get("score"),
            "stop_reason": row.get("stop_reason"),
            "done_by": row.get("done_by"),
            "iterations": row.get("iterations"),
            "wall_sec": row.get("wall_sec"),
        })
        figures[fid] = rec

    result = {"run": os.path.basename(os.path.normpath(root)), "figures": figures}
    if cd:
        jobs = []
        for row in rows:
            pred = row.get("mesh_path") or os.path.join(root, "figures", row["figure_id"], "best.stl")
            if row.get("gt_mesh_path") and os.path.exists(pred):
                jobs.append((row["figure_id"], row["gt_mesh_path"], pred, points))
        result["cd"] = _pool_map(_cd_worker, jobs, workers, agent_root, "CD")
        result["cd"]["points"] = list(points)
        result["cd"]["no_mesh"] = len(rows) - len(jobs)
    if gt_valid:
        jobs = [(row["figure_id"], row["gt_mesh_path"]) for row in rows if row.get("gt_mesh_path")]
        valid = _pool_map(_gt_valid_worker, jobs, workers, agent_root, "GT is_valid")
        for fid, figure in figures.items():
            figure["gt_valid"] = valid["values"].get(fid)
        result["gt_valid_errors"] = valid["errors"]
    return result


# ---------------------------------------------------------------- tables

def _mean(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(part, whole):
    return "—" if not whole else f"{100 * part / whole:.1f}%"


def _row(title, cells):
    return f"| {title} | " + " | ".join(cells) + " |"


def quality_table(runs, labels) -> list[str]:
    points = None
    for run in runs:
        if "cd" in run:
            points = run["cd"]["points"]
            break
    has_valid = any("gt_valid" in f for run in runs for f in run["figures"].values())
    head = ["IR", "mean IoU (watertight GT)"]
    if has_valid:
        head += ["GT `is_valid`", "mean IoU (GT `is_valid`)"]
    head += ["mean GMS", "mean score"]
    head += [f"median CD @{n}" for n in points or []]
    lines = ["| | parts | " + " | ".join(head) + " |", "|---" * (len(head) + 2) + "|"]
    for run, label in zip(runs, labels):
        figs = list(run["figures"].values())
        n = len(figs)
        ir = sum(1 for f in figs if f["failure"]) / n
        ious = [f["iou"] for f in figs if f["gt_watertight"] and f["iou"] is not None]
        cells = [str(n), f"{ir:.4f}", _f(_mean(ious))]
        if has_valid:
            valid = [f for f in figs if f.get("gt_valid")]
            cells += [f"{len(valid)} ({_pct(len(valid), n)})", _f(_mean(f["iou"] for f in valid))]
        cells += [_f(_mean(f["gms_norm"] for f in figs)), _f(_mean(f["score"] or 0.0 for f in figs))]
        for p in points or []:
            values = [v[str(p)] for v in run.get("cd", {}).get("values", {}).values()]
            cells.append(_f(st.median(values)) if values else "—")
        lines.append(_row(label, cells))
    return lines


def quality_notes(runs, labels) -> list[str]:
    lines = []
    for run, label in zip(runs, labels):
        figs = list(run["figures"].values())
        wt = [f for f in figs if f["gt_watertight"]]
        with_iou = sum(1 for f in wt if f["iou"] is not None)
        fails = Counter(f["failure"] for f in figs if f["failure"])
        text = (f"- {label}: IoU on {with_iou} of {len(wt)} parts with watertight GT "
                f"(of {len(figs)}); failures {sum(fails.values())}"
                + (f" ({', '.join(f'{k} {v}' for k, v in fails.most_common())})" if fails else ""))
        if "cd" in run:
            cd = run["cd"]
            values = list(cd["values"].values())
            p0 = str(cd["points"][0])
            text += (f"; CD over {len(values)} parts, errors {len(cd['errors'])}, "
                     f"no mesh {cd['no_mesh']}")
            if len(cd["points"]) > 1 and values:
                p1 = str(cd["points"][-1])
                ratio = st.median(v[p1] for v in values) / st.median(v[p0] for v in values)
                text += (f"; median@{p1}/median@{p0} = {ratio:.3f} "
                         f"(densities {int(p0) / int(p1):.3f}); max@{p0} "
                         f"{max(v[p0] for v in values):.1f}")
        if any("gt_valid" in f for f in figs):
            valid = [f for f in figs if f.get("gt_valid")]
            text += (f"; GT `is_valid` for {len(valid)}, of which IoU is present for "
                     f"{sum(1 for f in valid if f['iou'] is not None)}; watertight by the instrument but not "
                     f"`is_valid`: {sum(1 for f in figs if f['gt_watertight'] and not f.get('gt_valid'))}; "
                     f"check errors {len(run.get('gt_valid_errors') or {})}")
        stops = Counter(f["stop_reason"] for f in figs)
        text += "; outcomes: " + ", ".join(f"`{k}` {v}" for k, v in stops.most_common())
        lines.append(text)
    return lines


def calls_table(runs, labels) -> list[str]:
    lines = ["| per part | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    cols = []
    for run in runs:
        figs = list(run["figures"].values())
        n = len(figs)
        total = lambda key: sum(f[key] for f in figs)  # noqa: E731
        calls = total("calls")
        assistant, rejected = total("assistant"), total("rejected")
        cols.append({
            "turns": f"{total('turns') / n:.2f} / {st.median(f['turns'] for f in figs):g}",
            "calls": f"**{calls / n:.2f}** / {st.median(f['calls'] for f in figs):g}",
            "assistant": f"{assistant / n:.2f}",
            "reasked": f"{total('reasked') / n:.2f}",
            "fallback": f"{(calls - assistant) / n:.2f}",
            "rejected": f"{rejected / n:.2f}",
            "valid": f"**{_pct(assistant, assistant + rejected)}**",
            "rej_figs": _pct(sum(1 for f in figs if f["rejected"]), n),
            "new": f"{total('new_calls') / n:.2f} ({_pct(total('new_calls'), calls)})",
            "exec": f"**{total('exec_calls') / n:.2f}** ({_pct(total('exec_calls'), calls)})",
            "raised": f"{total('raised_calls') / n:.2f} ({_pct(total('raised_calls'), calls)})",
            "cands": f"{total('new_cands') / n:.2f} ({total('exec_cands') / n:.2f})",
        })
    rows = [
        ("assistant turns (model requests), mean / median", "turns"),
        ("**tool calls**, mean / median", "calls"),
        ("- by assistant choice", "assistant"),
        ("  of which named on a re-ask", "reasked"),
        ("- harness fallback (default action)", "fallback"),
        ("rejected by the harness (invalid proposal)", "rejected"),
        ("**share of valid proposals** from the assistant", "valid"),
        ("parts with at least one rejection", "rej_figs"),
        ("calls that yielded a new candidate (after dedup)", "new"),
        ("**calls that yielded an executable candidate**", "exec"),
        ("calls that raised the best part fitness", "raised"),
        ("new candidates per part (of which executed)", "cands"),
    ]
    for title, key in rows:
        lines.append(_row(title, [col[key] for col in cols]))
    return lines


def tools_table(runs, labels) -> list[str]:
    lines = ["| tool | set | calls per part | parts with a call | rejected "
             "| executable candidate | raised the best |", "|---|---|---|---|---|---|---|"]
    seen = {tool for run in runs for f in run["figures"].values() for tool in f["tools"]}
    order = [t for t in TOOLS_ORDER if t in seen] + sorted(seen - set(TOOLS_ORDER))
    for tool in order:
        for run, label in zip(runs, labels):
            figs = list(run["figures"].values())
            stats = [f["tools"].get(tool, {}) for f in figs]
            calls = sum(s.get("calls", 0) for s in stats)
            if not calls and not any(s.get("rejected") for s in stats):
                lines.append(_row(f"`{tool}`", [label, "0", "0", "0", "—", "—"]))
                continue
            exec_ = sum(s.get("exec", 0) for s in stats)
            raised = sum(s.get("raised", 0) for s in stats)
            lines.append(_row(f"`{tool}`", [
                label, f"{calls / len(figs):.3f}", str(sum(1 for s in stats if s.get("calls"))),
                str(sum(s.get("rejected", 0) for s in stats)),
                f"{exec_} ({_pct(exec_, calls)})", f"{raised} ({_pct(raised, calls)})"]))
    return lines


def reject_reasons(runs, labels) -> list[str]:
    reasons = Counter()
    per_run = []
    for run in runs:
        c = Counter()
        for f in run["figures"].values():
            c.update(f["reject_reasons"])
        per_run.append(c)
        reasons.update(c)
    lines = ["| rejection reason | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for reason, _ in reasons.most_common():
        lines.append(_row(reason, [str(c.get(reason, 0)) for c in per_run]))
    return lines


DEFINITIONS = """\
- **Call**: a `tool_call` event, an action accepted and executed by the harness.
  **By assistant choice**: outcome `chosen`, plus actions the assistant named
  on a re-ask (`reasked`, `reasked_unparsed`, `retried`).
  The rest is fallback: the harness took the default action.
- **Rejected**: a `plan_rejected` event: the assistant proposed an action and the
  harness did not let it through. **Share of valid** = calls by assistant choice /
  (the same + rejected).
- **New candidate**: the sum of `evaluate.n` after the call (up to the next call) is
  above zero, i.e. something survived dedup. **Executable**: the sum of `evaluate.ok > 0`.
- **Raised the best**: the maximum `harness_fitness` of the new candidates (the mean
  of the available `iou` and `gms_norm` by index) is strictly above the previous best
  of the part. The first evaluated call of a part is not counted.
- **IR**: share of parts with `metrics.failure`; **mean IoU (watertight GT)**: over parts
  whose GT the run instrument judged watertight (`gt_watertight`; in older runs at least
  one closed body, now the whole mesh, `metrics.is_valid_gt`) and with a computed IoU;
  **mean IoU (GT `is_valid`)**: over parts whose GT is entirely watertight with volume > 0
  (`is_valid`), with the IoU the run computed; **mean GMS**: `gms_norm` over all parts that have it;
  **mean score**: `score` over all parts, a failure counts as 0.
- **CD** is computed from `best.stl` and GT (`trimesh.load_mesh` → `normalize_for_metrics`
  → `compute_cd`), cicada scale: sum of mean squared distances, ×1000."""


IOU_BINS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)
GMS_BINS = (0.95, 0.97, 0.99)


def _bin_label(value, edges) -> str:
    lo = None
    for hi in edges:
        if value < hi:
            return f"< {hi}" if lo is None else f"{lo}–{hi}"
        lo = hi
    return f"≥ {edges[-1]}"


def _bin_order(edges) -> list[str]:
    return [_bin_label(e - 1e-9, edges) for e in edges] + [f"≥ {edges[-1]}"]


def loss_tables(run: dict, other: dict | None, target: float, worst: int) -> str:
    """Where IoU is lost: the loss `Σ(1 − IoU)` by IoU bin, final GMS and outcome.

    Parts are those with a final IoU and, if `--gt-valid` was collected, a GT that
    passes `is_valid`; otherwise those with `gt_watertight` from the run instrument.
    """
    figs = run["figures"]
    has_valid = any("gt_valid" in f for f in figs.values())
    keep = {fid: f for fid, f in figs.items() if f["iou"] is not None
            and (f.get("gt_valid") if has_valid else f["gt_watertight"])}
    if not keep:
        return "No parts with IoU.\n"
    n = len(keep)
    loss = sum(1 - f["iou"] for f in keep.values())
    mean = 1 - loss / n
    out = [f"## IoU loss: {run['run']}", "",
           f"Parts {n} ({'GT `is_valid`' if has_valid else 'instrument `gt_watertight`'}), mean IoU {mean:.4f}, "
           f"loss Σ(1 − IoU) {loss:.1f}; to reach a mean of {target} it must be cut by "
           f"{_pct(loss - n * (1 - target), loss)}.", "",
           "### By IoU bin", "", "| IoU | parts | loss share |", "|---|---|---|"]
    bins = defaultdict(list)
    for f in keep.values():
        bins[_bin_label(f["iou"], IOU_BINS)].append(f)
    for label in _bin_order(IOU_BINS):
        group = bins.get(label, [])
        out.append(_row(label, [str(len(group)), _pct(sum(1 - f["iou"] for f in group), loss)]))
    tail = sorted(keep.values(), key=lambda f: f["iou"])[:worst]
    tail_loss = sum(1 - f["iou"] for f in tail)
    out += ["", f"{len(tail)} worst parts: {_pct(tail_loss, loss)} of the loss; raising them to 1.0 gives a mean "
                f"{1 - (loss - tail_loss) / n:.4f}."]

    with_gms = [f for f in keep.values() if f["gms_norm"] is not None]
    out += ["", "### By shape closeness (final GMS)", "",
            f"'If raised' is the set's mean IoU if the IoU of groups below {target} is raised to {target}.", "",
            "| GMS | parts | loss share | of which IoU < " + f"{target} | mean IoU if raised |",
            "|---|---|---|---|---|"]
    gbins = defaultdict(list)
    for f in with_gms:
        gbins[_bin_label(f["gms_norm"], GMS_BINS)].append(f)
    for label in reversed(_bin_order(GMS_BINS)):
        group = gbins.get(label, [])
        gain = sum(max(0.0, target - f["iou"]) for f in group)
        out.append(_row(label, [str(len(group)), _pct(sum(1 - f["iou"] for f in group), loss),
                                str(sum(1 for f in group if f["iou"] < target)), f"{mean + gain / n:.4f}"]))
    if len(with_gms) < n:
        out.append(f"\nWithout GMS: {n - len(with_gms)}.")

    head = ["outcome", "parts", "loss share", "IoU < " + str(target), "turns, median", "mean IoU"]
    common = {}
    if other is not None:
        common = {fid: other["figures"][fid]["iou"] for fid in keep
                  if fid in other["figures"] and other["figures"][fid]["iou"] is not None}
        head += [f"mean IoU, {other['run']}", "max of two", "parts in the pair"]
    out += ["", "### By outcome", "", "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    outcomes = defaultdict(list)
    for fid, f in keep.items():
        outcomes[(f["stop_reason"], f.get("done_by"))].append(fid)
    for (stop, by), fids in sorted(outcomes.items(), key=lambda kv: -len(kv[1])):
        group = [keep[fid] for fid in fids]
        iters = [f["iterations"] for f in group if f.get("iterations") is not None]
        cells = [str(len(group)), _pct(sum(1 - f["iou"] for f in group), loss),
                 str(sum(1 for f in group if f["iou"] < target)),
                 f"{st.median(iters):g}" if iters else "—", _f(_mean(f["iou"] for f in group))]
        if other is not None:
            pair = [fid for fid in fids if fid in common]
            cells += [_f(_mean(common[fid] for fid in pair)),
                      _f(_mean(max(common[fid], keep[fid]["iou"]) for fid in pair)), str(len(pair))]
        out.append(_row(f"`{stop}`" + (f" / `{by}`" if by else ""), cells))
    if other is not None and common:
        out += ["", f"Per-part maximum of the two runs: {_mean(max(v, keep[fid]['iou']) for fid, v in common.items()):.4f} "
                    f"(both IoU on {len(common)}); this run on them: "
                    f"{_mean(keep[fid]['iou'] for fid in common):.4f}."]
    if not any(f.get("done_by") or f.get("iterations") is not None for f in keep.values()):
        out += ["", "The JSON has no `done_by` or turns; it was built by an older version, rebuild it with `collect`."]
    return "\n".join(out) + "\n"


def render(runs, labels) -> str:
    parts = ["## Quality", "", *quality_table(runs, labels), "", *quality_notes(runs, labels),
             "", "## Tool calls per part", "", *calls_table(runs, labels), "",
             "Percentages in parentheses are of the number of calls.", "", "### By tool", "",
             *tools_table(runs, labels), "", "### Rejection reasons", "",
             *reject_reasons(runs, labels), "", "### How things were counted", "", DEFINITIONS, ""]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("collect", help="per-part counters of one run to JSON (on the node)")
    one.add_argument("run")
    one.add_argument("--out", required=True)
    one.add_argument("--cd", action="store_true", help="compute CD from best.stl and GT")
    one.add_argument("--gt-valid", action="store_true",
                     help="check GT with the is_valid function (whole mesh watertight, volume > 0)")
    one.add_argument("--cd-points", default="8192,30000")
    one.add_argument("--workers", type=int, default=16)
    one.add_argument("--agent-root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
        help="directory with the cad_agent package (for --cd); the default is right only "
             "while the script lives in agent/tools of its tree")
    many = sub.add_parser("table", help="markdown tables from the collect JSON")
    many.add_argument("stats", nargs="+")
    many.add_argument("--label", action="append", help="column label, one per JSON")
    many.add_argument("--out")
    many.add_argument("--only-gt-valid", action="store_true",
                      help="all tables only over parts with GT is_valid (needs collect --gt-valid)")
    lost = sub.add_parser("loss", help="where IoU goes: loss by IoU bin, GMS and outcome")
    lost.add_argument("stats", help="JSON from collect (preferably with --gt-valid)")
    lost.add_argument("--vs", help="JSON of another run on the same parts: paired IoU by outcome")
    lost.add_argument("--target", type=float, default=0.99)
    lost.add_argument("--worst", type=int, default=200)
    lost.add_argument("--out")
    args = parser.parse_args()

    if args.command == "loss":
        loaded = []
        for path in (args.stats, args.vs):
            if path:
                with open(path, encoding="utf-8") as handle:
                    loaded.append(json.load(handle))
            else:
                loaded.append(None)
        text = loss_tables(loaded[0], loaded[1], args.target, args.worst)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text)
        print(text)
        return

    if args.command == "collect":
        points = tuple(int(p) for p in args.cd_points.split(","))
        result = collect(args.run, args.cd, points, args.workers, args.agent_root, args.gt_valid)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False)
        print(f"{result['run']}: {len(result['figures'])} parts → {args.out}", file=sys.stderr)
        return

    runs = []
    for path in args.stats:
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    if args.only_gt_valid:
        for run in runs:
            keep = {fid for fid, f in run["figures"].items() if f.get("gt_valid")}
            if not keep:
                parser.error(f"{run['run']}: no gt_valid, build with collect --gt-valid")
            run["figures"] = {fid: run["figures"][fid] for fid in keep}
            if "cd" in run:
                run["cd"]["values"] = {k: v for k, v in run["cd"]["values"].items() if k in keep}
                run["cd"]["errors"] = {k: v for k, v in run["cd"]["errors"].items() if k in keep}
    labels = args.label or [run["run"] for run in runs]
    if len(labels) != len(runs):
        parser.error("--label is needed once per JSON")
    text = render(runs, labels)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
