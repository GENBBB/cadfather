#!/usr/bin/env python3
"""Multi-step eval using the fast-mesh path.

For each sample, iteratively:
    iter 1: pick the best fast-mesh candidate from detect_*(gt_mesh)
    iter K: compute (ADD, CUT) = (gt \\ running, running \\ gt)
            run detectors on the largest residual; pick the candidate
            whose fast-mesh + running_pred → highest IoU vs gt
    stop when IoU plateaus (improvement < eps) or K hits max_iters

All boolean ops use trimesh (manifold3d C++ backend) — no cadquery
rendering on iter 2+ either.
"""
import argparse
import gc
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DET_RENDER_INPROCESS", "1")

import numpy as np
import trimesh

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from cadfit.candidates.detectors import (
    detect_silhouette_extrude,
    detect_revolve,
    detect_slice_fit,
    detect_planar_clusters,
    detect_loft,
    detect_helix,
    detect_local_extrudes,
    detect_local_revolves,
    detect_local_lofts,
    detect_local_sweeps,
)
from cadfit.candidates.residual import compute_residuals


class SampleTimeout(Exception):
    pass


def _alarm(*_):
    raise SampleTimeout()


# LOCAL detectors: fit one local element to a residual.  Run on BOTH
# residuals every iteration (CADFit loop).  GLOBAL detectors: try to
# capture the whole residual in one op — cheap, good for iter-1 and
# simple residuals.
_LOCAL_DETECTORS = [
    ("local_extrude",  detect_local_extrudes),
    ("local_revolve",  detect_local_revolves),
    ("local_loft",     detect_local_lofts),
    ("local_sweep",    detect_local_sweeps),
    ("helix",          detect_helix),
]
_GLOBAL_DETECTORS = [
    ("extrude",        detect_silhouette_extrude),
    ("revolve",        detect_revolve),
    ("slice_fit",      detect_slice_fit),
    ("planar_cluster", detect_planar_clusters),
    ("loft",           detect_loft),
]
_DETECTORS = _LOCAL_DETECTORS + _GLOBAL_DETECTORS


def normalize_pair(pred, gt):
    """Scale `pred` so its bbox matches gt's already-normalized [-1, 1] cube.
    Only used for iter-1 IoU when the detector's mesh wasn't built in
    gt's coordinate frame."""
    c = (pred.bounds[0] + pred.bounds[1]) / 2.0
    pred.apply_translation(-c)
    s_gt = float(np.max(gt.extents))
    s_pred = float(np.max(pred.extents))
    if s_pred > 0:
        pred.apply_scale(s_gt / s_pred)
    # Re-center
    cc = (pred.bounds[0] + pred.bounds[1]) / 2.0
    pred.apply_translation(-cc)
    return pred


def voxel_iou(a, b, pts):
    ia = a.contains(pts); ib = b.contains(pts)
    inter = int((ia & ib).sum()); union = int((ia | ib).sum())
    return inter / max(union, 1)


def _mesh_signature(m, q_vol: float = 2, q_pos: float = 2) -> tuple:
    """Geometric fingerprint of a candidate mesh for dedup.

    Two candidates with the same (rounded volume, bbox extents, centroid)
    are treated as duplicate sketch-derived operations -- the
    deterministic replacement for CADFit's learned sketch prior: adjacent
    slices produce near-identical contours, and this collapses them so
    only DISTINCT profiles survive into the (expensive) boolean-IoU loop.
    """
    try:
        vol = round(float(abs(m.volume)), q_vol)
        ext = tuple(np.round(m.extents, q_pos).tolist())
        ctr = tuple(np.round((m.bounds[0] + m.bounds[1]) / 2.0, q_pos).tolist())
        return (vol, ext, ctr)
    except Exception:
        return (id(m),)


def _dedupe_candidates(cands):
    """Collapse candidates with identical geometric signature, keeping
    the highest-scored one per (signature, op).  Input/return:
    list of (name, score, mesh, op)."""
    best = {}
    for name, score, m, op in cands:
        sig = (op,) + _mesh_signature(m)
        cur = best.get(sig)
        if cur is None or score > cur[1]:
            best[sig] = (name, score, m, op)
    out = list(best.values())
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _decimate(m, max_faces: int):
    """Quadric-decimate a mesh to <= max_faces (speed for shapely unions
    + booleans).  Returns the input unchanged on failure or if small."""
    try:
        if m is None or len(m.faces) <= max_faces:
            return m
        out = m.simplify_quadric_decimation(max_faces)
        if out is None or len(out.faces) == 0:
            return m
        return out
    except Exception:
        return m


def candidates_for_iter(gt_mesh, prev_pred, detectors, max_total):
    """Return list of (name, score, fast_mesh, op_for_iter2).

    For iter 1: pick top-k candidates across all detectors.
    For iter 2+: gather UNION (ADD residual) and CUT (CUT residual)
    candidates *separately*, then INTERLEAVE so neither op starves the
    other when we cap at max_total.  Earlier truncation-by-list-order
    silently dropped all CUT candidates when ADD generated >max_total.
    """
    if prev_pred is None:
        all_outs = []
        for name, fn in detectors:
            try:
                outs = fn(gt_mesh)
            except Exception:
                continue
            for o in outs:
                if o.mesh is not None and len(o.mesh.faces) > 0:
                    all_outs.append((name, o.score, o.mesh, "init"))
        # Dedup identical-geometry candidates (the learned-prior
        # replacement), THEN keep the top-k distinct profiles.
        all_outs = _dedupe_candidates(all_outs)
        return all_outs[:max_total]

    # iter 2+ (CADFit loop): compute BOTH residuals
    #   R+ = target \ pred   -> fit a local element, UNION it in
    #   R- = pred \ target   -> fit a local element, CUT it out
    # Run ALL detectors (local + global) on each residual.  Decimate
    # residual meshes first so per-iteration detection stays fast.
    # iter-2+ detector set: the LOCAL detectors (which fit one element
    # to the residual) + cheap global extrude/slice_fit.  Skip the
    # slow global revolve/loft/helix -- their local_* counterparts cover
    # the residual case and run ~5x faster.
    iter2_detectors = [(n, f) for (n, f) in detectors
                       if n.startswith("local_") or n in ("extrude", "slice_fit")]
    try:
        residuals = compute_residuals(prev_pred, gt_mesh)
    except Exception:
        return []
    union_list = []
    cut_list = []
    for op_name, res_mesh, bucket in (
        ("union", residuals.add_mesh, union_list),
        ("cut",   residuals.cut_mesh, cut_list),
    ):
        if res_mesh is None or len(res_mesh.faces) == 0:
            continue
        res_mesh = _decimate(res_mesh, 6000)
        for name, fn in iter2_detectors:
            try:
                outs = fn(res_mesh)
            except Exception:
                continue
            for o in outs:
                if o.mesh is not None and len(o.mesh.faces) > 0:
                    bucket.append((name, o.score, o.mesh, op_name))
    union_list = _dedupe_candidates(union_list)
    cut_list = _dedupe_candidates(cut_list)
    # Interleave so half slots go to each op.
    out_list = []
    half = max(max_total // 2, 1)
    out_list.extend(union_list[:half])
    out_list.extend(cut_list[:half])
    # Fill remaining slots with whichever has more candidates left.
    while len(out_list) < max_total and (
            len(union_list) > half or len(cut_list) > half):
        if len(union_list) > half:
            out_list.append(union_list[half])
            union_list.pop(half)
        elif len(cut_list) > half:
            out_list.append(cut_list[half])
            cut_list.pop(half)
        else:
            break
    return out_list[:max_total]


def combine(running, fast_mesh, op):
    if running is None:
        return fast_mesh
    try:
        if op == "init":
            return fast_mesh
        if op == "union":
            return trimesh.boolean.union([running, fast_mesh])
        if op == "cut":
            return trimesh.boolean.difference([running, fast_mesh])
    except Exception:
        return None
    return None


def eval_sample(stl_path, detectors, k, pts, max_iters, sample_timeout, eps):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(sample_timeout))
    best_iou = 0.0
    chain = []
    try:
        gt = trimesh.load(stl_path, process=True)
        if isinstance(gt, trimesh.Scene):
            gt = trimesh.util.concatenate(list(gt.geometry.values()))
        c = (gt.bounds[0] + gt.bounds[1]) / 2.0
        gt.apply_translation(-c)
        gt.apply_scale(2.0 / float(np.max(gt.extents)))
        # Decimate large meshes — shapely unions + boolean residuals
        # both scale with face count, so this is the single biggest
        # speed lever.  IoU is unaffected at ~8k faces.
        if len(gt.faces) > 8000:
            try:
                gt = gt.simplify_quadric_decimation(8000)
                gt.apply_translation(-(gt.bounds[0] + gt.bounds[1]) / 2.0)
            except Exception:
                pass

        # Occupancy-based scoring: point-in-mesh of a boolean result
        # equals the boolean of the operands' point-in-mesh.  So we
        # score every candidate with ONE contains() call (fast, embreex)
        # and a vectorized boolean over occupancy arrays -- NO mesh
        # boolean.  Only the winning candidate is materialized into a
        # real mesh (one boolean per accepted iteration) so the next
        # iteration can slice its residual.
        gt_in = gt.contains(pts)
        running = None
        running_in = np.zeros(len(pts), dtype=bool)
        for it in range(max_iters):
            if best_iou >= 0.99:
                break
            cands = candidates_for_iter(gt, running, detectors, k)
            if not cands:
                break
            # PHASE A — cheap pre-rank by occupancy IoU (one contains()
            # per candidate, NO mesh boolean).  Point-in-(A op B) equals
            # the boolean of point-memberships, so this estimates the
            # post-op IoU well enough to RANK candidates.
            scored = []
            for name, score, m, op in cands:
                cand_pred = m.copy()
                if op == "init" and running is None:
                    cand_pred = normalize_pair(cand_pred, gt)
                try:
                    cand_in = cand_pred.contains(pts)
                except Exception:
                    continue
                if op == "init":
                    new_in = cand_in
                elif op == "union":
                    new_in = running_in | cand_in
                elif op == "cut":
                    new_in = running_in & ~cand_in
                else:
                    continue
                pred_iou = (int((new_in & gt_in).sum())
                            / max(int((new_in | gt_in).sum()), 1))
                if pred_iou > best_iou + eps:
                    scored.append((pred_iou, name, cand_pred, op))
            if not scored:
                break
            # PHASE B — verify the top few with REAL booleans, pick the
            # true best.  Occupancy-predicted IoU can mismatch the
            # materialized result (boolean cleans up geometry); verifying
            # only the top-N keeps it cheap (~3 booleans/iter) while the
            # SELECTION uses true materialized IoU.
            scored.sort(key=lambda t: t[0], reverse=True)
            best_local = best_iou
            best_mat = None
            best_local_name = None
            best_mat_in = None
            for pred_iou, name, cand_pred, op in scored[:3]:
                materialized = combine(running, cand_pred, op)
                if materialized is None or len(materialized.faces) == 0:
                    continue
                try:
                    mat_in = materialized.contains(pts)
                except Exception:
                    continue
                true_iou = (int((mat_in & gt_in).sum())
                            / max(int((mat_in | gt_in).sum()), 1))
                if true_iou > best_local + eps:
                    best_local = true_iou
                    best_mat = materialized
                    best_mat_in = mat_in
                    best_local_name = (name, op)
            if best_mat is None or best_local <= best_iou + eps:
                break
            best_iou = best_local
            running = best_mat
            running_in = best_mat_in
            chain.append(best_local_name)
        del gt
    except SampleTimeout:
        if not chain:
            chain = [("TIMEOUT", "init")]
    finally:
        signal.alarm(0)
        gc.collect()
    return best_iou, chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-timeout", type=int, default=25)
    ap.add_argument("--iou-n", type=int, default=15_000)
    ap.add_argument("--eps", type=float, default=0.005)
    args = ap.parse_args()

    stl_paths = sorted(Path(args.dir).glob("*.stl"))
    if args.limit:
        stl_paths = stl_paths[:args.limit]

    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(args.iou_n, 3)).astype(np.float64)

    cat = (Path(args.dir).parent.name if Path(args.dir).name == "meshes"
           else Path(args.dir).name)
    print(f"eval_ms cat={cat} n={len(stl_paths)} k={args.k} max_iters={args.max_iters}",
          flush=True)
    results = []
    t0 = time.time()
    for i, p in enumerate(stl_paths):
        try:
            iou, chain = eval_sample(p, _DETECTORS, args.k, pts,
                                     args.max_iters, args.sample_timeout, args.eps)
        except Exception as e:
            iou, chain = 0.0, [(f"ERR:{type(e).__name__}", "init")]
        results.append((p.name, float(iou), chain))
        ch = " > ".join(f"{n}/{op[:3]}" for n, op in chain)[:50]
        print(f"[{i+1:2d}/{len(stl_paths)}] {p.name[:36]:36} IoU={iou:.4f} "
              f"steps={len(chain):d} {ch}", flush=True)

    ious = np.array([r[1] for r in results])
    elapsed = time.time() - t0
    print(f"\n=== summary multi-step {cat} ({elapsed:.0f}s) ===", flush=True)
    print(f"  mean    {ious.mean():.4f}", flush=True)
    print(f"  median  {float(np.median(ious)):.4f}", flush=True)
    print(f"  min     {ious.min():.4f}", flush=True)
    print(f"  >0.995  {int((ious > 0.995).sum())}/{len(ious)}", flush=True)
    print(f"  >0.95   {int((ious > 0.95).sum())}/{len(ious)}", flush=True)
    print(f"  >0.90   {int((ious > 0.90).sum())}/{len(ious)}", flush=True)


if __name__ == "__main__":
    main()
