"""
CAD Gradient Optimizer: Python entry point.

Usage:
    from python.optimizer_tool import optimize
    result = optimize(cadquery_code, stl_path, lr=1e-3, steps=200)

    result.best_code   -- optimized CadQuery string
    result.final_loss  -- loss at convergence
    result.history     -- list of (step, loss)
"""

import sys
import os
from dataclasses import dataclass, field

# Add build dir to path so _cad_grad can be imported
_build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

try:
    import _cad_grad
except ImportError as e:
    raise ImportError(
        f"_cad_grad C++ module not importable for {sys.executable} "
        f"(searched {os.path.abspath(_build_dir)}); build it with "
        f"`cmake -S . -B build && cmake --build build`: {e}") from e

from .cq_parser import parse_cadquery


def _filter_local(positions, distances, tree, base_band, verbose=False,
                  min_pts=2000, residual_threshold=None):
    """Keep only sample points where the *initial* prediction's surface is
    within a band around them.  Used by local_fit mode to prevent the
    optimizer from spanning the partial prediction over unrendered
    target features.

    Starts at `base_band` and expands geometrically (×1.5) until at
    least `min_pts` points survive — wide-band samples are spread thinly
    in 3D, so a fixed-band filter would otherwise discard too much.  The
    band still caps at 30× base, so far-away features (e.g. a feature on
    the opposite side of the target) stay excluded.

    If `residual_threshold` is provided (and >0), additionally drop any
    point whose INITIAL residual |pred_sdf(p) - target_sdf(p)| exceeds
    the threshold — these are the "spanning candidates" where the loss
    is already huge and minimising them pulls the optimizer far from
    the LLM's intended partial fit.  This implements the clamped
    one-sided loss as a static sample filter.

    tree must be set to the INITIAL parameter values before this call.
    """
    import numpy as np
    init = np.fromiter(
        (tree.eval_sdf(positions[i, 0], positions[i, 1], positions[i, 2])
         for i in range(len(positions))),
        dtype=np.float64, count=len(positions))
    band = base_band
    while band < 30.0 * base_band:
        mask = np.abs(init) < band
        if int(mask.sum()) >= min_pts:
            break
        band *= 1.5
    n_keep = int(mask.sum())
    if verbose:
        print(f'local_fit: kept {n_keep}/{len(positions)} pts '
              f'(band={band:.4f}, base={base_band:.4f})')
    if n_keep < 50:
        # Truly degenerate (initial pred is far from every sampled point).
        # Fall back to no filtering rather than running on near-empty data.
        return positions, distances, len(positions)
    # Apply residual-threshold filter on top of the locality filter.
    if residual_threshold is not None and residual_threshold > 0:
        residual = init[mask] - distances[mask]
        sub_mask = np.abs(residual) < residual_threshold
        n_after = int(sub_mask.sum())
        if verbose:
            print(f'local_fit: residual<{residual_threshold:.4f} kept '
                  f'{n_after}/{n_keep} pts')
        if n_after >= 50:
            kept_pos = positions[mask][sub_mask]
            kept_dist = distances[mask][sub_mask]
            return (np.ascontiguousarray(kept_pos, dtype=np.float64),
                    np.ascontiguousarray(kept_dist, dtype=np.float64),
                    n_after)
        # otherwise fall back to pre-threshold (still useful signal)
    return (np.ascontiguousarray(positions[mask], dtype=np.float64),
            np.ascontiguousarray(distances[mask], dtype=np.float64),
            n_keep)


# Target-mesh cache for sign fixing: optimize() samples three bands per call
# (coarse/fine/eval) and previously re-parsed the STL + rebuilt the ray-casting
# BVH each time.  Key by (path, mtime_ns) so a reused temp path (e.g. /tmp/_t.stl
# rewritten per part) never serves a stale mesh.  Bounded to a handful of
# entries; caches the trimesh object, whose .ray BVH is built once and reused.
_TM_CACHE = {}


def _load_target_tm(stl_path):
    import os
    import trimesh
    key = (stl_path, os.stat(stl_path).st_mtime_ns)
    tm = _TM_CACHE.get(key)
    if tm is None:
        tm = trimesh.load(stl_path)
        if len(_TM_CACHE) >= 4:
            _TM_CACHE.pop(next(iter(_TM_CACHE)))
        _TM_CACHE[key] = tm
    return tm


def _robust_sample_sdf_points(mesh, stl_path, n_count, band):
    """Wrap C++ sample_sdf_points and fix the signs using ray-casting
    via trimesh.  The C++ implementation uses pseudonormals, which can
    flip sign for points far from the surface (causing the optimiser to
    chase wrong targets — see SDF investigation notes).  Here we trust
    only the magnitude from C++ and compute the SIGN ourselves with
    trimesh.contains() which uses watertight ray-casting."""
    import numpy as np
    pos, dist = _cad_grad.sample_sdf_points(mesh, n_count, band)
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    try:
        tm = _load_target_tm(stl_path)
        if not tm.is_watertight:
            return pos, dist
        # contains() re-casts rays whose forward/backward parity disagree in a
        # direction drawn from the global np.random: seed it so the same
        # target and points always get the same signs.
        rng_state = np.random.get_state()
        np.random.seed(0)
        try:
            inside = tm.contains(pos)
        finally:
            np.random.set_state(rng_state)
        sign = np.where(inside, -1.0, 1.0)
        return pos, sign * np.abs(dist)
    except Exception:
        return pos, dist


@dataclass
class OptResult:
    best_code: str
    final_loss: float
    history: list[float]
    params: list[float]
    param_names: list[str]
    # Validity ladder: best_code first, then the same edit applied at decreasing
    # optimisation fraction alpha (toward the original).  A consumer that can
    # build cadquery (the pipeline) walks this and keeps the most-optimised entry
    # that rebuilds to a valid solid — guarantees a usable program even when the
    # full optimisation lands on a self-intersecting / degenerate config.
    valid_ladder: list[str] = field(default_factory=list)
    # Optional: when the safety guard ran, these are the 3D IoU (vs the
    # target STL) of the rendered prediction before/after Adam, with
    # iou_after_kept = max(before, after) if the guard reverted.
    iou_before: float = float('nan')
    iou_after: float = float('nan')
    guard_reverted: bool = False


def _inject_via_slots(code: str, slots, new_params: list) -> str:
    """Replace numeric literals in source code using ParamSlot positions.

    The parser stores internal values (e.g. half-extents) which differ
    from source literals.  We recover the conversion factor per slot by
    reading the original literal from the source and dividing by the
    internal value, then apply it to the optimized value.
    """
    lines = code.split('\n')
    edits = []
    for slot, internal_old, internal_new in zip(slots, [s.value for s in slots], new_params):
        li = slot.line - 1  # 1-based to 0-based
        if li < 0 or (slot.col == 0 and slot.end_col == 0):
            continue  # synthetic param, no source location
        # Read original source literal to compute conversion factor
        src_literal = lines[li][slot.col:slot.end_col]
        try:
            src_val = float(src_literal)
        except ValueError:
            continue
        if abs(internal_old) > 1e-12:
            factor = src_val / internal_old
        else:
            factor = 1.0
        new_src_val = internal_new * factor
        new_str = f"{new_src_val:.6g}"
        edits.append((li, slot.col, slot.end_col, new_str))

    # Dedupe edits that target the SAME source span (keep the last).  A reused
    # workplane (e.g. w0 used by several solids) makes the parser emit one param
    # per use, all pointing at the same origin=(x,y,z) literal; without this,
    # two edits at one span shift each other's positions and merge into garbage
    # (origin=(0,0,-21) -> '-0.3871220.311167').  Applying one value per span is
    # correct: those params are frozen wp_origin copies with identical values.
    _seen = {}
    for e in edits:
        _seen[(e[0], e[1], e[2])] = e[3]
    edits = [(li, col, ec, s) for (li, col, ec), s in _seen.items()]

    # Sort reverse so we edit from bottom-right to top-left
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for li, col, end_col, new_str in edits:
        line = lines[li]
        lines[li] = line[:col] + new_str + line[end_col:]

    return '\n'.join(lines)


def optimize(cadquery_code: str, stl_path: str,
             lr: float = 1e-3, steps: int = 200,
             batch_size: int = 512, sdf_band: float = -1.0,
             sdf_points: int = 10000, verbose: bool = False,
             normalize: bool = True,
             freeze_prefixes=None,
             identifiability_gate: bool = False,
             identifiability_rel: float = 1e-3,
             clamp_frac: float = 0.0,
             numerical_gradients: bool = False,
             numerical_fallback: bool = False,
             fallback_threshold: float = 0.1,
             safety_guard: bool = False,
             safety_iou_samples: int = 5000,
             max_drift: float = None,
             n_restarts: int = 1,
             restart_perturb: float = 0.2,
             local_fit: bool = False,
             local_band_frac: float = 0.1,
             loss_threshold: float = None,
             residual_importance: float = 0.0,
             resample_every: int = 25,
             occ_weight: float = 0.0,
             occ_points: int = 20000,
             occ_batch: int = 1024,
             occ_expand: float = 0.25,
             optimize_only_last_line: bool = False) -> OptResult:
    """
    Optimize numerical parameters of a CadQuery script to match a target STL.

    Args:
        cadquery_code: CadQuery Python code as a string
        stl_path: Path to the target .stl file
        lr: Adam learning rate
        steps: Maximum optimization steps
        batch_size: Mini-batch size for gradient estimation
        sdf_band: Band width for SDF point sampling
        sdf_points: Number of SDF reference points to sample
        verbose: Print loss at each step
        freeze_prefixes: List of param name prefixes to freeze during optimization.
            E.g. ["fillet", "chamfer"] freezes all fillet.r and chamfer.s params.
        numerical_gradients: If True, use finite-difference gradients instead of
            analytical. Slower (~Nx per step, where N = number of params) but more
            robust for boolean operations where analytical gradients may be zero.
        numerical_fallback: If True, run analytical first, then if loss doesn't
            improve enough, re-run with numerical gradients.
        fallback_threshold: Loss improvement ratio threshold for fallback.
            If final_loss / initial_loss > fallback_threshold, trigger numerical retry.

    Returns:
        OptResult with optimized code, loss, and history
    """
    if _cad_grad is None:
        raise RuntimeError("C++ backend not available. Build with pybind11.")

    # 0a. Auto-desugar the image2cad FUNCTIONAL cadgen format (best.py:
    # r=extrude(r, point, 'XY', "sketch()...", h) etc.) into plain method-chain
    # CadQuery.  Detection is a cheap regex; the rewrite runs cadgen's own
    # surface projections in the cad_utils interpreter (subprocess) so offsets
    # are exact.  On any failure the original code proceeds (parser will bail,
    # pick-best keeps base).  Gate: CAD_NO_AUTO_DESUGAR.
    import os as _os_ds
    if not _os_ds.environ.get('CAD_NO_AUTO_DESUGAR'):
        try:
            from .cadgen_desugar import is_cadgen_functional, desugar_cadgen
            if is_cadgen_functional(cadquery_code):
                cadquery_code = desugar_cadgen(cadquery_code)
        except Exception:
            pass

    # 0. Auto-resolve face-relative workplanes (copyWorkplane(face_wN) built on
    # PointOnFaceSelector faces).  The AST parser cannot resolve those frames
    # statically -- unresolved, the cut collapses to the root frame and the
    # optimiser distorts it.  Detection is a cheap regex, so codes without the
    # construct pay nothing; the rewrite execs the code once (cadquery resolves
    # the face) and is idempotent, so callers that already preprocessed via
    # resolve_face_frames are unaffected.  Gate: CAD_NO_AUTO_FACE_RESOLVE.
    import os as _os_ff
    if not _os_ff.environ.get('CAD_NO_AUTO_FACE_RESOLVE'):
        try:
            from .face_frames import needs_face_resolution, resolve_face_frames
            if needs_face_resolution(cadquery_code):
                cadquery_code = resolve_face_frames(cadquery_code)
        except Exception:
            pass

    # 1. Parse CadQuery code
    parse_result = parse_cadquery(cadquery_code)

    # 1a. Frame alignment (mandatory by default).
    # Convention: the CadQuery script lives in [-100, 100]^3.
    # We map it onto whatever frame the target lives in:
    #     scale = max(target bbox extent) / 200
    #     shift = target bbox center
    # If the target is already in [0, 1]^3 the resulting transform is
    # exactly (scale=1/200, shift=(0.5, 0.5, 0.5)).
    # If the target is in a different frame, the same formula re-targets.
    # Both wrapper params are frozen; the optimiser tunes the script's
    # literals within their native [-100, 100]^3 frame.
    if normalize:
        mesh_for_norm = _cad_grad.load_stl(stl_path)
        (tmnx, tmny, tmnz), (tmxx, tmxy, tmxz) = mesh_for_norm.bounds()
        target_max_extent = max(tmxx - tmnx, tmxy - tmny, tmxz - tmnz)
        target_center = ((tmnx + tmxx) / 2, (tmny + tmxy) / 2, (tmnz + tmxz) / 2)
        scale = target_max_extent / 200.0
        shift = target_center
        parse_result.tree_desc = {
            "type": "translate3d",
            "child": {"type": "scale3d", "child": parse_result.tree_desc},
        }
        from python.cq_parser import _SyntheticNode, _make_slot
        syn = _SyntheticNode()
        new_params = list(shift) + [scale] + list(parse_result.params)
        new_slots = [_make_slot(syn, v) for v in (*shift, scale)] + list(parse_result.param_slots)
        new_names = ['frozen.norm.tx', 'frozen.norm.ty', 'frozen.norm.tz',
                     'frozen.norm.scale'] + list(parse_result.param_names)
        parse_result.params = new_params
        parse_result.param_slots = new_slots
        parse_result.param_names = new_names

    tree_desc = parse_result.tree_desc
    params = parse_result.params

    # 2. Build C++ tree
    tree = _cad_grad.create_tree(tree_desc)
    tree.set_params(params)

    # 3. Load target mesh. Two-stage sampling:
    #    * wide band (full bbox diag) for the coarse pass — catches distant
    #      starts, gradient flows from volumetric SDF difference.
    #    * narrow band (~0.05 diag) for the fine pass — analytical boolean
    #      SDFs (max(A,±B)) diverge from true SDF in interiors; near-surface
    #      sampling sidesteps that so booleans can reach true zero loss.
    mesh = _cad_grad.load_stl(stl_path)
    (mnx, mny, mnz), (mxx, mxy, mxz) = mesh.bounds()
    diag = ((mxx - mnx) ** 2 + (mxy - mny) ** 2 + (mxz - mnz) ** 2) ** 0.5
    # Clamped one-sided loss (DEFAULT-OFF, clamp_frac=0.0): when enabled it drops
    # band points whose INITIAL residual exceeds clamp_frac*diag.  The intent was
    # to skip structural mismatch the prediction cannot represent (e.g. an
    # extrude+revolve approximating a sweep).  But it is a SILENT POINT-LEVEL GATE
    # that cannot tell "structural mismatch" from "large-but-FIXABLE parameter
    # error": a faithful cut/extrude whose depth is off by, say, 25 produces a
    # residual band of exactly the points that carry the depth gradient — clamping
    # drops them and FREEZES the depth (colleague's cut+extrude finding: with the
    # clamp on the depth never moved from its init; with it off it recovered to the
    # true value).  Pure-gradient Adam should optimise every prediction; the
    # pipeline's pick-best (keep max(base, opt)) is the safety net against the rare
    # structurally-broken case the clamp was meant to protect.  Pass clamp_frac>0
    # to opt back in (or loss_threshold directly).
    if loss_threshold is None and clamp_frac and clamp_frac > 0.0:
        loss_threshold = clamp_frac * diag
    user_band = sdf_band if (sdf_band is not None and sdf_band > 0) else None
    # Wide band for coarse pass: covers pred bbox even when far from target.
    # Narrow band for fine pass: tight enough to skip the interior of
    # boolean-approximated SDFs, wide enough to give continuous gradient.
    # eval_band: thin band around the actual target surface, where mesh
    # SDF is reliable; we report the loss here.
    # Coarse-pass band.  A FULL-diagonal band samples points deep in interiors /
    # far field where the analytical polygon/CSG SDF MAGNITUDE is unreliable
    # (max(A,+-B) is a bound, not a true distance) -- its gradient drags polygon/
    # boolean ops OFF the true minimum even when started AT it (extrude_16 0.69,
    # hole_15 0.50, cut_11 0.93 from-true; a narrow band holds them at ~1.0).
    # Use a modest catch radius instead of the full diagonal.  (start is normally
    # the LLM prediction, already near the target, so the wide catch isn't needed.)
    wide_band   = user_band if user_band is not None else max(diag * 0.1, 1.0)
    narrow_band = user_band if user_band is not None else max(diag * 0.05, 0.2)
    eval_band   = user_band if user_band is not None else max(diag * 0.01, 0.02)
    # In local_fit mode the prediction is assumed to be already near its
    # intended feature (LLM placed it), so the wide volumetric pass is
    # both unnecessary and harmful — its points spread across the bbox
    # would force the |init_pred| filter band to grow until far features
    # are no longer excluded.  Sample on the narrow band instead.
    coarse_band = narrow_band if local_fit else wide_band
    positions, distances = _robust_sample_sdf_points(mesh, stl_path, sdf_points, coarse_band)

    # Volumetric occupancy points for the sign regularizer (occ_weight > 0).
    # Uniform over the (expanded) target bbox, labelled inside/outside by the
    # target mesh.  The analytical CSG SDF's SIGN is exact everywhere, so this
    # supervises the WHOLE volume — penalising pred-solid-where-target-empty
    # (balloon) and pred-empty-where-target-solid (hollow) — which the thin
    # surface band cannot.  occ_weight is a fraction of the target diagonal
    # (scale-invariant); the C++ term uses a magnitude-robust linear hinge.
    occ_pos = occ_sig = None
    _occ_kw = {}
    if occ_weight and occ_weight > 0.0:
        try:
            import numpy as _onp
            import trimesh as _otm
            _gt = _otm.load(stl_path)
            if _gt.is_watertight:
                _bmin, _bmax = _gt.bounds
                _c = (_bmin + _bmax) / 2.0
                _e = (_bmax - _bmin)
                _lo = _c - _e * (0.5 + occ_expand)
                _hi = _c + _e * (0.5 + occ_expand)
                _rng = _onp.random.default_rng(0)
                _pts = _rng.uniform(_lo, _hi, (occ_points, 3))
                _inside = _gt.contains(_pts)
                occ_pos = _onp.ascontiguousarray(_pts, dtype=_onp.float64)
                occ_sig = _onp.where(_inside, -1.0, 1.0).astype(_onp.float64)
                _occ_kw = dict(occ_positions=occ_pos, occ_signs=occ_sig,
                               occ_weight=occ_weight * diag, occ_batch=occ_batch)
                if verbose:
                    print(f'occupancy reg: {int(_inside.sum())}/{occ_points} inside, '
                          f'occ_weight_abs={occ_weight * diag:.3f}')
            elif verbose:
                print('occupancy reg: target not watertight -> disabled')
        except Exception as _oe:
            if verbose:
                print(f'occupancy reg skipped: {_oe}')

    # 3a. Local-fit filter: restrict sample points to those near the
    # *initial* prediction's surface, so the optimizer cannot stretch the
    # current partial block to cover unrendered target features.  Tree is
    # already at initial params here.
    if local_fit:
        positions, distances, _ = _filter_local(
            positions, distances, tree, diag * local_band_frac,
            verbose=verbose, residual_threshold=loss_threshold)
    elif loss_threshold is not None and loss_threshold > 0:
        # Threshold-only mode: clamp the loss by DROPPING points whose
        # initial residual exceeds the threshold.  Equivalent to the
        # clamped one-sided loss as a static sample filter (no locality).
        import numpy as _np
        _init = _np.fromiter(
            (tree.eval_sdf(positions[i, 0], positions[i, 1], positions[i, 2])
             for i in range(len(positions))),
            dtype=_np.float64, count=len(positions))
        _resid = _init - distances
        _mask = _np.abs(_resid) < loss_threshold
        _n_after = int(_mask.sum())
        if verbose:
            print(f'loss_threshold<{loss_threshold:.4f}: kept '
                  f'{_n_after}/{len(positions)} pts')
        if _n_after >= 50:
            positions = _np.ascontiguousarray(positions[_mask], dtype=_np.float64)
            distances = _np.ascontiguousarray(distances[_mask], dtype=_np.float64)

    # 4. Build freeze mask.
    # Synthetic 3D primitive centers duplicate the DoF of a parent `.translate()`
    # and have no source literal to write back to — always freeze them.
    _SYNTHETIC_CENTERS = {
        'box.cx', 'box.cy', 'box.cz',
        'cylinder.cx', 'cylinder.cy', 'cylinder.cz',
        'sphere.cx', 'sphere.cy', 'sphere.cz',
        'hole.cx', 'hole.cy', 'hole.cz',
        'cbore_hole.cx', 'cbore_hole.cy', 'cbore_hole.cz',
        'cbore_pocket.cx', 'cbore_pocket.cy', 'cbore_pocket.cz',
    }
    names = parse_result.param_names
    freeze_mask = [
        n in _SYNTHETIC_CENTERS or n.startswith('frozen.')
        for n in names
    ]
    if freeze_prefixes:
        for i, n in enumerate(names):
            if any(n.startswith(pfx) for pfx in freeze_prefixes):
                freeze_mask[i] = True

    # Optional: freeze every literal that isn't on the LAST line of the
    # source script.  Useful for stepwise pipelines where prior blocks
    # are already-optimised and only the freshly-added line should move.
    if optimize_only_last_line:
        # Collect source lines from real (non-synthetic) slots.
        real_lines = [s.line for s in parse_result.param_slots
                      if s.line > 0]
        if real_lines:
            max_line = max(real_lines)
            for i, s in enumerate(parse_result.param_slots):
                if s.line > 0 and s.line < max_line:
                    freeze_mask[i] = True
            if verbose:
                n_frozen = sum(1 for i, s in enumerate(parse_result.param_slots)
                               if s.line > 0 and s.line < max_line)
                n_free = sum(1 for i, s in enumerate(parse_result.param_slots)
                             if s.line == max_line)
                print(f'optimize_only_last_line: max_line={max_line}, '
                      f'froze {n_frozen} params from earlier lines, '
                      f'kept {n_free} params from last line free')

    # 5. Coarse pass (wide band). Give it 70% of the budget; the fine pass
    # below is a polishing step and doesn't need many steps once we're close.
    coarse_steps = steps if user_band is not None else max((steps * 7) // 10, 1)

    # 5pre. Skip-converged + adaptive stagnation.  Compute initial RMS
    # error per sample as a fraction of target's diagonal — this is a
    # scale-invariant measure of how well our SDF model already matches
    # the target before any optimisation.
    #
    # If RMS/diag < 1e-3: we're at numerical noise floor — skip optimisation.
    # If RMS/diag < 1e-2: we're "close" — Adam can drift the noise without
    #   meaningfully decreasing loss.  Use AGGRESSIVE stagnation (W=25, rel=0.1)
    #   so Adam bails as soon as loss stops decreasing fast.
    # Else: there's real signal to optimise.  Use LAX stagnation (W=200,
    #   rel=0.002) so Adam can work through transient bad states and find
    #   the better minimum (e.g., 00605956: IoU crashes at step 200 then
    #   recovers to step 800).
    import math as _math
    init_full_loss = float(_cad_grad.eval_sdf_mse(tree, positions, distances))
    init_rms_over_diag = _math.sqrt(init_full_loss) / diag
    # Tightened from 1e-3 → 1e-4: 1e-3 was hacking the regression count
    # by skipping cases that USED to improve in baseline.  Only skip when
    # the SDF model is at TRUE numerical noise floor.
    if init_rms_over_diag < 1e-4:
        return OptResult(
            best_code=cadquery_code,
            final_loss=init_full_loss,
            history=[init_full_loss],
            params=list(parse_result.params),
            param_names=list(parse_result.param_names),
        )
    # Lax stagnation across all cases.  Earlier aggressive setting
    # (W=25, rel=0.1) caused 35+ previously-improving cases to stagnate
    # — the aggressive window stopped Adam BEFORE it found improvements
    # that baseline (W=200) could reach.  Trust Adam to converge with
    # the standard window.
    _stag_window, _stag_rel = 200, 0.002
    # Importance sampling perturbs the per-step batch loss (different points each
    # rebuild), which can trip the stagnation early-stop prematurely.  Disable it
    # when importance is active so Adam runs the full step budget.
    if residual_importance and residual_importance > 0.0:
        _stag_window = 0

    # Identifiability gate (DEFAULT-OFF): freeze params the LOSS is ~insensitive
    # to.  Intent was to stop Adam random-walking dead params and distorting
    # otherwise-faithful complex parts.  But the threshold is RELATIVE
    # (identifiability_rel * max_grad): when one param dominates the gradient (e.g.
    # a big box body), genuinely-identifiable secondary params — a cut depth, a
    # small feature — fall below it and get FROZEN, so they never optimise (the
    # colleague's pocket-cut depth froze here: "froze 1/11 low-gradient params").
    # Like the clamp, it is a guard that sacrifices fixable params; the pipeline's
    # part-level pick-best is the safety net against the drift it targeted.  Pass
    # identifiability_gate=True to opt back in (lower identifiability_rel to freeze
    # only TRULY-dead params rather than merely-weak ones).
    if identifiability_gate and not optimize_only_last_line:
        import numpy as _np
        _P = _np.ascontiguousarray(positions); _D = _np.ascontiguousarray(distances)
        if len(_P) > 4000:
            _ix = _np.linspace(0, len(_P) - 1, 4000).astype(int)
            _P = _np.ascontiguousarray(_P[_ix]); _D = _np.ascontiguousarray(_D[_ix])
        _base = float(_cad_grad.eval_sdf_mse(tree, _P, _D))
        _p0 = list(params); _eps = 1e-2; _g = [0.0] * len(_p0)
        for _i in range(len(_p0)):
            if freeze_mask[_i]:
                continue
            _pp = list(_p0); _pp[_i] += _eps; tree.set_params(_pp)
            _g[_i] = abs(float(_cad_grad.eval_sdf_mse(tree, _P, _D)) - _base) / _eps
        tree.set_params(_p0)
        _gmax = max(_g) if _g else 0.0
        if _gmax > 0.0:
            _thr = identifiability_rel * _gmax
            _nf = 0
            for _i in range(len(_p0)):
                if not freeze_mask[_i] and _g[_i] < _thr:
                    freeze_mask[_i] = True; _nf += 1
            if verbose:
                print(f"identifiability gate: froze {_nf}/{len(_p0)} low-gradient params")


    # Trust-region bounds: each tunable param is allowed to drift at most
    # `max_drift * max(|p_init|, 1)` from its initial value.  This caps
    # the harm the optimiser can do when it follows a biased gradient
    # (e.g. boolean SDF approximation pointing in the wrong direction)
    # without preventing genuine optimisation when the gradient is sound.
    p_init = list(params)
    if max_drift is not None and max_drift > 0:
        drift_lo, drift_hi = [], []
        for i, v in enumerate(p_init):
            if freeze_mask[i]:
                drift_lo.append(v); drift_hi.append(v)
            else:
                d = max_drift * max(abs(v), 1.0)
                drift_lo.append(v - d); drift_hi.append(v + d)
    else:
        drift_lo, drift_hi = [], []

    result = _cad_grad.optimize(
        tree, positions, distances,
        lr=lr, steps=coarse_steps, batch_size=batch_size,
        early_stop=1e-8, verbose=verbose,
        freeze_mask=freeze_mask,
        numerical_gradients=numerical_gradients,
        param_lo=drift_lo, param_hi=drift_hi,
        stagnation_window=_stag_window, stagnation_rel=_stag_rel,
        importance_mix=residual_importance, resample_every=resample_every,
        **_occ_kw,
    )
    opt_params = result["params"]
    loss_history = result["loss_history"]

    # 5a. Fine pass (narrow band) — only when user didn't pin sdf_band.
    # Keep coarse params if fine doesn't strictly improve on BOTH wide and
    # narrow metrics (the latter guards against the fine pass collapsing
    # small features that don't register in the narrow surface samples).
    if user_band is None:
        pos_fine, dist_fine = _robust_sample_sdf_points(mesh, stl_path, sdf_points, narrow_band)

        # Local-fit filter on the fine-pass sample too — must use the
        # INITIAL prediction surface, so temporarily reset tree params.
        if local_fit:
            tree.set_params(p_init)
            pos_fine, dist_fine, _ = _filter_local(
                pos_fine, dist_fine, tree, diag * local_band_frac,
                verbose=verbose, residual_threshold=loss_threshold)
            tree.set_params(opt_params)
        elif loss_threshold is not None and loss_threshold > 0:
            tree.set_params(p_init)
            import numpy as _np
            _init = _np.fromiter(
                (tree.eval_sdf(pos_fine[i, 0], pos_fine[i, 1], pos_fine[i, 2])
                 for i in range(len(pos_fine))),
                dtype=_np.float64, count=len(pos_fine))
            _mask = _np.abs(_init - dist_fine) < loss_threshold
            if int(_mask.sum()) >= 50:
                pos_fine  = _np.ascontiguousarray(pos_fine[_mask],  dtype=_np.float64)
                dist_fine = _np.ascontiguousarray(dist_fine[_mask], dtype=_np.float64)
            tree.set_params(opt_params)

        def _wide_mse(pvec):
            tree.set_params(pvec)
            return float(_cad_grad.eval_sdf_mse(tree, positions, distances))

        def _narrow_mse(pvec):
            tree.set_params(pvec)
            return float(_cad_grad.eval_sdf_mse(tree, pos_fine, dist_fine))

        coarse_wide   = _wide_mse(opt_params)
        coarse_narrow = _narrow_mse(opt_params)

        # If the coarse pass already converged, skip the fine pass — the
        # narrow-band sample can bias toward dominant features and cause
        # small ones (holes, thin cylinders) to collapse.
        if coarse_wide < 1e-6 and coarse_narrow < 1e-4:
            loss_history = loss_history + [coarse_narrow]
        else:
            tree.set_params(opt_params)
            # Fine pass: smaller lr because we're already in the basin —
            # large steps overshoot and lose the surface-touch precision.
            fine_lr = lr * 0.2
            result_fine = _cad_grad.optimize(
                tree, pos_fine, dist_fine,
                lr=fine_lr, steps=steps - coarse_steps, batch_size=batch_size,
                early_stop=1e-10, verbose=verbose,
                freeze_mask=freeze_mask,
                numerical_gradients=numerical_gradients,
                param_lo=drift_lo, param_hi=drift_hi,
                stagnation_window=_stag_window, stagnation_rel=_stag_rel,
                importance_mix=residual_importance, resample_every=resample_every,
                **_occ_kw,
            )
            fine_params = result_fine["params"]
            fine_wide   = _wide_mse(fine_params)
            fine_narrow = _narrow_mse(fine_params)

            # Accept fine if it improves narrow metric and doesn't blow up wide.
            if fine_narrow < coarse_narrow and fine_wide <= max(coarse_wide * 3.0, 1e-4):
                opt_params = fine_params
                loss_history = loss_history + result_fine["loss_history"]
                positions, distances = pos_fine, dist_fine
            else:
                loss_history = loss_history + [coarse_narrow]

    # 5b. Numerical fallback: if analytical didn't converge well, retry with numerical
    if numerical_fallback and not numerical_gradients and len(loss_history) >= 2:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        if initial_loss > 1e-10 and final_loss / initial_loss > fallback_threshold:
            if verbose:
                print(f"Analytical gradient stalled (ratio={final_loss/initial_loss:.3f}), "
                      f"retrying with numerical gradients...")
            # Reset tree to original params and retry
            tree.set_params(params)
            result_num = _cad_grad.optimize(
                tree, positions, distances,
                lr=lr, steps=steps, batch_size=batch_size,
                early_stop=1e-8, verbose=verbose,
                freeze_mask=freeze_mask,
                numerical_gradients=True,
                param_lo=drift_lo, param_hi=drift_hi,
            )
            num_loss = result_num["loss_history"]
            if num_loss and num_loss[-1] < final_loss:
                if verbose:
                    print(f"Numerical improved: {final_loss:.6f} -> {num_loss[-1]:.6f}")
                opt_params = result_num["params"]
                loss_history = loss_history + num_loss  # concatenate histories

    # 6. Fold synthetic center offsets (cx/cy/cz) into their parent translate,
    #    then write optimized params back into code using source locations.
    _norm_shift = 4 if normalize else 0
    names = parse_result.param_names

    def _build_code(vec):
        """Turn a raw optimised param vector into source code: tied-slot sync,
        synthetic-centre fold, arc through-point write-back, then inject."""
        adj = list(vec)
        # tied-slot synchronisation: closed-loop polygons copy MASTER -> FOLLOWER
        # so both literals match (else the wire fails to close -> Null shape).
        for master_idx, follower_idx in parse_result.tied_slots:
            adj[follower_idx + _norm_shift] = adj[master_idx + _norm_shift]
        for i, n in enumerate(names):
            if n in ('cylinder.cx', 'cylinder.cy', 'cylinder.cz',
                     'circle.cx', 'circle.cy',
                     'slot.cx', 'slot.cy',
                     'box.cx', 'box.cy', 'box.cz',
                     'sphere.cx', 'sphere.cy', 'sphere.cz'):
                # Fold ONLY SYNTHETIC centres (no source literal to write to).
                _slot = parse_result.param_slots[i]
                if not (_slot.col == 0 and _slot.end_col == 0):
                    continue
                offset = adj[i]
                if abs(offset) < 1e-9:
                    continue
                axis = n[-1]
                target_name = f'translate.{axis}'
                for j in range(i - 1, -1, -1):
                    if names[j] == target_name:
                        adj[j] += offset
                        adj[i] = 0.0
                        break
        # Arc through-point write-back (arcpoly minor arcs): recompute each arc's
        # through-point from the optimised endpoints so .arc(p1, through, p3) stays
        # a valid arc through the moved endpoints.
        wb_slots = list(parse_result.param_slots)
        wb_vals = list(adj)
        if getattr(parse_result, 'arc_writeback', None):
            from .cq_parser import _arc_through_point
            for aw in parse_result.arc_writeback:
                ai = aw['a_idx'] + _norm_shift
                bi = aw['b_idx'] + _norm_shift
                tx, ty = _arc_through_point(adj[ai], adj[ai + 1],
                                            adj[bi], adj[bi + 1],
                                            aw['r_s'], aw['side'], aw.get('major', False))
                wb_slots.append(aw['mx_slot']); wb_vals.append(tx)
                wb_slots.append(aw['my_slot']); wb_vals.append(ty)
        return _inject_via_slots(cadquery_code, wb_slots, wb_vals)

    adjusted = list(opt_params)
    best_code = _build_code(opt_params)
    # Validity ladder: the same edit at decreasing optimisation fraction (toward
    # the original).  A cadquery-capable consumer keeps the most-optimised entry
    # that rebuilds to a valid solid.  Cheap (text injects only, no cadquery here
    # so the optimiser stays env-light and <1s).
    _p_init = list(params)
    valid_ladder = [best_code]
    for _alpha in (0.7, 0.45, 0.25, 0.1):
        _vec = [pi + _alpha * (po - pi) for pi, po in zip(_p_init, opt_params)]
        valid_ladder.append(_build_code(_vec))
    valid_ladder.append(cadquery_code)  # alpha=0 (original) — always valid

    # Final loss = full-sample MSE on narrow band (surface accuracy).
    # Batch-averaged loss_history[-1] can early-stop on a lucky subsample and
    # grossly underreport error for shapes with rare high-error regions
    # (e.g. small holes inside a large plate).
    tree.set_params(opt_params)
    if user_band is None:
        pos_eval, dist_eval = _robust_sample_sdf_points(
            mesh, stl_path, sdf_points, eval_band)
    else:
        pos_eval, dist_eval = positions, distances
    true_final_loss = float(_cad_grad.eval_sdf_mse(tree, pos_eval, dist_eval))

    # Safety guard: render the ORIGINAL prediction and the OPTIMISED one via
    # cadquery, compute 3D IoU against the target for each, and keep the
    # higher one.  The Adam loop minimises an SDF-MSE on a thin surface band,
    # which is an approximation that can diverge from true shape similarity
    # (especially for boolean shapes where the analytical CSG SDF differs
    # from the mesh SDF in interiors).  This guarantees we never return a
    # worse shape than we started with.  IoU catches volume-growth failure
    # modes that surface-only metrics (chamfer) miss.
    iou_before = float('nan')
    iou_after  = float('nan')
    guard_reverted = False
    if safety_guard and not best_code == cadquery_code:
        try:
            iou_before, iou_after = _measure_mesh_iou_pair(
                cadquery_code, best_code, stl_path,
                n_samples=safety_iou_samples)
            if iou_after < iou_before - 0.005:
                if verbose:
                    print(f'safety guard reverting: IoU {iou_before:.4f} -> '
                          f'{iou_after:.4f}')
                best_code = cadquery_code
                opt_params = parse_result.params
                true_final_loss = float('nan')
                guard_reverted = True
        except Exception as e:
            if verbose:
                print(f'safety guard skipped: {e}')

    return OptResult(
        best_code=best_code,
        final_loss=true_final_loss,
        history=loss_history,
        params=opt_params,
        param_names=parse_result.param_names,
        valid_ladder=valid_ladder,
        iou_before=iou_before,
        iou_after=iou_after,
        guard_reverted=guard_reverted,
    )


def _measure_mesh_iou_pair(code_before: str, code_after: str,
                           stl_path: str, n_samples: int = 20000):
    """Render two CadQuery scripts via cadquery, then for each compute 3D
    IoU against the target STL.  Returns (iou_before, iou_after).  Caller
    catches any exception (cadquery failure, etc.)."""
    import os as _os, tempfile as _tf, uuid as _uuid
    import cadquery as _cq
    import trimesh as _trimesh
    import numpy as _np

    target = _trimesh.load(stl_path)
    tmin, tmax = target.bounds
    extent = float((tmax - tmin).max())
    centre = (tmin + tmax) / 2

    def _render(code_str):
        ns = {}
        exec(code_str, ns)
        obj = ns.get('result') or ns.get('r')
        tmp = _os.path.join(_tf.gettempdir(), f'cad_{_uuid.uuid4().hex}.stl')
        _cq.exporters.export(obj, tmp)
        m = _trimesh.load(tmp)
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        m.apply_scale(extent / 200.0)
        m.apply_translation(centre)
        return m

    def _iou(pred, tgt):
        rng = _np.random.default_rng(0)
        bmin = _np.minimum(pred.bounds[0], tgt.bounds[0])
        bmax = _np.maximum(pred.bounds[1], tgt.bounds[1])
        margin = (bmax - bmin) * 0.02
        bmin -= margin; bmax += margin
        pts = rng.uniform(bmin, bmax, (n_samples, 3))
        in_p = pred.contains(pts)
        in_t = tgt.contains(pts)
        inter = int((in_p & in_t).sum())
        union = int((in_p | in_t).sum())
        return inter / union if union > 0 else 0.0

    m_before = _render(code_before)
    m_after  = _render(code_after)
    return _iou(m_before, target), _iou(m_after, target)
