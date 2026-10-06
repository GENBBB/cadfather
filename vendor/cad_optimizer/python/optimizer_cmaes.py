"""
CAD Gradient Optimizer: CMA-ES (Covariance Matrix Adaptation) variant.

Same loss as optimizer_tool.optimize() (SDF-MSE against target sample
points) but the search is performed by pycma's CMA-ES rather than Adam
on per-parameter gradients.  Forward SDF evaluation still goes through
the C++ analytical tree — that's where almost all of the per-iteration
cost lives, so the search step (a numpy-bound CMA-ES update) is cheap
in comparison.

Why a CMA-ES variant exists:
  * It is GRADIENT-FREE — works on shapes where analytical CSG SDF
    gradients are zero or biased (boolean interiors, clipping).
  * It is INHERENTLY GLOBAL — population-based sampling escapes local
    minima that Adam gets stuck in.
  * It comes with a built-in trust-region equivalent (sigma), which
    behaves a lot like our `max_drift` clip but is adapted online.

When NOT to use it:
  * High-dimensional scripts (>50 tunable params): CMA-ES scales O(N²)
    per iter with full-cov, becomes slower than Adam-numerical-grad.
  * When analytical gradients are known-good: Adam converges in fewer
    forward evaluations.

Usage:
    from python.optimizer_cmaes import optimize
    result = optimize(cadquery_code, stl_path, sigma=0.1, steps=200)
"""

import sys
import os
from dataclasses import dataclass

# Reuse the analytical pipeline for parsing, normalization, sample-point
# generation, and the OptResult/inject_via_slots writeback.
from . import optimizer_tool as _ot
from .cq_parser import parse_cadquery

# Re-export so callers can use optimizer_cmaes.OptResult.
OptResult = _ot.OptResult

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


def optimize(cadquery_code: str, stl_path: str,
             sigma: float = 0.1,
             steps: int = 200,
             popsize: int = None,
             batch_size: int = 512,
             sdf_band: float = -1.0,
             sdf_points: int = 10000,
             verbose: bool = False,
             normalize: bool = True,
             freeze_prefixes=None,
             max_drift: float = None,
             local_fit: bool = False,
             local_band_frac: float = 0.1,
             loss_threshold: float = None,
             optimize_only_last_line: bool = False) -> OptResult:
    """Optimize CadQuery script parameters with CMA-ES.

    Args:
        sigma:   initial step size, as a fraction of `max_drift` (or 0.1
                 if max_drift is None).  CMA-ES adapts this online.
        steps:   max generations.
        popsize: population per generation.  Default = 4 + 3*log(N) where
                 N = number of tunable params (pycma's default).
        Other args match optimizer_tool.optimize() and are honoured
        identically (sampling, freeze mask, trust region, local_fit).

    Returns OptResult with best_code, final_loss, params, history.
    """
    import cma
    import numpy as np

    if _cad_grad is None:
        raise RuntimeError("C++ backend not available. Build with pybind11.")

    # ----- 1. Parse + (optionally) wrap with frame-normalization. -----
    parse_result = parse_cadquery(cadquery_code)

    if normalize:
        mesh_for_norm = _cad_grad.load_stl(stl_path)
        (tmnx, tmny, tmnz), (tmxx, tmxy, tmxz) = mesh_for_norm.bounds()
        target_max_extent = max(tmxx - tmnx, tmxy - tmny, tmxz - tmnz)
        target_center = ((tmnx + tmxx) / 2, (tmny + tmxy) / 2,
                         (tmnz + tmxz) / 2)
        scale = target_max_extent / 200.0
        shift = target_center
        parse_result.tree_desc = {
            "type": "translate3d",
            "child": {"type": "scale3d", "child": parse_result.tree_desc},
        }
        from .cq_parser import _SyntheticNode, _make_slot
        syn = _SyntheticNode()
        new_params = list(shift) + [scale] + list(parse_result.params)
        new_slots = [_make_slot(syn, v) for v in (*shift, scale)] + list(parse_result.param_slots)
        new_names = ['frozen.norm.tx', 'frozen.norm.ty', 'frozen.norm.tz',
                     'frozen.norm.scale'] + list(parse_result.param_names)
        parse_result.params = new_params
        parse_result.param_slots = new_slots
        parse_result.param_names = new_names

    tree = _cad_grad.create_tree(parse_result.tree_desc)
    tree.set_params(parse_result.params)
    params = list(parse_result.params)
    names  = list(parse_result.param_names)

    # ----- 2. Target sampling (reuse optimizer_tool helpers). -----
    mesh = _cad_grad.load_stl(stl_path)
    (mnx, mny, mnz), (mxx, mxy, mxz) = mesh.bounds()
    diag = ((mxx - mnx) ** 2 + (mxy - mny) ** 2 + (mxz - mnz) ** 2) ** 0.5
    user_band = sdf_band if (sdf_band is not None and sdf_band > 0) else None
    wide_band   = user_band if user_band is not None else max(diag, 1.0)
    narrow_band = user_band if user_band is not None else max(diag * 0.05, 0.2)
    eval_band   = user_band if user_band is not None else max(diag * 0.01, 0.02)
    coarse_band = narrow_band if local_fit else wide_band
    positions, distances = _ot._robust_sample_sdf_points(
        mesh, stl_path, sdf_points, coarse_band)
    if local_fit:
        positions, distances, _ = _ot._filter_local(
            positions, distances, tree, diag * local_band_frac,
            verbose=verbose, residual_threshold=loss_threshold)
    elif loss_threshold is not None and loss_threshold > 0:
        import numpy as _np
        _init = _np.fromiter(
            (tree.eval_sdf(positions[i, 0], positions[i, 1], positions[i, 2])
             for i in range(len(positions))),
            dtype=_np.float64, count=len(positions))
        _mask = _np.abs(_init - distances) < loss_threshold
        if int(_mask.sum()) >= 50:
            positions = _np.ascontiguousarray(positions[_mask], dtype=_np.float64)
            distances = _np.ascontiguousarray(distances[_mask], dtype=_np.float64)

    # ----- 3. Freeze mask: synthetic centers + explicit prefixes. -----
    _SYNTHETIC_CENTERS = {
        'box.cx', 'box.cy', 'box.cz',
        'cylinder.cx', 'cylinder.cy', 'cylinder.cz',
        'sphere.cx', 'sphere.cy', 'sphere.cz',
        'hole.cx', 'hole.cy', 'hole.cz',
        'cbore_hole.cx', 'cbore_hole.cy', 'cbore_hole.cz',
        'cbore_pocket.cx', 'cbore_pocket.cy', 'cbore_pocket.cz',
    }
    freeze_mask = [
        (n in _SYNTHETIC_CENTERS or n.startswith('frozen.'))
        for n in names
    ]
    if freeze_prefixes:
        for i, n in enumerate(names):
            if any(n.startswith(pfx) for pfx in freeze_prefixes):
                freeze_mask[i] = True

    # Freeze everything not on the last source line if requested.
    if optimize_only_last_line:
        real_lines = [s.line for s in parse_result.param_slots if s.line > 0]
        if real_lines:
            max_line = max(real_lines)
            for i, s in enumerate(parse_result.param_slots):
                if s.line > 0 and s.line < max_line:
                    freeze_mask[i] = True

    free_idx = [i for i, f in enumerate(freeze_mask) if not f]
    if not free_idx:
        # Nothing to optimize.
        true_final_loss = float(_cad_grad.eval_sdf_mse(tree, positions, distances))
        return OptResult(
            best_code=cadquery_code,
            final_loss=true_final_loss,
            history=[true_final_loss],
            params=list(params),
            param_names=list(names),
        )

    # ----- 4. Trust region in the FREE subspace. -----
    p_init = list(params)
    if max_drift is not None and max_drift > 0:
        drift_lo = [p_init[i] - max_drift * max(abs(p_init[i]), 1.0)
                    for i in free_idx]
        drift_hi = [p_init[i] + max_drift * max(abs(p_init[i]), 1.0)
                    for i in free_idx]
        bounds = [drift_lo, drift_hi]
    else:
        bounds = [None, None]

    # ----- 5. Skip-converged check (mirror optimizer_tool). -----
    import math as _math
    init_full_loss = float(_cad_grad.eval_sdf_mse(tree, positions, distances))
    init_rms_over_diag = _math.sqrt(init_full_loss) / diag
    if init_rms_over_diag < 1e-4:
        return OptResult(
            best_code=cadquery_code,
            final_loss=init_full_loss,
            history=[init_full_loss],
            params=list(params),
            param_names=list(names),
        )

    # ----- 6. Run CMA-ES over the free params. -----
    x0 = [p_init[i] for i in free_idx]

    # Initial sigma scaled per-param so it's meaningful across mixed units.
    # CMA-ES uses a single scalar sigma and a per-coordinate cov matrix —
    # we encode the per-param scale via cma's CMA_stds option.
    scales = [max(abs(p_init[i]), 1.0) for i in free_idx]
    if max_drift is not None and max_drift > 0:
        # Choose sigma so its 1-sigma ball ≈ sigma_frac * drift box.
        sigma0 = sigma * max_drift  # e.g. 0.1 * 0.3 = 0.03 of param scale
    else:
        sigma0 = sigma

    opts = {
        'CMA_stds':    scales,           # per-coord initial std
        'maxiter':     steps,
        'tolfun':      1e-10,
        'tolx':        1e-8,
        'verbose':     -9 if not verbose else 1,
        'verb_log':    0,
        'verb_disp':   0 if not verbose else 10,
    }
    if popsize is not None:
        opts['popsize'] = popsize
    if bounds[0] is not None:
        opts['bounds'] = bounds

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    full_params = list(p_init)
    loss_history = [init_full_loss]
    best_loss = init_full_loss
    best_x = list(x0)

    # Mini-batch sample indices (re-drawn per generation for stochastic
    # signal — same idea as Adam batch_size in optimizer_tool).
    rng = np.random.default_rng(42)

    def _eval(x_vec):
        # Inject free params into full vector and evaluate SDF-MSE.
        for k, i in enumerate(free_idx):
            full_params[i] = float(x_vec[k])
        tree.set_params(full_params)
        return float(_cad_grad.eval_sdf_mse(tree, batch_pos, batch_dist))

    n_pts = len(positions)
    while not es.stop():
        if batch_size and batch_size < n_pts:
            sel = rng.choice(n_pts, batch_size, replace=False)
            batch_pos = positions[sel]
            batch_dist = distances[sel]
        else:
            batch_pos = positions
            batch_dist = distances
        xs = es.ask()
        fitnesses = [_eval(x) for x in xs]
        es.tell(xs, fitnesses)
        # Track best at full-sample resolution.
        gen_best_i = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
        gen_best_x = xs[gen_best_i]
        for k, i in enumerate(free_idx):
            full_params[i] = float(gen_best_x[k])
        tree.set_params(full_params)
        full_loss = float(_cad_grad.eval_sdf_mse(tree, positions, distances))
        loss_history.append(full_loss)
        if full_loss < best_loss:
            best_loss = full_loss
            best_x = list(gen_best_x)

    # ----- 7. Inject best params back into source. -----
    for k, i in enumerate(free_idx):
        full_params[i] = float(best_x[k])
    # Apply tied-slot synchronisation.
    _norm_shift = 4 if normalize else 0
    adjusted = list(full_params)
    for master_idx, follower_idx in parse_result.tied_slots:
        adjusted[follower_idx + _norm_shift] = adjusted[master_idx + _norm_shift]
    # Fold synthetic center offsets into parent translate.
    for i, n in enumerate(names):
        if n in ('cylinder.cx', 'cylinder.cy', 'cylinder.cz',
                 'circle.cx', 'circle.cy',
                 'slot.cx', 'slot.cy',
                 'box.cx', 'box.cy', 'box.cz',
                 'sphere.cx', 'sphere.cy', 'sphere.cz'):
            offset = adjusted[i]
            if abs(offset) < 1e-9:
                continue
            axis = n[-1]
            target_name = f'translate.{axis}'
            for j in range(i - 1, -1, -1):
                if names[j] == target_name:
                    adjusted[j] += offset
                    adjusted[i] = 0.0
                    break

    best_code = _ot._inject_via_slots(cadquery_code, parse_result.param_slots,
                                      adjusted)
    tree.set_params(full_params)
    if user_band is None:
        pos_eval, dist_eval = _ot._robust_sample_sdf_points(
            mesh, stl_path, sdf_points, eval_band)
    else:
        pos_eval, dist_eval = positions, distances
    true_final_loss = float(_cad_grad.eval_sdf_mse(tree, pos_eval, dist_eval))

    return OptResult(
        best_code=best_code,
        final_loss=true_final_loss,
        history=loss_history,
        params=full_params,
        param_names=names,
    )
