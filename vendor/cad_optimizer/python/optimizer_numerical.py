"""
CAD Gradient Optimizer: NUMERICAL-gradient variant.

Same Adam loss (SDF-MSE against target sample points), same Python
plumbing as optimizer_tool.optimize().  The only difference: gradients
of the loss wrt each script parameter are computed by central
finite-differences inside the C++ optimizer, not by chain rule through
the analytical SDF tree.

Useful for:
  * boolean operations where the analytical CSG SDF (max(A,±B)) gives
    biased gradients in the interior — numerical gradients of the same
    expression are unbiased.
  * sanity-checking the analytical pipeline.
  * shapes whose analytical gradient is exactly zero (e.g. when the
    parameter only affects a subtree the boolean clips away).

Wall-clock cost: ~N× per step where N = number of tunable parameters,
because each step does 2N forward evaluations instead of 1 + 1
analytical-backward.  Tractable for scripts with ≲50 params.

Usage:
    from python.optimizer_numerical import optimize
    result = optimize(cadquery_code, stl_path, lr=1e-3, steps=200)
"""

import sys
import os
from dataclasses import dataclass

import numpy as np

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
from .polygon_thin import thin_code


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
    # Optional: clamped-residual filter on top of locality.
    if residual_threshold is not None and residual_threshold > 0:
        residual = init[mask] - distances[mask]
        sub_mask = np.abs(residual) < residual_threshold
        if int(sub_mask.sum()) >= 50:
            kept_pos = positions[mask][sub_mask]
            kept_dist = distances[mask][sub_mask]
            return (np.ascontiguousarray(kept_pos, dtype=np.float64),
                    np.ascontiguousarray(kept_dist, dtype=np.float64),
                    int(sub_mask.sum()))
    return (np.ascontiguousarray(positions[mask], dtype=np.float64),
            np.ascontiguousarray(distances[mask], dtype=np.float64),
            n_keep)


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
        import trimesh
        tm = trimesh.load(stl_path)
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

    # Sort reverse so we edit from bottom-right to top-left
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for li, col, end_col, new_str in edits:
        line = lines[li]
        lines[li] = line[:col] + new_str + line[end_col:]

    return '\n'.join(lines)


def _project_revolve(revolve_bounds, params: list, init: list,
                     shift: int = 0) -> None:
    """Keep revolve profiles on their side of the axis, in place:
    a vertex past the axis, or a rect whose near edge crossed it, makes OCC
    fail with `BRep_API: command not done`.  The crossing coordinate goes back
    to its initial value, not onto the axis, and a vertex that started on the
    axis stays there: a slanted edge ending on the axis is a cone apex, whose
    zero-area triangle breaks the mesh's watertightness."""
    for rb in revolve_bounds:
        ax, sd = rb["axis"], rb["side"]
        idx = [i + shift for i in rb["idx"]]
        if rb["kind"] == "poly":
            n = len(idx) // 2
            rad, axl = idx[:n], idx[n:]
            for i in rad:
                if (params[i] - ax) * sd < 0 or abs(init[i] - ax) < 1e-9:
                    params[i] = init[i]
            # an edge from an on-axis vertex that was perpendicular to the axis
            # stays so: the on-axis end takes its neighbour's axial coordinate
            for v in range(n):
                if abs(init[rad[v]] - ax) >= 1e-9:
                    continue
                for u in ((v - 1) % n, (v + 1) % n):
                    if (abs(init[rad[u]] - ax) >= 1e-9
                            and abs(init[axl[u]] - init[axl[v]]) < 1e-9):
                        params[axl[v]] = params[axl[u]]
                        break
        else:  # rect: centre and half-size across the axis go back together
            # (moving only the near edge back does not survive the 6-digit
            # write-back: the rounded edge lands past the axis again)
            c, h = idx
            near = params[c] - sd * params[h]
            near0 = init[c] - sd * init[h]
            if (near - ax) * sd < 0 or abs(near0 - ax) < 1e-9:
                params[c], params[h] = init[c], init[h]


def _crosses(p) -> bool:
    """Two non-adjacent edges of the closed polygon p (n x 2) cross; repeated
    vertices (zero-length edges) are skipped, touching does not count."""
    p = p[np.any(p != np.roll(p, 1, axis=0), axis=1)]
    n = len(p)
    if n < 4:
        return False
    a, b = p, np.roll(p, -1, axis=0)
    k = np.arange(n)

    def cross(o, u, v):
        return ((u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1])
                - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0]))
    for j0 in range(0, n, 256):  # rows in blocks: n x n pairs of a long polygon
        j = np.arange(j0, min(j0 + 256, n))
        A, B = a[j, None], b[j, None]
        C, D = a[None], b[None]
        hit = ((cross(A, B, C) * cross(A, B, D) < 0)
               & (cross(C, D, A) * cross(C, D, B) < 0))
        d = (k[None] - j[:, None]) % n
        if (hit & (d > 1) & (d < n - 1)).any():
            return True
    return False


def _untangle_polygons(polygon_bounds, params: list, init: list,
                       shift: int = 0) -> None:
    """Keep sketch polygons from folding over themselves, in place:
    each vertex moves on its own, and a contour whose moved vertices cross an
    edge makes OCC build an invalid body.  A polygon that crosses itself in the
    written code (6 digits) but not at the start takes half of the largest
    share t of its move that keeps it simple (bisection); t is per polygon,
    the other params keep their optimized values."""
    def rnd(q):
        return np.array([[float(f"{x:.6g}"), float(f"{y:.6g}")] for x, y in q])
    for pb in polygon_bounds:
        idx = [i + shift for i in pb]
        p0 = np.array([init[i] for i in idx]).reshape(-1, 2)
        p1 = np.array([params[i] for i in idx]).reshape(-1, 2)
        if np.array_equal(p0, p1) or not _crosses(rnd(p1)) or _crosses(rnd(p0)):
            continue
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            if _crosses(rnd(p0 + mid * (p1 - p0))):
                hi = mid
            else:
                lo = mid
        # at lo the contour just touches itself, and 6-digit rounding decides
        # whether it crosses: half of it (valid 160 of 187 vs 144 at lo)
        for i, v in zip(idx, (p0 + lo / 2 * (p1 - p0)).ravel()):
            params[i] = float(v)


def optimize(cadquery_code: str, stl_path: str,
             lr: float = 1e-3, steps: int = 200,
             batch_size: int = 512, sdf_band: float = -1.0,
             sdf_points: int = 10000, verbose: bool = False,
             normalize: bool = True,
             freeze_prefixes=None,
             safety_guard: bool = False,
             safety_iou_samples: int = 5000,
             max_drift: float = None,
             local_fit: bool = False,
             local_band_frac: float = 0.1,
             loss_threshold: float = None,
             optimize_only_last_line: bool = False,
             coarse_mode: str = "wide",
             pick_best: bool = True,
             thin_polygons: float = 0.0,
             untangle_polygons: bool = True) -> OptResult:
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
        coarse_mode: band of the coarse pass: "wide" — full bbox
            diagonal (snapshot); "mid" — 0.2 diag; "narrow" — the fine-pass
            band (5%); "skip_close" — wide, but no coarse pass when the start
            is close (RMS/diag < 1e-2 on the wide sample): the fine pass
            gets the whole budget.
        pick_best: return the start, coarse or fine result — whichever
            written-back code has the lowest MSE on the eval band (the
            final_loss sample). On by default since 2026-09-30; False —
            the snapshot (the fine pass as is).
        thin_polygons: drop polygon vertices collinear with their
            neighbours up to this tolerance before optimizing
            (python/polygon_thin.py): dense polygons otherwise fold
            into self-intersection; 1e-4 — the code's 4-digit rounding.
            0 — off (the snapshot, the default until checked by a run). The
            start is returned as given, not thinned.
        untangle_polygons: a sketch polygon that crosses itself in the
            written-back code takes half of the largest share of its move
            that keeps it simple (_untangle_polygons). On by default
            since 2026-10-01 (run v4_untangle); False — the fold is written
            as optimized.

        Gradients are always numerical (central finite-differences) in
        this variant — see module docstring.

    Returns:
        OptResult with optimized code, loss, and history
    """
    if _cad_grad is None:
        raise RuntimeError("C++ backend not available. Build with pybind11.")

    # 0. Thin sketch polygons; the start goes back as given
    source_code = cadquery_code
    if thin_polygons:
        cadquery_code = thin_code(cadquery_code, thin_polygons)[0]

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
    user_band = sdf_band if (sdf_band is not None and sdf_band > 0) else None
    # Wide band for coarse pass: covers pred bbox even when far from target.
    # Narrow band for fine pass: tight enough to skip the interior of
    # boolean-approximated SDFs, wide enough to give continuous gradient.
    # eval_band: thin band around the actual target surface, where mesh
    # SDF is reliable; we report the loss here.
    wide_band   = user_band if user_band is not None else max(diag, 1.0)
    narrow_band = user_band if user_band is not None else max(diag * 0.05, 0.2)
    eval_band   = user_band if user_band is not None else max(diag * 0.01, 0.02)
    # In local_fit mode the prediction is assumed to be already near its
    # intended feature (LLM placed it), so the wide volumetric pass is
    # both unnecessary and harmful — its points spread across the bbox
    # would force the |init_pred| filter band to grow until far features
    # are no longer excluded.  Sample on the narrow band instead.
    if coarse_mode not in ("wide", "mid", "narrow", "skip_close"):
        raise ValueError(f"coarse_mode: {coarse_mode!r}")
    if user_band is None and coarse_mode == "mid":
        wide_band = max(diag * 0.2, 0.2)
    elif user_band is None and coarse_mode == "narrow":
        wide_band = narrow_band
    coarse_band = narrow_band if local_fit else wide_band
    positions, distances = _robust_sample_sdf_points(mesh, stl_path, sdf_points, coarse_band)

    # 3a. Local-fit filter: restrict sample points to those near the
    # *initial* prediction's surface, so the optimizer cannot stretch the
    # current partial block to cover unrendered target features.  Tree is
    # already at initial params here.
    if local_fit:
        positions, distances, _ = _filter_local(
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

    # Freeze everything not on the last source line if requested.
    if optimize_only_last_line:
        real_lines = [s.line for s in parse_result.param_slots if s.line > 0]
        if real_lines:
            max_line = max(real_lines)
            for i, s in enumerate(parse_result.param_slots):
                if s.line > 0 and s.line < max_line:
                    freeze_mask[i] = True

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
            best_code=source_code,
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

    fine_steps = steps - coarse_steps
    if coarse_mode == "skip_close" and user_band is None and init_rms_over_diag < 1e-2:
        opt_params, loss_history = list(params), [init_full_loss]
        fine_steps = steps
    else:
        result = _cad_grad.optimize(
            tree, positions, distances,
            lr=lr, steps=coarse_steps, batch_size=batch_size,
            early_stop=1e-8, verbose=verbose,
            freeze_mask=freeze_mask,
            numerical_gradients=True,
            param_lo=drift_lo, param_hi=drift_hi,
            stagnation_window=_stag_window, stagnation_rel=_stag_rel,
        )
        opt_params = result["params"]
        loss_history = result["loss_history"]
    candidates = [list(params), opt_params]  # start, coarse

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
                lr=fine_lr, steps=fine_steps, batch_size=batch_size,
                early_stop=1e-10, verbose=verbose,
                freeze_mask=freeze_mask,
                numerical_gradients=True,
                param_lo=drift_lo, param_hi=drift_hi,
                stagnation_window=_stag_window, stagnation_rel=_stag_rel,
            )
            fine_params = result_fine["params"]
            candidates.append(fine_params)
            fine_wide   = _wide_mse(fine_params)
            fine_narrow = _narrow_mse(fine_params)

            # Accept fine if it improves narrow metric and doesn't blow up wide.
            if fine_narrow < coarse_narrow and fine_wide <= max(coarse_wide * 3.0, 1e-4):
                opt_params = fine_params
                loss_history = loss_history + result_fine["loss_history"]
                positions, distances = pos_fine, dist_fine
            else:
                loss_history = loss_history + [coarse_narrow]

    # 6. Fold synthetic center offsets (cx/cy/cz) into their parent translate,
    #    then write optimized params back into code using source locations.
    def _write_back(pvec):
        adjusted = list(pvec)
        # Apply tied-slot synchronisation: for closed-loop polygons, copy the
        # MASTER param value to its FOLLOWER so both source literals get the
        # same value (otherwise cadquery's wire fails to close → Null TopoDS_Shape).
        # The frame normalization wrapper added 4 params at the start, so all
        # downstream slot indices shift by 4.
        _norm_shift = 4 if normalize else 0
        # before the tie sync, so a tied pair stays equal after projection
        _project_revolve(parse_result.revolve_bounds, adjusted, p_init,
                         _norm_shift)
        for master_idx, follower_idx in parse_result.tied_slots:
            adjusted[follower_idx + _norm_shift] = adjusted[master_idx + _norm_shift]
        # after the tie sync (it moves vertices too); a pair tied across
        # polygons is synced once more
        if untangle_polygons and parse_result.polygon_bounds:
            _untangle_polygons(parse_result.polygon_bounds, adjusted, p_init,
                               _norm_shift)
            for master_idx, follower_idx in parse_result.tied_slots:
                adjusted[follower_idx + _norm_shift] = adjusted[master_idx + _norm_shift]
        names = parse_result.param_names
        for i, n in enumerate(names):
            if n in ('cylinder.cx', 'cylinder.cy', 'cylinder.cz',
                     'circle.cx', 'circle.cy',
                     'slot.cx', 'slot.cy',
                     'box.cx', 'box.cy', 'box.cz',
                     'sphere.cx', 'sphere.cy', 'sphere.cz'):
                offset = adjusted[i]
                if abs(offset) < 1e-9:
                    continue
                # Find the nearest preceding translate param for same axis
                axis = n[-1]  # 'x', 'y', or 'z'
                target_name = f'translate.{axis}'
                for j in range(i - 1, -1, -1):
                    if names[j] == target_name:
                        adjusted[j] += offset
                        adjusted[i] = 0.0  # zero out the synthetic offset
                        break
        # a dropped closing vertex of a polygon takes its first vertex's value
        alias = parse_result.alias_slots
        return _inject_via_slots(
            cadquery_code, list(parse_result.param_slots) + [s for _, s in alias],
            adjusted + [adjusted[i + _norm_shift] for i, _ in alias])

    best_code = _write_back(opt_params)

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

    # pick_best: the coarse pass minimises MSE on the wide band and
    # can end worse at the surface than it started; judge the written-back
    # code of each candidate (write-back itself moves the loss) on the eval band.
    if pick_best:
        def _code_mse(code):
            try:
                pr = parse_cadquery(code)
                desc, p = pr.tree_desc, list(pr.params)
                if normalize:
                    desc = {"type": "translate3d", "child": {"type": "scale3d", "child": desc}}
                    p = list(params[:4]) + p
                t = _cad_grad.create_tree(desc)
                t.set_params(p)
                return float(_cad_grad.eval_sdf_mse(t, pos_eval, dist_eval))
            except Exception:
                return float("inf")
        scored = [(cadquery_code, params)] + [(_write_back(p), p) for p in candidates[1:]]
        losses = [_code_mse(c) for c, _ in scored]
        k = min(range(len(scored)), key=losses.__getitem__)  # ties: the earlier
        best_code, opt_params = scored[k]
        true_final_loss = losses[k]

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

    if best_code == cadquery_code:  # the start: as given, not thinned
        best_code = source_code

    return OptResult(
        best_code=best_code,
        final_loss=true_final_loss,
        history=loss_history,
        params=opt_params,
        param_names=parse_result.param_names,
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
