"""Helix / coil detector — for springs and helical sweeps.

Algorithm
---------
A helical body (spring, coil) has these signatures:
  - PCA principal axis = the helix's axis (one direction has dominant
    variance because the coil's height >> coil diameter).
  - Cross-sections perpendicular to that axis show ONE small circular
    section that orbits the axis as we move along it (not multiple
    concentric rings).

Given an axis hypothesis:
  1.  Translate mesh to centroid.
  2.  Slice at K evenly-spaced heights along the axis.
  3.  Each slice has 1 or 2 small circles (coil cross-section).  Fit
      CIRCLE to each slice using the C++ section fitter; track the
      centroid (cx, cy) as a function of z.
  4.  Fit (cx(z), cy(z)) = (R cos(2π z / pitch + φ), R sin(...)).
      The coil radius R and pitch are estimated by:
          R = sqrt(<(cx - cx̄)²> + <(cy - cȳ)²>) / sqrt(2)
          pitch from FFT of cx(z) / cy(z)
  5.  The wire-cross-section radius r_w is the fitted CIRCLE radius
      at each slice (averaged).

Fast-mesh emission
------------------
Trimesh has ``trimesh.creation.cylinder`` and ``trimesh.creation.box`` but
no built-in helix sweep.  We build a polyline along the helix curve, then
``trimesh.creation.sweep_polygon(circle, path)`` to sweep a circular
profile -- yields a watertight Trimesh of the helix.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

from .extrude import DetectorOutput


def _pca_axis(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        v = v - v.mean(axis=0, keepdims=True)
        cov = (v.T @ v) / max(len(v), 1)
        w, vec = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        # Principal axis = direction of largest variance.
        return vec[:, order[0]]
    except Exception:
        return None


def _slice_centroid_radius(mesh: trimesh.Trimesh, origin, normal
                           ) -> Optional[tuple[float, float, float]]:
    """Cut a slice perpendicular to ``normal`` at ``origin``; return
    (cx, cy, r_wire) where cx, cy are the centroid in the slice's local
    2D frame and r_wire is the section's effective radius (sqrt(area/π))
    -- a robust proxy for the wire-circle radius.
    """
    try:
        sec = mesh.section(plane_origin=list(origin),
                           plane_normal=list(normal))
        if sec is None:
            return None
        p2d, _ = sec.to_planar() if hasattr(sec, "to_planar") else (None, None)
        if p2d is None or not p2d.polygons_full:
            return None
        polys = list(p2d.polygons_full)
        if not polys:
            return None
        total_area = sum(p.area for p in polys)
        if total_area <= 0:
            return None
        # Centroid is weighted by area.
        cx = sum(p.centroid.x * p.area for p in polys) / total_area
        cy = sum(p.centroid.y * p.area for p in polys) / total_area
        # Effective radius = area-equivalent circle radius.
        r_wire = math.sqrt(total_area / math.pi)
        return float(cx), float(cy), float(r_wire)
    except Exception:
        return None


def _detect_helix_axis(mesh: trimesh.Trimesh, axis: np.ndarray,
                       n_slices: int = 30,
                       ) -> Optional[dict]:
    """Test ``axis`` as a helix axis using a VERTEX-based detector.

    Rationale: section-centroid based detection fails on tightly wound
    coils (multiple wire intersections per slice cancel out).  The
    vertex approach works directly with the mesh point cloud:
      1. Project all vertices to cylindrical (r, theta, z) coords
         around the candidate axis.
      2. The MAJORITY of vertices should lie near a single r value
         (the helix mean radius); flag if r-std / r-mean is too big.
      3. Fit theta_unwrap = a*z + b after sorting by z + chunked
         unwrap.  Low residual means the points lie on a clean helix.
    """
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    center_world = mesh.bounds.mean(axis=0)
    verts = np.asarray(mesh.vertices, dtype=np.float64) - center_world
    z_all = verts @ axis
    z_min, z_max = float(z_all.min()), float(z_all.max())
    if z_max - z_min < 1e-6:
        return None

    # Orthonormal basis perpendicular to axis.
    seed = (np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9
            else np.array([0.0, 1.0, 0.0]))
    u = np.cross(axis, seed); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(axis, u)
    x_loc = verts @ u
    y_loc = verts @ v
    r_loc = np.sqrt(x_loc**2 + y_loc**2)

    # Median r is the helix mean radius.
    R = float(np.median(r_loc))
    if R < 1e-4:
        return None
    r_iqr = float(np.percentile(r_loc, 75) - np.percentile(r_loc, 25))
    if r_iqr / R > 0.7:
        # Too much radial variance -> not a clean helix.
        return None
    r_wire = max(r_iqr, 0.5 * float(np.std(r_loc)))
    if r_wire < 1e-4:
        return None

    theta = np.arctan2(y_loc, x_loc)
    # DENSE-COIL ROBUST FIT: bin vertices into fine z-slices and take
    # the circular-mean angle per slice.  A raw z-sorted unwrap fails on
    # dense coils because, at nearly-equal z, vertices span the wire's
    # whole circumference -- consecutive points jump in theta and
    # np.unwrap breaks.  Per-slice circular mean collapses that spread
    # to one representative angle per height, so unwrap is clean.
    n_bins = 60
    z_lo, z_hi = z_all.min(), z_all.max()
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, z_all, side="right") - 1,
                      0, n_bins - 1)
    zc = []; thc = []
    for b_ in range(n_bins):
        mask = bin_idx == b_
        if mask.sum() < 3:
            continue
        mean_cos = float(np.cos(theta[mask]).mean())
        mean_sin = float(np.sin(theta[mask]).mean())
        zc.append(0.5 * (edges[b_] + edges[b_ + 1]))
        thc.append(math.atan2(mean_sin, mean_cos))
    if len(zc) < 6:
        return None
    zc = np.asarray(zc); thc = np.unwrap(np.asarray(thc))

    A = np.vstack([zc, np.ones_like(zc)]).T
    try:
        sol, _, _, _ = np.linalg.lstsq(A, thc, rcond=None)
    except Exception:
        return None
    a, b = float(sol[0]), float(sol[1])
    n_turns = abs(float(thc[-1] - thc[0])) / (2 * math.pi)
    # A helix must WIND.  Gate on the number of TURNS, not the absolute
    # angle-rate: a tall, large-pitch spring has a small |a| (e.g. 0.2 rad/unit
    # over a 200-unit coil = 6.9 turns) yet is unmistakably a helix.  The old
    # ``|a| >= 0.5`` gate wrongly rejected exactly those.  Keep a tiny floor to
    # exclude flat/degenerate fits.
    if n_turns < 1.5 or abs(a) < 0.02:
        return None
    resid = float(np.std(thc - (a * zc + b)))
    # Per-slice means have much tighter residual than raw vertices.
    if resid > 0.6:
        return None
    zs_fit = zc

    pitch = 2 * math.pi / abs(a)
    # CONICAL taper: fit the coil radius linearly in z (mean per-vertex radius
    # ≈ centreline radius at that height).  Falls back to constant R if the fit
    # is degenerate/wild -> a cylindrical spring is just ra≈0.
    rA = np.vstack([z_all, np.ones_like(z_all)]).T
    rra, rrb = np.linalg.lstsq(rA, r_loc, rcond=None)[0]
    r_at_min = float(rra * z_min + rrb)
    r_at_max = float(rra * z_max + rrb)
    if min(r_at_min, r_at_max) <= 1e-6 or max(r_at_min, r_at_max) > 3.0 * R:
        r_at_min = r_at_max = R
    return {
        "axis": axis,
        "center": center_world,
        "R_helix": R,
        "r_at_min": r_at_min,
        "r_at_max": r_at_max,
        "r_wire": r_wire,
        "pitch": pitch,
        "z_min": z_min,
        "z_max": z_max,
        "angle_rate": a,
        "angle_offset": b,
        "n_slices_used": int(len(zs_fit)),
        "angle_resid": resid,
        "n_turns": n_turns,
    }


def _build_helix_mesh(params: dict, n_turns_extra: float = 0.0,
                      n_per_turn: int = 48
                      ) -> Optional[trimesh.Trimesh]:
    """Build a swept-circle helix Trimesh from params (axis, R, r_wire,
    pitch, z range).
    """
    try:
        axis = np.asarray(params["axis"], dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        R = float(params["R_helix"])
        r_wire = float(params["r_wire"])
        pitch = float(params["pitch"])
        z_min = float(params["z_min"])
        z_max = float(params["z_max"])
        center = np.asarray(params["center"], dtype=np.float64)
        angle_rate = float(params["angle_rate"])
        angle_offset = float(params["angle_offset"])
        r_at_min = float(params.get("r_at_min", R))   # conical taper
        r_at_max = float(params.get("r_at_max", R))

        z_span = z_max - z_min
        # Wind via the lstsq angle_rate (theta = a*z + b): the robust turn-count
        # estimate.  The theta-endpoint span (params['n_turns']) OVER-counts on
        # noisy end-coils (verified: 6.85 -> IoU 0.12 vs angle_rate 6.53 -> 0.30).
        n_turns = max(1.0, z_span / max(pitch, 1e-6) + n_turns_extra)
        n_points = max(int(n_per_turn * n_turns), 32)

        # Build orthonormal basis (u, v) perpendicular to axis.
        seed = (np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9
                else np.array([0.0, 1.0, 0.0]))
        u = np.cross(axis, seed)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(axis, u)

        zs_curve = np.linspace(z_min, z_max, n_points)
        path_pts = []
        for z in zs_curve:
            theta = angle_rate * z + angle_offset
            frac = (z - z_min) / max(z_span, 1e-9)
            Rz = r_at_min + frac * (r_at_max - r_at_min)   # radius at this height
            pt = (center
                  + axis * z
                  + Rz * (u * math.cos(theta) + v * math.sin(theta)))
            path_pts.append(pt)
        path_pts = np.asarray(path_pts, dtype=np.float64)

        # Build a small circular profile (cross-section of wire).
        from shapely.geometry import Polygon
        try:
            from trimesh.creation import sweep_polygon
        except Exception:
            return None
        n_circle = 24
        circle = [(r_wire * math.cos(2 * math.pi * i / n_circle),
                   r_wire * math.sin(2 * math.pi * i / n_circle))
                  for i in range(n_circle)]
        poly = Polygon(circle)
        if not poly.is_valid or poly.area <= 0:
            return None

        try:
            m3 = sweep_polygon(poly, path_pts)
        except Exception:
            return None
        if m3 is None or len(m3.faces) == 0:
            return None
        if float(m3.volume) < 0:
            m3.invert()
        return m3
    except Exception:
        return None


def _helix_path_points(params: dict, n_per_turn: int = 48) -> Optional[np.ndarray]:
    """Compute the 3D helix curve points (same recipe as the fast mesh)."""
    try:
        axis = np.asarray(params["axis"], dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        R = float(params["R_helix"]); pitch = float(params["pitch"])
        z_min = float(params["z_min"]); z_max = float(params["z_max"])
        center = np.asarray(params["center"], dtype=np.float64)
        a_rate = float(params["angle_rate"]); a_off = float(params["angle_offset"])
        r_at_min = float(params.get("r_at_min", R)); r_at_max = float(params.get("r_at_max", R))
        seed = (np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9
                else np.array([0.0, 1.0, 0.0]))
        u = np.cross(axis, seed); u /= np.linalg.norm(u) + 1e-12
        v = np.cross(axis, u)
        z_span = z_max - z_min
        n_turns = max(1.0, z_span / max(pitch, 1e-6))
        n_points = max(int(n_per_turn * n_turns), 32)
        zs = np.linspace(z_min, z_max, n_points)
        pts = []
        for z in zs:
            th = a_rate * z + a_off
            frac = (z - z_min) / max(z_span, 1e-9)
            Rz = r_at_min + frac * (r_at_max - r_at_min)
            pts.append(center + axis * z
                       + Rz * (u * math.cos(th) + v * math.sin(th)))
        return np.asarray(pts, dtype=np.float64)
    except Exception:
        return None


def _emit_helix_program(params: dict) -> str:
    """Emit a CadQuery sweep of a circular wire profile along the helix
    curve.  Renderable + parameter-optimizable on the server."""
    axis = np.asarray(params["axis"], dtype=np.float64)
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-9:
        return ""
    axis = axis / nrm
    R = float(params["R_helix"]); r_wire = float(params["r_wire"])
    z_min = float(params["z_min"]); z_max = float(params["z_max"])
    height = abs(z_max - z_min)
    center = np.asarray(params["center"], dtype=np.float64)
    r_at_min = float(params.get("r_at_min", R)); r_at_max = float(params.get("r_at_max", R))
    a_rate = float(params.get("angle_rate", 1.0))
    # PITCH = 2pi/angle_rate (the detector's lstsq slope).  makeHelix lays
    # turns = height/pitch, matching the fast mesh's a*z winding (the robust
    # turn count; the theta-endpoint n_turns over-counts on noisy end-coils).
    pitch = float(params.get("pitch", 0.0))
    if pitch <= 1e-6:
        pitch = height / max(float(params.get("n_turns", 1.0)), 0.5)
    if height <= 1e-6 or pitch <= 1e-6 or r_wire <= 1e-6 or max(r_at_min, r_at_max) <= 1e-6:
        return ""

    # Build the canonical spring along +Z at the ORIGIN (ref: gen_api Helix --
    # profile on the **XZ** plane at (R,0), swept along makeHelix), then rotate
    # (+Z -> helix axis) and translate to the coil's BOTTOM end (makeHelix runs
    # `height` UP from center, so center must be the bottom, not the bbox centre).
    # Orient so the radius GROWS along +local-Z (bottom = smaller-radius end) so
    # the conical `angle` (cone semi-angle, deg) is positive.
    if r_at_min <= r_at_max:
        r_bot, r_top, ldir, p0 = r_at_min, r_at_max, axis, center + z_min * axis
    else:
        r_bot, r_top, ldir, p0 = r_at_max, r_at_min, -axis, center + z_max * axis
    r_bot = max(r_bot, 1e-3)
    rw = min(r_wire, 0.49 * pitch)                                # cap (ref: d=min(wire_d,pitch))
    angle_arg = ""
    if (r_top - r_bot) / max(r_top, 1e-9) > 0.03:                 # conical taper
        semi = math.degrees(math.atan((r_top - r_bot) / max(height, 1e-9)))
        angle_arg = f", angle={semi:.5f}"
    lh_arg = ", lefthand=True" if a_rate < 0 else ""             # handedness from theta(z) slope sign
    # rotation mapping +Z -> ldir (about Z x ldir)
    zhat = np.array([0.0, 0.0, 1.0])
    cosang = float(np.clip(np.dot(zhat, ldir), -1.0, 1.0))
    rot_deg = math.degrees(math.acos(cosang))
    rax = np.cross(zhat, ldir)
    if float(np.linalg.norm(rax)) < 1e-9:
        rax = np.array([1.0, 0.0, 0.0]); rot_deg = 0.0 if cosang > 0 else 180.0
    else:
        rax = rax / np.linalg.norm(rax)

    # PHASE alignment: makeHelix starts the coil at its surface X-dir (+X before
    # our rotate), but the detector fit has the bottom coil at angle theta_bot in
    # its own (u, v) basis.  Spin the swept spring about its axis (ldir) by the
    # signed angle that maps the render's start dir onto the fit's start dir, so
    # the coils overlap GT instead of being rotationally offset.
    def _rodrigues(vv, k, deg):
        k = k / (np.linalg.norm(k) + 1e-12); th = math.radians(deg)
        return (vv * math.cos(th) + np.cross(k, vv) * math.sin(th)
                + k * float(np.dot(k, vv)) * (1.0 - math.cos(th)))
    seed = (np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    u = np.cross(axis, seed); u /= np.linalg.norm(u) + 1e-12; v = np.cross(axis, u)
    z_bot = z_min if r_at_min <= r_at_max else z_max
    th_bot = a_rate * z_bot + float(params.get("angle_offset", 0.0))
    dir_fast = u * math.cos(th_bot) + v * math.sin(th_bot)
    dir_render = _rodrigues(np.array([1.0, 0.0, 0.0]), rax, rot_deg)
    cphi = float(np.clip(np.dot(dir_render, dir_fast), -1.0, 1.0))
    sphi = float(np.dot(np.cross(dir_render, dir_fast), ldir))
    phase_deg = math.degrees(math.atan2(sphi, cphi))

    return (
        "import cadquery as cq\n"
        f"_wire = cq.Wire.makeHelix(pitch={pitch:.5f}, height={height:.5f}, "
        f"radius={r_bot:.5f}{angle_arg}{lh_arg})\n"
        f"_spring = (cq.Workplane('XZ').center({r_bot:.5f}, 0.0).circle({rw:.5f})"
        f".sweep(cq.Workplane(obj=_wire), isFrenet=True))\n"
        f"result = (_spring"
        f".rotate((0, 0, 0), ({rax[0]:.5f}, {rax[1]:.5f}, {rax[2]:.5f}), {rot_deg:.4f})"
        f".rotate((0, 0, 0), ({ldir[0]:.5f}, {ldir[1]:.5f}, {ldir[2]:.5f}), {phase_deg:.4f})"
        f".translate(({p0[0]:.5f}, {p0[1]:.5f}, {p0[2]:.5f})))\n"
    )


def detect_helix(mesh: trimesh.Trimesh) -> list[DetectorOutput]:
    """Try PCA axis + each cardinal axis as helix candidate; emit one
    DetectorOutput per successful detection.
    """
    outs: list[DetectorOutput] = []
    axes_to_try: list[np.ndarray] = []
    pca = _pca_axis(mesh)
    if pca is not None:
        axes_to_try.append(pca)
    for i in (0, 1, 2):
        e = np.zeros(3); e[i] = 1.0
        axes_to_try.append(e)
    # Dedup by cosine.
    seen = []
    unique = []
    for a in axes_to_try:
        if any(abs(float(np.dot(a, s))) > 0.99 for s in seen):
            continue
        seen.append(a)
        unique.append(a)

    for a in unique:
        params = _detect_helix_axis(mesh, a)
        if params is None:
            continue
        fast_mesh = _build_helix_mesh(params)
        if fast_mesh is None or len(fast_mesh.faces) == 0:
            continue
        # Score: lower angular residual = higher confidence.
        score = float(max(0.0, 1.0 - 2.0 * params["angle_resid"]))
        outs.append(DetectorOutput(
            program=_emit_helix_program(params),
            score=score,
            debug={
                "detector": "helix",
                "R_helix": params["R_helix"],
                "r_wire": params["r_wire"],
                "pitch": params["pitch"],
                "angle_rate": params["angle_rate"],
                "angle_resid": params["angle_resid"],
                "n_slices_used": params["n_slices_used"],
            },
            mesh=fast_mesh,
        ))
    outs.sort(key=lambda o: o.score, reverse=True)
    return outs
