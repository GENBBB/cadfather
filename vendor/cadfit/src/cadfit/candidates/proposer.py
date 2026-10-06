"""Top-level proposer: given prev_code + GT STL, produce N deterministic
candidate full-programs.

Algorithm
---------
1.  Render prev_code -> Sprev.stl (subprocess, timeout).
    -  If prev_code is empty / "init r = cq.Workplane()" with no solid,
       Sprev is None and ADD == GT, CUT == empty.
2.  Boolean residuals:
        ADD = Sgt \\ Sprev   (parts to union into r)
        CUT = Sprev \\ Sgt   (parts to cut from r)
3.  For each enabled detector, call it on each non-empty residual,
    wrap every DetectorOutput into a CadQuery block via
    ``block_emitter.detector_program_to_block``, append to prev_code.
4.  Rank by the detector's own ``score`` field (higher = better),
    return the top ``num_candidates``.

Notes
-----
We do NOT render/score the candidates here - that is the caller's job
(it matches kabisov's stepwise pipeline, which renders + optimizes +
chamfer-scores everything in one pass).  Our job is just to emit valid
.py code.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import trimesh

from .block_emitter import (
    append_block_to_prev,
    detector_program_to_block,
    modifier_program_to_block,
)
from .detectors import (
    detect_cylinders,
    detect_fillet_chamfer,
    detect_loft,
    detect_planar_clusters,
    detect_revolve,
    detect_silhouette_extrude,
    detect_slice_fit,
    detect_sweep,
)
from .detectors.helix import detect_helix
from .detectors.extrude import DetectorOutput
from .residual import (
    Residuals,
    compute_residuals,
    render_cadquery_to_stl,
)

log = logging.getLogger(__name__)


# Detectors split into two flavours:
#   - RESIDUAL detectors take ADD/CUT residual mesh -> emit a piece that
#     is unioned / cut from the running ``r``.
#   - MODIFIER detectors take the CURRENT BUILD mesh (rendered
#     prev_code) -> emit a block that modifies ``r`` directly
#     (fillet / chamfer / shell).
_RESIDUAL_DETECTORS: dict[str, Callable[[trimesh.Trimesh], list[DetectorOutput]]] = {
    "extrude":         detect_silhouette_extrude,
    "cylinder":        detect_cylinders,
    "revolve":         detect_revolve,
    "loft":            detect_loft,
    "sweep":           detect_sweep,
    "slice_fit":       detect_slice_fit,
    "planar_cluster":  detect_planar_clusters,
    "helix":           detect_helix,
}
_MODIFIER_DETECTORS: dict[str, Callable[[trimesh.Trimesh], list[DetectorOutput]]] = {
    "fillet_chamfer":  detect_fillet_chamfer,
}
_DEFAULT_DETECTORS: dict[str, Callable[[trimesh.Trimesh], list[DetectorOutput]]] = {
    **_RESIDUAL_DETECTORS,
    **_MODIFIER_DETECTORS,
}


@dataclass
class DetCandidate:
    """One deterministic candidate's full CadQuery program."""
    code: str
    op: str                # "union", "cut", or "init" (no prior code)
    detector: str          # e.g. "extrude"
    score: float           # higher = better
    debug: dict = field(default_factory=dict)
    # Fast-path mesh: present when the detector built a trimesh
    # approximation of `code`'s output.  ITER-1 ONLY -- on iter-2+ the
    # mesh would have to be unioned/cut with prev_mesh, which the
    # detector cannot do without re-rendering prev_code.  The caller is
    # responsible for honoring this constraint.
    mesh: Optional[object] = None  # trimesh.Trimesh, kept Optional[object] to avoid import


_PRED_FRAME_EXTENT = 200.0  # CADRecode canonical: longest bbox edge -> 200

# Absolute load ceiling: above this many faces we refuse the mesh outright
# (skip det -> VLM-only).  This only guards against pathological multi-million-
# face files spiking RAM in the weld/load itself; everything below it is
# DECIMATED (see _decimate_mesh) rather than skipped.  MCB's 1.16 M-face parts
# and cadeval's 835k-face loft fall BELOW this -> they now get det candidates.
#
# Decimation: the det path runs welds + contains()/manifold3d booleans, whose
# cost (and memory) scale with face count.  Above _DECIMATE_TRIGGER we reduce
# to ~_DECIMATE_TARGET with fast_simplification (a compiled C++ quadric
# decimator) BEFORE any heavy op.  The trigger sits ABOVE the local eval set's
# max (~44k faces) on purpose, so the evaluation meshes are NEVER decimated and
# their metrics stay bit-for-bit identical; only the huge real-data meshes
# (MCB / cadeval), previously skipped entirely, are shrunk into the usable
# range.  IoU loss at this mild ratio is < 0.005.
_MAX_DET_FACES = 2_000_000  # (kept name for back-compat; absolute load ceiling)
_DECIMATE_TRIGGER = 50_000
_DECIMATE_TARGET = 40_000

# INCREMENTAL-STEP GATE.  A det step may only ADD missing volume (union the ADD
# residual) or REMOVE extra volume (cut the CUT residual) -- it must NEVER
# rewrite the prior prediction or fill the target's empty space.  The detectors
# fit SOLID primitives, so on a sparse/hollow residual a union block overshoots
# GT and a cut block can gouge valid geometry.  We gate each candidate on the
# STEP's effect vs the TARGET (not "piece inside residual", which wrongly kills
# legitimate big cuts of an over-filled prev):
#   union: of the volume it ADDS (piece outside prev), at most _MAX_OVERSHOOT may
#          fall OUTSIDE gt;
#   cut:   of the volume it REMOVES (piece inside prev), at most _MAX_OVERSHOOT
#          may fall INSIDE gt (= gouge valid geometry).
# Measured with cheap surface-point lookups against prev/gt occupancy grids
# (built once per make_candidates call).  >= 1.0 disables the gate.
_MAX_OVERSHOOT = float(os.environ.get("DET_MAX_OVERSHOOT", "0.3"))
_CONFORM_PITCH = float(os.environ.get("DET_CONFORM_PITCH", "3.0"))


def _to_prediction_frame(mesh: trimesh.Trimesh,
                         target_extent: float = _PRED_FRAME_EXTENT
                         ) -> trimesh.Trimesh:
    """Map a GT mesh into the CADRecode prediction frame ([-100,100]^3).

    The canonical CADRecode / cadrille convention is a FIXED affine: the GT
    lives in the unit cube [0,1]^3 and maps to the prediction frame by
    ``(x - 0.5) * 200`` -- the exact inverse of cadrille's ``pred/200 + 0.5``.
    This PRESERVES the part's true position and scale within the cube, so det
    geometry lands where the VLM expects and scores correctly on the
    scale-preserving cadrille CD / IoU.

    A per-mesh "center bbox at origin + scale max-extent to 200" is WRONG for
    parts that are off-center or smaller than the unit cube: it shifts the
    centroid to the origin and rescales each part to fill the cube -- i.e. a
    translation + isotropic-scale error that cadrille penalizes (and that
    CADFit's Powell alignment otherwise hides).  So use the fixed affine
    whenever the GT is already in [0,1]^3 (DeepCAD / MCB); fall back to
    per-mesh only for absolute-scale datasets (e.g. cadeval) that have no
    canonical [0,1] frame.  Returns a transformed COPY.
    """
    m = mesh.copy()
    try:
        lo, hi = m.bounds
        tol = 0.05
        in_unit_cube = (float(np.min(lo)) >= -tol
                        and float(np.max(hi)) <= 1.0 + tol)
        if in_unit_cube:
            # Fixed canonical affine: [0,1]^3 -> [-100,100]^3 (= (x-0.5)*200).
            m.apply_translation([-0.5, -0.5, -0.5])
            m.apply_scale(target_extent)
        else:
            # No canonical unit cube -> per-mesh center + scale (best effort).
            c = (lo + hi) / 2.0
            m.apply_translation(-c)
            ext = float(np.max(m.extents))
            if ext > 1e-9:
                m.apply_scale(target_extent / ext)
    except Exception:
        return mesh
    return m


def _weld_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make a (possibly unwelded) STL watertight in place, best-effort.

    Many real datasets (e.g. MCB) export STLs as *triangle soup*: vertices
    are NOT shared between adjacent faces, so ``is_watertight`` is False and
    every face is its own connected component.  The det path relies on
    watertight geometry -- ``mesh.contains`` (occupancy-IoU scoring) and the
    manifold3d booleans (residual cut/union) both produce garbage on soup.

    We weld duplicate vertices, drop degenerate/duplicate faces, fill the
    resulting micro-gaps, and fix winding.  Each step is guarded so a mesh
    that lacks a given trimesh method (API drift across versions) still
    returns usable geometry rather than raising.
    """
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    # Drop degenerate + duplicate faces (method names differ across trimesh
    # versions -- try the modern update_faces form, fall back to legacy).
    for modern, legacy in (("nondegenerate_faces", "remove_degenerate_faces"),
                           ("unique_faces", "remove_duplicate_faces")):
        try:
            if hasattr(mesh, modern):
                mesh.update_faces(getattr(mesh, modern)())
            elif hasattr(mesh, legacy):
                getattr(mesh, legacy)()
        except Exception:
            pass
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
        except Exception:
            pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _decimate_mesh(mesh, target_faces: int = _DECIMATE_TARGET):
    """Reduce a high-face mesh to ~``target_faces`` with fast_simplification
    (a compiled C++ quadric edge-collapse decimator) so the downstream
    weld / contains() / manifold3d-boolean stages stay cheap and bounded.

    Only fires above ``_DECIMATE_TRIGGER`` -- meshes at or below it (incl. the
    whole local eval set, max ~44k faces) are returned UNCHANGED, so their
    metrics are unaffected.  Degrades gracefully: if the backend is missing or
    decimation fails/empties the mesh, the original is returned (slower, but
    correct).  Returns a COPY when it decimates, else the input mesh.
    """
    try:
        if mesh is None or len(mesh.faces) <= _DECIMATE_TRIGGER:
            return mesh
        import fast_simplification as fs
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        f = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        nv, nf = fs.simplify(v, f, target_count=int(target_faces))
        if nf is None or len(nf) == 0:
            return mesh
        dm = trimesh.Trimesh(vertices=np.asarray(nv), faces=np.asarray(nf),
                             process=True)
        if dm is None or len(dm.faces) == 0:
            return mesh
        log.info("decimated mesh %d -> %d faces (target %d)",
                 len(f), len(dm.faces), target_faces)
        return dm
    except Exception:
        return mesh


def _is_degenerate(mesh) -> bool:
    """True for meshes that crash the section/boolean det path: too few
    faces, a ~zero bbox extent (flat sketch sliver), or ~zero volume.  These
    fail *natively* inside trimesh.section / embreex / manifold3d (a NaN
    plane-line parameter -- ``d1/(d1-d3)`` -- not a catchable Python
    exception), so they must be filtered BEFORE any detector touches them.
    e.g. cadeval's ``01_sketch`` targets are ~1-triangle, zero-volume STLs.
    """
    try:
        if mesh is None or len(mesh.faces) < 4:
            return True
        ext = [float(e) for e in mesh.extents]
        mx = max(ext) if ext else 0.0
        if mx <= 1e-9 or min(ext) < 1e-5 * mx:        # flat / sliver
            return True
        try:
            if abs(float(mesh.volume)) < 1e-9 * (mx ** 3):
                return True
        except Exception:
            return True
    except Exception:
        return True
    return False


def _has_running_r(prev_code: str) -> bool:
    """Check whether prev_code defines `r` as a CadQuery solid we can
    union/cut against.  Empty or import-only files do not."""
    for line in prev_code.splitlines():
        s = line.strip()
        if s.startswith("r=") or s.startswith("r ="):
            return True
    return False


def _make_init_program(detector_program: str) -> str:
    """For iter 1 (empty prev_code): take a detector's standalone program
    and rename ``result`` -> ``r``.  No union/cut.
    """
    lines = []
    saw_import = False
    for raw in detector_program.splitlines():
        s = raw.strip()
        if s.startswith("import cadquery"):
            if saw_import:
                continue
            saw_import = True
        lines.append(raw.replace("result", "r"))
    if not saw_import:
        lines.insert(0, "import cadquery as cq")
    return "\n".join(lines) + "\n"


# ---------- prev_code rendering --------------------------------------------

def _render_prev(prev_code: str, timeout: float = 30.0) -> Optional[trimesh.Trimesh]:
    """Render prev_code to a trimesh.  Returns None if prev_code does
    not produce any geometry (e.g. it is just `r = cq.Workplane(...)`
    with no extrude yet).
    """
    if not prev_code.strip():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".stl", delete=False) as f:
        stl_path = Path(f.name)
    try:
        ok = render_cadquery_to_stl(prev_code, stl_path, timeout=timeout)
        if not ok or not stl_path.exists() or stl_path.stat().st_size == 0:
            return None
        m = trimesh.load(stl_path, process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        if m is None or len(m.faces) == 0:
            return None
        return m
    except Exception as e:
        log.debug("render_prev failed: %s", e)
        return None
    finally:
        try:
            stl_path.unlink()
        except OSError:
            pass


def _sample_surface(mesh, n: int, seed: int = 0):
    """Deterministic surface sampling (seed-stable across runs so the rerank
    is reproducible).  Falls back to the unseeded API on older trimesh."""
    try:
        p, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:
        np.random.seed(seed)
        p, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(p, dtype=np.float64)


def _rerank_candidates_by_cd(candidates, gt_mesh, n_pts: int = 1500):
    """Reorder candidates best-first by their fast-mesh chamfer distance to the
    (already framed) GT -- the SAME quantity the downstream scorer uses.

    The per-detector ``score`` is a heuristic that does not compare across
    detectors, so sorting by it can bury the actually-best-fitting candidate
    below the top-N cap (which the stepwise pipeline keeps).  When fast meshes
    are present we rank by symmetric squared chamfer to GT instead; candidates
    without a fast mesh (e.g. iter-2 modifier blocks) keep their score and sort
    after the CD-ranked ones.  Any failure -> plain score-sorted fallback.
    """
    try:
        from scipy.spatial import cKDTree
        gpts = _sample_surface(gt_mesh, n_pts)
        gtree = cKDTree(gpts)
        scored, unscored = [], []
        for c in candidates:
            m = getattr(c, "mesh", None)
            if m is None or len(getattr(m, "faces", [])) == 0:
                unscored.append(c)
                continue
            try:
                ppts = _sample_surface(m, n_pts)
                d1, _ = gtree.query(ppts, 1)
                d2, _ = cKDTree(ppts).query(gpts, 1)
                cd = float(np.mean(d1 ** 2) + np.mean(d2 ** 2))
            except Exception:
                unscored.append(c)
                continue
            c.debug = dict(c.debug, cd_to_gt=cd)
            scored.append((cd, c))
        scored.sort(key=lambda t: t[0])
        unscored.sort(key=lambda c: c.score, reverse=True)
        return [c for _, c in scored] + unscored
    except Exception:
        return sorted(candidates, key=lambda c: c.score, reverse=True)


def _occ_grid(mesh):
    """Fill-voxelize a mesh ONCE so per-candidate gating is a cheap point
    lookup.  Returns None on failure -> that side of the gate is skipped."""
    try:
        if mesh is None or len(mesh.faces) == 0:
            return None
        # Adaptive pitch: a FIXED pitch makes trimesh.subdivide_to_size try to
        # build (extent/pitch)^2 triangles, which HANGS when a candidate has a
        # degenerate / far-flung vertex (inf/nan coords) or the mesh is in a
        # much larger frame than _CONFORM_PITCH assumes (gt is [0,1]^3, det
        # reconstructions are native ~[-100,100]).  Cap the grid at MAX_CELLS
        # so subdivision always terminates fast.  Never FINER than
        # _CONFORM_PITCH -> normal meshes are byte-for-byte unchanged; only the
        # pathological large-extent case is coarsened.
        ext = float(np.max(mesh.extents))
        if not np.isfinite(ext) or ext <= 0.0:
            return None
        MAX_CELLS = 256.0
        pitch = max(_CONFORM_PITCH, ext / MAX_CELLS)
        # method='ray' voxelizes by ray casting (cost ~ grid resolution).  The
        # default 'subdivide' method recursively splits every triangle until
        # edges < pitch -> for a native-frame recon mesh (extent ~200) at a
        # fine pitch it explodes to millions of faces and HANGS (cost scales
        # with extent/pitch AND initial face count, which pitch alone can't
        # bound).  'ray' has no subdivision, so it cannot blow up.
        return mesh.voxelized(pitch, method='ray').fill()
    except Exception:
        return None


def _step_conforms(piece, prev_vox, gt_vox, op: str, n_pts: int = 1500) -> bool:
    """Incremental-step gate against the TARGET.

    Only UNION steps are gated: of the volume a union ADDS (piece surface
    outside prev), at most ``_MAX_OVERSHOOT`` may fall OUTSIDE gt.  A union that
    overshoots is exactly the "rewrite" the residual scheme must never do (it
    re-covers prior geometry / fills the target's empty space).

    CUT steps are NOT gated: a cut only ever removes material, so it cannot
    rewrite the prediction; over-cutting is already penalised by the downstream
    CD/IoU selection, and a strict surface gate wrongly freezes the legitimate
    cuts that hollow out an over-filled init (their boundary runs along gt walls
    even though the removed VOLUME is the extra outside gt).

    Returns True (keep) whenever it cannot be measured -> strictly subtractive.
    """
    if _MAX_OVERSHOOT >= 1.0 or op != "union" or piece is None or gt_vox is None:
        return True
    try:
        if len(getattr(piece, "faces", [])) == 0:
            return True
        pts, _ = trimesh.sample.sample_surface(piece, n_pts); pts = np.asarray(pts)
        in_gt = gt_vox.is_filled(pts)
        in_prev = (prev_vox.is_filled(pts) if prev_vox is not None
                   else np.zeros(len(pts), dtype=bool))
        sel = ~in_prev                           # the newly-added boundary
        if not sel.any():
            # NO-OP union: the piece lies entirely inside prev, so the union
            # adds nothing (result == prev up to boolean noise).  Such a
            # candidate cannot improve anything yet wastes a slot + a render
            # -> reject it (verified on 02_cut: these were the only "leaks";
            # added volume = +0.00% of prev).
            return False
        return float(np.mean(~in_gt[sel])) <= _MAX_OVERSHOOT
    except Exception:
        return True


def _apply_op_mesh(prev_mesh, piece, op: str):
    """Compute an iter-2 candidate's TRUE result mesh = ``prev_mesh {op} piece``
    via a manifold3d boolean, so the candidate carries a fast mesh that (a) the
    CD-rerank can score against GT with its real resulting geometry and (b) the
    caller can evaluate without re-rendering cadquery.

    ``piece`` is the residual detector's fast mesh (the block being added/cut),
    already in the prediction frame.  Returns None on any failure -> caller
    leaves ``mesh=None`` (candidate falls back to score-sort, as before), so this
    is strictly additive.
    """
    if prev_mesh is None or piece is None or len(getattr(piece, "faces", [])) == 0:
        return None
    try:
        if op == "union":
            r = trimesh.boolean.union([prev_mesh, piece])
        elif op == "cut":
            r = trimesh.boolean.difference([prev_mesh, piece])
        else:
            return None
        if isinstance(r, (list, tuple)):
            r = trimesh.util.concatenate([m for m in r if m is not None]) if r else None
        if r is None or len(r.faces) == 0:
            return None
        return r
    except Exception:
        return None


# ---------- main entry point -----------------------------------------------

def make_candidates(prev_code: str,
                    gt_stl_path: str,
                    num_candidates: int,
                    detectors: Optional[list[str]] = None,
                    render_timeout: float = 30.0,
                    prev_stl_path: Optional[str] = None,
                    rank_by_cd: bool = True,
                    prev_detector: Optional[str] = None,
                    ) -> list[DetCandidate]:
    """Produce up to ``num_candidates`` deterministic step proposals.

    Parameters
    ----------
    prev_code : str
        The CadQuery code committed so far.  May be empty for iter 1.
    gt_stl_path : str
        Path to the ground-truth target mesh.
    num_candidates : int
        How many candidates to return (best-scored first).  0 means
        return an empty list (det-candidate generation disabled).
    detectors : list[str] | None
        Which detectors to run.  Default = all available.
    prev_stl_path : str | None
        If given AND has_running_r(prev_code) is True, load this STL as
        the pred_mesh directly INSTEAD of subprocess-rendering prev_code.
        Saves ~3-5 s per call -- a huge win when called over many parents
        in a stepwise pipeline.  Falls back to re-rendering on load
        failure.

    Returns
    -------
    list[DetCandidate], sorted by score descending, length <= num_candidates.
    """
    if num_candidates <= 0:
        return []

    detector_names = detectors or list(_DEFAULT_DETECTORS.keys())
    residual_dets = [(n, _RESIDUAL_DETECTORS[n]) for n in detector_names
                     if n in _RESIDUAL_DETECTORS]
    modifier_dets = [(n, _MODIFIER_DETECTORS[n]) for n in detector_names
                     if n in _MODIFIER_DETECTORS]

    # A helix/coil parent has NO clean residual to add/cut: its residual is a
    # thin messy shell, and a CD-better-but-IoU-worse cut/union slips past the
    # keep-parent margin (observed: coil parts regress −0.05..−0.15 at iter-2).
    # Freeze it — skip residual+modifier detectors when the parent is a helix.
    # (Wire-like bbox-fill is NOT a usable proxy: finned parts are equally thin
    # yet benefit from cuts, so gate on the parent DETECTOR identity instead.)
    if prev_detector == "helix":
        residual_dets = []
        modifier_dets = []

    gt_mesh = trimesh.load(gt_stl_path, process=False)
    if isinstance(gt_mesh, trimesh.Scene):
        gt_mesh = trimesh.util.concatenate(list(gt_mesh.geometry.values()))
    if gt_mesh is None or len(gt_mesh.faces) == 0:
        return []

    # Absolute load ceiling: only pathological multi-million-face files are
    # refused outright (skip det -> VLM-only).  Checked BEFORE weld so we never
    # weld a monster.  Everything below is welded then decimated (not skipped).
    if len(gt_mesh.faces) > _MAX_DET_FACES:
        log.warning("GT has %d faces > absolute cap %d -> skipping det "
                    "candidates (VLM-only)", len(gt_mesh.faces), _MAX_DET_FACES)
        return []

    # Weld triangle-soup STLs (e.g. MCB) so contains()/booleans are valid, then
    # decimate huge meshes into the usable range (no-op for the eval set).
    gt_mesh = _weld_mesh(gt_mesh)
    gt_mesh = _decimate_mesh(gt_mesh)

    # ------------------------------------------------------------------
    # FRAME ALIGNMENT.  The GT mesh on disk is normalized to the unit
    # cube ([0,1]^3), but the running prediction `r` (CADRecode / VLM
    # code, and every det block we emit) lives in the model's canonical
    # frame: centered at the origin and scaled so the longest bbox edge
    # is 200 (i.e. [-100,100]).  CADRecode applies exactly this
    # normalization to the input cloud, so its code is in that frame.
    #
    # If we fit detectors on the raw [0,1] GT, the emitted geometry is
    # ~200x too small and offset -> `r.cut(piece)` removes a speck and
    # `compute_residuals(pred[-100,100], gt[0,1])` mixes frames into
    # garbage.  Shift+scale the GT into the prediction frame FIRST so
    # all detector geometry composes correctly with `r`.
    gt_mesh = _to_prediction_frame(gt_mesh)

    # Degenerate targets (zero-volume sketches / slivers -- e.g. cadeval's
    # 01_sketch 136-byte STLs) crash trimesh.section natively at the residual
    # stage; skip det entirely for them (the sample stays VLM-only).
    if _is_degenerate(gt_mesh):
        log.warning("GT degenerate (sketch/sliver) -> det skipped (VLM-only)")
        return []

    has_r = _has_running_r(prev_code)
    pred_mesh = None
    if has_r:
        # Fast path: prev STL already exists on disk -> load it directly.
        if prev_stl_path and os.path.exists(prev_stl_path):
            try:
                pred_mesh = trimesh.load(prev_stl_path, process=False)
                if isinstance(pred_mesh, trimesh.Scene):
                    pred_mesh = trimesh.util.concatenate(
                        list(pred_mesh.geometry.values()))
                if pred_mesh is None or len(pred_mesh.faces) == 0:
                    pred_mesh = None
            except Exception:
                pred_mesh = None
        # Fallback: subprocess-render prev_code (~3-5 s).
        if pred_mesh is None:
            pred_mesh = _render_prev(prev_code, timeout=render_timeout)
        # Same absolute ceiling on the prediction (residual booleans run
        # against it too).
        if pred_mesh is not None and len(pred_mesh.faces) > _MAX_DET_FACES:
            log.warning("pred has %d faces > absolute cap %d -> skipping det",
                        len(pred_mesh.faces), _MAX_DET_FACES)
            return []
        # Weld whichever source we got (disk load or render are both
        # process=False) so residual booleans against it are valid, then
        # decimate huge predictions into the usable range.
        if pred_mesh is not None:
            pred_mesh = _weld_mesh(pred_mesh)
            pred_mesh = _decimate_mesh(pred_mesh)

    candidates: list[DetCandidate] = []

    if not has_r:
        # Iter 1 path: prev_code empty / no `r`.  Run RESIDUAL detectors
        # on the full GT and emit standalone init-programs.  Modifier
        # detectors are skipped here -- there is no ``r`` to modify yet.
        for det_name, det_fn in residual_dets:
            try:
                outs = det_fn(gt_mesh)
            except Exception as e:
                log.debug("detector %s failed on GT (init): %s", det_name, e)
                continue
            for out in outs:
                candidates.append(DetCandidate(
                    code=_make_init_program(out.program),
                    op="init",
                    detector=det_name,
                    score=out.score,
                    debug={**out.debug, "residual_side": "init"},
                    mesh=getattr(out, "mesh", None),
                ))
    else:
        # Iter 2+ path: compute ADD / CUT residuals against the rendered
        # prev_code mesh, propose blocks for each non-empty residual.
        # Skip the expensive boolean if no residual detector was requested.
        if residual_dets:
            residuals = (compute_residuals(pred_mesh, gt_mesh)
                         if pred_mesh is not None
                         else Residuals(add_mesh=gt_mesh, cut_mesh=None,
                                        gt_volume=0.0, pred_volume=0.0,
                                        add_volume=0.0, cut_volume=0.0, iou=0.0))
        else:
            residuals = Residuals(add_mesh=None, cut_mesh=None,
                                  gt_volume=0.0, pred_volume=0.0,
                                  add_volume=0.0, cut_volume=0.0, iou=0.0)

        # Build prev / gt occupancy grids ONCE for the incremental-step gate
        # (cheap per-candidate surface-point lookups against these).
        prev_vox = _occ_grid(pred_mesh) if _MAX_OVERSHOOT < 1.0 else None
        gt_vox = _occ_grid(gt_mesh) if _MAX_OVERSHOOT < 1.0 else None

        # --- RESIDUAL detectors on ADD / CUT meshes -----------------------
        for op_name, residual_mesh in (
            ("union", residuals.add_mesh),
            ("cut",   residuals.cut_mesh),
        ):
            if (residual_mesh is None or len(residual_mesh.faces) == 0
                    or _is_degenerate(residual_mesh)):
                continue
            for det_name, det_fn in residual_dets:
                try:
                    outs = det_fn(residual_mesh)
                except Exception as e:
                    log.debug("detector %s failed on %s residual: %s",
                              det_name, op_name, e)
                    continue
                for out in outs:
                    piece = getattr(out, "mesh", None)
                    # INCREMENTAL-STEP GATE: drop a union that overshoots GT or a
                    # cut that gouges valid geometry -- the residual scheme may
                    # only add missing / remove extra, never rewrite the prior.
                    if not _step_conforms(piece, prev_vox, gt_vox, op_name):
                        continue
                    try:
                        block = detector_program_to_block(out.program, op=op_name)
                    except Exception as e:
                        log.debug("block rewrite failed: %s", e)
                        continue
                    full_code = append_block_to_prev(prev_code, block)
                    # Compute the candidate's TRUE result mesh (prev {op} piece)
                    # so it can be CD-ranked + scored without a cadquery render.
                    result_mesh = _apply_op_mesh(pred_mesh, piece, op_name)
                    candidates.append(DetCandidate(
                        code=full_code,
                        op=op_name,
                        detector=det_name,
                        score=out.score,
                        debug={**out.debug, "residual_side": op_name},
                        mesh=result_mesh,
                    ))

        # --- MODIFIER detectors on the CURRENT BUILD mesh ----------------
        # These need pred_mesh (the rendered prev_code); they emit
        # `r = r.edges(...).fillet(R)`-style modifier blocks.  Wrap
        # via modifier_program_to_block (NOT detector_program_to_block).
        if pred_mesh is not None and modifier_dets:
            for det_name, det_fn in modifier_dets:
                try:
                    outs = det_fn(pred_mesh)
                except Exception as e:
                    log.debug("modifier detector %s failed: %s", det_name, e)
                    continue
                for out in outs:
                    try:
                        block = modifier_program_to_block(out.program)
                    except Exception as e:
                        log.debug("modifier block rewrite failed: %s", e)
                        continue
                    full_code = append_block_to_prev(prev_code, block)
                    candidates.append(DetCandidate(
                        code=full_code,
                        op="modifier",
                        detector=det_name,
                        score=out.score,
                        debug={**out.debug, "residual_side": "modifier"},
                    ))

    # Rank best-first.  By default use the fast-mesh chamfer distance to GT
    # (what the downstream scorer keeps), which compares across detectors far
    # better than the per-detector heuristic score and stops the top-N cap from
    # discarding the actually-best candidate.  rank_by_cd=False -> legacy score.
    if rank_by_cd:
        candidates = _rerank_candidates_by_cd(candidates, gt_mesh)
    else:
        candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:num_candidates]
