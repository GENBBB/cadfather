"""Section analysis — CADFit-style.

Slice a mesh perpendicular to a chosen axis at N equally-spaced heights,
extract the 2D contour per section, fit primitives via the C++
section_fit submodule, then classify the trajectory across sections to
suggest the right CadQuery op (extrude / loft / sweep / revolve).

The actual op-emission lives in detector modules (``loft.py``,
``sweep.py``, ``revolve.py``); this file just builds the section data
and the per-axis trajectory hint.

The C++ fitter (`_cad_grad.section.fit_all`) is loaded lazily — if the
build is missing we fall back to ``None`` and the caller can decide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh


# ---------------------------- C++ fitter loader ---------------------------

def _load_cad_grad():
    """C++ module of det (`cadfit._native`: `cluster_normals` + `section`).

    A miss is never silent: without the module `planar_cluster` falls back to the
    Python branch and the `section` detectors return `[]`, so det output changes.
    Hence a warning is always emitted, and with `CADFIT_REQUIRE_NATIVE=1` an
    ImportError is raised. State: `cadfit.native_status()`.
    """
    import os
    import warnings
    try:
        from .. import _native
        return _native
    except Exception as exc:
        if os.environ.get("CADFIT_REQUIRE_NATIVE") == "1":
            raise ImportError(
                "cadfit._native cannot be imported, and CADFIT_REQUIRE_NATIVE=1; "
                "build it with: native/build.sh <python>") from exc
        warnings.warn(
            f"cadfit._native cannot be imported ({type(exc).__name__}: {exc}) — det "
            "runs without C++: section detectors are empty, clustering is slower. "
            "Build it with: native/build.sh <python>", RuntimeWarning, stacklevel=2)
        return None


_cg = _load_cad_grad()
_HAS_CPP_FITS = _cg is not None and hasattr(_cg, "section")


# ----------------------------- Data classes -------------------------------

@dataclass
class PrimitiveFitRecord:
    """Plain-Python mirror of cadopt::section::PrimitiveFit, easier to pickle."""
    kind: str           # "LINE" / "ARC" / "CIRCLE" / "RECT" / "POLYGON"
    params: list[float]
    residual: float
    support_frac: float
    score: float


@dataclass
class Section:
    z: float                                    # height along axis_idx
    axis_idx: int                               # 0=X, 1=Y, 2=Z
    contour: np.ndarray                         # (N, 2) points in 2D
    fits: list[PrimitiveFitRecord]              # sorted by score (best first)

    @property
    def best_kind(self) -> str:
        return self.fits[0].kind if self.fits else "NONE"

    @property
    def best_params(self) -> list[float]:
        return list(self.fits[0].params) if self.fits else []


@dataclass
class TrajectoryHint:
    op: str                                     # "extrude" / "loft" / "sweep" / "general"
    confidence: float                           # in [0, 1]
    axis_idx: int
    sections: list[Section]
    debug: dict = field(default_factory=dict)


# ----------------------------- Slicing ------------------------------------

def _slice_contour(mesh: trimesh.Trimesh, axis_idx: int, z: float
                   ) -> Optional[np.ndarray]:
    """Return the largest 2D contour at height z along axis_idx, or None."""
    normal = [0.0, 0.0, 0.0]
    normal[axis_idx] = 1.0
    origin = [0.0, 0.0, 0.0]
    origin[axis_idx] = z
    try:
        sec3d = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec3d is None:
            return None
        # trimesh 4.x renamed Path3D.to_2D -> to_planar; keep fallback for 3.x.
        if hasattr(sec3d, "to_planar"):
            p2d, _ = sec3d.to_planar()
        elif hasattr(sec3d, "to_2D"):
            p2d, _ = sec3d.to_2D()
        else:
            return None
        if p2d is None or not p2d.polygons_full:
            return None
        # Pick the largest polygon (matches our multi-component extrude convention).
        poly = max(p2d.polygons_full, key=lambda g: g.area)
        coords = np.asarray(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return None
        return coords.astype(np.float64)
    except Exception:
        return None


def _fits_for_contour(contour: np.ndarray, tol: float = -1.0
                      ) -> list[PrimitiveFitRecord]:
    """Run the C++ fit_all on a (N, 2) contour and convert to Python records."""
    if not _HAS_CPP_FITS:
        return []
    raw = _cg.section.fit_all(contour, tol)
    recs = []
    for f in raw:
        recs.append(PrimitiveFitRecord(
            kind=str(f.kind).split(".")[-1],
            params=list(f.params),
            residual=float(f.residual),
            support_frac=float(f.support_frac),
            score=float(f.score),
        ))
    return recs


def extract_sections(mesh: trimesh.Trimesh, axis_idx: int,
                     n_sections: int = 8,
                     margin_frac: float = 0.05,
                     ) -> list[Section]:
    """Slice the mesh into ``n_sections`` cross-sections perpendicular to
    ``axis_idx``, fit primitives on each contour.  Returns only sections
    that produced a valid contour.
    """
    lo, hi = mesh.bounds
    span = hi[axis_idx] - lo[axis_idx]
    if span < 1e-9:
        return []
    margin = span * margin_frac
    zs = np.linspace(lo[axis_idx] + margin, hi[axis_idx] - margin, n_sections)
    secs: list[Section] = []
    for z in zs:
        c = _slice_contour(mesh, axis_idx, float(z))
        if c is None:
            continue
        secs.append(Section(
            z=float(z),
            axis_idx=axis_idx,
            contour=c,
            fits=_fits_for_contour(c),
        ))
    return secs


# --------------------------- Trajectory classifier ------------------------

def _params_close(a: list[float], b: list[float], rtol: float = 0.1) -> bool:
    """Are two parameter vectors close (same length, near-equal values)?"""
    if len(a) != len(b):
        return False
    ar = np.asarray(a)
    br = np.asarray(b)
    denom = np.maximum(np.abs(ar), np.abs(br))
    denom = np.where(denom < 1e-9, 1.0, denom)
    return bool(np.all(np.abs(ar - br) / denom < rtol))


def classify_trajectory(sections: list[Section],
                        scale_rtol: float = 0.1,
                        ) -> TrajectoryHint:
    """Look at how the best primitive evolves across sections.

    Heuristics for v1:
      - All sections share the same best_kind AND have near-equal params
            → ``extrude`` (constant cross-section)
      - All sections share the same best_kind, params differ
            → ``loft`` (will need richer scoring in v2)
      - Otherwise / not enough valid sections
            → ``general`` (fall back to silhouette extrude)
    """
    if not sections or len(sections) < 2:
        return TrajectoryHint("general", 0.0, axis_idx=-1, sections=sections)
    axis_idx = sections[0].axis_idx
    kinds = [s.best_kind for s in sections]
    same_kind = len(set(kinds)) == 1
    if not same_kind:
        return TrajectoryHint("general", 0.3, axis_idx=axis_idx,
                              sections=sections,
                              debug={"kinds": kinds})

    p0 = sections[0].best_params
    all_close = all(_params_close(p0, s.best_params, scale_rtol)
                    for s in sections[1:])
    if all_close:
        return TrajectoryHint("extrude", 0.9, axis_idx=axis_idx,
                              sections=sections,
                              debug={"kind": kinds[0]})
    # Same kind but params drift → loft candidate.
    return TrajectoryHint("loft", 0.6, axis_idx=axis_idx,
                          sections=sections,
                          debug={"kind": kinds[0]})
