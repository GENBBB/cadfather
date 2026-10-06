// section_fit.h — 2D primitive fits for CADFit-style section analysis.
//
// The Python side (det_candidates.section_analyzer) slices a residual
// mesh perpendicular to a chosen axis, extracts a 2D contour per
// section, and calls fit_all() to identify which primitive (line, arc,
// circle, oriented-rect, polygon) best explains that contour.
//
// The trajectory of fits across sections then tells the high-level
// classifier which CadQuery op to emit:
//     same primitive + same params  every section  -> extrude
//     same primitive + scaled params               -> loft
//     same primitive + centroid follows curve      -> sweep
//     (r,z) profile across axial sections          -> revolve
//
// All routines take a contiguous double2 array of contour points and
// return a `PrimitiveFit` with `residual` (mean perpendicular distance
// from input points to the fitted curve), `support_frac` (fraction of
// points within `tol`), and `score` (lower = better -- residual-aware
// composite).
//
// No external dependencies (no Eigen) -- linear algebra is inline 3x3.
#pragma once
#include "types.h"
#include <cstddef>
#include <vector>

namespace cadopt::section {

enum class PrimitiveKind : int {
  LINE    = 0,
  ARC     = 1,
  CIRCLE  = 2,
  RECT    = 3,
  POLYGON = 4,
};

struct PrimitiveFit {
  PrimitiveKind kind = PrimitiveKind::POLYGON;

  // Params layout per kind:
  //   LINE:    [nx, ny, c]                                     (n·p = c)
  //   CIRCLE:  [cx, cy, r]
  //   ARC:     [cx, cy, r, theta_min, theta_max]                (radians)
  //   RECT:    [cx, cy, w, h, theta]                            (half-widths w/2 h/2)
  //   POLYGON: flattened [x0,y0,x1,y1,...] simplified contour
  std::vector<double> params;

  double residual     = 0.0;   // mean abs perpendicular distance, fit curve <- pts
  double support_frac = 0.0;   // fraction with |d| < tol
  double score        = 0.0;   // residual + outlier penalty (lower = better)
};

// Single-primitive fits.
PrimitiveFit fit_line   (const double2* pts, int n);
PrimitiveFit fit_circle (const double2* pts, int n);
PrimitiveFit fit_arc    (const double2* pts, int n);
PrimitiveFit fit_rect   (const double2* pts, int n);
PrimitiveFit fit_polygon(const double2* pts, int n, double simplify_tol);

// Try every primitive; return all fits sorted by `score` ascending
// (best first).  ``tol`` is the inlier threshold used to compute
// `support_frac`.  Pass tol < 0 to auto-pick = 1% of the contour's
// bbox diagonal.
std::vector<PrimitiveFit> fit_all(const double2* pts, int n, double tol);

} // namespace cadopt::section
