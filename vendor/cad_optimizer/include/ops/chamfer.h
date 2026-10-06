#pragma once
#include "types.h"

namespace cadopt {

// Chamfer: replaces a sharp boolean intersection edge with a 45-degree cut.
// chamfer_intersect(d1, d2, size) = max(d1, d2, (d1 + d2)*sqrt(0.5) - size)
//
// This creates a flat 45-degree bevel at the intersection edge.
//
// Params (1): size (chamfer distance)
//
// Gradient:
//   Let d3 = (d1 + d2) * sqrt(0.5) - size
//   result = max(d1, d2, d3)
//   Derivatives follow the active branch (whichever of d1, d2, d3 is max).
double chamfer_intersect(double d1, double d2, double size,
                         double* dcham_dd1, double* dcham_dd2, double* dcham_dsize);

} // namespace cadopt
