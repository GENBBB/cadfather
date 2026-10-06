#pragma once
#include "types.h"

namespace cadopt {

// Polygon (closed polyline): 2D signed distance to a closed polygon.
// Params: N vertices, each (x, y) = 2N params total.
// SDF: min distance to any edge, sign from winding number.
//
// Gradient: only the two vertices of the closest edge have non-zero
// gradients. Sign from winding number is piecewise constant (grad = 0).
//
// Line segment distance helper:
// Returns UNSIGNED distance from 2D point to line segment (a, b),
// plus gradients w.r.t. a, b, and p.
double dist_line_segment(double2 a, double2 b, double2 p,
                         double2* dd_da, double2* dd_db, double2* dd_dp);

// Returns signed distance from 2D point to closed polygon defined by
// vertices[0..n-1]. Writes gradients into dparams[0..2n-1] and dp[0..1].
double sdf_polygon(const double2* vertices, int n, double2 p,
                   double* dparams, double* dp);

} // namespace cadopt
