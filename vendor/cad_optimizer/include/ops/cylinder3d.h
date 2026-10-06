#pragma once
#include "types.h"

namespace cadopt {

// Cylinder3D: 5 params (cx, cy, cz, radius, half_height)
// Z-axis aligned cylinder centered at (cx, cy, cz).
//
// SDF:
//   r_xy = sqrt((p.x-cx)^2 + (p.y-cy)^2) - radius
//   h = |p.z - cz| - half_height
//   sdf = length(max((r_xy, h), 0)) + min(max(r_xy, h), 0)
//
// Multiple regions: corner, tube, cap, interior.
double sdf_cylinder3d(double3 center, double radius, double half_height,
                      double3 p, double* dparams, double* dp);

} // namespace cadopt
