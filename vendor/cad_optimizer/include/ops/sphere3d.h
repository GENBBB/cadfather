#pragma once
#include "types.h"

namespace cadopt {

// Sphere: 4 params (cx, cy, cz, radius)
// SDF = length(p - center) - radius
//
// Gradient:
//   n = normalize(p - center)
//   d(sdf)/d(cx) = -n.x,  d(sdf)/d(cy) = -n.y,  d(sdf)/d(cz) = -n.z
//   d(sdf)/d(radius) = -1
//   d(sdf)/d(p) = n
double sdf_sphere3d(double3 center, double radius, double3 p,
                    double* dparams, double* dp);

} // namespace cadopt
