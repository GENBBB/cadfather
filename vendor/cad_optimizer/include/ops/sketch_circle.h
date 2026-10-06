#pragma once
#include "types.h"

namespace cadopt {

// SDF of a 2D circle centered at `center` with given `radius`.
// Evaluated in XY plane (p.z ignored for the 2D SDF).
//
// Params (3): center_x, center_y, radius
// dcsg_dparams[0..2]: d(sdf)/d(cx), d(sdf)/d(cy), d(sdf)/d(radius)
// dcsg_dp[0..2]:      d(sdf)/d(p.x), d(sdf)/d(p.y), d(sdf)/d(p.z)
double sdf_sketch_circle(double2 center, double radius, double3 p,
                         double* dcsg_dparams, double* dcsg_dp);

} // namespace cadopt
