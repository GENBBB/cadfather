#pragma once
#include "types.h"

namespace cadopt {

// SDF of a 2D axis-aligned rectangle centered at `center` with half-extents `half_size`.
// Evaluated in XY plane (p.z ignored for the 2D SDF).
//
// Params (4): center_x, center_y, half_w, half_h
// dcsg_dparams[0..3]: d(sdf)/d(center_x), d(sdf)/d(center_y), d(sdf)/d(half_w), d(sdf)/d(half_h)
// dcsg_dp[0..2]:      d(sdf)/d(p.x), d(sdf)/d(p.y), d(sdf)/d(p.z)
double sdf_sketch_rect(double2 center, double2 half_size, double3 p,
                       double* dcsg_dparams, double* dcsg_dp);

} // namespace cadopt
