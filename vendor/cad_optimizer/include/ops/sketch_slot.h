#pragma once
#include "types.h"

namespace cadopt {

// SDF of a 2D slot (stadium/discorectangle) centered at `center` with
// half-length `half_len` along its axis, cap radius `radius`, and
// orientation `angle` (radians from X axis).
//
// Geometrically: Minkowski sum of a line segment (length 2*half_len)
// and a disk (radius `radius`).
//
// Params (5): center_x, center_y, half_len, radius, angle
// dcsg_dparams[0..4]: d(sdf)/d(cx), d(sdf)/d(cy), d(sdf)/d(half_len),
//                     d(sdf)/d(radius), d(sdf)/d(angle)
// dcsg_dp[0..2]:      d(sdf)/d(p.x), d(sdf)/d(p.y), d(sdf)/d(p.z)
double sdf_sketch_slot(double2 center, double half_len, double radius,
                       double angle, double3 p,
                       double* dcsg_dparams, double* dcsg_dp);

} // namespace cadopt
