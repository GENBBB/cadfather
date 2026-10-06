#pragma once
#include "types.h"

namespace cadopt {

// Revolve: rotates a 2D profile (in XY plane) around the Y axis to produce a solid.
// For a 3D query point p, we compute r = sqrt(p.x^2 + p.z^2) and evaluate the
// 2D child SDF at (r, p.y, 0).
//
// For partial revolution (angle < 2*PI), we intersect with a wedge.
// Full revolution (angle = 2*PI) is the common case.
//
// Params (1): angle (radians, 0 to 2*PI)
//
// Gradient derivation:
//   Let r = sqrt(x^2 + z^2), p' = (r, y, 0)
//   sdf = child(p')
//   d(sdf)/d(x) = d(child)/d(p'.x) * d(r)/d(x) = dc/dr * x/r
//   d(sdf)/d(y) = d(child)/d(p'.y)
//   d(sdf)/d(z) = d(child)/d(p'.x) * d(r)/d(z) = dc/dr * z/r
//   d(sdf)/d(child_params) propagates directly since p' is independent of child params.

} // namespace cadopt
