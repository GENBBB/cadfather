#pragma once
#include "types.h"

namespace cadopt {

// Box3D: 6 params (cx, cy, cz, hx, hy, hz)
// SDF = standard 3D box distance field
//   d = abs(p - center) - half_size
//   sdf = length(max(d, 0)) + min(max(d.x, d.y, d.z), 0)
//
// Multiple regions: corner (3 positive), edge (2 positive),
// face (1 positive), interior (0 positive).
double sdf_box3d(double3 center, double3 half_size, double3 p,
                 double* dparams, double* dp);

} // namespace cadopt
