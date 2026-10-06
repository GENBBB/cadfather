#pragma once
#include "types.h"

namespace cadopt {

// SDF of extrusion: extends a 2D SDF symmetrically along one axis by `depth` (half-extent).
// ICAD convention: sketch in XZ plane, extrude along Y.
// CadQuery convention: sketch in XY plane, extrude along Z.
// We use CadQuery convention: the 2D SDF is evaluated in XY, extrude along Z.
//
// Params (1): depth (half-extent, so full extrusion height = 2*depth)
// dcsg_dparams[0]:  d(sdf)/d(depth)
// dcsg_dp[0..2]:    d(sdf)/d(p.x), d(sdf)/d(p.y), d(sdf)/d(p.z)
// dcsg_d_child:     d(sdf)/d(child_sdf) -- for chain rule propagation
double sdf_extrude(double depth, double3 p, double child_sdf,
                   double* dcsg_dparams, double* dcsg_dp, double* dcsg_d_child);

} // namespace cadopt
