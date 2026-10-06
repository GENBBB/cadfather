#pragma once
#include "types.h"

namespace cadopt {

// Shell: hollows out a solid by removing everything except a thin wall
// of given thickness around the surface.
//
// sdf_shell(p) = abs(child_sdf(p)) - thickness/2
//
// Points at distance < thickness/2 from the original surface are inside the shell.
//
// Params (1): thickness
//
// Gradient:
//   d(shell)/d(thickness) = -0.5
//   d(shell)/d(child_params) = sign(child_sdf) * d(child_sdf)/d(child_params)
//   d(shell)/d(p) = sign(child_sdf) * d(child_sdf)/d(p)

} // namespace cadopt
