#pragma once
#include "types.h"

namespace cadopt {

// Translate3D: shift = (sx, sy, sz), 3 params.
// SDF: child->sdf(p - shift)
// d/d(shift_i) = -d(child)/d(p_i)
// d/d(p_i) = d(child)/d(p_i)
// Implemented directly in node class.

} // namespace cadopt
