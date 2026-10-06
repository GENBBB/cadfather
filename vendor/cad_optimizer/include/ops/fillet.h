#pragma once
#include "types.h"

namespace cadopt {

// Fillet (smooth union/intersection): replaces a sharp boolean with a rounded blend.
// Uses polynomial smooth-min with radius k.
//
// Smooth union: smin(d1, d2, k) = mix(d2, d1, h) - k*h*(1-h)
//   where h = clamp(0.5 + 0.5*(d2-d1)/k, 0, 1)
//
// Smooth intersection: smax(d1, d2, k) = -smin(-d1, -d2, k)
//
// Params (1): radius k
//
// Gradient:
//   When 0 < h < 1 (in the blend zone):
//     dh/d(d1) = -0.5/k, dh/d(d2) = 0.5/k, dh/dk = -0.5*(d2-d1)/k^2
//     dsdf/d(d1) = (1-h) + (d1-d2) * dh/d(d1) - k*(1-2h)*dh/d(d1)
//     dsdf/d(d2) = h + (d1-d2) * dh/d(d2) - k*(1-2h)*dh/d(d2)
//     dsdf/dk = (d1-d2)*dh/dk - h*(1-h) - k*(1-2h)*dh/dk
//   When h = 0: sdf = d2, derivatives pass through to d2
//   When h = 1: sdf = d1, derivatives pass through to d1

// Returns the smooth-min of d1 and d2 with radius k.
// Writes d(smin)/d(d1), d(smin)/d(d2), d(smin)/d(k) into the output pointers.
double smooth_min(double d1, double d2, double k,
                  double* dsmin_dd1, double* dsmin_dd2, double* dsmin_dk);

// Same but smooth-max (for smooth intersection)
double smooth_max(double d1, double d2, double k,
                  double* dsmax_dd1, double* dsmax_dd2, double* dsmax_dk);

} // namespace cadopt
