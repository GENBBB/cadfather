#include "ops/fillet.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

double smooth_min(double d1, double d2, double k,
                  double* dsmin_dd1, double* dsmin_dd2, double* dsmin_dk) {
  // h = clamp(0.5 + 0.5*(d2-d1)/k, 0, 1)
  double raw_h = 0.5 + 0.5 * (d2 - d1) / k;
  double h = std::clamp(raw_h, 0.0, 1.0);

  // sdf = mix(d2, d1, h) - k*h*(1-h) = d2*(1-h) + d1*h - k*h*(1-h)
  double sdf = d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h);

  if (raw_h <= 0.0) {
    // h clamped to 0: sdf = d2
    *dsmin_dd1 = 0.0;
    *dsmin_dd2 = 1.0;
    *dsmin_dk = 0.0;
  } else if (raw_h >= 1.0) {
    // h clamped to 1: sdf = d1
    *dsmin_dd1 = 1.0;
    *dsmin_dd2 = 0.0;
    *dsmin_dk = 0.0;
  } else {
    // In the blend zone: h is a function of d1, d2, k
    double dh_dd1 = -0.5 / k;
    double dh_dd2 =  0.5 / k;
    double dh_dk  = -0.5 * (d2 - d1) / (k * k);

    // dsdf/d(d1) = h + (d1 - d2) * dh_dd1 - k*(1 - 2h)*dh_dd1
    // dsdf/d(d2) = (1-h) + (d1 - d2) * dh_dd2 - k*(1 - 2h)*dh_dd2
    // dsdf/dk = (d1 - d2)*dh_dk - h*(1-h) - k*(1 - 2h)*dh_dk
    double diff = d1 - d2;
    double blend_term = k * (1.0 - 2.0 * h);

    *dsmin_dd1 = h + diff * dh_dd1 - blend_term * dh_dd1;
    *dsmin_dd2 = (1.0 - h) + diff * dh_dd2 - blend_term * dh_dd2;
    *dsmin_dk = diff * dh_dk - h * (1.0 - h) - blend_term * dh_dk;
  }

  return sdf;
}

double smooth_max(double d1, double d2, double k,
                  double* dsmax_dd1, double* dsmax_dd2, double* dsmax_dk) {
  // smooth_max(d1, d2, k) = -smooth_min(-d1, -d2, k)
  double dd1, dd2, dk;
  double smin = smooth_min(-d1, -d2, k, &dd1, &dd2, &dk);

  *dsmax_dd1 = dd1;   // d(-smin)/d(d1) = -d(smin)/d(-d1) * (-1) = dd1
  *dsmax_dd2 = dd2;
  *dsmax_dk = -dk;     // d(-smin)/dk = -dk

  return -smin;
}

} // namespace cadopt
