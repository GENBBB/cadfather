#include "ops/extrude.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

// One-sided extrusion to match cadquery's `.extrude(d)`: shape occupies
// z ∈ [min(0, depth), max(0, depth)].  Supports negative depth (extrude
// in -Z direction) without an outer translate — when Adam tunes depth,
// the extent shrinks toward z=0 (cadquery anchors the sketch end at the
// workplane, regardless of sign).
double sdf_extrude(double depth, double3 p, double child_sdf,
                   double* dcsg_dparams, double* dcsg_dp, double* dcsg_d_child) {
  double wx = child_sdf;
  double dmin = std::min(0.0, depth);
  double dmax = std::max(0.0, depth);
  double wy_lo = dmin - p.z;         // > 0 when below z = dmin
  double wy_hi = p.z - dmax;         // > 0 when above z = dmax
  double wy, dwy_dpz, dwy_ddepth;
  if (wy_lo > wy_hi) {
    wy = wy_lo;
    dwy_dpz = -1.0;
    // d(dmin)/d(depth) = 1 if depth<0, else 0
    dwy_ddepth = (depth < 0.0) ? 1.0 : 0.0;
  } else {
    wy = wy_hi;
    dwy_dpz = 1.0;
    // d(dmax)/d(depth) = 1 if depth>0; minus sign on (p.z - dmax)
    dwy_ddepth = (depth > 0.0) ? -1.0 : 0.0;
  }

  double wx_max = std::max(wx, 0.0);
  double wy_max = std::max(wy, 0.0);
  double dist_outside = std::sqrt(wx_max * wx_max + wy_max * wy_max);
  double dist_inside = std::min(std::max(wx, wy), 0.0);
  double sdf = dist_outside + dist_inside;

  bool outside_2d = wx > 0.0;
  bool outside_z = wy > 0.0;

  if (outside_2d && outside_z) {
    double inv_dist = (dist_outside > 1e-15) ? (1.0 / dist_outside) : 0.0;
    dcsg_dparams[0] = (wy * inv_dist) * dwy_ddepth;
    dcsg_dp[2]      = (wy * inv_dist) * dwy_dpz;
    *dcsg_d_child   = wx * inv_dist;
  } else if (outside_2d && !outside_z) {
    dcsg_dparams[0] = 0.0;
    dcsg_dp[2] = 0.0;
    *dcsg_d_child = 1.0;
  } else if (!outside_2d && outside_z) {
    dcsg_dparams[0] = dwy_ddepth;
    dcsg_dp[2] = dwy_dpz;
    *dcsg_d_child = 0.0;
  } else {
    if (wx > wy) {
      dcsg_dparams[0] = 0.0;
      dcsg_dp[2] = 0.0;
      *dcsg_d_child = 1.0;
    } else {
      dcsg_dparams[0] = dwy_ddepth;
      dcsg_dp[2] = dwy_dpz;
      *dcsg_d_child = 0.0;
    }
  }
  dcsg_dp[0] = 0.0;
  dcsg_dp[1] = 0.0;

  return sdf;
}

} // namespace cadopt
