#include "ops/sketch_rect.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

// Ported from ICAD csg_nodes_diff_shared.h: calculate_csg_2D_box_with_diff
// Converted float -> double
double sdf_sketch_rect(double2 center, double2 half_size, double3 p,
                       double* dcsg_dparams, double* dcsg_dp) {
  double2 rel_p = double2(p.x, p.y) - center;
  double2 q = abs2(rel_p) - half_size;

  double2 q_max = max2(q, double2(0.0));
  double dist_outside = length(q_max);
  double dist_inside = std::min(std::max(q.x, q.y), 0.0);
  double sdf = dist_outside + dist_inside;

  bool outside_x = q.x > 0.0;
  bool outside_y = q.y > 0.0;
  double sign_x = (rel_p.x >= 0.0) ? 1.0 : -1.0;
  double sign_y = (rel_p.y >= 0.0) ? 1.0 : -1.0;

  // Param order: [0]=center_x, [1]=center_y, [2]=half_w, [3]=half_h
  if (outside_x && outside_y) {
    double inv_dist = (dist_outside > 1e-15) ? (1.0 / dist_outside) : 0.0;
    dcsg_dparams[2] = -q.x * inv_dist;        // d/d(half_w)
    dcsg_dparams[3] = -q.y * inv_dist;        // d/d(half_h)
    dcsg_dparams[0] = -sign_x * q.x * inv_dist; // d/d(cx)
    dcsg_dparams[1] = -sign_y * q.y * inv_dist; // d/d(cy)
    dcsg_dp[0] = sign_x * q.x * inv_dist;
    dcsg_dp[1] = sign_y * q.y * inv_dist;
  } else if (outside_x && !outside_y) {
    dcsg_dparams[2] = -1.0; dcsg_dparams[3] = 0.0;
    dcsg_dparams[0] = -sign_x; dcsg_dparams[1] = 0.0;
    dcsg_dp[0] = sign_x; dcsg_dp[1] = 0.0;
  } else if (!outside_x && outside_y) {
    dcsg_dparams[2] = 0.0; dcsg_dparams[3] = -1.0;
    dcsg_dparams[0] = 0.0; dcsg_dparams[1] = -sign_y;
    dcsg_dp[0] = 0.0; dcsg_dp[1] = sign_y;
  } else {
    if (q.x > q.y) {
      dcsg_dparams[2] = -1.0; dcsg_dparams[3] = 0.0;
      dcsg_dparams[0] = -sign_x; dcsg_dparams[1] = 0.0;
      dcsg_dp[0] = sign_x; dcsg_dp[1] = 0.0;
    } else {
      dcsg_dparams[2] = 0.0; dcsg_dparams[3] = -1.0;
      dcsg_dparams[0] = 0.0; dcsg_dparams[1] = -sign_y;
      dcsg_dp[0] = 0.0; dcsg_dp[1] = sign_y;
    }
  }
  dcsg_dp[2] = 0.0; // No z-component for 2D primitive
  return sdf;
}

} // namespace cadopt
