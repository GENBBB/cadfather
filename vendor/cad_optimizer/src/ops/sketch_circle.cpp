#include "ops/sketch_circle.h"
#include <cmath>

namespace cadopt {

// Ported from ICAD csg_nodes_diff_shared.h: calculate_csg_2D_circle_with_diff
double sdf_sketch_circle(double2 center, double radius, double3 p,
                         double* dcsg_dparams, double* dcsg_dp) {
  double dx = center.x - p.x;
  double dy = center.y - p.y;
  double dist = std::sqrt(dx * dx + dy * dy);
  double inv_dist = (dist > 1e-15) ? (1.0 / dist) : 0.0;

  // Param order: [0]=cx, [1]=cy, [2]=radius
  dcsg_dparams[0] = dx * inv_dist;    // d/d(cx)
  dcsg_dparams[1] = dy * inv_dist;    // d/d(cy)
  dcsg_dparams[2] = -1.0;             // d/d(radius)

  dcsg_dp[0] = -dx * inv_dist;        // d/d(p.x)
  dcsg_dp[1] = -dy * inv_dist;        // d/d(p.y)
  dcsg_dp[2] = 0.0;                   // no z

  return dist - radius;
}

} // namespace cadopt
