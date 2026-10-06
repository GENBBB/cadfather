#include "ops/sketch_slot.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

double sdf_sketch_slot(double2 center, double half_len, double radius,
                       double angle, double3 p,
                       double* dcsg_dparams, double* dcsg_dp) {
  double ca = std::cos(angle);
  double sa = std::sin(angle);
  double rx = p.x - center.x;
  double ry = p.y - center.y;

  // Rotate to local frame (slot axis along local X)
  double lx =  ca * rx + sa * ry;
  double ly = -sa * rx + ca * ry;

  // Clamp to segment [-half_len, half_len]
  double fx = std::clamp(lx, -half_len, half_len);
  double dx = lx - fx;

  // Distance to nearest point on segment centerline
  double dist = std::sqrt(dx * dx + ly * ly);
  double inv_dist = (dist > 1e-15) ? (1.0 / dist) : 0.0;

  double sdf = dist - radius;

  // --- Gradients ---
  // d(sdf)/d(radius) = -1
  dcsg_dparams[3] = -1.0;

  double ddist_dlx, ddist_dhl;
  if (lx > half_len) {
    // Right cap: dx = lx - half_len > 0
    ddist_dlx = dx * inv_dist;
    ddist_dhl = -dx * inv_dist;
  } else if (lx < -half_len) {
    // Left cap: dx = lx + half_len < 0
    ddist_dlx = dx * inv_dist;
    ddist_dhl = dx * inv_dist;  // d(dx)/d(hl) = d(lx - (-hl))/d(hl) = 1, but dx<0
                                // Actually: fx = -hl, dx = lx-fx = lx+hl
                                // d(dx)/d(hl) = 1
                                // d(dist)/d(hl) = (dx/dist) * 1 = dx * inv_dist
  } else {
    // Body region: fx = lx, dx = 0, dist = |ly|
    ddist_dlx = 0.0;
    ddist_dhl = 0.0;
  }
  double ddist_dly = ly * inv_dist;

  // d(sdf)/d(half_len)
  dcsg_dparams[2] = ddist_dhl;

  // Chain rule through rotation:
  // d(lx)/d(cx) = -ca,  d(lx)/d(cy) = -sa
  // d(ly)/d(cx) =  sa,  d(ly)/d(cy) = -ca
  // d(lx)/d(angle) = -sa*rx + ca*ry = ly  (rotation derivative)
  // d(ly)/d(angle) = -ca*rx - sa*ry = -lx

  // d(sdf)/d(cx)
  dcsg_dparams[0] = ddist_dlx * (-ca) + ddist_dly * sa;
  // d(sdf)/d(cy)
  dcsg_dparams[1] = ddist_dlx * (-sa) + ddist_dly * (-ca);
  // d(sdf)/d(angle)
  dcsg_dparams[4] = ddist_dlx * ly + ddist_dly * (-lx);

  // Spatial gradients: d(sdf)/d(p.x), d(sdf)/d(p.y)
  // d(lx)/d(px) = ca,   d(lx)/d(py) = sa
  // d(ly)/d(px) = -sa,  d(ly)/d(py) = ca
  dcsg_dp[0] = ddist_dlx * ca + ddist_dly * (-sa);
  dcsg_dp[1] = ddist_dlx * sa + ddist_dly * ca;
  dcsg_dp[2] = 0.0;  // 2D primitive, no z

  return sdf;
}

} // namespace cadopt
