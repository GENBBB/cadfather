#include "ops/box3d.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

double sdf_box3d(double3 center, double3 half_size, double3 p,
                 double* dparams, double* dp) {
  double3 rel = p - center;
  double dx = std::abs(rel.x) - half_size.x;
  double dy = std::abs(rel.y) - half_size.y;
  double dz = std::abs(rel.z) - half_size.z;

  double sx = (rel.x >= 0) ? 1.0 : -1.0;
  double sy = (rel.y >= 0) ? 1.0 : -1.0;
  double sz = (rel.z >= 0) ? 1.0 : -1.0;

  double qx = std::max(dx, 0.0);
  double qy = std::max(dy, 0.0);
  double qz = std::max(dz, 0.0);

  int ox = dx > 0, oy = dy > 0, oz = dz > 0;
  int num_out = ox + oy + oz;

  if (num_out >= 2) {
    double len = std::sqrt(qx * qx + qy * qy + qz * qz);
    if (len < 1e-15) len = 1e-15;
    double inv = 1.0 / len;
    // dparams: [cx, cy, cz, hx, hy, hz]
    dparams[0] = ox ? -sx * qx * inv : 0.0;
    dparams[1] = oy ? -sy * qy * inv : 0.0;
    dparams[2] = oz ? -sz * qz * inv : 0.0;
    dparams[3] = ox ? -qx * inv : 0.0;
    dparams[4] = oy ? -qy * inv : 0.0;
    dparams[5] = oz ? -qz * inv : 0.0;
    dp[0] = ox ? sx * qx * inv : 0.0;
    dp[1] = oy ? sy * qy * inv : 0.0;
    dp[2] = oz ? sz * qz * inv : 0.0;
    return len;
  }

  // Face or interior: sdf = max(dx, dy, dz)
  // Derivative follows whichever axis is dominant
  std::fill_n(dparams, 6, 0.0);
  std::fill_n(dp, 3, 0.0);

  if (dx >= dy && dx >= dz) {
    dparams[0] = -sx; dparams[3] = -1.0;
    dp[0] = sx;
    return dx;
  } else if (dy >= dx && dy >= dz) {
    dparams[1] = -sy; dparams[4] = -1.0;
    dp[1] = sy;
    return dy;
  } else {
    dparams[2] = -sz; dparams[5] = -1.0;
    dp[2] = sz;
    return dz;
  }
}

} // namespace cadopt
