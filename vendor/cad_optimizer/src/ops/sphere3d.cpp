#include "ops/sphere3d.h"
#include <cmath>

namespace cadopt {

double sdf_sphere3d(double3 center, double radius, double3 p,
                    double* dparams, double* dp) {
  double3 rel = p - center;
  double dist = length(rel);
  double sdf = dist - radius;

  if (dist > 1e-15) {
    double inv = 1.0 / dist;
    double nx = rel.x * inv, ny = rel.y * inv, nz = rel.z * inv;
    dparams[0] = -nx;  // d/dcx
    dparams[1] = -ny;  // d/dcy
    dparams[2] = -nz;  // d/dcz
    dparams[3] = -1.0; // d/dradius
    dp[0] = nx;
    dp[1] = ny;
    dp[2] = nz;
  } else {
    dparams[0] = 0.0; dparams[1] = 0.0; dparams[2] = 0.0;
    dparams[3] = -1.0;
    dp[0] = 0.0; dp[1] = 0.0; dp[2] = 0.0;
  }
  return sdf;
}

} // namespace cadopt
