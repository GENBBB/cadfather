#include "ops/cylinder3d.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

double sdf_cylinder3d(double3 center, double radius, double half_height,
                      double3 p, double* dparams, double* dp) {
  double dx = p.x - center.x;
  double dy = p.y - center.y;
  double r_dist = std::sqrt(dx * dx + dy * dy);
  double r_xy = r_dist - radius;
  double h = std::abs(p.z - center.z) - half_height;
  double sz = (p.z - center.z >= 0) ? 1.0 : -1.0;

  // Radial gradient direction
  double rx = 0, ry = 0; // d(r_dist)/d(px), d(r_dist)/d(py)
  if (r_dist > 1e-15) {
    rx = dx / r_dist;
    ry = dy / r_dist;
  }

  bool out_r = r_xy > 0, out_h = h > 0;

  if (out_r && out_h) {
    // Corner: outside both tube and cap
    double len = std::sqrt(r_xy * r_xy + h * h);
    if (len < 1e-15) len = 1e-15;
    double inv = 1.0 / len;
    double wr = r_xy * inv, wh = h * inv;

    dparams[0] = -rx * wr;     // d/dcx
    dparams[1] = -ry * wr;     // d/dcy
    dparams[2] = -sz * wh;     // d/dcz
    dparams[3] = -wr;          // d/dradius
    dparams[4] = -wh;          // d/dhalf_height
    dp[0] = rx * wr;
    dp[1] = ry * wr;
    dp[2] = sz * wh;
    return len;
  }

  if (out_r && !out_h) {
    // Tube: outside radially, inside vertically
    dparams[0] = -rx;          // d/dcx
    dparams[1] = -ry;          // d/dcy
    dparams[2] = 0.0;
    dparams[3] = -1.0;         // d/dradius
    dparams[4] = 0.0;
    dp[0] = rx; dp[1] = ry; dp[2] = 0.0;
    return r_xy;
  }

  if (!out_r && out_h) {
    // Cap: inside radially, outside vertically
    dparams[0] = 0.0; dparams[1] = 0.0;
    dparams[2] = -sz;
    dparams[3] = 0.0;
    dparams[4] = -1.0;
    dp[0] = 0.0; dp[1] = 0.0; dp[2] = sz;
    return h;
  }

  // Interior: sdf = max(r_xy, h), both <= 0
  if (r_xy >= h) {
    dparams[0] = -rx; dparams[1] = -ry;
    dparams[2] = 0.0; dparams[3] = -1.0; dparams[4] = 0.0;
    dp[0] = rx; dp[1] = ry; dp[2] = 0.0;
    return r_xy;
  } else {
    dparams[0] = 0.0; dparams[1] = 0.0;
    dparams[2] = -sz; dparams[3] = 0.0; dparams[4] = -1.0;
    dp[0] = 0.0; dp[1] = 0.0; dp[2] = sz;
    return h;
  }
}

} // namespace cadopt
