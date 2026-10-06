#include "ops/polygon.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace cadopt {

double dist_line_segment(double2 a, double2 b, double2 p,
                         double2* dd_da, double2* dd_db, double2* dd_dp) {
  const double eps_seg = 1e-12;
  const double eps_dist = 1e-8;

  double2 e = b - a;
  double2 w = p - a;
  double ee = dot(e, e);

  // Degenerate segment: collapse to point a
  if (ee <= eps_seg) {
    double d = length(w);
    if (d > eps_dist) {
      double inv = 1.0 / d;
      *dd_dp = w * inv;
      *dd_da = w * (-inv);
    } else {
      *dd_dp = {0, 0};
      *dd_da = {0, 0};
    }
    *dd_db = {0, 0};
    return d;
  }

  double inv_ee = 1.0 / ee;
  double t_raw = dot(w, e) * inv_ee;

  // Region 1: closest to a (t <= 0)
  if (t_raw <= 0.0) {
    double d = length(w);
    if (d > eps_dist) {
      double inv = 1.0 / d;
      *dd_dp = w * inv;
      *dd_da = w * (-inv);
    } else {
      *dd_dp = {0, 0};
      *dd_da = {0, 0};
    }
    *dd_db = {0, 0};
    return d;
  }

  // Region 2: closest to b (t >= 1)
  if (t_raw >= 1.0) {
    double2 r = p - b;
    double d = length(r);
    if (d > eps_dist) {
      double inv = 1.0 / d;
      *dd_dp = r * inv;
      *dd_db = r * (-inv);
    } else {
      *dd_dp = {0, 0};
      *dd_db = {0, 0};
    }
    *dd_da = {0, 0};
    return d;
  }

  // Region 3: interior of segment (0 < t < 1)
  double t = t_raw;
  double2 closest = a + e * t;
  double2 r = p - closest;
  double d = length(r);

  if (d <= eps_dist) {
    *dd_da = {0, 0};
    *dd_db = {0, 0};
    *dd_dp = {0, 0};
    return d;
  }

  double inv_d = 1.0 / d;
  double rs = dot(r, e);
  double vDotS = dot(w, e);
  double inv_ee2 = inv_ee * inv_ee;

  // Port from ICAD reference: interior-case derivatives
  double2 A = (e + w) * (-inv_ee) + e * (2.0 * vDotS * inv_ee2);
  double2 B = w * inv_ee - e * (2.0 * vDotS * inv_ee2);

  *dd_da = (r * (-(1.0 - t)) - A * rs) * inv_d;
  *dd_db = (r * (-t) - B * rs) * inv_d;
  *dd_dp = (r - e * (rs * inv_ee)) * inv_d;

  return d;
}

double sdf_polygon(const double2* vertices, int n, double2 p,
                   double* dparams, double* dp) {
  // Zero all gradients
  std::memset(dparams, 0, 2 * n * sizeof(double));
  dp[0] = dp[1] = 0.0;

  double min_dist = 1e30;
  int min_i = 0;
  double2 min_dd_da{}, min_dd_db{}, min_dd_dp{};

  // Find closest edge and compute its unsigned distance gradient
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    double2 dd_da, dd_db, dd_dp;
    double d = dist_line_segment(vertices[i], vertices[j], p,
                                 &dd_da, &dd_db, &dd_dp);
    if (d < min_dist) {
      min_dist = d;
      min_i = i;
      min_dd_da = dd_da;
      min_dd_db = dd_db;
      min_dd_dp = dd_dp;
    }
  }

  // Sign via NON-ZERO winding number (matches cadquery/OCC's BRep fill rule).
  // Same as even-odd for simple polygons, but for SELF-INTERSECTING profiles
  // (the model sometimes emits these) the rules differ; OCC builds the face with
  // non-zero, so this keeps the SDF sign faithful in the overlap region.
  int wn = 0;
  for (int i = 0; i < n; i++) {
    double2 a = vertices[i], b = vertices[(i + 1) % n];
    double isLeft = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
    if (a.y <= p.y) {
      if (b.y > p.y && isLeft > 0.0) wn++;
    } else {
      if (b.y <= p.y && isLeft < 0.0) wn--;
    }
  }
  double sign = (wn != 0) ? -1.0 : 1.0;

  // Write signed gradients for the closest edge
  int j = (min_i + 1) % n;
  dparams[2 * min_i]     = sign * min_dd_da.x;
  dparams[2 * min_i + 1] = sign * min_dd_da.y;
  dparams[2 * j]         = sign * min_dd_db.x;
  dparams[2 * j + 1]     = sign * min_dd_db.y;
  dp[0] = sign * min_dd_dp.x;
  dp[1] = sign * min_dd_dp.y;

  return sign * min_dist;
}

} // namespace cadopt
