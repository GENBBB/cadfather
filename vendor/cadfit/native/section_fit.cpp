// section_fit.cpp — 2D primitive fits for section analysis.
//
// Implementations:
//   - fit_line:    total least squares via 2x2 covariance eigen-vector.
//   - fit_circle:  Kasa algebraic least squares (linear system, 3 unknowns).
//   - fit_arc:     fit_circle + angle range detection from largest angular gap.
//   - fit_rect:    minimum-area oriented bounding box (rotating calipers
//                  on the convex hull).
//   - fit_polygon: Ramer-Douglas-Peucker simplification, pass-through fit.
//
// Self-contained C++17, no external libraries.
#include "section_fit.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace cadopt::section {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

double bbox_diagonal(const double2* pts, int n) {
  double lo_x = std::numeric_limits<double>::infinity();
  double lo_y = lo_x;
  double hi_x = -lo_x, hi_y = hi_x;
  for (int i = 0; i < n; ++i) {
    lo_x = std::min(lo_x, pts[i].x);
    lo_y = std::min(lo_y, pts[i].y);
    hi_x = std::max(hi_x, pts[i].x);
    hi_y = std::max(hi_y, pts[i].y);
  }
  double dx = hi_x - lo_x;
  double dy = hi_y - lo_y;
  return std::sqrt(dx * dx + dy * dy);
}

double2 centroid(const double2* pts, int n) {
  double2 c{0.0, 0.0};
  for (int i = 0; i < n; ++i) c += pts[i];
  if (n > 0) { c.x /= n; c.y /= n; }
  return c;
}

// Solve 3x3 linear system A x = b in place (Gaussian elimination with
// partial pivoting).  Returns false on near-singular.
bool solve_3x3(double A[3][3], double b[3], double x[3]) {
  for (int i = 0; i < 3; ++i) {
    int pivot = i;
    double max_abs = std::abs(A[i][i]);
    for (int r = i + 1; r < 3; ++r) {
      double v = std::abs(A[r][i]);
      if (v > max_abs) { max_abs = v; pivot = r; }
    }
    if (max_abs < 1e-14) return false;
    if (pivot != i) {
      for (int c = 0; c < 3; ++c) std::swap(A[i][c], A[pivot][c]);
      std::swap(b[i], b[pivot]);
    }
    for (int r = i + 1; r < 3; ++r) {
      double f = A[r][i] / A[i][i];
      for (int c = i; c < 3; ++c) A[r][c] -= f * A[i][c];
      b[r] -= f * b[i];
    }
  }
  for (int i = 2; i >= 0; --i) {
    double s = b[i];
    for (int c = i + 1; c < 3; ++c) s -= A[i][c] * x[c];
    x[i] = s / A[i][i];
  }
  return true;
}

void scoring_pass(PrimitiveFit& fit, const double2* pts, int n,
                  double tol, double scale,
                  // distance functor: takes (p, params) -> perpendicular distance
                  double (*dist_fn)(const double2&, const std::vector<double>&)) {
  if (n <= 0) {
    fit.residual = 0.0;
    fit.support_frac = 0.0;
    fit.score = std::numeric_limits<double>::infinity();
    return;
  }
  double sum_abs = 0.0;
  int n_in = 0;
  for (int i = 0; i < n; ++i) {
    double d = std::abs(dist_fn(pts[i], fit.params));
    sum_abs += d;
    if (d < tol) ++n_in;
  }
  fit.residual = sum_abs / n;
  fit.support_frac = static_cast<double>(n_in) / n;
  double outlier_penalty = (1.0 - fit.support_frac) * std::max(scale, 1e-9);
  fit.score = (fit.residual / std::max(scale, 1e-9)) + outlier_penalty;
}

double dist_line(const double2& p, const std::vector<double>& pr) {
  // pr = [nx, ny, c];  signed dist = nx*x + ny*y - c
  return pr[0] * p.x + pr[1] * p.y - pr[2];
}

double dist_circle(const double2& p, const std::vector<double>& pr) {
  // pr = [cx, cy, r];  signed dist = sqrt((x-cx)^2+(y-cy)^2) - r
  double dx = p.x - pr[0];
  double dy = p.y - pr[1];
  return std::sqrt(dx * dx + dy * dy) - pr[2];
}

double dist_rect(const double2& p, const std::vector<double>& pr) {
  // pr = [cx, cy, w, h, theta].  Signed dist via OBB SDF.
  double cx = pr[0], cy = pr[1], w = pr[2], h = pr[3], th = pr[4];
  double ct = std::cos(th), st = std::sin(th);
  // Rotate point into rect-local frame.
  double dx = p.x - cx, dy = p.y - cy;
  double lx =  ct * dx + st * dy;
  double ly = -st * dx + ct * dy;
  double qx = std::abs(lx) - w * 0.5;
  double qy = std::abs(ly) - h * 0.5;
  double outside = std::sqrt(std::max(qx, 0.0) * std::max(qx, 0.0)
                           + std::max(qy, 0.0) * std::max(qy, 0.0));
  double inside  = std::min(std::max(qx, qy), 0.0);
  return outside + inside;
}

double dist_polygon(const double2& /*p*/, const std::vector<double>& /*pr*/) {
  // For polygon kind we don't compute a curve-distance — the polygon
  // *is* the data.  Use 0 (perfect fit by definition; score deferred to
  // the simplification reduction ratio assigned in fit_polygon).
  return 0.0;
}

} // namespace


// ----------------------------- LINE ---------------------------------------

PrimitiveFit fit_line(const double2* pts, int n) {
  PrimitiveFit f;
  f.kind = PrimitiveKind::LINE;
  f.params.assign(3, 0.0);
  if (n < 2) return f;

  double2 mu = centroid(pts, n);
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (int i = 0; i < n; ++i) {
    double dx = pts[i].x - mu.x;
    double dy = pts[i].y - mu.y;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  // 2x2 covariance:  [sxx sxy; sxy syy].  Smaller eigenvalue's
  // eigenvector is perpendicular to the best line.
  double trace = sxx + syy;
  double det   = sxx * syy - sxy * sxy;
  double disc  = std::max(0.0, trace * trace * 0.25 - det);
  double lam_small = trace * 0.5 - std::sqrt(disc);
  // eigenvector for lam_small:  (sxy, lam_small - sxx)  (unnormalized)
  double nx = sxy;
  double ny = lam_small - sxx;
  double nlen = std::sqrt(nx * nx + ny * ny);
  if (nlen < 1e-14) {
    // Degenerate (all points coincide) — return a vertical line.
    nx = 1.0; ny = 0.0; nlen = 1.0;
  }
  nx /= nlen; ny /= nlen;
  double c = nx * mu.x + ny * mu.y;
  f.params = {nx, ny, c};
  return f;
}


// ----------------------------- CIRCLE -------------------------------------

PrimitiveFit fit_circle(const double2* pts, int n) {
  PrimitiveFit f;
  f.kind = PrimitiveKind::CIRCLE;
  f.params.assign(3, 0.0);
  if (n < 3) return f;

  // Kasa algebraic fit.  Solve for (a, b, c) such that
  //   x^2 + y^2 + a*x + b*y + c = 0
  // for each input point in a least-squares sense.  Then
  //   cx = -a/2, cy = -b/2, r = sqrt(cx^2 + cy^2 - c).
  double Sxx = 0, Syy = 0, Sxy = 0;
  double Sx  = 0, Sy  = 0, S = static_cast<double>(n);
  double Sxz = 0, Syz = 0, Sz  = 0;
  for (int i = 0; i < n; ++i) {
    double x = pts[i].x, y = pts[i].y;
    double z = x * x + y * y;
    Sxx += x * x;  Syy += y * y;  Sxy += x * y;
    Sx  += x;       Sy  += y;
    Sxz += x * z;   Syz += y * z;  Sz  += z;
  }
  // Solve [[Sxx Sxy Sx],[Sxy Syy Sy],[Sx Sy S]] * [a,b,c] = -[Sxz, Syz, Sz]
  double A[3][3] = {{Sxx, Sxy, Sx}, {Sxy, Syy, Sy}, {Sx, Sy, S}};
  double b_rhs[3] = {-Sxz, -Syz, -Sz};
  double sol[3] = {0, 0, 0};
  if (!solve_3x3(A, b_rhs, sol)) return f;
  double cx = -0.5 * sol[0];
  double cy = -0.5 * sol[1];
  double rr2 = cx * cx + cy * cy - sol[2];
  if (rr2 <= 0) return f;
  double r = std::sqrt(rr2);
  f.params = {cx, cy, r};
  return f;
}


// ----------------------------- ARC ----------------------------------------

PrimitiveFit fit_arc(const double2* pts, int n) {
  PrimitiveFit base = fit_circle(pts, n);
  PrimitiveFit f;
  f.kind = PrimitiveKind::ARC;
  if (base.params.size() != 3) return f;
  const double cx = base.params[0], cy = base.params[1], r = base.params[2];

  std::vector<double> theta;
  theta.reserve(n);
  for (int i = 0; i < n; ++i) {
    theta.push_back(std::atan2(pts[i].y - cy, pts[i].x - cx));
  }
  std::sort(theta.begin(), theta.end());

  // Find the largest angular gap (wrap-around included).
  double max_gap = 0.0;
  int gap_idx = -1;
  for (int i = 0; i < n; ++i) {
    int j = (i + 1) % n;
    double g = theta[j] - theta[i];
    if (j == 0) g += kTwoPi;     // wrap
    if (g > max_gap) { max_gap = g; gap_idx = i; }
  }

  // If the largest gap is small, the points cover the full circle.
  if (max_gap < kPi / 12.0 /* 15 degrees */) {
    f.kind = PrimitiveKind::CIRCLE;
    f.params = base.params;
  } else {
    // Arc spans from theta[gap_idx+1] -> theta[gap_idx] (going around).
    int start = (gap_idx + 1) % n;
    int end   = gap_idx;
    f.params = {cx, cy, r, theta[start], theta[end]};
  }
  return f;
}


// ----------------------------- RECT ---------------------------------------

namespace {

// Returns convex hull in counter-clockwise order.  Andrew's monotone
// chain on a local copy of the input points (sorted by x then y).
std::vector<double2> convex_hull(const double2* pts, int n) {
  std::vector<double2> p(pts, pts + n);
  std::sort(p.begin(), p.end(), [](double2 a, double2 b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  // Remove duplicates to avoid collinear weirdness.
  p.erase(std::unique(p.begin(), p.end(),
                      [](double2 a, double2 b){
                        return a.x == b.x && a.y == b.y;
                      }), p.end());
  int m = static_cast<int>(p.size());
  if (m < 3) return p;

  auto cross2 = [](double2 O, double2 A, double2 B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
  };

  std::vector<double2> H(2 * m);
  int k = 0;
  // Lower hull
  for (int i = 0; i < m; ++i) {
    while (k >= 2 && cross2(H[k - 2], H[k - 1], p[i]) <= 0.0) --k;
    H[k++] = p[i];
  }
  // Upper hull
  for (int i = m - 2, t = k + 1; i >= 0; --i) {
    while (k >= t && cross2(H[k - 2], H[k - 1], p[i]) <= 0.0) --k;
    H[k++] = p[i];
  }
  H.resize(k - 1);
  return H;
}

} // namespace


PrimitiveFit fit_rect(const double2* pts, int n) {
  PrimitiveFit f;
  f.kind = PrimitiveKind::RECT;
  f.params.assign(5, 0.0);
  if (n < 3) return f;

  std::vector<double2> H = convex_hull(pts, n);
  int m = static_cast<int>(H.size());
  if (m < 3) return f;

  double best_area = std::numeric_limits<double>::infinity();
  double best_cx = 0, best_cy = 0, best_w = 0, best_h = 0, best_th = 0;
  for (int i = 0; i < m; ++i) {
    double2 a = H[i];
    double2 b = H[(i + 1) % m];
    double ex = b.x - a.x, ey = b.y - a.y;
    double el = std::sqrt(ex * ex + ey * ey);
    if (el < 1e-14) continue;
    double ux = ex / el, uy = ey / el;
    double vx = -uy,     vy = ux;
    double min_u = std::numeric_limits<double>::infinity();
    double max_u = -min_u;
    double min_v = min_u, max_v = max_u;
    for (int j = 0; j < m; ++j) {
      double pu = H[j].x * ux + H[j].y * uy;
      double pv = H[j].x * vx + H[j].y * vy;
      min_u = std::min(min_u, pu);
      max_u = std::max(max_u, pu);
      min_v = std::min(min_v, pv);
      max_v = std::max(max_v, pv);
    }
    double w = max_u - min_u;
    double h = max_v - min_v;
    double area = w * h;
    if (area < best_area) {
      best_area = area;
      double cu = 0.5 * (min_u + max_u);
      double cv = 0.5 * (min_v + max_v);
      best_cx = cu * ux + cv * vx;
      best_cy = cu * uy + cv * vy;
      best_w  = w;
      best_h  = h;
      best_th = std::atan2(uy, ux);
    }
  }
  f.params = {best_cx, best_cy, best_w, best_h, best_th};
  return f;
}


// ----------------------------- POLYGON ------------------------------------

namespace {

double perp_dist(double2 p, double2 a, double2 b) {
  double dx = b.x - a.x;
  double dy = b.y - a.y;
  double len = std::sqrt(dx * dx + dy * dy);
  if (len < 1e-14) {
    double ex = p.x - a.x, ey = p.y - a.y;
    return std::sqrt(ex * ex + ey * ey);
  }
  // unsigned perpendicular distance from p to segment AB extended.
  double num = std::abs(dx * (a.y - p.y) - (a.x - p.x) * dy);
  return num / len;
}

void rdp(const std::vector<double2>& p, int lo, int hi, double tol,
         std::vector<int>& keep) {
  if (hi <= lo + 1) return;
  double max_d = 0.0;
  int idx = lo;
  for (int i = lo + 1; i < hi; ++i) {
    double d = perp_dist(p[i], p[lo], p[hi]);
    if (d > max_d) { max_d = d; idx = i; }
  }
  if (max_d > tol) {
    keep.push_back(idx);
    rdp(p, lo, idx, tol, keep);
    rdp(p, idx, hi, tol, keep);
  }
}

} // namespace


PrimitiveFit fit_polygon(const double2* pts, int n, double simplify_tol) {
  PrimitiveFit f;
  f.kind = PrimitiveKind::POLYGON;
  if (n < 3) return f;

  std::vector<double2> p(pts, pts + n);
  std::vector<int> keep = {0, n - 1};
  rdp(p, 0, n - 1, simplify_tol, keep);
  std::sort(keep.begin(), keep.end());
  keep.erase(std::unique(keep.begin(), keep.end()), keep.end());

  f.params.reserve(2 * keep.size());
  for (int i : keep) {
    f.params.push_back(p[i].x);
    f.params.push_back(p[i].y);
  }
  return f;
}


// ----------------------------- fit_all ------------------------------------

std::vector<PrimitiveFit> fit_all(const double2* pts, int n, double tol) {
  std::vector<PrimitiveFit> out;
  if (n < 2) return out;

  double scale = bbox_diagonal(pts, n);
  if (scale < 1e-14) scale = 1.0;
  double t = tol > 0.0 ? tol : 0.01 * scale;

  PrimitiveFit lf = fit_line(pts, n);
  if (lf.params.size() == 3)
    scoring_pass(lf, pts, n, t, scale, dist_line);
  out.push_back(std::move(lf));

  PrimitiveFit cf = fit_circle(pts, n);
  if (cf.params.size() == 3)
    scoring_pass(cf, pts, n, t, scale, dist_circle);
  out.push_back(std::move(cf));

  PrimitiveFit af = fit_arc(pts, n);
  if (!af.params.empty()) {
    // The distance to an arc curve is the same as to the underlying
    // circle, since arc points lie on the circle.
    scoring_pass(af, pts, n, t, scale, dist_circle);
  }
  out.push_back(std::move(af));

  PrimitiveFit rf = fit_rect(pts, n);
  if (rf.params.size() == 5)
    scoring_pass(rf, pts, n, t, scale, dist_rect);
  out.push_back(std::move(rf));

  PrimitiveFit pf = fit_polygon(pts, n, t);
  // polygon is the data — give it the worst score by definition unless
  // every other fit fails.  We set residual=0 but score is "scale" so it
  // ranks last when *any* primitive matches with residual<scale.
  pf.residual = 0.0;
  pf.support_frac = 1.0;
  pf.score = scale;
  out.push_back(std::move(pf));

  // Sort by score ascending; tie-break by complexity so degenerate
  // higher-DOF fits (e.g. a line is also a 0-width RECT) lose to the
  // simpler primitive.  Complexity order: LINE < CIRCLE < ARC < RECT < POLYGON.
  auto complexity = [](PrimitiveKind k) {
    switch (k) {
      case PrimitiveKind::LINE:    return 0;
      case PrimitiveKind::CIRCLE:  return 1;
      case PrimitiveKind::ARC:     return 2;
      case PrimitiveKind::RECT:    return 3;
      case PrimitiveKind::POLYGON: return 4;
    }
    return 5;
  };
  // Pick the "near-tie" threshold from the median score: anything
  // within 1e-9 of the best is considered tied.
  std::sort(out.begin(), out.end(),
            [&](const PrimitiveFit& a, const PrimitiveFit& b) {
              constexpr double kTieEps = 1e-9;
              if (std::abs(a.score - b.score) < kTieEps)
                return complexity(a.kind) < complexity(b.kind);
              return a.score < b.score;
            });
  return out;
}

} // namespace cadopt::section
