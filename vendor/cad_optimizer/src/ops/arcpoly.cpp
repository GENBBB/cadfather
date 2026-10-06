#include "ops/arcpoly.h"
#include "ops/polygon.h"   // dist_line_segment
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace cadopt {

// ---------------------------------------------------------------------------
// Unsigned distance from P to the circular arc (A->B on circle centre O radius r)
// + gradients w.r.t. r, P, A, B, O.  Direct port of the ICAD reference
// distance_point_to_arc_diff (csg_nodes_diff_shared.h:316), float -> double.
// ---------------------------------------------------------------------------
static double distance_point_to_arc_diff(
    double r, double2 P, double2 A, double2 B, double2 O,
    double& ddist_dr, double2& ddist_dP, double2& ddist_dA,
    double2& ddist_dB, double2& ddist_dO) {
  ddist_dr = 0.0; ddist_dP = {0,0}; ddist_dA = {0,0}; ddist_dB = {0,0}; ddist_dO = {0,0};

  double2 OP = P - O;
  double distToCenter = length(OP);
  double2 dirToP = distToCenter > 1e-9 ? OP * (1.0 / distToCenter) : double2{1, 0};
  double2 closestOnCircle = O + dirToP * r;

  double2 OA = A - O, OB = B - O, OC = closestOnCircle - O;
  double crossAC = cross_2D(OA, OC);
  double crossAB = cross_2D(OA, OB);
  double crossCB = cross_2D(OC, OB);
  bool onArc = crossAB >= 0 ? (crossAC >= 0 && crossCB >= 0)
                            : !(crossAC < 0 && crossCB < 0);

  // Branch 1: closest point on the circle (arc interior)
  double dist1 = std::abs(distToCenter - r);
  double sign1 = (distToCenter - r) >= 0.0 ? 1.0 : -1.0;
  double2 dlen_dOP; length_diff(OP, &dlen_dOP);
  double2 ddist1_dOP = sign1 * dlen_dOP;
  double2 ddist1_dP = ddist1_dOP;       // OP = P - O
  double2 ddist1_dO = -ddist1_dOP;
  double  ddist1_dr = -sign1;

  // Branch 2: nearest endpoint
  double2 PA = P - A, PB = P - B;
  double2 dlen_dPA, dlen_dPB;
  double distPA = length_diff(PA, &dlen_dPA);
  double distPB = length_diff(PB, &dlen_dPB);
  bool useA = distPA <= distPB;
  double dist2 = useA ? distPA : distPB;
  double2 ddist2_dP = useA ? dlen_dPA : dlen_dPB;
  double2 ddist2_dA = useA ? -dlen_dPA : double2{0,0};
  double2 ddist2_dB = useA ? double2{0,0} : -dlen_dPB;

  if (onArc) {
    ddist_dr = ddist1_dr; ddist_dP = ddist1_dP; ddist_dO = ddist1_dO;
    return dist1;
  } else {
    ddist_dP = ddist2_dP; ddist_dA = ddist2_dA; ddist_dB = ddist2_dB;
    return dist2;
  }
}

// Forward-only arc geometry (centre, radius) used for the winding densify.
static void arc_geometry(double2 a, double2 b, double r_s, double side,
                         double2& c, double& r) {
  double2 mid = (a + b) * 0.5;
  double2 seg = b - a;
  double d = length(seg);
  double2 n = normalize2(double2{-seg.y, seg.x});
  r = 0.5 * d + r_s;
  double h2 = r * r - 0.25 * d * d;
  double h = h2 > 0 ? std::sqrt(h2) : 0.0;
  double inv_sign = side < 0 ? -1.0 : 1.0;
  c = mid + inv_sign * h * n;
}

double dist_arc_segment(double2 a, double2 b, double r_s, double side, bool major, double2 p,
                        double2* dd_da, double2* dd_db, double2* dd_dp) {
  bool inverted = side < 0;
  double inv_sign = inverted ? -1.0 : 1.0;

  double2 mid = (a + b) * 0.5;
  double2 seg = b - a;
  double d = length(seg);
  if (d < 1e-9) { // degenerate -> point distance
    return dist_line_segment(a, b, p, dd_da, dd_db, dd_dp);
  }
  double inv_d = 1.0 / d;
  double2 n = normalize2(double2{-seg.y, seg.x});
  double r = 0.5 * d + r_s;
  double h2 = r * r - 0.25 * d * d;
  double h = h2 > 1e-18 ? std::sqrt(h2) : 0.0;
  double inv_2h = h > 1e-9 ? 1.0 / (2.0 * h) : 0.0;
  double2 c = mid + inv_sign * h * n;

  // Selecting the minor vs the MAJOR arc (>180 deg) between a,b on the same
  // circle = swapping the A,B order in the arc-membership test (the complement).
  bool swap = inverted ^ major;
  double2 A = swap ? b : a;
  double2 B = swap ? a : b;

  double ddist_dr; double2 ddist_dP, ddist_dA, ddist_dB, ddist_dO;
  double dist = distance_point_to_arc_diff(r, p, A, B, c,
                                           ddist_dr, ddist_dP, ddist_dA, ddist_dB, ddist_dO);

  *dd_dp = ddist_dP;

  // chord-length derivatives
  double2 dd_da_chord = -seg * inv_d;  // dd(=|b-a|)/da
  double2 dd_db_chord = seg * inv_d;
  double  dh_dd = r_s * inv_2h;
  double2 dh_da = dh_dd * dd_da_chord;
  double2 dh_db = dh_dd * dd_db_chord;
  double2 dr_da = 0.5 * dd_da_chord;
  double2 dr_db = 0.5 * dd_db_chord;

  // normalize-derivative of n: dn/dparam = (I - n n^T)/d * dn_raw/dparam
  auto proj = [&](double2 v) -> double2 { return v - n * dot(n, v); };
  double2 dn_da_x = proj(double2{0, -1}) * inv_d;
  double2 dn_da_y = proj(double2{1, 0}) * inv_d;
  double2 dn_db_x = proj(double2{0, 1}) * inv_d;
  double2 dn_db_y = proj(double2{-1, 0}) * inv_d;

  // dc/d{a,b} contracted with ddist_dO  (dmid/da = 0.5 I)
  double dO_dc_da_x = dot(ddist_dO, double2{0.5, 0.0} + inv_sign * (dh_da.x * n + h * dn_da_x));
  double dO_dc_da_y = dot(ddist_dO, double2{0.0, 0.5} + inv_sign * (dh_da.y * n + h * dn_da_y));
  double dO_dc_db_x = dot(ddist_dO, double2{0.5, 0.0} + inv_sign * (dh_db.x * n + h * dn_db_x));
  double dO_dc_db_y = dot(ddist_dO, double2{0.0, 0.5} + inv_sign * (dh_db.y * n + h * dn_db_y));

  double2 ddist_da_endpoint = swap ? ddist_dB : ddist_dA;  // endpoint that equals a
  double2 ddist_db_endpoint = swap ? ddist_dA : ddist_dB;

  *dd_da = ddist_dr * dr_da + ddist_da_endpoint + double2{dO_dc_da_x, dO_dc_da_y};
  *dd_db = ddist_dr * dr_db + ddist_db_endpoint + double2{dO_dc_db_x, dO_dc_db_y};
  return dist;
}

// Append sampled points of the arc a->b (excluding a, including b) for winding.
static void sample_arc(double2 a, double2 b, double r_s, double side, bool major,
                       std::vector<double2>& out, int n_seg = 20) {
  double2 c, r; double rr;
  arc_geometry(a, b, r_s, side, c, rr);
  double2 va = a - c, vb = b - c;
  double la = length(va);
  if (la < 1e-9 || rr < 1e-9) { out.push_back(b); return; }
  // through-point (apex) determines which way the arc sweeps.  Minor: near apex
  // at mid - (r-h)*n (opposite the centre).  Major: far apex at mid + (r+h)*n.
  double2 mid = (a + b) * 0.5;
  double2 seg = b - a; double d = length(seg);
  double2 nrm = normalize2(double2{-seg.y, seg.x});
  double h = rr * rr - 0.25 * d * d; h = h > 0 ? std::sqrt(h) : 0.0;
  double inv_sign = side < 0 ? -1.0 : 1.0;
  double2 apex = major ? (mid + inv_sign * (rr + h) * nrm)
                       : (mid - inv_sign * (rr - h) * nrm);
  double ang_b = std::atan2(cross_2D(va, vb), dot(va, vb));
  double2 vap = apex - c;
  double ang_ap = std::atan2(cross_2D(va, vap), dot(va, vap));
  double sweep = ang_b;
  // pick the arc (short/long) that passes through the apex
  if (!((ang_ap >= 0) == (ang_b >= 0) && std::abs(ang_ap) <= std::abs(ang_b) + 1e-9)) {
    sweep = ang_b - (ang_b >= 0 ? 2.0 * M_PI : -2.0 * M_PI);
  }
  for (int k = 1; k < n_seg; k++) {
    double t = sweep * (double)k / (double)n_seg;
    double cs = std::cos(t), sn = std::sin(t);
    double2 v{va.x * cs - va.y * sn, va.x * sn + va.y * cs};
    out.push_back(c + v);
  }
  out.push_back(b);
}

double sdf_arcpoly(const double2* vertices, int n, const ArcEdge* edges, double2 p,
                   double* dparams, double* dp) {
  std::memset(dparams, 0, 2 * n * sizeof(double));
  dp[0] = dp[1] = 0.0;
  if (n < 2) return 1e30;

  // 1) closest edge (exact line/arc distance)
  double min_dist = 1e30; int min_i = 0;
  double2 min_dda{}, min_ddb{}, min_ddp{};
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    double2 dda, ddb, ddp;
    double dd = edges[i].is_arc
        ? dist_arc_segment(vertices[i], vertices[j], edges[i].r_s, edges[i].side, edges[i].major, p, &dda, &ddb, &ddp)
        : dist_line_segment(vertices[i], vertices[j], p, &dda, &ddb, &ddp);
    if (dd < min_dist) { min_dist = dd; min_i = i; min_dda = dda; min_ddb = ddb; min_ddp = ddp; }
  }

  // 2) sign via densified winding (arcs sampled; distance stays exact)
  std::vector<double2> poly; poly.reserve(n * 4);
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    poly.push_back(vertices[i]);
    if (edges[i].is_arc)
      sample_arc(vertices[i], vertices[j], edges[i].r_s, edges[i].side, edges[i].major, poly);  // adds interiors + endpoint
    // for line edges the next vertices[i] (= vertices[j]) is pushed next loop
  }
  int m = (int)poly.size();
  // NON-ZERO winding number (matches cadquery / OCC's BRep fill rule).  Identical
  // to even-odd for simple profiles, but for SELF-INTERSECTING profiles (which the
  // model sometimes emits) the two rules disagree in the overlap region; non-zero
  // is the one OCC uses to build the face, so this keeps the SDF sign faithful.
  int wn = 0;
  for (int i = 0; i < m; i++) {
    double2 a = poly[i], b = poly[(i + 1) % m];
    double isLeft = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
    if (a.y <= p.y) {
      if (b.y > p.y && isLeft > 0.0) wn++;
    } else {
      if (b.y <= p.y && isLeft < 0.0) wn--;
    }
  }
  double sign = (wn != 0) ? -1.0 : 1.0;

  int j = (min_i + 1) % n;
  dparams[2 * min_i]     = sign * min_dda.x;
  dparams[2 * min_i + 1] = sign * min_dda.y;
  dparams[2 * j]         = sign * min_ddb.x;
  dparams[2 * j + 1]     = sign * min_ddb.y;
  dp[0] = sign * min_ddp.x;
  dp[1] = sign * min_ddp.y;
  return sign * min_dist;
}

} // namespace cadopt
