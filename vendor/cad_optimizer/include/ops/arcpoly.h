#pragma once
#include "types.h"

namespace cadopt {

// Per-edge descriptor for an arc-aware closed profile.
// Edge i connects vertices[i] -> vertices[(i+1)%n].
//   is_arc == false : straight line segment.
//   is_arc == true  : circular arc with radius-offset r_s (radius = 0.5*chord + r_s)
//                     bulging to the `side` (+1 / -1).  r_s and side are FROZEN
//                     (computed at parse time); only the endpoint vertices optimize.
struct ArcEdge {
  bool is_arc = false;
  double r_s = 0.0;
  double side = 1.0;
  bool major = false;   // true => arc sweeps the long way (>180 deg)
};

// Unsigned distance from point p to a circular-arc edge (a -> b) with radius-offset
// r_s and bulge side.  Sets gradients w.r.t. the two endpoints a, b and the query p.
// Ported from the ICAD inverse_CAD reference (distance_point_to_arc{,_diff}), with
// r_s held constant.
double dist_arc_segment(double2 a, double2 b, double r_s, double side, bool major, double2 p,
                        double2* dd_da, double2* dd_db, double2* dd_dp);

// Signed distance from p to a closed profile of n vertices whose edges are a mix of
// lines and arcs (edges[i] describes edge i).  Writes gradients into dparams[0..2n-1]
// (w.r.t. the vertices) and dp[0..1] (spatial).  Distance is exact for arcs; the
// inside/outside sign uses a densified (arc-sampled) winding test.
double sdf_arcpoly(const double2* vertices, int n, const ArcEdge* edges, double2 p,
                   double* dparams, double* dp);

} // namespace cadopt
