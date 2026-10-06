#pragma once
#include "types.h"
#include <vector>
#include <string>

namespace cadopt {

struct Triangle {
  double3 v[3];
  double3 normal;
};

struct AABB {
  double3 mn{1e18, 1e18, 1e18};
  double3 mx{-1e18, -1e18, -1e18};
  void expand(double3 p) {
    mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
    mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
  }
  void expand(const AABB& o) { expand(o.mn); expand(o.mx); }
  double3 center() const { return (mn + mx) * 0.5; }
  int longestAxis() const {
    double3 d = mx - mn;
    if (d.x >= d.y && d.x >= d.z) return 0;
    if (d.y >= d.z) return 1;
    return 2;
  }
};

struct BVHNode {
  AABB box;
  int left = -1, right = -1; // child indices, or if leaf: left=triIdx, right=-1
  bool isLeaf() const { return right == -1 && left >= 0; }
};

struct Mesh {
  std::vector<Triangle> tris;
  std::vector<BVHNode> bvh;
  AABB bounds;

  void buildBVH();
  double closestDistance(double3 p) const;           // unsigned
  double signedDistance(double3 p) const;             // signed via pseudonormal
};

// I/O
Mesh load_stl(const std::string& path);

// Sampling
std::vector<double3> sample_surface_points(const Mesh& mesh, int count, unsigned seed = 42);

// Sample random points in the bounding box, keep those within `band` of the surface.
// Returns (positions, signed_distances) pairs.
void sample_sdf_points(const Mesh& mesh, int count, double band,
                       std::vector<double3>& positions, std::vector<double>& distances,
                       unsigned seed = 42);

} // namespace cadopt
