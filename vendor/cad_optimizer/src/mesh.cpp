#include "mesh.h"
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstring>
#include <stdexcept>

namespace cadopt {

// ---- STL loading ----

Mesh load_stl(const std::string& path) {
  Mesh mesh;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open STL: " + path);

  char header[80];
  f.read(header, 80);

  uint32_t num_tris = 0;
  f.read(reinterpret_cast<char*>(&num_tris), 4);
  mesh.tris.resize(num_tris);

  for (uint32_t i = 0; i < num_tris; i++) {
    float buf[12];
    f.read(reinterpret_cast<char*>(buf), 48);
    uint16_t attr;
    f.read(reinterpret_cast<char*>(&attr), 2);

    mesh.tris[i].normal = {buf[0], buf[1], buf[2]};
    mesh.tris[i].v[0]   = {buf[3], buf[4], buf[5]};
    mesh.tris[i].v[1]   = {buf[6], buf[7], buf[8]};
    mesh.tris[i].v[2]   = {buf[9], buf[10], buf[11]};

    mesh.bounds.expand(mesh.tris[i].v[0]);
    mesh.bounds.expand(mesh.tris[i].v[1]);
    mesh.bounds.expand(mesh.tris[i].v[2]);
  }

  mesh.buildBVH();
  return mesh;
}

// ---- BVH construction ----

static AABB triBox(const Triangle& t) {
  AABB b;
  b.expand(t.v[0]); b.expand(t.v[1]); b.expand(t.v[2]);
  return b;
}

void Mesh::buildBVH() {
  if (tris.empty()) return;

  int n = (int)tris.size();
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  bvh.clear();
  bvh.reserve(2 * n);

  struct BuildTask { int nodeIdx; int begin; int end; };
  std::vector<BuildTask> stack;

  bvh.push_back({});
  stack.push_back({0, 0, n});

  while (!stack.empty()) {
    auto [nodeIdx, begin, end] = stack.back();
    stack.pop_back();

    AABB box;
    for (int i = begin; i < end; i++)
      box.expand(triBox(tris[indices[i]]));
    bvh[nodeIdx].box = box;

    int count = end - begin;
    if (count <= 4) {
      // Leaf: store range [begin, end) as triIdx = begin, right = end
      // For simplicity, we'll use a small-leaf approach: store first tri index
      // and iterate linearly for leaves with <= 4 tris
      bvh[nodeIdx].left = begin;
      bvh[nodeIdx].right = -(end); // negative marks leaf with end index
      continue;
    }

    int axis = box.longestAxis();
    int mid = (begin + end) / 2;
    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
      [&](int a, int b) { return triBox(tris[a]).center()[axis] < triBox(tris[b]).center()[axis]; });

    int leftIdx = (int)bvh.size();
    bvh.push_back({});
    int rightIdx = (int)bvh.size();
    bvh.push_back({});
    bvh[nodeIdx].left = leftIdx;
    bvh[nodeIdx].right = rightIdx;

    stack.push_back({rightIdx, mid, end});
    stack.push_back({leftIdx, begin, mid});
  }

  // Reorder triangles by BVH index order for cache locality
  std::vector<Triangle> reordered(n);
  for (int i = 0; i < n; i++)
    reordered[i] = tris[indices[i]];
  tris = std::move(reordered);
}

// ---- Closest point on triangle ----

static double3 closestPointOnTriangle(double3 p, const Triangle& tri, double3& pseudonormal) {
  double3 a = tri.v[0], b = tri.v[1], c = tri.v[2];
  double3 ab = b - a, ac = c - a, ap = p - a;

  double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) { pseudonormal = tri.normal; return a; }

  double3 bp = p - b;
  double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) { pseudonormal = tri.normal; return b; }

  double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    double v = d1 / (d1 - d3);
    pseudonormal = tri.normal;
    return a + ab * v;
  }

  double3 cp = p - c;
  double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) { pseudonormal = tri.normal; return c; }

  double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    double w = d2 / (d2 - d6);
    pseudonormal = tri.normal;
    return a + ac * w;
  }

  double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    pseudonormal = tri.normal;
    return b + (c - b) * w;
  }

  double denom = 1.0 / (va + vb + vc);
  double v = vb * denom, w = vc * denom;
  pseudonormal = tri.normal;
  return a + ab * v + ac * w;
}

// ---- BVH traversal for closest distance ----

static double aabbDist2(const AABB& box, double3 p) {
  double dx = std::max({box.mn.x - p.x, 0.0, p.x - box.mx.x});
  double dy = std::max({box.mn.y - p.y, 0.0, p.y - box.mx.y});
  double dz = std::max({box.mn.z - p.z, 0.0, p.z - box.mx.z});
  return dx * dx + dy * dy + dz * dz;
}

double Mesh::closestDistance(double3 p) const {
  return std::abs(signedDistance(p));
}

double Mesh::signedDistance(double3 p) const {
  if (bvh.empty()) return 1e18;

  double bestDist2 = 1e30;
  double3 bestNormal{0, 1, 0};
  double3 bestClosest{0, 0, 0};

  struct StackEntry { int nodeIdx; };
  StackEntry stack[64];
  int top = 0;
  stack[top++] = {0};

  while (top > 0) {
    int ni = stack[--top].nodeIdx;
    const BVHNode& node = bvh[ni];

    if (aabbDist2(node.box, p) >= bestDist2)
      continue;

    if (node.right <= 0) {
      // Leaf: tris in range [node.left, -node.right)
      int begin = node.left;
      int end = -node.right;
      for (int i = begin; i < end; i++) {
        double3 pn;
        double3 cp = closestPointOnTriangle(p, tris[i], pn);
        double3 diff = p - cp;
        double d2 = dot(diff, diff);
        if (d2 < bestDist2) {
          bestDist2 = d2;
          bestNormal = pn;
          bestClosest = cp;
        }
      }
    } else {
      // Push both children; visit closer one first
      double dL = aabbDist2(bvh[node.left].box, p);
      double dR = aabbDist2(bvh[node.right].box, p);
      if (dL < dR) {
        stack[top++] = {node.right};
        stack[top++] = {node.left};
      } else {
        stack[top++] = {node.left};
        stack[top++] = {node.right};
      }
    }
  }

  double dist = std::sqrt(bestDist2);
  double sign = dot(p - bestClosest, bestNormal) >= 0 ? 1.0 : -1.0;
  return sign * dist;
}

// ---- Sampling ----

std::vector<double3> sample_surface_points(const Mesh& mesh, int count, unsigned seed) {
  std::vector<double3> samples;
  samples.reserve(count);
  int n = (int)mesh.tris.size();
  if (n == 0) return samples;

  // Build area-weighted CDF
  std::vector<double> cdf(n);
  double total = 0;
  for (int i = 0; i < n; i++) {
    double3 e1 = mesh.tris[i].v[1] - mesh.tris[i].v[0];
    double3 e2 = mesh.tris[i].v[2] - mesh.tris[i].v[0];
    total += 0.5 * length(cross(e1, e2));
    cdf[i] = total;
  }
  for (auto& v : cdf) v /= total;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int s = 0; s < count; s++) {
    double r = dist(rng);
    int ti = (int)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
    ti = std::min(ti, n - 1);
    const auto& t = mesh.tris[ti];

    double u = dist(rng), v = dist(rng);
    double su = std::sqrt(u);
    double a = 1.0 - su, b = su * (1.0 - v), c = su * v;
    samples.push_back(t.v[0] * a + t.v[1] * b + t.v[2] * c);
  }
  return samples;
}

void sample_sdf_points(const Mesh& mesh, int count, double band,
                       std::vector<double3>& positions, std::vector<double>& distances,
                       unsigned seed) {
  positions.clear();
  distances.clear();
  positions.reserve(count);
  distances.reserve(count);

  // Expand bounding box by band
  double3 mn = mesh.bounds.mn - double3(band);
  double3 mx = mesh.bounds.mx + double3(band);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dx(mn.x, mx.x);
  std::uniform_real_distribution<double> dy(mn.y, mx.y);
  std::uniform_real_distribution<double> dz(mn.z, mx.z);

  int tries = 0;
  while ((int)positions.size() < count && tries < 100 * count) {
    double3 p{dx(rng), dy(rng), dz(rng)};
    double sd = mesh.signedDistance(p);
    if (std::abs(sd) < band) {
      positions.push_back(p);
      distances.push_back(sd);
    }
    tries++;
  }
}

} // namespace cadopt
