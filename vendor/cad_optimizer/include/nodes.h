#pragma once
#include "csg_tree.h"
#include "ops/sketch_rect.h"
#include "ops/sketch_circle.h"
#include "ops/sketch_slot.h"
#include "ops/extrude.h"
#include "ops/rotate.h"
#include "ops/fillet.h"
#include "ops/chamfer.h"
#include "ops/sphere3d.h"
#include "ops/box3d.h"
#include "ops/cylinder3d.h"
#include "ops/polygon.h"
#include "ops/arcpoly.h"
#include "mesh.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>

namespace cadopt {

// Frozen mesh leaf: a fixed sub-solid baked to a triangle mesh (BVH signed
// distance).  Used for cadgen ops with no analytical SDF (gear/sweep/spring/
// shell/loft) so the surrounding SUPPORTED ops can still be optimized against
// a faithful frozen backdrop.  Zero params; zero spatial gradient because the
// frozen branch carries no tunable params (only the branch VALUE participates
// in the boolean min/max) — keeps it to one BVH query per eval.
class MeshNode : public CsgLeaf {
public:
  std::shared_ptr<Mesh> mesh;

  uint32_t getParamCount() const override { return 0; }
  uint32_t getSelfParamCount() const override { return 0; }
  void setParams(const double*) override {}
  void getParams(double*) const override {}
  void getParamInfo(ParamInfo*) const override {}

  double calcSdf(double3 p) const override {
    return mesh ? mesh->signedDistance(p) : 1e18;
  }
  double calcSdfWithDiff(double3 p, double* /*dparams*/, double* dp) const override {
    dp[0] = dp[1] = dp[2] = 0.0;
    return mesh ? mesh->signedDistance(p) : 1e18;
  }
};

// ============================================================
// 2D Sketch primitives
// ============================================================

class RectNode : public CsgLeaf {
public:
  double2 center{0, 0};
  double2 half_size{0.1, 0.1};

  uint32_t getParamCount() const override { return 4; }
  uint32_t getSelfParamCount() const override { return 4; }

  void setParams(const double* p) override {
    center = {p[0], p[1]};
    half_size = {p[2], p[3]};
  }
  void getParams(double* o) const override {
    o[0] = center.x; o[1] = center.y; o[2] = half_size.x; o[3] = half_size.y;
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD); o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::SIZE);  o[3] = ParamInfo(ParamType::SIZE);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    return sdf_sketch_rect(center, half_size, p, dparams, dp);
  }
  double calcSdf(double3 p) const override {
    double2 rel = abs2(double2(p.x, p.y) - center) - half_size;
    double2 q = max2(rel, double2(0.0));
    return length(q) + std::min(std::max(rel.x, rel.y), 0.0);
  }
};

class CircleNode : public CsgLeaf {
public:
  double2 center{0, 0};
  double radius = 0.1;

  uint32_t getParamCount() const override { return 3; }
  uint32_t getSelfParamCount() const override { return 3; }

  void setParams(const double* p) override {
    center = {p[0], p[1]}; radius = p[2];
  }
  void getParams(double* o) const override {
    o[0] = center.x; o[1] = center.y; o[2] = radius;
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD); o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::SIZE);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    return sdf_sketch_circle(center, radius, p, dparams, dp);
  }
  double calcSdf(double3 p) const override {
    return length(double2(p.x, p.y) - center) - radius;
  }
};

// Slot/stadium: Minkowski sum of a line segment and a disk.
// Params (5): center_x, center_y, length, width, angle
// Internally: half_len = (length - width) / 2, radius = width / 2
// The param interface uses CadQuery's (length, width) so write-back is trivial.
class SlotNode : public CsgLeaf {
public:
  double2 center{0, 0};
  double length = 1.0;   // total slot length (CadQuery convention)
  double width = 0.5;    // total slot width  (CadQuery convention)
  double angle = 0.0;    // radians

  uint32_t getParamCount() const override { return 5; }
  uint32_t getSelfParamCount() const override { return 5; }

  void setParams(const double* p) override {
    center = {p[0], p[1]}; length = p[2]; width = p[3]; angle = p[4];
  }
  void getParams(double* o) const override {
    o[0] = center.x; o[1] = center.y; o[2] = length; o[3] = width; o[4] = angle;
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD); o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::SIZE);  o[3] = ParamInfo(ParamType::SIZE);
    o[4] = ParamInfo(ParamType::ANGLE);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double hl = (length - width) * 0.5;
    double r  = width * 0.5;
    // sdf_sketch_slot outputs grads w.r.t. (cx, cy, half_len, radius, angle)
    // We need to chain-rule to (cx, cy, length, width, angle)
    double raw_dp[5] = {};
    double sdf = sdf_sketch_slot(center, hl, r, angle, p, raw_dp, dp);
    // d(hl)/d(length) = 0.5, d(hl)/d(width) = -0.5
    // d(r)/d(width)   = 0.5
    dparams[0] = raw_dp[0];  // d/d(cx)
    dparams[1] = raw_dp[1];  // d/d(cy)
    dparams[2] = raw_dp[2] * 0.5;  // d/d(length) = d/d(hl) * 0.5
    dparams[3] = raw_dp[2] * (-0.5) + raw_dp[3] * 0.5;  // d/d(width) = d/d(hl)*(-0.5) + d/d(r)*0.5
    dparams[4] = raw_dp[4];  // d/d(angle)
    return sdf;
  }
  double calcSdf(double3 p) const override {
    double hl = (length - width) * 0.5;
    double r  = width * 0.5;
    double ca = std::cos(angle), sa = std::sin(angle);
    double rx = p.x - center.x, ry = p.y - center.y;
    double lx =  ca * rx + sa * ry;
    double ly = -sa * rx + ca * ry;
    double fx = std::clamp(lx, -hl, hl);
    double dx = lx - fx;
    return std::sqrt(dx * dx + ly * ly) - r;
  }
};

// ============================================================
// Extrude: sketch in XY, extrude along Z
// ============================================================

class ExtrudeNode : public CsgUnary {
public:
  double depth = 0.1; // full extent; shape occupies z ∈ [0, depth]

  uint32_t getParamCount() const override { return 1 + child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    depth = p[0];
    child->setParams(p + 1);
  }
  void getParams(double* o) const override {
    o[0] = depth;
    child->getParams(o + 1);
  }
  void getParamInfo(ParamInfo* o) const override {
    // Extrude depth is SIGNED: a downward / -Z extrusion has negative depth.
    // Typing it ParamType::SIZE clamps to [1e-4, 1e6], which forces a negative
    // depth (e.g. -42) to +1e-4 and collapses the solid to a zero-height sheet.
    // Bound it to the sign of the initial depth with a small margin from 0 so
    // the magnitude optimizes freely but the solid can't degenerate or flip.
    if (depth < 0.0)
      o[0] = ParamInfo(ParamType::COORD, -1e6, -1e-4);
    else
      o[0] = ParamInfo(ParamType::COORD, 1e-4, 1e6);
    child->getParamInfo(o + 1);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    // Evaluate child 2D SDF in XY plane
    double child_dp[3] = {};
    uint32_t ls = 1;
    double child_sdf = child->calcSdfWithDiff(double3(p.x, p.y, 0), dparams + ls, child_dp);

    // Compute extrude SDF + its own gradients
    double d_child = 0;
    double sdf = sdf_extrude(depth, p, child_sdf, dparams, dp, &d_child);

    // Chain rule: d(extrude)/dp.x = d(extrude)/d(child_sdf) * d(child_sdf)/dp.x
    dp[0] = d_child * child_dp[0];
    dp[1] = d_child * child_dp[1];
    // dp[2] already set by sdf_extrude

    // Chain rule for child params: d(extrude)/d(child_param) = d_child * d(child_sdf)/d(param)
    uint32_t le = getParamCount();
    for (uint32_t i = ls; i < le; i++)
      dparams[i] *= d_child;

    return sdf;
  }

  double calcSdf(double3 p) const override {
    double d2d = child->calcSdf(double3(p.x, p.y, 0));
    double wx = d2d;
    // 1D SDF to [min(0,depth), max(0,depth)] — works for negative depth too.
    double dmin = std::min(0.0, depth);
    double dmax = std::max(0.0, depth);
    double wy = std::max(dmin - p.z, p.z - dmax);
    double2 w = max2(double2(wx, wy), double2(0.0));
    return length(w) + std::min(std::max(wx, wy), 0.0);
  }
};

// ============================================================
// Boolean operations
// ============================================================

class UnionNode : public CsgBinary {
public:
  uint32_t getParamCount() const override { return left->getParamCount() + right->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }

  void setParams(const double* p) override {
    left->setParams(p);
    right->setParams(p + left->getParamCount());
  }
  void getParams(double* o) const override {
    left->getParams(o);
    right->getParams(o + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    left->getParamInfo(o);
    right->getParamInfo(o + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double dp1[3] = {}, dp2[3] = {};
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double d1 = left->calcSdfWithDiff(p, dparams, dp1);
    double d2 = right->calcSdfWithDiff(p, dparams + lp, dp2);
    if (d1 <= d2) {
      std::fill_n(dparams + lp, rp, 0.0);
      std::copy_n(dp1, 3, dp);
      return d1;
    } else {
      std::fill_n(dparams, lp, 0.0);
      std::copy_n(dp2, 3, dp);
      return d2;
    }
  }
  double calcSdf(double3 p) const override {
    return std::min(left->calcSdf(p), right->calcSdf(p));
  }
};

class IntersectNode : public CsgBinary {
public:
  uint32_t getParamCount() const override { return left->getParamCount() + right->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }

  void setParams(const double* p) override {
    left->setParams(p); right->setParams(p + left->getParamCount());
  }
  void getParams(double* o) const override {
    left->getParams(o); right->getParams(o + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    left->getParamInfo(o); right->getParamInfo(o + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double dp1[3] = {}, dp2[3] = {};
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double d1 = left->calcSdfWithDiff(p, dparams, dp1);
    double d2 = right->calcSdfWithDiff(p, dparams + lp, dp2);
    if (d1 >= d2) {
      std::fill_n(dparams + lp, rp, 0.0);
      std::copy_n(dp1, 3, dp);
      return d1;
    } else {
      std::fill_n(dparams, lp, 0.0);
      std::copy_n(dp2, 3, dp);
      return d2;
    }
  }
  double calcSdf(double3 p) const override {
    return std::max(left->calcSdf(p), right->calcSdf(p));
  }
};

class SubtractNode : public CsgBinary {
public:
  uint32_t getParamCount() const override { return left->getParamCount() + right->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }

  void setParams(const double* p) override {
    left->setParams(p); right->setParams(p + left->getParamCount());
  }
  void getParams(double* o) const override {
    left->getParams(o); right->getParams(o + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    left->getParamInfo(o); right->getParamInfo(o + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double dp1[3] = {}, dp2[3] = {};
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double d1 = left->calcSdfWithDiff(p, dparams, dp1);
    double d2 = right->calcSdfWithDiff(p, dparams + lp, dp2);
    if (d1 >= -d2) {
      std::fill_n(dparams + lp, rp, 0.0);
      std::copy_n(dp1, 3, dp);
      return d1;
    } else {
      for (uint32_t i = lp; i < lp + rp; i++) dparams[i] *= -1.0;
      std::fill_n(dparams, lp, 0.0);
      for (int i = 0; i < 3; i++) dp[i] = -dp2[i];
      return -d2;
    }
  }
  double calcSdf(double3 p) const override {
    return std::max(left->calcSdf(p), -right->calcSdf(p));
  }
};

// ============================================================
// Translate3D: 3 params (sx, sy, sz)
// ============================================================

class Translate3DNode : public CsgUnary {
public:
  double3 shift{0, 0, 0};

  uint32_t getParamCount() const override { return 3 + child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 3; }

  void setParams(const double* p) override {
    shift = {p[0], p[1], p[2]};
    child->setParams(p + 3);
  }
  void getParams(double* o) const override {
    o[0] = shift.x; o[1] = shift.y; o[2] = shift.z;
    child->getParams(o + 3);
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD);
    o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::COORD);
    child->getParamInfo(o + 3);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double child_dp[3] = {};
    double sdf = child->calcSdfWithDiff(p - shift, dparams + 3, child_dp);
    dparams[0] = -child_dp[0];
    dparams[1] = -child_dp[1];
    dparams[2] = -child_dp[2];
    dp[0] = child_dp[0]; dp[1] = child_dp[1]; dp[2] = child_dp[2];
    return sdf;
  }
  double calcSdf(double3 p) const override {
    return child->calcSdf(p - shift);
  }
};

// ============================================================
// Rotate3D: 3 params (angle_x, angle_y, angle_z) Euler ZYX
// ============================================================

class Rotate3DNode : public CsgUnary {
public:
  double3 angles{0, 0, 0};
  Mat3 R, dR_dax, dR_day, dR_daz;

  uint32_t getParamCount() const override { return 3 + child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 3; }

  void setParams(const double* p) override {
    angles = {p[0], p[1], p[2]};
    Mat3 Rx = rotX(angles.x), Ry = rotY(angles.y), Rz = rotZ(angles.z);
    R = Rz * Ry * Rx;
    dR_dax = Rz * Ry * dRotX(angles.x);
    dR_day = Rz * dRotY(angles.y) * Rx;
    dR_daz = dRotZ(angles.z) * Ry * Rx;
    child->setParams(p + 3);
  }
  void getParams(double* o) const override {
    o[0] = angles.x; o[1] = angles.y; o[2] = angles.z;
    child->getParams(o + 3);
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::ANGLE);
    o[1] = ParamInfo(ParamType::ANGLE);
    o[2] = ParamInfo(ParamType::ANGLE);
    child->getParamInfo(o + 3);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double3 p_rot = R * p;
    double child_dp[3] = {};
    double sdf = child->calcSdfWithDiff(p_rot, dparams + 3, child_dp);
    double3 g{child_dp[0], child_dp[1], child_dp[2]};

    double3 dp_dax = dR_dax * p;
    dparams[0] = dot(g, dp_dax);
    double3 dp_day = dR_day * p;
    dparams[1] = dot(g, dp_day);
    double3 dp_daz = dR_daz * p;
    dparams[2] = dot(g, dp_daz);

    double3 dp_out = R.transposed() * g;
    dp[0] = dp_out.x; dp[1] = dp_out.y; dp[2] = dp_out.z;
    return sdf;
  }
  double calcSdf(double3 p) const override {
    return child->calcSdf(R * p);
  }
};

// ============================================================
// Scale3D: 1 param (uniform scale)
// ============================================================

class Scale3DNode : public CsgUnary {
public:
  double scale = 1.0;

  uint32_t getParamCount() const override { return 1 + child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    scale = p[0];
    child->setParams(p + 1);
  }
  void getParams(double* o) const override {
    o[0] = scale;
    child->getParams(o + 1);
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::MULTIPLIER);
    child->getParamInfo(o + 1);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double inv_s = 1.0 / scale;
    double3 ps = p * inv_s;
    double child_dp[3] = {};
    double csdf = child->calcSdfWithDiff(ps, dparams + 1, child_dp);

    dparams[0] = csdf
      + child_dp[0] * (-p.x * inv_s)
      + child_dp[1] * (-p.y * inv_s)
      + child_dp[2] * (-p.z * inv_s);

    dp[0] = child_dp[0]; dp[1] = child_dp[1]; dp[2] = child_dp[2];

    uint32_t le = getParamCount();
    for (uint32_t i = 1; i < le; i++) dparams[i] *= scale;

    return scale * csdf;
  }
  double calcSdf(double3 p) const override {
    return scale * child->calcSdf(p / scale);
  }
};

// ============================================================
// Mirror nodes (0 params, flip one axis)
// ============================================================

class MirrorXYNode : public CsgUnary { // flip Z
public:
  uint32_t getParamCount() const override { return child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }
  void setParams(const double* p) override { child->setParams(p); }
  void getParams(double* o) const override { child->getParams(o); }
  void getParamInfo(ParamInfo* o) const override { child->getParamInfo(o); }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double cdp[3] = {};
    double sdf = child->calcSdfWithDiff({p.x, p.y, -p.z}, dparams, cdp);
    dp[0] = cdp[0]; dp[1] = cdp[1]; dp[2] = -cdp[2];
    return sdf;
  }
  double calcSdf(double3 p) const override { return child->calcSdf({p.x, p.y, -p.z}); }
};

class MirrorXZNode : public CsgUnary { // flip Y
public:
  uint32_t getParamCount() const override { return child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }
  void setParams(const double* p) override { child->setParams(p); }
  void getParams(double* o) const override { child->getParams(o); }
  void getParamInfo(ParamInfo* o) const override { child->getParamInfo(o); }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double cdp[3] = {};
    double sdf = child->calcSdfWithDiff({p.x, -p.y, p.z}, dparams, cdp);
    dp[0] = cdp[0]; dp[1] = -cdp[1]; dp[2] = cdp[2];
    return sdf;
  }
  double calcSdf(double3 p) const override { return child->calcSdf({p.x, -p.y, p.z}); }
};

class MirrorYZNode : public CsgUnary { // flip X
public:
  uint32_t getParamCount() const override { return child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }
  void setParams(const double* p) override { child->setParams(p); }
  void getParams(double* o) const override { child->getParams(o); }
  void getParamInfo(ParamInfo* o) const override { child->getParamInfo(o); }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double cdp[3] = {};
    double sdf = child->calcSdfWithDiff({-p.x, p.y, p.z}, dparams, cdp);
    dp[0] = -cdp[0]; dp[1] = cdp[1]; dp[2] = cdp[2];
    return sdf;
  }
  double calcSdf(double3 p) const override { return child->calcSdf({-p.x, p.y, p.z}); }
};

// ============================================================
// Inverse (negate SDF, 0 params)
// ============================================================

class InverseNode : public CsgUnary {
public:
  uint32_t getParamCount() const override { return child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }
  void setParams(const double* p) override { child->setParams(p); }
  void getParams(double* o) const override { child->getParams(o); }
  void getParamInfo(ParamInfo* o) const override { child->getParamInfo(o); }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double sdf = child->calcSdfWithDiff(p, dparams, dp);
    uint32_t n = getParamCount();
    for (uint32_t i = 0; i < n; i++) dparams[i] = -dparams[i];
    for (int i = 0; i < 3; i++) dp[i] = -dp[i];
    return -sdf;
  }
  double calcSdf(double3 p) const override { return -child->calcSdf(p); }
};

// ============================================================
// Revolve: rotate 2D profile around Y axis (full revolution)
// ============================================================

class RevolveNode : public CsgUnary {
public:
  // axis_along_x == 0  → revolve around a sketch-Y (vertical) axis.
  // axis_along_x == 1  → revolve around a sketch-X (horizontal) axis.
  int axis_along_x = 0;
  // Perpendicular position of the axis in the sketch plane (sketch-u for a
  // sketch-Y axis, sketch-v for a sketch-X axis).  The radial coordinate is
  // measured from here, so OFF-AXIS revolves (axis not through the origin) are
  // faithful instead of being silently revolved about the origin.
  double axis_offset = 0.0;
  // Sweep angle in degrees.  < 360 => partial revolve, intersected with an
  // angular wedge [0, angle] about the axis.
  double angle_deg = 360.0;
  // +1 if the profile lies on the +radial side of the axis (profile plane at
  // phi=0), -1 if on the -side (profile plane at phi=pi).  Orients the wedge.
  double profile_side = 1.0;
  // Rotational direction of the partial-revolve sweep.  cadquery sweeps by the
  // right-hand rule about (axisEnd-axisStart); profile_side flips `a` which
  // reverses the wedge's angular sense, so the parser sets this to
  // -profile_side * sign(axis_dir) to keep the wedge on the side cadquery
  // actually fills (else +side partial revolves miss ~half the solid).
  double wedge_b_sign = 1.0;

  uint32_t getParamCount() const override { return child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 0; }
  void setParams(const double* p) override { child->setParams(p); }
  void getParams(double* o) const override { child->getParams(o); }
  void getParamInfo(ParamInfo* o) const override { child->getParamInfo(o); }

  // Signed distance to the angular wedge [0, theta] (negative inside).  (a,b)
  // are the rotation-plane coords with phi=0 at the profile's start plane.
  // Writes d(sdf)/da, d(sdf)/db.  The wedge is purely geometric (no child
  // params).  Uses the bisector + half-plane line-distance form, exact near the
  // solid (where phi is near [0,theta]).
  static double wedgeSdf(double a, double b, double theta, double& dda, double& ddb) {
    double beta = 0.5 * theta, alpha = 0.5 * theta;
    double cbe = std::cos(beta), sbe = std::sin(beta);
    double cal = std::cos(alpha), sal = std::sin(alpha);
    double a2 =  a * cbe + b * sbe;     // rotate by -beta
    double bp = -a * sbe + b * cbe;
    double sgn = bp >= 0.0 ? 1.0 : -1.0;
    double bb = sgn * bp;               // |bp|
    dda = cal * (sgn * -sbe) - sal * cbe;
    ddb = cal * (sgn *  cbe) - sal * sbe;
    return bb * cal - a2 * sal;
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    uint32_t pc = child->getParamCount();
    double dp_pos[3] = {}, dp_neg[3] = {};
    // Dynamic, not a fixed [64] stack buffer: a revolve PROFILE can have far more
    // than 64 params (complex polygons/arcpolys — a screw has 156).  The old fixed
    // buffer overflowed the stack inside child->calcSdfWithDiff -> abort/segfault
    // in optimize() (the forward eval_sdf was fine; only the gradient overflowed).
    std::vector<double> dparams_pos(pc, 0.0), dparams_neg(pc, 0.0);
    // Rotation-plane coords (a,b): the profile lives at b=0 (phi=0), swept into b.
    double a, b;            // a = signed radial offset along the profile axis, b = out-of-plane
    if (axis_along_x == 0) { a = p.x - axis_offset; b = p.z; }
    else                   { a = p.y - axis_offset; b = p.z; }
    double r = std::sqrt(a * a + b * b);
    double up =  r, un = -r;            // profile coordinate is axis_offset +/- r
    double sdf_pos, sdf_neg;
    if (axis_along_x == 0) {
      sdf_pos = child->calcSdfWithDiff(double3(axis_offset + up, p.y, 0.0), dparams_pos.data(), dp_pos);
      sdf_neg = child->calcSdfWithDiff(double3(axis_offset + un, p.y, 0.0), dparams_neg.data(), dp_neg);
    } else {
      sdf_pos = child->calcSdfWithDiff(double3(p.x, axis_offset + up, 0.0), dparams_pos.data(), dp_pos);
      sdf_neg = child->calcSdfWithDiff(double3(p.x, axis_offset + un, 0.0), dparams_neg.data(), dp_neg);
    }
    bool use_pos = sdf_pos < sdf_neg;
    double sdf = use_pos ? sdf_pos : sdf_neg;
    double *src_dp = use_pos ? dp_pos : dp_neg;
    double *src_dparams = use_pos ? dparams_pos.data() : dparams_neg.data();
    for (uint32_t i = 0; i < pc; i++) dparams[i] = src_dparams[i];
    double sign_r = use_pos ? 1.0 : -1.0;
    // d(profile coord)/d(spatial): child-arg = axis_offset + sign_r*r;
    // dr/da = a/r, dr/db = b/r (a,b are spatial up to the constant offset).
    double da_term, db_term;
    if (r > 1e-15) { double inv_r = 1.0 / r; da_term = sign_r * a * inv_r; db_term = sign_r * b * inv_r; }
    else           { da_term = sign_r; db_term = 0.0; }
    if (axis_along_x == 0) {
      dp[0] = src_dp[0] * da_term;   // a = p.x - off
      dp[1] = src_dp[1];             // axial = p.y
      dp[2] = src_dp[0] * db_term;   // b = p.z
    } else {
      dp[0] = src_dp[0];             // axial = p.x
      dp[1] = src_dp[1] * da_term;   // a = p.y - off
      dp[2] = src_dp[1] * db_term;   // b = p.z
    }
    // Partial revolve: intersect (max) with the angular wedge.
    if (angle_deg < 359.999) {
      double theta = angle_deg * (M_PI / 180.0), dwa, dwb;
      double sw = wedgeSdf(profile_side * a, wedge_b_sign * b, theta, dwa, dwb);
      dwa *= profile_side;       // d/da of wedgeSdf(side*a, ...) = dwa * side
      dwb *= wedge_b_sign;       // d/db of wedgeSdf(.., bsign*b) = dwb * bsign
      if (sw > sdf) {
        sdf = sw;
        for (uint32_t i = 0; i < pc; i++) dparams[i] = 0.0;  // wedge has no child params
        if (axis_along_x == 0) { dp[0] = dwa; dp[1] = 0.0; dp[2] = dwb; }
        else                   { dp[0] = 0.0; dp[1] = dwa; dp[2] = dwb; }
      }
    }
    return sdf;
  }

  double calcSdf(double3 p) const override {
    double a, b;
    if (axis_along_x == 0) { a = p.x - axis_offset; b = p.z; }
    else                   { a = p.y - axis_offset; b = p.z; }
    double r = std::sqrt(a * a + b * b);
    double s1, s2;
    if (axis_along_x == 0) {
      s1 = child->calcSdf(double3(axis_offset + r, p.y, 0.0));
      s2 = child->calcSdf(double3(axis_offset - r, p.y, 0.0));
    } else {
      s1 = child->calcSdf(double3(p.x, axis_offset + r, 0.0));
      s2 = child->calcSdf(double3(p.x, axis_offset - r, 0.0));
    }
    double sdf = std::min(s1, s2);
    if (angle_deg < 359.999) {
      double theta = angle_deg * (M_PI / 180.0), dwa, dwb;
      double sw = wedgeSdf(profile_side * a, wedge_b_sign * b, theta, dwa, dwb);
      sdf = std::max(sdf, sw);
    }
    return sdf;
  }
};

// ============================================================
// Fillet (smooth) boolean operations: 1 self param (radius k)
// ============================================================

class FilletUnionNode : public CsgBinary {
public:
  double radius = 0.1;

  uint32_t getParamCount() const override { return 1 + left->getParamCount() + right->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    radius = p[0];
    left->setParams(p + 1);
    right->setParams(p + 1 + left->getParamCount());
  }
  void getParams(double* o) const override {
    o[0] = radius;
    left->getParams(o + 1);
    right->getParams(o + 1 + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::SIZE);
    left->getParamInfo(o + 1);
    right->getParamInfo(o + 1 + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double dp1[3] = {}, dp2[3] = {};
    double d1 = left->calcSdfWithDiff(p, dparams + 1, dp1);
    double d2 = right->calcSdfWithDiff(p, dparams + 1 + lp, dp2);

    double dsmin_dd1, dsmin_dd2, dsmin_dk;
    double sdf = smooth_min(d1, d2, radius, &dsmin_dd1, &dsmin_dd2, &dsmin_dk);

    dparams[0] = dsmin_dk;
    for (uint32_t i = 0; i < lp; i++) dparams[1 + i] *= dsmin_dd1;
    for (uint32_t i = 0; i < rp; i++) dparams[1 + lp + i] *= dsmin_dd2;
    for (int i = 0; i < 3; i++) dp[i] = dsmin_dd1 * dp1[i] + dsmin_dd2 * dp2[i];
    return sdf;
  }
  double calcSdf(double3 p) const override {
    double d1 = left->calcSdf(p), d2 = right->calcSdf(p);
    double raw_h = 0.5 + 0.5 * (d2 - d1) / radius;
    double h = std::clamp(raw_h, 0.0, 1.0);
    return d2 * (1.0 - h) + d1 * h - radius * h * (1.0 - h);
  }
};

class FilletIntersectNode : public CsgBinary {
public:
  double radius = 0.1;

  uint32_t getParamCount() const override { return 1 + left->getParamCount() + right->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    radius = p[0];
    left->setParams(p + 1);
    right->setParams(p + 1 + left->getParamCount());
  }
  void getParams(double* o) const override {
    o[0] = radius;
    left->getParams(o + 1);
    right->getParams(o + 1 + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::SIZE);
    left->getParamInfo(o + 1);
    right->getParamInfo(o + 1 + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double dp1[3] = {}, dp2[3] = {};
    double d1 = left->calcSdfWithDiff(p, dparams + 1, dp1);
    double d2 = right->calcSdfWithDiff(p, dparams + 1 + lp, dp2);

    double dsmax_dd1, dsmax_dd2, dsmax_dk;
    double sdf = smooth_max(d1, d2, radius, &dsmax_dd1, &dsmax_dd2, &dsmax_dk);

    dparams[0] = dsmax_dk;
    for (uint32_t i = 0; i < lp; i++) dparams[1 + i] *= dsmax_dd1;
    for (uint32_t i = 0; i < rp; i++) dparams[1 + lp + i] *= dsmax_dd2;
    for (int i = 0; i < 3; i++) dp[i] = dsmax_dd1 * dp1[i] + dsmax_dd2 * dp2[i];
    return sdf;
  }
  double calcSdf(double3 p) const override {
    double nd1 = -left->calcSdf(p), nd2 = -right->calcSdf(p);
    double raw_h = 0.5 + 0.5 * (nd2 - nd1) / radius;
    double h = std::clamp(raw_h, 0.0, 1.0);
    return -(nd2 * (1.0 - h) + nd1 * h - radius * h * (1.0 - h));
  }
};

// ============================================================
// Chamfer intersection: 1 self param (size)
// ============================================================

class ChamferIntersectNode : public CsgBinary {
public:
  double size = 0.1;

  uint32_t getParamCount() const override { return 1 + left->getParamCount() + right->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    size = p[0];
    left->setParams(p + 1);
    right->setParams(p + 1 + left->getParamCount());
  }
  void getParams(double* o) const override {
    o[0] = size;
    left->getParams(o + 1);
    right->getParams(o + 1 + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::SIZE);
    left->getParamInfo(o + 1);
    right->getParamInfo(o + 1 + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double dp1[3] = {}, dp2[3] = {};
    double d1 = left->calcSdfWithDiff(p, dparams + 1, dp1);
    double d2 = right->calcSdfWithDiff(p, dparams + 1 + lp, dp2);

    double dcham_dd1, dcham_dd2, dcham_dsize;
    double sdf = chamfer_intersect(d1, d2, size, &dcham_dd1, &dcham_dd2, &dcham_dsize);

    dparams[0] = dcham_dsize;
    for (uint32_t i = 0; i < lp; i++) dparams[1 + i] *= dcham_dd1;
    for (uint32_t i = 0; i < rp; i++) dparams[1 + lp + i] *= dcham_dd2;
    for (int i = 0; i < 3; i++) dp[i] = dcham_dd1 * dp1[i] + dcham_dd2 * dp2[i];
    return sdf;
  }
  double calcSdf(double3 p) const override {
    static const double INV_SQRT2 = 1.0 / std::sqrt(2.0);
    double d1 = left->calcSdf(p), d2 = right->calcSdf(p);
    double d3 = (d1 + d2) * INV_SQRT2 - size;
    return std::max({d1, d2, d3});
  }
};

// ============================================================
// Offset: expand/contract SDF boundary, 1 self param (offset)
// sdf = child_sdf - offset
// Used for: rounded box (box.fillet), offset2D, etc.
// ============================================================

class OffsetNode : public CsgUnary {
public:
  double offset = 0.0;

  uint32_t getParamCount() const override { return 1 + child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    offset = p[0];
    child->setParams(p + 1);
  }
  void getParams(double* o) const override {
    o[0] = offset;
    child->getParams(o + 1);
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::SIZE);
    child->getParamInfo(o + 1);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double child_dp[3] = {};
    double csdf = child->calcSdfWithDiff(p, dparams + 1, child_dp);
    dparams[0] = -1.0;  // d(sdf)/d(offset) = -1
    dp[0] = child_dp[0]; dp[1] = child_dp[1]; dp[2] = child_dp[2];
    return csdf - offset;
  }
  double calcSdf(double3 p) const override {
    return child->calcSdf(p) - offset;
  }
};

// ============================================================
// Shell: hollow out a solid, 1 self param (thickness)
// ============================================================

class ShellNode : public CsgUnary {
public:
  double thickness = 0.1;

  uint32_t getParamCount() const override { return 1 + child->getParamCount(); }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    thickness = p[0];
    child->setParams(p + 1);
  }
  void getParams(double* o) const override {
    o[0] = thickness;
    child->getParams(o + 1);
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::SIZE);
    child->getParamInfo(o + 1);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double child_dp[3] = {};
    double csdf = child->calcSdfWithDiff(p, dparams + 1, child_dp);
    double sign_csdf = (csdf >= 0.0) ? 1.0 : -1.0;

    dparams[0] = -0.5;
    uint32_t le = getParamCount();
    for (uint32_t i = 1; i < le; i++) dparams[i] *= sign_csdf;
    for (int i = 0; i < 3; i++) dp[i] = sign_csdf * child_dp[i];
    return std::abs(csdf) - thickness * 0.5;
  }
  double calcSdf(double3 p) const override {
    return std::abs(child->calcSdf(p)) - thickness * 0.5;
  }
};

// ============================================================
// 3D Leaf primitives: Sphere, Box, Cylinder
// ============================================================

class SphereNode : public CsgLeaf {
public:
  double3 center{0, 0, 0};
  double radius = 0.1;

  uint32_t getParamCount() const override { return 4; }
  uint32_t getSelfParamCount() const override { return 4; }

  void setParams(const double* p) override {
    center = {p[0], p[1], p[2]}; radius = p[3];
  }
  void getParams(double* o) const override {
    o[0] = center.x; o[1] = center.y; o[2] = center.z; o[3] = radius;
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD); o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::COORD); o[3] = ParamInfo(ParamType::SIZE);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    return sdf_sphere3d(center, radius, p, dparams, dp);
  }
  double calcSdf(double3 p) const override {
    return length(p - center) - radius;
  }
};

class BoxNode : public CsgLeaf {
public:
  double3 center{0, 0, 0};
  double3 half_size{0.1, 0.1, 0.1};

  uint32_t getParamCount() const override { return 6; }
  uint32_t getSelfParamCount() const override { return 6; }

  void setParams(const double* p) override {
    center = {p[0], p[1], p[2]}; half_size = {p[3], p[4], p[5]};
  }
  void getParams(double* o) const override {
    o[0] = center.x; o[1] = center.y; o[2] = center.z;
    o[3] = half_size.x; o[4] = half_size.y; o[5] = half_size.z;
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD); o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::COORD);
    o[3] = ParamInfo(ParamType::SIZE);  o[4] = ParamInfo(ParamType::SIZE);
    o[5] = ParamInfo(ParamType::SIZE);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    return sdf_box3d(center, half_size, p, dparams, dp);
  }
  double calcSdf(double3 p) const override {
    double3 d{std::abs(p.x - center.x) - half_size.x,
              std::abs(p.y - center.y) - half_size.y,
              std::abs(p.z - center.z) - half_size.z};
    double3 q{std::max(d.x, 0.0), std::max(d.y, 0.0), std::max(d.z, 0.0)};
    return length(q) + std::min(std::max({d.x, d.y, d.z}), 0.0);
  }
};

class CylinderNode : public CsgLeaf {
public:
  double3 center{0, 0, 0};
  double radius = 0.1;
  double half_height = 0.1;

  uint32_t getParamCount() const override { return 5; }
  uint32_t getSelfParamCount() const override { return 5; }

  void setParams(const double* p) override {
    center = {p[0], p[1], p[2]}; radius = p[3]; half_height = p[4];
  }
  void getParams(double* o) const override {
    o[0] = center.x; o[1] = center.y; o[2] = center.z;
    o[3] = radius; o[4] = half_height;
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::COORD); o[1] = ParamInfo(ParamType::COORD);
    o[2] = ParamInfo(ParamType::COORD);
    o[3] = ParamInfo(ParamType::SIZE);  o[4] = ParamInfo(ParamType::SIZE);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    return sdf_cylinder3d(center, radius, half_height, p, dparams, dp);
  }
  double calcSdf(double3 p) const override {
    double dx = p.x - center.x, dy = p.y - center.y;
    double r_xy = std::sqrt(dx*dx + dy*dy) - radius;
    double h = std::abs(p.z - center.z) - half_height;
    double qr = std::max(r_xy, 0.0), qh = std::max(h, 0.0);
    return std::sqrt(qr*qr + qh*qh) + std::min(std::max(r_xy, h), 0.0);
  }
};

// ============================================================
// Loft: linear interpolation between two 2D profiles along Z
// Binary node: left = bottom profile (z = -half_height),
//              right = top profile (z = +half_height)
// 1 self param: half_height
// ============================================================

class LoftNode : public CsgBinary {
public:
  double half_height = 0.1;

  uint32_t getParamCount() const override {
    return 1 + left->getParamCount() + right->getParamCount();
  }
  uint32_t getSelfParamCount() const override { return 1; }

  void setParams(const double* p) override {
    half_height = p[0];
    left->setParams(p + 1);
    right->setParams(p + 1 + left->getParamCount());
  }
  void getParams(double* o) const override {
    o[0] = half_height;
    left->getParams(o + 1);
    right->getParams(o + 1 + left->getParamCount());
  }
  void getParamInfo(ParamInfo* o) const override {
    o[0] = ParamInfo(ParamType::SIZE);
    left->getParamInfo(o + 1);
    right->getParamInfo(o + 1 + left->getParamCount());
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    uint32_t lp = left->getParamCount(), rp = right->getParamCount();
    double dp_bot[3] = {}, dp_top[3] = {};
    double sdf_bot = left->calcSdfWithDiff({p.x, p.y, 0}, dparams + 1, dp_bot);
    double sdf_top = right->calcSdfWithDiff({p.x, p.y, 0}, dparams + 1 + lp, dp_top);

    double hh2 = 2.0 * half_height;
    double t_raw = (hh2 > 1e-15) ? (p.z + half_height) / hh2 : 0.5;
    bool clamped = (t_raw <= 0.0 || t_raw >= 1.0);
    double t = std::clamp(t_raw, 0.0, 1.0);

    // Interpolated 2D SDF (like child_sdf in extrude)
    double wx = (1.0 - t) * sdf_bot + t * sdf_top;
    double wy = std::abs(p.z) - half_height;
    double sign_z = (p.z >= 0.0) ? 1.0 : -1.0;

    // Extrude combination: same as sdf_extrude
    double wx_max = std::max(wx, 0.0);
    double wy_max = std::max(wy, 0.0);
    double dist_out = std::sqrt(wx_max * wx_max + wy_max * wy_max);
    double sdf = dist_out + std::min(std::max(wx, wy), 0.0);

    // Gradient of outer extrude w.r.t. wx and wy
    double dsdf_dwx, dsdf_dwy;
    bool out_2d = wx > 0.0, out_z = wy > 0.0;
    if (out_2d && out_z) {
      double inv = (dist_out > 1e-15) ? 1.0 / dist_out : 0.0;
      dsdf_dwx = wx_max * inv;
      dsdf_dwy = wy_max * inv;
    } else if (out_2d || (!out_2d && !out_z && wx > wy)) {
      dsdf_dwx = 1.0; dsdf_dwy = 0.0;
    } else {
      dsdf_dwx = 0.0; dsdf_dwy = 1.0;
    }

    // d(t)/d(half_height) = -p.z / (2 * hh²) when not clamped
    double dt_dhh = (!clamped && hh2 > 1e-15) ? (-p.z / (half_height * hh2)) : 0.0;
    double dwx_dhh = (sdf_top - sdf_bot) * dt_dhh;  // d(wx)/d(hh)
    dparams[0] = dsdf_dwx * dwx_dhh + dsdf_dwy * (-1.0);

    // Child param gradients
    for (uint32_t i = 0; i < lp; i++)
      dparams[1 + i] *= dsdf_dwx * (1.0 - t);
    for (uint32_t i = 0; i < rp; i++)
      dparams[1 + lp + i] *= dsdf_dwx * t;

    // Spatial gradients
    double dwx_dpx = (1.0 - t) * dp_bot[0] + t * dp_top[0];
    double dwx_dpy = (1.0 - t) * dp_bot[1] + t * dp_top[1];
    double dt_dpz = (!clamped && hh2 > 1e-15) ? 1.0 / hh2 : 0.0;
    double dwx_dpz = (sdf_top - sdf_bot) * dt_dpz;

    dp[0] = dsdf_dwx * dwx_dpx;
    dp[1] = dsdf_dwx * dwx_dpy;
    dp[2] = dsdf_dwx * dwx_dpz + dsdf_dwy * sign_z;

    return sdf;
  }

  double calcSdf(double3 p) const override {
    double hh2 = 2.0 * half_height;
    double t = (hh2 > 1e-15) ? std::clamp((p.z + half_height) / hh2, 0.0, 1.0) : 0.5;
    double d2d = (1.0 - t) * left->calcSdf({p.x, p.y, 0}) + t * right->calcSdf({p.x, p.y, 0});
    double wy = std::abs(p.z) - half_height;
    double2 w = max2({d2d, wy}, {0.0, 0.0});
    return length(w) + std::min(std::max(d2d, wy), 0.0);
  }
};

// ============================================================
// Sweep: sweep a 2D profile along a 3D polyline path
// Self params: path vertices (3 * n_path_verts doubles)
// Child: 2D profile
// The profile is evaluated in the plane perpendicular to the path
// at the closest point on the polyline.
// ============================================================

class SweepNode : public CsgUnary {
public:
  int n_path_verts = 2;
  std::vector<double3> path;
  std::vector<double3> seg_n, seg_b;   // per-segment frame
  bool use_frenet = false;             // isFrenet=True sweeps -> Frenet frame
  double profile_angle = 0.0;          // rotate profile within frame plane (workplane convention)

  SweepNode() = default;
  SweepNode(int n) : n_path_verts(n), path(n) {}

  uint32_t getParamCount() const override {
    return 3 * n_path_verts + child->getParamCount();
  }
  uint32_t getSelfParamCount() const override { return 3 * n_path_verts; }

  void setParams(const double* p) override {
    for (int i = 0; i < n_path_verts; i++)
      path[i] = {p[3*i], p[3*i+1], p[3*i+2]};
    child->setParams(p + 3 * n_path_verts);
    computeFrames();
  }
  void getParams(double* o) const override {
    for (int i = 0; i < n_path_verts; i++) {
      o[3*i] = path[i].x; o[3*i+1] = path[i].y; o[3*i+2] = path[i].z;
    }
    child->getParams(o + 3 * n_path_verts);
  }
  void getParamInfo(ParamInfo* o) const override {
    for (int i = 0; i < 3 * n_path_verts; i++)
      o[i] = ParamInfo(ParamType::COORD);
    child->getParamInfo(o + 3 * n_path_verts);
  }

  // Find closest point on polyline, return segment index and parameter t
  void findClosest(double3 p, int& best_seg, double& best_t, double3& closest) const {
    double min_dist2 = 1e30;
    best_seg = 0; best_t = 0; closest = path[0];
    for (int i = 0; i < n_path_verts - 1; i++) {
      double3 ab = path[i+1] - path[i];
      double len2 = dot(ab, ab);
      double t = (len2 > 1e-12) ? std::clamp(dot(p - path[i], ab) / len2, 0.0, 1.0) : 0.0;
      double3 c = path[i] + ab * t;
      double d2 = dot(p - c, p - c);
      if (d2 < min_dist2) {
        min_dist2 = d2; best_seg = i; best_t = t; closest = c;
      }
    }
  }

  // Rotation-minimizing (parallel-transport) frame per segment, via the
  // double-reflection method (Wang et al. 2008).  A fixed up-vector frame
  // twists the profile along a curved path (a helical spring); RMF keeps the
  // profile orientation consistent as CadQuery's sweep does.
  void computeFrames() {
    int nseg = std::max(0, n_path_verts - 1);
    seg_n.assign(nseg, {0,0,0}); seg_b.assign(nseg, {0,0,0});
    if (nseg == 0) return;
    if (use_frenet) {
      // Frenet frame: normal = curvature direction (dT/ds).  On a helix this
      // points toward the axis (radial) and ROTATES with the coil, matching
      // CadQuery's sweep(..., isFrenet=True).  RMF (below) deliberately does NOT
      // rotate, so it mis-orients asymmetric profiles (flat rects) on helices.
      std::vector<double3> T(nseg);
      for (int i = 0; i < nseg; i++) T[i] = normalize(path[i+1] - path[i]);
      for (int i = 0; i < nseg; i++) {
        double3 nrm = T[std::min(i+1, nseg-1)] - T[std::max(i-1, 0)];  // dT
        nrm = nrm - T[i] * dot(nrm, T[i]);                            // _|_ tangent
        double L = length(nrm);
        if (L < 1e-9) {  // straight: any perpendicular reference
          double3 up = (std::abs(T[i].z) < 0.9) ? double3{0,0,1} : double3{1,0,0};
          nrm = up - T[i] * dot(up, T[i]); L = length(nrm);
        }
        double3 n_i = nrm * (1.0 / std::max(L, 1e-12));
        seg_n[i] = n_i; seg_b[i] = cross(T[i], n_i);
      }
      return;
    }
    double3 t0 = normalize(path[1] - path[0]);
    double3 up = (std::abs(t0.z) < 0.9) ? double3{0,0,1} : double3{1,0,0};
    double3 n0 = normalize(up - t0 * dot(up, t0));
    seg_n[0] = n0; seg_b[0] = cross(t0, n0);
    double3 t_prev = t0, n_prev = n0, x_prev = (path[0] + path[1]) * 0.5;
    for (int i = 1; i < nseg; i++) {
      double3 t_i = normalize(path[i+1] - path[i]);
      double3 x_i = (path[i] + path[i+1]) * 0.5;
      double3 v1 = x_i - x_prev; double c1 = dot(v1, v1);
      double3 nL = n_prev, tL = t_prev;
      if (c1 > 1e-18) {
        nL = n_prev - v1 * (2.0/c1 * dot(v1, n_prev));
        tL = t_prev - v1 * (2.0/c1 * dot(v1, t_prev));
      }
      double3 v2 = t_i - tL; double c2 = dot(v2, v2);
      double3 n_i = (c2 > 1e-18) ? (nL - v2 * (2.0/c2 * dot(v2, nL))) : nL;
      n_i = normalize(n_i - t_i * dot(n_i, t_i));
      seg_n[i] = n_i; seg_b[i] = cross(t_i, n_i);
      t_prev = t_i; n_prev = n_i; x_prev = x_i;
    }
  }

  // Local frame at segment `seg`: tangent, normal, binormal (RMF).
  void buildFrame(int seg, double3& tangent, double3& normal, double3& binormal) const {
    tangent = normalize(path[seg+1] - path[seg]);
    if (seg < (int)seg_n.size() && dot(seg_n[seg], seg_n[seg]) > 0.5) {
      normal = seg_n[seg]; binormal = seg_b[seg];
    } else {
      double3 up = (std::abs(dot(tangent, double3{0,0,1})) < 0.99) ? double3{0,0,1} : double3{0,1,0};
      normal = normalize(cross(up, tangent)); binormal = cross(tangent, normal);
    }
    if (profile_angle != 0.0) {   // rotate profile orientation within the frame plane
      double c = std::cos(profile_angle), s = std::sin(profile_angle);
      double3 n2 = normal * c + binormal * s;
      double3 b2 = binormal * c - normal * s;
      normal = n2; binormal = b2;
    }
  }

  // Axial "beyond the cap" distance: >0 only past the two GLOBAL path ends,
  // so the swept tube becomes finite with proper end caps.
  double axialOut(int seg, double t, double3 delta, double3 tan) const {
    if (seg == 0 && t <= 1e-9) return std::max(0.0, -dot(delta, tan));
    if (seg == n_path_verts - 2 && t >= 1.0 - 1e-9) return std::max(0.0, dot(delta, tan));
    return 0.0;
  }

  double calcSdf(double3 p) const override {
    int seg; double t; double3 closest;
    findClosest(p, seg, t, closest);
    double3 tan, norm, binorm;
    buildFrame(seg, tan, norm, binorm);
    double3 delta = p - closest;
    double d2d = child->calcSdf({dot(delta, norm), dot(delta, binorm), 0});
    double ao = axialOut(seg, t, delta, tan);
    if (ao > 0.0) return (d2d > 0.0) ? std::sqrt(d2d*d2d + ao*ao) : ao;
    return d2d;
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    int seg; double t; double3 closest;
    findClosest(p, seg, t, closest);
    double3 tan, norm, binorm;
    buildFrame(seg, tan, norm, binorm);

    double3 delta = p - closest;
    double local_x = dot(delta, norm);
    double local_y = dot(delta, binorm);

    uint32_t sp = 3 * n_path_verts;
    double child_dp[3] = {};
    double d2d = child->calcSdfWithDiff({local_x, local_y, 0}, dparams + sp, child_dp);
    double ds_dlx = child_dp[0], ds_dly = child_dp[1];

    // Axial end cap: past a global end the sdf blends the profile distance with
    // the axial overshoot.  dS_dd2d / dS_dao are the chain-rule factors.
    double ao = axialOut(seg, t, delta, tan);
    double sdf = d2d, dS_dd2d = 1.0, dS_dao = 0.0;
    if (ao > 0.0) {
      if (d2d > 0.0) { sdf = std::sqrt(d2d*d2d + ao*ao); dS_dd2d = d2d/sdf; dS_dao = ao/sdf; }
      else           { sdf = ao;                          dS_dd2d = 0.0;     dS_dao = 1.0; }
    }
    double axdir = (seg == 0 && t <= 1e-9) ? -1.0 : 1.0;

    // Spatial gradient = scaled profile part + axial part
    dp[0] = dS_dd2d * (ds_dlx * norm.x + ds_dly * binorm.x) + dS_dao * axdir * tan.x;
    dp[1] = dS_dd2d * (ds_dlx * norm.y + ds_dly * binorm.y) + dS_dao * axdir * tan.y;
    dp[2] = dS_dd2d * (ds_dlx * norm.z + ds_dly * binorm.z) + dS_dao * axdir * tan.z;

    // Scale child (profile) param gradients by the cap factor.
    uint32_t cpc = child->getParamCount();
    for (uint32_t k = 0; k < cpc; k++) dparams[sp + k] *= dS_dd2d;

    // Path vertex gradients (approximate; path is typically frozen).
    double3 ds_dclosest;
    ds_dclosest.x = -dS_dd2d * (ds_dlx * norm.x + ds_dly * binorm.x);
    ds_dclosest.y = -dS_dd2d * (ds_dlx * norm.y + ds_dly * binorm.y);
    ds_dclosest.z = -dS_dd2d * (ds_dlx * norm.z + ds_dly * binorm.z);
    std::fill_n(dparams, sp, 0.0);
    int a = seg, b = seg + 1;
    dparams[3*a]   = (1.0 - t) * ds_dclosest.x;
    dparams[3*a+1] = (1.0 - t) * ds_dclosest.y;
    dparams[3*a+2] = (1.0 - t) * ds_dclosest.z;
    dparams[3*b]   = t * ds_dclosest.x;
    dparams[3*b+1] = t * ds_dclosest.y;
    dparams[3*b+2] = t * ds_dclosest.z;
    return sdf;
  }
};

// ============================================================
// Polygon: closed 2D polyline sketch primitive (variable params)
// ============================================================

class PolygonNode : public CsgLeaf {
public:
  std::vector<double2> vertices;

  PolygonNode() = default;
  PolygonNode(int n) : vertices(n) {}

  uint32_t getParamCount() const override { return 2 * (uint32_t)vertices.size(); }
  uint32_t getSelfParamCount() const override { return 2 * (uint32_t)vertices.size(); }

  void setParams(const double* p) override {
    for (size_t i = 0; i < vertices.size(); i++)
      vertices[i] = {p[2*i], p[2*i+1]};
  }
  void getParams(double* o) const override {
    for (size_t i = 0; i < vertices.size(); i++) {
      o[2*i] = vertices[i].x; o[2*i+1] = vertices[i].y;
    }
  }
  void getParamInfo(ParamInfo* o) const override {
    for (size_t i = 0; i < vertices.size() * 2; i++)
      o[i] = ParamInfo(ParamType::COORD);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double dp2d[2] = {};
    double sdf = sdf_polygon(vertices.data(), (int)vertices.size(),
                             {p.x, p.y}, dparams, dp2d);
    dp[0] = dp2d[0]; dp[1] = dp2d[1]; dp[2] = 0.0;
    return sdf;
  }
  double calcSdf(double3 p) const override {
    // Quick unsigned distance + sign via winding
    int n = (int)vertices.size();
    double min_dist_sq = 1e30;
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % n;
      double2 a = vertices[i], b = vertices[j];
      double2 e = b - a, w = {p.x - a.x, p.y - a.y};
      double ee = dot(e, e);
      double t = (ee > 1e-12) ? std::clamp(dot(w, e) / ee, 0.0, 1.0) : 0.0;
      double2 d = w - e * t;
      min_dist_sq = std::min(min_dist_sq, dot(d, d));
    }
    double sign = 1.0;
    for (int i = 0, j = n-1; i < n; j = i, i++) {
      double2 vi = vertices[i], vj = vertices[j];
      bool c1 = p.y >= vi.y, c2 = p.y < vj.y;
      double ex = vj.x - vi.x, ey = vj.y - vi.y;
      double wx = p.x - vi.x, wy = p.y - vi.y;
      bool c3 = ex * wy > ey * wx;
      if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) sign *= -1.0;
    }
    return sign * std::sqrt(min_dist_sq);
  }
};

// Arc-aware closed profile.  Same param contract as PolygonNode (2N vertex coords);
// per-edge (is_arc, r_s, side) are constants set at build time.  Lets the optimizer
// tune the shared endpoint vertices of arc edges (the dominant boundary error in
// resolution-limited VLM sketches) without baking arcs into fixed polylines.
class ArcPolyNode : public CsgLeaf {
public:
  std::vector<double2> vertices;
  std::vector<ArcEdge> edges;   // edges.size() == vertices.size()

  ArcPolyNode() = default;
  ArcPolyNode(int n) : vertices(n), edges(n) {}

  uint32_t getParamCount() const override { return 2 * (uint32_t)vertices.size(); }
  uint32_t getSelfParamCount() const override { return 2 * (uint32_t)vertices.size(); }

  void setParams(const double* p) override {
    for (size_t i = 0; i < vertices.size(); i++)
      vertices[i] = {p[2*i], p[2*i+1]};
  }
  void getParams(double* o) const override {
    for (size_t i = 0; i < vertices.size(); i++) {
      o[2*i] = vertices[i].x; o[2*i+1] = vertices[i].y;
    }
  }
  void getParamInfo(ParamInfo* o) const override {
    for (size_t i = 0; i < vertices.size() * 2; i++)
      o[i] = ParamInfo(ParamType::COORD);
  }

  double calcSdfWithDiff(double3 p, double* dparams, double* dp) const override {
    double dp2d[2] = {};
    double sdf = sdf_arcpoly(vertices.data(), (int)vertices.size(), edges.data(),
                             {p.x, p.y}, dparams, dp2d);
    dp[0] = dp2d[0]; dp[1] = dp2d[1]; dp[2] = 0.0;
    return sdf;
  }
  double calcSdf(double3 p) const override {
    double dummy_dp[2] = {};
    std::vector<double> dummy(2 * vertices.size());
    return sdf_arcpoly(vertices.data(), (int)vertices.size(), edges.data(),
                       {p.x, p.y}, dummy.data(), dummy_dp);
  }
};

} // namespace cadopt
