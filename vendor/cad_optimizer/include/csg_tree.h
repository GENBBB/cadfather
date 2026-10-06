#pragma once
#include "types.h"
#include "params.h"
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>

namespace cadopt {

// Base class for all CSG tree nodes.
// Every node implements calcSdfWithDiff() returning SDF value
// and writing analytical gradients into the provided buffers.
class CsgNode {
public:
  virtual ~CsgNode() = default;

  // Total parameter count in this subtree (self + all children)
  virtual uint32_t getParamCount() const = 0;

  // Self parameter count (owned by this node only)
  virtual uint32_t getSelfParamCount() const = 0;

  // Set parameters in DFS order: self first, then children left-to-right
  virtual void setParams(const double* params) = 0;

  // Get current parameters in DFS order
  virtual void getParams(double* out) const = 0;

  // Get parameter info (type + limits) in DFS order
  virtual void getParamInfo(ParamInfo* out) const = 0;

  // Evaluate SDF and compute analytical gradients.
  //   p:             query point
  //   dcsg_dparams:  [getParamCount()] output -- d(sdf)/d(param_i) for all subtree params
  //   dcsg_dp:       [3] output -- d(sdf)/d(p.x), d(sdf)/d(p.y), d(sdf)/d(p.z)
  //   returns:       signed distance at p
  virtual double calcSdfWithDiff(double3 p, double* dcsg_dparams, double* dcsg_dp) const = 0;

  // Evaluate SDF only (no gradients, faster)
  virtual double calcSdf(double3 p) const = 0;

  // Helper: get params as vector
  std::vector<double> getParamsVec() const {
    std::vector<double> v(getParamCount());
    getParams(v.data());
    return v;
  }

  // Helper: get param info as vector
  std::vector<ParamInfo> getParamInfoVec() const {
    std::vector<ParamInfo> v(getParamCount());
    getParamInfo(v.data());
    return v;
  }
};

// Leaf node: no children (2D sketch primitives)
class CsgLeaf : public CsgNode {};

// Unary node: one child
class CsgUnary : public CsgNode {
public:
  std::unique_ptr<CsgNode> child;
};

// Binary node: two children
class CsgBinary : public CsgNode {
public:
  std::unique_ptr<CsgNode> left, right;
};

// Convenience: build a CsgLiteTree-like wrapper
struct CsgTree {
  std::unique_ptr<CsgNode> root;
  std::vector<double> params;
  std::vector<ParamInfo> param_info;

  void syncFromRoot() {
    if (!root) return;
    params.resize(root->getParamCount());
    param_info.resize(root->getParamCount());
    root->getParams(params.data());
    root->getParamInfo(param_info.data());
  }

  void syncToRoot() {
    if (!root) return;
    root->setParams(params.data());
  }
};

} // namespace cadopt
