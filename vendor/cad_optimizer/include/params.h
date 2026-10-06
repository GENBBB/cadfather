#pragma once
#include "types.h"
#include <vector>
#include <cmath>

namespace cadopt {

enum class ParamType {
  SIZE,        // positive, bounded (e.g., radius, thickness)
  COORD,       // can be negative, bounded by scene extents
  ANGLE,       // radians, [-PI, PI]
  MULTIPLIER,  // positive, larger range than SIZE
};

struct ParamInfo {
  ParamType type = ParamType::SIZE;
  double lo = 1e-4;
  double hi = 1e6;

  ParamInfo() = default;
  ParamInfo(ParamType t) : type(t) {
    switch (t) {
    case ParamType::SIZE:       lo = 1e-4; hi = 1e6; break;
    case ParamType::COORD:      lo = -1e6; hi = 1e6; break;
    case ParamType::ANGLE:      lo = -2 * M_PI; hi = 2 * M_PI; break;
    case ParamType::MULTIPLIER: lo = 1e-4; hi = 1e6; break;
    }
  }
  ParamInfo(ParamType t, double lo_, double hi_) : type(t), lo(lo_), hi(hi_) {}
};

} // namespace cadopt
