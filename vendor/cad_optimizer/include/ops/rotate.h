#pragma once
#include "types.h"
#include <cmath>

namespace cadopt {

// 3x3 rotation matrix (row-major)
struct Mat3 {
  double m[9] = {1,0,0, 0,1,0, 0,0,1};

  double3 operator*(double3 v) const {
    return {m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z};
  }

  Mat3 operator*(const Mat3& b) const {
    Mat3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        r.m[i*3+j] = 0;
        for (int k = 0; k < 3; k++)
          r.m[i*3+j] += m[i*3+k] * b.m[k*3+j];
      }
    return r;
  }

  Mat3 transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r.m[i*3+j] = m[j*3+i];
    return r;
  }
};

inline Mat3 rotX(double a) {
  double c = std::cos(a), s = std::sin(a);
  return {1,0,0, 0,c,-s, 0,s,c};
}
inline Mat3 rotY(double a) {
  double c = std::cos(a), s = std::sin(a);
  return {c,0,s, 0,1,0, -s,0,c};
}
inline Mat3 rotZ(double a) {
  double c = std::cos(a), s = std::sin(a);
  return {c,-s,0, s,c,0, 0,0,1};
}

inline Mat3 dRotX(double a) {
  double c = std::cos(a), s = std::sin(a);
  return {0,0,0, 0,-s,-c, 0,c,-s};
}
inline Mat3 dRotY(double a) {
  double c = std::cos(a), s = std::sin(a);
  return {-s,0,c, 0,0,0, -c,0,-s};
}
inline Mat3 dRotZ(double a) {
  double c = std::cos(a), s = std::sin(a);
  return {-s,-c,0, c,-s,0, 0,0,0};
}

} // namespace cadopt
