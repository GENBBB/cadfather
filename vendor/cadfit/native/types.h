#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace cadopt {

struct double2 {
  double x = 0, y = 0;

  double2() = default;
  constexpr double2(double x_, double y_) : x(x_), y(y_) {}
  explicit constexpr double2(double v) : x(v), y(v) {}

  double2 operator+(double2 b) const { return {x + b.x, y + b.y}; }
  double2 operator-(double2 b) const { return {x - b.x, y - b.y}; }
  double2 operator*(double s) const { return {x * s, y * s}; }
  double2 operator*(double2 b) const { return {x * b.x, y * b.y}; }
  double2 operator/(double s) const { return {x / s, y / s}; }
  double2 operator-() const { return {-x, -y}; }
  double2& operator+=(double2 b) { x += b.x; y += b.y; return *this; }
  double2& operator-=(double2 b) { x -= b.x; y -= b.y; return *this; }
  double2& operator*=(double s) { x *= s; y *= s; return *this; }
};

inline double2 operator*(double s, double2 v) { return {s * v.x, s * v.y}; }
inline double dot(double2 a, double2 b) { return a.x * b.x + a.y * b.y; }
inline double length(double2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }
inline double2 abs2(double2 v) { return {std::abs(v.x), std::abs(v.y)}; }
inline double2 max2(double2 a, double2 b) { return {std::max(a.x, b.x), std::max(a.y, b.y)}; }
// Helpers used by the arc-profile SDF (ops/arcpoly).
inline double cross_2D(double2 a, double2 b) { return a.x * b.y - a.y * b.x; }
inline double2 normalize2(double2 v) {
  double l = std::sqrt(v.x * v.x + v.y * v.y);
  return l > 1e-12 ? double2{v.x / l, v.y / l} : double2{1.0, 0.0};
}
// length + d|v|/dv (= v/|v|), parallel to the reference's length_diff.
inline double length_diff(double2 v, double2* dlen_dv) {
  double l = std::sqrt(v.x * v.x + v.y * v.y);
  if (l > 1e-12) *dlen_dv = {v.x / l, v.y / l};
  else *dlen_dv = {0.0, 0.0};
  return l;
}

struct double3 {
  double x = 0, y = 0, z = 0;

  double3() = default;
  constexpr double3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  explicit constexpr double3(double v) : x(v), y(v), z(v) {}

  double& operator[](int i) { return (&x)[i]; }
  double operator[](int i) const { return (&x)[i]; }

  double3 operator+(double3 b) const { return {x + b.x, y + b.y, z + b.z}; }
  double3 operator-(double3 b) const { return {x - b.x, y - b.y, z - b.z}; }
  double3 operator*(double s) const { return {x * s, y * s, z * s}; }
  double3 operator/(double s) const { return {x / s, y / s, z / s}; }
  double3 operator-() const { return {-x, -y, -z}; }
  double3& operator+=(double3 b) { x += b.x; y += b.y; z += b.z; return *this; }
  double3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
};

inline double3 operator*(double s, double3 v) { return {s * v.x, s * v.y, s * v.z}; }
inline double dot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double length(double3 v) { return std::sqrt(dot(v, v)); }
inline double3 cross(double3 a, double3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double3 normalize(double3 v) { double l = length(v); return l > 1e-15 ? v / l : double3(0); }

} // namespace cadopt
