#include "ops/chamfer.h"
#include <cmath>
#include <algorithm>

namespace cadopt {

double chamfer_intersect(double d1, double d2, double size,
                         double* dcham_dd1, double* dcham_dd2, double* dcham_dsize) {
  // d3 = (d1 + d2) * INV_SQRT2 - size
  static const double INV_SQRT2 = 1.0 / std::sqrt(2.0);
  double d3 = (d1 + d2) * INV_SQRT2 - size;

  // result = max(d1, d2, d3)
  if (d1 >= d2 && d1 >= d3) {
    *dcham_dd1 = 1.0;
    *dcham_dd2 = 0.0;
    *dcham_dsize = 0.0;
    return d1;
  } else if (d2 >= d1 && d2 >= d3) {
    *dcham_dd1 = 0.0;
    *dcham_dd2 = 1.0;
    *dcham_dsize = 0.0;
    return d2;
  } else {
    *dcham_dd1 = INV_SQRT2;
    *dcham_dd2 = INV_SQRT2;
    *dcham_dsize = -1.0;
    return d3;
  }
}

} // namespace cadopt
