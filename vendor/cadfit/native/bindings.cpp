// cadfit._native: the C++ part of det (`cluster_normals` and the `section` submodule).
//
// Taken from the cad_optimizer snapshot's bindings.cpp with no logic changes;
// everything belonging to the parameter optimizer (nodes, optimizer, mesh,
// csg_tree, ops) is intentionally left out.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "section_fit.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using namespace cadopt;

PYBIND11_MODULE(_native, m) {
  m.doc() = "cadfit det native: cluster_normals + section fits";

  // Greedy face-normal clustering (planar_cluster hot path).  Mirrors
  // planar_cluster._greedy_normal_clusters exactly but in C++: for each
  // unassigned face in area-descending order, grow a cluster of faces whose
  // normal is within angle_thresh of the running area-weighted mean normal.
  // Returns clusters (area-desc) as list[(indices[int64], normal[3], origin[3], area)].
  m.def("cluster_normals", [](py::array_t<double> normals, py::array_t<double> areas,
                              py::array_t<double> centroids, double angle_thresh_deg,
                              double min_area_frac) {
    auto Nn = normals.unchecked<2>();
    auto Aa = areas.unchecked<1>();
    auto Cc = centroids.unchecked<2>();
    int n = (int)Nn.shape(0);
    py::list out;
    if (n == 0) return out;
    const double PI = 3.14159265358979323846;
    double cos_thresh = std::cos(angle_thresh_deg * PI / 180.0);
    double total = 0.0;
    for (int i = 0; i < n; i++) total += Aa(i);
    if (total < 1e-12) return out;
    double min_area = total * min_area_frac;

    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    // largest area first; ties broken by ascending index to match
    // numpy's stable argsort (deterministic, identical to the Python path).
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ if (Aa(a) != Aa(b)) return Aa(a) > Aa(b); return a < b; });

    std::vector<char> assigned(n, 0);
    struct Cl { std::vector<long long> members; double nrm[3]; double org[3]; double area; };
    std::vector<Cl> clusters;

    for (int si = 0; si < n; si++) {
      int seed = order[si];
      if (assigned[seed]) continue;
      std::vector<long long> members; members.push_back(seed);
      double rep[3] = { Nn(seed,0), Nn(seed,1), Nn(seed,2) };
      double rep_area = Aa(seed);
      assigned[seed] = 1;
      for (int jj = 0; jj < n; jj++) {
        int j = order[jj];
        if (assigned[j]) continue;
        double cos_a = Nn(j,0)*rep[0] + Nn(j,1)*rep[1] + Nn(j,2)*rep[2];
        if (cos_a < cos_thresh) continue;
        members.push_back(j);
        assigned[j] = 1;
        double aj = Aa(j), new_area = rep_area + aj;
        for (int k = 0; k < 3; k++) rep[k] = (rep[k]*rep_area + Nn(j,k)*aj) / new_area;
        double nlen = std::sqrt(rep[0]*rep[0]+rep[1]*rep[1]+rep[2]*rep[2]);
        if (nlen > 1e-12) for (int k = 0; k < 3; k++) rep[k] /= nlen;
        rep_area = new_area;
      }
      if (rep_area < min_area) continue;
      double org[3] = {0,0,0}, wsum = 0.0;
      for (long long mm : members) { double w = Aa((int)mm); wsum += w;
        for (int k = 0; k < 3; k++) org[k] += Cc((int)mm,k)*w; }
      double wd = (wsum > 1e-12 ? wsum : 1.0);
      Cl cl; cl.members = std::move(members);
      for (int k = 0; k < 3; k++) { cl.nrm[k] = rep[k]; cl.org[k] = org[k]/wd; }
      cl.area = rep_area;
      clusters.push_back(std::move(cl));
    }
    std::sort(clusters.begin(), clusters.end(),
              [](const Cl& a, const Cl& b){ return a.area > b.area; });
    for (auto& cl : clusters) {
      py::array_t<long long> idx((py::ssize_t)cl.members.size(), cl.members.data());
      py::array_t<double> nrm(3, cl.nrm);
      py::array_t<double> org(3, cl.org);
      out.append(py::make_tuple(idx, nrm, org, cl.area));
    }
    return out;
  }, "Greedy face-normal clustering (C++ port of planar_cluster hot path)",
     py::arg("normals"), py::arg("areas"), py::arg("centroids"),
     py::arg("angle_thresh_deg"), py::arg("min_area_frac"));

  // ---------------------------------------------------------------------
  // section_fit — 2D primitive fits for CADFit-style section analysis.
  // ---------------------------------------------------------------------
  auto sec = m.def_submodule("section",
    "2D primitive fits (line/arc/circle/rect/polygon) on contour points");

  py::enum_<section::PrimitiveKind>(sec, "PrimitiveKind")
    .value("LINE",    section::PrimitiveKind::LINE)
    .value("ARC",     section::PrimitiveKind::ARC)
    .value("CIRCLE",  section::PrimitiveKind::CIRCLE)
    .value("RECT",    section::PrimitiveKind::RECT)
    .value("POLYGON", section::PrimitiveKind::POLYGON);

  py::class_<section::PrimitiveFit>(sec, "PrimitiveFit")
    .def_readonly("kind",         &section::PrimitiveFit::kind)
    .def_readonly("params",       &section::PrimitiveFit::params)
    .def_readonly("residual",     &section::PrimitiveFit::residual)
    .def_readonly("support_frac", &section::PrimitiveFit::support_frac)
    .def_readonly("score",        &section::PrimitiveFit::score);

  // Helper: turn (N, 2) float64 numpy into a vector of double2.
  auto to_double2 = [](py::array_t<double> arr) -> std::vector<double2> {
    auto b = arr.unchecked<2>();
    if (b.shape(1) != 2)
      throw std::invalid_argument("points array must be shape (N, 2)");
    std::vector<double2> v((size_t)b.shape(0));
    for (py::ssize_t i = 0; i < b.shape(0); ++i)
      v[i] = double2(b(i, 0), b(i, 1));
    return v;
  };

  sec.def("fit_line", [to_double2](py::array_t<double> arr) {
    auto v = to_double2(arr);
    return section::fit_line(v.data(), (int)v.size());
  }, py::arg("points"));

  sec.def("fit_circle", [to_double2](py::array_t<double> arr) {
    auto v = to_double2(arr);
    return section::fit_circle(v.data(), (int)v.size());
  }, py::arg("points"));

  sec.def("fit_arc", [to_double2](py::array_t<double> arr) {
    auto v = to_double2(arr);
    return section::fit_arc(v.data(), (int)v.size());
  }, py::arg("points"));

  sec.def("fit_rect", [to_double2](py::array_t<double> arr) {
    auto v = to_double2(arr);
    return section::fit_rect(v.data(), (int)v.size());
  }, py::arg("points"));

  sec.def("fit_polygon", [to_double2](py::array_t<double> arr, double tol) {
    auto v = to_double2(arr);
    return section::fit_polygon(v.data(), (int)v.size(), tol);
  }, py::arg("points"), py::arg("simplify_tol") = 0.01);

  sec.def("fit_all", [to_double2](py::array_t<double> arr, double tol) {
    auto v = to_double2(arr);
    return section::fit_all(v.data(), (int)v.size(), tol);
  }, py::arg("points"), py::arg("tol") = -1.0,
     "Try every primitive, return list sorted by `score` ascending. "
     "`tol < 0` auto-picks 1%% of the bbox diagonal.");
}
