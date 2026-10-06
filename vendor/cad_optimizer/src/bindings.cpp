#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "nodes.h"
#include "optimizer.h"
#include "mesh.h"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;
using namespace cadopt;

// Factory: create a node from a description dict (recursive)
static std::unique_ptr<CsgNode> build_tree(const py::dict& desc) {
  std::string type = desc["type"].cast<std::string>();

  if (type == "rect") {
    return std::make_unique<RectNode>();
  } else if (type == "circle") {
    return std::make_unique<CircleNode>();
  } else if (type == "extrude") {
    auto node = std::make_unique<ExtrudeNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "union") {
    auto node = std::make_unique<UnionNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "intersect") {
    auto node = std::make_unique<IntersectNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "subtract") {
    auto node = std::make_unique<SubtractNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "translate3d") {
    auto node = std::make_unique<Translate3DNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "rotate3d") {
    auto node = std::make_unique<Rotate3DNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "scale3d") {
    auto node = std::make_unique<Scale3DNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "mirror_xy") {
    auto node = std::make_unique<MirrorXYNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "mirror_xz") {
    auto node = std::make_unique<MirrorXZNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "mirror_yz") {
    auto node = std::make_unique<MirrorYZNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "inverse") {
    auto node = std::make_unique<InverseNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "revolve") {
    auto node = std::make_unique<RevolveNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    if (desc.contains("axis_along_x"))
      node->axis_along_x = desc["axis_along_x"].cast<int>();
    if (desc.contains("axis_offset"))
      node->axis_offset = desc["axis_offset"].cast<double>();
    if (desc.contains("angle_deg"))
      node->angle_deg = desc["angle_deg"].cast<double>();
    if (desc.contains("profile_side"))
      node->profile_side = desc["profile_side"].cast<double>();
    if (desc.contains("wedge_b_sign"))
      node->wedge_b_sign = desc["wedge_b_sign"].cast<double>();
    return node;
  } else if (type == "fillet_union") {
    auto node = std::make_unique<FilletUnionNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "fillet_intersect") {
    auto node = std::make_unique<FilletIntersectNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "chamfer_intersect") {
    auto node = std::make_unique<ChamferIntersectNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "offset") {
    auto node = std::make_unique<OffsetNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "shell") {
    auto node = std::make_unique<ShellNode>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "sphere") {
    return std::make_unique<SphereNode>();
  } else if (type == "box") {
    return std::make_unique<BoxNode>();
  } else if (type == "cylinder") {
    return std::make_unique<CylinderNode>();
  } else if (type == "mesh") {
    // Frozen sub-solid baked to an STL (exotic cadgen ops).  Load once + BVH.
    auto node = std::make_unique<MeshNode>();
    node->mesh = std::make_shared<Mesh>(load_stl(desc["path"].cast<std::string>()));
    node->mesh->buildBVH();
    return node;
  } else if (type == "loft") {
    auto node = std::make_unique<LoftNode>();
    node->left = build_tree(desc["left"].cast<py::dict>());
    node->right = build_tree(desc["right"].cast<py::dict>());
    return node;
  } else if (type == "sweep") {
    int n_verts = desc["n_path_verts"].cast<int>();
    auto node = std::make_unique<SweepNode>(n_verts);
    if (desc.contains("frenet")) node->use_frenet = desc["frenet"].cast<bool>();
    if (desc.contains("profile_angle")) node->profile_angle = desc["profile_angle"].cast<double>();
    node->child = build_tree(desc["child"].cast<py::dict>());
    return node;
  } else if (type == "slot") {
    return std::make_unique<SlotNode>();
  } else if (type == "polygon") {
    int n_verts = desc["n_vertices"].cast<int>();
    return std::make_unique<PolygonNode>(n_verts);
  } else if (type == "arcpoly") {
    int n_verts = desc["n_vertices"].cast<int>();
    auto node = std::make_unique<ArcPolyNode>(n_verts);
    py::list edges = desc["edges"].cast<py::list>();
    for (int i = 0; i < n_verts && i < (int)edges.size(); i++) {
      py::dict e = edges[i].cast<py::dict>();
      node->edges[i].is_arc = e["is_arc"].cast<int>() != 0;
      node->edges[i].r_s  = e.contains("r_s")  ? e["r_s"].cast<double>()  : 0.0;
      node->edges[i].side = e.contains("side") ? e["side"].cast<double>() : 1.0;
      node->edges[i].major = e.contains("major") ? (e["major"].cast<int>() != 0) : false;
    }
    return node;
  }
  throw std::runtime_error("Unknown node type: " + type);
}

// Wrapper that owns the tree and exposes it to Python
struct PyTree {
  std::unique_ptr<CsgNode> root;

  void set_params(const std::vector<double>& p) { root->setParams(p.data()); }
  std::vector<double> get_params() const { return root->getParamsVec(); }
  std::vector<ParamInfo> get_param_info() const { return root->getParamInfoVec(); }
  uint32_t param_count() const { return root->getParamCount(); }

  double eval_sdf(double x, double y, double z) const {
    return root->calcSdf({x, y, z});
  }
};

PYBIND11_MODULE(_cad_grad, m) {
  m.doc() = "CAD Gradient Optimizer C++ backend";

  py::class_<ParamInfo>(m, "ParamInfo")
    .def_readwrite("lo", &ParamInfo::lo)
    .def_readwrite("hi", &ParamInfo::hi);

  py::class_<PyTree>(m, "CsgTree")
    .def("set_params", &PyTree::set_params)
    .def("get_params", &PyTree::get_params)
    .def("param_count", &PyTree::param_count)
    .def("eval_sdf", &PyTree::eval_sdf);

  m.def("eval_sdf_mse", [](PyTree& tree,
                           py::array_t<double> pos_arr,
                           py::array_t<double> dist_arr) {
    auto pos = pos_arr.unchecked<2>();
    auto dist = dist_arr.unchecked<1>();
    int n = (int)pos.shape(0);
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      double sdf = tree.root->calcSdf({pos(i, 0), pos(i, 1), pos(i, 2)});
      double d = sdf - dist(i);
      sum += d * d;
    }
    return sum / n;
  }, "MSE between analytical SDF and reference distances at sample positions.");

  m.def("create_tree", [](const py::dict& desc) {
    auto t = std::make_unique<PyTree>();
    t->root = build_tree(desc);
    return t;
  }, "Build a CSG tree from a description dict");

  py::class_<Mesh>(m, "Mesh")
    .def("bounds", [](const Mesh& mesh) {
      return py::make_tuple(
        py::make_tuple(mesh.bounds.mn.x, mesh.bounds.mn.y, mesh.bounds.mn.z),
        py::make_tuple(mesh.bounds.mx.x, mesh.bounds.mx.y, mesh.bounds.mx.z));
    }, "Axis-aligned bounding box ((mn.x,mn.y,mn.z), (mx.x,mx.y,mx.z)).");

  m.def("load_stl", [](const std::string& path) {
    return load_stl(path);
  }, "Load an STL mesh file");

  m.def("sample_sdf_points", [](const Mesh& mesh, int count, double band, unsigned seed) {
    std::vector<double3> pos;
    std::vector<double> dist;
    sample_sdf_points(mesh, count, band, pos, dist, seed);

    // Convert to numpy arrays
    py::array_t<double> pos_arr({(int)pos.size(), 3});
    auto pos_buf = pos_arr.mutable_unchecked<2>();
    for (size_t i = 0; i < pos.size(); i++) {
      pos_buf(i, 0) = pos[i].x; pos_buf(i, 1) = pos[i].y; pos_buf(i, 2) = pos[i].z;
    }
    py::array_t<double> dist_arr(dist.size(), dist.data());
    return py::make_tuple(pos_arr, dist_arr);
  }, "Sample SDF points near mesh surface",
     py::arg("mesh"), py::arg("count") = 10000, py::arg("band") = 0.1, py::arg("seed") = 42);

  m.def("optimize", [](PyTree& tree, py::array_t<double> pos_arr, py::array_t<double> dist_arr,
                        double lr, int steps, int batch_size, double early_stop, bool verbose,
                        std::vector<bool> freeze_mask, bool numerical_gradients, double fd_eps,
                        std::vector<double> param_lo, std::vector<double> param_hi,
                        int stagnation_window, double stagnation_rel,
                        double importance_mix, int resample_every,
                        py::array_t<double> occ_pos_arr, py::array_t<double> occ_sign_arr,
                        double occ_weight, int occ_batch) {
    auto pos_buf = pos_arr.unchecked<2>();
    auto dist_buf = dist_arr.unchecked<1>();
    int n = (int)pos_buf.shape(0);

    std::vector<double3> positions(n);
    std::vector<double> distances(n);
    for (int i = 0; i < n; i++) {
      positions[i] = {pos_buf(i, 0), pos_buf(i, 1), pos_buf(i, 2)};
      distances[i] = dist_buf(i);
    }

    // Optional occupancy points for the volumetric sign regularizer.
    std::vector<double3> occ_positions;
    std::vector<double> occ_signs;
    if (occ_weight > 0.0 && occ_pos_arr.size() > 0) {
      auto op = occ_pos_arr.unchecked<2>();
      auto os = occ_sign_arr.unchecked<1>();
      int m_occ = (int)op.shape(0);
      occ_positions.resize(m_occ);
      occ_signs.resize(m_occ);
      for (int i = 0; i < m_occ; i++) {
        occ_positions[i] = {op(i, 0), op(i, 1), op(i, 2)};
        occ_signs[i] = os(i);
      }
    }

    auto params = tree.get_params();
    auto pinfo = tree.get_param_info();
    // Apply optional bound overrides — useful for trust-region-style
    // drift clipping (caller passes [p_init - delta, p_init + delta]).
    if (!param_lo.empty() && param_lo.size() == pinfo.size())
      for (size_t i = 0; i < pinfo.size(); i++) pinfo[i].lo = param_lo[i];
    if (!param_hi.empty() && param_hi.size() == pinfo.size())
      for (size_t i = 0; i < pinfo.size(); i++) pinfo[i].hi = param_hi[i];

    OptSettings settings;
    settings.learning_rate = lr;
    settings.max_steps = steps;
    settings.batch_size = batch_size;
    settings.early_stop_loss = early_stop;
    settings.verbose = verbose;
    settings.freeze_mask = std::move(freeze_mask);
    settings.numerical_gradients = numerical_gradients;
    settings.fd_eps = fd_eps;
    settings.stagnation_window = stagnation_window;
    settings.stagnation_rel = stagnation_rel;
    settings.importance_mix = importance_mix;
    settings.resample_every = resample_every;
    settings.occ_weight = occ_weight;
    settings.occ_batch = occ_batch;
    settings.occ_positions = std::move(occ_positions);
    settings.occ_signs = std::move(occ_signs);

    auto result = optimize_adam(settings, tree.root.get(), positions, distances, params, pinfo);
    tree.set_params(result.params);

    return py::dict(
      py::arg("params") = result.params,
      py::arg("loss_history") = result.loss_history
    );
  }, "Run Adam optimization",
     py::arg("tree"), py::arg("positions"), py::arg("distances"),
     py::arg("lr") = 1e-3, py::arg("steps") = 200, py::arg("batch_size") = 512,
     py::arg("early_stop") = 0.0, py::arg("verbose") = false,
     py::arg("freeze_mask") = std::vector<bool>(),
     py::arg("numerical_gradients") = false,
     py::arg("fd_eps") = 1e-4,
     py::arg("param_lo") = std::vector<double>{},
     py::arg("param_hi") = std::vector<double>{},
     py::arg("stagnation_window") = 200,
     py::arg("stagnation_rel") = 0.002,
     py::arg("importance_mix") = 0.0,
     py::arg("resample_every") = 25,
     py::arg("occ_positions") = py::array_t<double>(),
     py::arg("occ_signs") = py::array_t<double>(),
     py::arg("occ_weight") = 0.0,
     py::arg("occ_batch") = 0);
}
