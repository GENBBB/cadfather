#pragma once
#include "csg_tree.h"
#include "mesh.h"
#include <vector>

namespace cadopt {

struct OptSettings {
  double learning_rate = 1e-3;
  double adam_beta1 = 0.9;
  double adam_beta2 = 0.999;
  double adam_eps = 1e-8;
  int max_steps = 200;
  int batch_size = 512;
  double early_stop_loss = 0.0;
  bool verbose = false;
  std::vector<bool> freeze_mask;  // if non-empty, freeze_mask[i]=true skips param i
  bool numerical_gradients = false;  // use finite-difference gradients instead of analytical
  double fd_eps = 1e-4;  // finite-difference step size
  // Stagnation early-stop: if the loss min over the last W steps is no
  // smaller than the loss min over the W steps before that — by more than
  // a fraction `rel` — the gradient is mostly noise.  Stop.
  int stagnation_window = 0;        // 0 disables
  double stagnation_rel = 0.001;    // 0.1%
  // Residual-importance mini-batch sampling (analytical path only).  When
  // importance_mix > 0, the per-step mini-batch is drawn from
  //   p_i = (1-mix)/N + mix * r2_i / sum(r2)
  // (r2_i = last-seen squared residual of point i), and each drawn point's
  // loss/grad contribution is multiplied by the importance weight
  // iw_i = 1/(N*p_i) so the estimator stays EXACTLY unbiased for the plain
  // SDF-MSE gradient — pure variance reduction that gives weak/localized
  // features (a cut depth, a small hole) a clean low-variance gradient
  // without a larger global point budget.  mix=0 → verbatim uniform path.
  double importance_mix = 0.0;      // 0 disables (default = byte-identical)
  int resample_every = 25;          // rebuild the sampling distribution every N steps

  // --- Volumetric occupancy (sign) regularizer ---
  // The surface-band MSE only constrains a thin shell around the target
  // surface (the analytical CSG SDF has the right SIGN everywhere but the
  // wrong MAGNITUDE in interiors, so magnitude can only be regressed near
  // the surface).  That under-constrains high-DOF shapes: parameters can
  // balloon into unsampled space with near-zero band loss (the catastrophic
  // post-opt IoU collapse).  This term samples occupancy points across the
  // whole volume and penalises WRONG-SIGN points with a LINEAR hinge:
  //   L_occ = occ_weight * mean_over_batch( relu(-sign_i * sdf(p_i)) )
  // Gradient magnitude is sign-only (occ_weight * (-sign) * d(sdf)/dparam),
  // so it is robust to the unreliable interior SDF magnitude while its
  // DIRECTION (from the active CSG branch) is reliable.  It is active only
  // where the sign is wrong (ballooned / hollowed), so it never fights the
  // surface-band fit.  occ_weight = 0 disables (byte-identical to before).
  double occ_weight = 0.0;                 // 0 disables
  int occ_batch = 0;                       // 0 -> use batch_size
  std::vector<double3> occ_positions;      // volumetric points (target frame)
  std::vector<double> occ_signs;           // -1 inside gt, +1 outside gt
};

struct OptResult {
  std::vector<double> params;       // optimized parameters
  std::vector<double> loss_history; // loss at each step
};

// Run Adam optimization on tree params to minimize SDF error against reference points.
// `tree` is modified in-place (params updated). `positions` and `ref_distances` come
// from sample_sdf_points() on the target mesh.
OptResult optimize_adam(const OptSettings& settings,
                        CsgNode* tree,
                        const std::vector<double3>& positions,
                        const std::vector<double>& ref_distances,
                        std::vector<double>& params,
                        const std::vector<ParamInfo>& param_info,
                        unsigned seed = 42);

} // namespace cadopt
