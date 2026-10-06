#include "optimizer.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace cadopt {

OptResult optimize_adam(const OptSettings& settings,
                        CsgNode* tree,
                        const std::vector<double3>& positions,
                        const std::vector<double>& ref_distances,
                        std::vector<double>& params,
                        const std::vector<ParamInfo>& param_info,
                        unsigned seed) {
  const uint32_t pc = (uint32_t)params.size();
  const int point_count = (int)positions.size();
  const int batch = std::min(settings.batch_size, point_count);
  const double inv_batch = 1.0 / batch;

  std::vector<double> m1(pc, 0.0), m2(pc, 0.0);
  std::vector<double> grad(pc), grad_tmp(pc);

  OptResult result;
  result.loss_history.reserve(settings.max_steps);

  std::mt19937 rng(seed);

  double b1_pow = 1.0, b2_pow = 1.0;

  // Track the BEST params seen during the run.  Adam can drift once it
  // hits a plateau (loss near noise floor); best-tracking gives us back
  // the lowest-loss visited rather than the random-walked endpoint.
  std::vector<double> best_params = params;
  double best_loss = std::numeric_limits<double>::infinity();

  // --- Residual-importance sampling state (analytical path only) ---
  const bool use_importance =
      settings.importance_mix > 0.0 && !settings.numerical_gradients;
  const double imix = settings.importance_mix;
  const int resample_every = std::max(1, settings.resample_every);
  std::vector<double> r2, prob, cdf;   // per-point sq-residual cache, p_i, prefix-sum
  std::uniform_real_distribution<double> uni01(0.0, 1.0);
  if (use_importance) {
    r2.assign(point_count, 0.0);
    prob.assign(point_count, 1.0 / point_count);
    cdf.assign(point_count, 0.0);
  }

  for (int step = 0; step < settings.max_steps; step++) {
    // Clamp params
    for (uint32_t j = 0; j < pc; j++)
      params[j] = std::clamp(params[j], param_info[j].lo, param_info[j].hi);

    tree->setParams(params.data());

    std::fill(grad.begin(), grad.end(), 0.0);
    double loss = 0.0;

    // Importance sampling: after a uniform warm-up window (lets the bulk body
    // fit and populates the residual cache), rebuild the sampling distribution
    // every resample_every steps from the cached squared residuals.
    const bool importance_active = use_importance && (step >= resample_every);
    if (importance_active && (step % resample_every == 0)) {
      double S = 0.0;
      for (int i = 0; i < point_count; i++) S += r2[i];
      const double base = (1.0 - imix) / point_count;
      double c = 0.0;
      if (S > 0.0) {
        for (int i = 0; i < point_count; i++) {
          prob[i] = base + imix * (r2[i] / S);
          c += prob[i]; cdf[i] = c;
        }
      } else {
        for (int i = 0; i < point_count; i++) {
          prob[i] = 1.0 / point_count; c += prob[i]; cdf[i] = c;
        }
      }
    }

    // Select batch indices + per-sample importance weights (shared between
    // analytical and numerical paths).  When importance is inactive this is the
    // verbatim uniform draw with iw=1 (seed-identical to the pre-change build).
    std::vector<int> batch_idx(batch);
    std::vector<double> batch_iw(batch, 1.0);
    if (importance_active) {
      const double total = cdf[point_count - 1];
      for (int b = 0; b < batch; b++) {
        double u = uni01(rng) * total;
        int idx = (int)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        if (idx >= point_count) idx = point_count - 1;
        batch_idx[b] = idx;
        batch_iw[b] = 1.0 / (point_count * prob[idx]);  // unbiased correction
      }
    } else {
      for (int b = 0; b < batch; b++)
        batch_idx[b] = rng() % point_count;
    }

    if (!settings.numerical_gradients) {
      // --- Analytical gradients (parallel) ---
      // The batch evals are independent (calcSdfWithDiff is const with local
      // scratch; the root call fully overwrites all pc gradient entries — the
      // boolean nodes zero the losing branch).  Parallelise over a FIXED
      // number of chunks with per-chunk accumulators reduced in chunk order:
      // results are bitwise identical for ANY thread count (only the FP
      // addition order vs the old strictly-serial loop differs).  RNG draw
      // order is untouched (indices pre-drawn above), so batch selection is
      // seed-identical to the serial build.
      constexpr int NCHUNK = 16;
      // Pre-draw occupancy indices HERE (same rng position as the old
      // in-loop draws: after batch_idx, before any evaluation).
      const bool use_occ = settings.occ_weight > 0.0 && !settings.occ_positions.empty();
      const int occ_n = use_occ ? (int)settings.occ_positions.size() : 0;
      const int ob = use_occ ? std::min(
          settings.occ_batch > 0 ? settings.occ_batch : settings.batch_size, occ_n) : 0;
      std::vector<int> occ_idx(ob);
      // (drawn after the batch-eval below in the OLD code path; keep the same
      // stream order: batch_idx draws happened above, occ draws come next.)
      for (int b = 0; b < ob; b++) occ_idx[b] = rng() % occ_n;

      std::vector<double> chunk_loss(NCHUNK, 0.0);
      std::vector<double> chunk_grad((size_t)NCHUNK * pc, 0.0);
      std::vector<double> diff2(batch);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (batch >= 512)
#endif
      for (int c = 0; c < NCHUNK; c++) {
        std::vector<double> gtmp(pc, 0.0);
        double* cg = &chunk_grad[(size_t)c * pc];
        int b0 = (int)((long long)batch * c / NCHUNK);
        int b1 = (int)((long long)batch * (c + 1) / NCHUNK);
        for (int b = b0; b < b1; b++) {
          int idx = batch_idx[b];
          double w = inv_batch * batch_iw[b];
          double dp[3];
          double sdf = tree->calcSdfWithDiff(positions[idx], gtmp.data(), dp);
          double diff = sdf - ref_distances[idx];
          // Report/track the UNWEIGHTED batch loss (low-variance progress
          // signal for best_params + stagnation); weight ONLY the gradient by
          // iw so the descent direction stays the unbiased SDF-MSE gradient.
          chunk_loss[c] += inv_batch * diff * diff;
          for (uint32_t k = 0; k < pc; k++)
            cg[k] += w * diff * gtmp[k];
          diff2[b] = diff * diff;
        }
      }
      // Deterministic chunk-ordered reduction.
      for (int c = 0; c < NCHUNK; c++) {
        loss += chunk_loss[c];
        const double* cg = &chunk_grad[(size_t)c * pc];
        for (uint32_t k = 0; k < pc; k++) grad[k] += cg[k];
      }
      // Residual cache for the importance distribution: serial writes avoid
      // duplicate-index races (overwrite, not smooth — a sampling cache).
      if (use_importance)
        for (int b = 0; b < batch; b++) r2[batch_idx[b]] = diff2[b];
      // multiply by 2 for MSE derivative (d/dx of (x-y)^2 = 2*(x-y))
      for (uint32_t k = 0; k < pc; k++)
        grad[k] *= 2.0;

      // --- Volumetric occupancy (sign) regularizer (parallel, same scheme) ---
      // Only wrong-sign points contribute, with a linear (magnitude-robust)
      // hinge: penalise pred solid where target is empty (balloon) and pred
      // empty where target is solid (hollow).  Uses a SEPARATE occ mini-batch
      // so it complements — not replaces — the surface-band gradient above.
      if (use_occ) {
        const double inv_ob = 1.0 / ob;
        std::fill(chunk_loss.begin(), chunk_loss.end(), 0.0);
        std::fill(chunk_grad.begin(), chunk_grad.end(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (ob >= 512)
#endif
        for (int c = 0; c < NCHUNK; c++) {
          std::vector<double> gtmp(pc, 0.0);
          double* cg = &chunk_grad[(size_t)c * pc];
          int b0 = (int)((long long)ob * c / NCHUNK);
          int b1 = (int)((long long)ob * (c + 1) / NCHUNK);
          for (int b = b0; b < b1; b++) {
            int idx = occ_idx[b];
            double3 p = settings.occ_positions[idx];
            double s = settings.occ_signs[idx];    // -1 inside target, +1 outside
            double dp3[3];
            double sdf = tree->calcSdfWithDiff(p, gtmp.data(), dp3);
            double u = s * sdf;                     // >0 => pred agrees with target
            if (u < 0.0) {                          // wrong sign: ballooned / hollowed
              chunk_loss[c] += settings.occ_weight * inv_ob * (-u);
              double gcoef = settings.occ_weight * inv_ob * (-s);
              for (uint32_t k = 0; k < pc; k++)
                cg[k] += gcoef * gtmp[k];
            }
          }
        }
        for (int c = 0; c < NCHUNK; c++) {
          loss += chunk_loss[c];
          const double* cg = &chunk_grad[(size_t)c * pc];
          for (uint32_t k = 0; k < pc; k++) grad[k] += cg[k];
        }
      }
    } else {
      // --- Numerical (finite-difference) gradients ---
      // First compute base loss
      for (int b = 0; b < batch; b++) {
        int idx = batch_idx[b];
        double diff = tree->calcSdf(positions[idx]) - ref_distances[idx];
        loss += inv_batch * diff * diff;
      }
      // For each parameter, compute forward-difference gradient
      const double eps = settings.fd_eps;
      std::vector<double> params_copy(params);
      for (uint32_t j = 0; j < pc; j++) {
        if (!settings.freeze_mask.empty() && settings.freeze_mask[j])
          continue;
        // Perturb parameter j forward
        params_copy[j] = params[j] + eps;
        tree->setParams(params_copy.data());
        double loss_plus = 0.0;
        for (int b = 0; b < batch; b++) {
          int idx = batch_idx[b];
          double diff = tree->calcSdf(positions[idx]) - ref_distances[idx];
          loss_plus += inv_batch * diff * diff;
        }
        grad[j] = (loss_plus - loss) / eps;
        // Restore
        params_copy[j] = params[j];
      }
      // Restore original params
      tree->setParams(params.data());
    }

    // Adam update
    b1_pow *= settings.adam_beta1;
    b2_pow *= settings.adam_beta2;
    for (uint32_t j = 0; j < pc; j++) {
      if (!settings.freeze_mask.empty() && settings.freeze_mask[j])
        continue;  // skip frozen params
      m1[j] = settings.adam_beta1 * m1[j] + (1.0 - settings.adam_beta1) * grad[j];
      m2[j] = settings.adam_beta2 * m2[j] + (1.0 - settings.adam_beta2) * grad[j] * grad[j];
      double m1_hat = m1[j] / (1.0 - b1_pow);
      double m2_hat = m2[j] / (1.0 - b2_pow);
      params[j] -= settings.learning_rate * m1_hat / (std::sqrt(m2_hat) + settings.adam_eps);
      params[j] = std::clamp(params[j], param_info[j].lo, param_info[j].hi);
    }

    tree->setParams(params.data());
    result.loss_history.push_back(loss);

    if (loss < best_loss) {
      best_loss = loss;
      best_params = params;
    }

    if (settings.verbose)
      printf("step %4d  loss = %.8f\n", step, loss);

    if (settings.early_stop_loss > 0 && loss < settings.early_stop_loss)
      break;

    // Stagnation early-stop: if the loss has not meaningfully decreased
    // over the past `stagnation_window` steps, the gradient is dominated
    // by noise (which is exactly the failure mode causing IoU regressions:
    // the loss plateaus while params drift in random directions).  Bail
    // out and report the best loss we've seen so far.
    if (settings.stagnation_window > 0 &&
        (int)result.loss_history.size() > settings.stagnation_window) {
      int W = settings.stagnation_window;
      int n = (int)result.loss_history.size();
      double recent_min = result.loss_history[n - 1];
      double past_min   = result.loss_history[n - 1 - W];
      // Compare min over recent W steps to min over preceding W steps.
      for (int k = 1; k < W && (n - 1 - k) >= 0; k++)
        recent_min = std::min(recent_min, result.loss_history[n - 1 - k]);
      for (int k = 1; k < W && (n - 1 - W - k) >= 0; k++)
        past_min = std::min(past_min, result.loss_history[n - 1 - W - k]);
      // If the recent block didn't improve more than `stagnation_rel`
      // *and* loss is already small (so we're polishing, not making
      // big strides), stop.
      if (past_min > 0 && recent_min > past_min * (1.0 - settings.stagnation_rel)) {
        if (settings.verbose)
          printf("early stop: stagnation at step %d (recent_min=%.4e, past_min=%.4e)\n",
                 step, recent_min, past_min);
        break;
      }
    }
  }

  // Return the best-seen params (lowest batch loss visited).
  result.params = best_params;
  return result;
}

} // namespace cadopt
