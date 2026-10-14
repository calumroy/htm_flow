// Systematic tuning for MAXIMUM temporal pooling on single repeating sequence
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

struct PoolingResult {
  std::string config_name;
  double pooled_on;       // Final pooled value with TP ON
  double pooled_off;      // Final pooled value with TP OFF
  double gap;             // Difference (pp)
  int steps_to_90;        // Steps to reach 0.90 (-1 if never)
  double avg_active_cols; // Average L1 active columns (sparsity check)
  int total_steps;        // Total training steps
};

// Base config - will modify specific params
HtmPipelineHarness::Config makeConfig(
    int pot_w,
    int pot_h,
    int cells_per_col,
    int max_segments_per_cell,
    int max_synapses_per_segment,
    float spatial_perm_inc,
    float active_predict_scale,
    float post_active_scale,
    float seq_perm_inc,
    int activation_thresh,
    int desired_local_activity,
    unsigned seed = 123u) {
  
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = 16;
  cfg.col_rows = 12;
  cfg.col_cols = 12;
  cfg.pot_h = pot_h;
  cfg.pot_w = pot_w;
  cfg.center_pot_synapses = false;
  cfg.wrap_input = true;
  cfg.inhib_w = 7;
  cfg.inhib_h = 7;
  cfg.desired_local_activity = desired_local_activity;
  cfg.connected_perm = 0.3f;
  cfg.min_overlap = 2;
  cfg.min_potential_overlap = 1;
  cfg.spatial_perm_inc = 0.05f;
  cfg.spatial_perm_dec = 0.02f;
  cfg.active_col_perm_dec = 0.01f;
  cfg.cells_per_column = cells_per_col;
  cfg.max_segments_per_cell = max_segments_per_cell;
  cfg.max_synapses_per_segment = max_synapses_per_segment;
  cfg.min_num_syn_threshold = 3;
  cfg.new_syn_permanence = 0.3f;
  cfg.connect_permanence = 0.2f;
  cfg.activation_threshold = activation_thresh;
  cfg.seq_perm_inc = 0.05f;
  cfg.seq_perm_dec = 0.02f;
  cfg.temp_spatial_perm_inc = spatial_perm_inc;
  cfg.temp_active_predict_proximal_scale = active_predict_scale;
  cfg.temp_post_active_proximal_scale = post_active_scale;
  cfg.temp_seq_perm_inc = seq_perm_inc;
  cfg.rng_seed = seed;
  
  return cfg;
}

PoolingResult runExperiment(const std::string& name, HtmPipelineHarness::Config cfg, 
                           int total_steps, bool tp_enabled) {
  if (!tp_enabled) {
    cfg.temp_post_active_proximal_scale = 0.0f;
    cfg.temp_active_predict_proximal_scale = 0.0f;
  }
  
  HtmPipelineHarness htm(cfg);
  VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
  inputs.setSequenceProbability(1.0);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  
  const int seq_len = inputs.seqLen();
  TemporalPoolingMeasure measure;
  
  double pooled = 0.0;
  int steps_to_90 = -1;
  double sum_active_cols = 0.0;
  int count_active_cols = 0;
  
  for (int t = 1; t <= total_steps; ++t) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(t, in);
    
    pooled = measure.temporalPoolingPercent(htm.learnCells01(t));
    
    // Track when we hit 0.90
    if (steps_to_90 < 0 && pooled >= 0.90) {
      steps_to_90 = t;
    }
    
    // Track sparsity (last 100 steps)
    if (t > total_steps - 100) {
      int active_count = 0;
      const auto& active_cols = htm.activeColumnsInt01();
      for (size_t i = 0; i < active_cols.size(); ++i) {
        if (active_cols[i] == 1) ++active_count;
      }
      sum_active_cols += active_count;
      count_active_cols++;
    }
  }
  
  PoolingResult result;
  result.config_name = name;
  if (tp_enabled) {
    result.pooled_on = pooled;
    result.pooled_off = 0.0;
  } else {
    result.pooled_on = 0.0;
    result.pooled_off = pooled;
  }
  result.gap = 0.0; // Will compute later
  result.steps_to_90 = steps_to_90;
  result.avg_active_cols = count_active_cols > 0 ? sum_active_cols / count_active_cols : 0.0;
  result.total_steps = total_steps;
  
  return result;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== MAXIMUM TEMPORAL POOLING EXPERIMENT ===\n";
  std::cout << "Target: >0.90 fraction of L1 learning cells carried over step-to-step\n\n";
  
  std::vector<PoolingResult> results;
  
  // Baseline (from existing suite4 test, but single-layer)
  std::cout << "1. Baseline config...\n";
  auto cfg_base = makeConfig(4, 12, 4, 2, 10, 0.05f, 0.25f, 0.25f, 0.05f, 3, 6);
  auto r1_on = runExperiment("Baseline", cfg_base, 1000, true);
  auto r1_off = runExperiment("Baseline", cfg_base, 1000, false);
  r1_on.pooled_off = r1_off.pooled_off;
  r1_on.gap = r1_on.pooled_on - r1_on.pooled_off;
  results.push_back(r1_on);
  std::cout << "  ON: " << r1_on.pooled_on << ", OFF: " << r1_on.pooled_off 
            << ", Gap: " << r1_on.gap << "\n\n";
  
  // 2. Increase receptive field (pot_w, pot_h)
  std::cout << "2. Larger receptive field (pot_w=8, pot_h=16)...\n";
  auto cfg_rf = makeConfig(8, 16, 4, 2, 10, 0.05f, 0.25f, 0.25f, 0.05f, 3, 6);
  auto r2_on = runExperiment("LargeRF", cfg_rf, 1000, true);
  auto r2_off = runExperiment("LargeRF", cfg_rf, 1000, false);
  r2_on.pooled_off = r2_off.pooled_off;
  r2_on.gap = r2_on.pooled_on - r2_on.pooled_off;
  results.push_back(r2_on);
  std::cout << "  ON: " << r2_on.pooled_on << ", OFF: " << r2_on.pooled_off 
            << ", Gap: " << r2_on.gap << "\n\n";
  
  // 3. More segments per cell (user's hypothesis)
  std::cout << "3. More segments (max_seg=8, max_syn=20)...\n";
  auto cfg_seg = makeConfig(4, 12, 4, 8, 20, 0.05f, 0.25f, 0.25f, 0.05f, 3, 6);
  auto r3_on = runExperiment("MoreSegments", cfg_seg, 1000, true);
  auto r3_off = runExperiment("MoreSegments", cfg_seg, 1000, false);
  r3_on.pooled_off = r3_off.pooled_off;
  r3_on.gap = r3_on.pooled_on - r3_on.pooled_off;
  results.push_back(r3_on);
  std::cout << "  ON: " << r3_on.pooled_on << ", OFF: " << r3_on.pooled_off 
            << ", Gap: " << r3_on.gap << "\n\n";
  
  // 4. More cells per column
  std::cout << "4. More cells per column (cells=8)...\n";
  auto cfg_cells = makeConfig(4, 12, 8, 2, 10, 0.05f, 0.25f, 0.25f, 0.05f, 3, 6);
  auto r4_on = runExperiment("MoreCells", cfg_cells, 1000, true);
  auto r4_off = runExperiment("MoreCells", cfg_cells, 1000, false);
  r4_on.pooled_off = r4_off.pooled_off;
  r4_on.gap = r4_on.pooled_on - r4_on.pooled_off;
  results.push_back(r4_on);
  std::cout << "  ON: " << r4_on.pooled_on << ", OFF: " << r4_on.pooled_off 
            << ", Gap: " << r4_on.gap << "\n\n";
  
  // 5. Strong TP parameters
  std::cout << "5. Strong TP params (post_active=2.0, spatial=0.15, seq=0.20)...\n";
  auto cfg_strong = makeConfig(4, 12, 4, 2, 10, 0.15f, 0.5f, 2.0f, 0.20f, 3, 6);
  auto r5_on = runExperiment("StrongTP", cfg_strong, 1000, true);
  auto r5_off = runExperiment("StrongTP", cfg_strong, 1000, false);
  r5_on.pooled_off = r5_off.pooled_off;
  r5_on.gap = r5_on.pooled_on - r5_on.pooled_off;
  results.push_back(r5_on);
  std::cout << "  ON: " << r5_on.pooled_on << ", OFF: " << r5_on.pooled_off 
            << ", Gap: " << r5_on.gap << "\n\n";
  
  // 6. Lower activation threshold (easier prediction)
  std::cout << "6. Lower activation threshold (thresh=2)...\n";
  auto cfg_thresh = makeConfig(4, 12, 4, 2, 10, 0.15f, 0.5f, 2.0f, 0.20f, 2, 6);
  auto r6_on = runExperiment("LowThresh", cfg_thresh, 1000, true);
  auto r6_off = runExperiment("LowThresh", cfg_thresh, 1000, false);
  r6_on.pooled_off = r6_off.pooled_off;
  r6_on.gap = r6_on.pooled_on - r6_on.pooled_off;
  results.push_back(r6_on);
  std::cout << "  ON: " << r6_on.pooled_on << ", OFF: " << r6_on.pooled_off 
            << ", Gap: " << r6_on.gap << "\n\n";
  
  // 7. Higher sparsity (more columns active)
  std::cout << "7. Higher sparsity (desired_activity=10)...\n";
  auto cfg_sparse = makeConfig(4, 12, 4, 2, 10, 0.15f, 0.5f, 2.0f, 0.20f, 2, 10);
  auto r7_on = runExperiment("HighSparsity", cfg_sparse, 1000, true);
  auto r7_off = runExperiment("HighSparsity", cfg_sparse, 1000, false);
  r7_on.pooled_off = r7_off.pooled_off;
  r7_on.gap = r7_on.pooled_on - r7_on.pooled_off;
  results.push_back(r7_on);
  std::cout << "  ON: " << r7_on.pooled_on << ", OFF: " << r7_on.pooled_off 
            << ", Gap: " << r7_on.gap << "\n\n";
  
  // 8. Combined: RF + Strong TP + Low Thresh
  std::cout << "8. Combined approach (RF + Strong TP + thresh=2)...\n";
  auto cfg_comb = makeConfig(8, 16, 4, 2, 10, 0.15f, 0.5f, 2.0f, 0.20f, 2, 6);
  auto r8_on = runExperiment("Combined", cfg_comb, 1000, true);
  auto r8_off = runExperiment("Combined", cfg_comb, 1000, false);
  r8_on.pooled_off = r8_off.pooled_off;
  r8_on.gap = r8_on.pooled_on - r8_on.pooled_off;
  results.push_back(r8_on);
  std::cout << "  ON: " << r8_on.pooled_on << ", OFF: " << r8_on.pooled_off 
            << ", Gap: " << r8_on.gap << "\n\n";
  
  // 9. More training steps
  std::cout << "9. Extended training (3000 steps)...\n";
  auto r9_on = runExperiment("Extended", cfg_comb, 3000, true);
  auto r9_off = runExperiment("Extended", cfg_comb, 3000, false);
  r9_on.pooled_off = r9_off.pooled_off;
  r9_on.gap = r9_on.pooled_on - r9_on.pooled_off;
  results.push_back(r9_on);
  std::cout << "  ON: " << r9_on.pooled_on << ", OFF: " << r9_on.pooled_off 
            << ", Gap: " << r9_on.gap << "\n\n";
  
  // 10. Ultra-strong TP
  std::cout << "10. Ultra-strong TP (post_active=3.0, spatial=0.20)...\n";
  auto cfg_ultra = makeConfig(8, 16, 4, 2, 10, 0.20f, 1.0f, 3.0f, 0.25f, 2, 6);
  auto r10_on = runExperiment("UltraStrong", cfg_ultra, 3000, true);
  auto r10_off = runExperiment("UltraStrong", cfg_ultra, 3000, false);
  r10_on.pooled_off = r10_off.pooled_off;
  r10_on.gap = r10_on.pooled_on - r10_on.pooled_off;
  results.push_back(r10_on);
  std::cout << "  ON: " << r10_on.pooled_on << ", OFF: " << r10_on.pooled_off 
            << ", Gap: " << r10_on.gap << "\n\n";
  
  // Summary table
  std::cout << "\n=== RESULTS SUMMARY ===\n\n";
  std::cout << "UNITS: Pooled = fraction 0-1 of L1 learning cells at t-1 still active at t\n";
  std::cout << "       Gap = ON - OFF in percentage points\n";
  std::cout << "       Target: >0.90 (90% carry-over)\n\n";
  
  std::cout << std::setw(20) << "Config"
            << std::setw(12) << "Pooled ON"
            << std::setw(12) << "Pooled OFF"
            << std::setw(12) << "Gap (pp)"
            << std::setw(15) << "Steps to 90%"
            << std::setw(15) << "Avg Active Cols"
            << std::setw(12) << "Total Steps"
            << "\n";
  std::cout << std::string(95, '-') << "\n";
  
  for (const auto& r : results) {
    std::cout << std::setw(20) << r.config_name
              << std::setw(12) << r.pooled_on
              << std::setw(12) << r.pooled_off
              << std::setw(12) << r.gap
              << std::setw(15) << (r.steps_to_90 > 0 ? std::to_string(r.steps_to_90) : "never")
              << std::setw(15) << r.avg_active_cols
              << std::setw(12) << r.total_steps
              << "\n";
  }
  
  // Find best config
  auto best = results[0];
  for (const auto& r : results) {
    if (r.pooled_on > best.pooled_on) best = r;
  }
  
  std::cout << "\n=== BEST CONFIG ===\n";
  std::cout << "Name: " << best.config_name << "\n";
  std::cout << "Pooled ON: " << best.pooled_on << " (" << (best.pooled_on * 100) << "%)\n";
  std::cout << "Pooled OFF: " << best.pooled_off << " (" << (best.pooled_off * 100) << "%)\n";
  std::cout << "Gap: " << best.gap << " (" << (best.gap * 100) << " pp)\n";
  std::cout << "Steps to 90%: " << (best.steps_to_90 > 0 ? std::to_string(best.steps_to_90) : "never") << "\n";
  std::cout << "Avg active cols: " << best.avg_active_cols << " / 144 (" 
            << (best.avg_active_cols / 144.0 * 100) << "%)\n";
  std::cout << "Achieved target: " << (best.pooled_on >= 0.90 ? "YES" : "NO") << "\n";
  
  return 0;
}
