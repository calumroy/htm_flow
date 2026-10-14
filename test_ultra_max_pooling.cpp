// Push to >90% temporal pooling with ultra-aggressive settings
#include <iostream>
#include <iomanip>
#include <vector>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

struct Result {
  std::string name;
  double pooled_on;
  double pooled_off;
  int steps_to_90;
  double avg_active_cols;
};

HtmPipelineHarness::Config makeConfig(
    float spatial_perm_inc,
    float active_predict_scale,
    float post_active_scale,
    float seq_perm_inc,
    int activation_thresh) {
  
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = 16;
  cfg.col_rows = 12;
  cfg.col_cols = 12;
  cfg.pot_h = 12;
  cfg.pot_w = 4;
  cfg.center_pot_synapses = false;
  cfg.wrap_input = true;
  cfg.inhib_w = 7;
  cfg.inhib_h = 7;
  cfg.desired_local_activity = 6;
  cfg.connected_perm = 0.3f;
  cfg.min_overlap = 2;
  cfg.min_potential_overlap = 1;
  cfg.spatial_perm_inc = 0.05f;
  cfg.spatial_perm_dec = 0.02f;
  cfg.active_col_perm_dec = 0.01f;
  cfg.cells_per_column = 4;
  cfg.max_segments_per_cell = 2;
  cfg.max_synapses_per_segment = 10;
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
  cfg.rng_seed = 123u;
  
  return cfg;
}

Result runTest(const std::string& name, HtmPipelineHarness::Config cfg, int total_steps, bool tp_on) {
  if (!tp_on) {
    cfg.temp_post_active_proximal_scale = 0.0f;
    cfg.temp_active_predict_proximal_scale = 0.0f;
  }
  
  HtmPipelineHarness htm(cfg);
  VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
  inputs.setSequenceProbability(1.0);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  
  TemporalPoolingMeasure measure;
  double pooled = 0.0;
  int steps_to_90 = -1;
  double sum_cols = 0.0;
  int count_cols = 0;
  
  // Print progress every 500 steps
  for (int t = 1; t <= total_steps; ++t) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(t, in);
    pooled = measure.temporalPoolingPercent(htm.learnCells01(t));
    
    if (steps_to_90 < 0 && pooled >= 0.90) {
      steps_to_90 = t;
    }
    
    if (t > total_steps - 100) {
      int active = 0;
      const auto& cols = htm.activeColumnsInt01();
      for (auto c : cols) if (c == 1) ++active;
      sum_cols += active;
      count_cols++;
    }
    
    if (tp_on && t % 500 == 0) {
      std::cout << "  Step " << t << ": pooled=" << pooled << "\n";
    }
  }
  
  Result r;
  r.name = name;
  r.pooled_on = tp_on ? pooled : 0.0;
  r.pooled_off = tp_on ? 0.0 : pooled;
  r.steps_to_90 = steps_to_90;
  r.avg_active_cols = count_cols > 0 ? sum_cols / count_cols : 0.0;
  return r;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== ULTRA-AGGRESSIVE TEMPORAL POOLING ===\n";
  std::cout << "Target: >0.90 (90% carry-over)\n\n";
  
  std::vector<Result> results;
  
  // Test increasingly aggressive configs
  struct ConfigSpec {
    std::string name;
    float spatial;
    float active_pred;
    float post_active;
    float seq;
    int thresh;
    int steps;
  };
  
  std::vector<ConfigSpec> configs = {
    {"Aggressive1", 0.20f, 1.0f, 3.0f, 0.25f, 2, 3000},
    {"Aggressive2", 0.25f, 1.5f, 4.0f, 0.30f, 2, 3000},
    {"Aggressive3", 0.30f, 2.0f, 5.0f, 0.35f, 1, 3000},
    {"Aggressive4", 0.35f, 2.0f, 6.0f, 0.40f, 1, 5000},
    {"Aggressive5", 0.40f, 2.5f, 8.0f, 0.45f, 1, 5000},
  };
  
  for (const auto& spec : configs) {
    std::cout << "\nTesting " << spec.name << " (steps=" << spec.steps << "):\n";
    std::cout << "  spatial=" << spec.spatial << ", active_pred=" << spec.active_pred 
              << ", post_active=" << spec.post_active << ", seq=" << spec.seq 
              << ", thresh=" << spec.thresh << "\n";
    
    auto cfg = makeConfig(spec.spatial, spec.active_pred, spec.post_active, 
                          spec.seq, spec.thresh);
    
    std::cout << "  Running with TP ON:\n";
    auto r_on = runTest(spec.name, cfg, spec.steps, true);
    
    std::cout << "  Running with TP OFF:\n";
    auto r_off = runTest(spec.name, cfg, spec.steps, false);
    
    r_on.pooled_off = r_off.pooled_off;
    results.push_back(r_on);
    
    std::cout << "  Result: ON=" << r_on.pooled_on << " (" << (r_on.pooled_on*100) << "%)"
              << ", OFF=" << r_on.pooled_off << " (" << (r_on.pooled_off*100) << "%)"
              << ", Gap=" << (r_on.pooled_on - r_on.pooled_off) << "\n";
    std::cout << "  Steps to 90%: " << (r_on.steps_to_90 > 0 ? std::to_string(r_on.steps_to_90) : "never") << "\n";
    std::cout << "  Avg active cols: " << r_on.avg_active_cols << " / 144\n";
    
    if (r_on.pooled_on >= 0.90) {
      std::cout << "  ✓✓✓ TARGET ACHIEVED! ✓✓✓\n";
      break;
    }
  }
  
  // Summary
  std::cout << "\n=== SUMMARY ===\n\n";
  std::cout << std::setw(20) << "Config"
            << std::setw(12) << "Pooled ON"
            << std::setw(12) << "Pooled OFF"
            << std::setw(12) << "Gap (pp)"
            << std::setw(15) << "Steps to 90%"
            << std::setw(12) << "Avg Cols"
            << "\n";
  std::cout << std::string(83, '-') << "\n";
  
  for (const auto& r : results) {
    std::cout << std::setw(20) << r.name
              << std::setw(12) << r.pooled_on
              << std::setw(12) << r.pooled_off
              << std::setw(12) << (r.pooled_on - r.pooled_off)
              << std::setw(15) << (r.steps_to_90 > 0 ? std::to_string(r.steps_to_90) : "never")
              << std::setw(12) << r.avg_active_cols
              << "\n";
  }
  
  return 0;
}
