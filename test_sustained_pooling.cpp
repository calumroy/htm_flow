// Validate sustained >90% temporal pooling
#include <iostream>
#include <iomanip>
#include <vector>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

HtmPipelineHarness::Config makeConfig(
    int desired_activity,
    float spatial_perm_inc,
    float post_active_scale) {
  
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
  cfg.desired_local_activity = desired_activity;
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
  cfg.activation_threshold = 1;
  cfg.seq_perm_inc = 0.05f;
  cfg.seq_perm_dec = 0.02f;
  cfg.temp_spatial_perm_inc = spatial_perm_inc;
  cfg.temp_active_predict_proximal_scale = 2.0f;
  cfg.temp_post_active_proximal_scale = post_active_scale;
  cfg.temp_seq_perm_inc = 0.40f;
  cfg.rng_seed = 123u;
  
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== SUSTAINED 90% TEMPORAL POOLING TEST ===\n\n";
  
  // Test config that showed promise
  auto cfg = makeConfig(25, 0.35f, 7.0f);
  
  // Run with TP ON
  std::cout << "Running with TP ON (desired_activity=25, spatial=0.35, post_active=7.0)...\n";
  HtmPipelineHarness htm_on(cfg);
  VerticalLineInputs inputs_on(cfg.input_cols, cfg.input_rows, cfg.input_cols);
  inputs_on.setSequenceProbability(1.0);
  inputs_on.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  
  TemporalPoolingMeasure measure_on;
  std::vector<double> pooling_values;
  int first_90 = -1;
  int sustained_count = 0;
  double sum_pooled_last1000 = 0.0;
  
  for (int t = 1; t <= 3000; ++t) {
    const std::vector<int> in = inputs_on.next(htm_on.rng());
    htm_on.step(t, in);
    double pooled = measure_on.temporalPoolingPercent(htm_on.learnCells01(t));
    
    if (t % 100 == 0) pooling_values.push_back(pooled);
    
    if (pooled >= 0.90) {
      if (first_90 < 0) first_90 = t;
      if (t > 1000) sustained_count++;
    }
    
    if (t > 2000) sum_pooled_last1000 += pooled;
    
    if (t % 500 == 0) {
      std::cout << "  Step " << t << ": pooled=" << pooled << "\n";
    }
  }
  
  double avg_last1000 = sum_pooled_last1000 / 1000.0;
  
  // Get sparsity
  int active_on = 0;
  const auto& cols_on = htm_on.activeColumnsInt01();
  for (auto c : cols_on) if (c == 1) ++active_on;
  
  // Run with TP OFF (control)
  std::cout << "\nRunning with TP OFF (control)...\n";
  cfg.temp_post_active_proximal_scale = 0.0f;
  cfg.temp_active_predict_proximal_scale = 0.0f;
  
  HtmPipelineHarness htm_off(cfg);
  VerticalLineInputs inputs_off(cfg.input_cols, cfg.input_rows, cfg.input_cols);
  inputs_off.setSequenceProbability(1.0);
  inputs_off.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  
  TemporalPoolingMeasure measure_off;
  double pooled_off = 0.0;
  double sum_pooled_off_last1000 = 0.0;
  
  for (int t = 1; t <= 3000; ++t) {
    const std::vector<int> in = inputs_off.next(htm_off.rng());
    htm_off.step(t, in);
    pooled_off = measure_off.temporalPoolingPercent(htm_off.learnCells01(t));
    
    if (t > 2000) sum_pooled_off_last1000 += pooled_off;
    
    if (t % 500 == 0) {
      std::cout << "  Step " << t << ": pooled=" << pooled_off << "\n";
    }
  }
  
  double avg_off_last1000 = sum_pooled_off_last1000 / 1000.0;
  
  int active_off = 0;
  const auto& cols_off = htm_off.activeColumnsInt01();
  for (auto c : cols_off) if (c == 1) ++active_off;
  
  // Results
  std::cout << "\n=== RESULTS ===\n\n";
  std::cout << "TP ON:\n";
  std::cout << "  Average (last 1000 steps): " << avg_last1000 << " (" << (avg_last1000*100) << "%)\n";
  std::cout << "  First reached 90%: step " << (first_90 > 0 ? std::to_string(first_90) : "never") << "\n";
  std::cout << "  Steps >90% (after step 1000): " << sustained_count << " / 2000\n";
  std::cout << "  Active columns: " << active_on << " / 144 (" << (active_on*100.0/144) << "%)\n";
  
  std::cout << "\nTP OFF (control):\n";
  std::cout << "  Average (last 1000 steps): " << avg_off_last1000 << " (" << (avg_off_last1000*100) << "%)\n";
  std::cout << "  Active columns: " << active_off << " / 144 (" << (active_off*100.0/144) << "%)\n";
  
  std::cout << "\nGap: " << (avg_last1000 - avg_off_last1000) << " (" << ((avg_last1000 - avg_off_last1000)*100) << " pp)\n";
  
  std::cout << "\n=== VALIDATION ===\n";
  bool target_met = avg_last1000 >= 0.90;
  bool gap_sufficient = (avg_last1000 - avg_off_last1000) >= 0.30;
  bool sparsity_ok = active_on >= 10 && active_on <= 80;  // 7-55% of columns
  
  std::cout << "✓ Target >90%: " << (target_met ? "PASS" : "FAIL") 
            << " (avg=" << avg_last1000 << ")\n";
  std::cout << "✓ Gap >30pp from control: " << (gap_sufficient ? "PASS" : "FAIL") 
            << " (gap=" << (avg_last1000 - avg_off_last1000) << ")\n";
  std::cout << "✓ Non-degenerate sparsity: " << (sparsity_ok ? "PASS" : "FAIL") 
            << " (cols=" << active_on << ")\n";
  
  if (target_met && gap_sufficient && sparsity_ok) {
    std::cout << "\n✓✓✓ ALL CRITERIA MET! ✓✓✓\n";
    std::cout << "\nFINAL CONFIG:\n";
    std::cout << "  desired_local_activity: 25\n";
    std::cout << "  temp_spatial_perm_inc: 0.35\n";
    std::cout << "  temp_active_predict_proximal_scale: 2.0\n";
    std::cout << "  temp_post_active_proximal_scale: 7.0\n";
    std::cout << "  temp_seq_perm_inc: 0.40\n";
    std::cout << "  activation_threshold: 1\n";
  } else {
    std::cout << "\n✗ Target not met, need more tuning.\n";
  }
  
  return 0;
}
