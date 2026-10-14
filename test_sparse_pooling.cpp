// Try high sparsity approach for >90% pooling
#include <iostream>
#include <iomanip>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

HtmPipelineHarness::Config makeConfig(
    int desired_activity,
    float spatial_perm_inc,
    float post_active_scale,
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
  cfg.activation_threshold = activation_thresh;
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
  
  // Test: Very high sparsity + ultra-strong TP
  std::cout << "\nTesting HIGH SPARSITY approach:\n";
  std::cout << "Hypothesis: More active columns = more overlap opportunity\n\n";
  
  struct TestSpec {
    int desired_activity;
    float spatial;
    float post_active;
    int thresh;
  };
  
  std::vector<TestSpec> tests = {
    {15, 0.30f, 6.0f, 1},  // 15 cols active (~10%)
    {20, 0.30f, 6.0f, 1},  // 20 cols active (~14%)
    {25, 0.35f, 7.0f, 1},  // 25 cols active (~17%)
    {30, 0.35f, 8.0f, 1},  // 30 cols active (~21%)
    {40, 0.40f, 10.0f, 1}, // 40 cols active (~28%)
  };
  
  for (const auto& spec : tests) {
    std::cout << "Testing desired_activity=" << spec.desired_activity 
              << " (" << (spec.desired_activity * 100.0 / 144) << "% of cols)\n";
    
    auto cfg = makeConfig(spec.desired_activity, spec.spatial, spec.post_active, spec.thresh);
    
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs.setSequenceProbability(1.0);
    inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure measure;
    double pooled = 0.0;
    int steps_to_90 = -1;
    double max_pooled = 0.0;
    
    for (int t = 1; t <= 5000; ++t) {
      const std::vector<int> in = inputs.next(htm.rng());
      htm.step(t, in);
      pooled = measure.temporalPoolingPercent(htm.learnCells01(t));
      
      if (pooled > max_pooled) max_pooled = pooled;
      
      if (steps_to_90 < 0 && pooled >= 0.90) {
        steps_to_90 = t;
        std::cout << "  ✓✓✓ REACHED 90% at step " << t << "! ✓✓✓\n";
      }
      
      if (t % 1000 == 0) {
        std::cout << "  Step " << t << ": pooled=" << pooled << " (max=" << max_pooled << ")\n";
      }
    }
    
    // Get active columns
    int active_count = 0;
    const auto& cols = htm.activeColumnsInt01();
    for (auto c : cols) if (c == 1) ++active_count;
    
    std::cout << "  Final: pooled=" << pooled << " (" << (pooled * 100) << "%)\n";
    std::cout << "  Max observed: " << max_pooled << " (" << (max_pooled * 100) << "%)\n";
    std::cout << "  Active cols: " << active_count << " / 144\n";
    std::cout << "  Steps to 90%: " << (steps_to_90 > 0 ? std::to_string(steps_to_90) : "never") << "\n\n";
    
    if (steps_to_90 > 0) {
      std::cout << "SUCCESS! Found config that reaches 90%!\n";
      std::cout << "Config: desired_activity=" << spec.desired_activity 
                << ", spatial=" << spec.spatial 
                << ", post_active=" << spec.post_active 
                << ", thresh=" << spec.thresh << "\n";
      break;
    }
  }
  
  return 0;
}
