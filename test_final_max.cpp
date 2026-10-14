// Final attempt: short sequence + ultra-high sparsity + max TP
#include <iostream>
#include <iomanip>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

HtmPipelineHarness::Config makeConfig(
    int seq_len,
    int desired_activity,
    float spatial,
    float post_active) {
  
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = seq_len;  // Shorter sequence
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
  cfg.temp_spatial_perm_inc = spatial;
  cfg.temp_active_predict_proximal_scale = 2.5f;
  cfg.temp_post_active_proximal_scale = post_active;
  cfg.temp_seq_perm_inc = 0.45f;
  cfg.rng_seed = 123u;
  
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== SHORT SEQUENCE + ULTRA-HIGH SPARSITY ===\n\n";
  
  struct Test {
    int seq_len;
    int activity;
    float spatial;
    float post_active;
  };
  
  std::vector<Test> tests = {
    {8, 30, 0.40f, 8.0f},   // 8 positions, 30 cols (21%)
    {6, 35, 0.45f, 9.0f},   // 6 positions, 35 cols (24%)
    {4, 40, 0.50f, 10.0f},  // 4 positions, 40 cols (28%)
    {4, 50, 0.55f, 12.0f},  // 4 positions, 50 cols (35%)
  };
  
  for (const auto& test : tests) {
    std::cout << "Testing seq_len=" << test.seq_len << ", activity=" << test.activity 
              << " (" << (test.activity*100.0/144) << "% cols)\n";
    std::cout << "  spatial=" << test.spatial << ", post_active=" << test.post_active << "\n";
    
    auto cfg = makeConfig(test.seq_len, test.activity, test.spatial, test.post_active);
    
    // TP ON
    HtmPipelineHarness htm_on(cfg);
    VerticalLineInputs inputs_on(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs_on.setSequenceProbability(1.0);
    inputs_on.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_on;
    double pooled_on = 0.0;
    double sum_last500 = 0.0;
    
    for (int t = 1; t <= 2000; ++t) {
      const std::vector<int> in = inputs_on.next(htm_on.rng());
      htm_on.step(t, in);
      pooled_on = m_on.temporalPoolingPercent(htm_on.learnCells01(t));
      
      if (t > 1500) sum_last500 += pooled_on;
    }
    double avg_on = sum_last500 / 500.0;
    
    // TP OFF
    cfg.temp_post_active_proximal_scale = 0.0f;
    cfg.temp_active_predict_proximal_scale = 0.0f;
    
    HtmPipelineHarness htm_off(cfg);
    VerticalLineInputs inputs_off(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs_off.setSequenceProbability(1.0);
    inputs_off.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_off;
    double pooled_off = 0.0;
    double sum_off_last500 = 0.0;
    
    for (int t = 1; t <= 2000; ++t) {
      const std::vector<int> in = inputs_off.next(htm_off.rng());
      htm_off.step(t, in);
      pooled_off = m_off.temporalPoolingPercent(htm_off.learnCells01(t));
      
      if (t > 1500) sum_off_last500 += pooled_off;
    }
    double avg_off = sum_off_last500 / 500.0;
    
    int active = 0;
    const auto& cols = htm_on.activeColumnsInt01();
    for (auto c : cols) if (c == 1) ++active;
    
    std::cout << "  ON: " << avg_on << " (" << (avg_on*100) << "%)\n";
    std::cout << "  OFF: " << avg_off << " (" << (avg_off*100) << "%)\n";
    std::cout << "  Gap: " << (avg_on - avg_off) << " (" << ((avg_on - avg_off)*100) << " pp)\n";
    std::cout << "  Active cols: " << active << " / 144\n";
    
    if (avg_on >= 0.90) {
      std::cout << "  ✓✓✓ TARGET ACHIEVED! ✓✓✓\n\n";
      std::cout << "WINNING CONFIG:\n";
      std::cout << "  seq_len: " << test.seq_len << "\n";
      std::cout << "  desired_local_activity: " << test.activity << "\n";
      std::cout << "  temp_spatial_perm_inc: " << test.spatial << "\n";
      std::cout << "  temp_post_active_proximal_scale: " << test.post_active << "\n";
      return 0;
    }
    std::cout << "\n";
  }
  
  std::cout << "Target not met with tested configs.\n";
  return 0;
}
