// Test with LOWER sparsity for proper baseline
#include <iostream>
#include <iomanip>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

int count_active(const std::vector<int>& vec) {
  int count = 0;
  for (auto v : vec) if (v) ++count;
  return count;
}

HtmPipelineHarness::Config makeConfig(int desired_activity, float post_active, float spatial) {
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = 8;
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
  cfg.temp_active_predict_proximal_scale = 3.0f;
  cfg.temp_post_active_proximal_scale = post_active;
  cfg.temp_seq_perm_inc = 0.55f;
  cfg.rng_seed = 123u;
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== LOW SPARSITY TEST (PROPER BASELINE) ===\n\n";
  
  struct Test {
    int activity;
    float post_active;
    float spatial;
    int steps;
  };
  
  std::vector<Test> tests = {
    {6, 12.0f, 0.50f, 8000},   // Low sparsity, strong TP
    {6, 20.0f, 0.70f, 8000},   // Low sparsity, ultra-strong TP
    {10, 12.0f, 0.50f, 8000},  // Medium sparsity
  };
  
  for (const auto& t : tests) {
    std::cout << "Config: activity=" << t.activity << ", post_active=" << t.post_active 
              << ", spatial=" << t.spatial << "\n";
    
    // TP ON
    auto cfg_on = makeConfig(t.activity, t.post_active, t.spatial);
    HtmPipelineHarness htm_on(cfg_on);
    VerticalLineInputs inputs_on(cfg_on.input_cols, cfg_on.input_rows, cfg_on.input_cols);
    inputs_on.setSequenceProbability(1.0);
    inputs_on.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_on;
    double sum_on = 0.0;
    int count_on = 0;
    int first_90 = -1;
    
    for (int step = 1; step <= t.steps; ++step) {
      const std::vector<int> in = inputs_on.next(htm_on.rng());
      htm_on.step(step, in);
      double pooled = m_on.temporalPoolingPercent(htm_on.learnCells01(step));
      
      if (step > t.steps - 1000) {
        sum_on += pooled;
        count_on++;
      }
      
      if (first_90 < 0 && pooled >= 0.90) first_90 = step;
      
      if (step % 2000 == 0) {
        std::cout << "  Step " << step << ": " << pooled << " (" << (pooled*100) << "%)\n";
      }
    }
    double avg_on = count_on > 0 ? sum_on / count_on : 0.0;
    
    // TP OFF
    auto cfg_off = makeConfig(t.activity, 0.0f, 0.0f);
    HtmPipelineHarness htm_off(cfg_off);
    VerticalLineInputs inputs_off(cfg_off.input_cols, cfg_off.input_rows, cfg_off.input_cols);
    inputs_off.setSequenceProbability(1.0);
    inputs_off.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_off;
    double sum_off = 0.0;
    int count_off = 0;
    
    for (int step = 1; step <= t.steps; ++step) {
      const std::vector<int> in = inputs_off.next(htm_off.rng());
      htm_off.step(step, in);
      double pooled = m_off.temporalPoolingPercent(htm_off.learnCells01(step));
      
      if (step > t.steps - 1000) {
        sum_off += pooled;
        count_off++;
      }
    }
    double avg_off = count_off > 0 ? sum_off / count_off : 0.0;
    
    auto cols_on = htm_on.activeColumnsInt01();
    int active_on = count_active(cols_on);
    
    std::cout << "  ON: " << avg_on << " (" << (avg_on*100) << "%)\n";
    std::cout << "  OFF: " << avg_off << " (" << (avg_off*100) << "%)\n";
    std::cout << "  Gap: " << (avg_on - avg_off) << " (" << ((avg_on - avg_off)*100) << " pp)\n";
    std::cout << "  Active cols: " << active_on << " / 144\n";
    std::cout << "  First 90%: " << (first_90 > 0 ? std::to_string(first_90) : "never") << "\n";
    
    if (avg_on >= 0.90) {
      std::cout << "  ✓✓✓ TARGET MET! ✓✓✓\n";
    }
    std::cout << "\n";
  }
  
  return 0;
}
