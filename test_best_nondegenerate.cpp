// Find best non-degenerate config
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
    float post_active,
    int steps_multiplier = 1) {
  
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = seq_len;
  cfg.col_rows = 12;
  cfg.col_cols = 12;
  cfg.pot_h = 12;  // Larger RF as user suggested
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
  cfg.temp_seq_perm_inc = 0.50f;
  cfg.rng_seed = 123u;
  
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== FIND BEST NON-DEGENERATE CONFIG ===\n";
  std::cout << "Goal: Maximize pooling while maintaining:\n";
  std::cout << "  - Gap >20pp from control\n";
  std::cout << "  - Active cols 20-60 / 144 (14-42%)\n";
  std::cout << "  - Representation changes across sequence\n\n";
  
  struct Test {
    int seq_len;
    int activity;
    float spatial;
    float post_active;
    int train_steps;
  };
  
  std::vector<Test> tests = {
    {8, 30, 0.45f, 9.0f, 5000},
    {8, 32, 0.50f, 10.0f, 5000},
    {8, 35, 0.55f, 11.0f, 5000},
    {6, 32, 0.55f, 11.0f, 5000},
    {6, 35, 0.60f, 12.0f, 5000},
  };
  
  double best_pooled = 0.0;
  Test best_test = tests[0];
  
  for (const auto& test : tests) {
    std::cout << "Testing seq=" << test.seq_len << ", act=" << test.activity 
              << ", sp=" << test.spatial << ", pa=" << test.post_active << "\n";
    
    auto cfg = makeConfig(test.seq_len, test.activity, test.spatial, test.post_active);
    
    // TP ON
    HtmPipelineHarness htm_on(cfg);
    VerticalLineInputs inputs_on(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs_on.setSequenceProbability(1.0);
    inputs_on.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_on;
    double pooled_on = 0.0;
    double sum_last1000 = 0.0;
    
    for (int t = 1; t <= test.train_steps; ++t) {
      const std::vector<int> in = inputs_on.next(htm_on.rng());
      htm_on.step(t, in);
      pooled_on = m_on.temporalPoolingPercent(htm_on.learnCells01(t));
      
      if (t > test.train_steps - 1000) sum_last1000 += pooled_on;
    }
    double avg_on = sum_last1000 / 1000.0;
    
    // TP OFF
    cfg.temp_post_active_proximal_scale = 0.0f;
    cfg.temp_active_predict_proximal_scale = 0.0f;
    
    HtmPipelineHarness htm_off(cfg);
    VerticalLineInputs inputs_off(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs_off.setSequenceProbability(1.0);
    inputs_off.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_off;
    double sum_off_last1000 = 0.0;
    
    for (int t = 1; t <= test.train_steps; ++t) {
      const std::vector<int> in = inputs_off.next(htm_off.rng());
      htm_off.step(t, in);
      double pooled_off = m_off.temporalPoolingPercent(htm_off.learnCells01(t));
      
      if (t > test.train_steps - 1000) sum_off_last1000 += pooled_off;
    }
    double avg_off = sum_off_last1000 / 1000.0;
    
    int active = 0;
    const auto& cols = htm_on.activeColumnsInt01();
    for (auto c : cols) if (c == 1) ++active;
    
    double gap = avg_on - avg_off;
    bool valid_gap = gap >= 0.20;
    bool valid_sparsity = active >= 20 && active <= 60;
    
    std::cout << "  ON: " << avg_on << " (" << (avg_on*100) << "%)\n";
    std::cout << "  OFF: " << avg_off << " (" << (avg_off*100) << "%)\n";
    std::cout << "  Gap: " << gap << " (" << (gap*100) << " pp) " << (valid_gap ? "✓" : "✗") << "\n";
    std::cout << "  Active: " << active << " / 144 " << (valid_sparsity ? "✓" : "✗") << "\n";
    
    if (valid_gap && valid_sparsity && avg_on > best_pooled) {
      best_pooled = avg_on;
      best_test = test;
      std::cout << "  >>> NEW BEST <<<\n";
    }
    std::cout << "\n";
  }
  
  std::cout << "=== BEST NON-DEGENERATE CONFIG ===\n";
  std::cout << "Achieved: " << best_pooled << " (" << (best_pooled*100) << "%)\n";
  std::cout << "Config:\n";
  std::cout << "  input_cols (seq_len): " << best_test.seq_len << "\n";
  std::cout << "  desired_local_activity: " << best_test.activity << "\n";
  std::cout << "  pot_w: 4, pot_h: 12\n";
  std::cout << "  temp_spatial_perm_inc: " << best_test.spatial << "\n";
  std::cout << "  temp_active_predict_proximal_scale: 2.5\n";
  std::cout << "  temp_post_active_proximal_scale: " << best_test.post_active << "\n";
  std::cout << "  temp_seq_perm_inc: 0.50\n";
  std::cout << "  activation_threshold: 1\n";
  std::cout << "  Training steps: " << best_test.train_steps << "\n";
  
  if (best_pooled >= 0.90) {
    std::cout << "\n✓✓✓ TARGET MET! ✓✓✓\n";
  } else {
    std::cout << "\nNote: Achieving >90% non-degenerate pooling on an 8-position repeating\n";
    std::cout << "sequence may not be feasible. Best achieved: " << (best_pooled*100) << "%\n";
  }
  
  return 0;
}
