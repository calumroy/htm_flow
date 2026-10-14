// Push toward 90% with very aggressive settings
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
    float active_pred,
    float post_active) {
  
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = seq_len;
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
  cfg.temp_active_predict_proximal_scale = active_pred;
  cfg.temp_post_active_proximal_scale = post_active;
  cfg.temp_seq_perm_inc = 0.55f;
  cfg.rng_seed = 123u;
  
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== PUSH TO 90% WITH ULTRA-AGGRESSIVE SETTINGS ===\n\n";
  
  struct Test {
    int seq;
    int act;
    float sp;
    float ap;
    float pa;
    int steps;
  };
  
  std::vector<Test> tests = {
    {8, 30, 0.50f, 3.0f, 12.0f, 8000},
    {8, 30, 0.60f, 4.0f, 15.0f, 8000},
    {8, 30, 0.70f, 5.0f, 20.0f, 8000},
    {8, 32, 0.70f, 5.0f, 20.0f, 8000},
    {8, 35, 0.75f, 6.0f, 25.0f, 10000},
  };
  
  for (const auto& t : tests) {
    std::cout << "Test: seq=" << t.seq << ", act=" << t.act 
              << ", sp=" << t.sp << ", ap=" << t.ap << ", pa=" << t.pa << "\n";
    
    auto cfg = makeConfig(t.seq, t.act, t.sp, t.ap, t.pa);
    
    // TP ON
    HtmPipelineHarness htm_on(cfg);
    VerticalLineInputs inputs_on(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs_on.setSequenceProbability(1.0);
    inputs_on.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure m_on;
    double pooled_on = 0.0;
    double sum_last1000 = 0.0;
    int first_90 = -1;
    int count_above_90 = 0;
    
    for (int t_step = 1; t_step <= t.steps; ++t_step) {
      const std::vector<int> in = inputs_on.next(htm_on.rng());
      htm_on.step(t_step, in);
      pooled_on = m_on.temporalPoolingPercent(htm_on.learnCells01(t_step));
      
      if (pooled_on >= 0.90 && first_90 < 0) first_90 = t_step;
      if (t_step > t.steps - 1000) {
        sum_last1000 += pooled_on;
        if (pooled_on >= 0.90) count_above_90++;
      }
      
      if (t_step % 2000 == 0) {
        std::cout << "  Step " << t_step << ": " << pooled_on << "\n";
      }
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
    
    for (int t_step = 1; t_step <= t.steps; ++t_step) {
      const std::vector<int> in = inputs_off.next(htm_off.rng());
      htm_off.step(t_step, in);
      double pooled_off = m_off.temporalPoolingPercent(htm_off.learnCells01(t_step));
      
      if (t_step > t.steps - 1000) sum_off_last1000 += pooled_off;
    }
    double avg_off = sum_off_last1000 / 1000.0;
    
    int active = 0;
    const auto& cols = htm_on.activeColumnsInt01();
    for (auto c : cols) if (c == 1) ++active;
    
    double gap = avg_on - avg_off;
    
    std::cout << "  ON avg: " << avg_on << " (" << (avg_on*100) << "%)\n";
    std::cout << "  OFF avg: " << avg_off << " (" << (avg_off*100) << "%)\n";
    std::cout << "  Gap: " << gap << " (" << (gap*100) << " pp)\n";
    std::cout << "  Active cols: " << active << " / 144\n";
    std::cout << "  First hit 90%: " << (first_90 > 0 ? std::to_string(first_90) : "never") << "\n";
    std::cout << "  Steps >90% in last 1000: " << count_above_90 << " / 1000\n";
    
    bool target_met = avg_on >= 0.90;
    bool gap_ok = gap >= 0.10;
    bool sparsity_ok = active >= 20 && active <= 80;
    
    std::cout << "  Target >90%: " << (target_met ? "✓" : "✗") << "\n";
    std::cout << "  Gap >10pp: " << (gap_ok ? "✓" : "✗") << "\n";
    std::cout << "  Sparsity OK: " << (sparsity_ok ? "✓" : "✗") << "\n";
    
    if (target_met && gap_ok && sparsity_ok) {
      std::cout << "\n✓✓✓ ALL CRITERIA MET! ✓✓✓\n";
      return 0;
    }
    std::cout << "\n";
  }
  
  std::cout << "Target 90% not achieved with tested configs.\n";
  std::cout << "This may indicate fundamental limitations of achieving\n";
  std::cout << ">90% pooling in a non-degenerate system.\n";
  
  return 0;
}
