// Comprehensive diagnostic: Why does TP plateau at ~54%?
#include <iostream>
#include <iomanip>
#include <vector>

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

double set_overlap(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.empty()) return 0.0;
  int overlap = 0, count_a = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (a[i]) count_a++;
    if (a[i] && b[i]) overlap++;
  }
  return count_a > 0 ? static_cast<double>(overlap) / count_a : 0.0;
}

HtmPipelineHarness::Config makeConfig(int pot_w, int pot_h, int activity) {
  HtmPipelineHarness::Config cfg;
  cfg.input_rows = 24;
  cfg.input_cols = 8;
  cfg.col_rows = 12;
  cfg.col_cols = 12;
  cfg.pot_h = pot_h;
  cfg.pot_w = pot_w;
  cfg.center_pot_synapses = false;
  cfg.wrap_input = true;
  cfg.inhib_w = 7;
  cfg.inhib_h = 7;
  cfg.desired_local_activity = activity;
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
  cfg.temp_spatial_perm_inc = 0.50f;
  cfg.temp_active_predict_proximal_scale = 3.0f;
  cfg.temp_post_active_proximal_scale = 12.0f;
  cfg.temp_seq_perm_inc = 0.55f;
  cfg.rng_seed = 123u;
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== COMPREHENSIVE TP PLATEAU DIAGNOSTIC ===\n\n";
  
  // Test: pot_w/pot_h independently with low sparsity
  std::cout << "1. RECEPTIVE FIELD SIZE (TP ON, activity=10):\n\n";
  
  struct RFTest {
    int pot_w;
    int pot_h;
    std::string name;
  };
  
  std::vector<RFTest> rf_tests = {
    {2, 6, "Small (2x6)"},
    {4, 12, "Baseline (4x12)"},
    {6, 18, "Large (6x18)"},
    {8, 24, "XLarge (8x24)"},
  };
  
  for (const auto& test : rf_tests) {
    std::cout << test.name << ":\n";
    
    auto cfg = makeConfig(test.pot_w, test.pot_h, 10);
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs.setSequenceProbability(1.0);
    inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure measure;
    double sum_pooled = 0.0;
    double sum_col_overlap = 0.0;
    int count = 0;
    std::vector<int> prev_cols;
    
    for (int t = 1; t <= 3000; ++t) {
      const std::vector<int> in = inputs.next(htm.rng());
      htm.step(t, in);
      
      if (t > 2000) {
        double pooled = measure.temporalPoolingPercent(htm.learnCells01(t));
        sum_pooled += pooled;
        
        const auto& cols = htm.activeColumnsInt01();
        if (!prev_cols.empty()) {
          sum_col_overlap += set_overlap(prev_cols, cols);
        }
        prev_cols = cols;
        count++;
      }
    }
    
    double avg_pooled = count > 0 ? sum_pooled / count : 0.0;
    double avg_col_overlap = count > 0 ? sum_col_overlap / count : 0.0;
    auto final_cols = htm.activeColumnsInt01();
    int active = count_active(final_cols);
    
    std::cout << "  Cell pooling: " << avg_pooled << " (" << (avg_pooled*100) << "%)\n";
    std::cout << "  Column overlap: " << avg_col_overlap << " (" << (avg_col_overlap*100) << "%)\n";
    std::cout << "  Active cols: " << active << " / 144\n\n";
  }
  
  // Test: segments vs baseline at low sparsity
  std::cout << "2. SEGMENTS VS BASELINE (TP ON, activity=10, pot=4x12):\n\n";
  
  struct SegTest {
    int max_seg;
    int max_syn;
    std::string name;
  };
  
  std::vector<SegTest> seg_tests = {
    {2, 10, "Baseline (seg=2, syn=10)"},
    {4, 20, "More (seg=4, syn=20)"},
    {8, 30, "Many (seg=8, syn=30)"},
  };
  
  for (const auto& test : seg_tests) {
    std::cout << test.name << ":\n";
    
    auto cfg = makeConfig(4, 12, 10);
    cfg.max_segments_per_cell = test.max_seg;
    cfg.max_synapses_per_segment = test.max_syn;
    
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs.setSequenceProbability(1.0);
    inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure measure;
    double sum_pooled = 0.0;
    int count = 0;
    
    for (int t = 1; t <= 3000; ++t) {
      const std::vector<int> in = inputs.next(htm.rng());
      htm.step(t, in);
      
      if (t > 2000) {
        sum_pooled += measure.temporalPoolingPercent(htm.learnCells01(t));
        count++;
      }
    }
    
    double avg_pooled = count > 0 ? sum_pooled / count : 0.0;
    std::cout << "  Pooling: " << avg_pooled << " (" << (avg_pooled*100) << "%)\n\n";
  }
  
  std::cout << "=== ANALYSIS ===\n";
  std::cout << "If receptive field size helps significantly, report that.\n";
  std::cout << "If segments hurt at low sparsity too, it confirms the mechanism.\n";
  std::cout << "If pooling plateaus regardless, it's a fundamental limit.\n";
  
  return 0;
}
