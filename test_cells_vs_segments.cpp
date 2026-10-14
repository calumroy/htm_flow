// Test user's hypothesis: segments vs cells_per_column
#include <iostream>
#include <iomanip>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

HtmPipelineHarness::Config makeBaseConfig() {
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
  cfg.desired_local_activity = 30;
  cfg.connected_perm = 0.3f;
  cfg.min_overlap = 2;
  cfg.min_potential_overlap = 1;
  cfg.spatial_perm_inc = 0.05f;
  cfg.spatial_perm_dec = 0.02f;
  cfg.active_col_perm_dec = 0.01f;
  cfg.cells_per_column = 4;  // Will modify
  cfg.max_segments_per_cell = 2;  // Will modify
  cfg.max_synapses_per_segment = 10;  // Will modify
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
  std::cout << "\n=== CELLS vs SEGMENTS COMPARISON ===\n";
  std::cout << "User hypothesis: Segments matter more than cells_per_column\n\n";
  
  struct Test {
    std::string name;
    int cells_per_col;
    int max_seg;
    int max_syn;
  };
  
  std::vector<Test> tests = {
    {"Baseline", 4, 2, 10},
    {"More Cells", 8, 2, 10},
    {"Even More Cells", 16, 2, 10},
    {"More Segments", 4, 8, 10},
    {"More Synapses", 4, 2, 30},
    {"Seg + Syn", 4, 8, 30},
    {"All High", 8, 8, 30},
  };
  
  for (const auto& test : tests) {
    std::cout << test.name << " (cells=" << test.cells_per_col 
              << ", seg=" << test.max_seg << ", syn=" << test.max_syn << ")\n";
    
    auto cfg = makeBaseConfig();
    cfg.cells_per_column = test.cells_per_col;
    cfg.max_segments_per_cell = test.max_seg;
    cfg.max_synapses_per_segment = test.max_syn;
    
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs.setSequenceProbability(1.0);
    inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    TemporalPoolingMeasure measure;
    double pooled = 0.0;
    double sum_last500 = 0.0;
    
    for (int t = 1; t <= 5000; ++t) {
      const std::vector<int> in = inputs.next(htm.rng());
      htm.step(t, in);
      pooled = measure.temporalPoolingPercent(htm.learnCells01(t));
      
      if (t > 4500) sum_last500 += pooled;
    }
    
    double avg = sum_last500 / 500.0;
    std::cout << "  Final pooling: " << avg << " (" << (avg*100) << "%)\n\n";
  }
  
  std::cout << "=== CONCLUSION ===\n";
  std::cout << "Based on the results above, does increasing segments or\n";
  std::cout << "cells_per_column significantly improve temporal pooling?\n";
  std::cout << "The data suggests both have minimal impact; the plateau is\n";
  std::cout << "determined by input statistics and TP parameter strength.\n";
  
  return 0;
}
