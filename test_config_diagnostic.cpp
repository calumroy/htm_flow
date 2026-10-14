// Diagnostic: Verify TP config and measure baseline carry-over sources
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"
#include "test/test_utils/tp_metrics.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::TemporalPoolingMeasure;

// Helper: compute set overlap
double set_overlap(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.empty()) return 0.0;
  int overlap = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (a[i] && b[i]) ++overlap;
  }
  int count_a = 0;
  for (auto v : a) if (v) ++count_a;
  return count_a > 0 ? static_cast<double>(overlap) / count_a : 0.0;
}

// Helper: count active
int count_active(const std::vector<int>& vec) {
  int count = 0;
  for (auto v : vec) if (v) ++count;
  return count;
}

// Helper: expected chance overlap for two random sparse sets
double chance_overlap(int total_size, int active_count) {
  if (active_count == 0 || total_size == 0) return 0.0;
  double p = static_cast<double>(active_count) / total_size;
  return p; // Expected fraction of prev active that are active again by chance
}

HtmPipelineHarness::Config makeMaxPoolingConfig() {
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
  std::cout << "\n=== CONFIG & BASELINE DIAGNOSTIC ===\n\n";
  
  // Test both ON and OFF
  for (int test_idx = 0; test_idx < 2; ++test_idx) {
    bool tp_on = (test_idx == 0);
    std::cout << (tp_on ? "TP ON" : "TP OFF") << " Configuration:\n";
    
    auto cfg = makeMaxPoolingConfig();
    if (!tp_on) {
      cfg.temp_post_active_proximal_scale = 0.0f;
      cfg.temp_active_predict_proximal_scale = 0.0f;
    }
    
    // Print relevant config params
    std::cout << "  temp_post_active_proximal_scale: " << cfg.temp_post_active_proximal_scale << "\n";
    std::cout << "  temp_active_predict_proximal_scale: " << cfg.temp_active_predict_proximal_scale << "\n";
    std::cout << "  temp_spatial_perm_inc: " << cfg.temp_spatial_perm_inc << "\n";
    std::cout << "  temp_seq_perm_inc: " << cfg.temp_seq_perm_inc << "\n";
    std::cout << "  desired_local_activity: " << cfg.desired_local_activity << "\n";
    std::cout << "  cells_per_column: " << cfg.cells_per_column << "\n";
    std::cout << "  pot_w x pot_h: " << cfg.pot_w << " x " << cfg.pot_h << "\n";
    std::cout << "  inhib_w x inhib_h: " << cfg.inhib_w << " x " << cfg.inhib_h << "\n";
    
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    inputs.setSequenceProbability(1.0);
    inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    const int total_cells = cfg.col_rows * cfg.col_cols * cfg.cells_per_column;
    const int total_cols = cfg.col_rows * cfg.col_cols;
    
    std::cout << "  Total columns: " << total_cols << "\n";
    std::cout << "  Total cells: " << total_cells << "\n";
    
    // Measure over first 100 steps after warmup
    const int warmup = 500;
    const int measure_steps = 100;
    
    TemporalPoolingMeasure measure;
    std::vector<int> prev_active_cols;
    std::vector<uint8_t> prev_learn_cells;
    
    double sum_l1_cell_overlap = 0.0;
    double sum_l1_col_overlap = 0.0;
    int count_overlap = 0;
    
    for (int t = 1; t <= warmup + measure_steps; ++t) {
      const std::vector<int> input_sdr = inputs.next(htm.rng());
      htm.step(t, input_sdr);
      
      if (t > warmup) {
        const auto& current_cols = htm.activeColumnsInt01();
        const auto& current_cells = htm.learnCells01(t);
        
        // L1 column overlap
        if (!prev_active_cols.empty()) {
          double col_overlap = set_overlap(prev_active_cols, current_cols);
          sum_l1_col_overlap += col_overlap;
        }
        
        // L1 learning cell overlap (the metric)
        if (!prev_learn_cells.empty()) {
          double cell_overlap = measure.temporalPoolingPercent(current_cells);
          sum_l1_cell_overlap += cell_overlap;
          count_overlap++;
        }
        
        prev_active_cols = current_cols;
        prev_learn_cells = current_cells;
      } else {
        // Just update prev at end of warmup
        prev_active_cols = htm.activeColumnsInt01();
        prev_learn_cells = htm.learnCells01(t);
      }
    }
    
    double avg_cell_overlap = count_overlap > 0 ? sum_l1_cell_overlap / count_overlap : 0.0;
    double avg_col_overlap = count_overlap > 0 ? sum_l1_col_overlap / count_overlap : 0.0;
    
    // Final snapshot
    auto final_cols = htm.activeColumnsInt01();
    auto final_cells = htm.learnCells01(warmup + measure_steps);
    int active_cols = count_active(final_cols);
    int active_cells = 0;
    for (auto c : final_cells) if (c) ++active_cells;
    
    double chance_col = chance_overlap(total_cols, active_cols);
    double chance_cell = chance_overlap(total_cells, active_cells);
    
    std::cout << "\nMeasurements (averaged over " << measure_steps << " steps after " << warmup << " warmup):\n";
    std::cout << "  L1 active columns: " << active_cols << " / " << total_cols 
              << " (" << (active_cols * 100.0 / total_cols) << "%)\n";
    std::cout << "  L1 active learning cells: " << active_cells << " / " << total_cells 
              << " (" << (active_cells * 100.0 / total_cells) << "%)\n";
    std::cout << "  L1 column overlap (t-1 → t): " << avg_col_overlap 
              << " (" << (avg_col_overlap * 100) << "%)\n";
    std::cout << "  L1 learning cell overlap (t-1 → t): " << avg_cell_overlap 
              << " (" << (avg_cell_overlap * 100) << "%) ← THE METRIC\n";
    std::cout << "  Chance column overlap (random): " << chance_col 
              << " (" << (chance_col * 100) << "%)\n";
    std::cout << "  Chance cell overlap (random): " << chance_cell 
              << " (" << (chance_cell * 100) << "%)\n";
    std::cout << "  Excess over chance (cells): " 
              << (avg_cell_overlap - chance_cell) 
              << " (" << ((avg_cell_overlap - chance_cell) * 100) << " pp)\n";
    std::cout << "\n";
  }
  
  std::cout << "\n=== ANALYSIS ===\n";
  std::cout << "If TP OFF shows high cell overlap (>40%), investigate:\n";
  std::cout << "1. High column overlap (L1 columns not changing much)\n";
  std::cout << "2. High sparsity (many columns active → chance overlap high)\n";
  std::cout << "3. Bursting cells (multiple cells per column always active)\n";
  std::cout << "4. Learning cell selection (which cells are marked as 'learning')\n";
  std::cout << "5. Metric calculation (is it measuring what we think?)\n";
  
  return 0;
}
