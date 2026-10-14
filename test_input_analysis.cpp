// Analyze input sequence and receptive field effects
#include <iostream>
#include <iomanip>
#include <vector>

#include "test/test_utils/tp_harness.hpp"
#include "test/test_utils/tp_inputs.hpp"

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::VerticalLineInputs;

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

int count_active(const std::vector<int>& vec) {
  int count = 0;
  for (auto v : vec) if (v) ++count;
  return count;
}

HtmPipelineHarness::Config makeConfig(int pot_w, int pot_h, int desired_activity) {
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
  // TP OFF
  cfg.temp_spatial_perm_inc = 0.0f;
  cfg.temp_active_predict_proximal_scale = 0.0f;
  cfg.temp_post_active_proximal_scale = 0.0f;
  cfg.temp_seq_perm_inc = 0.0f;
  cfg.rng_seed = 123u;
  return cfg;
}

int main() {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== INPUT SEQUENCE & RECEPTIVE FIELD ANALYSIS ===\n\n";
  
  // First, analyze raw input sequence overlap
  std::cout << "1. RAW INPUT SEQUENCE (8-position vertical lines, LeftToRight):\n";
  VerticalLineInputs inputs(8, 24, 8);
  inputs.setSequenceProbability(1.0);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  
  std::mt19937 rng(123);
  std::vector<std::vector<int>> seq_inputs;
  for (int i = 0; i < 10; ++i) {
    seq_inputs.push_back(inputs.next(rng));
  }
  
  // Measure input overlap
  double sum_input_overlap = 0.0;
  for (size_t i = 1; i < seq_inputs.size(); ++i) {
    double overlap = set_overlap(seq_inputs[i-1], seq_inputs[i]);
    sum_input_overlap += overlap;
    if (i <= 8) {
      int active_prev = count_active(seq_inputs[i-1]);
      int active_curr = count_active(seq_inputs[i]);
      std::cout << "  Step " << (i-1) << " → " << i << ": overlap=" << overlap 
                << " (" << (overlap*100) << "%), active=" << active_prev << "/" << (8*24) << "\n";
    }
  }
  double avg_input_overlap = sum_input_overlap / (seq_inputs.size() - 1);
  std::cout << "  Average input overlap: " << avg_input_overlap << " (" << (avg_input_overlap*100) << "%)\n\n";
  
  // Test different receptive field sizes with TP OFF
  std::cout << "2. RECEPTIVE FIELD SIZE EFFECT (TP OFF, desired_activity=30):\n\n";
  
  struct Test {
    int pot_w;
    int pot_h;
    std::string name;
  };
  
  std::vector<Test> tests = {
    {4, 12, "Baseline (4x12)"},
    {2, 6, "Small (2x6)"},
    {8, 24, "Large (8x24=full)"},
  };
  
  for (const auto& test : tests) {
    std::cout << test.name << " (pot_w=" << test.pot_w << ", pot_h=" << test.pot_h << "):\n";
    
    auto cfg = makeConfig(test.pot_w, test.pot_h, 30);
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs test_inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    test_inputs.setSequenceProbability(1.0);
    test_inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    const int warmup = 500;
    const int measure = 100;
    std::vector<int> prev_cols;
    double sum_col_overlap = 0.0;
    int count_overlap = 0;
    
    for (int t = 1; t <= warmup + measure; ++t) {
      const std::vector<int> input_sdr = test_inputs.next(htm.rng());
      htm.step(t, input_sdr);
      
      if (t > warmup) {
        const auto& current_cols = htm.activeColumnsInt01();
        if (!prev_cols.empty()) {
          double overlap = set_overlap(prev_cols, current_cols);
          sum_col_overlap += overlap;
          count_overlap++;
        }
        prev_cols = current_cols;
      } else if (t == warmup) {
        prev_cols = htm.activeColumnsInt01();
      }
    }
    
    double avg_col_overlap = count_overlap > 0 ? sum_col_overlap / count_overlap : 0.0;
    auto final_cols = htm.activeColumnsInt01();
    int active_cols = count_active(final_cols);
    
    std::cout << "  Column overlap: " << avg_col_overlap << " (" << (avg_col_overlap*100) << "%)\n";
    std::cout << "  Active columns: " << active_cols << " / 144\n\n";
  }
  
  // Test different sparsity levels
  std::cout << "3. SPARSITY EFFECT (TP OFF, pot_w=4, pot_h=12):\n\n";
  
  std::vector<int> sparsity_levels = {6, 15, 30, 45};
  
  for (int desired_activity : sparsity_levels) {
    std::cout << "desired_local_activity=" << desired_activity << ":\n";
    
    auto cfg = makeConfig(4, 12, desired_activity);
    HtmPipelineHarness htm(cfg);
    VerticalLineInputs test_inputs(cfg.input_cols, cfg.input_rows, cfg.input_cols);
    test_inputs.setSequenceProbability(1.0);
    test_inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
    
    const int warmup = 500;
    const int measure = 100;
    std::vector<int> prev_cols;
    double sum_col_overlap = 0.0;
    int count_overlap = 0;
    
    for (int t = 1; t <= warmup + measure; ++t) {
      const std::vector<int> input_sdr = test_inputs.next(htm.rng());
      htm.step(t, input_sdr);
      
      if (t > warmup) {
        const auto& current_cols = htm.activeColumnsInt01();
        if (!prev_cols.empty()) {
          double overlap = set_overlap(prev_cols, current_cols);
          sum_col_overlap += overlap;
          count_overlap++;
        }
        prev_cols = current_cols;
      } else if (t == warmup) {
        prev_cols = htm.activeColumnsInt01();
      }
    }
    
    double avg_col_overlap = count_overlap > 0 ? sum_col_overlap / count_overlap : 0.0;
    auto final_cols = htm.activeColumnsInt01();
    int active_cols = count_active(final_cols);
    double chance = static_cast<double>(active_cols) / 144;
    
    std::cout << "  Column overlap: " << avg_col_overlap << " (" << (avg_col_overlap*100) << "%)\n";
    std::cout << "  Active columns: " << active_cols << " / 144 (" << (active_cols*100.0/144) << "%)\n";
    std::cout << "  Chance overlap: " << chance << " (" << (chance*100) << "%)\n";
    std::cout << "  Excess over chance: " << (avg_col_overlap - chance) 
              << " (" << ((avg_col_overlap - chance)*100) << " pp)\n\n";
  }
  
  std::cout << "=== CONCLUSION ===\n";
  std::cout << "High baseline column overlap (60%+) comes from:\n";
  std::cout << "1. Repeating 8-position sequence (input has structure, not random)\n";
  std::cout << "2. Adjacent positions share context (vertical line shifts one column)\n";
  std::cout << "3. High sparsity (33% active) → higher chance overlap\n";
  std::cout << "4. Wide receptive fields see overlapping input\n";
  std::cout << "5. Columns learn stable responses to their preferred patterns\n";
  
  return 0;
}
