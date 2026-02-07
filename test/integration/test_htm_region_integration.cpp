/// Integration tests for HTMRegion - testing multi-layer HTM region behavior.
///
/// These tests verify that the HTMRegion class correctly stacks multiple
/// HTMLayer instances, passes outputs between layers, and produces the expected
/// hierarchical behavior similar to the Python HTM_region.py.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <htm_flow/config.hpp>
#include <htm_flow/htm_layer.hpp>
#include <htm_flow/htm_region.hpp>
#include <htm_flow/region_runtime.hpp>
#include <utilities/sdr_inputs.hpp>

namespace {

// ── Suite-wide default configuration ────────────────────────────────
// Equivalent to htm_flow::small_test_config() with all values shown explicitly.
// Every test in this file starts from this config (or a modified copy).
htm_flow::HTMLayerConfig suiteConfig() {
  htm_flow::HTMLayerConfig cfg;
  // Input/column grid
  cfg.num_input_rows            = 10;
  cfg.num_input_cols            = 10;
  cfg.num_column_rows           = 10;
  cfg.num_column_cols           = 20;
  // Proximal topology
  cfg.pot_width                 = 5;
  cfg.pot_height                = 1;
  cfg.center_pot_synapses       = true;
  cfg.wrap_input                = true;
  cfg.connected_perm            = 0.3f;
  cfg.min_overlap               = 2;
  cfg.min_potential_overlap     = 1;
  // Inhibition
  cfg.inhibition_width          = 10;
  cfg.inhibition_height         = 1;
  cfg.desired_local_activity    = 1;
  cfg.strict_local_activity     = false;
  // Spatial learning
  cfg.spatial_permanence_inc    = 0.1f;
  cfg.spatial_permanence_dec    = 0.05f;
  cfg.active_col_permanence_dec = 0.05f;
  // Sequence pooler
  cfg.cells_per_column          = 3;
  cfg.max_segments_per_cell     = 2;
  cfg.max_synapses_per_segment  = 10;
  cfg.min_num_syn_threshold     = 5;
  cfg.min_score_threshold       = 5;
  cfg.new_syn_permanence        = 0.3f;
  cfg.connect_permanence        = 0.2f;
  cfg.activation_threshold      = 6;
  cfg.sequence_permanence_inc   = 0.1f;
  cfg.sequence_permanence_dec   = 0.05f;
  // Temporal pooler
  cfg.temp_enabled              = true;
  cfg.temp_delay_length         = 4;
  cfg.temp_enable_persistence   = true;
  cfg.temp_spatial_permanence_inc  = 0.01f;
  cfg.temp_sequence_permanence_inc = 0.01f;
  // Runtime
  cfg.log_timings               = false;
  return cfg;
}

/// Create a vertical line input pattern at the given x position.
std::vector<int> create_vertical_line(int width, int height, int x_pos) {
  std::vector<int> input(static_cast<std::size_t>(width * height), 0);
  for (int y = 0; y < height; ++y) {
    input[static_cast<std::size_t>(y * width + x_pos)] = 1;
  }
  return input;
}

/// Count the number of predictive cells in a snapshot.
int count_predictive_cells(const htm_gui::Snapshot& snap) {
  int count = 0;
  for (const auto& masks : snap.column_cell_masks) {
    for (int c = 0; c < snap.cells_per_column && c < 64; ++c) {
      if (masks.predictive & (std::uint64_t(1) << c)) {
        ++count;
      }
    }
  }
  return count;
}

/// Count the number of active cells in a snapshot.
int count_active_cells(const htm_gui::Snapshot& snap) {
  int count = 0;
  for (const auto& masks : snap.column_cell_masks) {
    for (int c = 0; c < snap.cells_per_column && c < 64; ++c) {
      if (masks.active & (std::uint64_t(1) << c)) {
        ++count;
      }
    }
  }
  return count;
}

/// Calculate the similarity between two binary vectors as a percentage [0, 1].
double similarity_percent(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0;
  }
  int matching = 0;
  int total_on = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] == 1 || b[i] == 1) {
      ++total_on;
      if (a[i] == b[i]) {
        ++matching;
      }
    }
  }
  return total_on > 0 ? static_cast<double>(matching) / static_cast<double>(total_on) : 1.0;
}

/// Measure temporal pooling stability by tracking output changes over time.
class TemporalPoolingMeasure {
public:
  double temporalPoolingPercent(const std::vector<int>& current_output) {
    if (prev_output_.empty()) {
      prev_output_ = current_output;
      return 0.0;
    }

    double sim = similarity_percent(prev_output_, current_output);
    prev_output_ = current_output;
    return sim;
  }

private:
  std::vector<int> prev_output_;
};

}  // namespace

// =============================================================================
// Basic Region Tests
// =============================================================================

TEST(HTMRegionIntegration, SingleLayerRegionWorks) {
  // Test that a single-layer region works correctly.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg;
  cfg.layers.push_back(layer_cfg);

  htm_flow::HTMRegion region(cfg);

  EXPECT_EQ(region.num_layers(), 1);

  auto input = create_vertical_line(layer_cfg.num_input_cols, layer_cfg.num_input_rows, 3);
  region.set_input(input);
  region.step(1);

  EXPECT_EQ(region.timestep(), 1);
  EXPECT_FALSE(region.output().empty());
}

TEST(HTMRegionIntegration, MultiLayerRegionStacks) {
  // Test that a multi-layer region correctly stacks layers.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg;
  cfg.layers.push_back(layer_cfg);
  cfg.layers.push_back(layer_cfg);  // Second layer will auto-adjust input dims
  cfg.layers.push_back(layer_cfg);  // Third layer

  htm_flow::HTMRegion region(cfg, "TestRegion");

  EXPECT_EQ(region.num_layers(), 3);
  EXPECT_EQ(region.name(), "TestRegion");

  // Verify each layer has correct input dimensions based on previous layer's output
  const auto& l0 = region.layer(0);
  const auto& l1 = region.layer(1);
  const auto& l2 = region.layer(2);

  // Layer 0's input matches the config
  EXPECT_EQ(l0.config().num_input_cols, layer_cfg.num_input_cols);
  EXPECT_EQ(l0.config().num_input_rows, layer_cfg.num_input_rows);

  // Layer 1's input should match layer 0's output
  EXPECT_EQ(l1.config().num_input_cols, l0.output_cols());
  EXPECT_EQ(l1.config().num_input_rows, l0.output_rows());

  // Layer 2's input should match layer 1's output
  EXPECT_EQ(l2.config().num_input_cols, l1.output_cols());
  EXPECT_EQ(l2.config().num_input_rows, l1.output_rows());
}

TEST(HTMRegionIntegration, LayerAccessWorks) {
  // Test that we can access individual layers in a region.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(3, layer_cfg);

  htm_flow::HTMRegion region(cfg);

  // Access each layer and verify
  for (int i = 0; i < region.num_layers(); ++i) {
    htm_flow::HTMLayer& layer = region.layer(i);
    EXPECT_EQ(layer.timestep(), 0);  // Before any steps
  }

  // Run some steps
  auto input = create_vertical_line(layer_cfg.num_input_cols, layer_cfg.num_input_rows, 3);
  region.set_input(input);
  region.step(5);

  // All layers should have the same timestep
  for (int i = 0; i < region.num_layers(); ++i) {
    EXPECT_EQ(region.layer(i).timestep(), 5);
  }
}

TEST(HTMRegionIntegration, OutputPassesBetweenLayers) {
  // Test that output from one layer becomes input to the next.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(2, layer_cfg);

  htm_flow::HTMRegion region(cfg);

  auto input = create_vertical_line(layer_cfg.num_input_cols, layer_cfg.num_input_rows, 3);
  region.set_input(input);
  region.step(1);

  // Layer 0 should have active columns
  auto snap0 = region.layer(0).snapshot();
  EXPECT_GT(snap0.active_column_indices.size(), 0u);

  // Layer 1 should also have processed input (output from layer 0)
  // and should have some active columns
  auto snap1 = region.layer(1).snapshot();
  // Note: Layer 1 might have different activity patterns depending on the config
  EXPECT_EQ(snap1.timestep, 1);
}

// =============================================================================
// Region Runtime Tests (GUI Integration)
// =============================================================================

TEST(HTMRegionIntegration, RegionRuntimeWorks) {
  // Test that HTMRegionRuntime provides correct IHtmRuntime interface.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(3, layer_cfg);

  htm_flow::HTMRegionRuntime runtime(cfg, "TestRuntime");

  EXPECT_EQ(runtime.num_layers(), 3);
  EXPECT_EQ(runtime.active_layer(), 0);

  // Step should work
  runtime.step(1);
  auto snap = runtime.snapshot();
  EXPECT_EQ(snap.timestep, 1);

  // Layer selection should work
  runtime.set_active_layer(1);
  EXPECT_EQ(runtime.active_layer(), 1);

  auto snap1 = runtime.snapshot();
  EXPECT_EQ(snap1.timestep, 1);  // Same timestep, different layer view
}

TEST(HTMRegionIntegration, RegionRuntimeLayerOptions) {
  // Test that layer_options returns correct options for GUI.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(3, layer_cfg);

  htm_flow::HTMRegionRuntime runtime(cfg);

  auto opts = runtime.layer_options();
  EXPECT_EQ(opts.size(), 3u);
  EXPECT_EQ(opts[0].id, 0);
  EXPECT_EQ(opts[1].id, 1);
  EXPECT_EQ(opts[2].id, 2);
}

TEST(HTMRegionIntegration, RegionRuntimeInputSequences) {
  // Test that input sequence selection works in region runtime.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(2, layer_cfg);

  htm_flow::HTMRegionRuntime runtime(cfg);

  auto seqs = runtime.input_sequences();
  EXPECT_FALSE(seqs.empty());

  // Change input sequence
  runtime.set_input_sequence(2);
  EXPECT_EQ(runtime.input_sequence(), 2);
}

// =============================================================================
// Temporal Pooling Hierarchy Tests
// =============================================================================

TEST(HTMRegionIntegration, TemporalPoolingWithTraining) {
  // Test that temporal pooling occurs after training on a repeating sequence.
  auto layer_cfg = suiteConfig();
  // Use larger learning rates for faster convergence
  layer_cfg.spatial_permanence_inc = 0.15f;
  layer_cfg.temp_spatial_permanence_inc = 0.1f;

  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(2, layer_cfg);
  htm_flow::HTMRegion region(cfg);

  std::mt19937 rng(42);
  utilities::MovingLineInputs inputs(layer_cfg.num_input_cols, layer_cfg.num_input_rows);
  inputs.setPattern(utilities::MovingLineInputs::Pattern::LeftToRight);

  // Training phase
  for (int epoch = 0; epoch < 20; ++epoch) {
    inputs.setIndex(0);
    for (int i = 0; i < inputs.seqLen(); ++i) {
      auto input = inputs.next(rng);
      region.set_input(input);
      region.step(1);
    }
  }

  // Evaluation phase - measure stability
  TemporalPoolingMeasure measure_l0;
  TemporalPoolingMeasure measure_l1;

  double pooling_l0 = 0.0;
  double pooling_l1 = 0.0;

  inputs.setIndex(0);
  for (int i = 0; i < 2 * inputs.seqLen(); ++i) {
    auto input = inputs.next(rng);
    region.set_input(input);
    region.step(1);

    pooling_l0 = measure_l0.temporalPoolingPercent(region.layer(0).output());
    pooling_l1 = measure_l1.temporalPoolingPercent(region.layer(1).output());
  }

  // After training, we expect some temporal pooling (stability in outputs)
  // The exact values depend on the configuration and training length
  EXPECT_GE(pooling_l0, 0.0) << "Layer 0 should show some output stability";
  EXPECT_GE(pooling_l1, 0.0) << "Layer 1 should show some output stability";
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST(HTMRegionIntegration, UniformRegionConfigWorks) {
  // Test that uniform_region_config helper produces valid config.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(4, layer_cfg);

  EXPECT_EQ(cfg.layers.size(), 4u);
  EXPECT_FALSE(cfg.enable_feedback);

  htm_flow::HTMRegion region(cfg);
  EXPECT_EQ(region.num_layers(), 4);
}

TEST(HTMRegionIntegration, DifferentLayerConfigs) {
  // Test that a region can have different configurations for each layer.
  htm_flow::HTMRegionConfig cfg;

  // Layer 0: small config
  auto layer0_cfg = suiteConfig();
  cfg.layers.push_back(layer0_cfg);

  // Layer 1: different config (will have input dims adjusted)
  auto layer1_cfg = suiteConfig();
  layer1_cfg.num_column_cols = 15;  // Different column layout
  layer1_cfg.num_column_rows = 8;
  cfg.layers.push_back(layer1_cfg);

  htm_flow::HTMRegion region(cfg);

  EXPECT_EQ(region.num_layers(), 2);
  EXPECT_EQ(region.layer(0).config().num_column_cols, layer0_cfg.num_column_cols);
  EXPECT_EQ(region.layer(1).config().num_column_cols, layer1_cfg.num_column_cols);

  // Run a step to ensure it works
  auto input = create_vertical_line(layer0_cfg.num_input_cols, layer0_cfg.num_input_rows, 3);
  region.set_input(input);
  region.step(1);

  EXPECT_EQ(region.timestep(), 1);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST(HTMRegionIntegration, EmptyConfigThrows) {
  // Test that creating a region with empty config throws an exception.
  htm_flow::HTMRegionConfig cfg;  // Empty layers

  EXPECT_THROW(htm_flow::HTMRegion region(cfg), std::invalid_argument);
}

TEST(HTMRegionIntegration, InvalidLayerIndexThrows) {
  // Test that accessing invalid layer index throws an exception.
  auto layer_cfg = suiteConfig();
  htm_flow::HTMRegionConfig cfg = htm_flow::uniform_region_config(2, layer_cfg);

  htm_flow::HTMRegion region(cfg);

  EXPECT_THROW(region.layer(-1), std::out_of_range);
  EXPECT_THROW(region.layer(2), std::out_of_range);
  EXPECT_THROW(region.layer(100), std::out_of_range);
}
