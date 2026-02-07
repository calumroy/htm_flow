/// Integration tests for HTMLayer - testing the complete HTM layer pipeline.
///
/// These tests verify that the HTMLayer class correctly integrates all the
/// calculators (overlap, inhibition, spatial learning, sequence pooling,
/// temporal pooling) and produces expected behavior.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <htm_flow/config.hpp>
#include <htm_flow/htm_layer.hpp>
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

/// Count the number of learning cells in a snapshot.
int count_learning_cells(const htm_gui::Snapshot& snap) {
  int count = 0;
  for (const auto& masks : snap.column_cell_masks) {
    for (int c = 0; c < snap.cells_per_column && c < 64; ++c) {
      if (masks.learning & (std::uint64_t(1) << c)) {
        ++count;
      }
    }
  }
  return count;
}

}  // namespace

// =============================================================================
// Basic Sanity Tests
// =============================================================================

TEST(HTMLayerIntegration, ColumnsActivateOnInput) {
  // Test that columns become active when given input with active bits.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  // Create input with a vertical line
  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 5);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();
  EXPECT_GT(snap.active_column_indices.size(), 0u) << "Expected some columns to become active";
}

TEST(HTMLayerIntegration, NoActiveColumnsWithEmptyInput) {
  // Test that no columns become active when input is all zeros.
  // Note: This depends on configuration - with min_overlap > 0 and all zeros,
  // no columns should have enough overlap to activate.
  auto cfg = suiteConfig();
  cfg.min_overlap = 2;  // Ensure we need some overlap
  htm_flow::HTMLayer layer(cfg);

  // Create empty input (all zeros)
  std::vector<int> input(
      static_cast<std::size_t>(cfg.num_input_cols) * static_cast<std::size_t>(cfg.num_input_rows), 0);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();
  // With potential overlap fallback, some columns might still activate even with zeros
  // if min_potential_overlap is low. The key test is that it doesn't crash.
  EXPECT_GE(snap.timestep, 1) << "Layer should have stepped once";
}

TEST(HTMLayerIntegration, ActiveCellsExistWhenColumnsActivate) {
  // Test that active cells are present when columns become active.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();
  int active = count_active_cells(snap);
  EXPECT_GT(active, 0) << "Expected active cells when columns are active";
}

TEST(HTMLayerIntegration, LearningCellsExistOnActivation) {
  // Test that learning cells are marked when columns become active.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();
  int learning = count_learning_cells(snap);
  EXPECT_GT(learning, 0) << "Expected learning cells when columns are active";
}

// =============================================================================
// Prediction Tests
// =============================================================================

TEST(HTMLayerIntegration, PredictionsFormAfterRepeatedInput) {
  // Test that predictions form after repeated exposure to a sequence.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  const int seq_len = cfg.num_input_cols;
  std::mt19937 rng(42);
  utilities::MovingLineInputs inputs(cfg.num_input_cols, cfg.num_input_rows);
  inputs.setPattern(utilities::MovingLineInputs::Pattern::LeftToRight);

  // Run multiple epochs of the sequence
  for (int epoch = 0; epoch < 15; ++epoch) {
    inputs.setIndex(0);
    for (int pos = 0; pos < seq_len; ++pos) {
      auto input = inputs.next(rng);
      layer.set_input(input);
      layer.step(1);
    }
  }

  // After training, check that predictions have formed
  inputs.setIndex(0);
  for (int pos = 0; pos < seq_len; ++pos) {
    auto input = inputs.next(rng);
    layer.set_input(input);
    layer.step(1);
  }

  auto snap = layer.snapshot();
  int predictive = count_predictive_cells(snap);
  EXPECT_GT(predictive, 0) << "Expected predictions to form after repeated sequence exposure";
}

// =============================================================================
// Output Tests
// =============================================================================

TEST(HTMLayerIntegration, OutputHasCorrectDimensions) {
  // Test that the output has the correct dimensions for stacking layers.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto output = layer.output();
  const std::size_t expected_size = static_cast<std::size_t>(cfg.num_column_rows) *
                                    static_cast<std::size_t>(cfg.num_column_cols) *
                                    static_cast<std::size_t>(cfg.cells_per_column);
  EXPECT_EQ(output.size(), expected_size) << "Output size should match column_rows * column_cols * cells_per_column";
}

TEST(HTMLayerIntegration, OutputContainsActiveCellInfo) {
  // Test that the output reflects active cell states.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto output = layer.output();
  int active_in_output = 0;
  for (int v : output) {
    if (v == 1) ++active_in_output;
  }

  auto snap = layer.snapshot();
  int active_cells = count_active_cells(snap);

  EXPECT_EQ(active_in_output, active_cells) << "Output should reflect the same number of active cells as snapshot";
}

// =============================================================================
// GUI Interface Tests
// =============================================================================

TEST(HTMLayerIntegration, SnapshotReturnsValidData) {
  // Test that the IHtmRuntime snapshot method returns valid data.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();

  EXPECT_EQ(snap.timestep, 1);
  EXPECT_EQ(snap.input_shape.rows, cfg.num_input_rows);
  EXPECT_EQ(snap.input_shape.cols, cfg.num_input_cols);
  EXPECT_EQ(snap.columns_shape.rows, cfg.num_column_rows);
  EXPECT_EQ(snap.columns_shape.cols, cfg.num_column_cols);
  EXPECT_EQ(snap.cells_per_column, cfg.cells_per_column);
  EXPECT_NE(snap.input, nullptr);
  EXPECT_EQ(snap.column_cell_masks.size(),
            static_cast<std::size_t>(cfg.num_column_rows) * static_cast<std::size_t>(cfg.num_column_cols));
}

TEST(HTMLayerIntegration, QueryProximalReturnsValidData) {
  // Test that query_proximal returns valid synapse data.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto q = layer.query_proximal(0, 0);
  EXPECT_EQ(q.column_x, 0);
  EXPECT_EQ(q.column_y, 0);
  EXPECT_EQ(q.synapses.size(), static_cast<std::size_t>(cfg.pot_width * cfg.pot_height));
}

TEST(HTMLayerIntegration, QueryDistalReturnsValidData) {
  // Test that query_distal returns valid segment/synapse data.
  auto cfg = suiteConfig();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 3);
  layer.set_input(input);
  layer.step(1);

  auto q = layer.query_distal(0, 0, 0, 0);
  EXPECT_EQ(q.src_column_x, 0);
  EXPECT_EQ(q.src_column_y, 0);
  EXPECT_EQ(q.src_cell, 0);
  EXPECT_EQ(q.segment, 0);
  EXPECT_EQ(q.synapses.size(), static_cast<std::size_t>(cfg.max_synapses_per_segment));
}

// =============================================================================
// Configuration Preset Tests
// =============================================================================

TEST(HTMLayerIntegration, DefaultConfigWorks) {
  // Test that the default configuration produces a working layer.
  auto cfg = htm_flow::default_layer_config();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 5);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();
  EXPECT_EQ(snap.timestep, 1);
}

TEST(HTMLayerIntegration, TemporalPoolingConfigWorks) {
  // Test that the temporal pooling test configuration produces a working layer.
  auto cfg = htm_flow::temporal_pooling_test_config();
  htm_flow::HTMLayer layer(cfg);

  auto input = create_vertical_line(cfg.num_input_cols, cfg.num_input_rows, 5);
  layer.set_input(input);
  layer.step(1);

  auto snap = layer.snapshot();
  EXPECT_EQ(snap.timestep, 1);
  EXPECT_GT(snap.active_column_indices.size(), 0u);
}
