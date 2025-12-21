#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <htm_flow/inhibition.hpp>
#include <htm_flow/overlap.hpp>
#include <htm_flow/spatiallearn.hpp>

namespace spatial_pooling_test_utils {

// -----------------------------------------------------------------------------
// Spatial-pooler-only end-to-end harness:
//   input -> overlap -> inhibition -> spatial learning (proximal permanence update)
//
// This intentionally stops BEFORE any temporal memory / sequence learning stages.
//
// Why this exists:
// - The legacy Python spatialPooling suites compare *active columns* output SDRs.
// - In htm_flow, "spatial pooler" maps cleanly to:
//     overlap::OverlapCalculator + inhibition::InhibitionCalculator + SpatialLearnCalculator
// - We want deterministic integration tests that exercise the real pipeline wiring
//   without pulling in temporal modules.
// -----------------------------------------------------------------------------
class SpatialPoolerHarness {
public:
  struct Config {
    // Input grid shape (rows, cols)
    int input_rows = 60;
    int input_cols = 30;

    // Column grid shape (rows, cols)
    int col_rows = 30;
    int col_cols = 10;

    // Potential pool size (must be <= input dims when wrap_input=false)
    int pot_h = 4;
    int pot_w = 4;

    bool center_pot_synapses = true;
    bool wrap_input = false;

    // Inhibition neighborhood
    int inhib_w = 3;
    int inhib_h = 3;
    int desired_local_activity = 2;

    // Overlap thresholds
    float connected_perm = 0.3f;
    int min_overlap = 3;

    // Spatial learning
    float spatial_perm_inc = 0.10f;
    float spatial_perm_dec = 0.02f;
    float active_col_perm_dec = 0.02f;

    // Determinism
    std::uint32_t rng_seed = 123u;
  };

  explicit SpatialPoolerHarness(const Config& cfg)
      : cfg_(cfg),
        num_columns_(cfg_.col_rows * cfg_.col_cols),
        num_pot_syn_(cfg_.pot_h * cfg_.pot_w),
        overlap_calc_(cfg_.pot_w,
                     cfg_.pot_h,
                     /*columns_width=*/cfg_.col_cols,
                     /*columns_height=*/cfg_.col_rows,
                     /*input_width=*/cfg_.input_cols,
                     /*input_height=*/cfg_.input_rows,
                     cfg_.center_pot_synapses,
                     cfg_.connected_perm,
                     cfg_.min_overlap,
                     cfg_.wrap_input),
        // strictLocalActivity=true forces serial inhibition (deterministic).
        inhibition_calc_(/*width=*/cfg_.col_cols,
                         /*height=*/cfg_.col_rows,
                         /*potentialInhibWidth=*/cfg_.inhib_w,
                         /*potentialInhibHeight=*/cfg_.inhib_h,
                         /*desiredLocalActivity=*/cfg_.desired_local_activity,
                         /*minOverlap=*/cfg_.min_overlap,
                         /*centerInhib=*/true,
                         /*wrapMode=*/cfg_.wrap_input,
                         /*strictLocalActivity=*/true,
                         /*debug=*/false,
                         /*useTieBreaker=*/true),
        spatial_learn_calc_(num_columns_,
                            num_pot_syn_,
                            cfg_.spatial_perm_inc,
                            cfg_.spatial_perm_dec,
                            cfg_.active_col_perm_dec),
        rng_(cfg_.rng_seed) {
    EXPECT_GT(cfg_.input_rows, 0);
    EXPECT_GT(cfg_.input_cols, 0);
    EXPECT_GT(cfg_.col_rows, 0);
    EXPECT_GT(cfg_.col_cols, 0);
    EXPECT_GT(cfg_.pot_h, 0);
    EXPECT_GT(cfg_.pot_w, 0);

    col_syn_perm_.assign(static_cast<std::size_t>(num_columns_ * num_pot_syn_), 0.0f);
    std::uniform_real_distribution<float> up(0.0f, 1.0f);
    for (auto& p : col_syn_perm_) p = up(rng_);

    col_syn_perm_shape_ = {num_columns_, num_pot_syn_};
    input_shape_ = {cfg_.input_rows, cfg_.input_cols};
    col_grid_shape_ = {cfg_.col_rows, cfg_.col_cols};
  }

  void step(const std::vector<int>& input_grid01) {
    ASSERT_EQ(static_cast<int>(input_grid01.size()), cfg_.input_rows * cfg_.input_cols)
        << "Input size mismatch: expected input_rows*input_cols";

    overlap_calc_.calculate_overlap(col_syn_perm_, col_syn_perm_shape_, input_grid01, input_shape_);
    const std::vector<float> col_overlap_scores = overlap_calc_.get_col_overlaps();

    inhibition_calc_.calculate_inhibition(col_overlap_scores, col_grid_shape_, col_overlap_scores, col_grid_shape_);
    const std::vector<int>& active_col_indices = inhibition_calc_.get_active_column_indices();

    spatial_learn_calc_.calculate_spatiallearn_1d_active_indices(col_syn_perm_,
                                                                 col_syn_perm_shape_,
                                                                 overlap_calc_.get_col_pot_inputs(),
                                                                 overlap_calc_.get_col_pot_inputs_shape(),
                                                                 active_col_indices);
  }

  // Dense 0/1 vector of length num_columns (col_rows*col_cols) representing active columns.
  std::vector<uint8_t> activeColumns01() {
    const std::vector<int> active_cols = inhibition_calc_.get_active_columns();
    std::vector<uint8_t> out(active_cols.size(), 0);
    for (std::size_t i = 0; i < active_cols.size(); ++i) out[i] = (active_cols[i] != 0) ? 1 : 0;
    return out;
  }

  std::mt19937& rng() { return rng_; }
  const Config& cfg() const { return cfg_; }

private:
  Config cfg_;
  int num_columns_;
  int num_pot_syn_;

  overlap::OverlapCalculator overlap_calc_;
  inhibition::InhibitionCalculator inhibition_calc_;
  spatiallearn::SpatialLearnCalculator spatial_learn_calc_;

  std::vector<float> col_syn_perm_;
  std::pair<int, int> col_syn_perm_shape_;
  std::pair<int, int> input_shape_;
  std::pair<int, int> col_grid_shape_;

  std::mt19937 rng_;
};

} // namespace spatial_pooling_test_utils


