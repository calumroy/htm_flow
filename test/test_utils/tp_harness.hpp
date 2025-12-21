#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <htm_flow/inhibition.hpp>
#include <htm_flow/overlap.hpp>
#include <htm_flow/sequence_pooler/active_cells.hpp>
#include <htm_flow/sequence_pooler/predict_cells.hpp>
#include <htm_flow/sequence_pooler/sequence_learning.hpp>
#include <htm_flow/sequence_pooler/sequence_types.hpp>
#include <htm_flow/spatiallearn.hpp>
#include <htm_flow/temporal_pooler/temporal_pooler.hpp>

namespace temporal_pooling_test_utils {

// -----------------------------------------------------------------------------
// End-to-end pipeline harnesses (mirror htm_flow/src/main.cpp, but deterministic)
// -----------------------------------------------------------------------------

class HtmPipelineHarness {
public:
  struct Config {
    // Input grid shape (rows, cols)
    int input_rows = 24;
    int input_cols = 16;

    // Column grid shape (rows, cols)
    int col_rows = 12;
    int col_cols = 12;

    // Potential pool size
    int pot_h = 12;
    int pot_w = 4;

    bool center_pot_synapses = false;
    bool wrap_input = true;

    // Inhibition neighborhood
    int inhib_w = 7;
    int inhib_h = 7;
    int desired_local_activity = 6;

    // Overlap thresholds
    float connected_perm = 0.3f;
    int min_overlap = 2;

    // Spatial learning
    float spatial_perm_inc = 0.05f;
    float spatial_perm_dec = 0.02f;
    float active_col_perm_dec = 0.01f;

    // Sequence pooler / TM-ish parameters
    int cells_per_column = 4;
    int max_segments_per_cell = 2;
    int max_synapses_per_segment = 10;
    int min_num_syn_threshold = 1;
    int min_score_threshold = 1;
    float new_syn_permanence = 0.3f;
    float connect_permanence = 0.2f;
    int activation_threshold = 3;

    // Sequence learning
    float seq_perm_inc = 0.05f;
    float seq_perm_dec = 0.02f;

    // Temporal pooler
    float temp_spatial_perm_inc = 0.05f;
    float temp_seq_perm_inc = 0.05f;
    int temp_delay_length = 4;
    bool temp_enable_persistence = true;

    // Determinism
    std::uint32_t rng_seed = 123u;
  };

  explicit HtmPipelineHarness(const Config& cfg)
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
        active_cells_calc_(sequence_pooler::ActiveCellsCalculator::Config{
            num_columns_,
            cfg_.cells_per_column,
            cfg_.max_segments_per_cell,
            cfg_.max_synapses_per_segment,
            cfg_.min_num_syn_threshold,
            cfg_.min_score_threshold,
            cfg_.new_syn_permanence,
            cfg_.connect_permanence,
        }),
        predict_cells_calc_(sequence_pooler::PredictCellsCalculator::Config{
            num_columns_,
            cfg_.cells_per_column,
            cfg_.max_segments_per_cell,
            cfg_.max_synapses_per_segment,
            cfg_.connect_permanence,
            cfg_.activation_threshold,
        }),
        seq_learn_calc_(sequence_pooler::SequenceLearningCalculator::Config{
            num_columns_,
            cfg_.cells_per_column,
            cfg_.max_segments_per_cell,
            cfg_.max_synapses_per_segment,
            cfg_.connect_permanence,
            cfg_.seq_perm_inc,
            cfg_.seq_perm_dec,
        }),
        temporal_pool_calc_(temporal_pooler::TemporalPoolerCalculator::Config{
            num_columns_,
            cfg_.cells_per_column,
            cfg_.max_segments_per_cell,
            cfg_.max_synapses_per_segment,
            num_pot_syn_,
            cfg_.temp_spatial_perm_inc,
            cfg_.temp_seq_perm_inc,
            cfg_.min_num_syn_threshold,
            cfg_.new_syn_permanence,
            cfg_.connect_permanence,
            cfg_.temp_delay_length,
            cfg_.temp_enable_persistence,
        }),
        rng_(cfg_.rng_seed) {
    col_syn_perm_.assign(static_cast<std::size_t>(num_columns_ * num_pot_syn_), 0.0f);
    std::uniform_real_distribution<float> up(0.0f, 1.0f);
    for (auto& p : col_syn_perm_) {
      p = up(rng_);
    }

    const std::size_t distal_count = static_cast<std::size_t>(num_columns_) *
                                     static_cast<std::size_t>(cfg_.cells_per_column) *
                                     static_cast<std::size_t>(cfg_.max_segments_per_cell) *
                                     static_cast<std::size_t>(cfg_.max_synapses_per_segment);
    distal_synapses_.assign(distal_count, sequence_pooler::DistalSynapse{0, 0, 0.0f});

    std::uniform_int_distribution<int> col_dis(0, num_columns_ - 1);
    std::uniform_int_distribution<int> cell_dis(0, cfg_.cells_per_column - 1);
    for (std::size_t i = 0; i < distal_count; ++i) {
      distal_synapses_[i] = sequence_pooler::DistalSynapse{col_dis(rng_), cell_dis(rng_), up(rng_)};
    }

    col_active01_.assign(static_cast<std::size_t>(num_columns_), 0);
    prev_active_col_indices_.clear();

    col_syn_perm_shape_ = {num_columns_, num_pot_syn_};
    input_shape_ = {cfg_.input_rows, cfg_.input_cols};
    col_grid_shape_ = {cfg_.col_rows, cfg_.col_cols};
  }

  void step(int time_step, const std::vector<int>& input_grid01) {
    overlap_calc_.calculate_overlap(col_syn_perm_, col_syn_perm_shape_, input_grid01, input_shape_);
    const std::vector<float> col_overlap_scores = overlap_calc_.get_col_overlaps();

    inhibition_calc_.calculate_inhibition(col_overlap_scores, col_grid_shape_, col_overlap_scores, col_grid_shape_);
    const std::vector<int>& active_col_indices = inhibition_calc_.get_active_column_indices();

    for (int c : prev_active_col_indices_) {
      col_active01_[static_cast<std::size_t>(c)] = 0;
    }
    for (int c : active_col_indices) {
      col_active01_[static_cast<std::size_t>(c)] = 1;
    }
    prev_active_col_indices_ = active_col_indices;

    spatial_learn_calc_.calculate_spatiallearn_1d_active_indices(col_syn_perm_,
                                                                 col_syn_perm_shape_,
                                                                 overlap_calc_.get_col_pot_inputs(),
                                                                 overlap_calc_.get_col_pot_inputs_shape(),
                                                                 active_col_indices);

    active_cells_calc_.calculate_active_cells(time_step,
                                              active_col_indices,
                                              predict_cells_calc_.get_predict_cells_time(),
                                              predict_cells_calc_.get_active_segs_time(),
                                              distal_synapses_);

    predict_cells_calc_.calculate_predict_cells(time_step, active_cells_calc_.get_active_cells_time(), distal_synapses_);

    seq_learn_calc_.calculate_sequence_learning(time_step,
                                                active_cells_calc_.get_active_cells_time(),
                                                active_cells_calc_.get_learn_cells_time(),
                                                predict_cells_calc_.get_predict_cells_time(),
                                                distal_synapses_,
                                                active_cells_calc_.get_seg_ind_update_active(),
                                                active_cells_calc_.get_seg_active_syn_active(),
                                                active_cells_calc_.get_seg_ind_new_syn_active(),
                                                active_cells_calc_.get_seg_new_syn_active(),
                                                predict_cells_calc_.get_seg_ind_update_mutable(),
                                                predict_cells_calc_.get_seg_active_syn_mutable());

    temporal_pool_calc_.update_proximal(time_step,
                                        overlap_calc_.get_col_pot_inputs(),
                                        col_active01_,
                                        col_syn_perm_,
                                        active_cells_calc_.get_burst_cols_time());
    temporal_pool_calc_.update_distal(time_step,
                                      active_cells_calc_.get_current_learn_cells_list(),
                                      active_cells_calc_.get_learn_cells_time(),
                                      predict_cells_calc_.get_predict_cells_time_mutable(),
                                      active_cells_calc_.get_active_cells_time(),
                                      predict_cells_calc_.get_active_segs_time(),
                                      distal_synapses_);
  }

  std::vector<int> activeColumnsInt01() {
    const std::vector<int> active_cols = inhibition_calc_.get_active_columns();
    std::vector<int> out(active_cols.size(), 0);
    for (std::size_t i = 0; i < active_cols.size(); ++i) {
      out[i] = (active_cols[i] != 0) ? 1 : 0;
    }
    return out;
  }

  std::vector<uint8_t> learnCells01(int time_step) const {
    const std::vector<int>& lc = active_cells_calc_.get_learn_cells_time();
    std::vector<uint8_t> out(static_cast<std::size_t>(num_columns_ * cfg_.cells_per_column), 0);
    for (int col = 0; col < num_columns_; ++col) {
      for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
        const int i0 = (col * cfg_.cells_per_column + cell) * 2 + 0;
        const int i1 = (col * cfg_.cells_per_column + cell) * 2 + 1;
        const bool learn =
            (lc[static_cast<std::size_t>(i0)] == time_step) || (lc[static_cast<std::size_t>(i1)] == time_step);
        out[static_cast<std::size_t>(col * cfg_.cells_per_column + cell)] = learn ? 1 : 0;
      }
    }
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
  sequence_pooler::ActiveCellsCalculator active_cells_calc_;
  sequence_pooler::PredictCellsCalculator predict_cells_calc_;
  sequence_pooler::SequenceLearningCalculator seq_learn_calc_;
  temporal_pooler::TemporalPoolerCalculator temporal_pool_calc_;

  std::vector<float> col_syn_perm_;
  std::pair<int, int> col_syn_perm_shape_;
  std::pair<int, int> input_shape_;
  std::pair<int, int> col_grid_shape_;
  std::vector<sequence_pooler::DistalSynapse> distal_synapses_;

  std::vector<uint8_t> col_active01_;
  std::vector<int> prev_active_col_indices_;
  std::mt19937 rng_;
};

class TwoLayerHtmHarness {
public:
  struct Config {
    HtmPipelineHarness::Config l0{};
    HtmPipelineHarness::Config l1{};
    std::uint32_t rng_seed = 123u;
  };

  explicit TwoLayerHtmHarness(Config cfg)
      : cfg_(cfg),
        rng_(cfg_.rng_seed),
        l0_(withSeed(cfg_.l0, cfg_.rng_seed)),
        l1_(withSeed(cfg_.l1, cfg_.rng_seed + 1u)) {
    EXPECT_EQ(l1_.cfg().input_rows, l0_.cfg().col_rows);
    EXPECT_EQ(l1_.cfg().input_cols, l0_.cfg().col_cols);
  }

  void step(int time_step, const std::vector<int>& input0) {
    l0_.step(time_step, input0);
    const std::vector<int> l0_out = l0_.activeColumnsInt01();
    l1_.step(time_step, l0_out);
  }

  HtmPipelineHarness& layer0() { return l0_; }
  HtmPipelineHarness& layer1() { return l1_; }
  const HtmPipelineHarness& layer0() const { return l0_; }
  const HtmPipelineHarness& layer1() const { return l1_; }

  std::mt19937& rng() { return rng_; }

private:
  static HtmPipelineHarness::Config withSeed(HtmPipelineHarness::Config c, std::uint32_t seed) {
    c.rng_seed = seed;
    return c;
  }

  Config cfg_;
  std::mt19937 rng_;
  HtmPipelineHarness l0_;
  HtmPipelineHarness l1_;
};

// Run N steps with an injected step function and input generator.
template <typename StepFn, typename InputFn>
inline void runSteps(int start_time_step, int num_steps, StepFn&& step_fn, InputFn&& next_input) {
  for (int dt = 0; dt < num_steps; ++dt) {
    const int t = start_time_step + dt;
    step_fn(t, next_input());
  }
}

} // namespace temporal_pooling_test_utils

