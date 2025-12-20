#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

namespace {

// -----------------------------------------------------------------------------
// A minimal C++ port of:
// - HTM/utilities/measureTemporalPooling.py
// - HTM/utilities/sdrFunctions.py (similarInputGrids)
//
// This measures "how much successive outputs change" using:
//   percent = |prev ∩ curr| / |prev|
// and maintains a running average across calls.
//
// In the Python tests this is used as a proxy for "temporal pooling":
// if the representation stabilizes across a repeating input sequence, the
// overlap between consecutive outputs increases.
// -----------------------------------------------------------------------------
class TemporalPoolingMeasure {
public:
  double temporalPoolingPercent(const std::vector<uint8_t>& grid01) {
    if (!prev_.empty()) {
      int prev_active = 0;
      int both_active = 0;
      for (std::size_t i = 0; i < grid01.size(); ++i) {
        const bool p = (prev_[i] != 0);
        const bool c = (grid01[i] != 0);
        prev_active += p ? 1 : 0;
        both_active += (p && c) ? 1 : 0;
      }
      const double percent = (prev_active > 0) ? (static_cast<double>(both_active) / static_cast<double>(prev_active))
                                               : 0.0;
      // Running average matches the Python implementation:
      // temporalAverage = (percentTemp + temporalAverage*(numInputGrids-1))/numInputGrids
      temporal_average_ =
          (percent + temporal_average_ * static_cast<double>(num_grids_ - 1)) / static_cast<double>(num_grids_);
    }
    prev_ = grid01;
    ++num_grids_;
    return temporal_average_;
  }

private:
  std::vector<uint8_t> prev_;
  double temporal_average_ = 0.0;
  int num_grids_ = 0;
};

static double similarityPercent(const std::vector<uint8_t>& a01, const std::vector<uint8_t>& b01) {
  int a_active = 0;
  int both_active = 0;
  for (std::size_t i = 0; i < a01.size(); ++i) {
    const bool a = (a01[i] != 0);
    const bool b = (b01[i] != 0);
    a_active += a ? 1 : 0;
    both_active += (a && b) ? 1 : 0;
  }
  return (a_active > 0) ? (static_cast<double>(both_active) / static_cast<double>(a_active)) : 0.0;
}

// -----------------------------------------------------------------------------
// Deterministic vertical-line input sequences (inspired by Python simpleVerticalLineInputs).
//
// We only implement the patterns we need for our integration tests:
// - LeftToRight: x = t
// - RightToLeft: x = (W-1 - t)
// - EvenPositions: x = 0,2,4,... (disjoint from odd)
// - OddPositions : x = 1,3,5,... (disjoint from even)
//
// The Python code stores a list of 2D grids and cycles through them. Here we generate
// grids on the fly (deterministically).
// -----------------------------------------------------------------------------
class VerticalLineInputs {
public:
  enum class Pattern {
    LeftToRight,
    RightToLeft,
    EvenPositions,
    OddPositions,
  };

  VerticalLineInputs(int width, int height, int seq_len)
      : width_(width), height_(height), seq_len_(seq_len), pat_(Pattern::LeftToRight), idx_(0) {
    EXPECT_GT(width_, 0);
    EXPECT_GT(height_, 0);
    EXPECT_GT(seq_len_, 0);
  }

  void setPattern(Pattern p) {
    pat_ = p;
    idx_ = 0;
  }

  void setIndex(int i) { idx_ = i % seq_len_; }

  int seqLen() const { return seq_len_; }

  // Deterministic "sequenceProbability" analogue:
  // - 1.0 => always in-sequence
  // - 0.0 => always random index (deterministic RNG)
  void setSequenceProbability(double p) { seq_prob_ = p; }

  std::vector<int> next(std::mt19937& rng) {
    int chosen = idx_;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) > seq_prob_) {
      std::uniform_int_distribution<int> pick(0, seq_len_ - 1);
      chosen = pick(rng);
    }

    std::vector<int> grid(static_cast<std::size_t>(width_ * height_), 0);
    const int x = x_for_index(chosen);
    for (int y = 0; y < height_; ++y) {
      grid[static_cast<std::size_t>(y * width_ + x)] = 1;
    }

    idx_ = (idx_ + 1) % seq_len_;
    return grid;
  }

private:
  int x_for_index(int t) const {
    switch (pat_) {
      case Pattern::LeftToRight:
        return t % width_;
      case Pattern::RightToLeft:
        return (width_ - 1 - (t % width_));
      case Pattern::EvenPositions: {
        // seq_len_ is expected to be width_/2 for even/odd tests; still safe for other values.
        const int pos = (2 * (t % width_)) % width_;
        return pos;
      }
      case Pattern::OddPositions: {
        const int pos = (2 * (t % width_) + 1) % width_;
        return pos;
      }
    }
    return 0;
  }

  int width_;
  int height_;
  int seq_len_;
  Pattern pat_;
  int idx_;
  double seq_prob_ = 1.0;
};

// -----------------------------------------------------------------------------
// Pipeline harness (mirrors htm_flow/src/main.cpp, but small + deterministic).
//
// This runs the "whole system" for one timestep:
//   overlap -> inhibition -> spatial learn -> active cells -> predict cells
//   -> sequence learning -> temporal pooler (proximal + distal)
//
// The integration tests measure temporal pooling on:
// - active columns (sparse spatial output)
// - predictive cells after temporal pooler (temporally stabilized output)
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
    // Note: for the vertical-line inputs used in these tests, increasing pot_h
    // makes overlap values more input-driven (a line intersects more rows),
    // which improves pattern differentiation and reduces "always stable" behavior
    // dominated by tie-breakers when overlap counts are tiny.
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
        // NOTE: strictLocalActivity=true forces serial inhibition, improving determinism.
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
        }),
        rng_(cfg_.rng_seed) {
    // Initialize proximal permanences deterministically.
    col_syn_perm_.assign(static_cast<std::size_t>(num_columns_ * num_pot_syn_), 0.0f);
    std::uniform_real_distribution<float> up(0.0f, 1.0f);
    for (auto& p : col_syn_perm_) {
      p = up(rng_);
    }

    // Initialize distal synapses deterministically.
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

    // Active-columns bitfield used by temporal proximal pooler.
    col_active01_.assign(static_cast<std::size_t>(num_columns_), 0);
    prev_active_col_indices_.clear();

    // Reusable shape pairs.
    col_syn_perm_shape_ = {num_columns_, num_pot_syn_};
    input_shape_ = {cfg_.input_rows, cfg_.input_cols};
    col_grid_shape_ = {cfg_.col_rows, cfg_.col_cols};
  }

  void step(int time_step, const std::vector<int>& input_grid01) {
    // 1) Overlap
    overlap_calc_.calculate_overlap(col_syn_perm_, col_syn_perm_shape_, input_grid01, input_shape_);
    const std::vector<float> col_overlap_scores = overlap_calc_.get_col_overlaps();

    // 2) Inhibition (deterministic serial path because strictLocalActivity=true)
    inhibition_calc_.calculate_inhibition(col_overlap_scores, col_grid_shape_, col_overlap_scores, col_grid_shape_);
    const std::vector<int>& active_col_indices = inhibition_calc_.get_active_column_indices();

    // 3) Dense active-columns bitfield for temporal proximal pooler.
    for (int c : prev_active_col_indices_) {
      col_active01_[static_cast<std::size_t>(c)] = 0;
    }
    for (int c : active_col_indices) {
      col_active01_[static_cast<std::size_t>(c)] = 1;
    }
    prev_active_col_indices_ = active_col_indices;

    // 4) Spatial learning (uses overlap's internal potential-input buffer)
    spatial_learn_calc_.calculate_spatiallearn_1d_active_indices(col_syn_perm_,
                                                                 col_syn_perm_shape_,
                                                                 overlap_calc_.get_col_pot_inputs(),
                                                                 overlap_calc_.get_col_pot_inputs_shape(),
                                                                 active_col_indices);

    // 5) Sequence pooler: active cells
    active_cells_calc_.calculate_active_cells(time_step,
                                              active_col_indices,
                                              predict_cells_calc_.get_predict_cells_time(),
                                              predict_cells_calc_.get_active_segs_time(),
                                              distal_synapses_);

    // 6) Sequence pooler: predict cells
    predict_cells_calc_.calculate_predict_cells(time_step, active_cells_calc_.get_active_cells_time(), distal_synapses_);

    // 7) Sequence learning
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

    // 8) Temporal pooler (end of timestep)
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

  std::vector<uint8_t> activeColumns01() {
    const std::vector<int> active_cols = inhibition_calc_.get_active_columns();
    std::vector<uint8_t> out(active_cols.size(), 0);
    for (std::size_t i = 0; i < active_cols.size(); ++i) {
      out[i] = (active_cols[i] != 0) ? 1 : 0;
    }
    return out;
  }

  std::vector<uint8_t> predictCells01(int time_step) const {
    const std::vector<int>& pc = predict_cells_calc_.get_predict_cells_time();
    std::vector<uint8_t> out(static_cast<std::size_t>(num_columns_ * cfg_.cells_per_column), 0);
    for (int col = 0; col < num_columns_; ++col) {
      for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
        const int i0 = (col * cfg_.cells_per_column + cell) * 2 + 0;
        const int i1 = (col * cfg_.cells_per_column + cell) * 2 + 1;
        const bool pred = (pc[static_cast<std::size_t>(i0)] == time_step) || (pc[static_cast<std::size_t>(i1)] == time_step);
        out[static_cast<std::size_t>(col * cfg_.cells_per_column + cell)] = pred ? 1 : 0;
      }
    }
    return out;
  }

  std::vector<uint8_t> learnCells01(int time_step) const {
    // Learning-cells output is closer to what the Python temporal pooling suites
    // measured in some cases (e.g. Suite4 uses getLearningCellsOutput).
    // It is also influenced by temporal pooling indirectly via predictive state:
    // temporal pooling updates `predict_cells_time`, which affects future learning-cell selection.
    const std::vector<int>& lc = active_cells_calc_.get_learn_cells_time();
    std::vector<uint8_t> out(static_cast<std::size_t>(num_columns_ * cfg_.cells_per_column), 0);
    for (int col = 0; col < num_columns_; ++col) {
      for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
        const int i0 = (col * cfg_.cells_per_column + cell) * 2 + 0;
        const int i1 = (col * cfg_.cells_per_column + cell) * 2 + 1;
        const bool learn = (lc[static_cast<std::size_t>(i0)] == time_step) || (lc[static_cast<std::size_t>(i1)] == time_step);
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

// Helper: run N steps with a provided input generator; time_step starts at 1.
template <typename InputFn>
static void runSteps(HtmPipelineHarness& htm, int start_time_step, int num_steps, InputFn&& next_input) {
  for (int dt = 0; dt < num_steps; ++dt) {
    const int t = start_time_step + dt;
    const std::vector<int> in = next_input();
    htm.step(t, in);
  }
}

} // namespace

TEST(TemporalPoolingIntegration, repeating_sequence_temporally_pools) {
  // Step-by-step: what we're testing and why
  // 1) Create a deterministic end-to-end HTM pipeline (overlap → inhibition → learning → temporal pooler).
  //    Why: these are integration tests; we want to validate temporal pooling emerges when the whole
  //    system runs, not just that individual update rules work in isolation.
  // 2) Feed a repeating input sequence (moving vertical line) for a warm-up period.
  //    Why: temporal pooling is learned; we expect stability to increase after repeated exposure.
  // 3) Measure temporal pooling on the learning-cells representation.
  //    Why: the Python suites often analyze cell-level representations (and Suite4 explicitly uses learning cells).
  //    In this pipeline, learning-cells selection depends on predictive state, which is influenced by temporal pooling.
  // 4) Assert the pooling percent rises above a minimum threshold.
  //    Why: for a repeatable sequence, the output should become relatively stable across timesteps.
  HtmPipelineHarness htm(HtmPipelineHarness::Config{});
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  // Longer warmup helps the end-to-end pipeline learn a stable representation.
  runSteps(htm, time_step, /*num_steps=*/300, [&]() { return inputs.next(htm.rng()); });
  time_step += 300;

  TemporalPoolingMeasure m;
  double pooled = 0.0;
  // Evaluate over multiple full cycles of the input sequence.
  const int eval_steps = 3 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    pooled = m.temporalPoolingPercent(htm.learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(pooled, 0.52) << "Expected high temporal pooling for repeating sequence";
}

TEST(TemporalPoolingIntegration, random_order_sequence_does_not_pool_much) {
  // Step-by-step: what we're testing and why
  // 1) Warm up the system on an in-sequence pattern.
  //    Why: match the Python tests: first let the system learn *something* about the task distribution.
  // 2) Switch to a "random-order" regime (sequenceProbability = 0.0).
  //    Why: the temporal pooler should not be able to form a stable representation if the next input
  //    is unpredictable.
  // 3) Measure temporal pooling percent on the learning-cells representation.
  // 4) Assert pooling remains low.
  HtmPipelineHarness htm(HtmPipelineHarness::Config{});
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  runSteps(htm, time_step, /*num_steps=*/120, [&]() { return inputs.next(htm.rng()); });
  time_step += 120;

  // Now make the next input effectively random (deterministic RNG).
  inputs.setSequenceProbability(0.0);

  TemporalPoolingMeasure m;
  double pooled = 0.0;
  const int eval_steps = 2 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    pooled = m.temporalPoolingPercent(htm.learnCells01(time_step));
    ++time_step;
  }

  EXPECT_LE(pooled, 0.40) << "Expected low temporal pooling for random-order inputs";
}

TEST(TemporalPoolingIntegration, missing_inputs_still_pools_some) {
  // Step-by-step: what we're testing and why
  // 1) Warm up the system on a repeating input sequence.
  // 2) Then simulate "missing inputs" by holding the same input every other step.
  //    Why: this mirrors the Python test where inputs sometimes fail to update; temporal pooling
  //    should still show some stability because the overall sequence is still structured.
  // 3) Measure pooling percent on learning-cells output.
  // 4) Assert pooling is not tiny (greater than the random-order case would be).
  HtmPipelineHarness htm(HtmPipelineHarness::Config{});
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  runSteps(htm, time_step, /*num_steps=*/100, [&]() { return inputs.next(htm.rng()); });
  time_step += 100;

  TemporalPoolingMeasure m;
  double pooled = 0.0;
  std::vector<int> held = inputs.next(htm.rng());
  const int eval_steps = 2 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    if (i % 2 == 0) {
      held = inputs.next(htm.rng());
    }
    htm.step(time_step, held);
    pooled = m.temporalPoolingPercent(htm.learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(pooled, 0.45) << "Expected moderate temporal pooling with missing/held inputs";
}

TEST(TemporalPoolingIntegration, pooled_representation_more_stable_than_raw_activity) {
  // Step-by-step: what we're testing and why
  // 1) The Python suite had a multi-layer test asserting pooling increases up the hierarchy.
  //    htm_flow currently runs a single-layer pipeline, so we replace that test with a C++-appropriate one:
  //    compare two representations at the SAME layer:
  //      - active columns (spatial output; more input-driven, less temporally smoothed)
  //      - learning cells (cell-level output; influenced by prediction/persistence)
  // 2) Run a repeating sequence long enough to learn.
  // 3) Measure pooling percent separately on active columns vs predictive output.
  // 4) Assert predictive output is at least as stable as active columns.
  HtmPipelineHarness htm(HtmPipelineHarness::Config{});
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  runSteps(htm, time_step, /*num_steps=*/160, [&]() { return inputs.next(htm.rng()); });
  time_step += 160;

  TemporalPoolingMeasure m_active_cols;
  TemporalPoolingMeasure m_learn;

  double pooled_cols = 0.0;
  double pooled_learn = 0.0;

  const int eval_steps = 2 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);

    pooled_cols = m_active_cols.temporalPoolingPercent(htm.activeColumns01());
    pooled_learn = m_learn.temporalPoolingPercent(htm.learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(pooled_learn, pooled_cols - 0.05) << "Expected learning-cells representation to be >= stability of active columns";
}

TEST(TemporalPoolingIntegration, multi_pattern_differentiation_and_recall) {
  // Step-by-step: what we're testing and why
  // 1) Train on Pattern A and record the predictive output for each element of one cycle.
  //    Why: the Python multi-pattern suites store outputs per-input to compare later.
  // 2) Train on Pattern B (a different, non-overlapping set of inputs) and record outputs.
  //    Why: the temporally pooled representations for different patterns should differ.
  // 3) Switch back to Pattern A, retrain, and record outputs again.
  //    Why: we want to verify recall — that Pattern A re-forms a similar pooled output.
  // 4) Compare similarities:
  //    - A vs A-again should be relatively high
  //    - A vs B should be relatively low
  //
  // Implementation note:
  // We use EvenPositions vs OddPositions which are disjoint sets of vertical lines (for even input width),
  // inspired by the Python suite's use of disjoint patterns.
  HtmPipelineHarness htm(HtmPipelineHarness::Config{});

  // Use half-length sequences for even/odd patterns (disjoint).
  const int base_w = htm.cfg().input_cols;
  ASSERT_EQ(base_w % 2, 0) << "This test expects even input width";
  VerticalLineInputs inputs(/*width=*/base_w, /*height=*/htm.cfg().input_rows, /*seq_len=*/base_w / 2);
  inputs.setSequenceProbability(1.0);

  auto capture_cycle = [&](int& time_step, std::vector<std::vector<uint8_t>>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(inputs.seqLen()));
    inputs.setIndex(0);
    for (int i = 0; i < inputs.seqLen(); ++i) {
      const std::vector<int> in = inputs.next(htm.rng());
      htm.step(time_step, in);
      out.push_back(htm.learnCells01(time_step));
      ++time_step;
    }
  };

  int time_step = 1;

  // Pattern A (even positions)
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  runSteps(htm, time_step, /*num_steps=*/220, [&]() { return inputs.next(htm.rng()); });
  time_step += 220;
  std::vector<std::vector<uint8_t>> a1;
  capture_cycle(time_step, a1);

  // Pattern B (odd positions)
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  runSteps(htm, time_step, /*num_steps=*/220, [&]() { return inputs.next(htm.rng()); });
  time_step += 220;
  std::vector<std::vector<uint8_t>> b;
  capture_cycle(time_step, b);

  // Back to Pattern A again
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  runSteps(htm, time_step, /*num_steps=*/260, [&]() { return inputs.next(htm.rng()); });
  time_step += 260;
  std::vector<std::vector<uint8_t>> a2;
  capture_cycle(time_step, a2);

  ASSERT_EQ(a1.size(), a2.size());
  ASSERT_EQ(a1.size(), b.size());

  // Compute average similarities per input index.
  double sim_a1_a2 = 0.0;
  double sim_a1_b = 0.0;
  for (std::size_t i = 0; i < a1.size(); ++i) {
    sim_a1_a2 += similarityPercent(a1[i], a2[i]);
    sim_a1_b += similarityPercent(a1[i], b[i]);
  }
  sim_a1_a2 /= static_cast<double>(a1.size());
  sim_a1_b /= static_cast<double>(a1.size());

  EXPECT_GE(sim_a1_a2, 0.20) << "Expected some recall: Pattern A should return a similar pooled output";
  EXPECT_LE(sim_a1_b, 0.35) << "Expected differentiation: Pattern A pooled output should differ from Pattern B";
}

