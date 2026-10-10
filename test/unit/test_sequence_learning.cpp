/*
 * Sequence Learning Regression Tests
 * ===================================
 *
 * These tests verify critical invariants in the sequence learning subsystem,
 * particularly around the interaction between predict-cells and sequence-learning.
 */

#include <gtest/gtest.h>

#include <htm_flow/sequence_pooler/predict_cells.hpp>
#include <htm_flow/sequence_pooler/sequence_learning.hpp>
#include <htm_flow/sequence_pooler/sequence_types.hpp>

using sequence_pooler::DistalSynapse;
using sequence_pooler::PredictCellsCalculator;
using sequence_pooler::SequenceLearningCalculator;
using sequence_pooler::idx_distal_synapse;

namespace {

inline int idx_cell_time(int cells_per_col, int col, int cell, int slot) {
  return (col * cells_per_col + cell) * 2 + slot;
}

} // namespace

struct ActiveUpdateState {
  std::vector<int> segment;
  std::vector<int8_t> active_synapses;
  std::vector<int> new_segment;
  std::vector<DistalSynapse> new_synapses;
};

ActiveUpdateState make_active_update_state(int num_cells, int max_synapses) {
  return ActiveUpdateState{
      std::vector<int>(num_cells, -1),
      std::vector<int8_t>(num_cells * max_synapses, 0),
      std::vector<int>(num_cells, -1),
      std::vector<DistalSynapse>(
          num_cells * max_synapses, DistalSynapse{0, 0, -1.0f}),
  };
}

void run_sequence_learning(SequenceLearningCalculator& learning,
                           PredictCellsCalculator& prediction,
                           int time_step,
                           const std::vector<int>& active_cells_time,
                           const std::vector<int>& learn_cells_time,
                           std::vector<DistalSynapse>& distal,
                           ActiveUpdateState& active_updates) {
  learning.calculate_sequence_learning(
      time_step,
      active_cells_time,
      learn_cells_time,
      prediction.get_prev_active_segment_syn_bits(),
      prediction.get_current_best_segments(),
      prediction.get_current_active_segment_syn_bits(),
      prediction.synapse_words_per_segment(),
      distal,
      active_updates.segment,
      active_updates.active_synapses,
      active_updates.new_segment,
      active_updates.new_synapses,
      prediction.get_seg_ind_update_mutable(),
      prediction.get_seg_active_syn_mutable());
}

TEST(SequenceLearningRegression, previous_prediction_reinforces_on_learning) {
  constexpr int num_columns = 1;
  constexpr int cells_per_column = 1;
  constexpr int max_segments = 1;
  constexpr int max_synapses = 2;

  PredictCellsCalculator prediction(PredictCellsCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f, /*activation_threshold=*/0});
  SequenceLearningCalculator learning(SequenceLearningCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f,
      /*permanence_inc=*/0.1f,
      /*permanence_dec=*/0.05f});

  std::vector<DistalSynapse> distal(
      num_columns * cells_per_column * max_segments * max_synapses,
      DistalSynapse{0, 0, 0.3f});
  std::vector<int> active_cells_time(num_columns * cells_per_column * 2, -1);
  std::vector<int> learn_cells_time(num_columns * cells_per_column * 2, -1);
  ActiveUpdateState active_updates =
      make_active_update_state(num_columns * cells_per_column, max_synapses);

  active_cells_time[idx_cell_time(cells_per_column, 0, 0, 0)] = 1;
  prediction.calculate_predict_cells(1, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 1, active_cells_time, learn_cells_time, distal, active_updates);

  active_cells_time[idx_cell_time(cells_per_column, 0, 0, 1)] = 2;
  learn_cells_time[idx_cell_time(cells_per_column, 0, 0, 0)] = 2;
  prediction.calculate_predict_cells(2, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 2, active_cells_time, learn_cells_time, distal, active_updates);

  EXPECT_FLOAT_EQ(distal[0].perm, 0.4f);
  EXPECT_FLOAT_EQ(distal[1].perm, 0.4f);
}

TEST(SequenceLearningRegression, segment_handoff_punishes_a_and_queues_b) {
  constexpr int num_columns = 3;
  constexpr int cells_per_column = 1;
  constexpr int max_segments = 2;
  constexpr int max_synapses = 1;

  PredictCellsCalculator prediction(PredictCellsCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f, /*activation_threshold=*/0});
  SequenceLearningCalculator learning(SequenceLearningCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f,
      /*permanence_inc=*/0.1f,
      /*permanence_dec=*/0.05f});

  std::vector<DistalSynapse> distal(
      num_columns * cells_per_column * max_segments * max_synapses,
      DistalSynapse{0, 0, 0.0f});
  const std::size_t seg_a = idx_distal_synapse(0, 0, 0, 0, 1, 2, 1);
  const std::size_t seg_b = idx_distal_synapse(0, 0, 1, 0, 1, 2, 1);
  distal[seg_a] = DistalSynapse{1, 0, 0.5f};
  distal[seg_b] = DistalSynapse{2, 0, 0.5f};

  std::vector<int> active_cells_time(num_columns * 2, -1);
  std::vector<int> learn_cells_time(num_columns * 2, -1);
  ActiveUpdateState active_updates =
      make_active_update_state(num_columns, max_synapses);

  active_cells_time[idx_cell_time(1, 1, 0, 0)] = 1;
  prediction.calculate_predict_cells(1, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 1, active_cells_time, learn_cells_time, distal, active_updates);

  active_cells_time[idx_cell_time(1, 2, 0, 0)] = 2;
  prediction.calculate_predict_cells(2, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 2, active_cells_time, learn_cells_time, distal, active_updates);

  EXPECT_FLOAT_EQ(distal[seg_a].perm, 0.45f);
  EXPECT_FLOAT_EQ(distal[seg_b].perm, 0.5f);
  EXPECT_EQ(prediction.get_seg_ind_update()[0], 1);

  active_cells_time[idx_cell_time(1, 0, 0, 0)] = 3;
  learn_cells_time[idx_cell_time(1, 0, 0, 0)] = 3;
  prediction.calculate_predict_cells(3, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 3, active_cells_time, learn_cells_time, distal, active_updates);

  EXPECT_FLOAT_EQ(distal[seg_a].perm, 0.45f);
  EXPECT_FLOAT_EQ(distal[seg_b].perm, 0.6f);
}

TEST(SequenceLearningRegression, all_failed_segments_are_decremented_once) {
  constexpr int num_columns = 3;
  constexpr int cells_per_column = 1;
  constexpr int max_segments = 2;
  constexpr int max_synapses = 1;

  PredictCellsCalculator prediction(PredictCellsCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f, /*activation_threshold=*/0});
  SequenceLearningCalculator learning(SequenceLearningCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f,
      /*permanence_inc=*/0.1f,
      /*permanence_dec=*/0.05f});

  std::vector<DistalSynapse> distal(
      num_columns * cells_per_column * max_segments * max_synapses,
      DistalSynapse{0, 0, 0.0f});
  const std::size_t seg_a = idx_distal_synapse(0, 0, 0, 0, 1, 2, 1);
  const std::size_t seg_b = idx_distal_synapse(0, 0, 1, 0, 1, 2, 1);
  distal[seg_a] = DistalSynapse{1, 0, 0.5f};
  distal[seg_b] = DistalSynapse{2, 0, 0.5f};

  std::vector<int> active_cells_time(num_columns * 2, -1);
  std::vector<int> learn_cells_time(num_columns * 2, -1);
  ActiveUpdateState active_updates =
      make_active_update_state(num_columns, max_synapses);

  active_cells_time[idx_cell_time(1, 1, 0, 0)] = 1;
  active_cells_time[idx_cell_time(1, 2, 0, 0)] = 1;
  prediction.calculate_predict_cells(1, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 1, active_cells_time, learn_cells_time, distal, active_updates);

  prediction.calculate_predict_cells(2, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 2, active_cells_time, learn_cells_time, distal, active_updates);

  EXPECT_FLOAT_EQ(distal[seg_a].perm, 0.45f);
  EXPECT_FLOAT_EQ(distal[seg_b].perm, 0.45f);
}

TEST(SequenceLearningRegression, failed_segment_decrements_only_causal_synapses) {
  constexpr int num_columns = 3;
  constexpr int cells_per_column = 1;
  constexpr int max_segments = 1;
  constexpr int max_synapses = 2;

  PredictCellsCalculator prediction(PredictCellsCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f, /*activation_threshold=*/0});
  SequenceLearningCalculator learning(SequenceLearningCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f,
      /*permanence_inc=*/0.1f,
      /*permanence_dec=*/0.05f});

  std::vector<DistalSynapse> distal(
      num_columns * cells_per_column * max_segments * max_synapses,
      DistalSynapse{0, 0, 0.0f});
  distal[0] = DistalSynapse{1, 0, 0.5f};
  distal[1] = DistalSynapse{2, 0, 0.5f};
  std::vector<int> active_cells_time(num_columns * 2, -1);
  std::vector<int> learn_cells_time(num_columns * 2, -1);
  ActiveUpdateState active_updates =
      make_active_update_state(num_columns, max_synapses);

  active_cells_time[idx_cell_time(1, 1, 0, 0)] = 1;
  prediction.calculate_predict_cells(1, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 1, active_cells_time, learn_cells_time, distal, active_updates);

  prediction.calculate_predict_cells(2, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 2, active_cells_time, learn_cells_time, distal, active_updates);

  EXPECT_FLOAT_EQ(distal[0].perm, 0.45f);
  EXPECT_FLOAT_EQ(distal[1].perm, 0.5f);
}

TEST(SequenceLearningRegression, active_cell_does_not_punish_predicting_segment) {
  constexpr int num_columns = 2;
  constexpr int cells_per_column = 1;
  constexpr int max_segments = 1;
  constexpr int max_synapses = 1;

  PredictCellsCalculator prediction(PredictCellsCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f, /*activation_threshold=*/0});
  SequenceLearningCalculator learning(SequenceLearningCalculator::Config{
      num_columns, cells_per_column, max_segments, max_synapses,
      /*connect_permanence=*/0.2f,
      /*permanence_inc=*/0.1f,
      /*permanence_dec=*/0.05f});

  std::vector<DistalSynapse> distal(
      num_columns * cells_per_column * max_segments * max_synapses,
      DistalSynapse{0, 0, 0.0f});
  distal[0] = DistalSynapse{1, 0, 0.5f};
  std::vector<int> active_cells_time(num_columns * 2, -1);
  std::vector<int> learn_cells_time(num_columns * 2, -1);
  ActiveUpdateState active_updates =
      make_active_update_state(num_columns, max_synapses);

  active_cells_time[idx_cell_time(1, 1, 0, 0)] = 1;
  prediction.calculate_predict_cells(1, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 1, active_cells_time, learn_cells_time, distal, active_updates);

  active_cells_time[idx_cell_time(1, 0, 0, 0)] = 2;
  prediction.calculate_predict_cells(2, active_cells_time, distal);
  run_sequence_learning(
      learning, prediction, 2, active_cells_time, learn_cells_time, distal, active_updates);

  EXPECT_FLOAT_EQ(distal[0].perm, 0.5f);
}



