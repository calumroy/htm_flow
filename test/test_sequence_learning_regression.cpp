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

TEST(SequenceLearningRegression, predict_update_struct_persists_and_reinforces_on_learning) {
  // Topology: single column, single cell, single segment, two synapses.
  const int num_columns = 1;
  const int cells_per_column = 1;
  const int max_segments = 1;
  const int max_syn = 2;

  PredictCellsCalculator pred(PredictCellsCalculator::Config{
      num_columns,
      cells_per_column,
      max_segments,
      max_syn,
      /*connect_permanence=*/0.2f,
      /*activation_threshold=*/0, // any connected active synapse will trigger prediction
  });

  SequenceLearningCalculator sl(SequenceLearningCalculator::Config{
      num_columns,
      cells_per_column,
      max_segments,
      max_syn,
      /*connect_permanence=*/0.2f,
      /*permanence_inc=*/0.1f,
      /*permanence_dec=*/0.05f,
  });

  // Distal synapses: origin (0,0,seg0) -> both endpoints point to (0,0) with perm 0.3.
  std::vector<DistalSynapse> distal(static_cast<std::size_t>(num_columns * cells_per_column * max_segments * max_syn));
  for (int syn = 0; syn < max_syn; ++syn) {
    const std::size_t idx = idx_distal_synapse(0, 0, 0, static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cells_per_column),
                                               static_cast<std::size_t>(max_segments),
                                               static_cast<std::size_t>(max_syn));
    distal[idx] = DistalSynapse{/*target_col=*/0, /*target_cell=*/0, /*perm=*/0.3f};
  }

  // State tensors.
  std::vector<int> active_cells_time(num_columns * cells_per_column * 2, -1);
  std::vector<int> learn_cells_time(num_columns * cells_per_column * 2, -1);
  // NOTE: predict_cells_time is owned by PredictCellsCalculator.

  // ----------------------------------------------------------------------------
  // t=1: make the cell active so it enters predictive state and emits update tensors.
  // ----------------------------------------------------------------------------
  active_cells_time[idx_cell_time(cells_per_column, 0, 0, 0)] = 1;
  pred.calculate_predict_cells(/*time_step=*/1, active_cells_time, distal);

  // The cell should now be predictive at t=1.
  {
    const auto& pct = pred.get_predict_cells_time();
    const int p0 = pct[idx_cell_time(cells_per_column, 0, 0, 0)];
    const int p1 = pct[idx_cell_time(cells_per_column, 0, 0, 1)];
    EXPECT_TRUE(p0 == 1 || p1 == 1);
  }

  // ----------------------------------------------------------------------------
  // t=2: the cell enters learning (learn_now && !learn_prev), and sequence learning
  // should consume the predict-cells update structure created when the cell ENTERED
  // predictive state. This is the behavior that was broken when predict-cells cleared
  // update tensors each timestep.
  // ----------------------------------------------------------------------------
  learn_cells_time[idx_cell_time(cells_per_column, 0, 0, 0)] = 2;

  // No active-cells-side update structures in this unit test.
  std::vector<int> seg_ind_update_active(num_columns * cells_per_column, -1);
  std::vector<int8_t> seg_active_syn_active(num_columns * cells_per_column * max_syn, 0);
  std::vector<int> seg_ind_new_syn_active(num_columns * cells_per_column, -1);
  std::vector<DistalSynapse> seg_new_syn_active(num_columns * cells_per_column * max_syn, DistalSynapse{0, 0, -1.0f});

  sl.calculate_sequence_learning(/*time_step=*/2,
                                active_cells_time,
                                learn_cells_time,
                                pred.get_predict_cells_time(),
                                distal,
                                seg_ind_update_active,
                                seg_active_syn_active,
                                seg_ind_new_syn_active,
                                seg_new_syn_active,
                                pred.get_seg_ind_update_mutable(),
                                pred.get_seg_active_syn_mutable());

  // At least one synapse should have increased above 0.3 due to positive reinforcement.
  float max_perm = 0.0f;
  for (const auto& s : distal) {
    max_perm = std::max(max_perm, s.perm);
  }
  EXPECT_GT(max_perm, 0.3f);
}


