#include <gtest/gtest.h>

#include <htm_flow/temporal_pooler/temporal_pooler.hpp>

using temporal_pooler::TemporalPoolerCalculator;
using sequence_pooler::DistalSynapse;

namespace {

inline int idx_cell_time(int num_cells_per_col, int col, int cell, int slot) {
  return (col * num_cells_per_col + cell) * 2 + slot;
}

} // namespace

TEST(TemporalPooler, proximal_updates_expected_synapses) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/3,
      /*spatial_permanence_inc=*/0.1f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.0f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  std::vector<float> col_syn_perm(2 * 3, 0.0f);
  std::vector<int> burst_cols_time(2 * 2, -1);

  // Step 1: establish prev buffers (no increments because prev buffers are zeroed).
  std::vector<int> pot_inputs_t1 = {
      1, 0, 1, // col0
      0, 1, 0  // col1
  };
  std::vector<uint8_t> col_active_t1 = {1, 0};
  tp.update_proximal(/*time_step=*/1, pot_inputs_t1, col_active_t1, col_syn_perm, burst_cols_time);

  // Step 2: increments based on prev inputs + prev active.
  std::vector<int> pot_inputs_t2 = {
      0, 1, 0, // col0
      1, 1, 1  // col1
  };
  std::vector<uint8_t> col_active_t2 = {1, 1};
  tp.update_proximal(/*time_step=*/2, pot_inputs_t2, col_active_t2, col_syn_perm, burst_cols_time);

  // Column 0:
  // - active now => increment where prev inputs were 1 (syn0, syn2)
  // - was active prev => increment where current inputs are 1 (syn1)
  EXPECT_FLOAT_EQ(col_syn_perm[0 * 3 + 0], 0.1f);
  EXPECT_FLOAT_EQ(col_syn_perm[0 * 3 + 1], 0.1f);
  EXPECT_FLOAT_EQ(col_syn_perm[0 * 3 + 2], 0.1f);

  // Column 1:
  // - active now => increment where prev inputs were 1 (syn1 only)
  // - was NOT active prev => no rule-B increment
  EXPECT_FLOAT_EQ(col_syn_perm[1 * 3 + 0], 0.0f);
  EXPECT_FLOAT_EQ(col_syn_perm[1 * 3 + 1], 0.1f);
  EXPECT_FLOAT_EQ(col_syn_perm[1 * 3 + 2], 0.0f);
}

TEST(TemporalPooler, proximal_skips_bursting_columns_for_ruleA) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/1,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/3,
      /*spatial_permanence_inc=*/0.1f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.0f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  std::vector<float> col_syn_perm(1 * 3, 0.0f);
  std::vector<int> burst_cols_time(1 * 2, -1);

  std::vector<int> pot_inputs_t1 = {1, 1, 1};
  std::vector<uint8_t> col_active_t1 = {1};
  tp.update_proximal(/*time_step=*/1, pot_inputs_t1, col_active_t1, col_syn_perm, burst_cols_time);

  // Mark the column as bursting at time_step 2.
  burst_cols_time[0] = 2;

  std::vector<int> pot_inputs_t2 = {0, 0, 0};
  std::vector<uint8_t> col_active_t2 = {1};
  tp.update_proximal(/*time_step=*/2, pot_inputs_t2, col_active_t2, col_syn_perm, burst_cols_time);

  // Rule A should be skipped (bursting now), so synapses should NOT have increased
  // based on prev inputs.
  EXPECT_FLOAT_EQ(col_syn_perm[0], 0.0f);
  EXPECT_FLOAT_EQ(col_syn_perm[1], 0.0f);
  EXPECT_FLOAT_EQ(col_syn_perm[2], 0.0f);
}

TEST(TemporalPooler, distal_reinforces_best_matching_segment) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/1,
      /*spatial_permanence_inc=*/0.0f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.0f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  // Distal synapses: shape (num_columns=2, cells=2, seg=1, syn=2)
  std::vector<DistalSynapse> distal(2 * 2 * 1 * 2);
  // Origin cell: (col0, cell0, seg0)
  distal[0] = DistalSynapse{/*target_col=*/1, /*target_cell=*/1, /*perm=*/0.5f};
  distal[1] = DistalSynapse{/*target_col=*/0, /*target_cell=*/1, /*perm=*/0.5f};

  // time history tensors: (2,2,2) => 8
  std::vector<int> learn_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_cells_time(2 * 2 * 2, -1);
  std::vector<int> predict_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_segs_time(2 * 2 * 1, -1);

  // Make (col0,cell0) active_predict at t=2 by setting it active at 2 and predicted at 1.
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 2;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;

  // Make target (col1,cell1) active at t=2, so syn0 counts as active synapse for reinforcement.
  active_cells_time[idx_cell_time(2, 1, 1, 0)] = 2;

  // Ensure (col1,cell1) enters learning at t=2 so it's in the prev2 set used for best-match selection.
  std::vector<std::pair<int, int>> new_learn_cells_list = {{1, 1}};

  tp.update_distal(/*time_step=*/2,
                   new_learn_cells_list,
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // Synapse 0 should have been incremented because its endpoint was active at time 2.
  EXPECT_FLOAT_EQ(distal[0].perm, 0.6f);
}

TEST(TemporalPooler, distal_persistence_extends_predictive_state) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/1,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/1,
      /*spatial_permanence_inc=*/0.0f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.0f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  std::vector<DistalSynapse> distal(1 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  std::vector<int> learn_cells_time(1 * 2 * 2, -1);
  std::vector<int> active_cells_time(1 * 2 * 2, -1);
  std::vector<int> predict_cells_time(1 * 2 * 2, -1);
  std::vector<int> active_segs_time(1 * 2 * 1, -1);

  // Step 1 (t=1): make cell (0,0) active_predict by setting it active at 1 and predicted at 0.
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 0;
  tp.update_distal(/*time_step=*/1,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // Step 2 (t=2): keep the cell active, but ensure it was NOT predicted at t=1.
  // The previous tracking streak should update avg_persist and then persistence should
  // extend predictive state at t=2.
  // Preserve the history: slot0 holds t=1, slot1 holds t=2.
  active_cells_time[idx_cell_time(2, 0, 0, 1)] = 2;
  tp.update_distal(/*time_step=*/2,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  const int p0 = predict_cells_time[idx_cell_time(2, 0, 0, 0)];
  const int p1 = predict_cells_time[idx_cell_time(2, 0, 0, 1)];
  EXPECT_TRUE(p0 == 2 || p1 == 2);
}

TEST(TemporalPooler, distal_decrements_inactive_synapses) {
  // Verify that when a segment is reinforced for an active-predictive cell,
  // synapses whose target cells are NOT active get decremented by seq_permanence_dec.
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/1,
      /*spatial_permanence_inc=*/0.0f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.05f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  // Distal synapses: shape (num_columns=2, cells=2, seg=1, syn=2)
  std::vector<DistalSynapse> distal(2 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  // Origin cell: (col0, cell0, seg0)
  // syn0: target (col1, cell1) -- will be active => should be incremented
  distal[0] = DistalSynapse{/*target_col=*/1, /*target_cell=*/1, /*perm=*/0.5f};
  // syn1: target (col0, cell1) -- will NOT be active => should be decremented
  distal[1] = DistalSynapse{/*target_col=*/0, /*target_cell=*/1, /*perm=*/0.5f};

  // time history tensors: (2,2,2) => 8
  std::vector<int> learn_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_cells_time(2 * 2 * 2, -1);
  std::vector<int> predict_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_segs_time(2 * 2 * 1, -1);

  // Make (col0,cell0) active_predict at t=2 by setting it active at 2 and predicted at 1.
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 2;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;

  // Make target (col1,cell1) active at t=2, so syn0 is "active" for reinforcement.
  // Do NOT make (col0,cell1) active, so syn1 is "inactive".
  active_cells_time[idx_cell_time(2, 1, 1, 0)] = 2;

  // Ensure (col1,cell1) enters learning at t=2 so it's in the prev2 set.
  std::vector<std::pair<int, int>> new_learn_cells_list = {{1, 1}};

  tp.update_distal(/*time_step=*/2,
                   new_learn_cells_list,
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // syn0: target was active => incremented: 0.5 + 0.1 = 0.6
  EXPECT_FLOAT_EQ(distal[0].perm, 0.6f);
  // syn1: target was NOT active => decremented: 0.5 - 0.05 = 0.45
  EXPECT_FLOAT_EQ(distal[1].perm, 0.45f);
}

TEST(TemporalPooler, distal_replaces_dead_synapses) {
  // Verify that when an inactive synapse's permanence decays to 0, it gets replaced
  // with a new synapse targeting a recent learning cell.
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/1,
      /*spatial_permanence_inc=*/0.0f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.1f, // large enough to kill the synapse in one step
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  // Distal synapses: shape (num_columns=2, cells=2, seg=1, syn=2)
  std::vector<DistalSynapse> distal(2 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  // Origin cell: (col0, cell0, seg0)
  // syn0: target (col1, cell1) -- will be active => incremented
  distal[0] = DistalSynapse{/*target_col=*/1, /*target_cell=*/1, /*perm=*/0.5f};
  // syn1: target (col0, cell1) -- NOT active, perm=0.05 => will decay to 0 and get replaced
  distal[1] = DistalSynapse{/*target_col=*/0, /*target_cell=*/1, /*perm=*/0.05f};

  std::vector<int> learn_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_cells_time(2 * 2 * 2, -1);
  std::vector<int> predict_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_segs_time(2 * 2 * 1, -1);

  // Make (col0,cell0) active_predict at t=2.
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 2;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;

  // Make target (col1,cell1) active at t=2 (so syn0 is "active").
  active_cells_time[idx_cell_time(2, 1, 1, 0)] = 2;

  // Provide a learning cell so there is a candidate for replacement.
  std::vector<std::pair<int, int>> new_learn_cells_list = {{1, 1}};

  tp.update_distal(/*time_step=*/2,
                   new_learn_cells_list,
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // syn0: active => incremented
  EXPECT_FLOAT_EQ(distal[0].perm, 0.6f);

  // syn1: was 0.05, decremented by 0.1 => would be -0.05, clamped to 0 => replaced.
  // The replacement synapse should have new_syn_permanence (0.3) and target a prev2 cell.
  EXPECT_FLOAT_EQ(distal[1].perm, 0.3f);
  // The replacement should target a cell from prev2_cells (which includes (1,1)).
  EXPECT_EQ(distal[1].target_col, 1);
  EXPECT_EQ(distal[1].target_cell, 1);
}

TEST(TemporalPooler, distal_persistence_does_not_become_sticky_without_activity) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/1,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/1,
      /*spatial_permanence_inc=*/0.0f,
      /*seq_permanence_inc=*/0.1f,
      /*seq_permanence_dec=*/0.0f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.3f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  std::vector<DistalSynapse> distal(1 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  std::vector<int> learn_cells_time(1 * 2 * 2, -1);
  std::vector<int> active_cells_time(1 * 2 * 2, -1);
  std::vector<int> predict_cells_time(1 * 2 * 2, -1);
  std::vector<int> active_segs_time(1 * 2 * 1, -1);

  // t=1: create a single-step active_predict streak for (0,0).
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 0;
  tp.update_distal(/*time_step=*/1,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // t=2: cell is active but was not predicted at t=1 => streak ends and persistence should extend.
  active_cells_time[idx_cell_time(2, 0, 0, 1)] = 2;
  tp.update_distal(/*time_step=*/2,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);
  {
    const int p0 = predict_cells_time[idx_cell_time(2, 0, 0, 0)];
    const int p1 = predict_cells_time[idx_cell_time(2, 0, 0, 1)];
    EXPECT_TRUE(p0 == 2 || p1 == 2);
  }

  // t=3: cell is NOT active. Persistence should not keep the cell predicting indefinitely.
  tp.update_distal(/*time_step=*/3,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);
  {
    const int p0 = predict_cells_time[idx_cell_time(2, 0, 0, 0)];
    const int p1 = predict_cells_time[idx_cell_time(2, 0, 0, 1)];
    EXPECT_FALSE(p0 == 3 || p1 == 3);
  }
}

