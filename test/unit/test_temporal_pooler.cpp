#include <gtest/gtest.h>

#include <htm_flow/temporal_pooler/temporal_pooler.hpp>

using temporal_pooler::TemporalPoolerCalculator;
using sequence_pooler::DistalSynapse;

namespace {

inline int idx_cell_time(int num_cells_per_col, int col, int cell, int slot) {
  return (col * num_cells_per_col + cell) * 2 + slot;
}

inline int idx_cell_seg(int num_cells_per_col, int max_segments_per_cell, int col, int cell, int seg) {
  return (col * num_cells_per_col + cell) * max_segments_per_cell + seg;
}

} // namespace

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
  tp.set_proximal_reinforcement_scales(/*active_predict_scale=*/1.0f,
                                       /*predictive_non_active_scale=*/0.0f,
                                       /*post_active_scale=*/0.0f);

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

TEST(TemporalPooler, distal_persistence_extends_predictive_state_with_segment_evidence) {
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
  // extend predictive state at t=2, but only because we provide segment evidence at t=1.
  active_segs_time[idx_cell_seg(/*num_cells_per_col=*/2, /*max_segments_per_cell=*/1, 0, 0, 0)] = 1;
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
  EXPECT_EQ(active_segs_time[idx_cell_seg(/*num_cells_per_col=*/2, /*max_segments_per_cell=*/1, 0, 0, 0)], 2);
}

TEST(TemporalPooler, distal_persistence_requires_recent_segment_evidence) {
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

  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 0;
  tp.update_distal(/*time_step=*/1,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

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
  EXPECT_FALSE(p0 == 2 || p1 == 2);
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
  active_segs_time[idx_cell_seg(/*num_cells_per_col=*/2, /*max_segments_per_cell=*/1, 0, 0, 0)] = 1;
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

TEST(TemporalPooler, distal_reuses_subconnected_prev2_segment) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/1,
      /*spatial_permanence_inc=*/0.0f,
      /*seq_permanence_inc=*/0.04f,
      /*seq_permanence_dec=*/0.02f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.1f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  std::vector<DistalSynapse> distal(2 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  // Origin cell: (col0, cell0, seg0)
  // syn0 is sub-connected and matches the prev2 set. It is intentionally NOT
  // active now; TP distal should reinforce the selected prev2 context, not the
  // current active context.
  distal[0] = DistalSynapse{/*target_col=*/1, /*target_cell=*/1, /*perm=*/0.1f};
  // syn1 is sub-connected and active now, but outside the prev2 set, so it should
  // decay instead of being rewarded.
  distal[1] = DistalSynapse{/*target_col=*/0, /*target_cell=*/1, /*perm=*/0.1f};

  std::vector<int> learn_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_cells_time(2 * 2 * 2, -1);
  std::vector<int> predict_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_segs_time(2 * 2 * 1, -1);

  // Make origin cell active-predict at t=2.
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 2;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;

  // Make syn1's endpoint active now to prove current activity is not the TP distal
  // reinforcement criterion.
  active_cells_time[idx_cell_time(2, 0, 1, 0)] = 2;
  std::vector<std::pair<int, int>> new_learn_cells_list = {{1, 1}};

  tp.update_distal(/*time_step=*/2,
                   new_learn_cells_list,
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // The segment should be reused and reinforced, not overwritten just because it is
  // still below connect_permanence.
  EXPECT_EQ(distal[0].target_col, 1);
  EXPECT_EQ(distal[0].target_cell, 1);
  EXPECT_FLOAT_EQ(distal[0].perm, 0.14f);
  EXPECT_EQ(distal[1].target_col, 0);
  EXPECT_EQ(distal[1].target_cell, 1);
  EXPECT_FLOAT_EQ(distal[1].perm, 0.08f);
}

TEST(TemporalPooler, proximal_update_uses_last_active_predict_support) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/3,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/3,
      /*spatial_permanence_inc=*/0.04f,
      /*seq_permanence_inc=*/0.04f,
      /*seq_permanence_dec=*/0.02f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.1f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });

  std::vector<DistalSynapse> distal(2 * 3 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  std::vector<int> learn_cells_time(2 * 3 * 2, -1);
  std::vector<int> active_cells_time(2 * 3 * 2, -1);
  std::vector<int> predict_cells_time(2 * 3 * 2, -1);
  std::vector<int> active_segs_time(2 * 3 * 1, -1);

  // Two cells in column 0 are active-predict at t=2. Column 1 has only a
  // generic prediction, so it should not receive TP proximal reinforcement.
  active_cells_time[idx_cell_time(3, 0, 0, 0)] = 2;
  active_cells_time[idx_cell_time(3, 0, 1, 0)] = 2;
  predict_cells_time[idx_cell_time(3, 0, 0, 0)] = 1;
  predict_cells_time[idx_cell_time(3, 0, 1, 0)] = 1;
  predict_cells_time[idx_cell_time(3, 1, 0, 0)] = 1;

  tp.update_distal(/*time_step=*/2,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  std::vector<int> pot_inputs = {
      1, 0, 1, // col0
      1, 1, 1  // col1
  };
  std::vector<float> proximal_perm = {
      0.25f, 0.0f, 0.35f, // col0
      0.0f, 0.0f, 0.0f    // col1
  };
  const auto stats =
      tp.update_proximal(/*support_time=*/2,
                         pot_inputs,
                         proximal_perm);

  EXPECT_EQ(stats.reinforced_inputs, 2);
  EXPECT_EQ(stats.reinforced_columns, 1);
  // Two active-predict cells in col0 give 2 * 0.04 reinforcement to active
  // proximal inputs. Col1 has a generic prediction only, so it is unchanged.
  EXPECT_FLOAT_EQ(proximal_perm[0], 0.33f);
  EXPECT_FLOAT_EQ(proximal_perm[1], 0.0f);
  EXPECT_FLOAT_EQ(proximal_perm[2], 0.43f);
  EXPECT_FLOAT_EQ(proximal_perm[3], 0.0f);
  EXPECT_FLOAT_EQ(proximal_perm[4], 0.0f);
  EXPECT_FLOAT_EQ(proximal_perm[5], 0.0f);

  std::vector<float> stale_perm(2 * 3, 0.0f);
  EXPECT_EQ(tp.update_proximal(/*support_time=*/1,
                               pot_inputs,
                               stale_perm).reinforced_inputs,
            0);
  EXPECT_FLOAT_EQ(stale_perm[0], 0.0f);
  EXPECT_FLOAT_EQ(stale_perm[2], 0.0f);
}

TEST(TemporalPooler, proximal_update_includes_predictive_non_active_columns) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/3,
      /*spatial_permanence_inc=*/0.04f,
      /*seq_permanence_inc=*/0.04f,
      /*seq_permanence_dec=*/0.02f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.1f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });
  tp.set_proximal_reinforcement_scales(/*active_predict_scale=*/1.0f,
                                       /*predictive_non_active_scale=*/1.0f,
                                       /*post_active_scale=*/0.0f);

  std::vector<DistalSynapse> distal(2 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  std::vector<int> learn_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_cells_time(2 * 2 * 2, -1);
  std::vector<int> predict_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_segs_time(2 * 2 * 1, -1);

  // Column 0 is active-predict and gets the existing active support path.
  active_cells_time[idx_cell_time(2, 0, 0, 0)] = 2;
  predict_cells_time[idx_cell_time(2, 0, 0, 0)] = 1;

  // Column 1 is predictive at t=2 with real segment support, but it did not win
  // inhibition. TP should still give its current active proximal inputs a local
  // permanence nudge so distal prediction can become future overlap.
  predict_cells_time[idx_cell_time(2, 1, 0, 0)] = 2;
  active_segs_time[idx_cell_seg(2, 1, 1, 0, 0)] = 2;

  tp.update_distal(/*time_step=*/2,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  std::vector<uint8_t> col_active = {1, 0};
  std::vector<int> pot_inputs = {
      1, 0, 1, // col0
      0, 1, 1  // col1
  };
  std::vector<float> proximal_perm = {
      0.0f, 0.0f, 0.35f, // col0
      0.0f, 0.29f, 0.0f  // col1
  };

  const auto stats =
      tp.update_proximal(/*support_time=*/2,
                         pot_inputs,
                         proximal_perm,
                         &col_active,
                         &predict_cells_time,
                         &active_segs_time);

  EXPECT_EQ(stats.reinforced_inputs, 4);

  EXPECT_FLOAT_EQ(proximal_perm[0], 0.04f);
  EXPECT_FLOAT_EQ(proximal_perm[1], 0.0f);
  EXPECT_FLOAT_EQ(proximal_perm[2], 0.39f);
  EXPECT_FLOAT_EQ(proximal_perm[3], 0.0f);
  EXPECT_FLOAT_EQ(proximal_perm[4], 0.33f);
  EXPECT_FLOAT_EQ(proximal_perm[5], 0.04f);
}

TEST(TemporalPooler, proximal_update_bridges_post_active_predictive_non_active_columns) {
  TemporalPoolerCalculator tp(TemporalPoolerCalculator::Config{
      /*num_columns=*/2,
      /*cells_per_column=*/2,
      /*max_segments_per_cell=*/1,
      /*max_synapses_per_segment=*/2,
      /*num_pot_synapses=*/3,
      /*spatial_permanence_inc=*/0.04f,
      /*seq_permanence_inc=*/0.04f,
      /*seq_permanence_dec=*/0.02f,
      /*min_num_syn_threshold=*/0,
      /*new_syn_permanence=*/0.1f,
      /*connect_permanence=*/0.2f,
      /*delay_length=*/4,
  });
  tp.set_proximal_reinforcement_scales(/*active_predict_scale=*/1.0f,
                                       /*predictive_non_active_scale=*/1.0f,
                                       /*post_active_scale=*/1.0f);

  std::vector<DistalSynapse> distal(2 * 2 * 1 * 2, DistalSynapse{0, 0, 0.0f});
  std::vector<int> learn_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_cells_time(2 * 2 * 2, -1);
  std::vector<int> predict_cells_time(2 * 2 * 2, -1);
  std::vector<int> active_segs_time(2 * 2 * 1, -1);

  // t=2: column 1 is correctly predicted and active. TP records active-predict
  // support for the column after its distal update.
  active_cells_time[idx_cell_time(2, 1, 0, 0)] = 2;
  predict_cells_time[idx_cell_time(2, 1, 0, 0)] = 1;
  tp.update_distal(/*time_step=*/2,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  // t=3: the same column did not win inhibition, but it is still predictive
  // with segment evidence. This is the late-side bridge case.
  predict_cells_time[idx_cell_time(2, 1, 0, 1)] = 3;
  active_segs_time[idx_cell_seg(2, 1, 1, 0, 0)] = 3;
  tp.update_distal(/*time_step=*/3,
                   /*new_learn_cells_list=*/{},
                   learn_cells_time,
                   predict_cells_time,
                   active_cells_time,
                   active_segs_time,
                   distal);

  std::vector<uint8_t> col_active = {1, 0};
  std::vector<int> pot_inputs = {
      1, 1, 1, // col0
      0, 1, 1  // col1
  };
  std::vector<float> proximal_perm = {
      0.0f, 0.0f, 0.0f,  // col0
      0.0f, 0.29f, 0.0f  // col1
  };

  const auto stats =
      tp.update_proximal(/*support_time=*/3,
                         pot_inputs,
                         proximal_perm,
                         &col_active,
                         &predict_cells_time,
                         &active_segs_time);

  EXPECT_EQ(stats.reinforced_inputs, 2);

  // The predictive non-active bridge gives one increment, and the stricter
  // post-active bridge gives one more because the column was active-predict at
  // the previous timestep and is still segment-backed predictive now.
  EXPECT_FLOAT_EQ(proximal_perm[3], 0.0f);
  EXPECT_FLOAT_EQ(proximal_perm[4], 0.37f);
  EXPECT_FLOAT_EQ(proximal_perm[5], 0.08f);
}


