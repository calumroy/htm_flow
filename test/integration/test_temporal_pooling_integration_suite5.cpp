#include <gtest/gtest.h>

#include "../test_utils/tp_harness.hpp"
#include "../test_utils/tp_inputs.hpp"
#include "../test_utils/tp_metrics.hpp"

#include <cstdint>
#include <vector>

using temporal_pooling_test_utils::TemporalPoolingMeasure;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::VerticalLineInputs;

namespace {

// ── Suite-wide configuration ────────────────────────────────────────
// Layer 0 is the control layer. Its temporal-pooling learning rates are zero.
// Layer 1 starts as an exact copy, then changes only the input shape and the
// settings below that increase temporal pooling:
// - A full-input potential pool gives each column more spatial input to pool.
// - More distal segments give each cell more temporal-context capacity.
// - Strong distal learning makes useful predictions form before proximal
//   pooling changes which columns win.
// - Strong proximal learning on columns that were predicted at t-1 but lost
//   inhibition at t (`temp_prev_predict_proximal_scale`) makes those columns
//   learn the current input and win for it later without bursting.
// - `temp_active_predict_proximal_scale` only strengthens inputs a column
//   already wins for. It is kept at a moderate value so pooled columns stay
//   pooled; it does not create pooling on its own.
TwoLayerHtmHarness::Config suiteConfig() {
  TwoLayerHtmHarness::Config cfg;

  // Layer 0 — Suite4 spatial and sequence-memory config, with TP learning off.
  cfg.l0.input_rows               = 24;
  cfg.l0.input_cols               = 16;
  cfg.l0.col_rows                 = 12;
  cfg.l0.col_cols                 = 12;
  cfg.l0.pot_h                    = 12;
  cfg.l0.pot_w                    = 4;
  cfg.l0.center_pot_synapses      = false;
  cfg.l0.wrap_input               = true;
  cfg.l0.inhib_w                  = 7;
  cfg.l0.inhib_h                  = 7;
  cfg.l0.desired_local_activity   = 6;
  cfg.l0.connected_perm           = 0.3f;
  cfg.l0.min_overlap              = 4;
  cfg.l0.min_potential_overlap    = 1;
  cfg.l0.spatial_perm_inc         = 0.05f;
  cfg.l0.spatial_perm_dec         = 0.02f;
  cfg.l0.active_col_perm_dec      = 0.01f;
  cfg.l0.cells_per_column         = 4;
  cfg.l0.max_segments_per_cell    = 2;
  cfg.l0.max_synapses_per_segment = 10;
  cfg.l0.min_num_syn_threshold    = 3;
  cfg.l0.new_syn_permanence       = 0.3f;
  cfg.l0.connect_permanence       = 0.2f;
  cfg.l0.activation_threshold     = 3;
  cfg.l0.seq_perm_inc             = 0.05f;
  cfg.l0.seq_perm_dec             = 0.02f;
  cfg.l0.temp_spatial_perm_inc    = 0.0f;
  cfg.l0.temp_seq_perm_inc        = 0.0f;
  cfg.l0.temp_seq_perm_dec        = 0.0f;
  cfg.l0.rng_seed                 = 123u;

  // Layer 1 — copy Layer 0, then increase only temporal-pooling capacity
  // and learning. Its input shape must match Layer 0's column grid.
  cfg.l1 = cfg.l0;
  cfg.l1.input_rows                        = cfg.l0.col_rows;
  cfg.l1.input_cols                        = cfg.l0.col_cols;
  cfg.l1.pot_h                             = cfg.l1.input_rows;
  cfg.l1.pot_w                             = cfg.l1.input_cols;
  cfg.l1.desired_local_activity            = 3;
  cfg.l1.max_segments_per_cell             = 4;
  cfg.l1.temp_spatial_perm_inc              = 0.1f;
  cfg.l1.temp_active_predict_proximal_scale = 2.0f;
  cfg.l1.temp_prev_predict_proximal_scale   = 5.0f;
  cfg.l1.temp_seq_perm_inc                  = 0.1f;

  cfg.rng_seed = 123u;
  return cfg;
}

// Active cells, with bursting columns removed.
//
// A bursting column has every cell active. A column that keeps winning while
// bursting stays bursting (ActiveCellsCalculator carries the burst forward),
// so it would add `cells_per_column` "pooled" cells every step without any
// correct prediction. Counting those cells would let a layer pass this test by
// bursting. Only columns with at least one inactive cell are counted.
std::vector<uint8_t> nonBurstingActiveCells01(
    const temporal_pooling_test_utils::HtmPipelineHarness& layer) {
  const auto snapshot = layer.snapshot();
  std::vector<uint8_t> active(
      static_cast<std::size_t>(snapshot.columns_shape.rows * snapshot.columns_shape.cols *
                               snapshot.cells_per_column),
      0);
  const std::uint64_t all_cells = (std::uint64_t{1} << snapshot.cells_per_column) - 1;
  for (std::size_t col = 0; col < snapshot.column_cell_masks.size(); ++col) {
    if ((snapshot.column_cell_masks[col].active & all_cells) == all_cells) {
      continue;
    }
    for (int cell = 0; cell < snapshot.cells_per_column; ++cell) {
      const std::uint64_t bit = std::uint64_t{1} << cell;
      active[col * static_cast<std::size_t>(snapshot.cells_per_column) +
             static_cast<std::size_t>(cell)] =
          (snapshot.column_cell_masks[col].active & bit) != 0 ? 1 : 0;
    }
  }
  return active;
}

} // namespace

TEST(TemporalPoolingIntegrationSuite5, test_layer1_maximises_temporal_pooling_over_layer0) {
  /*
  Purpose:
  Show that a hierarchy can concentrate temporal pooling in Layer 1 while
  Layer 0 remains a spatial and sequence-memory input layer.

  What:
  Both layers see the same repeating sequence through the hierarchy.
  Layer 0 has temporal-pooling learning disabled.
  Layer 1 has a full-input potential pool, more distal-segment capacity,
  strong distal temporal learning, and strong proximal temporal learning.
  We compare how much each layer's active-cell output stays active from one
  sequence position to the next. Bursting columns are excluded from the
  measurement. A column that bursts and keeps winning stays bursting, and its
  cells would count as pooled without any correct prediction.

  Why the configuration is asymmetric:
  Layer 0 should preserve changing sequence positions for the next layer.
  Layer 1 should combine those positions into a more stable representation.
  Copying Layer 0 before applying the Layer 1 overrides keeps unrelated spatial
  and sequence-memory settings equal.

  Layer 0 is not expected to reach zero. Its receptive fields span several
  line positions and spatial learning raises permanence to every input a
  column wins for, so some Layer 0 columns win two consecutive positions.
  That is the spatial floor the Layer 1 margin is measured against.

  Pass:
  Layer 1 retains at least 0.65 of its non-bursting active cells between
  sequence positions.
  Layer 1 temporal pooling is at least 0.50 above Layer 0.

  Fail:
  Layer 1 does not form a stable non-bursting active-cell representation,
  or its output is not more stable than the Layer 0 control.

  Steps:
  1. Run a repeating left-to-right line sequence for train_steps.
  2. Run three more complete cycles.
  3. Measure successive non-bursting active-cell overlap in both layers.
  4. Require Layer 1 to pool strongly and exceed Layer 0.
  */

  TwoLayerHtmHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  const int train_steps = 3000;
  int time_step = 1;
  htm_test_gui::startGui(htm);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += train_steps;

  TemporalPoolingMeasure layer0_measure;
  TemporalPoolingMeasure layer1_measure;
  double layer0_pooled = 0.0;
  double layer1_pooled = 0.0;

  const int eval_steps = 3 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    layer0_pooled =
        layer0_measure.temporalPoolingPercent(nonBurstingActiveCells01(htm.layer0()));
    layer1_pooled =
        layer1_measure.temporalPoolingPercent(nonBurstingActiveCells01(htm.layer1()));
    ++time_step;
  }

  EXPECT_GE(layer1_pooled, 0.65)
      << "Layer 1 should keep most of its non-bursting active cells active across the sequence";
  EXPECT_GE(layer1_pooled, layer0_pooled + 0.50)
      << "Layer 1 should pool more strongly than the Layer 0 control";
}
