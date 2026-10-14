#include <gtest/gtest.h>

#include "../test_utils/tp_harness.hpp"
#include "../test_utils/tp_inputs.hpp"
#include "../test_utils/tp_metrics.hpp"

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
// - Moderate proximal learning lets predicted columns learn successive inputs
//   without making the same columns dominate every input.
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
  cfg.l0.min_overlap              = 2;
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
  cfg.l1.max_segments_per_cell             = 4;
  cfg.l1.temp_spatial_perm_inc             = 0.12f;
  cfg.l1.temp_active_predict_proximal_scale = 0.25f;
  cfg.l1.temp_post_active_proximal_scale    = 0.5f;
  cfg.l1.temp_seq_perm_inc                 = 0.21f;

  cfg.rng_seed = 123u;
  return cfg;
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
  strong distal temporal learning, and moderate proximal temporal learning.
  We compare how much each layer's learning-cell output stays active from one
  sequence position to the next.

  Why the configuration is asymmetric:
  Layer 0 should preserve changing sequence positions for the next layer.
  Layer 1 should combine those positions into a more stable representation.
  Copying Layer 0 before applying the Layer 1 overrides keeps unrelated spatial
  and sequence-memory settings equal.

  Pass:
  Layer 1 temporal pooling is at least 0.20.
  Layer 1 temporal pooling is at least 0.10 above Layer 0.

  Fail:
  Layer 1 does not form a stable learning-cell representation,
  or its output is not more stable than the Layer 0 control.

  Steps:
  1. Run a repeating left-to-right line sequence for train_steps.
  2. Run three more complete cycles.
  3. Measure successive learning-cell overlap in both layers.
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
        layer0_measure.temporalPoolingPercent(htm.layer0().learnCells01(time_step));
    layer1_pooled =
        layer1_measure.temporalPoolingPercent(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(layer1_pooled, 0.20)
      << "Layer 1 should form a stable temporally pooled representation";
  EXPECT_GE(layer1_pooled, layer0_pooled + 0.10)
      << "Layer 1 should pool more strongly than the Layer 0 control";
}
