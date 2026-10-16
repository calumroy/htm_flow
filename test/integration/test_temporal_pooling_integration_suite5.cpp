#include <gtest/gtest.h>

#include "../test_utils/tp_harness.hpp"
#include "../test_utils/tp_inputs.hpp"
#include "../test_utils/tp_metrics.hpp"

#include <cstdint>
#include <iostream>
#include <sstream>
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

// Per-layer running measurements over the evaluation steps.
//
// Pooling is measured on columns, not cells: the fraction of columns active at
// t-1 that are still active at t. Temporal pooling is a column staying active
// across consecutive inputs, so that is the quantity to measure.
//
// Bursting is measured separately from ActiveCellsCalculator's burst record.
// It is not folded into the pooling number. A layer can score high pooling by
// bursting (a column that keeps winning while bursting stays bursting), so the
// test asserts a low burst fraction as a second condition.
struct LayerMeasure {
  TemporalPoolingMeasure pooling;
  double pooled = 0.0;
  double burst_sum = 0.0;
  int steps = 0;

  void record(temporal_pooling_test_utils::HtmPipelineHarness& layer, int time_step) {
    const std::vector<uint8_t> cols = layer.activeColumns01();
    pooled = pooling.temporalPoolingPercent(cols);
    burst_sum += temporal_pooling_test_utils::burstFraction(cols, layer.burst_columns_time(),
                                                            time_step);
    ++steps;
  }

  double burstFraction() const { return steps > 0 ? burst_sum / steps : 0.0; }
};

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
  We compare how much each layer's set of winning columns stays the same from
  one sequence position to the next. Separately, we measure what fraction of
  Layer 1's winning columns are bursting, read from ActiveCellsCalculator's
  burst record. A column that bursts and keeps winning stays bursting, so high
  column pooling with high bursting is not temporal pooling.

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
  Layer 1 keeps at least kLayer1MinPooling of its winning columns between
  sequence positions.
  Layer 1 column pooling is at least kLayer1MinMargin above Layer 0.
  At most kLayer1MaxBurstFraction of Layer 1's winning columns are bursting.

  Fail:
  Layer 1 columns do not stay active across positions, Layer 1 is not more
  stable than the Layer 0 control, or Layer 1 reaches its stability by
  bursting.

  Steps:
  1. Run a repeating left-to-right line sequence for train_steps.
  2. Run three more complete cycles.
  3. Each step, record successive winning-column overlap in both layers and
     the bursting fraction of Layer 1's winning columns.
  4. Require Layer 1 to pool strongly, exceed Layer 0, and not burst.
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

  LayerMeasure layer0;
  LayerMeasure layer1;

  const int eval_steps = 3 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    layer0.record(htm.layer0(), time_step);
    layer1.record(htm.layer1(), time_step);
    ++time_step;
  }

  const double kLayer1MinPooling = 0.65;
  const double kLayer1MinMargin = 0.50;
  const double kLayer1MaxBurstFraction = 0.10;

  const auto summary = [&]() {
    std::ostringstream os;
    os << "layer0 column pooling=" << layer0.pooled
       << " layer1 column pooling=" << layer1.pooled
       << " layer0 burst fraction=" << layer0.burstFraction()
       << " layer1 burst fraction=" << layer1.burstFraction();
    return os.str();
  };

  std::cout << "[ suite5 ] " << summary() << "\n";

  EXPECT_GE(layer1.pooled, kLayer1MinPooling)
      << "Layer 1 should keep most of its winning columns active across the sequence. "
      << summary();
  EXPECT_GE(layer1.pooled, layer0.pooled + kLayer1MinMargin)
      << "Layer 1 should pool more strongly than the Layer 0 control. " << summary();
  EXPECT_LE(layer1.burstFraction(), kLayer1MaxBurstFraction)
      << "Layer 1 should reach its stability by prediction, not by bursting. " << summary();
}
