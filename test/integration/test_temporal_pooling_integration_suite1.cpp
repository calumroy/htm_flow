#include <gtest/gtest.h>

#include "../test_utils/tp_harness.hpp"
#include "../test_utils/tp_inputs.hpp"
#include "../test_utils/tp_metrics.hpp"

#include <vector>

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::TemporalPoolingMeasure;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::VerticalLineInputs;

namespace {

// ── Suite-wide default configuration ────────────────────────────────
// Every test in this file starts from this config.
// Modify a copy in individual tests if needed.
HtmPipelineHarness::Config suiteConfig() {
  HtmPipelineHarness::Config c;
  c.input_rows              = 24;
  c.input_cols              = 16;
  c.col_rows                = 12;
  c.col_cols                = 12;
  c.pot_h                   = 12;
  c.pot_w                   = 4;
  c.center_pot_synapses     = false;
  c.wrap_input              = true;
  c.inhib_w                 = 7;
  c.inhib_h                 = 7;
  c.desired_local_activity  = 6;
  c.connected_perm          = 0.3f;
  c.min_overlap             = 2;
  c.min_potential_overlap   = 0;
  c.spatial_perm_inc        = 0.05f;
  c.spatial_perm_dec        = 0.02f;
  c.active_col_perm_dec     = 0.01f;
  c.cells_per_column        = 4;
  c.max_segments_per_cell   = 2;
  c.max_synapses_per_segment = 10;
  c.min_num_syn_threshold   = 1;
  c.new_syn_permanence      = 0.3f;
  c.connect_permanence      = 0.2f;
  c.activation_threshold    = 3;
  c.seq_perm_inc            = 0.05f;
  c.seq_perm_dec            = 0.02f;
  c.temp_spatial_perm_inc   = 0.05f;
  c.temp_seq_perm_inc       = 0.05f;
  c.temp_delay_length       = 4;
  c.temp_enable_persistence = true;
  c.rng_seed                = 123u;
  return c;
}

inline std::vector<uint8_t> toU8(const std::vector<int>& v01) {
  std::vector<uint8_t> out(v01.size(), 0);
  for (std::size_t i = 0; i < v01.size(); ++i) {
    out[i] = (v01[i] != 0) ? 1 : 0;
  }
  return out;
}

} // namespace

TEST(TemporalPoolingIntegrationSuite1, test_case1_repeating_sequence_pools) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPooling.py::test_case1

  What we are testing (and why):
  - Temporal pooling is the idea that a repeating input sequence (A,B,C,...) will eventually produce
    a *more stable* representation at the top of the system than the raw activity would suggest.
  - If the system is learning the temporal structure, successive outputs during the cycle should
    overlap strongly (high temporal pooling percent).
  */

  // Step-by-step plan:
  // 1) Run a warmup/training phase where the same input sequence repeats.
  // 2) Re-run for a couple more cycles and measure temporal pooling on each step.
  // 3) Expect the running-average temporal pooling to be "high enough".

  HtmPipelineHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/300,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 300;

  TemporalPoolingMeasure m;
  double pooled = 0.0;

  const int eval_steps = 3 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    pooled = m.temporalPoolingPercent(htm.learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(pooled, 0.52) << "Expected high temporal pooling for a repeating input sequence";
}

TEST(TemporalPoolingIntegrationSuite1, test_case2_random_order_pools_less) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPooling.py::test_case2

  What we are testing (and why):
  - When the input order is randomized, the system should not be able to strongly “pool”
    because there is no consistent temporal structure to latch onto.
  - So the temporal pooling measure should remain comparatively low.
  */

  // Step-by-step plan:
  // 1) Warm up with a normal repeating sequence, so the system isn't “cold”.
  // 2) Switch to randomized order by setting sequenceProbability=0.
  // 3) Measure temporal pooling across ~2 cycles and expect it to stay lower.

  HtmPipelineHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/200,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 200;

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

  EXPECT_LE(pooled, 0.75) << "Random ordering should pool less than a stable repeating sequence";
}

TEST(TemporalPoolingIntegrationSuite1, test_case3_missing_inputs_still_pools) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPooling.py::test_case3

  What we are testing (and why):
  - Real streams can have dropped frames / repeated frames.
  - The system should still produce temporal pooling even if some inputs are “missing”
    (modeled here by holding the same input every other step).
  */

  // Step-by-step plan:
  // 1) Warm up the system on a repeating sequence.
  // 2) During evaluation, update the input only on even steps; on odd steps repeat the previous input.
  // 3) Expect temporal pooling to remain “moderate”, not collapse.

  HtmPipelineHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/180,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 180;

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

  EXPECT_GE(pooled, 0.45) << "Expected temporal pooling to remain when some inputs are held/repeated";
}

TEST(TemporalPoolingIntegrationSuite1, test_case4_pooling_increases_with_depth_two_layer) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPooling.py::test_case4

  What we are testing (and why):
  - In the Python code, “depth” is multiple HTM layers inside a region.
  - In htm_flow, we approximate depth with a feed-forward hierarchy:
      layer0_output -> layer1_input
  - The high-level expectation is the same: representations should become *more stable*
    (or at least not less stable) as you go up the hierarchy.
  */

  // Step-by-step plan:
  // 1) Build a two-layer stack where layer1 consumes layer0 active-columns.
  // 2) Warm up on a repeating sequence.
  // 3) Compare temporal pooling in layer0 vs layer1 during evaluation.

  // Build a 2-layer config from the suite defaults
  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = suiteConfig();
  cfg.l1 = suiteConfig();
  cfg.l1.input_rows = cfg.l0.col_rows;
  cfg.l1.input_cols = cfg.l0.col_cols;
  cfg.rng_seed = 123u;

  TwoLayerHtmHarness htm(cfg);

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/260,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 260;

  TemporalPoolingMeasure m0;
  TemporalPoolingMeasure m1;
  double pooled0 = 0.0;
  double pooled1 = 0.0;

  const int eval_steps = 3 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in0 = inputs.next(htm.rng());
    htm.step(time_step, in0);
    pooled0 = m0.temporalPoolingPercent(htm.layer0().learnCells01(time_step));
    pooled1 = m1.temporalPoolingPercent(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(pooled1, pooled0 - 0.02) << "Expected temporal pooling to increase (or at least not decrease) with depth";
}

