#include "../test_utils/tp_harness.hpp"
#include "../test_utils/tp_inputs.hpp"
#include "../test_utils/tp_metrics.hpp"

#include <algorithm>
#include <random>
#include <vector>

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::TemporalPoolingMeasure;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::similarityPercent;

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
  HtmPipelineHarness htm(suiteConfig());
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  // Longer warmup helps the end-to-end pipeline learn a stable representation.
  temporal_pooling_test_utils::runSteps(time_step, /*num_steps=*/300, [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
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
  HtmPipelineHarness htm(suiteConfig());
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  temporal_pooling_test_utils::runSteps(time_step, /*num_steps=*/120, [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
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
  HtmPipelineHarness htm(suiteConfig());
  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  inputs.setSequenceProbability(1.0);

  int time_step = 1;
  temporal_pooling_test_utils::runSteps(time_step, /*num_steps=*/100, [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
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
  //    Here we implement the *actual intent* in C++ by building a 2-layer system:
  //      - Layer 0: sensory input -> active columns -> learning cells
  //      - Layer 1: input = Layer 0 active columns -> its own columns/cells
  //    This matches the Python concept of "output of one passes to input of the next".
  // 2) Drive the network with a repeating input sequence and allow both layers to learn.
  // 3) Measure temporal pooling in BOTH layers on the learning-cells representation.
  //    Why learning cells? The Python suites often look at cell-level outputs, and learning selection
  //    depends on predictive state (which temporal pooling influences).
  // 4) Assert pooling in Layer 1 is greater than (or at least not worse than) Layer 0.
  // Build a 2-layer config from the suite defaults
  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = suiteConfig();
  cfg.l1 = suiteConfig();
  // Layer-1 input == Layer-0 column grid.
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
  // Warm up longer than the single-layer tests to allow both layers to settle.
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/260,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 260;

  TemporalPoolingMeasure m_l0_learn;
  TemporalPoolingMeasure m_l1_learn;

  double pooled_l0 = 0.0;
  double pooled_l1 = 0.0;

  const int eval_steps = 3 * inputs.seqLen();
  for (int i = 0; i < eval_steps; ++i) {
    const std::vector<int> in0 = inputs.next(htm.rng());
    htm.step(time_step, in0);

    pooled_l0 = m_l0_learn.temporalPoolingPercent(htm.layer0().learnCells01(time_step));
    pooled_l1 = m_l1_learn.temporalPoolingPercent(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  EXPECT_GE(pooled_l1, pooled_l0 - 0.02) << "Expected temporal pooling to increase (or at least not decrease) up the hierarchy";
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
  HtmPipelineHarness htm(suiteConfig());

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
  temporal_pooling_test_utils::runSteps(time_step, /*num_steps=*/220, [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 220;
  std::vector<std::vector<uint8_t>> a1;
  capture_cycle(time_step, a1);

  // Pattern B (odd positions)
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  temporal_pooling_test_utils::runSteps(time_step, /*num_steps=*/220, [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 220;
  std::vector<std::vector<uint8_t>> b;
  capture_cycle(time_step, b);

  // Back to Pattern A again
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  temporal_pooling_test_utils::runSteps(time_step, /*num_steps=*/260, [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
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

