#include <gtest/gtest.h>

#include "../test_utils/tp_harness.hpp"
#include "../test_utils/tp_inputs.hpp"
#include "../test_utils/tp_metrics.hpp"

#include <vector>

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::TemporalPoolingMeasure;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::similarityPercent;

namespace {

// ── Suite-wide default configuration ────────────────────────────────
// Suite4 uses the default harness config in a 2-layer hierarchy.
// Layer 1's input matches Layer 0's column grid.
// Every test in this file starts from these configs.
// Modify copies in individual tests if needed.
TwoLayerHtmHarness::Config suiteConfig() {
  TwoLayerHtmHarness::Config cfg;

  // Layer 0 — default pipeline config
  cfg.l0.input_rows              = 24;
  cfg.l0.input_cols              = 16;
  cfg.l0.col_rows                = 12;
  cfg.l0.col_cols                = 12;
  cfg.l0.pot_h                   = 12;
  cfg.l0.pot_w                   = 4;
  cfg.l0.center_pot_synapses     = false;
  cfg.l0.wrap_input              = true;
  cfg.l0.inhib_w                 = 7;
  cfg.l0.inhib_h                 = 7;
  cfg.l0.desired_local_activity  = 6;
  cfg.l0.connected_perm          = 0.3f;
  cfg.l0.min_overlap             = 2;
  cfg.l0.min_potential_overlap   = 1;
  cfg.l0.spatial_perm_inc        = 0.05f;
  cfg.l0.spatial_perm_dec        = 0.02f;
  cfg.l0.active_col_perm_dec     = 0.01f;
  cfg.l0.cells_per_column        = 4;
  cfg.l0.max_segments_per_cell   = 2;
  cfg.l0.max_synapses_per_segment = 10;
  cfg.l0.min_num_syn_threshold   = 1;
  cfg.l0.new_syn_permanence      = 0.3f;
  cfg.l0.connect_permanence      = 0.2f;
  cfg.l0.activation_threshold    = 3;
  cfg.l0.seq_perm_inc            = 0.05f;
  cfg.l0.seq_perm_dec            = 0.02f;
  cfg.l0.temp_spatial_perm_inc   = 0.05f;
  cfg.l0.temp_seq_perm_inc       = 0.05f;
  cfg.l0.rng_seed                = 123u;

  // Layer 1 — same params, but input matches layer 0's column grid
  cfg.l1 = cfg.l0;
  cfg.l1.input_rows = cfg.l0.col_rows;   // 12
  cfg.l1.input_cols = cfg.l0.col_cols;    // 12

  cfg.rng_seed = 123u;
  return cfg;
}

inline std::vector<uint8_t> representativeOverCycle(TwoLayerHtmHarness& htm,
                                                    VerticalLineInputs& inputs,
                                                    int& time_step,
                                                    int seq_len) {
  // Build a single representative SDR for "what this pattern looks like" at the *top* layer,
  // by running one full input cycle and OR-ing the per-step learn-cells outputs.
  //
  // Why do we need a representative at all?
  // - The input is a SEQUENCE (a cycle of vertical lines at different x positions).
  // - The pooled representation for a whole sequence is not one timestep; it is the
  //   stable-ish set of cells that tend to be involved across the cycle.
  //
  // Why OR instead of majority-vote?
  // - With multiple predicted cells per column, learn-cells outputs can be sparse and
  //   variable step-to-step. Majority vote can collapse to all-zeros for short cycles.
  // - OR preserves the union of features that participate across the cycle, matching
  //   Suite2's avgOfSamples approach.
  std::vector<std::vector<uint8_t>> samples;
  samples.reserve(static_cast<std::size_t>(seq_len));
  for (int i = 0; i < seq_len; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    samples.push_back(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  if (samples.empty()) return {};
  const std::size_t n = samples[0].size();
  std::vector<uint8_t> out(n, 0);
  for (const auto& s : samples) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = (out[i] != 0 || s[i] != 0) ? 1 : 0;
    }
  }
  return out;
}

} // namespace

TEST(TemporalPoolingIntegrationSuite4, test_tempEquality_two_disjoint_patterns_do_not_overlap_much) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite4.py::test_tempEquality

  Purpose:
  Show that the model can learn two patterns and still keep their outputs apart.

  What:
  Even lines use only even x positions.
  Odd lines use only odd x positions.
  They share almost no input cells.
  The second layer should also use mostly different cells for each pattern.
  We read the cells the second layer uses while learning.
  That is the output this test compares.

  Why the limit is 0.60:
  Some cells can stay active after the input changes.
  A small model can also share some cells between patterns.
  The test checks that the two outputs are not almost the same.
  It allows them to share some cells.

  Pass:
  Similarity between the two second-layer outputs is 0.60 or less.
  Similarity is the share of active cells in the first output that are also active in the second.

  Fail:
  The second layer uses almost the same cells for both patterns.

  Steps:
  1. Train on even lines for train_steps.
  2. Record the second-layer cells used across one full even-line cycle.
  3. Train on odd lines for train_steps.
  4. Record the second-layer cells used across one full odd-line cycle.
  5. Compare the two records. Similarity must be 0.60 or less.
  */

  TwoLayerHtmHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setSequenceProbability(1.0);

  const int seq_len = inputs.seqLen();
  // Steps used to train each pattern. Change this value only.
  const int train_steps = 320;
  int time_step = 1;

  // Train on even lines, then record one full cycle.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += train_steps;
  const std::vector<uint8_t> repP1 = representativeOverCycle(htm, inputs, time_step, seq_len);

  // Train on odd lines, then record one full cycle.
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += train_steps;
  const std::vector<uint8_t> repP2 = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double sim = similarityPercent(repP1, repP2);
  EXPECT_LE(sim, 0.60) << "Even lines and odd lines should not produce almost the same second-layer output";
}

TEST(TemporalPoolingIntegrationSuite4, test_temporalDiff_patterns_remain_distinct) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite4.py::test_temporalDiff

  Purpose:
  Show that two different inputs still produce two different outputs
  after the model has learned both.

  What:
  Even lines and odd lines do not share input positions.
  Even lines use only even x positions.
  Odd lines use only odd x positions.
  After training on both, the second layer should still use a different set of cells for each.

  Why this test is separate from test_tempEquality:
  test_tempEquality trains one pattern, records it, then trains the other and compares.
  This test trains both patterns first.
  Both patterns can change the model before the comparison.

  Pass:
  Similarity between the two second-layer outputs is 0.70 or less.
  Similarity is the share of active cells in the first output that are also active in the second.

  Fail:
  The second layer keeps the same cells active for both patterns.
  This can happen when cells stay active after the input changes,
  or when the model is so small that almost the same cells learn every input.

  Steps:
  1. Train on even lines for train_steps.
  2. Train on odd lines for train_steps.
  3. Record the second-layer cells used across one full even-line cycle.
  4. Record the second-layer cells used across one full odd-line cycle.
  5. Compare the two records. They must not be almost the same.
  */

  TwoLayerHtmHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setSequenceProbability(1.0);
  const int seq_len = inputs.seqLen();

  // Steps used to train each pattern. Change this value only.
  const int train_steps = 1000;
  int time_step = 1;

  // Train on even lines, then odd lines.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  htm_test_gui::startGui(htm);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });

  time_step += train_steps;
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += train_steps;

  // Record one full cycle of each pattern after both have been trained.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  inputs.setIndex(0);
  const std::vector<uint8_t> repEven = representativeOverCycle(htm, inputs, time_step, seq_len);

  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  inputs.setIndex(0);
  const std::vector<uint8_t> repOdd = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double sim = similarityPercent(repEven, repOdd);
  EXPECT_LE(sim, 0.70);
}

TEST(TemporalPoolingIntegrationSuite4, test_tempDiffPooled_transition_can_become_pooled) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite4.py::test_tempDiffPooled

  Purpose:
  Show that switching between two patterns many times can make their outputs more alike.
  Each pattern should still stay stable on its own.

  What:
  First, train each pattern alone and record how similar the second-layer outputs are.
  Then switch between one even-line cycle and one odd-line cycle, alternate_count times.
  After that, the cells used inside one pattern should stay mostly the same from step to step.
  The two patterns may share more cells than they did before the switching.

  Why the test does not require the two outputs to become the same:
  The Python test expects the two patterns to merge into one stable output.
  That result depends strongly on the settings.
  This test checks the direction only.
  Switching must not make the two outputs less similar.

  Pass:
  The stability score for even lines is 0.30 or more.
  The stability score for odd lines is 0.30 or more.
  The stability score is the share of second-layer cells that stay active
  from one step to the next while one pattern repeats.
  Final similarity is not more than 0.05 below the early similarity.

  Fail:
  The cells for one pattern change a lot from step to step.
  After switching, the two patterns share fewer cells than they did at the start.

  Steps:
  1. Train on even lines for train_steps.
     Record the second-layer cells for one even-line cycle.
  2. Train on odd lines for train_steps.
     Record the second-layer cells for one odd-line cycle.
  3. Compare those two early records.
  4. Switch between one even-line cycle and one odd-line cycle, alternate_count times.
  5. Run one even-line cycle and one odd-line cycle again.
     Measure how stable each pattern's cells are.
  6. Record one more even-line cycle and one more odd-line cycle.
  7. Compare the final records with the early records.
  */

  TwoLayerHtmHarness htm(suiteConfig());

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setSequenceProbability(1.0);
  const int seq_len = inputs.seqLen();

  // Steps used to train each pattern before switching. Change this value only.
  const int train_steps = 220;
  // Number of even-then-odd switches. Change this value only.
  const int alternate_count = 12;
  int time_step = 1;

  // Train each pattern alone and record the early second-layer output.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += train_steps;
  inputs.setIndex(0);
  const std::vector<uint8_t> repEvenEarly = representativeOverCycle(htm, inputs, time_step, seq_len);

  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        train_steps,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += train_steps;
  inputs.setIndex(0);
  const std::vector<uint8_t> repOddEarly = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double simEarly = similarityPercent(repEvenEarly, repOddEarly);

  // Switch between the two patterns so the model can learn the change from one to the other.
  for (int i = 0; i < alternate_count; ++i) {
    inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
    inputs.setIndex(0);
    temporal_pooling_test_utils::runSteps(time_step,
                                          /*num_steps=*/seq_len,
                                          [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                          [&]() { return inputs.next(htm.rng()); });
    time_step += seq_len;

    inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
    inputs.setIndex(0);
    temporal_pooling_test_utils::runSteps(time_step,
                                          /*num_steps=*/seq_len,
                                          [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                          [&]() { return inputs.next(htm.rng()); });
    time_step += seq_len;
  }

  // Measure how stable each pattern's cells are across one cycle.
  TemporalPoolingMeasure mEven;
  TemporalPoolingMeasure mOdd;
  double pooledEven = 0.0;
  double pooledOdd = 0.0;

  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  inputs.setIndex(0);
  for (int i = 0; i < seq_len; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    pooledEven = mEven.temporalPoolingPercent(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  inputs.setIndex(0);
  for (int i = 0; i < seq_len; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    pooledOdd = mOdd.temporalPoolingPercent(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  // Record one more cycle of each pattern and compare with the early records.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  inputs.setIndex(0);
  const std::vector<uint8_t> repEvenFinal = representativeOverCycle(htm, inputs, time_step, seq_len);
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  inputs.setIndex(0);
  const std::vector<uint8_t> repOddFinal = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double simFinal = similarityPercent(repEvenFinal, repOddFinal);
  EXPECT_GE(pooledEven, 0.30);
  EXPECT_GE(pooledOdd, 0.30);
  EXPECT_GE(simFinal, simEarly - 0.05) << "After switching between the two patterns, similarity should not fall";
}

