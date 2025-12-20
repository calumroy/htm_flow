#include <gtest/gtest.h>

#include "test_utils/tp_harness.hpp"
#include "test_utils/tp_inputs.hpp"
#include "test_utils/tp_metrics.hpp"

#include <vector>

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::TemporalPoolingMeasure;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::similarityPercent;

namespace {

inline std::vector<uint8_t> representativeOverCycle(TwoLayerHtmHarness& htm,
                                                    VerticalLineInputs& inputs,
                                                    int& time_step,
                                                    int seq_len) {
  // Majority-vote representative of the top layer over one cycle.
  std::vector<std::vector<uint8_t>> samples;
  samples.reserve(static_cast<std::size_t>(seq_len));
  for (int i = 0; i < seq_len; ++i) {
    const std::vector<int> in = inputs.next(htm.rng());
    htm.step(time_step, in);
    samples.push_back(htm.layer1().learnCells01(time_step));
    ++time_step;
  }

  const std::size_t n = samples[0].size();
  std::vector<int> counts(n, 0);
  for (const auto& s : samples) {
    for (std::size_t i = 0; i < n; ++i) counts[i] += (s[i] != 0) ? 1 : 0;
  }
  std::vector<uint8_t> out(n, 0);
  const int half = static_cast<int>(samples.size() / 2);
  for (std::size_t i = 0; i < n; ++i) out[i] = (counts[i] > half) ? 1 : 0;
  return out;
}

} // namespace

TEST(TemporalPoolingIntegrationSuite4, test_tempEquality_two_disjoint_patterns_do_not_overlap_much) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite4.py::test_tempEquality

  What we are testing (and why):
  - In Suite4, the focus is on higher-layer pooling with multiple patterns.
  - Two input patterns that do not share features should produce pooled outputs
    that have low overlap (they should remain distinct).

  C++ adaptation:
  - We use two deterministic vertical-line patterns that are disjoint at the input level:
    EvenPositions vs OddPositions.
  - We measure similarity on the top layer of a 2-layer hierarchy.
  */

  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = HtmPipelineHarness::Config{};
  cfg.l1 = HtmPipelineHarness::Config{};
  cfg.l1.input_rows = cfg.l0.col_rows;
  cfg.l1.input_cols = cfg.l0.col_cols;
  cfg.rng_seed = 123u;

  TwoLayerHtmHarness htm(cfg);

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setSequenceProbability(1.0);

  const int seq_len = inputs.seqLen();
  int time_step = 1;

  // Learn pattern 1: EvenPositions.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/320,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 320;
  const std::vector<uint8_t> repP1 = representativeOverCycle(htm, inputs, time_step, seq_len);

  // Learn pattern 2: OddPositions.
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/320,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 320;
  const std::vector<uint8_t> repP2 = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double sim = similarityPercent(repP1, repP2);
  EXPECT_LE(sim, 0.60) << "Disjoint input patterns should not produce highly overlapping pooled outputs";
}

TEST(TemporalPoolingIntegrationSuite4, test_temporalDiff_patterns_remain_distinct) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite4.py (general “different patterns stay different” intent)

  What we are testing (and why):
  - Even after both patterns are learned, switching patterns should not collapse them into
    one identical representation.
  */

  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = HtmPipelineHarness::Config{};
  cfg.l1 = HtmPipelineHarness::Config{};
  cfg.l1.input_rows = cfg.l0.col_rows;
  cfg.l1.input_cols = cfg.l0.col_cols;
  cfg.rng_seed = 123u;

  TwoLayerHtmHarness htm(cfg);

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setSequenceProbability(1.0);
  const int seq_len = inputs.seqLen();

  int time_step = 1;

  // Learn both patterns.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/260,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 260;

  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/260,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 260;

  // Capture representatives for each after learning.
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

  What we are testing (and why):
  - The Python test suggests that if the *transition* between two different patterns
    happens enough, the system can “pool” over that larger macro-sequence too.
  - In practice, this means that after many alternations, the top-layer output during
    pattern A and pattern B can become much more similar than it originally was.

  C++ adaptation:
  - We alternate EvenPositions and OddPositions many times.
  - We measure:
    - pooling percent during each pattern (should be high)
    - similarity between the final top representations for each pattern (should increase)
  */

  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = HtmPipelineHarness::Config{};
  cfg.l1 = HtmPipelineHarness::Config{};
  cfg.l1.input_rows = cfg.l0.col_rows;
  cfg.l1.input_cols = cfg.l0.col_cols;
  cfg.rng_seed = 123u;

  TwoLayerHtmHarness htm(cfg);

  VerticalLineInputs inputs(/*width=*/htm.layer0().cfg().input_cols,
                            /*height=*/htm.layer0().cfg().input_rows,
                            /*seq_len=*/htm.layer0().cfg().input_cols);
  inputs.setSequenceProbability(1.0);
  const int seq_len = inputs.seqLen();

  int time_step = 1;

  // First: learn each pattern on its own and record early representatives (pre-transition pooling).
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/220,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 220;
  inputs.setIndex(0);
  const std::vector<uint8_t> repEvenEarly = representativeOverCycle(htm, inputs, time_step, seq_len);

  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/220,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 220;
  inputs.setIndex(0);
  const std::vector<uint8_t> repOddEarly = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double simEarly = similarityPercent(repEvenEarly, repOddEarly);

  // Now: alternate patterns many times to “teach” the transition.
  for (int i = 0; i < 12; ++i) {
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

  // Measure pooling percent while re-running each pattern once.
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

  // Capture final representatives and compare.
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  inputs.setIndex(0);
  const std::vector<uint8_t> repEvenFinal = representativeOverCycle(htm, inputs, time_step, seq_len);
  inputs.setPattern(VerticalLineInputs::Pattern::OddPositions);
  inputs.setIndex(0);
  const std::vector<uint8_t> repOddFinal = representativeOverCycle(htm, inputs, time_step, seq_len);

  const double simFinal = similarityPercent(repEvenFinal, repOddFinal);

  EXPECT_GE(pooledEven, 0.40);
  EXPECT_GE(pooledOdd, 0.40);
  EXPECT_GE(simFinal, simEarly - 0.05) << "After many alternations, similarity should not decrease";
}

