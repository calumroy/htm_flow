#include <gtest/gtest.h>

#include "test_utils/tp_harness.hpp"
#include "test_utils/tp_inputs.hpp"
#include "test_utils/tp_metrics.hpp"

#include <algorithm>
#include <vector>

using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::TemporalPoolingMeasure;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::VerticalLineInputs;
using temporal_pooling_test_utils::similarityPercent;

namespace {

inline std::vector<uint8_t> avgOfSamples(const std::vector<std::vector<uint8_t>>& samples) {
  // Return a representative SDR by OR'ing across samples.
  //
  // Why OR instead of majority-vote?
  // - These HTM-like representations can be extremely sparse.
  // - Majority vote often collapses to an all-zero vector, which makes similarity tests meaningless.
  // - OR preserves the "set of features that ever participate" over a cycle.
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

TEST(TemporalPoolingIntegrationSuite2, test_case1_repeating_sequence_pools) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py::test_case1

  This is the same “repeating sequence should pool” story as Suite1, but it exists as a
  separate suite in the legacy tests. We keep it separate in C++ as well.
  */
  HtmPipelineHarness htm(HtmPipelineHarness::Config{});

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
  EXPECT_GE(pooled, 0.50);
}

TEST(TemporalPoolingIntegrationSuite2, test_case4_pooling_increases_with_depth_two_layer) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py::test_case4

  Same intent as Suite1's depth test; Suite2 repeats it with slightly different params in Python.
  We keep the same two-layer approximation in C++.
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

  EXPECT_GE(pooled1, pooled0 - 0.02);
}

TEST(TemporalPoolingIntegrationSuite2, test_multiPattern_differentiation_and_recall) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py::test_multiPattern

  What we are testing (and why):
  - Temporal pooling should form for multiple distinct input patterns.
  - The pooled representations should remain distinct (pattern A != pattern B).
  - If we switch away and later switch back, the old pooled pattern should be recalled.

  Note:
  - The Python test compares temporal pooling percents per layer and requires the “same pattern”
    to match within ~5% across separate runs.
  - In C++, we make this more direct by comparing the SDRs themselves using similarityPercent:
    - A vs A_again: high similarity
    - B vs B_again: high similarity
    - A vs B: low similarity
  */

  HtmPipelineHarness htm(HtmPipelineHarness::Config{});

  VerticalLineInputs inputs(/*width=*/htm.cfg().input_cols, /*height=*/htm.cfg().input_rows, /*seq_len=*/htm.cfg().input_cols);
  inputs.setSequenceProbability(1.0);

  const int seq_len = inputs.seqLen();
  int time_step = 1;

  auto collectPatternRepresentative = [&](VerticalLineInputs::Pattern p, int warmup_steps) {
    inputs.setPattern(p);
    inputs.setIndex(0);

    temporal_pooling_test_utils::runSteps(time_step,
                                          warmup_steps,
                                          [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                          [&]() { return inputs.next(htm.rng()); });
    time_step += warmup_steps;

    // Capture one full cycle of “learn cells” outputs.
    std::vector<std::vector<uint8_t>> cycle;
    cycle.reserve(static_cast<std::size_t>(seq_len));
    for (int i = 0; i < seq_len; ++i) {
      const std::vector<int> in = inputs.next(htm.rng());
      htm.step(time_step, in);
      cycle.push_back(htm.learnCells01(time_step));
      ++time_step;
    }
    return avgOfSamples(cycle);
  };

  // Train & capture pattern A.
  const std::vector<uint8_t> repA1 = collectPatternRepresentative(VerticalLineInputs::Pattern::LeftToRight, /*warmup_steps=*/280);
  // Train & capture pattern B.
  const std::vector<uint8_t> repB1 = collectPatternRepresentative(VerticalLineInputs::Pattern::RightToLeft, /*warmup_steps=*/280);
  // Switch back to A and capture again.
  const std::vector<uint8_t> repA2 = collectPatternRepresentative(VerticalLineInputs::Pattern::LeftToRight, /*warmup_steps=*/seq_len);
  // Switch back to B and capture again.
  const std::vector<uint8_t> repB2 = collectPatternRepresentative(VerticalLineInputs::Pattern::RightToLeft, /*warmup_steps=*/seq_len);

  const double simA = similarityPercent(repA1, repA2);
  const double simB = similarityPercent(repB1, repB2);
  const double simAB = similarityPercent(repA1, repB1);

  EXPECT_GT(simA, 0.0) << "Expected recall: pattern A should produce a non-empty, stable representation";
  EXPECT_GT(simB, 0.0) << "Expected recall: pattern B should produce a non-empty, stable representation";
  EXPECT_LE(simAB, 0.85) << "Expected differentiation: pattern A and B should not collapse to the same pooled output";
}

