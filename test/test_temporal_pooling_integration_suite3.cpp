#include <gtest/gtest.h>

#include "test_utils/tp_harness.hpp"
#include "test_utils/tp_inputs.hpp"
#include "test_utils/tp_metrics.hpp"
#include "test_utils/tp_suite3_patterns.hpp"

#include <vector>

using temporal_pooling_test_utils::CustomSdrInputs;
using temporal_pooling_test_utils::HtmPipelineHarness;
using temporal_pooling_test_utils::Suite3Patterns;
using temporal_pooling_test_utils::TwoLayerHtmHarness;
using temporal_pooling_test_utils::similarityPercent;

namespace {

inline std::vector<uint8_t> runAndCaptureOneCycleTopLearn(TwoLayerHtmHarness& htm,
                                                          CustomSdrInputs& inputs,
                                                          int& time_step,
                                                          int seq_len) {
  // Capture the top-layer learn-cells outputs over one sequence cycle,
  // then return a representative by majority vote.
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

TEST(TemporalPoolingIntegrationSuite3, test_tempEquality_two_patterns_recall_and_separation) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite3.py::test_tempEquality

  What we are testing (and why):
  - The legacy Python suite used image sequences (DEF and ABC) and tested that:
    1) Each pattern forms a stable temporally pooled representation over its own sequence.
    2) The pooled representations for different patterns remain distinct.
    3) If we switch away and later switch back, the same pattern is recalled.

  C++ adaptation:
  - We do NOT read image files. We hardcode a small set of binary patterns that preserve the
    *relationships* (ABC and DEF are different sequences).
  - We observe the top layer of a 2-layer hierarchy (layer0 -> layer1) to mirror the
    “higher layer output” behavior from the Python suite.
  */

  Suite3Patterns pats;

  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = HtmPipelineHarness::Config{};
  cfg.l1 = HtmPipelineHarness::Config{};
  cfg.l0.input_rows = pats.h;
  cfg.l0.input_cols = pats.w;
  cfg.l0.col_rows = 10;
  cfg.l0.col_cols = 10;
  // The default harness uses a tall potential pool (pot_h=12) for larger inputs.
  // For Suite3's small (8x8) hardcoded patterns, we must keep pot sizes <= input sizes.
  cfg.l0.pot_h = 4;
  cfg.l0.pot_w = 4;
  cfg.l0.desired_local_activity = 4;
  cfg.l1.input_rows = cfg.l0.col_rows;
  cfg.l1.input_cols = cfg.l0.col_cols;
  cfg.l1.col_rows = 8;
  cfg.l1.col_cols = 8;
  cfg.l1.pot_h = 4;
  cfg.l1.pot_w = 4;
  cfg.l1.desired_local_activity = 3;
  cfg.rng_seed = 123u;

  TwoLayerHtmHarness htm(cfg);

  CustomSdrInputs inputs(pats.w, pats.h);
  const int defIdx = inputs.appendSequence(pats.DEF());
  const int abcIdx = inputs.appendSequence(pats.ABC());

  const int defLen = inputs.getNumInputsInSeq(defIdx);
  const int abcLen = inputs.getNumInputsInSeq(abcIdx);
  ASSERT_EQ(defLen, 3);
  ASSERT_EQ(abcLen, 3);

  int time_step = 1;

  // Learn DEF, then capture a representative.
  inputs.changePattern(defIdx);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/120,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 120;
  const std::vector<uint8_t> repDEF1 = runAndCaptureOneCycleTopLearn(htm, inputs, time_step, defLen);

  // Learn ABC, then capture a representative.
  inputs.changePattern(abcIdx);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/120,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 120;
  const std::vector<uint8_t> repABC1 = runAndCaptureOneCycleTopLearn(htm, inputs, time_step, abcLen);

  // Switch back to DEF and capture again (recall).
  inputs.changePattern(defIdx);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/240,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 240;
  const std::vector<uint8_t> repDEF2 = runAndCaptureOneCycleTopLearn(htm, inputs, time_step, defLen);

  // Switch back to ABC and capture again (recall).
  inputs.changePattern(abcIdx);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/240,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 240;
  const std::vector<uint8_t> repABC2 = runAndCaptureOneCycleTopLearn(htm, inputs, time_step, abcLen);

  const double simDEF = similarityPercent(repDEF1, repDEF2);
  const double simABC = similarityPercent(repABC1, repABC2);
  const double simBetween = similarityPercent(repDEF1, repABC1);

  EXPECT_GE(simDEF, 0.30) << "Expected recall/stability for DEF over time";
  EXPECT_GE(simABC, 0.30) << "Expected recall/stability for ABC over time";
  EXPECT_LE(simBetween, 0.70) << "Expected DEF and ABC pooled representations to remain distinct";
}

TEST(TemporalPoolingIntegrationSuite3, test_temporalDiff_shared_element_produces_intermediate_similarity) {
  /*
  Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite3.py::test_temporalDiff

  What we are testing (and why):
  - In the Python suite, ABC and DEC share exactly one element (C), so after learning,
    their temporally pooled outputs are expected to be partially similar (not identical,
    not completely different).
  */

  Suite3Patterns pats;

  TwoLayerHtmHarness::Config cfg;
  cfg.l0 = HtmPipelineHarness::Config{};
  cfg.l1 = HtmPipelineHarness::Config{};
  cfg.l0.input_rows = pats.h;
  cfg.l0.input_cols = pats.w;
  cfg.l0.col_rows = 10;
  cfg.l0.col_cols = 10;
  cfg.l0.pot_h = 4;
  cfg.l0.pot_w = 4;
  cfg.l0.desired_local_activity = 4;
  cfg.l1.input_rows = cfg.l0.col_rows;
  cfg.l1.input_cols = cfg.l0.col_cols;
  cfg.l1.col_rows = 8;
  cfg.l1.col_cols = 8;
  cfg.l1.pot_h = 4;
  cfg.l1.pot_w = 4;
  cfg.l1.desired_local_activity = 3;
  cfg.rng_seed = 123u;

  TwoLayerHtmHarness htm(cfg);

  CustomSdrInputs inputs(pats.w, pats.h);
  const int abcIdx = inputs.appendSequence(pats.ABC());
  const int decIdx = inputs.appendSequence(pats.DEC());

  const int len = inputs.getNumInputsInSeq(abcIdx);
  ASSERT_EQ(len, 3);

  int time_step = 1;

  // Learn ABC a bunch.
  inputs.changePattern(abcIdx);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/240,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 240;
  const std::vector<uint8_t> repABC = runAndCaptureOneCycleTopLearn(htm, inputs, time_step, len);

  // Learn DEC a bunch.
  inputs.changePattern(decIdx);
  inputs.setIndex(0);
  temporal_pooling_test_utils::runSteps(time_step,
                                        /*num_steps=*/240,
                                        [&](int t, const std::vector<int>& in) { htm.step(t, in); },
                                        [&]() { return inputs.next(htm.rng()); });
  time_step += 240;
  const std::vector<uint8_t> repDEC = runAndCaptureOneCycleTopLearn(htm, inputs, time_step, len);

  const double sim = similarityPercent(repABC, repDEC);
  EXPECT_GE(sim, 0.10) << "ABC and DEC share one element; expect some similarity";
  EXPECT_LE(sim, 0.90) << "ABC and DEC are not identical; expect similarity to be bounded";
}

