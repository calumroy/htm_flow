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
  // Build one "representative" SDR for the whole sequence by aggregating the TOP layer output
  // across one full cycle.
  //
  // Why do we need this?
  // - Suite3's patterns are SEQUENCES (e.g. A->B->C), not single static images.
  // - Temporal pooling is about learning a stable representation over the sequence.
  // - So we need a single summary representation for comparisons like:
  //     pattern X at time T0  vs  pattern X at time T1     (recall/stability)
  //     pattern X             vs  pattern Y               (separation)
  //
  // Why majority vote here?
  // - For these tiny sequence lengths (3 steps), majority vote tends to keep cells that
  //   are consistently involved across the sequence and drop one-off burst artifacts.
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
  Provenance:
  - Python: HTM/tests/temporalPooling/test_temporalPoolingSuite3.py::test_tempEquality

  What Suite3 is about (conceptually):
  - The original Python suite feeds IMAGE sequences (read from disk) into a multi-layer HTM and checks:
    (1) each sequence becomes temporally pooled (stable-ish top representation),
    (2) different sequences pool to different top representations,
    (3) switching away and switching back recalls the earlier pooled representation.

  C++ adaptation details (so you do NOT need to read Python to understand the test):
  - We do not use image files; we hardcode small 8x8 binary patterns that act like "letters".
  - We form sequences DEF and ABC (3 steps each).
  - We approximate a "higher layer" by using a 2-layer feed-forward hierarchy:
      layer0(active columns) -> layer1(input grid)
    and we measure layer1 learn-cells output as the pooled representation.

  What we measure and why it maps to the intent:
  - We compute a representative SDR for each pattern by running one full cycle and aggregating
    layer1 learn-cells outputs.
  - We compare representatives using similarityPercent = |A ∩ B| / |A|.
    - recall: repDEF(before) should overlap repDEF(after)
    - separation: repDEF should not overlap repABC too much

  Why the numeric thresholds are "moderate" (not super tight):
  - These are integration tests over a small deterministic configuration, not a tuned benchmark.
  - The goal is to catch regressions where outputs collapse or everything becomes identical,
    without making CI brittle.
  */

  // Step-by-step:
  // 1) Configure a small deterministic 2-layer hierarchy suitable for 8x8 inputs.
  //    (pot_h/pot_w must be <= input dims to avoid invalid neighborhoods)
  // 2) Train on DEF and capture representative repDEF1.
  // 3) Train on ABC and capture representative repABC1.
  // 4) Switch back to DEF, train more, capture repDEF2 (recall).
  // 5) Switch back to ABC, train more, capture repABC2 (recall).
  // 6) Assert: DEF recalls itself, ABC recalls itself, and DEF != ABC.

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
  Provenance:
  - Python: HTM/tests/temporalPooling/test_temporalPoolingSuite3.py::test_temporalDiff

  What we are testing:
  - Two sequences that share ONE element should have pooled representations that are:
    - not identical (because the sequences differ),
    - not completely unrelated (because they share some structure).

  How we set that up in C++:
  - We build two 3-step sequences:
      ABC and DEC
    which share the final element C.
  - After training each sequence, we capture a representative top-layer SDR for each.
  - We then assert the similarity is bounded away from both extremes.

  Why this is a reasonable proxy:
  - If the temporal pooling stage is doing anything meaningful, shared input structure tends
    to produce some shared higher-level structure (some overlap), but not total collapse.
  */

  // Step-by-step:
  // 1) Configure the same small deterministic 2-layer hierarchy for 8x8 inputs.
  // 2) Train on ABC; capture representative repABC.
  // 3) Train on DEC; capture representative repDEC.
  // 4) Assert similarity(repABC, repDEC) is neither ~0 nor ~1.

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

