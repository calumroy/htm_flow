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
  // What problem is this solving?
  // - Our "pattern" is not a single frame; it is a SEQUENCE over a cycle length (seq_len).
  // - The system output can vary slightly at each timestep (especially near pattern switches).
  // - We want ONE stable-ish summary SDR so we can do meaningful comparisons like:
  //     "pattern A now" vs "pattern A later"  (recall)
  //     "pattern A" vs "pattern B"            (differentiation)
  //
  // Why OR instead of majority-vote?
  // - These learn-cells outputs can be very sparse and bursty.
  // - Majority vote can collapse to all-zeros, which then makes overlap/similarity comparisons
  //   meaningless (everything looks identical).
  // - OR preserves the union of features that participate across the cycle, which is closer to
  //   the Python suite's intent of capturing the "temporally pooled pattern" for the whole sequence.
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
  Provenance (legacy behavior being ported):
  - Python: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py::test_case1

  What we are testing:
  - If we repeatedly present the SAME input sequence, the model should learn temporal structure and
    produce a representation that changes less over time (i.e., it "pools" over the sequence).

  How we measure "pools":
  - We use TemporalPoolingMeasure on the learn-cells output:
      temporalPoolingPercent = running average of |prev ∩ curr| / |prev|
    A higher value means successive outputs overlap more (more stability).

  Why learn-cells output:
  - It is the closest analogue to what the Python suite examines at the tested layer:
    a sparse set of cells representing the temporally pooled state (not raw input bits).
  */

  // Step-by-step:
  // 1) Warm up the model by replaying a deterministic repeating sequence.
  // 2) Run for a few more cycles and compute temporalPoolingPercent each step.
  // 3) Expect the running average to be "reasonably high".
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
  Provenance:
  - Python: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py::test_case4

  What we are testing:
  - Temporal pooling should increase with depth: higher-level representations should be at least as
    stable as lower-level representations (often more stable).

  Why a 2-layer hierarchy in C++:
  - The Python project has multiple layers inside a region.
  - htm_flow does not have an identical "region with N internal layers" concept, so we approximate
    depth using two full pipelines in series:
      layer0(active columns) -> layer1(input grid)

  What we measure:
  - We compute the temporal pooling metric for learn-cells at layer0 and layer1 over the same
    evaluation window, and require layer1 to not be less stable than layer0 (within a small tolerance).
  */

  // Step-by-step:
  // 1) Build a deterministic two-layer feed-forward stack.
  // 2) Warm up on a repeating sequence long enough for pooling to emerge.
  // 3) During evaluation, track pooling in both layers and compare.
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
  Provenance:
  - Python: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py::test_multiPattern

  What we are testing:
  - The model can learn temporally pooled representations for MULTIPLE patterns.
  - Those representations remain distinct (pattern A ≠ pattern B).
  - If we switch away and later switch back, the model "recalls" the previous pooled state for that pattern.

  What is a "pattern" here?
  - Each pattern is a repeating SEQUENCE of vertical-line inputs over a fixed cycle length.
  - Pattern A: LeftToRight (line sweeps left->right)
  - Pattern B: RightToLeft (line sweeps right->left)

  What representation do we compare?
  - We compare a representative SDR of the learn-cells output aggregated over ONE cycle.
  - This is a practical analogue to the Python suite's approach of storing per-input SDRs and
    comparing "same pattern at different times" vs "different patterns".

  What does similarityPercent mean?
  - similarityPercent(A,B) = |A ∩ B| / |A|
  - We use it for:
    - recall: A1 vs A2 should overlap (non-zero overlap is a minimum sanity check)
    - differentiation: A vs B should not be almost identical

  Important note about strictness:
  - These integration tests run a small, deterministic configuration.
  - Exact similarity thresholds can be brittle across parameter changes.
  - So we assert the *qualitative* behaviors (non-degenerate recall + non-collapse) rather than
    locking to very tight numeric values.
  */

  // Step-by-step:
  // 1) Train on Pattern A long enough for a pooled representation to form; capture representative A1.
  // 2) Train on Pattern B long enough for a pooled representation to form; capture representative B1.
  // 3) Switch back to A briefly; capture representative A2 (recall).
  // 4) Switch back to B briefly; capture representative B2 (recall).
  // 5) Assert: A1 overlaps A2, B1 overlaps B2, and A1 is not almost identical to B1.

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

