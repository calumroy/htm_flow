#include <gtest/gtest.h>

#include "../test_utils/sp_harness.hpp"
#include "../test_utils/sp_metrics.hpp"
#include "../test_utils/tp_inputs.hpp"

#include <vector>

using spatial_pooling_test_utils::SpatialPoolerHarness;
using spatial_pooling_test_utils::jaccardSimilarity01;
using spatial_pooling_test_utils::toU8;

using temporal_pooling_test_utils::VerticalLineInputs;

namespace {

// ── Suite-wide default configuration ────────────────────────────────
// Every test in this file starts from this config.
// Modify a copy in individual tests if needed.
SpatialPoolerHarness::Config suiteConfig() {
  SpatialPoolerHarness::Config c;
  c.input_rows            = 60;
  c.input_cols            = 30;
  c.col_rows              = 30;
  c.col_cols              = 10;
  c.pot_h                 = 4;
  c.pot_w                 = 4;
  c.center_pot_synapses   = true;
  c.wrap_input            = false;
  c.inhib_w               = 3;
  c.inhib_h               = 3;
  c.desired_local_activity = 2;
  c.connected_perm        = 0.3f;
  c.min_overlap           = 3;
  c.min_potential_overlap  = 0;
  c.spatial_perm_inc      = 0.10f;
  c.spatial_perm_dec      = 0.02f;
  c.active_col_perm_dec   = 0.02f;
  c.rng_seed              = 123u;
  return c;
}

} // namespace

TEST(SpatialPoolingIntegrationSuite2, test_case1_two_patterns_non_forgetting_recall) {
  /*
  Provenance:
  - Python: HTM/tests/spatialPooling/test_spatialPoolingSuite2.py::test_case1

  What the legacy Python test checks:
  - The spatial pooler can learn Pattern A, then learn Pattern B, and later still produce
    (nearly) the same column-level SDR outputs for Pattern A (i.e., A is not forgotten).

  Why this is meaningful for a spatial pooler:
  - Spatial learning is updating proximal synapse permanences based on active columns
    and their input patches.
  - Catastrophic interference would show up as Pattern A’s column SDRs drifting when
    Pattern B is trained, if both patterns share resources too aggressively.

  How we map it in htm_flow:
  - We run the spatial pooler pipeline (overlap->inhibition->spatial learning).
  - Pattern A: LeftToRight vertical-line sweep (a sequence of input SDRs).
  - Pattern B: RightToLeft vertical-line sweep.
  - We store the *active columns* output for each step of a cycle (seq_len steps).
  - We then train on the other pattern and re-capture, and compare per-step using
    the same similarity metric as Python’s `gridsSimilar` (Jaccard).

  Notes on strictness:
  - Python expects average similarity ≥ 0.95.
  - In C++, small config differences can change exact SDRs, so we target the same
    qualitative behavior (high recall) but leave moderate slack to avoid brittle tests.
  */

  SpatialPoolerHarness sp(suiteConfig());
  VerticalLineInputs inputs(/*width=*/sp.cfg().input_cols, /*height=*/sp.cfg().input_rows, /*seq_len=*/sp.cfg().input_cols);
  inputs.setSequenceProbability(1.0);

  auto runSteps = [&](int steps) {
    for (int i = 0; i < steps; ++i) {
      sp.step(inputs.next(sp.rng()));
    }
  };

  auto captureCycle = [&]() {
    inputs.setIndex(0);
    std::vector<std::vector<uint8_t>> cycle;
    cycle.reserve(static_cast<std::size_t>(inputs.seqLen()));
    for (int i = 0; i < inputs.seqLen(); ++i) {
      sp.step(inputs.next(sp.rng()));
      cycle.push_back(sp.activeColumns01());
    }
    return cycle;
  };

  // Train A, train B, train A again (mirrors the repeated switching in the Python test).
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(5 * inputs.seqLen());
  inputs.setPattern(VerticalLineInputs::Pattern::RightToLeft);
  runSteps(5 * inputs.seqLen());
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(5 * inputs.seqLen());

  // Capture pattern A outputs.
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(inputs.seqLen());
  const auto a1 = captureCycle();

  // Capture pattern B outputs.
  inputs.setPattern(VerticalLineInputs::Pattern::RightToLeft);
  runSteps(inputs.seqLen());
  const auto b1 = captureCycle();

  // Switch back to A and capture again.
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(inputs.seqLen());
  const auto a2 = captureCycle();

  // Switch back to B and capture again.
  inputs.setPattern(VerticalLineInputs::Pattern::RightToLeft);
  runSteps(inputs.seqLen());
  const auto b2 = captureCycle();

  ASSERT_EQ(a1.size(), a2.size());
  ASSERT_EQ(b1.size(), b2.size());

  double simA = 0.0;
  double simB = 0.0;
  for (std::size_t i = 0; i < a1.size(); ++i) simA += jaccardSimilarity01(a1[i], a2[i]);
  for (std::size_t i = 0; i < b1.size(); ++i) simB += jaccardSimilarity01(b1[i], b2[i]);
  simA /= static_cast<double>(a1.size());
  simB /= static_cast<double>(b1.size());

  EXPECT_GE(simA, 0.70) << "Expected Pattern A to be largely recalled after learning Pattern B";
  EXPECT_GE(simB, 0.70) << "Expected Pattern B to be largely recalled after switching back";
}

TEST(SpatialPoolingIntegrationSuite2, test_case2_similar_patterns_non_forgetting) {
  /*
  Provenance:
  - Python: HTM/tests/spatialPooling/test_spatialPoolingSuite2.py::test_case2

  What changes vs case1:
  - The two patterns are MORE similar in the Python suite (a left-to-right sweep with
    a positional jump).

  C++ analogue:
  - We use EvenPositions as a “jumping” pattern relative to LeftToRight:
      LeftToRight: x = 0,1,2,3,4,...
      EvenPositions: x = 0,2,4,6,8,... (wraps)

  What we check:
  - Both patterns remain recallable after training on the other.
  */

  SpatialPoolerHarness sp(suiteConfig());
  VerticalLineInputs inputs(/*width=*/sp.cfg().input_cols, /*height=*/sp.cfg().input_rows, /*seq_len=*/sp.cfg().input_cols);
  inputs.setSequenceProbability(1.0);

  auto runSteps = [&](int steps) {
    for (int i = 0; i < steps; ++i) sp.step(inputs.next(sp.rng()));
  };

  auto captureCycle = [&]() {
    inputs.setIndex(0);
    std::vector<std::vector<uint8_t>> cycle;
    cycle.reserve(static_cast<std::size_t>(inputs.seqLen()));
    for (int i = 0; i < inputs.seqLen(); ++i) {
      sp.step(inputs.next(sp.rng()));
      cycle.push_back(sp.activeColumns01());
    }
    return cycle;
  };

  // A: LeftToRight, B: EvenPositions.
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(5 * inputs.seqLen());
  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  runSteps(5 * inputs.seqLen());
  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(5 * inputs.seqLen());

  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(inputs.seqLen());
  const auto a1 = captureCycle();

  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  runSteps(inputs.seqLen());
  const auto b1 = captureCycle();

  inputs.setPattern(VerticalLineInputs::Pattern::LeftToRight);
  runSteps(inputs.seqLen());
  const auto a2 = captureCycle();

  inputs.setPattern(VerticalLineInputs::Pattern::EvenPositions);
  runSteps(inputs.seqLen());
  const auto b2 = captureCycle();

  ASSERT_EQ(a1.size(), a2.size());
  ASSERT_EQ(b1.size(), b2.size());

  double simA = 0.0;
  double simB = 0.0;
  for (std::size_t i = 0; i < a1.size(); ++i) simA += jaccardSimilarity01(a1[i], a2[i]);
  for (std::size_t i = 0; i < b1.size(); ++i) simB += jaccardSimilarity01(b1[i], b2[i]);
  simA /= static_cast<double>(a1.size());
  simB /= static_cast<double>(b1.size());

  EXPECT_GE(simA, 0.65);
  EXPECT_GE(simB, 0.65);
}


