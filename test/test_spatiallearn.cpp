/*
 * Test for SpatialLearnCalculator
 *
 * This test verifies the correct behavior of the SpatialLearnCalculator class,
 * which is responsible for updating synapse permanence values based on column
 * activity and input patterns.
 *
 * It covers:
 * 1. Newly active columns: Synapses connected to active inputs are incremented,
 * others decremented.
 * 2. Previously active columns with CHANGED inputs: Active inputs increment,
 * inactive inputs decrement by a specific activeColPermanenceDec.
 * 3. Previously active columns with SAME inputs: No changes to permanence.
 * 4. Inactive columns: No changes.
 * 5. Value clamping: Permanence values are kept between 0.0 and 1.0.
 */
#include <gtest/gtest.h>
#include <htm_flow/spatiallearn.hpp>
#include <iostream>
#include <vector>

using namespace spatiallearn;

TEST(SpatialLearnCalculator, calculate_spatiallearn) {
  int numColumns = 4;
  int numPotSynapses = 4;
  float spatialPermanenceInc = 0.05f;
  float spatialPermanenceDec = 0.05f;
  float activeColPermanenceDec = 0.01f;

  SpatialLearnCalculator calculator(numColumns, numPotSynapses,
                                    spatialPermanenceInc, spatialPermanenceDec,
                                    activeColPermanenceDec);

  // Initial state (flattened: numColumns x numPotSynapses)
  std::vector<float> colSynPerm(numColumns * numPotSynapses, 0.5f);
  const std::pair<int, int> colSynPermShape = {numColumns, numPotSynapses};

  // --- Step 1: Activate Column 0 and 2 ---
  // Column 0: Inputs {1, 1, 0, 0}
  // Column 2: Inputs {0, 0, 1, 1}
  std::vector<int> colPotInputs = {
      1, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 1, 1,
      0, 0, 0, 0
  };
  const std::pair<int, int> colPotInputsShape = {numColumns, numPotSynapses};
  std::vector<int> activeColIndices = {0, 2};

  calculator.calculate_spatiallearn_1d_active_indices(
      colSynPerm, colSynPermShape,
      colPotInputs, colPotInputsShape,
      activeColIndices);

  // Expected:
  // Column 0: Newly active. Inputs match {1, 1, 0, 0}.
  //   Syn 0, 1 (input 1) += inc (0.05) -> 0.55
  //   Syn 2, 3 (input 0) -= dec (0.05) -> 0.45
  // Column 1: Inactive. No change -> 0.5
  // Column 2: Newly active. Inputs match {0, 0, 1, 1}.
  //   Syn 0, 1 (input 0) -= dec (0.05) -> 0.45
  //   Syn 2, 3 (input 1) += inc (0.05) -> 0.55
  // Column 3: Inactive. No change -> 0.5

  // Column 0 base = 0
  EXPECT_FLOAT_EQ(colSynPerm[0], 0.55f);
  EXPECT_FLOAT_EQ(colSynPerm[1], 0.55f);
  EXPECT_FLOAT_EQ(colSynPerm[2], 0.45f);
  EXPECT_FLOAT_EQ(colSynPerm[3], 0.45f);

  // Column 1 base = 4
  EXPECT_FLOAT_EQ(colSynPerm[4], 0.5f);

  // Column 2 base = 8
  EXPECT_FLOAT_EQ(colSynPerm[8], 0.45f);
  EXPECT_FLOAT_EQ(colSynPerm[9], 0.45f);
  EXPECT_FLOAT_EQ(colSynPerm[10], 0.55f);
  EXPECT_FLOAT_EQ(colSynPerm[11], 0.55f);

  // --- Step 2: Column 0 stays active, inputs change ---
  // Column 0: Active previously (was {1, 1, 0, 0}, now {1, 0, 1, 0})
  //   Logic: If prevActive is true AND inputs changed => update
  //   Syn 0 (input 1) += inc (0.05) -> 0.60
  //   Syn 1 (input 0) -= activeDec (0.01) -> 0.54 (Note: NOT spatialDec)
  //   Syn 2 (input 1) += inc (0.05) -> 0.50
  //   Syn 3 (input 0) -= activeDec (0.01) -> 0.44

  // Column 2 stays active, inputs SAME {0, 0, 1, 1}
  //   Logic: prevActive is true AND inputs SAME => NO update

  colPotInputs = {
      1, 0, 1, 0,  // Changed
      0, 0, 0, 0,
      0, 0, 1, 1,  // Same
      0, 0, 0, 0
  };
  activeColIndices = {0, 2};

  calculator.calculate_spatiallearn_1d_active_indices(
      colSynPerm, colSynPermShape,
      colPotInputs, colPotInputsShape,
      activeColIndices);

  // Column 0
  EXPECT_FLOAT_EQ(colSynPerm[0], 0.60f); // 0.55 + 0.05
  EXPECT_FLOAT_EQ(colSynPerm[1], 0.54f); // 0.55 - 0.01
  EXPECT_FLOAT_EQ(colSynPerm[2], 0.50f); // 0.45 + 0.05
  EXPECT_FLOAT_EQ(colSynPerm[3], 0.44f); // 0.45 - 0.01

  // Column 2 should be unchanged from step 1
  EXPECT_FLOAT_EQ(colSynPerm[8], 0.45f);
  EXPECT_FLOAT_EQ(colSynPerm[9], 0.45f);
  EXPECT_FLOAT_EQ(colSynPerm[10], 0.55f);
  EXPECT_FLOAT_EQ(colSynPerm[11], 0.55f);

  // --- Step 3: Limits check ---
  // Set column 1 very close to 1.0 and 0.0 and activate it
  colSynPerm[4] = 0.99f;
  colSynPerm[5] = 0.01f;
  colSynPerm[6] = 0.5f;
  colSynPerm[7] = 0.5f;

  colPotInputs[4] = 1;
  colPotInputs[5] = 0;
  colPotInputs[6] = 0;
  colPotInputs[7] = 0;
  activeColIndices = {1};

  // Calc
  calculator.calculate_spatiallearn_1d_active_indices(
      colSynPerm, colSynPermShape,
      colPotInputs, colPotInputsShape,
      activeColIndices);

  // Syn 0: 0.99 + 0.05 = 1.04 -> clamped to 1.0
  // Syn 1: 0.01 - 0.05 = -0.04 -> clamped to 0.0
  EXPECT_FLOAT_EQ(colSynPerm[4], 1.0f);
  EXPECT_FLOAT_EQ(colSynPerm[5], 0.0f);
}
