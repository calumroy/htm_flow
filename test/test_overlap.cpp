#include <gtest/gtest.h>
#include <src/overlap/overlap.cpp>
#include <src/overlap/overlap_utils.cpp>

// Using the overlap namespace
using namespace overlap_utils;
TEST(Overlap, flatten_index)
{
    ASSERT_EQ(3, flatten_index(1, 1, 2));
    ASSERT_EQ(11, flatten_index(3, 2, 4));
    ASSERT_EQ(7, flatten_index(7, 3, 0));
}

TEST(Images2NeibsTest, test1)
{
    // Test 1: Check that a 3x3 patch is extracted from a 5x5 matrix
    // No wrapping around the edges so padding of zero is used instead.
    std::vector<std::vector<int>> input = {{1, 2, 3, 4, 5},
                                           {6, 7, 8, 9, 10},
                                           {11, 12, 13, 14, 15},
                                           {16, 17, 18, 19, 20},
                                           {21, 22, 23, 24, 25}};
    std::pair<int, int> neib_shape = {3, 3};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = false;
    bool center_neigh = false;

    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    ASSERT_EQ(output.size(), 5);
    ASSERT_EQ(output[0].size(), 5);
    ASSERT_EQ(output[0][0].size(), 3);
    ASSERT_EQ(output[0][0][0].size(), 3);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{1, 2, 3}, {6, 7, 8}, {11, 12, 13}},
                                                                                {{2, 3, 4}, {7, 8, 9}, {12, 13, 14}},
                                                                                {{3, 4, 5}, {8, 9, 10}, {13, 14, 15}},
                                                                                {{4, 5, 0}, {9, 10, 0}, {14, 15, 0}},
                                                                                {{5, 0, 0}, {10, 0, 0}, {15, 0, 0}}},
                                                                               {{{6, 7, 8}, {11, 12, 13}, {16, 17, 18}},
                                                                                {{7, 8, 9}, {12, 13, 14}, {17, 18, 19}},
                                                                                {{8, 9, 10}, {13, 14, 15}, {18, 19, 20}},
                                                                                {{9, 10, 0}, {14, 15, 0}, {19, 20, 0}},
                                                                                {{10, 0, 0}, {15, 0, 0}, {20, 0, 0}}},
                                                                               {{{11, 12, 13}, {16, 17, 18}, {21, 22, 23}},
                                                                                {{12, 13, 14}, {17, 18, 19}, {22, 23, 24}},
                                                                                {{13, 14, 15}, {18, 19, 20}, {23, 24, 25}},
                                                                                {{14, 15, 0}, {19, 20, 0}, {24, 25, 0}},
                                                                                {{15, 0, 0}, {20, 0, 0}, {25, 0, 0}}},
                                                                               {{{16, 17, 18}, {21, 22, 23}, {0, 0, 0}},
                                                                                {{17, 18, 19}, {22, 23, 24}, {0, 0, 0}},
                                                                                {{18, 19, 20}, {23, 24, 25}, {0, 0, 0}},
                                                                                {{19, 20, 0}, {24, 25, 0}, {0, 0, 0}},
                                                                                {{20, 0, 0}, {25, 0, 0}, {0, 0, 0}}},
                                                                               {{{21, 22, 23}, {0, 0, 0}, {0, 0, 0}},
                                                                                {{22, 23, 24}, {0, 0, 0}, {0, 0, 0}},
                                                                                {{23, 24, 25}, {0, 0, 0}, {0, 0, 0}},
                                                                                {{24, 25, 0}, {0, 0, 0}, {0, 0, 0}},
                                                                                {{25, 0, 0}, {0, 0, 0}, {0, 0, 0}}}};
    ASSERT_EQ(output, expected_output);
}

TEST(Images2NeibsTest, test2_wrap)
{
    // Test 2: Check that a 2x2 patch is extracted from a 3x3 matrix.
    // Wrapping is used.
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3},
                                           {4, 5, 6},
                                           {7, 8, 9}};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = false;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{1, 2}, {4, 5}},
                                                                                {{2, 3}, {5, 6}},
                                                                                {{3, 1}, {6, 4}}},
                                                                               {{{4, 5}, {7, 8}},
                                                                                {{5, 6}, {8, 9}},
                                                                                {{6, 4}, {9, 7}}},
                                                                               {{{7, 8}, {1, 2}},
                                                                                {{8, 9}, {2, 3}},
                                                                                {{9, 7}, {3, 1}}}};
    ASSERT_EQ(output, expected_output);
}

TEST(Images2NeibsTest, test3_wrap_center)
{
    // Check that a 2x2 patch is extracted from a 3x3 matrix.
    // Wrapping is used and the centering of the patch over each input element is used.
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3},
                                           {4, 5, 6},
                                           {7, 8, 9}};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{9, 7}, {3, 1}},
                                                                                {{7, 8}, {1, 2}},
                                                                                {{8, 9}, {2, 3}}},
                                                                               {{{3, 1}, {6, 4}},
                                                                                {{1, 2}, {4, 5}},
                                                                                {{2, 3}, {5, 6}}},
                                                                               {{{6, 4}, {9, 7}},
                                                                                {{4, 5}, {7, 8}},
                                                                                {{5, 6}, {8, 9}}}};
    ASSERT_EQ(output, expected_output);
}

TEST(Images2NeibsTest, test4_wrap_center)
{
    // Check that a 3x3 patch is extracted from a 5x5 matrix
    // Wrapping is used and the centering of the patch over each input element is used.
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3, 4, 5},
                                           {6, 7, 8, 9, 10},
                                           {11, 12, 13, 14, 15},
                                           {16, 17, 18, 19, 20},
                                           {21, 22, 23, 24, 25}};
    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {3, 3};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{25, 21, 22}, {5, 1, 2}, {10, 6, 7}},
                                                                                {{21, 22, 23}, {1, 2, 3}, {6, 7, 8}},
                                                                                {{22, 23, 24}, {2, 3, 4}, {7, 8, 9}},
                                                                                {{23, 24, 25}, {3, 4, 5}, {8, 9, 10}},
                                                                                {{24, 25, 21}, {4, 5, 1}, {9, 10, 6}}},
                                                                               {{{5, 1, 2}, {10, 6, 7}, {15, 11, 12}},
                                                                                {{1, 2, 3}, {6, 7, 8}, {11, 12, 13}},
                                                                                {{2, 3, 4}, {7, 8, 9}, {12, 13, 14}},
                                                                                {{3, 4, 5}, {8, 9, 10}, {13, 14, 15}},
                                                                                {{4, 5, 1}, {9, 10, 6}, {14, 15, 11}}},
                                                                               {{{10, 6, 7}, {15, 11, 12}, {20, 16, 17}},
                                                                                {{6, 7, 8}, {11, 12, 13}, {16, 17, 18}},
                                                                                {{7, 8, 9}, {12, 13, 14}, {17, 18, 19}},
                                                                                {{8, 9, 10}, {13, 14, 15}, {18, 19, 20}},
                                                                                {{9, 10, 6}, {14, 15, 11}, {19, 20, 16}}},
                                                                               {{{15, 11, 12}, {20, 16, 17}, {25, 21, 22}},
                                                                                {{11, 12, 13}, {16, 17, 18}, {21, 22, 23}},
                                                                                {{12, 13, 14}, {17, 18, 19}, {22, 23, 24}},
                                                                                {{13, 14, 15}, {18, 19, 20}, {23, 24, 25}},
                                                                                {{14, 15, 11}, {19, 20, 16}, {24, 25, 21}}},
                                                                               {{{20, 16, 17}, {25, 21, 22}, {5, 1, 2}},
                                                                                {{16, 17, 18}, {21, 22, 23}, {1, 2, 3}},
                                                                                {{17, 18, 19}, {22, 23, 24}, {2, 3, 4}},
                                                                                {{18, 19, 20}, {23, 24, 25}, {3, 4, 5}},
                                                                                {{19, 20, 16}, {24, 25, 21}, {4, 5, 1}}}};

    ASSERT_EQ(output, expected_output);
}

TEST(Images2Neibs, test5_wrap_step1_2)
{
    // Check that a 2x2 patch is extracted from a 3x3 matrix.
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3},
                                           {4, 5, 6},
                                           {7, 8, 9}};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {2, 1};
    bool wrap_mode = true;
    bool center_neigh = false;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{1, 2}, {4, 5}},
                                                                                {{2, 3}, {5, 6}},
                                                                                {{3, 1}, {6, 4}}},
                                                                               {{{7, 8}, {1, 2}},
                                                                                {{8, 9}, {2, 3}},
                                                                                {{9, 7}, {3, 1}}}};

    // Check that the function produced the expected output
    EXPECT_EQ(output, expected_output);
}

TEST(Images2Neibs, test6_large)
{
    // Check that a 40x40 patch is extracted from a 400x400 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing (400x400) every element is 1
    int n_rows = 400;
    int n_cols = 400;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {40, 40};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Assert the output is the correct size
    ASSERT_EQ(output.size(), n_rows);
    ASSERT_EQ(output[0].size(), n_cols);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(parallel_Images2Neibs, test1_wrap)
{
    // Test 1: Check that a 2x2 patch is extracted from a 3x3 matrix
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3},
                                           {4, 5, 6},
                                           {7, 8, 9}};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = false;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = parallel_Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{1, 2}, {4, 5}},
                                                                                {{2, 3}, {5, 6}},
                                                                                {{3, 1}, {6, 4}}},
                                                                               {{{4, 5}, {7, 8}},
                                                                                {{5, 6}, {8, 9}},
                                                                                {{6, 4}, {9, 7}}},
                                                                               {{{7, 8}, {1, 2}},
                                                                                {{8, 9}, {2, 3}},
                                                                                {{9, 7}, {3, 1}}}};

    // Check that the function produced the expected output
    EXPECT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs, test2_wrap)
{
    // Check that a 3x3 patch is extracted from a 5x5 matrix
    // Apply wrapping around the edges.
    std::vector<std::vector<int>> input = {{1, 2, 3, 4, 5},
                                           {6, 7, 8, 9, 10},
                                           {11, 12, 13, 14, 15},
                                           {16, 17, 18, 19, 20},
                                           {21, 22, 23, 24, 25}};
    std::pair<int, int> neib_shape = {3, 3};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = false;

    std::vector<std::vector<std::vector<std::vector<int>>>> output = parallel_Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{1, 2, 3}, {6, 7, 8}, {11, 12, 13}},
                                                                                {{2, 3, 4}, {7, 8, 9}, {12, 13, 14}},
                                                                                {{3, 4, 5}, {8, 9, 10}, {13, 14, 15}},
                                                                                {{4, 5, 1}, {9, 10, 6}, {14, 15, 11}},
                                                                                {{5, 1, 2}, {10, 6, 7}, {15, 11, 12}}},
                                                                               {{{6, 7, 8}, {11, 12, 13}, {16, 17, 18}},
                                                                                {{7, 8, 9}, {12, 13, 14}, {17, 18, 19}},
                                                                                {{8, 9, 10}, {13, 14, 15}, {18, 19, 20}},
                                                                                {{9, 10, 6}, {14, 15, 11}, {19, 20, 16}},
                                                                                {{10, 6, 7}, {15, 11, 12}, {20, 16, 17}}},
                                                                               {{{11, 12, 13}, {16, 17, 18}, {21, 22, 23}},
                                                                                {{12, 13, 14}, {17, 18, 19}, {22, 23, 24}},
                                                                                {{13, 14, 15}, {18, 19, 20}, {23, 24, 25}},
                                                                                {{14, 15, 11}, {19, 20, 16}, {24, 25, 21}},
                                                                                {{15, 11, 12}, {20, 16, 17}, {25, 21, 22}}},
                                                                               {{{16, 17, 18}, {21, 22, 23}, {1, 2, 3}},
                                                                                {{17, 18, 19}, {22, 23, 24}, {2, 3, 4}},
                                                                                {{18, 19, 20}, {23, 24, 25}, {3, 4, 5}},
                                                                                {{19, 20, 16}, {24, 25, 21}, {4, 5, 1}},
                                                                                {{20, 16, 17}, {25, 21, 22}, {5, 1, 2}}},
                                                                               {{{21, 22, 23}, {1, 2, 3}, {6, 7, 8}},
                                                                                {{22, 23, 24}, {2, 3, 4}, {7, 8, 9}},
                                                                                {{23, 24, 25}, {3, 4, 5}, {8, 9, 10}},
                                                                                {{24, 25, 21}, {4, 5, 1}, {9, 10, 6}},
                                                                                {{25, 21, 22}, {5, 1, 2}, {10, 6, 7}}}};
    // Check that the function produced the expected output
    EXPECT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs, test3_wrap_center)
{
    // Check that a 3x3 patch is extracted from a 5x5 matrix
    // Wrapping is used and the centering of the patch over each input element is used.
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3, 4, 5},
                                           {6, 7, 8, 9, 10},
                                           {11, 12, 13, 14, 15},
                                           {16, 17, 18, 19, 20},
                                           {21, 22, 23, 24, 25}};
    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {3, 3};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = parallel_Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{25, 21, 22}, {5, 1, 2}, {10, 6, 7}},
                                                                                {{21, 22, 23}, {1, 2, 3}, {6, 7, 8}},
                                                                                {{22, 23, 24}, {2, 3, 4}, {7, 8, 9}},
                                                                                {{23, 24, 25}, {3, 4, 5}, {8, 9, 10}},
                                                                                {{24, 25, 21}, {4, 5, 1}, {9, 10, 6}}},
                                                                               {{{5, 1, 2}, {10, 6, 7}, {15, 11, 12}},
                                                                                {{1, 2, 3}, {6, 7, 8}, {11, 12, 13}},
                                                                                {{2, 3, 4}, {7, 8, 9}, {12, 13, 14}},
                                                                                {{3, 4, 5}, {8, 9, 10}, {13, 14, 15}},
                                                                                {{4, 5, 1}, {9, 10, 6}, {14, 15, 11}}},
                                                                               {{{10, 6, 7}, {15, 11, 12}, {20, 16, 17}},
                                                                                {{6, 7, 8}, {11, 12, 13}, {16, 17, 18}},
                                                                                {{7, 8, 9}, {12, 13, 14}, {17, 18, 19}},
                                                                                {{8, 9, 10}, {13, 14, 15}, {18, 19, 20}},
                                                                                {{9, 10, 6}, {14, 15, 11}, {19, 20, 16}}},
                                                                               {{{15, 11, 12}, {20, 16, 17}, {25, 21, 22}},
                                                                                {{11, 12, 13}, {16, 17, 18}, {21, 22, 23}},
                                                                                {{12, 13, 14}, {17, 18, 19}, {22, 23, 24}},
                                                                                {{13, 14, 15}, {18, 19, 20}, {23, 24, 25}},
                                                                                {{14, 15, 11}, {19, 20, 16}, {24, 25, 21}}},
                                                                               {{{20, 16, 17}, {25, 21, 22}, {5, 1, 2}},
                                                                                {{16, 17, 18}, {21, 22, 23}, {1, 2, 3}},
                                                                                {{17, 18, 19}, {22, 23, 24}, {2, 3, 4}},
                                                                                {{18, 19, 20}, {23, 24, 25}, {3, 4, 5}},
                                                                                {{19, 20, 16}, {24, 25, 21}, {4, 5, 1}}}};

    ASSERT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs, test4_wrap_step1_2)
{
    // Check that a 2x2 patch is extracted from a 3x3 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3},
                                           {4, 5, 6},
                                           {7, 8, 9}};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {2, 1};
    bool wrap_mode = true;
    bool center_neigh = false;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = parallel_Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Create the expected output
    std::vector<std::vector<std::vector<std::vector<int>>>> expected_output = {{{{1, 2}, {4, 5}},
                                                                                {{2, 3}, {5, 6}},
                                                                                {{3, 1}, {6, 4}}},
                                                                               {{{7, 8}, {1, 2}},
                                                                                {{8, 9}, {2, 3}},
                                                                                {{9, 7}, {3, 1}}}};

    // Check that the function produced the expected output
    EXPECT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs, test5_large)
{
    // Check that a 40x40 patch is extracted from a 400x400 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing (400x400) every element is 1
    int n_rows = 400;
    int n_cols = 400;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {40, 40};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = parallel_Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Assert the output is the correct size
    ASSERT_EQ(output.size(), n_rows);
    ASSERT_EQ(output[0].size(), n_cols);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(parallel_Images2Neibs, test6_very_large)
{
    // Check that a 40x40 patch is extracted from a 800x800 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing (800x800) every element is 1
    int n_rows = 800;
    int n_cols = 800;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {40, 40};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // Run the function and save the output
    std::vector<std::vector<std::vector<std::vector<int>>>> output = parallel_Images2Neibs(input, neib_shape, neib_step, wrap_mode, center_neigh);

    // Assert the output is the correct size
    ASSERT_EQ(output.size(), n_rows);
    ASSERT_EQ(output[0].size(), n_cols);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(parallel_Images2Neibs_1D, test1_small)
{

    // This simulates the 2D vector input
    // {{1, 2, 3},
    //  {4, 5, 6},
    //  {7, 8, 9}};
    // Create the input vector
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Define the input shape
    std::pair<int, int> input_shape = std::make_pair(3, 3);

    // Define the neighbourhood shape and step
    std::pair<int, int> neib_shape = std::make_pair(2, 2);
    std::pair<int, int> neib_step = std::make_pair(1, 1);

    // Set the wrap_mode and center_neigh flags
    bool wrap_mode = true;
    bool center_neigh = false;

    // This simulates the 4D vector output
    // {{{{1, 2}, {4, 5}},
    //   {{2, 3}, {5, 6}},
    //   {{3, 1}, {6, 4}}},
    //  {{{4, 5}, {7, 8}},
    //   {{5, 6}, {8, 9}},
    //   {{6, 4}, {9, 7}}},
    //  {{{7, 8}, {1, 2}},
    //   {{8, 9}, {2, 3}},
    //   {{9, 7}, {3, 1}}}};

    // Define the expected output vector
    std::vector<int> expected_output = {1, 2, 4, 5, 2, 3, 5, 6, 3, 1, 6, 4, 4, 5, 7, 8, 5, 6, 8, 9, 6, 4, 9, 7, 7, 8, 1, 2, 8, 9, 2, 3, 9, 7, 3, 1};

    // Define the output shape
    const int output_rows = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));
    const int output_cols = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second));
    const int output_channels = neib_shape.first * neib_shape.second;
    const int output_size = output_rows * output_cols * output_channels;

    std::vector<int> output_shape = {output_rows,
                                     output_cols,
                                     neib_shape.first,
                                     neib_shape.second};

    // Create the output vector and initialize it with zeros
    std::vector<int> output(output_size, 0);

    tf::Taskflow taskflow;

    // Call the parallel_Images2Neibs_1D function
    parallel_Images2Neibs_1D(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Create a taskflow executor to run the above taskflow
    tf::Executor executor;
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs_1D, test2_nowrap_center)
{

    // This simulates the 2D vector input
    // {{1, 2, 3},
    //  {4, 5, 6},
    //  {7, 8, 9}};
    // Create the input vector
    const std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Define the input shape
    const std::pair<int, int> input_shape = std::make_pair(3, 3);

    // Define the neighbourhood shape and step
    const std::pair<int, int> neib_shape = std::make_pair(2, 2);
    const std::pair<int, int> neib_step = std::make_pair(1, 1);

    // Set the wrap_mode and center_neigh flags
    bool wrap_mode = false;
    bool center_neigh = true;

    // This simulates the 4D vector output
    // {{{{0, 0}, {0, 1}},
    //   {{0, 0}, {1, 2}}
    //   {{0, 0}, {2, 3}}},
    //  {{{0, 1}, {0, 4}},
    //   {{1, 2}, {4, 5}},
    //   {{2, 3}, {5, 6}}},
    //  {{{0, 4}, {0, 7}},
    //   {{4, 5}, {7, 8}},
    //   {{5, 6}, {8, 9}}}}

    // Define the expected output vector
    std::vector<int> expected_output = {0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 1, 0, 4, 1, 2, 4, 5, 2, 3, 5, 6, 0, 4, 0, 7, 4, 5, 7, 8, 5, 6, 8, 9};

    // Define the output shape
    const int output_rows = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));
    const int output_cols = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second));
    const int output_channels = neib_shape.first * neib_shape.second;
    const int output_size = output_rows * output_cols * output_channels;

    std::vector<int> output_shape = {output_rows,
                                     output_cols,
                                     neib_shape.first,
                                     neib_shape.second};

    // Create the output vector and initialize it with zeros
    std::vector<int> output(output_size, 0);

    tf::Taskflow taskflow;

    // Call the parallel_Images2Neibs_1D function
    parallel_Images2Neibs_1D(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Create a taskflow executor to run the above taskflow
    tf::Executor executor;
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs_1D, test3_asymmetrical)
{

    // This simulates the 2D vector input
    // {{1, 2, 3, 4},
    //  {5, 6, 7, 8},
    //  {9, 10, 11, 12}};
    // Create the input vector
    const std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Define the input shape
    const std::pair<int, int> input_shape = std::make_pair(3, 4);

    // Define the neighbourhood shape and step
    const std::pair<int, int> neib_shape = std::make_pair(2, 2);
    const std::pair<int, int> neib_step = std::make_pair(1, 1);

    // Set the wrap_mode and center_neigh flags
    bool wrap_mode = true;
    bool center_neigh = true;

    // This simulates the 4D vector output
    // {{{{12, 9}, {4, 1},
    //   {{9, 10}, {1, 2}},
    //   {{10, 11}, {2, 3},
    //   {{11, 12}, {3, 4}}},
    //  {{{4, 1}, {8, 5}},
    //   {{1, 2}, {5, 6}},
    //   {{2, 3}, {6, 7}},
    //   {{3, 4}, {7, 8}}},
    //  {{{8, 5}, {12, 9}},
    //   {{5, 6}, {9, 10}},
    //   {{6, 7}, {10, 11}},
    //   {{7, 8}, {11, 12}}}}

    // Define the expected output vector
    std::vector<int> expected_output = {12, 9, 4, 1, 9, 10, 1, 2, 10, 11, 2, 3, 11, 12, 3, 4, 4, 1, 8, 5, 1, 2, 5, 6, 2, 3, 6, 7, 3, 4, 7, 8, 8, 5, 12, 9, 5, 6, 9, 10, 6, 7, 10, 11, 7, 8, 11, 12};
    // Define the maximum output shape (the total number of convolutions that can be acheived with the given input size and step sizes)
    const int max_output_rows = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));
    const int max_output_cols = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second));

    // We want to limit the size of the output to be slightly smaller then the total number of convolutions that can be acheived with the given input size and step sizes.
    // This test a situations where we might want the output size to be smaller then the total number of possible convolutions.
    std::vector<int> output_shape = {max_output_rows,
                                     max_output_cols,
                                     neib_shape.first,
                                     neib_shape.second};

    const int output_channels = neib_shape.first * neib_shape.second;
    const int output_size = output_shape[0] * output_shape[1] * output_channels; // 3 * 4 * 4 = 48

    // Create the output vector and initialize it with zeros
    std::vector<int> output(output_size, 0);

    tf::Taskflow taskflow;
    // Call the parallel_Images2Neibs_1D function
    parallel_Images2Neibs_1D(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Create a taskflow executor to run the above taskflow
    tf::Executor executor;
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs_1D, test4_limit_output_size)
{

    // This simulates the 2D vector input
    // {{1, 2, 3, 4},
    //  {5, 6, 7, 8},
    //  {9, 10, 11, 12}};
    // Create the input vector
    const std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Define the input shape
    const std::pair<int, int> input_shape = std::make_pair(3, 4);

    // Define the neighbourhood shape and step
    const std::pair<int, int> neib_shape = std::make_pair(2, 2);
    const std::pair<int, int> neib_step = std::make_pair(1, 1);

    // Define the shape of the output vector.
    // We want to limit the size of the output to be slightly smaller then the total number of convolutions that can be achieved with the given input size and step sizes.
    // This test a situations where we might want the output size to be smaller then the total number of possible convolutions.
    std::vector<int> output_shape = {3, // Limit the output size to 3 rows
                                     3, // Limit the output size to 3 columns
                                     neib_shape.first,
                                     neib_shape.second};

    const int output_channels = neib_shape.first * neib_shape.second;
    const int output_size = output_shape[0] * output_shape[1] * output_channels; // 3 * 3 * 4 = 36

    // Set the wrap_mode and center_neigh flags
    bool wrap_mode = true;
    bool center_neigh = true;

    // This simulates the 4D vector output
    // {{{{12, 9}, {4, 1},
    //   {{9, 10}, {1, 2}},
    //   {{10, 11}, {2, 3},
    //   {{11, 12}, {3, 4}}},
    //  {{{4, 1}, {8, 5}},
    //   {{1, 2}, {5, 6}},
    //   {{2, 3}, {6, 7}},
    //   {{3, 4}, {7, 8}}},
    //  {{{8, 5}, {12, 9}},
    //   {{5, 6}, {9, 10}},
    //   {{6, 7}, {10, 11}},
    //   {{7, 8}, {11, 12}}}}

    // Since the output size is limited to 3 columns 3 rows, the last column is not included in the output.
    // This means we want the output to be equal to the following 4D vector
    // {{{{12, 9}, {4, 1},
    //   {{9, 10}, {1, 2}},
    //   {{10, 11}, {2, 3}}},
    //  {{{4, 1}, {8, 5}},
    //   {{1, 2}, {5, 6}},
    //   {{2, 3}, {6, 7}}},
    //  {{{8, 5}, {12, 9}},
    //   {{5, 6}, {9, 10}},
    //   {{6, 7}, {10, 11}}}}

    // Define the expected output vector
    std::vector<int> expected_output = {12, 9, 4, 1, 9, 10, 1, 2, 10, 11, 2, 3, 4, 1, 8, 5, 1, 2, 5, 6, 2, 3, 6, 7, 8, 5, 12, 9, 5, 6, 9, 10, 6, 7, 10, 11};

    // Create the output vector and initialize it with zeros
    std::vector<int> output(output_size, 0);

    tf::Taskflow taskflow;
    // Call the parallel_Images2Neibs_1D function
    parallel_Images2Neibs_1D(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Create a taskflow executor to run the above taskflow
    tf::Executor executor;
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}

TEST(parallel_Images2Neibs_1D, test5_limit_output_size_2)
{

    // This simulates the 2D vector input
    // {{1, 2, 3, 4},
    //  {5, 6, 7, 8},
    //  {9, 10, 11, 12}};
    // Create the input vector
    const std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Define the input shape
    const std::pair<int, int> input_shape = std::make_pair(3, 4);

    // Define the neighbourhood shape and step
    const std::pair<int, int> neib_shape = std::make_pair(2, 2);
    const std::pair<int, int> neib_step = std::make_pair(1, 1);

    // Define the shape of the output vector.
    // We want to limit the size of the output to be slightly smaller then the total number of convolutions that can be achieved with the given input size and step sizes.
    // This test a situations where we might want the output size to be smaller then the total number of possible convolutions.
    std::vector<int> output_shape = {2, // Limit the output size to 2 rows
                                     3, // Limit the output size to 3 columns
                                     neib_shape.first,
                                     neib_shape.second};

    const int output_channels = neib_shape.first * neib_shape.second;
    const int output_size = output_shape[0] * output_shape[1] * output_channels; // 2 * 3 * 4 = 24

    // Set the wrap_mode and center_neigh flags
    bool wrap_mode = true;
    bool center_neigh = true;

    // This simulates the 4D vector output
    // {{{{12, 9}, {4, 1},
    //   {{9, 10}, {1, 2}},
    //   {{10, 11}, {2, 3},
    //   {{11, 12}, {3, 4}}},
    //  {{{4, 1}, {8, 5}},
    //   {{1, 2}, {5, 6}},
    //   {{2, 3}, {6, 7}},
    //   {{3, 4}, {7, 8}}},
    //  {{{8, 5}, {12, 9}},
    //   {{5, 6}, {9, 10}},
    //   {{6, 7}, {10, 11}},
    //   {{7, 8}, {11, 12}}}}

    // Since the output size is limited to  2 rows 3 columns, the last column or row is not included in the output.
    // This means we want the output to be equal to the following 4D vector
    // {{{{12, 9}, {4, 1},
    //   {{9, 10}, {1, 2}},
    //   {{10, 11}, {2, 3}}},
    //  {{{4, 1}, {8, 5}},
    //   {{1, 2}, {5, 6}},
    //   {{2, 3}, {6, 7}}}}

    // Define the expected output vector
    std::vector<int> expected_output = {12, 9, 4, 1, 9, 10, 1, 2, 10, 11, 2, 3, 4, 1, 8, 5, 1, 2, 5, 6, 2, 3, 6, 7};

    // Create the output vector and initialize it with zeros
    std::vector<int> output(output_size, 0);

    tf::Taskflow taskflow;
    // Call the parallel_Images2Neibs_1D function
    parallel_Images2Neibs_1D(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Create a taskflow executor to run the above taskflow
    tf::Executor executor;
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}

TEST(get_connected_syn_input, test1_small)
{
    // Test the get_connected_syn_input function with a small example.
    // This function is used to get the input to the connected synapses for each cortical column.

    // Example data
    std::vector<float> col_syn_perm2 = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<int> col_input_pot_syn2 = {1, 0, 1, 0};
    float connected_perm2 = 0.25f;
    int n_rows2 = 2;
    int n_cols2 = 2;
    std::vector<int> check_conn2(n_rows2 * n_cols2, 2);

    tf::Taskflow taskflow2;
    taskflow2.name("get_connected_syn_input");

    // Define the expected output vector
    std::vector<int> expected_output = {0, 0, 1, 0};

    // Execute the task flow graph
    tf::Executor executor2;

    // Call the function with the example data
    overlap_utils::get_connected_syn_input(col_syn_perm2, col_input_pot_syn2, connected_perm2, n_rows2, n_cols2,
                                           check_conn2, taskflow2);

    tf::Future<void> fu = executor2.run(taskflow2);
    fu.wait(); // block until the execution completes.
    executor2.run(taskflow2).wait();

    //  Print the output vector for the connected synapses
    std::cout << "Connected synapses2: ";

    for (int i = 0; i < check_conn2.size(); ++i)
    {
        std::cout << check_conn2[i] << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(check_conn2, expected_output);
}