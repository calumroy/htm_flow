#include <gtest/gtest.h>
#include <src/overlap/overlap.cpp>
#include <src/overlap/overlap_utils.cpp>
// Include the gpu_overlap.hpp header file from the gpu_overlap library
#include <overlap/gpu_overlap.hpp>

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
    // Check that a 40x40 patch is extracted from a 400x400 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing (400x400) every element is 1
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

TEST(gpu_Images2Neibs, test1_wrap)
{

    // Make a task_gpu_test object and run a gpu test
    std::cout << "Running GPU test" << std::endl;

    // Test 1: Check that a 2x2 patch is extracted from a 3x3 matrix
    // Create an input matrix for testing
    std::vector<std::vector<int>>
        input = {{1, 2, 3},
                 {4, 5, 6},
                 {7, 8, 9}};

    std::pair<int, int> input_shape = {input.size(), input[0].size()};
    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = false;

    // We need to flatten the input matrix
    std::vector<int> flat_input = gpu_overlap::flattenVector(input);

    // Print the flat_input
    std::cout << "flat_input: " << std::endl;
    for (int i = 0; i < flat_input.size(); i++)
    {
        std::cout << flat_input[i] << ", ";
    }

    // Run the function and save the output
    std::vector<int> flat_output = gpu_overlap::gpu_Images2Neibs(flat_input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh);

    // Print the flat output
    std::cout << "\n"
              << "flat_output: " << std::endl;
    for (int i = 0; i < flat_output.size(); i++)
    {
        std::cout << flat_output[i] << ", ";
    }

    // Unflatten the output
    auto output = gpu_overlap::unflattenVector(flat_output, input_shape.first, input_shape.second, neib_shape.first, neib_shape.second);

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

TEST(gpu_Images2Neibs, test2_large)
{
    // Check that a 40x40 patch is extracted from a 400x400 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing (400x400) every element is 1
    int n_rows = 400;
    int n_cols = 400;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));
    std::pair<int, int> input_shape = {input.size(), input[0].size()};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {40, 40};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // We need to flatten the input matrix
    std::vector<int> flat_input = gpu_overlap::flattenVector(input);

    // Run the function and save the output
    std::vector<int> flat_output = gpu_overlap::gpu_Images2Neibs(flat_input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh);

    // Unflatten the output
    auto output = gpu_overlap::unflattenVector(flat_output, input_shape.first, input_shape.second, neib_shape.first, neib_shape.second);

    // Assert the output is the correct size
    ASSERT_EQ(output.size(), n_rows);
    ASSERT_EQ(output[0].size(), n_cols);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(gpu_Images2Neibs, test3_very_large)
{
    // Check that a 40x40 patch is extracted from a 400x400 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing but create an already flatten version(400x400) every element is 1
    int n_rows = 800;
    int n_cols = 800;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));
    std::pair<int, int> input_shape = {n_rows, n_cols};

    // Create a flat version of the input every element is 1. The flat version has the same size as the input
    // but is a 1D vector. This is done to improve the gpu performance.
    // std::vector<int> flat_input = std::vector<int>(n_rows * n_cols, 1);

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {40, 40};
    std::pair<int, int> neib_step = {1, 1};
    bool wrap_mode = true;
    bool center_neigh = true;

    // We need to flatten the input matrix
    std::vector<int> flat_input = gpu_overlap::flattenVector(input);

    // Run the function and save the output
    std::vector<int> flat_output = gpu_overlap::gpu_Images2Neibs(flat_input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh);

    // Don't unflatten the output it is too cpu intensive for large matrices.
    // Just run checks on the flattened output

    // Assert the output is the correct size
    ASSERT_EQ(flat_output.size(), n_rows * n_cols * neib_shape.first * neib_shape.second);
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

    // Call the parallel_Images2Neibs_1D function
    parallel_Images2Neibs_1D(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh);

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}