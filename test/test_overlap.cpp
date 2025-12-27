#include <gtest/gtest.h>
#include <htm_flow/overlap.hpp>
#include <htm_flow/overlap_utils.hpp>
#include "logger.hpp"
#include <utilities/stopwatch.hpp>
#include <set>

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

// OverlapCalculator Integration Tests
TEST(OverlapCalculatorTest, BasicOverlapCalculation)
{
    // Set up test parameters
    int potential_width = 3;
    int potential_height = 3;
    int columns_width = 4;
    int columns_height = 4;
    int input_width = 6;
    int input_height = 6;
    bool center_pot_synapses = true;
    float connected_perm = 0.5f;
    int min_overlap = 1;
    bool wrap_input = false;

    // Create test input grid (6x6)
    std::vector<int> inputGrid = {
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1
    };
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create test synapse permanence values (4x4 columns, each with 3x3 potential synapses)
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column);
    
    // Initialize with alternating high and low permanence values
    for (int i = 0; i < colSynPerm.size(); ++i) {
        colSynPerm[i] = (i % 2 == 0) ? 0.7f : 0.3f;
    }
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create an instance of OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Run the overlap calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    // Retrieve the overlap scores
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    // Check that we got the expected number of overlap scores
    ASSERT_EQ(overlapScores.size(), num_columns);

    // Check that all overlap scores are non-negative
    for (float score : overlapScores) {
        ASSERT_GE(score, 0.0f);
    }

    LOG(DEBUG, "Basic Overlap Calculation Test Passed");
}

TEST(OverlapCalculatorTest, SmallGridTest)
{
    // Test with a very small grid to verify exact calculations
    int potential_width = 2;
    int potential_height = 2;
    int columns_width = 2;
    int columns_height = 2;
    int input_width = 3;
    int input_height = 3;
    bool center_pot_synapses = false;
    float connected_perm = 0.5f;
    int min_overlap = 0;
    bool wrap_input = false;

    // Create simple test input (3x3)
    std::vector<int> inputGrid = {
        1, 1, 0,
        1, 0, 1,
        0, 1, 1
    };
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create synapse permanence values where all synapses are connected
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column, 0.8f); // All above threshold
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Run calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    // Get results
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    // Verify we have the right number of scores
    ASSERT_EQ(overlapScores.size(), num_columns);

    LOG(DEBUG, "Small Grid Test - Overlap Scores:");
    overlap_utils::print_2d_vector(overlapScores, {columns_height, columns_width});

    LOG(DEBUG, "Small Grid Test Passed");
}

TEST(OverlapCalculatorTest, WrapInputTest)
{
    // Test with wrap_input enabled
    int potential_width = 3;
    int potential_height = 3;
    int columns_width = 3;
    int columns_height = 3;
    int input_width = 4;
    int input_height = 4;
    bool center_pot_synapses = true;
    float connected_perm = 0.4f;
    int min_overlap = 1;
    bool wrap_input = true;

    // Create test input (4x4)
    std::vector<int> inputGrid = {
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create synapse permanence values
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column);
    
    // Set varying permanence values
    for (int i = 0; i < colSynPerm.size(); ++i) {
        colSynPerm[i] = 0.3f + (i % 5) * 0.1f; // Values from 0.3 to 0.7
    }
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create OverlapCalculator with wrap_input enabled
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Run calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    // Get results
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    // Verify results
    ASSERT_EQ(overlapScores.size(), num_columns);
    
    LOG(DEBUG, "Wrap Input Test - Overlap Scores:");
    overlap_utils::print_2d_vector(overlapScores, {columns_height, columns_width});

    LOG(DEBUG, "Wrap Input Test Passed");
}

TEST(OverlapCalculatorTest, ColumnsWiderThanInputWrapDoesNotZeroTail)
{
    // Regression test:
    // When columns_width > input_width with wrap_input enabled, columns beyond the input width
    // should wrap around and still see valid input patches (not remain all zeros).
    const int potential_width = 3;
    const int potential_height = 3;
    const int columns_width = 40;
    const int columns_height = 1;
    const int input_width = 20;
    const int input_height = 4;
    const bool center_pot_synapses = false;
    const float connected_perm = 0.2f;
    const int min_overlap = 0;
    const bool wrap_input = true;

    // Input: 4x20 with a vertical line at x=0, so overlap should be >0 for many columns.
    std::vector<int> inputGrid(input_width * input_height, 0);
    for (int r = 0; r < input_height; ++r) {
        inputGrid[r * input_width + 0] = 1;
    }
    const std::pair<int, int> inputGridShape = {input_height, input_width};

    // Permanences: all synapses connected.
    const int num_columns = columns_width * columns_height;
    const int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column, 1.0f);
    const std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);
    const std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    ASSERT_EQ(overlapScores.size(), static_cast<size_t>(num_columns));

    // Column x=0 and x=20 should map to the same wrapped input start, so their integer overlap
    // (before per-column tie-breakers) should match and be >0.
    const int idx0 = 0;   // (y=0, x=0)
    const int idx20 = 20; // (y=0, x=20)

    const int ov0 = static_cast<int>(overlapScores[idx0]);
    const int ov20 = static_cast<int>(overlapScores[idx20]);

    ASSERT_GT(ov0, 0);
    ASSERT_EQ(ov0, ov20);
}

TEST(OverlapCalculatorTest, NoConnectedSynapsesTest)
{
    // Test case where no synapses are connected (all permanences below threshold)
    int potential_width = 2;
    int potential_height = 2;
    int columns_width = 3;
    int columns_height = 3;
    int input_width = 4;
    int input_height = 4;
    bool center_pot_synapses = false;
    float connected_perm = 0.8f; // High threshold
    int min_overlap = 0;
    bool wrap_input = false;

    // Create test input (all active)
    std::vector<int> inputGrid(input_width * input_height, 1);
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create synapse permanence values (all below threshold)
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column, 0.3f); // All below 0.8 threshold
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Run calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    // Get results
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    // All overlap scores should be very small (just tie-breaker values)
    for (float score : overlapScores) {
        ASSERT_LT(score, 1.0f); // Should be less than 1 (no connected synapses)
    }

    overlap_utils::print_2d_vector(overlapScores, {columns_height, columns_width});

    LOG(DEBUG, "No Connected Synapses Test Passed");
}

TEST(OverlapCalculatorTest, AllConnectedSynapsesTest)
{
    // Test case where all synapses are connected and all inputs are active
    int potential_width = 2;
    int potential_height = 2;
    int columns_width = 2;
    int columns_height = 2;
    int input_width = 3;
    int input_height = 3;
    bool center_pot_synapses = false;
    float connected_perm = 0.2f; // Low threshold
    int min_overlap = 0;
    bool wrap_input = true;

    // Create test input (all active)
    std::vector<int> inputGrid(input_width * input_height, 1);
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create synapse permanence values (all above threshold)
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column, 0.9f); // All above 0.2 threshold
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Run calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    // Get results
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    // All overlap scores should be close to the number of potential synapses
    for (float score : overlapScores) {
        ASSERT_GE(score, synapses_per_column - 1.0f); // Should be close to max possible
    }

    LOG(DEBUG, "All Connected Synapses Test Passed");
}

TEST(OverlapCalculatorTest, TieBreakerTest)
{
    // Test that tie-breaker values work correctly
    int potential_width = 2;
    int potential_height = 2;
    int columns_width = 3;
    int columns_height = 3;
    int input_width = 4;
    int input_height = 4;
    bool center_pot_synapses = false;
    float connected_perm = 0.5f;
    int min_overlap = 0;
    bool wrap_input = false;

    // Create test input
    std::vector<int> inputGrid = {
        1, 1, 0, 0,
        1, 1, 0, 0,
        0, 0, 1, 1,
        0, 0, 1, 1
    };
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create identical synapse permanence values for all columns
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column, 0.8f); // All identical
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Run calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    // Get results
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    overlap_utils::print_2d_vector(overlapScores, {columns_height, columns_width});

    // Check that tie-breaker values make scores unique
    std::set<float> uniqueScores(overlapScores.begin(), overlapScores.end());
    ASSERT_EQ(uniqueScores.size(), overlapScores.size()); // All scores should be unique

    LOG(DEBUG, "Tie-Breaker Test Passed");
}

TEST(OverlapCalculatorTest, LargeInputTest)
{
    // Performance test with larger input
    int potential_width = 10;
    int potential_height = 10;
    int columns_width = 500;
    int columns_height = 500;
    int input_width = 1000;
    int input_height = 1000;
    bool center_pot_synapses = true;
    float connected_perm = 0.5f;
    int min_overlap = 1;
    bool wrap_input = false;

    START_STOPWATCH();

    // Create test input (checkerboard pattern)
    std::vector<int> inputGrid(input_width * input_height);
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            inputGrid[i * input_width + j] = (i + j) % 2;
        }
    }
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create synapse permanence values
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column);
    
    // Initialize with random-like pattern
    for (int i = 0; i < colSynPerm.size(); ++i) {
        colSynPerm[i] = 0.3f + (i % 7) * 0.1f; // Values from 0.3 to 0.9
    }
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    STOP_STOPWATCH();
    unsigned long long setup_time = GET_ELAPSED_TIME();

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    START_STOPWATCH();

    // Run calculation
    overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

    STOP_STOPWATCH();

    // Get results
    std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

    // Verify results
    ASSERT_EQ(overlapScores.size(), num_columns);

    LOG(INFO, "Setup time ms: " + std::to_string(setup_time));
    LOG(INFO, "Large Input Overlap calculation time: ");
    PRINT_ELAPSED_TIME();

    LOG(DEBUG, "Large Input Test Passed");
}

TEST(OverlapCalculatorTest, PerformanceTest)
{
    // Performance test with multiple runs
    int numCycles = 5;
    int potential_width = 8;
    int potential_height = 8;
    int columns_width = 100;
    int columns_height = 100;
    int input_width = 200;
    int input_height = 200;
    bool center_pot_synapses = true;
    float connected_perm = 0.5f;
    int min_overlap = 1;
    bool wrap_input = false;

    // Create test input
    std::vector<int> inputGrid(input_width * input_height);
    for (int i = 0; i < inputGrid.size(); ++i) {
        inputGrid[i] = i % 3 == 0 ? 1 : 0; // Sparse pattern
    }
    std::pair<int, int> inputGridShape = {input_height, input_width};

    // Create synapse permanence values
    int num_columns = columns_width * columns_height;
    int synapses_per_column = potential_width * potential_height;
    std::vector<float> colSynPerm(num_columns * synapses_per_column);
    
    for (int i = 0; i < colSynPerm.size(); ++i) {
        colSynPerm[i] = 0.4f + (i % 6) * 0.1f; // Values from 0.4 to 0.9
    }
    std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    START_STOPWATCH();

    // Run multiple cycles
    for (int i = 0; i < numCycles; ++i) {
        overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);
    }

    STOP_STOPWATCH();

    LOG(INFO, "Total time for " + std::to_string(numCycles) + " overlap calculation cycles: ");
    PRINT_ELAPSED_TIME();

    // This is a performance test, so we just assert it completes
    ASSERT_TRUE(true);
}

TEST(OverlapCalculatorTest, DifferentParametersTest)
{
    // Test with various parameter combinations
    struct TestParams {
        int potential_width;
        int potential_height;
        int columns_width;
        int columns_height;
        int input_width;
        int input_height;
        bool center_pot_synapses;
        bool wrap_input;
        float connected_perm;
    };

    std::vector<TestParams> testCases = {
        {3, 3, 5, 5, 10, 10, true, false, 0.5f},
        {2, 4, 4, 3, 8, 6, false, true, 0.3f},
        {5, 2, 3, 6, 15, 12, true, true, 0.7f},
        {1, 1, 10, 10, 12, 12, false, false, 0.1f}
    };

    for (size_t testIdx = 0; testIdx < testCases.size(); ++testIdx) {
        const auto& params = testCases[testIdx];
        
        // Create test input
        std::vector<int> inputGrid(params.input_width * params.input_height);
        for (int i = 0; i < inputGrid.size(); ++i) {
            inputGrid[i] = (i + testIdx) % 3 == 0 ? 1 : 0;
        }
        std::pair<int, int> inputGridShape = {params.input_height, params.input_width};

        // Create synapse permanence values
        int num_columns = params.columns_width * params.columns_height;
        int synapses_per_column = params.potential_width * params.potential_height;
        std::vector<float> colSynPerm(num_columns * synapses_per_column);
        
        for (int i = 0; i < colSynPerm.size(); ++i) {
            colSynPerm[i] = 0.2f + (i % 8) * 0.1f;
        }
        std::pair<int, int> colSynPermShape = {num_columns, synapses_per_column};

        // Create OverlapCalculator
        overlap::OverlapCalculator overlapCalc(
            params.potential_width,
            params.potential_height,
            params.columns_width,
            params.columns_height,
            params.input_width,
            params.input_height,
            params.center_pot_synapses,
            params.connected_perm,
            1, // min_overlap
            params.wrap_input);

        // Run calculation
        overlapCalc.calculate_overlap(colSynPerm, colSynPermShape, inputGrid, inputGridShape);

        // Get results
        std::vector<float> overlapScores = overlapCalc.get_col_overlaps();

        // Verify basic properties
        ASSERT_EQ(overlapScores.size(), num_columns);
        for (float score : overlapScores) {
            ASSERT_GE(score, 0.0f);
        }

        LOG(DEBUG, "Test case " + std::to_string(testIdx + 1) + " passed");
    }

    LOG(DEBUG, "Different Parameters Test Passed");
}

TEST(OverlapCalculatorTest, TieBreakerConsistencyTest)
{
    // Test that tie-breaker values are consistent across multiple runs
    int potential_width = 3;
    int potential_height = 3;
    int columns_width = 4;
    int columns_height = 4;
    int input_width = 6;
    int input_height = 6;
    bool center_pot_synapses = true;
    float connected_perm = 0.5f;
    int min_overlap = 0;
    bool wrap_input = false;

    // Create OverlapCalculator
    overlap::OverlapCalculator overlapCalc(
        potential_width,
        potential_height,
        columns_width,
        columns_height,
        input_width,
        input_height,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    // Get tie-breaker values
    std::vector<float> tieBreaker1 = overlapCalc.get_pot_syn_tie_breaker();
    std::vector<float> tieBreaker2 = overlapCalc.get_pot_syn_tie_breaker();

    // Tie-breaker values should be identical across calls
    ASSERT_EQ(tieBreaker1, tieBreaker2);

    // Verify tie-breaker properties
    ASSERT_EQ(tieBreaker1.size(), columns_width * columns_height * potential_width * potential_height);
    
    // All values should be small and positive
    for (float val : tieBreaker1) {
        ASSERT_GT(val, 0.0f);
        ASSERT_LT(val, 0.5f);
    }

    // Print out the tieBreaker1 values in matrix
    overlap_utils::print_2d_vector(tieBreaker1, {columns_height * potential_height, columns_width * potential_width});

    LOG(DEBUG, "Tie-Breaker Consistency Test Passed");
}

