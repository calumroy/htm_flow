#include <gtest/gtest.h>
#include <src/overlap/overlap.cpp>

// Using the overlap namespace
using namespace overlap;
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
    // Test 1: Check that a 2x2 patch is extracted from a 3x3 matrix
    // Create an input matrix for testing
    std::vector<std::vector<int>> input = {{1, 2, 3},
                                           {4, 5, 6},
                                           {7, 8, 9}};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {2, 2};
    std::pair<int, int> neib_step = {1, 2};
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