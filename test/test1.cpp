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

TEST(Images2NeibsTest, Test1)
{
    // Test 1: Check that a 3x3 patch is extracted from a 5x5 matrix
    std::vector<std::vector<int>> input = {{1, 2, 3, 4, 5},
                                           {6, 7, 8, 9, 10},
                                           {11, 12, 13, 14, 15},
                                           {16, 17, 18, 19, 20},
                                           {21, 22, 23, 24, 25}};
    std::pair<int, int> neib_shape = {3, 3};
    std::pair<int, int> neib_step = {1, 1};
    bool mode = false;

    std::vector<std::vector<std::vector<std::vector<int>>>> output = Images2Neibs(input, neib_shape, neib_step, mode);

    ASSERT_EQ(output.size(), 5);
    ASSERT_EQ(output[0].size(), 5);
    ASSERT_EQ(output[0][0].size(), 3);
    ASSERT_EQ(output[0][0][0].size(), 3);

    ASSERT_EQ(output[0][0][0][0], 1);
    ASSERT_EQ(output[0][0][0][1], 2);
    ASSERT_EQ(output[0][0][0][2], 3);
    ASSERT_EQ(output[0][0][1][0], 6);
    ASSERT_EQ(output[0][0][1][1], 7);
    ASSERT_EQ(output[0][0][1][2], 8);
    ASSERT_EQ(output[0][0][2][0], 11);
    ASSERT_EQ(output[0][0][2][1], 12);
    ASSERT_EQ(output[0][0][2][2], 13);

    ASSERT_EQ(output[0][1][0][0], 2);
    ASSERT_EQ(output[0][1][0][1], 3);
    ASSERT_EQ(output[0][1][0][2], 4);
    ASSERT_EQ(output[0][1][1][0], 7);
    ASSERT_EQ(output[0][1][1][1], 8);
    ASSERT_EQ(output[0][1][1][2], 9);
    ASSERT_EQ(output[0][1][2][0], 12);
    ASSERT_EQ(output[0][1][2][1], 13);
    ASSERT_EQ(output[0][1][2][2], 14);

    ASSERT_EQ(output[0][2][0][0], 3);
    ASSERT_EQ(output[0][2][0][1], 4);
    ASSERT_EQ(output[0][2][0][2], 5);
    ASSERT_EQ(output[0][2][1][0], 8);
    ASSERT_EQ(output[0][2][1][1], 9);
    ASSERT_EQ(output[0][2][1][2], 10);
    ASSERT_EQ(output[0][2][2][0], 13);
    ASSERT_EQ(output[0][2][2][1], 14);
    ASSERT_EQ(output[0][2][2][2], 15);

    ASSERT_EQ(output[1][0][0][0], 6);
    ASSERT_EQ(output[1][0][0][1], 7);
    ASSERT_EQ(output[1][0][0][2], 8);
    ASSERT_EQ(output[1][0][1][0], 11);
    ASSERT_EQ(output[1][0][1][1], 12);
    ASSERT_EQ(output[1][0][1][2], 13);
    ASSERT_EQ(output[1][0][2][0], 16);
    ASSERT_EQ(output[1][0][2][1], 17);
    ASSERT_EQ(output[1][0][2][2], 18);

    // We could carry on the asserts for the rest of the matrix
}
