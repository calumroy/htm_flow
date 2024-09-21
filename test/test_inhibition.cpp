#include <gtest/gtest.h>
#include <htm_flow/inhibition.hpp>
#include <htm_flow/inhibition_utils.hpp>
#include "logger.hpp"  
#include <algorithm>
#include <random>
#include <climits>

TEST(ParallelSortIndTest, BasicSorting) {
    // Set up test data
    std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> values = {5, 2, 8, 6, 1, 7, 3, 4};

    // Expected result: indices sorted by corresponding values in descending order
    std::vector<int> expected_sorted_indices = {2, 5, 3, 0, 7, 6, 1, 4};

    // Create Taskflow and Executor
    tf::Taskflow taskflow;
    tf::Executor executor;

    // Run the parallel_sort_ind function
    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    // Run the constructed taskflow graph
    tf::Future<void> fu = executor.run(taskflow);
    fu.wait(); // Block until the execution completes

    // Check if the actual output matches the expected output
    ASSERT_EQ(indices, expected_sorted_indices);
}

TEST(ParallelSortIndTest, SmallSorting) {
    // Set up test data with a small number of elements
    const int size = 10;
    std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> values = {42, 15, 78, 30, 5, 61, 23, 50, 9, 37};

    // Create a copy of indices to sort using std::sort for comparison
    std::vector<int> expected_sorted_indices = indices;
    // Use stable_sort to preserve relative order of equal values
    std::stable_sort(expected_sorted_indices.begin(), expected_sorted_indices.end(),
                     [&values](int a, int b) { return values[a] > values[b]; });

    // Create Taskflow and Executor
    tf::Taskflow taskflow;
    tf::Executor executor;

    // Run the parallel_sort_ind function
    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    // Run the constructed taskflow graph
    tf::Future<void> fu = executor.run(taskflow);
    fu.wait(); // Block until the execution completes

    // Check if the actual output matches the expected output
    ASSERT_EQ(indices, expected_sorted_indices);

    // Print the sorted indices and their corresponding values for better understanding
    std::cout << "Sorted indices and their values:" << std::endl;
    for (int i : indices) {
        std::cout << "Index: " << i << ", Value: " << values[i] << std::endl;
    }
}

// Test with empty input vectors
TEST(ParallelSortIndTest, EmptyInput) {
    std::vector<int> indices;
    std::vector<int> values;

    std::vector<int> expected_sorted_indices;

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with a single-element vector
TEST(ParallelSortIndTest, SingleElement) {
    std::vector<int> indices = {0};
    std::vector<int> values = {42};

    std::vector<int> expected_sorted_indices = {0};

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with duplicate values in the vector
TEST(ParallelSortIndTest, DuplicateValues) {
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> values = {5, 2, 5, 2, 5};

    std::vector<int> expected_sorted_indices = indices;
    // Use stable_sort to preserve relative order of equal values
    std::stable_sort(expected_sorted_indices.begin(), expected_sorted_indices.end(),
                     [&values](int a, int b) { return values[a] > values[b]; });

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with negative values in the vector
TEST(ParallelSortIndTest, NegativeValues) {
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> values = {-1, -3, -2, -5, -4};

    std::vector<int> expected_sorted_indices = {0, 2, 1, 4, 3};

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with an already sorted (descending) vector
TEST(ParallelSortIndTest, AlreadySortedDescending) {
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> values = {100, 90, 80, 70, 60};

    std::vector<int> expected_sorted_indices = {0, 1, 2, 3, 4};

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with a vector sorted in ascending order
TEST(ParallelSortIndTest, SortedAscending) {
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> values = {10, 20, 30, 40, 50};

    std::vector<int> expected_sorted_indices = {4, 3, 2, 1, 0};

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Stress test with a large input vector
TEST(ParallelSortIndTest, LargeInput) {
    const int size = 1000000;
    std::vector<int> indices(size);
    std::vector<int> values(size);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 100);

    for (int i = 0; i < size; ++i) {
        indices[i] = i;
        values[i] = dist(rng);
    }

    // // Print out the values using overlap_utils::print_1d_vector
    // LOG(INFO, "Values: ");
    // overlap_utils::print_1d_vector(values);

    std::vector<int> expected_sorted_indices = indices;
    // Use stable_sort to preserve relative order of equal values
    std::stable_sort(expected_sorted_indices.begin(), expected_sorted_indices.end(),
                     [&values](int a, int b) { return values[a] > values[b]; });


    // LOG(INFO, "Expected sorted indices: ");
    // overlap_utils::print_1d_vector(expected_sorted_indices);

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    // // print out the values associated with the indices
    // LOG(INFO, "Sorted Indices: ");
    // overlap_utils::print_1d_vector(indices);
    // LOG(INFO, "Sorted Values: ");
    // for (int i : indices) {
    //     std::cout << "Index: " << i << ", Value: " << values[i] << std::endl;
    // }

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with all equal values in the vector
TEST(ParallelSortIndTest, AllEqualValues) {
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> values = {42, 42, 42, 42, 42};

    std::vector<int> expected_sorted_indices = indices;

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    // Since all values are equal, any permutation of indices is acceptable
    std::sort(indices.begin(), indices.end());
    std::sort(expected_sorted_indices.begin(), expected_sorted_indices.end());

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with mixed positive and negative values
TEST(ParallelSortIndTest, MixedPositiveNegativeValues) {
    std::vector<int> indices = {0, 1, 2, 3, 4, 5};
    std::vector<int> values = {5, -10, 15, -20, 25, -30};

    std::vector<int> expected_sorted_indices = {4, 2, 0, 1, 3, 5};

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with maximum and minimum integer values
TEST(ParallelSortIndTest, MaxMinIntValues) {
    std::vector<int> indices = {0, 1, 2, 3};
    std::vector<int> values = {INT_MAX, INT_MIN, 0, -1};

    std::vector<int> expected_sorted_indices = {0, 2, 3, 1};

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

// Test with random values for comprehensive coverage
TEST(ParallelSortIndTest, RandomValues) {
    const int size = 100;
    std::vector<int> indices(size);
    std::vector<int> values(size);

    std::mt19937 rng(67890);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (int i = 0; i < size; ++i) {
        indices[i] = i;
        values[i] = dist(rng);
    }

    std::vector<int> expected_sorted_indices = indices;
    // Use stable_sort to preserve relative order of equal values
    std::stable_sort(expected_sorted_indices.begin(), expected_sorted_indices.end(),
                     [&values](int a, int b) { return values[a] > values[b]; });

    tf::Taskflow taskflow;
    tf::Executor executor;

    inhibition_utils::parallel_sort_ind(indices, values, taskflow);

    tf::Future<void> fu = executor.run(taskflow);
    fu.wait();

    ASSERT_EQ(indices, expected_sorted_indices);
}

TEST(InhibitionCalculatorTest, BasicInhibitionCalculation)
{
    // Set up test parameters
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 2;
    int inhibition_height = 2;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = false;

    // Create test input for colOverlapGrid and potColOverlapGrid
    std::vector<int> colOverlapGrid = {
        3, 2, 4, 1,
        1, 3, 2, 2,
        4, 1, 3, 1,
        2, 4, 1, 3};
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<int> potColOverlapGrid = {
        2, 1, 2, 1,
        1, 2, 1, 2,
        2, 1, 2, 1,
        1, 2, 1, 2};
    std::pair<int, int> potColOverlapGridShape = {num_column_rows, num_column_cols};

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Define the expected output (modify according to your expectations)
    std::vector<int> expected_activeColumns = {
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1};

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}


TEST(InhibitionCalculatorTest, LargeInhibitionCalculation)
{
    // Set up test parameters
    int num_column_cols = 10;
    int num_column_rows = 10;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 3;
    int min_overlap = 1;
    bool center_pot_synapses = false;

    // Create test input for colOverlapGrid and potColOverlapGrid
    std::vector<int> colOverlapGrid = {
        3, 2, 4, 1, 5, 2, 3, 4, 1, 2,
        1, 3, 2, 2, 4, 1, 3, 2, 4, 1,
        4, 1, 3, 1, 2, 4, 1, 3, 1, 2,
        2, 4, 1, 3, 1, 2, 4, 1, 3, 1,
        3, 2, 4, 1, 5, 2, 3, 4, 1, 2,
        1, 3, 2, 2, 4, 1, 3, 2, 4, 1,
        4, 1, 3, 1, 2, 4, 1, 3, 1, 2,
        2, 4, 1, 3, 1, 2, 4, 1, 3, 1,
        3, 2, 4, 1, 5, 2, 3, 4, 1, 2,
        1, 3, 2, 2, 4, 1, 3, 2, 4, 1};
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<int> potColOverlapGrid = {
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
    std::pair<int, int> potColOverlapGridShape = {num_column_rows, num_column_cols};

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Define the expected output (modify according to your expectations)
    std::vector<int> expected_activeColumns = {
        0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
        0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0};

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}