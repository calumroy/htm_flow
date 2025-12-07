#include <gtest/gtest.h>
#include <htm_flow/inhibition.hpp>
#include <htm_flow/inhibition_utils.hpp>
#include "logger.hpp"  
#include <algorithm>
#include <random>
#include <climits>

#include <utilities/stopwatch.hpp>  // Include for the Stop watch Macros to time tests with.

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
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = false;
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = false;
    bool useTieBreaker = true;  // Use tie breaker when their are multiple columns with the same overlap score.

    // Create test input for colOverlapGrid and potColOverlapGrid
    std::vector<float> colOverlapGrid = {
        3, 2, 4, 1,
        1, 3, 2, 2,
        4, 1, 3, 1,
        2, 4, 1, 3};
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<float> potColOverlapGrid = {
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
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug,
        useTieBreaker);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Define the expected output - with strict_local_activity=true, the constraint
    // "at most 2 active in any neighborhood" is strictly enforced.
    std::vector<int> expected_activeColumns = {
        0, 0, 1, 1,
        0, 0, 0, 0,
        1, 0, 0, 0,
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
    bool wrapMode = false;
    bool strict_local_activity = true;
    // Create test input for colOverlapGrid and potColOverlapGrid
    std::vector<float> colOverlapGrid = {
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

    std::vector<float> potColOverlapGrid = {
        6, 5, 7, 4, 6, 5, 6, 5, 6, 5,
        5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
        7, 4, 6, 4, 6, 5, 6, 5, 6, 5,
        5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
        6, 5, 7, 4, 6, 5, 6, 5, 6, 5,
        5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
        7, 4, 6, 4, 6, 5, 6, 5, 6, 5,
        5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
        6, 5, 7, 4, 6, 5, 6, 5, 6, 5,
        5, 6, 5, 6, 5, 6, 5, 6, 5, 6};
    std::pair<int, int> potColOverlapGridShape = {num_column_rows, num_column_cols};

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Print actual output for debugging
    LOG(DEBUG, "Actual Active Columns (2D):");
    overlap_utils::print_2d_vector(activeColumns, colOverlapGridShape);

    // Define the expected output - with strict_local_activity=true, the constraint
    // "at most 3 active in any neighborhood" is strictly enforced.
    std::vector<int> expected_activeColumns = {
        1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 0, 1, 0, 0, 0, 1, 0};

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);

}

TEST(InhibitionCalculatorTest, Case1) {
    int num_column_cols = 8;
    int num_column_rows = 20;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = true;
    bool useTieBreaker = true;  // Use tie breaker when their are multiple columns with the same overlap score.

    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug,
        useTieBreaker);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns 
    std::vector<int> expected_activeColumns = {
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0
    };

    LOG(DEBUG, "Actual Active Columns (2D):");
    overlap_utils::print_2d_vector(activeColumns, {num_column_rows, num_column_cols});

    LOG(DEBUG, "Expected Active Columns (2D):");
    overlap_utils::print_2d_vector(expected_activeColumns, {num_column_rows, num_column_cols});

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}
TEST(InhibitionCalculatorTest, Case1_15Rows) {
    int num_column_cols = 8;
    int num_column_rows = 15;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = true;
    bool useTieBreaker = true;  // Use tie breaker when their are multiple columns with the same overlap score.

    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0,
        0, 0, 3, 3, 0, 0, 0, 0
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug,
        useTieBreaker);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns - with strict_local_activity=true
    std::vector<int> expected_activeColumns = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    LOG(DEBUG, "Actual Active Columns (2D):");
    overlap_utils::print_2d_vector(activeColumns, {num_column_rows, num_column_cols});

    LOG(DEBUG, "Expected Active Columns (2D):");
    overlap_utils::print_2d_vector(expected_activeColumns, {num_column_rows, num_column_cols});

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}
TEST(InhibitionCalculatorTest, Case2) {
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = true;

    std::vector<float> colOverlapGrid = {
        19, 15, 16, 20,
        21, 11, 12, 18,
        13, 19, 21, 15,
        18, 14, 10, 17
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<float> potColOverlapGrid = colOverlapGrid;

    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug);

    inhibitionCalc.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, colOverlapGridShape);

    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns - with strict_local_activity=true
    std::vector<int> expected_activeColumns = {
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    LOG(DEBUG, "Expected Active Columns (2D):");
    overlap_utils::print_2d_vector(expected_activeColumns, {num_column_rows, num_column_cols});

    ASSERT_EQ(activeColumns, expected_activeColumns);
}

TEST(InhibitionCalculatorTest, Case3) {
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  // centerInhib = 1
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = true;
    bool useTieBreaker = true;
    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        8, 4, 5, 8,
        8, 6, 1, 6,
        7, 7, 9, 4,
        2, 3, 1, 5
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug,
        useTieBreaker);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns - with strict_local_activity=true, the constraint
    // "at most 2 active in any 3x3 neighborhood" is strictly enforced.
    // The previous expected values violated this constraint (e.g., position (1,1)
    // had 4 active columns in its neighborhood: cols 0, 2, 4, 10).
    std::vector<int> expected_activeColumns = {
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}

TEST(InhibitionCalculatorTest, Case3_NonStrict) {
    // Same as Case3 but with strict_local_activity = false
    // This uses the parallel implementation which is faster but may not
    // strictly enforce the local activity constraint in all neighborhoods.
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  // centerInhib = 1
    bool wrapMode = false;
    bool strict_local_activity = false;  // Non-strict mode uses parallel implementation
    bool debug = true;
    bool useTieBreaker = true;
    
    // Initialize colOverlapGrid (same as Case3)
    std::vector<float> colOverlapGrid = {
        8, 4, 5, 8,
        8, 6, 1, 6,
        7, 7, 9, 4,
        2, 3, 1, 5
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug,
        useTieBreaker);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // With strict_local_activity=false, the parallel implementation is used.
    // This may produce more active columns but doesn't guarantee the strict
    // constraint is met in all neighborhoods. The result may also be
    // non-deterministic due to thread scheduling.
    // We just verify that some columns are active (basic sanity check).
    int activeCount = 0;
    for (int val : activeColumns) {
        activeCount += val;
    }
    
    LOG(INFO, "Non-strict mode produced " + std::to_string(activeCount) + " active columns");
    overlap_utils::print_2d_vector(activeColumns, {num_column_rows, num_column_cols});
    
    // Should have at least some active columns
    ASSERT_GT(activeCount, 0);
}

TEST(InhibitionCalculatorTest, Case3_Determinism) {
    // This test verifies that the inhibition calculator produces deterministic results
    // by running the same calculation multiple times and checking all results are identical.
    // It also verifies that the strict local activity constraint is respected.
    const int NUM_ITERATIONS = 100;
    
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  // centerInhib = 1
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = false;  // Disable debug for performance
    bool useTieBreaker = true;
    
    // Initialize colOverlapGrid (same as Case3)
    std::vector<float> colOverlapGrid_original = {
        8, 4, 5, 8,
        8, 6, 1, 6,
        7, 7, 9, 4,
        2, 3, 1, 5
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // Store results from all iterations to compare
    std::vector<std::vector<int>> allResults;
    allResults.reserve(NUM_ITERATIONS);

    LOG(INFO, "Running Case3 determinism test with " + std::to_string(NUM_ITERATIONS) + " iterations...");

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Create fresh copies of the overlap grids for each iteration
        // (since tie-breakers modify the grids)
        std::vector<float> colOverlapGrid = colOverlapGrid_original;
        std::vector<float> potColOverlapGrid = colOverlapGrid_original;

        // Create a new instance of InhibitionCalculator for each iteration
        inhibition::InhibitionCalculator inhibitionCalc(
            num_column_cols,
            num_column_rows,
            inhibition_width,
            inhibition_height,
            desired_local_activity,
            min_overlap,
            center_pot_synapses,
            wrapMode,
            strict_local_activity,
            debug,
            useTieBreaker);

        // Run the inhibition calculation
        inhibitionCalc.calculate_inhibition(
            colOverlapGrid, colOverlapGridShape,
            potColOverlapGrid, colOverlapGridShape);

        // Retrieve the active columns
        std::vector<int> activeColumns = inhibitionCalc.get_active_columns();
        allResults.push_back(activeColumns);
    }

    // Print the first result for reference
    LOG(INFO, "First iteration result:");
    overlap_utils::print_2d_vector(allResults[0], {num_column_rows, num_column_cols});

    // Verify all results are identical to the first result
    const std::vector<int>& firstResult = allResults[0];
    int mismatchCount = 0;
    for (int iter = 1; iter < NUM_ITERATIONS; ++iter) {
        if (allResults[iter] != firstResult) {
            ++mismatchCount;
            LOG(ERROR, "Determinism failure! Iteration " + std::to_string(iter) + 
                       " differs from iteration 0");
            LOG(ERROR, "Iteration 0 result:");
            overlap_utils::print_2d_vector(firstResult, {num_column_rows, num_column_cols});
            LOG(ERROR, "Iteration " + std::to_string(iter) + " result:");
            overlap_utils::print_2d_vector(allResults[iter], {num_column_rows, num_column_cols});
        }
    }

    if (mismatchCount > 0) {
        FAIL() << mismatchCount << " out of " << (NUM_ITERATIONS - 1) << " iterations differed from the first result";
    }

    LOG(INFO, "All " + std::to_string(NUM_ITERATIONS) + " iterations produced identical results!");

    // Verify the constraint is respected: no neighborhood should have more than
    // desired_local_activity active columns
    LOG(INFO, "Verifying strict local activity constraint...");
    for (int row = 0; row < num_column_rows; ++row) {
        for (int col = 0; col < num_column_cols; ++col) {
            // Count active columns in this column's 3x3 neighborhood
            int activeInNeighborhood = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = row + dy;
                    int nx = col + dx;
                    if (ny >= 0 && ny < num_column_rows && nx >= 0 && nx < num_column_cols) {
                        int idx = ny * num_column_cols + nx;
                        if (firstResult[idx] == 1) {
                            activeInNeighborhood++;
                        }
                    }
                }
            }
            if (activeInNeighborhood > desired_local_activity) {
                LOG(ERROR, "Constraint violation at (" + std::to_string(row) + "," + 
                           std::to_string(col) + "): " + std::to_string(activeInNeighborhood) +
                           " active in neighborhood, max allowed is " + std::to_string(desired_local_activity));
                FAIL() << "Local activity constraint violated";
            }
        }
    }
    LOG(INFO, "Constraint verified: all neighborhoods have <= " + std::to_string(desired_local_activity) + " active columns");
    
    SUCCEED();
}

TEST(InhibitionCalculatorTest, WrapModeCase1) {
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 2;
    int inhibition_height = 2;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = false;  
    bool wrapMode = true;
    bool strict_local_activity = true;

    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        8, 4, 5, 8,
        8, 6, 1, 6,
        7, 7, 9, 4,
        2, 3, 1, 5
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns - with strict_local_activity=true
    std::vector<int> expected_activeColumns = {
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0
    };

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}

TEST(InhibitionCalculatorTest, WrapModeCase2) {
    int num_column_cols = 6;
    int num_column_rows = 6;
    int inhibition_width = 5;
    int inhibition_height = 4;
    int desired_local_activity = 4;
    int min_overlap = 1;
    bool center_pot_synapses = false;  
    bool wrapMode = true;
    bool strict_local_activity = true;

    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        8, 4, 5, 8, 3, 2,
        8, 6, 1, 6, 7, 5,
        7, 7, 9, 4, 2, 1,
        2, 3, 1, 5, 6, 8,
        4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns - with strict_local_activity=true
    std::vector<int> expected_activeColumns = {
        1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0
    };

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}

TEST(InhibitionCalculatorTest, Case4) {
    int num_column_cols = 5;
    int num_column_rows = 6;
    int inhibition_width = 2;
    int inhibition_height = 2;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  // centerInhib = 1
    bool wrapMode = false;
    bool strict_local_activity = true;

    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        0, 0, 3.10, 3.11, 0,
        0, 0, 3.02, 3.01, 0,
        0, 0, 3.13, 3.14, 0,
        0, 0, 3.03, 3.04, 0,
        0, 0, 3.15, 3.16, 0,
        0, 0, 3.07, 3.08, 0
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns - with strict_local_activity=true
    std::vector<int> expected_activeColumns = {
        0, 0, 1, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 0,
        0, 0, 0, 0, 0
    };

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}

TEST(InhibitionCalculatorTest, Case5) {
    int num_column_cols = 3;
    int num_column_rows = 3;
    int inhibition_width = 2;
    int inhibition_height = 2;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  // centerInhib = 1
    bool wrapMode = false;
    bool strict_local_activity = true;

    // Initialize colOverlapGrid
    std::vector<float> colOverlapGrid = {
        0, 0, 4,  
        0, 0, 3,  
        0, 0, 5  
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    // Retrieve the active columns
    std::vector<int> activeColumns = inhibitionCalc.get_active_columns();

    // Expected active columns
    std::vector<int> expected_activeColumns = {
        0, 0, 1, 
        0, 0, 0, 
        0, 0, 1 
    };

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}

TEST(InhibitionCalculatorTest, LargeInput) {
    int num_column_cols = 1000;
    int num_column_rows = 1000; 
    int inhibition_width = 20;
    int inhibition_height = 20;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  
    bool wrapMode = true;
    bool strict_local_activity = false;

    START_STOPWATCH();

    // Generate colOverlapGrid with incremental values
    std::vector<float> colOverlapGrid(num_column_cols * num_column_rows);
    for (int row = 0; row < num_column_rows; ++row) {
        for (int col = 0; col < num_column_cols; ++col) {
            colOverlapGrid[row * num_column_cols + col] = 1 + col + num_column_cols * row;
        }
    }
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid plus 1
    std::vector<float> potColOverlapGrid = colOverlapGrid;
    for (int i = 0; i < num_column_cols * num_column_rows; ++i) {
        potColOverlapGrid[i] += 1;
    }

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    STOP_STOPWATCH();
    unsigned long long startup_time_taken = GET_ELAPSED_TIME();
    
    START_STOPWATCH();

    // Run the inhibition calculation
    inhibitionCalc.calculate_inhibition(
        colOverlapGrid, colOverlapGridShape,
        potColOverlapGrid, colOverlapGridShape);

    STOP_STOPWATCH();
    // Print out the startup time 
    LOG(INFO, "Startup time ms: ");
    LOG(INFO, std::to_string(startup_time_taken));
    // Print out the inhibition calculation time
    LOG(INFO, "Inhibition calculation time: ");
    PRINT_ELAPSED_TIME();

    // No assertions; this is a performance test
    ASSERT_TRUE(true);
}

TEST(InhibitionCalculatorTest, RunTime) {
    // This test is to measure the performance of the inhibition calculation
    // The inhibition calculation is run multiple times to test performance
    // Running this test on my laptop: 16 CPU(s) 11th Gen Intel(R) Core(TM) i9-11900H
    //         Build in Release mode
    //         [INFO] TestBody (897): Total time for 10 cycles: 
    //         Elapsed time: 454 milliseconds
    int numCycles = 10;
    int num_column_cols = 1000;
    int num_column_rows = 1000;
    int inhibition_width = 10;
    int inhibition_height = 10;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;  
    bool wrapMode = false;
    bool strict_local_activity = false;
    // Generate colOverlapGrid with incremental values
    std::vector<float> colOverlapGrid(num_column_cols * num_column_rows);
    for (int row = 0; row < num_column_rows; ++row) {
        for (int col = 0; col < num_column_cols; ++col) {
            colOverlapGrid[row * num_column_cols + col] = 1 + col + num_column_cols * row;
        }
    }
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    // potColOverlapGrid is the same as colOverlapGrid
    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Create an instance of InhibitionCalculator
    inhibition::InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    START_STOPWATCH();

    // Run the inhibition calculation multiple times to test performance
    for (int i = 0; i < numCycles; ++i) {
        inhibitionCalc.calculate_inhibition(
            colOverlapGrid, colOverlapGridShape,
            potColOverlapGrid, colOverlapGridShape);
    }

    STOP_STOPWATCH();
    LOG(INFO, "Total time for " + std::to_string(numCycles) + " cycles: ");
    PRINT_ELAPSED_TIME();

    // No assertions; this is a performance test
    ASSERT_TRUE(true);
}

TEST(InhibitionCalculatorTest, SerialSortedVsParallel) {
    // Test to verify that the serial sorted implementation produces the same results as the parallel implementation
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 3;
    int inhibition_height = 3;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = true;
    bool wrapMode = false;
    bool strict_local_activity = true;
    bool debug = false;

    std::vector<float> colOverlapGrid = {
        19, 15, 16, 20,
        21, 11, 12, 18,
        13, 19, 21, 15,
        18, 14, 10, 17
    };
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<float> potColOverlapGrid = colOverlapGrid;

    // Test with parallel implementation
    inhibition::InhibitionCalculator inhibitionCalcParallel(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug);

    inhibitionCalcParallel.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, colOverlapGridShape, false);
    std::vector<int> parallelActiveColumns = inhibitionCalcParallel.get_active_columns();

    // Test with serial sorted implementation
    inhibition::InhibitionCalculator inhibitionCalcSerial(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrapMode,
        strict_local_activity,
        debug);

    inhibitionCalcSerial.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, colOverlapGridShape, true);
    std::vector<int> serialActiveColumns = inhibitionCalcSerial.get_active_columns();

    // Compare results
    LOG(DEBUG, "Parallel Active Columns (2D):");
    overlap_utils::print_2d_vector(parallelActiveColumns, {num_column_rows, num_column_cols});
    
    LOG(DEBUG, "Serial Active Columns (2D):");
    overlap_utils::print_2d_vector(serialActiveColumns, {num_column_rows, num_column_cols});

    // The results should be the same
    ASSERT_EQ(parallelActiveColumns, serialActiveColumns);
}

TEST(InhibitionCalculatorTest, SerialSortedBasicTest) {
    // Basic test specifically for the serial sorted implementation
    int num_column_cols = 4;
    int num_column_rows = 4;
    int inhibition_width = 2;
    int inhibition_height = 2;
    int desired_local_activity = 2;
    int min_overlap = 1;
    bool center_pot_synapses = false;
    bool wrapMode = false;
    bool strict_local_activity = false;

    // Create test input for colOverlapGrid and potColOverlapGrid
    std::vector<float> colOverlapGrid = {
        3, 2, 4, 1,
        1, 3, 2, 2,
        4, 1, 3, 1,
        2, 4, 1, 3};
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<float> potColOverlapGrid = {
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
        center_pot_synapses,
        wrapMode,
        strict_local_activity);

    // Run the inhibition calculation with serial sorted implementation
    inhibitionCalc.calculate_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape, true);

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
