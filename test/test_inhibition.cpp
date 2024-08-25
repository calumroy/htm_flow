#include <gtest/gtest.h>
#include <htm_flow/inhibition.hpp>
#include <htm_flow/inhibition_utils.hpp>

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
        0, 1, 0, 0,
        1, 0, 1, 0,
        0, 1, 0, 1};

    // Check if the actual output matches the expected output
    ASSERT_EQ(activeColumns, expected_activeColumns);
}


