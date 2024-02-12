#include <gtest/gtest.h>

// Include the gpu_overlap.hpp header file from the gpu_overlap library
#include "overlap/gpu_overlap.hpp"
#include "overlap.hpp"
#include "overlap_utils.hpp"

#include "logger.hpp"
#include <utilities/stopwatch.hpp>

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

    // Unflatten the output. Note if the step size is more then 1 in any direction then the output upp er dimensions will be smaller then the input grid.
    int N = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));  // Number of rows in output matrix
    int M = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second)); // Number of columns in output matrix
    auto output = gpu_overlap::unflattenVector(flat_output, N, M, neib_shape.first, neib_shape.second);

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

    // Unflatten the output. Note if the step size is more then 1 in any direction then the output upp er dimensions will be smaller then the input grid.
    int N = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));  // Number of rows in output matrix
    int M = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second)); // Number of columns in output matrix
    auto output = gpu_overlap::unflattenVector(flat_output, N, M, neib_shape.first, neib_shape.second);
    
    // Assert the output is the correct size
    ASSERT_EQ(output.size(), n_rows);
    ASSERT_EQ(output[0].size(), n_cols);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(gpu_Images2Neibs, test3_very_large)
{
    // Check that a 40x40 patch is extracted from a 800x800 matrix
    // Use a step size of 1 in the column direction and 2 in the row direction.
    // Create an input matrix for testing but create an already flatten version(800x800) every element is 1
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

TEST(gpu_Images2Neibs, test4_medium_2_step)
{
    // Check that a patch is extracted from a  matrix
    // Use a step size of 2 in the column direction and 2 in the row direction.
    // Create an input matrix for testing every element is 1
    int n_rows = 60;
    int n_cols = 60;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));
    std::pair<int, int> input_shape = {input.size(), input[0].size()};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {20, 20};
    std::pair<int, int> neib_step = {2, 2};
    bool wrap_mode = true;
    bool center_neigh = true;

    // We need to flatten the input matrix
    std::vector<int> flat_input = gpu_overlap::flattenVector(input);

    // Run the function and save the output
    std::vector<int> flat_output = gpu_overlap::gpu_Images2Neibs(flat_input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh);

    // Unflatten the output. Note if the step size is more then 1 in any direction then the output upper dimensions will be smaller then the input grid.
    int N = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));  // Number of rows in output matrix
    int M = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second)); // Number of columns in output matrix
    auto output = gpu_overlap::unflattenVector(flat_output, N, M, neib_shape.first, neib_shape.second);
    
    // Assert the output is the correct size
    ASSERT_EQ(output.size(), N);
    ASSERT_EQ(output[0].size(), M);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(gpu_Images2Neibs, test5_large_2_step)
{
    // Check that a patch is extracted from a  matrix
    // Use a step size of 2 in the column direction and 2 in the row direction.
    // Create an input matrix for testing every element is 1
    int n_rows = 200;
    int n_cols = 200;
    std::vector<std::vector<int>> input = std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols, 1));
    std::pair<int, int> input_shape = {input.size(), input[0].size()};

    // Set the neighbourhood shape and step size
    std::pair<int, int> neib_shape = {30, 30};
    std::pair<int, int> neib_step = {2, 2};
    bool wrap_mode = true;
    bool center_neigh = true;

    // We need to flatten the input matrix
    std::vector<int> flat_input = gpu_overlap::flattenVector(input);

    // Run the function and save the output
    std::vector<int> flat_output = gpu_overlap::gpu_Images2Neibs(flat_input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh);

    // Unflatten the output. Note if the step size is more then 1 in any direction then the output upper dimensions will be smaller then the input grid.
    int N = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));  // Number of rows in output matrix
    int M = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second)); // Number of columns in output matrix
    auto output = gpu_overlap::unflattenVector(flat_output, N, M, neib_shape.first, neib_shape.second);
    
    // Assert the output is the correct size
    ASSERT_EQ(output.size(), N);
    ASSERT_EQ(output[0].size(), M);
    ASSERT_EQ(output[0][0].size(), neib_shape.first);
    ASSERT_EQ(output[0][0][0].size(), neib_shape.second);
}

TEST(gpu_overlap, test1_small)
{
    // Test that the total gpu only implementation produces an expected result.
    // This runs the function calculate_overlap_gpu from the gpu_overlap library.
    // The function calculates the overlap scores for a given input.

    // Create the required inputs for the function
    int pot_width = 2;
    int pot_height = 2;
    bool center_pot_synapses = false;
    int num_input_rows = 10;
    int num_input_cols = 10;
    int num_column_rows = 5;
    int num_column_cols = 5;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = num_column_rows * num_column_cols;
    bool wrap_input = false;
    bool center_neigh = false;

    // Create random colSynPerm array. This is an array representing the permanence values of columns synapses.
    // It stores for each column the permanence values of all potential synapses from that column connecting to the input.
    // It is a 1D vector simulating a 2D vector of size num_columns * num_pot_syn.
    std::vector<float> col_syn_perm(num_columns * num_pot_syn);
    std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn}; // Store the shape of the simulated col_syn_perm 2D vector.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    // TODO: remove this.
    // This is slow and only for testing.
    for (int i = 0; i < num_columns * num_pot_syn; ++i)
    {
        col_syn_perm[i] = dis(gen);
    }
    // Create a random input matrix. This is a matrix representing the input to the HTM layer.
    // It is a boolean input of 1 or 0.
    // It is a 1D vector simulating a 2D vector of size num_input_rows * num_input_cols.
    std::vector<int> new_input_mat(num_input_rows * num_input_cols);
    std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols}; // Store the shape of the simulated 2D vector input matrix.
    std::uniform_int_distribution<> dis2(0, 1);
    for (int i = 0; i < num_input_rows * num_input_cols; ++i)
    {
        new_input_mat[i] = dis2(gen);
    }
    // Print the input matrix
    LOG(INFO, "Input matrix: ");
    overlap_utils::print_2d_vector(new_input_mat, std::pair(num_input_rows, num_input_cols));

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, num_column_cols, num_column_rows, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    //std::vector<int> flat_overlap_output = {0};

    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           num_column_cols,
                                           num_column_rows,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(num_column_rows, num_column_cols));
    PRINT_ELAPSED_TIME();

    LOG(INFO, "Starting the GPU overlap calculation.");
    START_STOPWATCH();

    // int n_rows2 = 10;
    // int n_cols2 = 10;
    // std::vector<std::vector<int>> input2 = std::vector<std::vector<int>>(n_rows2, std::vector<int>(n_cols2, 1));
    // std::pair<int, int> input_shape2 = {n_rows2, n_cols2};

    // // Set the neighbourhood shape and step size
    // std::pair<int, int> neib_shape2 = {2, 2};
    // std::pair<int, int> neib_step2 = {1, 1};
    // bool wrap_mode2 = true;
    // bool center_neigh2 = true;

    // // We need to flatten the input matrix
    // std::vector<int> flat_input2 = gpu_overlap::flattenVector(input2);

    // // Run the function and save the output
    // std::vector<int> flat_output = gpu_overlap::gpu_Images2Neibs(flat_input2, input_shape2, neib_shape2, neib_step2, wrap_mode2, center_neigh2);
    // // Print the flat output
    // overlap_utils::print_1d_vector(flat_output);

    // Run the function and save the output
    auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu( col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mat, 
                                        new_input_mat_shape, 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh
                                        );
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap calculation!");
    PRINT_ELAPSED_TIME();
    
    // Print the flat output
    overlap_utils::print_2d_vector(flat_overlap_output,  std::pair(num_column_rows, num_column_cols));


}





