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

    // It first runs the CPU only overlap calulation and then the GPU only overlap calculation.
    // This is so we can compare the results of the GPU only implementation to the CPU only implementation.

    // Create the required inputs for the function
    int pot_width = 2;
    int pot_height = 2;
    bool center_pot_synapses = false;
    int num_input_rows = 5;
    int num_input_cols = 5;
    int height_cortical_cols = 5;
    int width_cortical_cols = 5;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
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
    LOG(INFO, "col_syn_perm: ");
    std::vector<int> perm_shape = {height_cortical_cols, width_cortical_cols, pot_height, pot_width};
    overlap_utils::print_4d_vector(col_syn_perm, perm_shape);
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
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    LOG(INFO, "Setting up the CPU overlap calculation.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(height_cortical_cols, width_cortical_cols));
    PRINT_ELAPSED_TIME();

    LOG(INFO, "Starting the GPU overlap calculation.");
    START_STOPWATCH();

    // Run the function and save the output
    auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu(
                                        width_cortical_cols, height_cortical_cols,
                                        col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mat, 
                                        new_input_mat_shape, 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh,
                                        connected_perm
                                        );
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap calculation!");
    PRINT_ELAPSED_TIME();
    
    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output
    overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}

TEST(gpu_overlap, test2_small_diff_pot_w_h)
{
    // Test that the total gpu only implementation produces an expected result.
    // This runs the function calculate_overlap_gpu from the gpu_overlap library.
    // The function calculates the overlap scores for a given input.

    // It first runs the CPU only overlap calulation and then the GPU only overlap calculation.
    // This is so we can compare the results of the GPU only implementation to the CPU only implementation.

    // Create the required inputs for the function
    int pot_width = 1;
    int pot_height = 2;
    bool center_pot_synapses = false;
    int num_input_rows = 6;
    int num_input_cols = 6;
    int height_cortical_cols = 5;
    int width_cortical_cols = 5;
    float connected_perm = 0.0;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
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
    LOG(INFO, "col_syn_perm: ");
    std::vector<int> perm_shape = {height_cortical_cols, width_cortical_cols, pot_height, pot_width};
    overlap_utils::print_4d_vector(col_syn_perm, perm_shape);
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

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};
    LOG(INFO, "neib_step: " + std::to_string(neib_step.first) + ", " + std::to_string(neib_step.second));


    LOG(INFO, "Setting up the CPU overlap calculation.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    // Print the input matrix
    LOG(INFO, "Input matrix: ");
    overlap_utils::print_2d_vector(new_input_mat, std::pair(num_input_rows, num_input_cols));
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(height_cortical_cols, width_cortical_cols));
    PRINT_ELAPSED_TIME();

    LOG(INFO, "Starting the GPU overlap calculation.");
    START_STOPWATCH();

    // Run the function and save the output
    auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu(
                                        width_cortical_cols, height_cortical_cols,
                                        col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mat, 
                                        new_input_mat_shape, 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh,
                                        connected_perm
                                        );
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap calculation!");
    PRINT_ELAPSED_TIME();
    
    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output
    overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}


TEST(gpu_overlap, test3_large)
{
    // Test that the total gpu only implementation produces an expected result.
    // This runs the function calculate_overlap_gpu from the gpu_overlap library.
    // The function calculates the overlap scores for a given input.

    // It first runs the CPU only overlap calulation and then the GPU only overlap calculation.
    // This is so we can compare the results of the GPU only implementation to the CPU only implementation.

    // Create the required inputs for the function
    int pot_width = 50;
    int pot_height = 50;
    bool center_pot_synapses = false;
    int num_input_rows = 1000;
    int num_input_cols = 1000;
    int height_cortical_cols = 500;
    int width_cortical_cols = 500;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
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
    // NOTE:
    // This is slow and only for testing.
    LOG(INFO, "Creating random col_syn_perm array for test case. Normally this is only done once. This is slow and only for testing.");
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

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    LOG(INFO, "Setting up CPU overlap calculator.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    PRINT_ELAPSED_TIME();

    LOG(INFO, "Starting the GPU overlap calculation.");
    START_STOPWATCH();

    // Run the function and save the output
    auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu(
                                        width_cortical_cols, height_cortical_cols,
                                        col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mat, 
                                        new_input_mat_shape, 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh,
                                        connected_perm
                                        );
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap calculation!");
    PRINT_ELAPSED_TIME();
    
    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output
    // overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}

TEST(gpu_overlap, test4_run_time_avg)
{
    // Compare the run times of the CPU only and GPU only overlap calculations.
    // Run multiple times and get the average run time for each.
    // The GPU only implementation should be faster once run initially as the GPU is warmed up.
    int num_test_run = 5;

    // Create the required inputs for the function
    int pot_width = 30;
    int pot_height = 30;
    bool center_pot_synapses = false;
    int num_input_rows = 500;
    int num_input_cols = 500;
    int height_cortical_cols = 400;
    int width_cortical_cols = 400;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
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
    // NOTE:
    // This is slow and only for testing.
    LOG(INFO, "Creating random col_syn_perm array for test case. Normally this is only done once. This is slow and only for testing.");
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

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    LOG(INFO, "Setting up CPU overlap calculator.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);

    // RUn the CPU only overlap calculation three times and get the average run time.
    float avg_cpu_run_time = 0;
    for (int i = 0; i < num_test_run; i++)
    {
        LOG(INFO, "Starting the CPU overlap calculation.");
        START_STOPWATCH();
        // Run the overlap calculation on the CPU
        overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
        STOP_STOPWATCH();
        LOG(INFO, "FINISHED CPU overlap calculation!");
        std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
        PRINT_ELAPSED_TIME();
        avg_cpu_run_time += GET_ELAPSED_TIME();
    }
    avg_cpu_run_time /= num_test_run;
    LOG(INFO, "Average CPU run time: " + std::to_string(avg_cpu_run_time));

    // Run the GPU only overlap calculation three times and get the average run time.
    float avg_gpu_run_time = 0;
    for (int i = 0; i < num_test_run; i++)
    {
        LOG(INFO, "Starting the GPU overlap calculation.");
        START_STOPWATCH();
        // Run the function and save the output
        auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu(
                                        width_cortical_cols, height_cortical_cols,
                                        col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mat, 
                                        new_input_mat_shape, 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh,
                                        connected_perm
                                        );
        STOP_STOPWATCH();
        LOG(INFO, "FINISHED GPU overlap calculation!");
        PRINT_ELAPSED_TIME();
        avg_gpu_run_time += GET_ELAPSED_TIME();
    }
    avg_gpu_run_time /= num_test_run;
    LOG(INFO, "Average GPU run time: " + std::to_string(avg_gpu_run_time));

}

TEST(gpu_overlap, test5_different_sizes)
{
    // Run the GPU overlap calculation for different input sizes 
    // and compare the results.

    // Define the input sizes and potential sizes to test
    std::vector<int> in_sizes_width = {10, 11, 12, 13, 14, 15, 20, 30, 40, 50};
    std::vector<int> in_sizes_height = {10, 11, 12, 13, 14, 15, 20, 30, 40, 50};
    std::vector<int> pot_sizes_width = {1, 2, 3, 4, 5};
    std::vector<int> pot_sizes_height = {1, 2, 3, 4, 5};

    // Create for loops to run the comparison for different input sizes and potential sizes.
    for (int pot_size_width : pot_sizes_width)
    {
        for (int pot_size_height : pot_sizes_height)
        {
            for (int in_size_width : in_sizes_width)
            {
                for (int in_size_height : in_sizes_height)
                {
                    // Print the current input and potential sizes
                    LOG(INFO, "Running test for input size: " + std::to_string(in_size_width) + "x" + std::to_string(in_size_height) + " and potential size: " + std::to_string(pot_size_width) + "x" + std::to_string(pot_size_height) + ".");

                    // Create the required inputs for the function
                    int pot_width = pot_size_width;
                    int pot_height = pot_size_height;
                    bool center_pot_synapses = false;
                    int num_input_rows = in_size_height;
                    int num_input_cols = in_size_width;
                    int height_cortical_cols = in_size_height / 2;  // Could have used different sizes here.
                    int width_cortical_cols = in_size_width / 2;    // Could have used different sizes here.
                    float connected_perm = 0.3;
                    int min_overlap = 3;
                    int num_pot_syn = pot_width * pot_height;
                    int num_columns = height_cortical_cols * width_cortical_cols;
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
                    // NOTE:
                    // This is slow and only for testing.
                    LOG(INFO, "Creating random col_syn_perm array for test case. Normally this is only done once per run. This is slow and only for testing.");
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

                    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
                    // neib_step_ = {step_x, step_y}
                    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
                    const std::pair<int, int> neib_shape = {pot_height, pot_width};

                    LOG(INFO, "Setting up CPU overlap calculator.");
                    // Create an instance of the overlap calculation class
                    // to compare the GPU only implementation to the CPU implementation.
                    overlap::OverlapCalculator overlapCalc(pot_width,
                                                        pot_height,
                                                        width_cortical_cols,
                                                        height_cortical_cols,
                                                        num_input_cols,
                                                        num_input_rows,
                                                        center_pot_synapses,
                                                        connected_perm,
                                                        min_overlap,
                                                        wrap_input);


                    LOG(INFO, "Starting the CPU overlap calculation.");
                    START_STOPWATCH();
                    // Run the overlap calculation on the CPU
                    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
                    STOP_STOPWATCH();
                    LOG(INFO, "FINISHED CPU overlap calculation!");
                    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
                    PRINT_ELAPSED_TIME();

                    // Run the GPU only overlap calculation three times and get the average run time.
                    LOG(INFO, "Starting the GPU overlap calculation.");
                    START_STOPWATCH();
                    // Run the function and save the output
                    auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu(
                                                    width_cortical_cols, height_cortical_cols,
                                                    col_syn_perm,
                                                    col_syn_perm_shape,
                                                    new_input_mat, 
                                                    new_input_mat_shape, 
                                                    neib_shape, 
                                                    neib_step, 
                                                    wrap_input, 
                                                    center_neigh,
                                                    connected_perm
                                                    );
                    STOP_STOPWATCH();
                    LOG(INFO, "FINISHED GPU overlap calculation!");
                    PRINT_ELAPSED_TIME();

                    // Compare the CPU and GPU outputs
                    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
                    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());
                    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
                }
            }
        }
    }
}

TEST(gpu_overlap_stream, test1_small)
{
    // Test that the total gpu only overlap calculations using streams produces an expected result.
    // This runs the function calculate_overlap_gpu_stream from the gpu_overlap library.
    // The function calculates the overlap scores for a given input.

    // It first runs the CPU only overlap calculation and then the GPU only overlap calculation.
    // This is so we can compare the results of the GPU only implementation to the CPU only implementation.

    // Create the required inputs for the function
    int pot_width = 2;
    int pot_height = 2;
    bool center_pot_synapses = false;
    int num_input_rows = 5;
    int num_input_cols = 5;
    int height_cortical_cols = 5;
    int width_cortical_cols = 5;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
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

    // NOTE:
    // This is slow and only for testing.
    LOG(INFO, "Creating random col_syn_perm array for test case. Normally this is only done once. This is slow and only for testing.");
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

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    LOG(INFO, "Setting up the CPU overlap calculation.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(height_cortical_cols, width_cortical_cols));
    PRINT_ELAPSED_TIME();

    // We need to setup the output vector for the first run of the GPU stream overlap calculation.
    std::vector<float> flat_overlap_output(height_cortical_cols * width_cortical_cols);  // Overlap output scores will be stored here.
    std::vector<float> flat_pot_overlap_output(height_cortical_cols * width_cortical_cols); // Overlap potential output scores will be stored here.
    LOG(INFO, "Starting the GPU overlap stream calculation.");
    START_STOPWATCH();
    // Run the function and save the output
    // Usage:
    // 1. Call initialize_gpu_memory() with the appropriate sizes
    // 2. Call calculate_overlap_gpu() as needed
    // 3. Call cleanup_gpu_memory() to free resources
    gpu_overlap::initialize_gpu_memory(num_input_rows, num_input_cols, height_cortical_cols, width_cortical_cols, neib_shape.first, neib_shape.second);
    // Run the function and save the output
    gpu_overlap::calculate_overlap_gpu_stream(
                                    width_cortical_cols, height_cortical_cols,
                                    col_syn_perm,
                                    col_syn_perm_shape,
                                    new_input_mat, 
                                    new_input_mat_shape, 
                                    neib_shape, 
                                    neib_step, 
                                    wrap_input, 
                                    center_neigh,
                                    connected_perm,
                                    flat_overlap_output,
                                    flat_pot_overlap_output
                                    );
    // Free the GPU memory
    gpu_overlap::cleanup_gpu_memory();
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap stream calculation!");
    PRINT_ELAPSED_TIME();

    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output overlap scores from GPU stream
    overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));
    // Also print the flat output overlap potential scores from GPU stream
    overlap_utils::print_2d_vector(flat_pot_overlap_output,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}

TEST(gpu_overlap_stream, test2_run_time_avg)
{
    // Compare the run times of the GPU only overlap calculations,
    // Compare the standard and stream versions of the GPU only overlap calculations.
    // Run multiple times and get the average run time for each.
    // The GPU stream implementation should be faster as it is not allocating memory on each call.
    
    int num_test_run = 10;

    // Create the required inputs for the function
    int pot_width = 30;
    int pot_height = 30;
    bool center_pot_synapses = false;
    int num_input_rows = 500;
    int num_input_cols = 500;
    int height_cortical_cols = 300;
    int width_cortical_cols = 300;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
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
    // NOTE:
    // This is slow and only for testing.
    LOG(INFO, "Creating random col_syn_perm array for test case. Normally this is only done once. This is slow and only for testing.");
    for (int i = 0; i < num_columns * num_pot_syn; ++i)
    {
        col_syn_perm[i] = dis(gen);
    }

    // Create a random input matrix. This is a matrix representing the input to the HTM layer.
    // It is a boolean input of 1 or 0.
    // It is a 1D vector simulating a 2D vector of size num_input_rows * num_input_cols.
    // Make a vector of length num_test_run of input vectors so we can use a new input each time we run the function.
    std::vector<std::vector<int>> new_input_mats(num_test_run);
    std::vector<std::pair<int, int>> new_input_mats_shapes(num_test_run);
    std::uniform_int_distribution<> dis2(0, 1);
    for (int i = 0; i < num_test_run; i++)
    {
        std::vector<int> new_input_mat(num_input_rows * num_input_cols);
        // NOTE: all the new input matrices should have the same shape.
        std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols}; // Store the shape of the simulated 2D vector input matrix.
        for (int i = 0; i < num_input_rows * num_input_cols; ++i)
        {
            new_input_mat[i] = dis2(gen);
        }
        new_input_mats[i] = new_input_mat;
        new_input_mats_shapes[i] = new_input_mat_shape;
    }

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    // Create a vector of output vectors to store the output of the GPU overlap calculation.
    std::vector<std::vector<float>> flat_overlap_outputs(num_test_run);
    for (int i = 0; i < num_test_run; i++)
    {
        flat_overlap_outputs[i].resize(height_cortical_cols * width_cortical_cols);
    }
    // ALso create a vector of output vectors to store the output of the GPU stream overlap calculation.
    std::vector<std::vector<float>> flat_overlap_stream_outputs(num_test_run);
    for (int i = 0; i < num_test_run; i++)
    {
        flat_overlap_stream_outputs[i].resize(height_cortical_cols * width_cortical_cols);
    }

    // Run the GPU only overlap calculation multiple times and get the average run time.
    float avg_gpu_run_time = 0;
    for (int i = 0; i < num_test_run; i++)
    {
        LOG(INFO, "Starting the GPU overlap calculation.");
        START_STOPWATCH();
        // Run the function and save the output
        auto flat_overlap_output = gpu_overlap::calculate_overlap_gpu(
                                        width_cortical_cols, height_cortical_cols,
                                        col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mats[i], 
                                        new_input_mats_shapes[i], 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh,
                                        connected_perm
                                        );
        STOP_STOPWATCH();
        LOG(INFO, "FINISHED GPU overlap calculation!");
        PRINT_ELAPSED_TIME();
        avg_gpu_run_time += GET_ELAPSED_TIME();
        // Save output to compare later
        flat_overlap_outputs[i] = flat_overlap_output;
    }
    avg_gpu_run_time /= num_test_run;
    LOG(INFO, "Average GPU run time ms: " + std::to_string(avg_gpu_run_time));

    // Now set up the GPU streaming overlap calculation and run it multiple times and get the average run time.
    float avg_gpu_stream_run_time = 0;
    // We need to setup the output vector for the first run of the GPU stream overlap calculation.
    std::vector<float> flat_overlap_output(height_cortical_cols * width_cortical_cols);
    std::vector<float> flat_pot_overlap_output(height_cortical_cols * width_cortical_cols);
    for (int i = 0; i < num_test_run; i++)
    {
        LOG(INFO, "Starting the GPU stream overlap calculation.");
        START_STOPWATCH();
        // Usage:
        // 1. Call initialize_gpu_memory() with the appropriate sizes
        // 2. Call calculate_overlap_gpu() as needed
        // 3. Call cleanup_gpu_memory() to free resources
        gpu_overlap::initialize_gpu_memory(num_input_rows, num_input_cols, height_cortical_cols, width_cortical_cols, neib_shape.first, neib_shape.second);
        // Run the function and save the output
        gpu_overlap::calculate_overlap_gpu_stream(
                                        width_cortical_cols, height_cortical_cols,
                                        col_syn_perm,
                                        col_syn_perm_shape,
                                        new_input_mats[i], 
                                        new_input_mats_shapes[i], 
                                        neib_shape, 
                                        neib_step, 
                                        wrap_input, 
                                        center_neigh,
                                        connected_perm,
                                        flat_overlap_output,
                                        flat_pot_overlap_output
                                        );
        // Free the GPU memory
        gpu_overlap::cleanup_gpu_memory();
        STOP_STOPWATCH();
        LOG(INFO, "FINISHED GPU stream overlap calculation!");
        PRINT_ELAPSED_TIME();
        avg_gpu_stream_run_time += GET_ELAPSED_TIME();
        // Save output to compare later
        flat_overlap_stream_outputs[i] = flat_overlap_output;
    }
    avg_gpu_stream_run_time /= num_test_run;
    LOG(INFO, "Average GPU stream run time ms: " + std::to_string(avg_gpu_stream_run_time));

    // Compare the outputs of the GPU and GPU stream overlap calculations.
    for (int i = 0; i < num_test_run; i++)
    {
        ASSERT_EQ(flat_overlap_outputs[i], flat_overlap_stream_outputs[i]);
    }

    // COmapre run times
    LOG(INFO, "Average GPU run time ms: " + std::to_string(avg_gpu_run_time));
    LOG(INFO, "Average GPU stream run time ms: " + std::to_string(avg_gpu_stream_run_time));

}


TEST(gpu_overlap_stream_opt, test1_small)
{
    // Test that the total gpu only overlap calculations using streams and optimised for reduced GPU memory produces an expected result.
    // This runs the function calculate_overlap_gpu_stream_opt from the gpu_overlap library.
    // The function calculates the overlap scores for a given input.

    // It first runs the CPU only overlap calculation and then the GPU only overlap calculation.
    // This is so we can compare the results of the GPU only implementation to the CPU only implementation.

    // Create the required inputs for the function
    int pot_width = 2;
    int pot_height = 2;
    bool center_pot_synapses = false;
    int num_input_rows = 5;
    int num_input_cols = 5;
    int height_cortical_cols = 5;
    int width_cortical_cols = 5;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
    bool wrap_input = false;
    bool center_neigh = false;

    std::vector<float> col_syn_perm(num_columns * num_pot_syn);
    std::vector<uint32_t> colConBits((num_columns * num_pot_syn + 31) / 32, 0);  // Initialize with 0s, sized to fit all bits
    std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn}; // Store the shape of the simulated col_syn_perm 2D vector.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    LOG(INFO, "Creating random col_syn_perm and colConBits arrays for test case. Normally this is only done once. This is slow and only for testing.");

    for (int i = 0; i < num_columns * num_pot_syn; ++i)
    {
        col_syn_perm[i] = dis(gen);  // Generate random permanence value

        // Calculate the index in the colConBits array and the bit position within the uint32_t element
        int bitIndex = i / 32;  // Index in the colConBits array
        int bitOffset = i % 32; // Bit position within the uint32_t element

        // Set the bit in colConBits if the synapse's permanence is above the connected_perm threshold
        if (col_syn_perm[i] > connected_perm)
        {
            colConBits[bitIndex] |= (1u << bitOffset);
        }
    }
    // Print the colConBits array
    LOG(INFO, "Printing the colConBits array.");
    const std::vector<int> colConBits_size = {height_cortical_cols, width_cortical_cols, pot_height, pot_width};
    overlap_utils::print_4d_bit_vector(colConBits, colConBits_size);
    
    // Modify new_input_mat to use uint32_t for the optimised version
    // It is a 1D vector of bits (stored in groups of 32 bits as uitn32_t) simulating a 2D vector of size num_input_rows * num_input_cols;
    // Change new_input_mat to std::vector<uint32_t> and adjust its setup

    int input_size = num_input_rows * num_input_cols;
    std::vector<uint32_t> new_input_mat((input_size + 31) / 32, 0);
    std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols};
    std::uniform_int_distribution<> dis2(0, 1);
    // Another version of the input matrix used for the CPU overlap calculation.
    std::vector<int> new_input_mat_CPU(num_input_rows * num_input_cols);

    for (int i = 0; i < input_size; ++i)
    {
        if (dis2(gen) == 1)
        {
            new_input_mat[i / 32] |= (1u << (i % 32));
            new_input_mat_CPU[i] = 1;
        }
    }

    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    LOG(INFO, "Setting up the CPU overlap calculation.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat_CPU, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(height_cortical_cols, width_cortical_cols));
    PRINT_ELAPSED_TIME();

    // We need to setup the output vector for the first run of the GPU stream overlap calculation.
    std::vector<float> flat_overlap_output(height_cortical_cols * width_cortical_cols);  // Overlap output scores will be stored here.
    std::vector<float> flat_pot_overlap_output(height_cortical_cols * width_cortical_cols); // Overlap potential output scores will be stored here.
    LOG(INFO, "Starting the GPU overlap stream calculation.");
    START_STOPWATCH();
    // Run the function and save the output
    // Usage:
    // 1. Call initialize_gpu_memory() with the appropriate sizes
    // 2. Call calculate_overlap_gpu() as needed
    // 3. Call cleanup_gpu_memory() to free resources
    bool optimised = true;  // Set to true to use the optimised version of the function to allocate less GPU memory
    gpu_overlap::initialize_gpu_memory(num_input_rows, num_input_cols, height_cortical_cols, width_cortical_cols, neib_shape.first, neib_shape.second, optimised);
    // Run the function and save the output
    gpu_overlap::calculate_overlap_gpu_stream_opt(
                                    width_cortical_cols, height_cortical_cols,
                                    colConBits,
                                    new_input_mat, 
                                    new_input_mat_shape, 
                                    neib_shape, 
                                    neib_step, 
                                    wrap_input, 
                                    center_neigh,
                                    flat_overlap_output,
                                    flat_pot_overlap_output
                                    );
    // Free the GPU memory
    gpu_overlap::cleanup_gpu_memory(optimised);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap stream calculation!");
    PRINT_ELAPSED_TIME();

    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output overlap scores from GPU stream
    overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));
    // Also print the flat output overlap potential scores from GPU stream
    overlap_utils::print_2d_vector(flat_pot_overlap_output,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}

TEST(gpu_overlap_stream_opt, test2_very_large)
{
    // Test that the total gpu only overlap calculations using streams and optimised for reduced GPU memory produces an expected result.
    // This runs the function calculate_overlap_gpu_stream_opt from the gpu_overlap library.
    // The function calculates the overlap scores for a given input.

    // It first runs the CPU only overlap calculation and then the GPU only overlap calculation.
    // This is so we can compare the results of the GPU only implementation to the CPU only implementation.

    // Create the required inputs for the function
    int pot_width = 40;
    int pot_height = 40;
    bool center_pot_synapses = false;
    int num_input_rows = 1000;
    int num_input_cols = 1000;
    int height_cortical_cols = 500;
    int width_cortical_cols = 500;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
    bool wrap_input = false;
    bool center_neigh = false;

    std::vector<float> col_syn_perm(num_columns * num_pot_syn);
    std::vector<uint32_t> colConBits((num_columns * num_pot_syn + 31) / 32, 0);  // Initialize with 0s, sized to fit all bits
    std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn}; // Store the shape of the simulated col_syn_perm 2D vector.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    LOG(INFO, "Creating random col_syn_perm and colConBits arrays for test case. Normally this is only done once. This is slow and only for testing.");

    for (int i = 0; i < num_columns * num_pot_syn; ++i)
    {
        col_syn_perm[i] = dis(gen);  // Generate random permanence value

        // Calculate the index in the colConBits array and the bit position within the uint32_t element
        int bitIndex = i / 32;  // Index in the colConBits array
        int bitOffset = i % 32; // Bit position within the uint32_t element

        // Set the bit in colConBits if the synapse's permanence is above the connected_perm threshold
        if (col_syn_perm[i] > connected_perm)
        {
            colConBits[bitIndex] |= (1u << bitOffset);
        }
    }

    // Create a random input matrix. This is a matrix representing the input to the HTM layer.
    // It is a boolean input of 1 or 0.
    // It is a 1D vector of bits (stored in groups of 32 bits as uitn32_t) simulating a 2D vector of size num_input_rows * num_input_cols;
    // Change new_input_mat to std::vector<uint32_t> and adjust its setup
    int input_size = num_input_rows * num_input_cols;
    std::vector<uint32_t> new_input_mat((input_size + 31) / 32, 0);
    std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols}; // Store the shape of the simulated 2D vector input matrix.
    std::uniform_int_distribution<> dis2(0, 1);
    // Another version of the input matrix used for the CPU overlap calculation.
    std::vector<int> new_input_mat_CPU(num_input_rows * num_input_cols);

    for (int i = 0; i < input_size; ++i)
    {
        if (dis2(gen) == 1)
        {
            new_input_mat[i / 32] |= (1u << (i % 32));
            new_input_mat_CPU[i] = 1;
        }
    }

    // Get the step sizes. How much to step over the input matrix for each patch connected to each cortical column.
    // neib_step_ = {step_x, step_y}
    const std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);
    const std::pair<int, int> neib_shape = {pot_height, pot_width};

    LOG(INFO, "Setting up the CPU overlap calculation.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat_CPU, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    //overlap_utils::print_2d_vector(col_overlap_scores, std::pair(height_cortical_cols, width_cortical_cols));
    PRINT_ELAPSED_TIME();

    // We need to setup the output vector for the first run of the GPU stream overlap calculation.
    std::vector<float> flat_overlap_output(height_cortical_cols * width_cortical_cols);  // Overlap output scores will be stored here.
    std::vector<float> flat_pot_overlap_output(height_cortical_cols * width_cortical_cols); // Overlap potential output scores will be stored here.
    LOG(INFO, "Starting the GPU overlap stream calculation.");
    START_STOPWATCH();
    // Run the function and save the output
    // Usage:
    // 1. Call initialize_gpu_memory() with the appropriate sizes
    // 2. Call calculate_overlap_gpu() as needed
    // 3. Call cleanup_gpu_memory() to free resources
    bool optimised = true;  // Set to true to use the optimised version of the function to allocate less GPU memory
    gpu_overlap::initialize_gpu_memory(num_input_rows, num_input_cols, height_cortical_cols, width_cortical_cols, neib_shape.first, neib_shape.second, optimised);
    // Run the function and save the output
    gpu_overlap::calculate_overlap_gpu_stream_opt(
                                    width_cortical_cols, height_cortical_cols,
                                    colConBits,
                                    new_input_mat, 
                                    new_input_mat_shape, 
                                    neib_shape, 
                                    neib_step, 
                                    wrap_input, 
                                    center_neigh,
                                    flat_overlap_output,
                                    flat_pot_overlap_output
                                    );
    // Free the GPU memory
    gpu_overlap::cleanup_gpu_memory(optimised);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap stream calculation!");
    PRINT_ELAPSED_TIME();

    // Convert the flat_overlap_output to a vector of ints instead of floats (we don't care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output overlap scores from GPU stream
    //overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));
    // Also print the flat output overlap potential scores from GPU stream
    //overlap_utils::print_2d_vector(flat_pot_overlap_output,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}


TEST(gpu_overlap_stream_opt_sparse, test1_small)
{
    // Test that the total GPU only overlap calculations using streams and optimized for sparse inputs produces an expected result.
    // This runs the function calculate_overlap_gpu_stream_opt_sparse from the gpu_overlap library.
    // The function calculates the overlap scores for a given sparse input.

    // It first runs the CPU only overlap calculation and then the GPU only calculation.
    // This is so we can compare the results of the GPU implementation to the CPU implementation.

    int pot_width = 3;
    int pot_height = 3;
    bool center_pot_synapses = false;
    int num_input_rows = 5;
    int num_input_cols = 5;
    int height_cortical_cols = 5;
    int width_cortical_cols = 5;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
    bool wrap_input = false;
    bool center_neigh = false;

    std::vector<float> col_syn_perm(num_columns * num_pot_syn);
    std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn}; // Store the shape of the simulated col_syn_perm 2D vector.
    std::vector<uint32_t> colConBits((num_columns * num_pot_syn + 31) / 32, 0); // Initialize with 0s
    std::vector<gpu_overlap::Int2> active_grid; // Vector to store the coordinates of active inputs
    int num_active = 0; // Number of active elements

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // Generate random permanence values and set connection bits
    for (int i = 0; i < num_columns * num_pot_syn; ++i)
    {   

        float perm = dis(gen);  // Generate random permanence value
        int bitIndex = i / 32;
        int bitOffset = i % 32;

        if (perm > connected_perm)
        {
            colConBits[bitIndex] |= (1u << bitOffset);
            col_syn_perm[i] = perm;
        }
    }
       // Print the colConBits array
    LOG(INFO, "Printing the colConBits array.");
    const std::vector<int> colConBits_size = {height_cortical_cols, width_cortical_cols, pot_height, pot_width};
    overlap_utils::print_4d_bit_vector(colConBits, colConBits_size);
    
    // Create sparse input matrix for the GPU overlap calculation.
    LOG(INFO, "Creating random sparse input matrix for test case.");
    int input_size = num_input_rows * num_input_cols;
    std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols};
    std::uniform_int_distribution<> dis2(0, 1);
    // Another version of the input matrix used for the CPU overlap calculation.
    std::vector<int> new_input_mat_CPU(num_input_rows * num_input_cols);

    // Generate sparse active input and the corresponding non sparse new_input_mat_CPU (used in the CPU calc)
    int inp_count = 0;
    for (int i = 0; i < num_input_rows; ++i)
    {
        for (int j = 0; j < num_input_cols; ++j)
        {
            if (dis2(gen) == 1)
            {
                active_grid.push_back({i, j});  // Store the coordinates of the active input for the GPU overlap calculation
                ++num_active;
                new_input_mat_CPU[inp_count] = 1;  // input used in the CPU overlap calculation
            }
            inp_count++;
        }
    }
    LOG(INFO, "Number of active inputs: " + std::to_string(num_active));

    std::pair<int, int> inputGrid_shape = {num_input_rows, num_input_cols};
    std::pair<int, int> neib_shape = {pot_height, pot_width};
    std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);

    LOG(INFO, "Setting up the CPU overlap calculation.");
    // Create an instance of the overlap calculation class
    // to compare the GPU only implementation to the CPU implementation.
    overlap::OverlapCalculator overlapCalc(pot_width,
                                           pot_height,
                                           width_cortical_cols,
                                           height_cortical_cols,
                                           num_input_cols,
                                           num_input_rows,
                                           center_pot_synapses,
                                           connected_perm,
                                           min_overlap,
                                           wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat_CPU, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    // Print the overlap scores
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(height_cortical_cols, width_cortical_cols));
    PRINT_ELAPSED_TIME();

    // Setup for GPU calculation
    // We need to setup the output vector for the first run of the GPU stream overlap calculation.
    std::vector<float> flat_overlap_output(height_cortical_cols * width_cortical_cols);
    std::vector<float> flat_pot_overlap_output(height_cortical_cols * width_cortical_cols);
    LOG(INFO, "Starting the GPU overlap stream calculation.");
    START_STOPWATCH();
    // Run the function and save the output
    // Usage:
    // 1. Call initialize_gpu_memory() with the appropriate sizes
    // 2. Call calculate_overlap_gpu() as needed
    // 3. Call cleanup_gpu_memory() to free resources
    // Initialize GPU memory
    gpu_overlap::initialize_gpu_memory_sparse(num_active, height_cortical_cols, width_cortical_cols, neib_shape.first, neib_shape.second);

    // Run the sparse overlap calculation on the GPU
    gpu_overlap::calculate_overlap_gpu_stream_opt_sparse(
        width_cortical_cols, height_cortical_cols,
        colConBits,
        active_grid,
        num_active,
        inputGrid_shape,
        neib_shape,
        neib_step,
        wrap_input,
        center_neigh,
        flat_overlap_output,
        flat_pot_overlap_output
    );

    // Cleanup GPU memory
    gpu_overlap::cleanup_gpu_memory_sparse();
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap stream calculation!");
    PRINT_ELAPSED_TIME();

    // COnvert the flat_overlap_output to a vector of ints instead of floats (we don;t care about the small tiebreaker values on the output).
    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    // Print the flat output overlap scores from GPU stream
    overlap_utils::print_2d_vector(flat_overlap_output_int,  std::pair(height_cortical_cols, width_cortical_cols));
    // Also print the flat output overlap potential scores from GPU stream
    overlap_utils::print_2d_vector(flat_pot_overlap_output,  std::pair(height_cortical_cols, width_cortical_cols));

    // Compare the CPU and GPU outputs
    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);

}

TEST(gpu_overlap_stream_opt_sparse, test2_very_large)
{
    // Test that the total GPU only overlap calculations using streams and optimized for sparse inputs produces an expected result.
    // This runs the function calculate_overlap_gpu_stream_opt_sparse from the gpu_overlap library.
    // The function calculates the overlap scores for a given sparse input.

    // It first runs the CPU only overlap calculation and then the GPU only calculation.
    // This is so we can compare the results of the GPU implementation to the CPU implementation.

    int pot_width = 40;
    int pot_height = 40;
    bool center_pot_synapses = false;
    int num_input_rows = 1000;
    int num_input_cols = 1000;
    int height_cortical_cols = 500;
    int width_cortical_cols = 500;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = height_cortical_cols * width_cortical_cols;
    bool wrap_input = false;
    bool center_neigh = false;

    std::vector<float> col_syn_perm(num_columns * num_pot_syn);
    std::vector<uint32_t> colConBits((num_columns * num_pot_syn + 31) / 32, 0);
    std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < num_columns * num_pot_syn; ++i)
    {
        float perm = dis(gen);
        int bitIndex = i / 32;
        int bitOffset = i % 32;
        if (perm > connected_perm)
        {
            colConBits[bitIndex] |= (1u << bitOffset);
            col_syn_perm[i] = perm;
        }
    }

    std::vector<gpu_overlap::Int2> active_grid;
    std::vector<uint32_t> new_input_mat((num_input_rows * num_input_cols + 31) / 32, 0);
    std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols};
    std::uniform_int_distribution<> dis2(0, 1);
    std::vector<int> new_input_mat_CPU(num_input_rows * num_input_cols);

    int num_active = 0;
    for (int i = 0; i < num_input_rows; ++i)
    {
        for (int j = 0; j < num_input_cols; ++j)
        {
            int index = i * num_input_cols + j;
            if (dis2(gen) == 1)
            {
                new_input_mat[index / 32] |= (1u << (index % 32));
                new_input_mat_CPU[index] = 1;
                active_grid.push_back({i, j});
                ++num_active;
            }
        }
    }

    std::pair<int, int> inputGrid_shape = {num_input_rows, num_input_cols};
    std::pair<int, int> neib_shape = {pot_height, pot_width};
    std::pair<int, int> neib_step = overlap_utils::get_step_sizes(num_input_cols, num_input_rows, width_cortical_cols, height_cortical_cols, pot_width, pot_height);

    overlap::OverlapCalculator overlapCalc(pot_width, pot_height, width_cortical_cols, height_cortical_cols, num_input_cols, num_input_rows, center_pot_synapses, connected_perm, min_overlap, wrap_input);
    LOG(INFO, "Starting the CPU overlap calculation.");
    START_STOPWATCH();
    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat_CPU, new_input_mat_shape);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED CPU overlap calculation!");
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    PRINT_ELAPSED_TIME();

    std::vector<float> flat_overlap_output(height_cortical_cols * width_cortical_cols);
    std::vector<float> flat_pot_overlap_output(height_cortical_cols * width_cortical_cols);

    
    gpu_overlap::initialize_gpu_memory_sparse(num_active, height_cortical_cols, width_cortical_cols, neib_shape.first, neib_shape.second);
    LOG(INFO, "Starting the GPU overlap stream calculation.");
    START_STOPWATCH();
    gpu_overlap::calculate_overlap_gpu_stream_opt_sparse(width_cortical_cols, height_cortical_cols, colConBits, active_grid, num_active, inputGrid_shape, neib_shape, neib_step, wrap_input, center_neigh, flat_overlap_output, flat_pot_overlap_output);
    STOP_STOPWATCH();
    LOG(INFO, "FINISHED GPU overlap stream calculation!");
    PRINT_ELAPSED_TIME();
    gpu_overlap::cleanup_gpu_memory_sparse();

    std::vector<int> flat_overlap_output_int(flat_overlap_output.begin(), flat_overlap_output.end());

    ASSERT_EQ(col_overlap_scores, flat_overlap_output_int);
}
