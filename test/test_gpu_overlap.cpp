#include <gtest/gtest.h>

// Include the gpu_overlap.hpp header file from the gpu_overlap library
#include <overlap/gpu_overlap.hpp>

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

TEST(gpu_Images2Neibs, test1_1D_wrap)
{
    // Test the gpu_Images2Neibs function version that has the same input and output parameters as the parallel_Images2Neibs_1D function.
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

    // Create the taskflow and executor
    tf::Taskflow taskflow;
    tf::Executor executor;

    // Call the gpu_Images2Neibs 1D function
    gpu_overlap::gpu_Images2Neibs(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Run the taskflow
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}

TEST(gpu_Images2Neibs, test2_1D_nowrap_center)
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

    // Create the taskflow and executor
    tf::Taskflow taskflow;
    tf::Executor executor;

    // Call the gpu_Images2Neibs function
    gpu_overlap::gpu_Images2Neibs(output, output_shape, input, input_shape, neib_shape, neib_step, wrap_mode, center_neigh, taskflow);

    // Run the taskflow
    executor.run(taskflow).wait();

    // Check that the output vector is equal to the expected output vector
    ASSERT_EQ(output, expected_output);
}