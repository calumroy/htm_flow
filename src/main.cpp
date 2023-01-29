#include <taskflow/taskflow.hpp>

#include <htm_flow/overlap.hpp>

// Include the gpu_overlap.hpp header file form the gpu_overlap library
#include <overlap/gpu_overlap.hpp>

// Function: main
int main(int argc, char *argv[])
{

    if (argc != 1)
    {
        std::cerr << "Usage: ./htm_flow" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    tf::Executor executor;
    tf::Taskflow taskflow;

    using overlap::OverlapCalculator;

    int pot_width = 3;
    int pot_height = 3;
    bool center_pot_synapses = true;
    int num_input_rows = 4;
    int num_input_cols = 5;
    int num_column_rows = 4;
    int num_column_cols = 4;
    float connected_perm = 0.3;
    int min_overlap = 3;
    int num_pot_syn = pot_width * pot_height;
    int num_columns = num_column_rows * num_column_cols;
    bool wrap_input = true;

    // Create random colSynPerm array. This is an array representing the permanence values of columns synapses.
    std::vector<std::vector<float>> col_syn_perm(num_columns, std::vector<float>(num_pot_syn));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < num_columns; ++i)
    {
        for (int j = 0; j < num_pot_syn; ++j)
        {
            col_syn_perm[i][j] = dis(gen);
        }
    }
    // Create a random input matrix. This is a matrix representing the input to the HTM layer.
    std::vector<std::vector<int>> new_input_mat(num_input_rows, std::vector<int>(num_input_cols));
    std::uniform_int_distribution<> dis2(0, 1);
    for (int i = 0; i < num_input_rows; ++i)
    {
        for (int j = 0; j < num_input_cols; ++j)
        {
            new_input_mat[i][j] = dis2(gen);
        }
    }

    // Create an instance of the overlap calculation class
    OverlapCalculator overlapCalc(pot_width,
                                  pot_height,
                                  num_column_cols,
                                  num_column_rows,
                                  num_input_cols,
                                  num_input_rows,
                                  center_pot_synapses,
                                  connected_perm,
                                  min_overlap,
                                  wrap_input);

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
    std::cout << "\nflat_output: " << std::endl;
    for (int i = 0; i < flat_output.size(); i++)
    {
        std::cout << flat_output[i] << ", ";
    }

    // Unflatten the output
    auto output = gpu_overlap::unflattenVector(flat_output, input_shape.first, input_shape.second, neib_shape.first, neib_shape.second);
}