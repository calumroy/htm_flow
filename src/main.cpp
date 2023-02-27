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
    // It stores for each column the permanence values of all potential synapses from that column connecting to the input.
    // It is a 1D vector simulating a 2D vector of size num_columns * num_pot_syn.
    std::vector<float> col_syn_perm(num_columns * num_pot_syn);
    std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn}; // Store the shape of the simulated col_syn_perm 2D vector.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
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

    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
}