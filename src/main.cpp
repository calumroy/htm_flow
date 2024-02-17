#include <taskflow/taskflow.hpp>

#include <htm_flow/overlap.hpp>

// // Include the gpu_overlap.hpp header file form the gpu_overlap library
// #include <overlap/gpu_overlap.hpp>

#include <utilities/logger.hpp>
#include <utilities/stopwatch.hpp>

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

    int pot_width = 30;
    int pot_height = 30;
    bool center_pot_synapses = false;
    int num_input_rows = 1200;
    int num_input_cols = 1200;
    int num_column_rows = 800;
    int num_column_cols = 800;
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


    START_STOPWATCH();
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

    LOG(INFO, "Starting the overlap calculation.");

    // Run the overlap calculation on the CPU
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    
    // // Print the input matrix
    // LOG(INFO, "Input matrix: ");
    // overlap_utils::print_2d_vector(new_input_mat, std::pair(num_input_rows, num_input_cols));

    // Print the overlap scores
    std::vector<int> col_overlap_scores = overlapCalc.get_col_overlaps();
    overlap_utils::print_2d_vector(col_overlap_scores, std::pair(num_column_rows, num_column_cols));
    PRINT_ELAPSED_TIME();
    
}