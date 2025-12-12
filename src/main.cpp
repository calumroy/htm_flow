#include <taskflow/taskflow.hpp>
#include <htm_flow/overlap.hpp>
// // Include the gpu_overlap.hpp header file form the gpu_overlap library
// #include <overlap/gpu_overlap.hpp>#include <inhibition/inhibition.hpp>
#include <htm_flow/inhibition.hpp>
#include <htm_flow/spatiallearn.hpp>
#include <htm_flow/sequence_pooler/active_cells.hpp>
#include <utilities/logger.hpp>
#include <utilities/stopwatch.hpp>
#include <cstdlib>
#include <random>

#define NUM_ITERATIONS 3

int main(int argc, char *argv[])
{
    if (argc != 1)
    {
        std::cerr << "Usage: ./htm_flow" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    using overlap::OverlapCalculator;
    using inhibition::InhibitionCalculator;
    using spatiallearn::SpatialLearnCalculator;
    using sequence_pooler::ActiveCellsCalculator;

    // Overlap calculation parameters (similar to your existing setup)
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
    bool strict_local_activity = false;
    
    // Inhibition calculation parameters
    int inhibition_width = 30;
    int inhibition_height = 30;
    int desired_local_activity = 10;

    // Spatial learning parameters
    float spatialPermanenceInc = 0.05f;
    float spatialPermanenceDec = 0.05f;
    float activeColPermanenceDec = 0.01f;

    // Sequence pooler (active cells) parameters (v1: predictive/segments are stubbed)
    int cells_per_column = 5;
    int max_segments_per_cell = 3;       // keep small for now; full TM wiring comes later
    int max_synapses_per_segment = 10;    // keep small for now; full TM wiring comes later
    int min_num_syn_threshold = 1;
    int min_score_threshold = 1;
    float new_syn_permanence = 0.3f;
    float connect_permanence = 0.2f;

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
    // Random data initialization for testing
    std::vector<float> colOverlapGrid(num_column_rows * num_column_cols, 1); // Placeholder for overlap grid
    std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};

    std::vector<float> potColOverlapGrid(num_column_rows * num_column_cols, 1); // Placeholder for potential overlap grid
    std::pair<int, int> potColOverlapGridShape = {num_column_rows, num_column_cols};

    // Create calculator instances once; this is the "pipeline" setup.
    OverlapCalculator overlapCalc(
        pot_width,
        pot_height,
        num_column_cols,
        num_column_rows,
        num_input_cols,
        num_input_rows,
        center_pot_synapses,
        connected_perm,
        min_overlap,
        wrap_input);

    InhibitionCalculator inhibitionCalc(
        num_column_cols,
        num_column_rows,
        inhibition_width,
        inhibition_height,
        desired_local_activity,
        min_overlap,
        center_pot_synapses,
        wrap_input,
        strict_local_activity);

    SpatialLearnCalculator spatialLearnCalc(
        num_columns,
        num_pot_syn,
        spatialPermanenceInc,
        spatialPermanenceDec,
        activeColPermanenceDec);

    ActiveCellsCalculator activeCellsCalc(ActiveCellsCalculator::Config{
        num_columns,
        cells_per_column,
        max_segments_per_cell,
        max_synapses_per_segment,
        min_num_syn_threshold,
        min_score_threshold,
        new_syn_permanence,
        connect_permanence,
    });

    // Run a couple of iterations to demonstrate stateful learning:
    // `col_syn_perm` persists and is updated by the spatial learning stage.

    for (int t = 0; t < NUM_ITERATIONS; ++t)
    {
        LOG(INFO, "=== Iteration " + std::to_string(t) + " ===");
        const int time_step = t + 1;

        // New random input each iteration (placeholder for real sensory input).
        for (int i = 0; i < num_input_rows * num_input_cols; ++i)
        {
            new_input_mat[i] = dis2(gen);
        }

        // Overlap
        START_STOPWATCH();
        LOG(INFO, "Starting the overlap calculation.");
        overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
        STOP_STOPWATCH();
        PRINT_ELAPSED_TIME();

        const std::vector<float> col_overlap_scores = overlapCalc.get_col_overlaps();

        // Inhibition
        START_STOPWATCH();
        LOG(INFO, "Starting the inhibition calculation.");
        inhibitionCalc.calculate_inhibition(col_overlap_scores, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);
        STOP_STOPWATCH();
        PRINT_ELAPSED_TIME();

        // Grab the already-computed active column indices (no recomputation / no scan).
        const std::vector<int>& activeColIndices = inhibitionCalc.get_active_column_indices();

        // Sequence pooler: active cells (v1 stub wiring).
        START_STOPWATCH();
        LOG(INFO, "Starting the sequence-pooler active-cells calculation.");
        activeCellsCalc.calculate_active_cells(time_step, activeColIndices);
        STOP_STOPWATCH();
        PRINT_ELAPSED_TIME();

        // Spatial learning consumes the SAME potential-inputs buffer produced by overlap
        // (no patch recomputation, no 2D conversions).
        START_STOPWATCH();
        LOG(INFO, "Starting the spatial learning calculation.");
        spatialLearnCalc.calculate_spatiallearn_1d_active_indices(
            col_syn_perm,
            col_syn_perm_shape,
            overlapCalc.get_col_pot_inputs(),
            overlapCalc.get_col_pot_inputs_shape(),
            activeColIndices);
        STOP_STOPWATCH();
        PRINT_ELAPSED_TIME();
    }
    return 0;
}
