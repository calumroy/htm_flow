#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <taskflow/taskflow.hpp>

#include <utilities/logger.hpp>
#include <htm_flow/overlap.hpp>
#include <htm_flow/overlap_utils.hpp>

namespace overlap
{

    // Define a class to calculate the overlap values for columns in a single HTM layer
    OverlapCalculator::OverlapCalculator(int potential_width, int potential_height, int columns_width, int columns_height,
                                         int input_width, int input_height, bool center_pot_synapses, float connected_perm,
                                         int min_overlap, bool wrap_input)
        : center_pot_synapses_(center_pot_synapses),
          wrap_input_(wrap_input),
          potential_width_(potential_width),
          potential_height_(potential_height),
          connected_perm_(connected_perm),
          min_overlap_(min_overlap),
          input_width_(input_width),
          input_height_(input_height),
          columns_width_(columns_width),
          columns_height_(columns_height),
          num_columns_(columns_width * columns_height),
          col_input_pot_syn_(num_columns_ * potential_height * potential_width, 0.0),
          col_inputs_shape_{columns_height_, columns_width_, potential_height_, potential_width_},
          col_pot_overlaps_(num_columns_, 0),
          step_x_(overlap_utils::get_step_sizes(input_width, input_height, columns_width, columns_height, potential_width, potential_height).first),
          step_y_(overlap_utils::get_step_sizes(input_width, input_height, columns_width, columns_height, potential_width, potential_height).second),
          neib_shape_(potential_height_, potential_width_),
          neib_step_(step_y_, step_x_),
          pot_syn_tie_breaker_(num_columns_ * potential_height * potential_width, 0.0),
          col_input_pot_syn_tie_(num_columns_ * potential_height * potential_width, 0.0),
          col_tie_breaker_(num_columns_, 0.0),
          con_syn_input_(num_columns_ * potential_height * potential_width, 0),
          col_overlaps_(num_columns_, 0),
          col_overlaps_tie_(num_columns_, 0.0)
    {
        // Initialize the random number generator
        std::random_device rd;
        rng_ = std::mt19937(rd());
        // Make the potential synapse tie breaker matrix.
        // Construct a tiebreaker matrix for the columns potential synapses.
        // It contains small values that help resolve any ties in potential overlap scores for columns.
        parallel_make_pot_syn_tie_breaker(pot_syn_tie_breaker_, std::make_pair(num_columns_, potential_height * potential_width));

        // Make the column input potential synapse tie breaker matrix.
        parallel_make_col_tie_breaker(col_tie_breaker_, columns_height_, columns_width_);

    }

    void OverlapCalculator::make_pot_syn_tie_breaker(std::vector<float> &pot_syn_tie_breaker, std::pair<int, int> size)
    {
        int input_height = size.first;
        int input_width = size.second;

        // Use the sum of all integer values less then or equal to formula.
        // This is because each row has its tie breaker values added together.
        // We want to make sure the result from adding the tie breaker values is
        // less then 0.5 but more then 0.0.
        float n = static_cast<float>(input_width);
        float norm_value = 0.5f / (n * (n + 1.0f) / 2.0f);

        std::vector<float> rows_tie(input_width);
        for (int i = 0; i < input_width; ++i)
        {
            rows_tie[i] = (i + 1) * norm_value;
        }

        // Use a seeded random sample of the above array
        std::mt19937 rng(1); // Mersenne Twister random number generator

        // Create a tiebreaker that changes for each row.
        for (int j = 0; j < input_height; ++j)
        {
            std::vector<float> row(input_width);
            std::sample(rows_tie.begin(), rows_tie.end(), row.begin(), input_width, rng);
            for (int i = 0; i < input_width; ++i)
            {
                pot_syn_tie_breaker[j * input_width + i] = row[i];
            }
        }
    }

    // Function to generate a matrix of tie-breaker values to resolve tie situations
    // in a grid. Each row in the grid has a unique pattern of tie-breaker values,
    // generated through random sampling, ensuring the sum of values in any row is less than 0.5.
    void OverlapCalculator::parallel_make_pot_syn_tie_breaker(std::vector<float> &pot_syn_tie_breaker, std::pair<int, int> size)
    {
        int input_height = size.first; // Number of rows in the tie-breaker matrix
        int input_width = size.second; // Number of columns in the tie-breaker matrix

        // Calculate the normalization value to ensure the sum of tie-breaker values
        // in any row is less than 0.5. This maintains the requirement for the total
        // to be less than 0.5 for any given row.
        float n = static_cast<float>(input_width);
        float norm_value = 0.5f / (n * (n + 1.0f) / 2.0f);

        // Pre-calculate tie-breaker values for a single row, each value being a multiple
        // of the normalization value. These are sequentially ordered from 1 to input_width,
        // each multiplied by the norm_value to ensure the sum of values in any row is less than 0.5.
        std::vector<float> rows_tie(input_width);
        for (int i = 0; i < input_width; ++i)
        {
            rows_tie[i] = (i + 1) * norm_value;
        }

        tf::Taskflow taskflow; // Taskflow object for parallel execution of row processing
        tf::Executor executor; // Executor to run the taskflow

        // Parallelly populate the tie-breaker matrix, each task handling one row.
        // This uses a fixed seed for reproducibility across runs.
        for (int j = 0; j < input_height; ++j)
        {
            taskflow.emplace([&pot_syn_tie_breaker, &rows_tie, j, input_width]()
                            {
                std::mt19937 rng(1); // Random number generator with a fixed seed for reproducibility
                std::vector<float> row(input_width);
                // Randomly sample values from rows_tie to create a unique pattern for each row,
                // similar to the Python implementation using random.sample.
                std::sample(rows_tie.begin(), rows_tie.end(), row.begin(), input_width, rng);
                for (int i = 0; i < input_width; ++i)
                {
                    // Populate the global tie-breaker matrix with the generated row values
                    pot_syn_tie_breaker[j * input_width + i] = row[i];
                } });
        }

        // Execute the taskflow and wait for all tasks to complete, ensuring the entire
        // tie-breaker matrix is populated.
        executor.run(taskflow).wait();
    }


    void OverlapCalculator::make_col_tie_breaker(std::vector<float> &tieBreaker, int columnsHeight, int columnsWidth)
    {
        int numColumns = columnsWidth * columnsHeight;

        // Create a vector of unique tie breaker values, similar to make_pot_syn_tie_breaker
        // Each value is a unique multiple of normValue to ensure no duplicates
        float normValue = 1.0f / float(2 * numColumns + 2);
        std::vector<float> uniqueValues(numColumns);
        for (int i = 0; i < numColumns; ++i) {
            uniqueValues[i] = (i + 1) * normValue;
        }
        
        // Create a vector of indices to shuffle
        std::vector<int> indices(numColumns);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Shuffle the indices using the same seed for reproducibility
        std::mt19937 gen(1);
        std::shuffle(indices.begin(), indices.end(), gen);

        // Assign the shuffled unique values to the tieBreaker
        for (int j = 0; j < tieBreaker.size(); j++)
        {
            int shuffledIndex = indices[j];
            tieBreaker[j] = uniqueValues[shuffledIndex];
        }
    }

    void OverlapCalculator::parallel_make_col_tie_breaker(std::vector<float> &tieBreaker, int columnsHeight, int columnsWidth)
    {
        int numColumns = columnsWidth * columnsHeight;
        
        // Create a vector of unique tie breaker values, similar to make_pot_syn_tie_breaker
        // Each value is a unique multiple of normValue to ensure no duplicates
        float normValue = 1.0f / float(2 * numColumns + 2);
        std::vector<float> uniqueValues(numColumns);
        for (int i = 0; i < numColumns; ++i) {
            uniqueValues[i] = (i + 1) * normValue;
        }
        
        // Create a vector of indices to shuffle
        std::vector<int> indices(numColumns);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Shuffle the indices using the same seed for reproducibility
        std::mt19937 gen(1);
        std::shuffle(indices.begin(), indices.end(), gen);

        tf::Taskflow taskflow;
        tf::Executor executor;

        taskflow.for_each_index(0, static_cast<int>(tieBreaker.size()), 1, [&tieBreaker, &uniqueValues, &indices](int j)
                                {
        // Use the shuffled index to get a unique value
        int shuffledIndex = indices[j];
        
        // Assign the unique shuffled value to this position
        tieBreaker[j] = uniqueValues[shuffledIndex]; })
            .name("make_col_tie_breaker_loop");

        executor.run(taskflow).wait();
    }

    void OverlapCalculator::check_new_input_params(
        const std::vector<float> &newColSynPerm,
        const std::pair<int, int> &colSynPerm_shape,
        const std::vector<int> &newInput,
        const std::pair<int, int> &inputGrid_shape)
    {
        assert(newInput.size() == inputGrid_shape.first * inputGrid_shape.second);
        assert(input_width_ == inputGrid_shape.second); // input_width_ equal to the number of columns in the input grid
        assert(input_height_ == inputGrid_shape.first); // input_height_ equal to the number of rows in the input grid
        assert(newColSynPerm.size() == colSynPerm_shape.first * colSynPerm_shape.second);
        assert(num_columns_ == colSynPerm_shape.first);
        assert(potential_width_ * potential_height_ == colSynPerm_shape.second);
    }

    // Return the potential synapse tie breaker values.
    std::vector<float> OverlapCalculator::get_pot_syn_tie_breaker()
    {
        return pot_syn_tie_breaker_;
    }

    // Return the column overlap values with tiebreaker values.
    std::vector<float> OverlapCalculator::get_col_overlaps()
    {
        // Return a copy of the col_overlaps_ vector.
        return col_overlaps_tie_;
    }

    const std::vector<int>& OverlapCalculator::get_col_pot_inputs() const
    {
        return col_input_pot_syn_;
    }

    std::pair<int, int> OverlapCalculator::get_col_pot_inputs_shape() const
    {
        return std::make_pair(num_columns_, potential_width_ * potential_height_);
    }

    const std::vector<float>& OverlapCalculator::get_col_pot_overlaps() const
    {
        return col_pot_overlaps_;
    }

    void OverlapCalculator::get_col_inputs(const std::vector<int> &inputGrid, const std::pair<int, int> &inputGrid_shape, std::vector<int> &col_inputs, tf::Taskflow &taskflow)
    {
        // This function uses a convolution function to return the inputs that each column potentially connects to.
        // inputGrid is a 1D vector simulating a 2D vector (matrix) of the input grid with the inputHeight and inputWidth
        // It outputs a simulated 4D vector using a 1D vector where each row (in the simulated 4D vector) represents the potential pool of inputs from the inputGrid to one "cortical column".
        // The output vector named "col_inputs" is a 1D vector of the flattened 4D vector.

        // Assert the output vector is the correct size.
        // The size of the output vector should be equal to the number of columns times the number of potential synapses.
        const int output_size = num_columns_ * potential_height_ * potential_width_;
        assert(col_inputs.size() == output_size);

        // IMPORTANT: We must iterate over the *column grid* here, not over the input grid.
        // If columns_width/height exceed input_width/height (with wrap_input enabled),
        // columns beyond the input extents should still map to wrapped input patches.
        overlap_utils::parallel_Columns2Neibs_1D(
            col_inputs,
            col_inputs_shape_,
            inputGrid,
            inputGrid_shape,
            neib_shape_,
            neib_step_,
            wrap_input_,
            center_pot_synapses_,
            taskflow);
    }

    void OverlapCalculator::calculate_overlap(const std::vector<float> &colSynPerm,
                                              const std::pair<int, int> &colSynPerm_shape,
                                              const std::vector<int> &inputGrid,
                                              const std::pair<int, int> &inputGrid_shape,
                                              bool debug)
    {
        check_new_input_params(colSynPerm, colSynPerm_shape, inputGrid, inputGrid_shape);

        // Create a Taskflow object to manage tasks and their dependencies.
        // There should be one taskflow object for the entire program.
        tf::Taskflow taskflow;
        tf::Executor executor;

        // Create sub taskflow graphs for each task that must be run after the previous task.
        tf::Taskflow tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8, tf9, tf10;

        // Calculate the inputs to each column.
        get_col_inputs(inputGrid, inputGrid_shape, col_input_pot_syn_, tf1);

        // Add a masked small tiebreaker value to the col_input_pot_syn_ scores (the inputs to the columns from potential synapses).
        overlap_utils::parallel_maskTieBreaker(col_input_pot_syn_, pot_syn_tie_breaker_, col_input_pot_syn_tie_, tf2);

        // Calculate the potential overlap scores for every column.
        // Sum the potential inputs for every column, Calculate the col_pot_overlaps_.
        overlap_utils::parallel_calcOverlap(col_input_pot_syn_tie_, num_columns_, potential_height_ * potential_width_, col_pot_overlaps_, tf3);

        // Calculate the connected synapse inputs for every column. The synapses who's permanence values are above the connected_perm_ threshold and are connected to an active input.
        // Calculate the con_syn_input_.
        overlap_utils::get_connected_syn_input(colSynPerm, col_input_pot_syn_, connected_perm_,
                                               num_columns_, potential_height_ * potential_width_,
                                               con_syn_input_, tf4);

        // Get the actual overlap scores for every column by summing the connected synapse inputs.
        // These are the sums for each cortical column of the number of connected synapses that are connected to an active input.
        overlap_utils::parallel_calcOverlap(con_syn_input_, num_columns_, potential_height_ * potential_width_, col_overlaps_, tf5);

        // Add a small tie breaker value to each cortical column's actual overlap score so draws in overlap scores can be resolved.
        overlap_utils::parallel_addVectors(col_overlaps_, col_tie_breaker_, col_overlaps_tie_, tf6);

        // Set the order of the tasks.
        tf::Task f1_task = taskflow.composed_of(tf1).name("get_col_inputs");
        tf::Task f2_task = taskflow.composed_of(tf2).name("parallel_maskTieBreaker");
        tf::Task f3_task = taskflow.composed_of(tf3).name("parallel_calcOverlap");
        tf::Task f4_task = taskflow.composed_of(tf4).name("get_connected_syn_input");
        tf::Task f5_task = taskflow.composed_of(tf5).name("parallel_calcOverlap");
        tf::Task f6_task = taskflow.composed_of(tf6).name("parallel_addVectors");
        f1_task.precede(f2_task);
        f2_task.precede(f3_task);
        f2_task.precede(f4_task);
        f4_task.precede(f5_task);
        f5_task.precede(f6_task);

        // dump the graph to a DOT file through std::cout
        if (debug) {
            taskflow.dump(std::cout);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Run the constructed taskflow graph.
        ///////////////////////////////////////////////////////////////////////////
        tf::Future<void> fu = executor.run(taskflow);
        fu.wait(); // block until the execution completes.
        executor.run(taskflow).wait();

        // Print debug information if requested
        if (debug) {
            // Print the col_input_pot_syn_ vector
            LOG(INFO, "inputGrid");
            overlap_utils::print_2d_vector(inputGrid, inputGrid_shape);
            // print step x y sizes step_x_
            LOG(INFO, "step_x_: " + std::to_string(step_x_));
            LOG(INFO, "step_y_: " + std::to_string(step_y_));
            LOG(INFO, "col_input_pot_syn_");
            overlap_utils::print_4d_vector(col_input_pot_syn_, col_inputs_shape_);
            LOG(INFO, "colSynPerm shape: " + std::to_string(colSynPerm_shape.first) + " " + std::to_string(colSynPerm_shape.second));
            overlap_utils::print_2d_vector(colSynPerm, colSynPerm_shape);

            LOG(INFO, "connected_perm_" + std::to_string(connected_perm_));
            // Print the con_syn_input_ vector
            const std::pair<int, int> con_syn_input_shape = {num_columns_, potential_height_ * potential_width_};
            LOG(INFO, "con_syn_input_ shape: " + std::to_string(con_syn_input_shape.first) + " " + std::to_string(con_syn_input_shape.second));
            overlap_utils::print_2d_vector(con_syn_input_, con_syn_input_shape);
            LOG(INFO, "col_pot_overlaps_ shape: " + std::to_string(col_pot_overlaps_.size()));
            overlap_utils::print_2d_vector(col_pot_overlaps_, {columns_width_, columns_height_});
            LOG(INFO, "col_overlaps_ shape: " + std::to_string(col_overlaps_.size()));
            overlap_utils::print_2d_vector(col_overlaps_, {columns_width_, columns_height_});
            LOG(INFO, "col_overlaps_tie_ shape: " + std::to_string(col_overlaps_tie_.size()));
            overlap_utils::print_1d_vector(col_overlaps_tie_);
            // Print the tie breaker values
            LOG(INFO, "col_tie_breaker_ shape: " + std::to_string(col_tie_breaker_.size()));
            overlap_utils::print_2d_vector(col_tie_breaker_, {columns_height_, columns_width_});

            LOG(DEBUG, "OverlapCalculator calculate_overlap Done.");
        }

        
    }

} // namespace overlap
