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

        LOG(DEBUG, "OverlapCalculator Constructor Done.");
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

    void OverlapCalculator::parallel_make_pot_syn_tie_breaker(std::vector<float> &pot_syn_tie_breaker, std::pair<int, int> size)
    {
        int input_height = size.first;
        int input_width = size.second;

        float n = static_cast<float>(input_width);
        float norm_value = 0.5f / (n * (n + 1.0f) / 2.0f);

        std::vector<float> rows_tie(input_width);
        for (int i = 0; i < input_width; ++i)
        {
            rows_tie[i] = (i + 1) * norm_value;
        }

        tf::Taskflow taskflow;
        tf::Executor executor;

        // Create a task for each row
        for (int j = 0; j < input_height; ++j)
        {
            taskflow.emplace([&pot_syn_tie_breaker, &rows_tie, j, input_width]()
                             {
            std::mt19937 rng(1);
            std::vector<float> row(input_width);
            std::sample(rows_tie.begin(), rows_tie.end(), row.begin(), input_width, rng);
            for (int i = 0; i < input_width; ++i)
            {
                pot_syn_tie_breaker[j * input_width + i] = row[i];
            } });
        }

        // Execute the taskflow
        executor.run(taskflow).wait();
    }

    void OverlapCalculator::make_col_tie_breaker(std::vector<float> &tieBreaker, int columnsHeight, int columnsWidth)
    {
        int numColumns = columnsWidth * columnsHeight;

        // Make a vector of tiebreaker values to add to the columns overlap values vector.
        float normValue = 1.0f / float(2 * numColumns + 2);

        // Initialise a random seed so we can get the same random numbers.
        // This means the tie breaker will be the same each time but it will
        // be randomly distributed over the cells.
        std::mt19937 gen(1);

        // Create a tiebreaker that is not biased to either side of the columns grid.
        for (int j = 0; j < tieBreaker.size(); j++)
        {
            // The tieBreaker is a flattened vector of the columns overlaps.

            // workout the row and col number of the non flattened matrix.
            int rowNum = std::floor(j / columnsWidth);
            int colNum = j % columnsWidth;
            if (std::uniform_real_distribution<float>(0, 1)(gen) > 0.5f)
            {
                // Some positions are bias to the bottom left
                tieBreaker[j] = ((rowNum + 1) * columnsWidth + (columnsWidth - colNum - 1)) * normValue;
            }
            else
            {
                // Some Positions are bias to the top right
                tieBreaker[j] = ((columnsHeight - rowNum) * columnsWidth + colNum) * normValue;
            }
        }
    }

    void OverlapCalculator::parallel_make_col_tie_breaker(std::vector<float> &tieBreaker, int columnsHeight, int columnsWidth)
    {
        int numColumns = columnsWidth * columnsHeight;
        float normValue = 1.0f / float(2 * numColumns + 2);

        std::mt19937 gen(1);

        tf::Taskflow taskflow;
        tf::Executor executor;

        taskflow.for_each_index(0, static_cast<int>(tieBreaker.size()), 1, [&tieBreaker, columnsHeight, columnsWidth, normValue, &gen](int j)
                                {
        int rowNum = std::floor(j / columnsWidth);
        int colNum = j % columnsWidth;

        if (std::uniform_real_distribution<float>(0, 1)(gen) > 0.5f) {
            // Some positions are bias to the bottom left
            tieBreaker[j] = ((rowNum + 1) * columnsWidth + (columnsWidth - colNum - 1)) * normValue;
        } else {
            // Some Positions are bias to the top right
            tieBreaker[j] = ((columnsHeight - rowNum) * columnsWidth + colNum) * normValue;
        } })
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

    // Return the column overlap values.
    std::vector<int> OverlapCalculator::get_col_overlaps()
    {
        // Return a copy of the col_overlaps_ vector.
        return col_overlaps_;
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

        // Call the parallel_Images2Neibs_1D function
        overlap_utils::parallel_Images2Neibs_1D(col_inputs, col_inputs_shape_, inputGrid, inputGrid_shape, neib_shape_, neib_step_, wrap_input_, center_pot_synapses_, taskflow);
    }

    void OverlapCalculator::calculate_overlap(const std::vector<float> &colSynPerm,
                                              const std::pair<int, int> &colSynPerm_shape,
                                              const std::vector<int> &inputGrid,
                                              const std::pair<int, int> &inputGrid_shape)
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
        taskflow.dump(std::cout);

        ///////////////////////////////////////////////////////////////////////////
        // Run the constructed taskflow graph.
        ///////////////////////////////////////////////////////////////////////////
        tf::Future<void> fu = executor.run(taskflow);
        fu.wait(); // block until the execution completes.
        executor.run(taskflow).wait();

        // // TODO remove these print outs.
        // // Print the col_input_pot_syn_ vector
        // LOG(INFO, "inputGrid");
        // overlap_utils::print_2d_vector(inputGrid, inputGrid_shape);
        // // print step x y sizes step_x_
        // LOG(INFO, "step_x_: " + std::to_string(step_x_));
        // LOG(INFO, "step_y_: " + std::to_string(step_y_));
        // LOG(INFO, "col_input_pot_syn_");
        // overlap_utils::print_4d_vector(col_input_pot_syn_, col_inputs_shape_);
        // LOG(INFO, "colSynPerm shape: " + std::to_string(colSynPerm_shape.first) + " " + std::to_string(colSynPerm_shape.second));
        // overlap_utils::print_2d_vector(colSynPerm, colSynPerm_shape);

        // // TODO remove this
        // LOG(INFO, "connected_perm_" + std::to_string(connected_perm_));
        // // Print the con_syn_input_ vector
        // const std::pair<int, int> con_syn_input_shape = {num_columns_, potential_height_ * potential_width_};
        // LOG(INFO, "con_syn_input_ shape: " + std::to_string(con_syn_input_shape.first) + " " + std::to_string(con_syn_input_shape.second));
        // overlap_utils::print_2d_vector(con_syn_input_, con_syn_input_shape);
        // LOG(INFO, "col_pot_overlaps_ shape: " + std::to_string(col_pot_overlaps_.size()));
        // overlap_utils::print_1d_vector(col_pot_overlaps_);
        // LOG(INFO, "col_overlaps_ shape: " + std::to_string(col_overlaps_.size()));
        // overlap_utils::print_1d_vector(col_overlaps_);
        // LOG(INFO, "col_overlaps_tie_ shape: " + std::to_string(col_overlaps_tie_.size()));
        // overlap_utils::print_1d_vector(col_overlaps_tie_);

        LOG(DEBUG, "OverlapCalculator calculate_overlap Done.");
    }

} // namespace overlap
