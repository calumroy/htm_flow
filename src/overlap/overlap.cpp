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
          step_x_(get_step_sizes(input_width, input_height, columns_width, columns_height, potential_width, potential_height).first),
          step_y_(get_step_sizes(input_width, input_height, columns_width, columns_height, potential_width, potential_height).second),
          pot_syn_tie_breaker_(num_columns_ * potential_height * potential_width, 0.0),
          col_input_pot_syn_tie_(num_columns_ * potential_height * potential_width, 0.0),
          col_tie_breaker_(num_columns_, 0.0),
          con_syn_input_(num_columns_ * potential_height * potential_width, 0)
    {
        // Initialize the random number generator
        std::random_device rd;
        rng_ = std::mt19937(rd());
        // Make the potential synapse tie breaker matrix.
        // Construct a tiebreaker matrix for the columns potential synapses.
        // It contains small values that help resolve any ties in potential overlap scores for columns.
        make_pot_syn_tie_breaker(pot_syn_tie_breaker_, std::make_pair(num_columns_, potential_height * potential_width));

        // Make the column input potential synapse tie breaker matrix.
        make_col_tie_breaker(col_tie_breaker_, columns_height_, columns_width_);

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

    std::pair<int, int> OverlapCalculator::get_step_sizes(int input_width, int input_height,
                                                          int col_width, int col_height,
                                                          int pot_width, int pot_height)
    {
        // Work out how large to make the step sizes so all of the
        // inputGrid can be covered as best as possible by the columns
        // potential synapses.
        int step_x = static_cast<int>(std::round(static_cast<float>(input_width) / static_cast<float>(col_width)));
        int step_y = static_cast<int>(std::round(static_cast<float>(input_height) / static_cast<float>(col_height)));

        // The step sizes may need to be increased if the potential sizes are too small.
        if (pot_width + (col_width - 1) * step_x < input_width)
        {
            // Calculate how many of the input elements cannot be covered with the current step_x value.
            int uncovered_x = (input_width - (pot_width + (col_width - 1) * step_x));
            // Use this to update the step_x value so all input elements are covered.
            step_x = step_x + static_cast<int>(std::ceil(static_cast<float>(uncovered_x) / static_cast<float>(col_width - 1)));
        }

        if (pot_height + (col_height - 1) * step_y < input_height)
        {
            int uncovered_y = (input_height - (pot_height + (col_height - 1) * step_y));
            step_y = step_y + static_cast<int>(std::ceil(static_cast<float>(uncovered_y) / static_cast<float>(col_height - 1)));
        }

        return std::make_pair(step_x, step_y);
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

    void OverlapCalculator::get_col_inputs(std::vector<int> &col_inputs, const std::vector<int> &inputGrid, const std::pair<int, int> &inputGrid_shape)
    {
        // This function uses a convolution function to return the inputs that each column potentially connects to.
        // inputGrid is a 1D vector simulating a 2D vector (matrix) of the input grid with the inputHeight and inputWidth
        // It outputs a simulated 4D vector using a 1D vector where each row (in the simulated 4D vector) represents the potential pool of inputs from the inputGrid to one "cortical column".
        // The output vector named "col_inputs" is a 1D vector of the flattened 4D vector.

        // Assert the output vector is the correct size.
        // The size of the output vector should be equal to the number of columns times the number of potential synapses.
        const int output_size = num_columns_ * potential_height_ * potential_width_;
        assert(col_inputs.size() == output_size);

        std::vector<int> col_inputs_shape = {columns_height_,
                                             columns_width_,
                                             potential_height_,
                                             potential_width_};

        // Define the neighbourhood shape and step for the parallel_Images2Neibs_1D
        const std::pair<int, int> neib_shape = std::make_pair(potential_height_, potential_width_);
        const std::pair<int, int> neib_step = std::make_pair(step_y_, step_x_);

        // Call the parallel_Images2Neibs_1D function
        overlap_utils::parallel_Images2Neibs_1D(col_inputs, col_inputs_shape, inputGrid, inputGrid_shape, neib_shape, neib_step, wrap_input_, center_pot_synapses_);
    }

    void OverlapCalculator::calculate_overlap(const std::vector<float> &colSynPerm,
                                              const std::pair<int, int> &colSynPerm_shape,
                                              const std::vector<int> &inputGrid,
                                              const std::pair<int, int> &inputGrid_shape)
    {

        // Create a Taskflow object to manage tasks and their dependencies.
        // There should be one taskflow object for the entire program.
        tf::Taskflow taskflow;
        tf::Executor executor;

        check_new_input_params(colSynPerm, colSynPerm_shape, inputGrid, inputGrid_shape);

        // Calculate the inputs to each column
        get_col_inputs(col_input_pot_syn_, inputGrid, inputGrid_shape);

        // TODO remove this
        // Print the col_input_pot_syn_ vector
        LOG(INFO, "inputGrid");
        overlap_utils::print_2d_vector(inputGrid, inputGrid_shape);
        std::vector<int> col_input_pot_shape = {columns_height_, columns_width_, potential_height_, potential_width_};
        LOG(INFO, "col_input_pot_syn_");
        overlap_utils::print_4d_vector(col_input_pot_syn_, col_input_pot_shape);
        LOG(INFO, "colSynPerm shape: " + std::to_string(colSynPerm_shape.first) + " " + std::to_string(colSynPerm_shape.second));
        overlap_utils::print_2d_vector(colSynPerm, colSynPerm_shape);

        // Add a masked small tiebreaker value to the col_input_pot_syn_ scores (the inputs to the columns from potential synapses).
        col_input_pot_syn_tie_ = overlap_utils::parallel_maskTieBreaker(col_input_pot_syn_, pot_syn_tie_breaker_);

        // Calculate the potential overlap scores for every column.
        // Sum the potential inputs for every column.
        col_pot_overlaps_ = overlap_utils::parallel_calcOverlap(col_input_pot_syn_tie_);

        overlap_utils::get_connected_syn_input(colSynPerm, col_input_pot_syn_, connected_perm_,
                                               num_columns_, potential_height_ * potential_width_,
                                               con_syn_input_, taskflow);

        // TODO remove this
        // Print the con_syn_input_ vector
        const std::pair<int, int> con_syn_input_shape = {num_columns_, potential_height_ * potential_width_};
        LOG(INFO, "con_syn_input_ shape: " + std::to_string(con_syn_input_shape.first) + " " + std::to_string(con_syn_input_shape.second));
        overlap_utils::print_2d_vector(con_syn_input_, con_syn_input_shape);

        // std::vector<std::vector<int>> connectedSynInputs =
        //     getConnectedSynInput(colSynPerm, colInputPotSyn);
        // std::vector<std::vector<int>> colOverlapVals = calcOverlap(connectedSynInputs);
        // colOverlapVals = addVectTieBreaker(colOverlapVals, colTieBreaker);
        // return std::make_pair(colOverlapVals, colInputPotSyn);

        LOG(DEBUG, "OverlapCalculator calculate_overlap Done.");
    }

} // namespace overlap
