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

namespace overlap
{

    // Define a function to convert 2D indices to 1D indices
    int flatten_index(int x, int y, int width)
    {
        return x + y * width;
    }

    // Define a function to wrap 2D indices around the input dimensions
    std::tuple<int, int> wrap_indices(int x, int y, int input_width, int input_height)
    {
        x = x % input_width;
        if (x < 0)
            x += input_width;
        y = y % input_height;
        if (y < 0)
            y += input_height;
        return std::make_tuple(x, y);
    }

    // Creates a new matrix by applying a sliding window operation to `input`.
    // The sliding window operation loops over points in `input` and stores
    // a rectangular neighbourhood of each point.
    template <typename T>
    std::vector<std::vector<std::vector<std::vector<T>>>> Images2Neibs(
        const std::vector<std::vector<T>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool mode)
    {
        // Determine the dimensions of the input matrix.
        const int rows = input.size();
        const int cols = input[0].size();

        // Check that the neighbourhood shape is valid.
        if (neib_shape.first > rows || neib_shape.second > cols)
        {
            throw std::invalid_argument("Neighbourhood shape must not be larger than the input matrix");
        }

        // Set the default step size to the neighbourhood shape.
        std::pair<int, int> step = neib_step;
        if (step.first == 0 && step.second == 0)
        {
            step = neib_shape;
        }

        // Create the output matrix.
        std::vector<std::vector<std::vector<std::vector<T>>>> output;

        // Apply the sliding window operation.
        for (int i = 0; i < rows; i += step.first)
        {
            std::vector<std::vector<std::vector<T>>> row_output;
            for (int j = 0; j < cols; j += step.second)
            {
                std::vector<std::vector<T>> patch;
                for (int ii = 0; ii < neib_shape.first; ++ii)
                {
                    std::vector<T> row;
                    for (int jj = 0; jj < neib_shape.second; ++jj)
                    {
                        int x = i + ii;
                        int y = j + jj;
                        if (mode)
                        {
                            x = x % rows;
                            y = y % cols;
                        }
                        if (x >= 0 && x < rows && y >= 0 && y < cols)
                        {
                            row.push_back(input[x][y]);
                        }
                        else
                        {
                            row.push_back(0);
                        }
                    }
                    patch.push_back(row);
                }
                row_output.push_back(patch);
            }
            output.push_back(row_output);
        }
        return output;
    }

    // The same function as above but parallelized using Taskflow.
    template <typename T>
    std::vector<std::vector<std::vector<std::vector<T>>>> parallel_Images2Neibs(
        const std::vector<std::vector<T>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool mode)
    {
        // Determine the dimensions of the input matrix.
        const int rows = input.size();
        const int cols = input[0].size();

        // Check that the neighbourhood shape is valid.
        if (neib_shape.first > rows || neib_shape.second > cols)
        {
            throw std::invalid_argument("Neighbourhood shape must not be larger than the input matrix");
        }

        // Set the default step size to the neighbourhood shape.
        std::pair<int, int> step = neib_step;
        if (step.first == 0 && step.second == 0)
        {
            step = neib_shape;
        }

        // Create the output matrix.
        std::vector<std::vector<std::vector<std::vector<T>>>> output;

        tf::Taskflow taskflow;
        tf::Executor executor;

        taskflow.for_each(rows, [&](int i)
                          {
    std::vector<std::vector<std::vector<T>>> row_output;
    for (int j = 0; j < cols; j += step.second)
    {
        std::vector<std::vector<T>> patch;
        for (int ii = 0; ii < neib_shape.first; ++ii)
        {
            std::vector<T> row;
            for (int jj = 0; jj < neib_shape.second; ++jj)
            {
                int x = i + ii;
                int y = j + jj;
                if (mode)
                {
                    x = x % rows;
                    y = y % cols;
                }
                if (x >= 0 && x < rows && y >= 0 && y < cols)
                {
                    row.push_back(input[x][y]);
                }
                else
                {
                    row.push_back(0);
                }
            }
            patch.push_back(row);
        }
        row_output.push_back(patch);
    }
    output.push_back(row_output); });

        // Run the taskflow.
        executor.run(taskflow).get();

        return output;
    }

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
          step_x_(get_step_sizes(input_width, input_height, columns_width, columns_height, potential_width, potential_height).first),
          step_y_(get_step_sizes(input_width, input_height, columns_width, columns_height, potential_width, potential_height).second),
          pot_syn_tie_breaker_(num_columns_, std::vector<float>(potential_height * potential_width, 0.0)),
          col_input_pot_syn_tie_(num_columns_, std::vector<float>(potential_height * potential_width, 0.0)),
          col_tie_breaker_(num_columns_, 0.0)
    {
        // Initialize the random number generator
        std::random_device rd;
        rng_ = std::mt19937(rd());
        // Make the potential synapse tie breaker matrix
        make_pot_syn_tie_breaker(pot_syn_tie_breaker_);

        LOG(DEBUG, "OverlapCalculator Constructor Done.");
    }

    void OverlapCalculator::make_pot_syn_tie_breaker(std::vector<std::vector<float>> &pot_syn_tie_breaker)
    {
        int input_height = pot_syn_tie_breaker.size();
        int input_width = pot_syn_tie_breaker[0].size();

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
            pot_syn_tie_breaker[j] = std::move(row);
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
        const std::vector<std::vector<int>> &newColSynPerm,
        const std::vector<std::vector<int>> &newInput)
    {
        assert(input_width_ == newInput[0].size());
        assert(input_height_ == newInput.size());
        assert(potential_width_ * potential_height_ == newColSynPerm[0].size());
        assert(num_columns_ == newColSynPerm.size());
    }

    void OverlapCalculator::calculate_overlap(const std::vector<std::vector<int>> &colSynPerm,
                                              const std::vector<std::vector<int>> &inputGrid)
    {
        check_new_input_params(colSynPerm, inputGrid);
        // colInputPotSyn = getColInputs(inputGrid);
        // colInputPotSynTie = maskTieBreaker(colInputPotSyn, potSynTieBreaker);
        // colPotOverlaps = calcOverlap(colInputPotSynTie);
        // std::vector<std::vector<int>> connectedSynInputs =
        //     getConnectedSynInput(colSynPerm, colInputPotSyn);
        // std::vector<std::vector<int>> colOverlapVals = calcOverlap(connectedSynInputs);
        // colOverlapVals = addVectTieBreaker(colOverlapVals, colTieBreaker);
        // return std::make_pair(colOverlapVals, colInputPotSyn);
    }

} // namespace overlap
