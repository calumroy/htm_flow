
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <taskflow/taskflow.hpp>

#include <utilities/logger.hpp>
#include <htm_flow/overlap_utils.hpp>

namespace overlap_utils
{

    // Define a function to convert 2D indices to 1D indices
    int flatten_index(int x, int y, int width)
    {
        return x + y * width;
    }

    // Define a function to convert 1D indices to 2D indices
    std::tuple<int, int> unflatten_index(int index, int width)
    {
        int x = index % width;
        int y = (index - x) / width;
        return std::make_tuple(x, y);
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
        bool wrap_mode,
        bool center_neigh)
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

                        // If the "center_neigh" flag is set, center the neighbourhood
                        // over the current element in the input matrix.
                        if (center_neigh)
                        {
                            x = i + ii - neib_shape.first / 2;
                            y = j + jj - neib_shape.second / 2;
                        }

                        if (wrap_mode)
                        {
                            x = (x + rows) % rows;
                            y = (y + cols) % cols;
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
        bool wrap_mode,
        bool center_neigh)
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
        // Initialize the taskflow output matrix size.
        // It should contain the number of rows of the ceiling of ((float)input.rows / (float)step.first))
        // each row runs in it's own thread.

        int N = static_cast<int>(ceil(static_cast<float>(rows) / step.first));  // Number of rows in output matrix
        int M = static_cast<int>(ceil(static_cast<float>(cols) / step.second)); // Number of columns in output matrix
        int O = neib_shape.first;                                               // Number of rows in each patch
        int P = neib_shape.second;                                              // Number of columns in each patch

        output.resize(N);
        for (int i = 0; i < N; ++i)
        {
            output[i].resize(M);
            for (int j = 0; j < M; ++j)
            {
                output[i][j].resize(O);
                for (int k = 0; k < O; ++k)
                {
                    output[i][j][k].resize(P);
                }
            }
        }

        tf::Taskflow taskflow;
        tf::Executor executor;

        taskflow.for_each_index(0, rows, step.first, [&](int i)
                                {
                                        // Add debug logging
                                        //LOG(DEBUG, "Thread i = " + std::to_string(i));
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

                                                    // If the "center_neigh" flag is set, center the neighbourhood
                                                    // over the current element in the input matrix.
                                                    if (center_neigh)
                                                    {
                                                        x = i + ii - neib_shape.first / 2;
                                                        y = j + jj - neib_shape.second / 2;
                                                    }

                                                    if (wrap_mode)
                                                    {
                                                        x = (x + rows) % rows;
                                                        y = (y + cols) % cols;
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
                                        // Set the output matrix for this row. 
                                        // Divide i the row index by the step size to get the correct row index to update.
                                        output[i / step.first] = row_output; });

        // Run the taskflow.
        executor.run(taskflow).get();

        return output;
    }

    // The same function as above as the parallelized Taskflow Images2Neibs but using 1D input and output vectors as parameters.
    template <typename T>
    void parallel_Images2Neibs_1D(
        std::vector<T> &output,
        std::vector<int> &output_shape,
        const std::vector<T> &input,
        const std::pair<int, int> &input_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh)
    {
        // Determine the dimensions of the input matrix.
        const int rows = input_shape.first;
        const int cols = input_shape.second;

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
        const int output_rows = static_cast<int>(ceil(static_cast<float>(rows) / step.first));
        const int output_cols = static_cast<int>(ceil(static_cast<float>(cols) / step.second));
        const int output_channels = neib_shape.first * neib_shape.second;
        const int output_size = output_rows * output_cols * output_channels;

        // If the output vector is not the correct size, resize it.
        if (output.size() != output_size)
        {
            output.resize(output_size, 0);
        }

        tf::Taskflow taskflow;
        tf::Executor executor;

        taskflow.for_each_index(0, rows, step.first, [&](int i)
                                {
        const int output_row = i / step.first;

        for (int j = 0; j < cols; j += step.second)
        {
            const int output_col = j / step.second;

            for (int ii = 0; ii < neib_shape.first; ++ii)
            {
                for (int jj = 0; jj < neib_shape.second; ++jj)
                {
                    int x = i + ii;
                    int y = j + jj;

                    // If the "center_neigh" flag is set, center the neighbourhood
                    // over the current element in the input matrix.
                    if (center_neigh)
                    {
                        x = i + ii - neib_shape.first / 2;
                        y = j + jj - neib_shape.second / 2;
                    }

                    if (wrap_mode)
                    {
                        x = (x + rows) % rows;
                        y = (y + cols) % cols;
                    }

                    if (x >= 0 && x < rows && y >= 0 && y < cols)
                    {
                        const int output_channel = (ii * neib_shape.second) + jj;
                        const int output_index = ((output_row * output_cols) + output_col) * output_channels + output_channel;

                        output[output_index] = input[x * cols + y];
                    }
                    else
                    {
                        const int output_channel = (ii * neib_shape.second) + jj;
                        const int output_index = ((output_row * output_cols) + output_col) * output_channels + output_channel;

                        output[output_index] = 0;
                    }
                }
            }
        } });

        // Run the taskflow.
        executor.run(taskflow).get();

        output_shape = {output_rows,
                        output_cols,
                        neib_shape.first,
                        neib_shape.second};
    }

    // Multiply two tensors element-wise
    std::vector<float>
    multiple(const std::vector<float> &a, const std::vector<float> &b)
    {
        std::vector<float> result;
        result.reserve(a.size());

        for (size_t i = 0; i < a.size(); ++i)
        {
            result.push_back(a[i] * b[i]);
        }

        return result;
    }

    // Apply tie breaker values to an input grid by multiplying the grid by the tie breaker values and adding the result to the grid.
    std::vector<float> maskTieBreaker(const std::vector<float> &grid, const std::vector<float> &tieBreaker)
    {
        std::vector<float> multi_vals;
        multi_vals.reserve(grid.size());

        for (size_t i = 0; i < grid.size(); ++i)
        {
            multi_vals.push_back(grid[i] * tieBreaker[i]);
        }

        std::vector<float> result;
        result.reserve(grid.size());

        for (size_t i = 0; i < grid.size(); ++i)
        {
            result.push_back(grid[i] + multi_vals[i]);
        }

        return result;
    }
} // namespace overlap_utils