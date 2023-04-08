///-----------------------------------------------------------------------------
///
/// @file overlap.hpp
///
/// @brief A calculator class to calculate the overlap scores for a group of HTM columns.
///        This is how well each column connects to a new active input (which is a 2d matrix of ones and zeros).
///
///-----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <taskflow/taskflow.hpp>

namespace overlap_utils
{
    // Define a function to convert 2D indices to 1D indices
    int flatten_index(int x, int y, int width);

    // Define a function to convert 1D indices to 2D indices
    std::tuple<int, int> unflatten_index(int index, int width);

    // Define a function to wrap 2D indices around the input dimensions
    std::tuple<int, int> wrap_indices(int x, int y, int input_width, int input_height);

    // Multiply two tensors element-wise
    std::vector<float> multiple(const std::vector<float> &a, const std::vector<float> &b);

    // Mask the grid with the tieBreaker values by multiplying them element-wise and then adding the result to the grid input.
    std::vector<float> maskTieBreaker(const std::vector<float> &grid, const std::vector<float> &tieBreaker);

    // Define a function to print out a 1D vector that is simulating a 2D vector.
    void print_2d_vector(const std::vector<int> &vec1D, const std::pair<int, int> &vec2D_shape);

    // Define a function to print out a 1D vector that is simulating a 4D vector.
    // vec4D_shape is a vector of size 4 that contains the shape of the 4D vector.
    void print_4d_vector(const std::vector<int> &vec1D, std::vector<int> &vec4D_shape);

    // Take a 1D vector and convert it to a 2D vector.
    std::vector<std::vector<int>> unflattenVector(const std::vector<int> &vec1D, size_t numRows, size_t numCols);

    // Take a 1D vector and convert it to a 4D vector.
    std::vector<std::vector<std::vector<std::vector<int>>>> unflattenVector(const std::vector<int> &vec1D, size_t numLayers, size_t numChannels, size_t numRows, size_t numCols);

    ///-----------------------------------------------------------------------------
    ///
    /// Images2Neibs       Creates a new matrix by applying a sliding window operation to `input`.
    ///                    The sliding window operation loops over points in `input` and stores
    ///                    a rectangular neighbourhood of each point.
    ///                    Throws an std::invalid_argument exception if the neighbourhood shape is
    ///                    larger than the input matrix shape.
    ///                    The output of this function is a 4D vector of type
    ///                    std::vector<std::vector<std::vector<std::vector<T>>>>, where T is the type of the elements
    ///                    in the input matrix. The dimensions of the output vector represent the following:
    ///
    ///                       The first dimension represents the row index of the input matrix.
    ///                       The second dimension represents the column index of the input matrix.
    ///                       The third dimension represents the row index of the neighbourhood patch within the sliding window.
    ///                       The fourth dimension represents the column index of the neighbourhood patch within the sliding window.
    ///
    ///                       For each element (i, j) in the input matrix, the output vector contains a 3D patch of size neib_shape starting at element (i, j).
    ///                       If the sliding window operation is being performed in "wrap around" mode (i.e. mode is true), then the patch will wrap around the edges of the input matrix. If the sliding window operation is not being performed in "wrap around" mode (i.e. mode is false), then the patch will be padded with zeros at the edges of the input matrix if necessary.
    ///
    /// @param[in] input         The input matrix (2D vector of ints).
    /// @param[in] neib_shape    The shape of the neighbourhood.
    /// @param[in] neib_step     The step size of the sliding window.
    /// @param[in] wrap_mode     Whether to wrap the patches around the edges if true or if false use padding of zero on the edges.
    /// @param[in] center_neigh  Whether to center the neighbourhood patch around the input element or not. If not "false" then place the neighbourhood patch top left corner at the input element.
    /// @return                  A 4D vector of ints. Each element stores the output patch for each element of the input matrix.
    ///-----------------------------------------------------------------------------
    template <typename T>
    std::vector<std::vector<std::vector<std::vector<T>>>> Images2Neibs(
        const std::vector<std::vector<T>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

    /// The same function as above but parallelized using Taskflow.
    template <typename T>
    std::vector<std::vector<std::vector<std::vector<T>>>> parallel_Images2Neibs(
        const std::vector<std::vector<T>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

    ///-----------------------------------------------------------------------------
    ///
    /// parallel_Images2Neibs_1D    The same function as the Images2Neibs function above but using 1D input vector that simulates a 2D input.
    ///                             The height and width of the input matrix are passed as parameters as well.
    ///                             Additionally the output is a 1D vector that simulates a 4D vector and it is passed in as
    ///                             a reference parameter to avoid copying the output.
    ///                             The output 1D vector simulates a 4D vector with the following dimensions:
    ///                                 The first dimension represents the row index of the input matrix.
    ///                                 The second dimension represents the column index of the input matrix.
    ///                                 The third dimension represents the row index of the neighbourhood patch within the sliding window.
    ///                                 The fourth dimension represents the column index of the neighbourhood patch within the sliding window.
    /// @param[out] output        The output matrix (1D vector of ints). The output is passed in as a reference parameter to avoid copying the output.
    /// @param[out] output_shape  The shape of the output matrix (numInputRows, numInputCols, numNeibRows, numNeibCols). Used to interpret the 1D output vector as a 4D matrix.
    /// @param[in] input         The input matrix (2D vector of ints).
    /// @param[in] input_shape   The shape of the input matrix (width, height) = (numcols, numrows). Used to interpret the 1D input vector as a 2D matrix.
    /// @param[in] neib_shape    The shape of the neighbourhood.
    /// @param[in] neib_step     The step size of the sliding window.
    /// @param[in] wrap_mode     Whether to wrap the patches around the edges if true or if false use padding of zero on the edges.
    /// @param[in] center_neigh  Whether to center the neighbourhood patch around the input element or not. If not "false" then place the neighbourhood patch top left corner at the input element.
    /// @return                  A 4D vector of ints. Each element stores the output patch for each element of the input matrix.
    ///-----------------------------------------------------------------------------
    template <typename T>
    void parallel_Images2Neibs_1D(
        std::vector<T> &output,
        std::vector<int> &output_shape,
        const std::vector<T> &input,
        const std::pair<int, int> &input_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

    ///-----------------------------------------------------------------------------
    ///
    /// parallel_maskTieBreaker   Applies a tie breaker to the input vector.
    ///                           The tie breaker is applied to the input vector element-wise.
    ///                           Multiply the tiebreaker values by the input grid then add them to it.
    //                            Since the grid contains ones and zeros some tiebreaker values are
    //                            masked out. This means the tie breaker will be different for each input
    //                            pattern.
    /// @param[in] grid           The input grid (1D vector) but could be simulating a 2D matrix.
    /// @param[in] tieBreaker     The tie breaker (1D vector must be same size as grid).
    /// @return                   A 1D vector of floats.
    template <typename T>
    std::vector<float> parallel_maskTieBreaker(const std::vector<T> &bool_grid, const std::vector<float> &tieBreaker);

    ///-----------------------------------------------------------------------------
    ///
    /// calcOverlap   Calculates the overlap between the input vector and the grid.
    ///               The input vector is a 1D vector that simulates a 2D matrix.
    ///               Calculate the potential overlap scores for every column.
    ///               Sum the potential inputs for every column.
    template <typename T>
    std::vector<float> parallel_calcOverlap(const std::vector<T> &b);

    // WIP
    ///-----------------------------------------------------------------------------
    ///
    /// get_connected_syn_input   Calculates the inputs to the "connected" cortical column synapses, the
    ///                           input to the column synapses that have a higher then a predefined permanence values.
    ///                           The cortical columns are represented as a simulated 2D matrix with n_rows and n_cols.
    ///                           The col_syn_perm and col_input_pot_syn are 1D vectors that simulate a 2D vector with the
    ///                           size number of cortical columns x number of potential synapses. Where each row in the
    ///                           col_syn_perm is the synapse permanence values for a column and each row in the col_input_pot_syn
    ///                           is the potential input to the synapses for a column.
    ///
    /// @param[in] col_syn_perm       The permanence values of the column synapses (1D vector simulating a 2D vector).
    /// @param[in] col_input_pot_syn  The potential input to the column synapses (1D vector of bools simulating a 2D vector of bools).
    /// @param[in] connected_perm     The permanence value that defines a synapse as "connected".
    /// @param[in] n_rows             The number of rows of the col_syn_perm or col_input_pot_syn simulated 2D vector inputs. This is equal to the number of cortical columns.
    /// @param[in] n_cols             The number of columns making up the col_syn_perm or col_input_pot_syn simulated 2D vector inputs. This is equal to the number of potential synapses.
    /// @param[out] check_conn        The input to the "connected" cortical column synapses (1D vector of bools simulating a 2D vector of bools).
    /// @param[in] taskflow           The taskflow graph object. Used so this function can add its tasks to the graph. See C++ taskflow library.
    template <typename T>
    void get_connected_syn_input(const std::vector<float> &col_syn_perm, const std::vector<T> &col_input_pot_syn,
                                 float connected_perm, int n_rows, int n_cols, std::vector<T> &check_conn,
                                 tf::Taskflow &taskflow);

    // -----------------------------------------------------------------------------
    // Header only implementations of the functions above.
    // templated functions must be defined in the header file.
    // -----------------------------------------------------------------------------

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
        const int input_rows = input_shape.first;
        const int input_cols = input_shape.second;

        // Check that the neighbourhood shape is valid.
        if (neib_shape.first > input_rows || neib_shape.second > input_cols)
        {
            throw std::invalid_argument("Neighbourhood shape must not be larger than the input matrix");
        }

        // Set the default step size to the neighbourhood shape.
        std::pair<int, int> step = neib_step;
        if (step.first == 0 && step.second == 0)
        {
            step = neib_shape;
        }

        // Check the output matrix size.
        const int no_output_rows = static_cast<int>(output_shape.at(0));
        const int no_output_cols = static_cast<int>(output_shape.at(1));
        const int output_channels = neib_shape.first * neib_shape.second;
        const int output_size = no_output_rows * no_output_cols * output_channels;

        // Define the maximum output shape (the total number of convolutions that can be achieved with the given input size and step sizes)
        const int max_output_rows = static_cast<int>(ceil(static_cast<float>(input_shape.first) / neib_step.first));
        const int max_output_cols = static_cast<int>(ceil(static_cast<float>(input_shape.second) / neib_step.second));

        // Assert the output vector is the correct size.
        assert(output.size() == output_size);

        tf::Taskflow taskflow;
        tf::Executor executor;

        // Parallelised over the rows of the input matrix, each row runs in it's own thread.
        taskflow.for_each_index(0, input_rows, neib_step.first, [&](int i)
                                {
        const int output_row = i / neib_step.first;

        // Make sure the output row is within the maximum output size.
        if (output_row < no_output_rows)
        {
            for (int j = 0; j < input_cols; j += neib_step.second)
            {
                const int output_col = j / neib_step.second;

                // Make sure the output column is within the maximum output size.
                if (output_col < no_output_cols)
                {
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
                                x = (x + input_rows) % input_rows;
                                y = (y + input_cols) % input_cols;
                            }

                            if (x >= 0 && x < input_rows && y >= 0 && y < input_cols)
                            {
                                const int output_channel = (ii * neib_shape.second) + jj;
                                const int output_index = ((output_row * no_output_cols) + output_col) * output_channels + output_channel;

                                output[output_index] = input[x * input_cols + y];
                            }
                            else
                            {
                                const int output_channel = (ii * neib_shape.second) + jj;
                                const int output_index = ((output_row * no_output_cols) + output_col) * output_channels + output_channel;

                                output[output_index] = 0;
                            }
                        }
                    }
                }
            }
        } });

        // Run the taskflow.
        executor.run(taskflow).get();
    }

    template <typename T>
    std::vector<float> parallel_maskTieBreaker(const std::vector<T> &bool_grid, const std::vector<float> &tieBreaker)
    {
        // Create a vector to hold the results of the computation.
        std::vector<float> result(bool_grid.size());

        int g_len = bool_grid.size();

        // Create a Taskflow object to manage tasks and their dependencies
        tf::Taskflow taskflow;
        tf::Executor executor;

        // Define a lambda function that multiplies each element of the input bool_grid
        // with the corresponding element of the tieBreaker values, and adds the result
        // to the corresponding element of the output vector.
        auto multiply_then_add = [&](int i)
        {
            // Output is a float.
            result[i] = static_cast<float>(bool_grid[i] * tieBreaker[i]) + bool_grid[i];
        };

        // Add a task to the Taskflow that applies the task_multiply_then_add function to each
        // element of the input bool_grid and the tieBreaker values.
        tf::Task task_multiply_then_add = taskflow.for_each_index(0, g_len, 1, multiply_then_add);

        // Execute the Taskflow and wait for it to finish.
        executor.run(taskflow).wait();

        return result;
    }

    template <typename T>
    std::vector<float> parallel_calcOverlap(const std::vector<T> &b)
    {
        // Create a vector to hold the results of the computation.
        std::vector<float> m(b.size());

        // Create a Taskflow object to manage tasks and their dependencies.
        tf::Taskflow taskflow;
        tf::Executor executor;

        // Define a lambda function that computes the sum of each column of the input b vector.
        auto compute_sum = [&](int j)
        {
            for (int i = 0; i < b.size(); i++)
            {
                if (i == j)
                {
                    m[i] += static_cast<float>(b[i]);
                }
            }
        };

        // Add a task to the Taskflow that applies the compute_sum function to each column of the input b vector.
        tf::Task task_compute_sum = taskflow.for_each_index(0, static_cast<int>(b.size()), 1, compute_sum);

        // Execute the Taskflow and wait for it to finish.
        executor.run(taskflow).wait();

        return m;
    }

    template <typename T>
    void get_connected_syn_input(const std::vector<float> &col_syn_perm, const std::vector<T> &col_input_pot_syn,
                                 float connected_perm, int n_rows, int n_cols, std::vector<T> &check_conn,
                                 tf::Taskflow &taskflow)
    {
        tf::Task check_conn_task = taskflow.emplace([&col_syn_perm, &col_input_pot_syn, connected_perm, n_rows, n_cols, &check_conn, &taskflow]()
                                                    { taskflow.for_each_index(0, n_rows * n_cols, 1, [&](int i)
                                                                              {
                const int row = i / n_cols;
                const int col = i % n_cols;
                // use `row` and `col` to compute the required values
                int index = row * n_cols + col;
                if (col_syn_perm[index] > connected_perm) 
                {
                    check_conn[index] = col_input_pot_syn[index];  
                } 
                else 
                {
                    check_conn[index] = 0;  // Set to 0 as the cortical column synape is not considered to be connected.
                } }); });
        tf::Task load_in1_task = taskflow.emplace([&col_syn_perm, n_rows, n_cols]() {});
        tf::Task load_in2_task = taskflow.emplace([&col_input_pot_syn, n_rows, n_cols]() {});

        check_conn_task.succeed(load_in1_task, load_in2_task);
    }

} // namespace overlap_utils