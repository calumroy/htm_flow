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

    // Multiply two tensors element-wise
    std::vector<float> multiple(const std::vector<float> &a, const std::vector<float> &b);

    // Mask the grid with the tieBreaker values by multiplying them element-wise and then adding the result to the grid input.
    std::vector<float> maskTieBreaker(const std::vector<float> &grid, const std::vector<float> &tieBreaker);

} // namespace overlap_utils