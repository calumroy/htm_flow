
#pragma once

#include <vector>
#include <tuple>

#include <taskflow/taskflow.hpp>

namespace gpu_overlap
{
    ///-----------------------------------------------------------------------------
    ///
    /// flattenVector              Flattens a 2D vector into a 1D vector.
    ///
    /// @param[in] vec2D           A 2D vector.
    ///
    /// @return                    A 1D vector.
    ///-----------------------------------------------------------------------------
    std::vector<int> flattenVector(const std::vector<std::vector<int>> &vec2D);
    std::vector<int> flattenVector(const std::vector<std::vector<std::vector<std::vector<int>>>> &vec4D);
    std::vector<std::vector<int>> unflattenVector(const std::vector<int> &vec1D, size_t numRows, size_t numCols);
    std::vector<std::vector<std::vector<std::vector<int>>>> unflattenVector(const std::vector<int> &vec1D, size_t numLayers, size_t numChannels, size_t numRows, size_t numCols);

    ///-----------------------------------------------------------------------------
    ///
    /// gpu_Images2Neibs           A function that performs a sliding window operation on a matrix.
    ///                            This function is designed to be called from the host. It allocates
    ///                            memory on the GPU, copies the input matrix to the GPU, launches the
    ///                            sliding_window_kernel kernel function, copies the output matrix from the GPU
    ///                            and frees the memory on the GPU.
    ///
    /// @param[in] input           A reference to the input matrix on the host. This is a 1D vector simulating a 2D matrix.
    /// @param[in] input_shape     A pair containing the number of rows and columns in the input matrix.
    /// @param[in] neib_shape      A pair containing the number of rows and columns in the neighbourhood.
    /// @param[in] neib_step       A pair containing the number of rows and columns to step the neighbourhood for each iteration.
    /// @param[in] wrap_mode       A flag indicating whether the neighbourhood should wrap around the input matrix.
    ///-----------------------------------------------------------------------------
    std::vector<int> gpu_Images2Neibs(
        const std::vector<int> &input,
        const std::pair<int, int> &input_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

    ///-----------------------------------------------------------------------------
    ///
    /// calculateOverlap   Calculate the overlap scores for a given input.
    ///                    This is how well each column connects to the active input.
    ///                    This is the main function of this class and its purpose.
    ///
    /// @param[in] colSynPerm   The synapse permanence values for each column. A 1D vector simulating a 2D vector of floats columns_width_ x columns_height_.
    /// @param[in] colSynPerm_shape  The shape of the colSynPerm vector height then width as a pair of ints.
    /// @param[in] inputGrid    The input grid as a 1D vector simulating a 2D vector of ints input_width_ x input_height_.
    /// @param[in] inputGrid_shape   The shape of the inputGrid vector height then width as a pair of ints.
    ///
    ///-----------------------------------------------------------------------------
    void calculate_overlap_gpu(const std::vector<float> &colSynPerm,
                               const std::pair<int, int> &colSynPerm_shape,
                               const std::vector<int> &inputGrid,
                               const std::pair<int, int> &inputGrid_shape,
                               const std::pair<int, int> &neib_shape,
                               const std::pair<int, int> &neib_step,
                               bool wrap_mode,
                               bool center_neigh);


    // DOES NOT WORK
    // ///-----------------------------------------------------------------------------
    // ///
    // /// gpu_Images2Neibs            The same function as the Images2Neibs function above but using 1D input vector that simulates a 2D input.
    // ///                             The width and height of the input matrix are passed as parameters as well.
    // ///                             Additionally the output is a 1D vector that simulates a 4D vector and it is passed in as
    // ///                             a reference parameter to avoid copying the output.
    // ///                             The output 1D vector simulates a 4D vector with the following dimensions:
    // ///                                 The first dimension represents the row index of the input matrix.
    // ///                                 The second dimension represents the column index of the input matrix.
    // ///                                 The third dimension represents the row index of the neighbourhood patch within the sliding window.
    // ///                                 The fourth dimension represents the column index of the neighbourhood patch within the sliding window.
    // /// @param[out] output        The output matrix (1D vector of ints). The output is passed in as a reference parameter to avoid copying the output.
    // /// @param[out] output_shape  The shape of the output matrix (numInputRows, numInputCols, numNeibRows, numNeibCols). Used to interpret the 1D output vector as a 4D matrix.
    // /// @param[in] input         The input matrix (2D vector of ints).
    // /// @param[in] input_shape   The shape of the input matrix (width, height) = (numcols, numrows). Used to interpret the 1D input vector as a 2D matrix.
    // /// @param[in] neib_shape    The shape of the neighbourhood.
    // /// @param[in] neib_step     The step size of the sliding window.
    // /// @param[in] wrap_mode     Whether to wrap the patches around the edges if true or if false use padding of zero on the edges.
    // /// @param[in] center_neigh  Whether to center the neighbourhood patch around the input element or not. If not "false" then place the neighbourhood patch top left corner at the input element.
    // /// @param[out] taskflow           The taskflow graph object. Used so this function can add its tasks to the graph. See C++ taskflow library.
    // ///-----------------------------------------------------------------------------
    // void gpu_Images2Neibs(
    //     std::vector<int> &output,
    //     std::vector<int> &output_shape,
    //     const std::vector<int> &input,
    //     const std::pair<int, int> &input_shape,
    //     const std::pair<int, int> &neib_shape,
    //     const std::pair<int, int> &neib_step,
    //     bool wrap_mode,
    //     bool center_neigh,
    //     tf::Taskflow &taskflow);

} // namespace gpu_overlap