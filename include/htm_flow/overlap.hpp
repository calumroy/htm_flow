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

namespace overlap
{
    // Define a function to convert 2D indices to 1D indices
    int flatten_index(int x, int y, int width);

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
    ///                       For each element (i, j) in the input matrix, the output vector contains a 3D patch of size neib_shape starting at element (i, j). If the sliding window operation is being performed in "wrap around" mode (i.e. mode is true), then the patch will wrap around the edges of the input matrix. If the sliding window operation is not being performed in "wrap around" mode (i.e. mode is false), then the patch will be padded with zeros at the edges of the input matrix if necessary.
    ///
    /// @param[in] input      The input matrix (2D vector of ints).
    /// @param[in] neib_shape The shape of the neighbourhood.
    /// @param[in] neib_step  The step size of the sliding window.
    /// @param[in] mode       Whether to center the patches around each location or not.
    ///                       If not (false), the top-left corner of the neighbourhood is used.
    /// @return               A 4D vector of ints. Each element stores the output patch for each element of the input matrix.
    ///-----------------------------------------------------------------------------
    template <typename T>
    std::vector<std::vector<std::vector<std::vector<T>>>> Images2Neibs(
        const std::vector<std::vector<T>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool mode);

    /// The same function as above but parallelized using Taskflow.
    template <typename T>
    std::vector<std::vector<std::vector<std::vector<T>>>> parallel_Images2Neibs(
        const std::vector<std::vector<T>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool mode);

    // Define a class to calculate the overlap values for columns in a single HTM layer
    class OverlapCalculator
    {
    public:
        // Constructor
        OverlapCalculator(int potential_width, int potential_height, int columns_width, int columns_height,
                          int input_width, int input_height, bool center_pot_synapses, float connected_perm,
                          int min_overlap, bool wrap_input);

        ///-----------------------------------------------------------------------------
        ///
        /// calculateOverlap   Calculate the overlap scores for a given input.
        ///                    This is how well each column connects to the active input.
        ///                    This is the main function of this class and its purpose.
        ///
        /// @param[in] colSynPerm   The synapse permanence values for each column.
        /// @param[in] inputGrid    The input grid (2D vector of ints).
        ///
        ///-----------------------------------------------------------------------------
        void calculate_overlap(const std::vector<std::vector<int>> &colSynPerm,
                               const std::vector<std::vector<int>> &inputGrid);

    private:
        // Calculate the potential synapses for a given column
        inline std::vector<std::tuple<int, int>> calculate_pot_syn(int column,
                                                                   const std::vector<std::vector<float>> &input_tensor);
        // Calculate the potential overlap scores for a given set of potential synapses
        inline std::vector<float> calculate_pot_overlap(const std::vector<std::tuple<int, int>> &pot_syn,
                                                        const std::vector<std::vector<float>> &input_tensor);
        // Calculate the actual overlap score for a given column based on its potential overlap scores
        int calculate_overlap(const std::vector<float> &pot_overlap,
                              const std::vector<std::tuple<int, int>> &pot_syn);
        // Get the step sizes for the potential synapses
        std::pair<int, int> get_step_sizes(int input_width, int input_height, int columns_width,
                                           int columns_height, int potential_width, int potential_height);
        // Make the potential synapse tie breaker matrix
        void make_pot_syn_tie_breaker(std::vector<std::vector<float>> &pot_syn_tie_breaker);

        // Check the new input parameter sizes to make sure they are the same as the inital input sizes.
        void check_new_input_params(
            const std::vector<std::vector<int>> &newColSynPerm,
            const std::vector<std::vector<int>> &newInput);

        // Member variables
        bool center_pot_synapses_;                              // Specifies if the potential synapses are centered over the columns
        bool wrap_input_;                                       // Use a wrap input function instead of padding the input to calcualte the overlap scores
        int potential_width_;                                   // Width of the potential synapses
        int potential_height_;                                  // Height of the potential synapses
        float connected_perm_;                                  // Probability that a column's potential synapses are connected to the input
        int min_overlap_;                                       // Minimum overlap required for a column to be considered for inhibition
        int input_width_;                                       // Width of the input
        int input_height_;                                      // Height of the input
        int columns_width_;                                     // Width of the columns
        int columns_height_;                                    // Height of the columns
        int num_columns_;                                       // Number of columns
        int step_x_;                                            // Step size in the x direction for the potential synapses
        int step_y_;                                            // Step size in the y direction for the potential synapses
        std::mt19937 rng_;                                      // Mersenne Twister random number generator
        std::vector<std::vector<float>> pot_syn_tie_breaker_;   // Potential synapse tie breaker matrix
        std::vector<std::vector<float>> col_input_pot_syn_tie_; // Potential synapse tie breaker matrix
        std::vector<float> col_tie_breaker_;                    // Potential synapse tie breaker matrix
    };

} // namespace overlap