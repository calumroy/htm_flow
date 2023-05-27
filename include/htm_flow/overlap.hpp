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

#include <htm_flow/overlap_utils.hpp>

namespace overlap
{

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
        /// @param[in] colSynPerm   The synapse permanence values for each column. A 1D vector simulating a 2D vector of floats columns_width_ x columns_height_.
        /// @param[in] colSynPerm_shape  The shape of the colSynPerm vector height then width as a pair of ints.
        /// @param[in] inputGrid    The input grid as a 1D vector simulating a 2D vector of ints input_width_ x input_height_.
        /// @param[in] inputGrid_shape   The shape of the inputGrid vector height then width as a pair of ints.
        ///
        ///-----------------------------------------------------------------------------
        void calculate_overlap(const std::vector<float> &colSynPerm,
                               const std::pair<int, int> &colSynPerm_shape,
                               const std::vector<int> &inputGrid,
                               const std::pair<int, int> &inputGrid_shape);

    private:
        // // Calculate the potential synapses for a given column
        // inline std::vector<std::tuple<int, int>> calculate_pot_syn(int column,
        //                                                            const std::vector<std::vector<float>> &input_tensor);
        // // Calculate the potential overlap scores for a given set of potential synapses
        // inline std::vector<float> calculate_pot_overlap(const std::vector<std::tuple<int, int>> &pot_syn,
        //                                                 const std::vector<std::vector<float>> &input_tensor);
        // // Calculate the actual overlap score for a given column based on its potential overlap scores
        // int calculate_overlap(const std::vector<float> &pot_overlap,
        //                       const std::vector<std::tuple<int, int>> &pot_syn);

        // Get the step sizes for the potential synapses
        std::pair<int, int> get_step_sizes(int input_width, int input_height, int columns_width,
                                           int columns_height, int potential_width, int potential_height);
        ///------------------------------------------------------------------------------------
        ///
        /// make_pot_syn_tie_breaker - Create a tie breaker matrix for potential synapses.
        ///
        /// This function takes a 1D vector simulating a 2D vector of floats representing the self.colInputPotSyn grid and creates a simulated 2D tie breaker matrix (outputs as a 1D vector)
        /// holding small values for each element. The tie breaker values are created such that for a particular row in the
        /// colInputPotSyn grid, the sum of all tie breaker values in that row is less than 0.5.
        /// The tie breaker values are all multiples of the same number and each row has a different pattern of tie breaker values.
        /// This is done by sliding the previous row's values along by 1 and wrapping at the end of the row.
        /// The tie breaker matrix is used to resolve situations where columns have the same overlap number.
        ///
        /// @param[in,out] pot_syn_tie_breaker The 2D vector of floats representing the tie breaker matrix.
        /// @param[in] size A pair of integers representing the height and width of the tie breaker matrix.
        ///
        ///------------------------------------------------------------------------------------
        void make_pot_syn_tie_breaker(std::vector<float> &pot_syn_tie_breaker, std::pair<int, int> size);

        ///------------------------------------------------------------------------------------
        ///
        /// parallel_make_pot_syn_tie_breaker - Same as the make_pot_syn_tie_breaker function but uses taskflow to parallelize the process.
        void parallel_make_pot_syn_tie_breaker(std::vector<float> &pot_syn_tie_breaker, std::pair<int, int> size);

        ///-----------------------------------------------------------------------------
        ///
        /// makeColTieBreaker Creates a tie-breaker vector with small values that are added
        /// to the overlap values between columns. The tie breaker values
        /// are randomly distributed over the cells and not biased to any side of the columns grid.
        /// The tie breaker values are multiples of the same number.
        /// For each column, the tie-breaker pattern is created by sliding the previous column's values along by 1
        /// and wrapping at the end of the row.
        /// The number of cortical columns is equal to columns_width_ * columns_height_.
        ///
        /// @param[in,out] tieBreaker A reference to a vector of floats representing the tie breaker values.
        /// @param[in] columnsWidth An integer representing the width of the columns grid.
        /// @param[in] columnsHeight An integer representing the height of the columns grid.
        ///-----------------------------------------------------------------------------
        void make_col_tie_breaker(std::vector<float> &tieBreaker, int columnsHeight, int columnsWidth);

        ///-----------------------------------------------------------------------------
        ///
        /// parallel_make_col_tie_breaker - Same as the make_col_tie_breaker function but uses taskflow to parallelize the process.
        void parallel_make_col_tie_breaker(std::vector<float> &tieBreaker, int columnsHeight, int columnsWidth);

        // Check the new input parameter sizes to make sure they are the same as the inital input sizes and
        // the 1D vectors are the right size for the simulated 2D vectors they represent.
        void check_new_input_params(const std::vector<float> &newColSynPerm,
                                    const std::pair<int, int> &colSynPerm_shape,
                                    const std::vector<int> &newInput,
                                    const std::pair<int, int> &inputGrid_shape);

        ///-----------------------------------------------------------------------------
        ///
        /// get_col_inputs This function uses a convolution function to
        /// return the inputs that each column potentially connects to.
        /// It outputs a 1D vector simulating a 2D vector where each row (in the sim 2D vector)
        /// represents the potential pool of inputs that one column in a layer can connect too via potential synapses.
        /// The input is also a 1D vector simulating a 2D vector (matrix) of the input grid.
        /// This inputGrid has the number of elements equal to the input_width_ x input_height_.
        /// @param[in] inputGrid A 1D vector simulating a 2D vector (matrix) of the input grid.
        /// @param[in] inputGrid_shape The shape of the inputGrid vector height and then width as a pair of ints.
        /// @param[out] col_inputs A 1D vector simulating a 2D vector (matrix) where each row represents the
        ///             potential pool of inputs that one column in a layer can connect too. The number of rows
        ///             is equal to the number of columns in the columns grid. The number of columns is equal to
        ///             the number of potential synapses for each column.
        /// @param[out] taskflow A tf::Taskflow object that the function gets added to. This can be run by a taskflow executor.
        void get_col_inputs(const std::vector<int> &inputGrid, const std::pair<int, int> &inputGrid_shape, std::vector<int> &col_inputs, tf::Taskflow &taskflow);

        // Member variables
        bool center_pot_synapses_; // Specifies if the potential synapses are centered over the columns
        bool wrap_input_;          // Use a wrap input function instead of padding the input to calcualte the overlap scores
        int potential_width_;      // Width of the potential synapses
        int potential_height_;     // Height of the potential synapses
        float connected_perm_;     // The permanence value that a column's potential synapses must be greater than for them to be considered "connected".
        int min_overlap_;          // Minimum overlap required for a column to be considered for inhibition
        int input_width_;          // Width of the input, equal to the number of columns in the input grid
        int input_height_;         // Height of the input, equal to the number of rows in the input grid
        int columns_width_;        // Width of the cortical columns
        int columns_height_;       // Height of the cortical columns
        int num_columns_;          // Number of columns making up this htm layer
        // Store the potential inputs to every column. Each row represents the inputs a columns potential synapses cover.
        std::vector<int> col_input_pot_syn_; // This is a 1D vector simulating a 2D vector with the size number of columns x number of potential synapses. columns_height_ x columns_width_ x potential_height_ x potential_width_
        std::vector<int> col_inputs_shape_;  // Equal to {columns_height_, columns_width_, potential_height_, potential_width_}, the shape of the col_input_pot_syn_ vector. A more convenient way to pass the shape of the col_input_pot_syn_ vector around.
        // Store the potential overlap value for every column
        std::vector<float> col_pot_overlaps_;      // This is a 1D vector with the size number of columns.
        int step_x_;                               // Step size in the x direction for the potential synapses
        int step_y_;                               // Step size in the y direction for the potential synapses
        const std::pair<int, int> neib_shape_;     // Equal to {potential_height_, potential_width_}, the shape of the neighborhood of a column. This is a pair of ints representing the height and width of the neighborhood.
        const std::pair<int, int> neib_step_;      // Equal to {step_x_, step_y_}, the step size in the x and y direction for the neighborhood. This is a pair of ints representing the step size in the x and y direction.
        std::mt19937 rng_;                         // Mersenne Twister random number generator
        std::vector<float> pot_syn_tie_breaker_;   // Potential synapse tie breaker matrix. It contains small values that help resolve any ties in potential overlap scores for columns. This is a 1D vector simulating a 4D vector with the size of the number of columns x number of potential synapses. columns_height_ x columns_width_ x potential_height_ x potential_width_
        std::vector<float> col_input_pot_syn_tie_; // Store the potential inputs to every column plus the tie breaker value. This is a 1D vector simulating a 2D vector with the size number of columns x number of potential synapses. columns_height_ x columns_width_ x potential_height_ x potential_width_
        std::vector<float> col_tie_breaker_;       // Store a tie breaker value for each column to be applied to overlap scores for each column to resolve overlap score ties.
        std::vector<int> con_syn_input_;           // Stores the connected inputs to every column. This is the same as col_input_pot_syn_ except any synapses that are not connected are set to 0. This is a 1D vector simulating a 2D vector with the size number of columns x number of potential synapses. columns_height_ x columns_width_ x potential_height_ x potential_width_
        // Store the actual overlap values for every column. This is the number of connected synapses that are active to a column
        std::vector<int> col_overlaps_;       // This is a 1D vector with the size number of columns.
        std::vector<float> col_overlaps_tie_; // Store the actual overlap values for every column plus a tie breaker value. This is the number of connected synapses that are active to a column plus a tie breaker value unique to every col to break draws in overlap values between columns.
    };

} // namespace overlap