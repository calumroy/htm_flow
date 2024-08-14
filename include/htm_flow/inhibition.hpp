///-----------------------------------------------------------------------------
///
/// @file inhibition.hpp
///
/// @brief A calculator class to determine active and inhibited columns in an HTM layer
///        based on overlap and potential overlap scores. It calculates the active columns
///        using a competitive inhibition mechanism to ensure a sparse distributed representation.
///
///-----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <vector>
#include <taskflow/taskflow.hpp>

namespace inhibition
{

    // Define a class to calculate the inhibition for columns in an HTM layer
    class InhibitionCalculator
    {
    public:
        // Constructor
        InhibitionCalculator(int width, int height, int potentialInhibWidth, int potentialInhibHeight,
                             int desiredLocalActivity, int minOverlap, bool centerInhib);

        ///-----------------------------------------------------------------------------
        ///
        /// calculate_inhibition   Determines the active and inhibited columns based on overlap scores.
        ///                        This function uses a competitive inhibition mechanism to activate columns.
        ///
        /// @param[in] colOverlapGrid   The overlap values for each column. A 1D vector simulating a 2D vector.
        /// @param[in] colOverlapGridShape  The shape of the colOverlapGrid vector as a pair of ints (rows, cols).
        /// @param[in] potColOverlapGrid    The potential overlap values for each column. A 1D vector simulating a 2D vector.
        /// @param[in] potColOverlapGridShape   The shape of the potColOverlapGrid vector as a pair of ints (rows, cols).
        ///
        ///-----------------------------------------------------------------------------
        void calculate_inhibition(const std::vector<int>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                  const std::vector<int>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape);

        ///-----------------------------------------------------------------------------
        ///
        /// get_active_columns - Returns the active state of each column.
        ///
        /// @return A 1D vector of ints representing the active state (1 for active, 0 for inactive) of each column.
        std::vector<int> get_active_columns();

        // ///-----------------------------------------------------------------------------
        // ///
        // /// calculateWinningCols   Determines which columns become active after applying the inhibition process.
        // ///
        // /// @param[in] colOverlapGrid   A vector representing the overlap values for each column in a 2D grid (flattened).
        // /// @param[in] colOverlapGridShape   The shape of the colOverlapGrid vector as a pair of ints (rows, cols).
        // /// @param[in] potColOverlapGrid    A vector representing the potential overlap values for each column in a 2D grid (flattened).
        // /// @param[in] potColOverlapGridShape   The shape of the potColOverlapGrid vector as a pair of ints (rows, cols).
        // ///
        // /// @return A vector representing the active state of each column (1 for active, 0 for inactive).
        // ///
        // /// Function:
        // /// 1. Add a tie-breaker to the overlaps grid based on position and previous activity.
        // /// 2. Sort all columns by their overlap values and process from highest to lowest.
        // /// 3. Determine if each column should be active based on local inhibition criteria.
        // /// 4. Apply the same process to the potential overlaps if necessary.
        // ///
        // ///-----------------------------------------------------------------------------
        // void calculateWinningCols(const std::vector<int>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
        //                           const std::vector<int>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape);



    private:
        ///-----------------------------------------------------------------------------
        ///
        /// add_tie_breaker   Adds a small value to each column's overlap score to help resolve ties.
        ///
        /// @param[in,out] overlapGrid The overlap scores for each column. A 1D vector simulating a 2D vector.
        /// @param[in] addColBias  A boolean indicating whether to add a bias for columns previously active.
        ///
        ///-----------------------------------------------------------------------------
        void add_tie_breaker(std::vector<int>& overlapGrid, bool addColBias);

        ///-----------------------------------------------------------------------------
        ///
        /// neighbours   Calculates the neighbouring columns for a given column based on its position.
        ///
        /// @param[in] pos_x The x-coordinate of the column.
        /// @param[in] pos_y The y-coordinate of the column.
        ///
        /// @return A vector of ints representing the indices of neighbouring columns.
        ///
        ///-----------------------------------------------------------------------------
        std::vector<int> neighbours(int pos_x, int pos_y) const;

        ///-----------------------------------------------------------------------------
        ///
        /// parallel_sort   Sorts a vector of indices based on corresponding values in parallel using Taskflow.
        ///
        /// @param[in,out] taskflow The Taskflow object for managing tasks.
        /// @param[in,out] indices  The indices to be sorted.
        /// @param[in] values The values based on which the sorting is to be performed.
        ///
        ///-----------------------------------------------------------------------------
        void parallel_sort(tf::Taskflow &taskflow, std::vector<int> &indices, const std::vector<int> &values);

        ///-----------------------------------------------------------------------------
        ///
        /// calculate_inhibition_for_column   Determines if a specific column should be active or inhibited.
        ///
        /// @param[in] colIndex The index of the column being considered.
        /// @param[in] overlapScore The overlap score of the column.
        ///
        ///-----------------------------------------------------------------------------
        void calculate_inhibition_for_column(int colIndex, int overlapScore);

        // Member variables
        int width_;                  // Width of the grid of columns
        int height_;                 // Height of the grid of columns
        int numColumns_;             // Total number of columns = width_ * height_
        int potentialWidth_;         // Width of the inhibition neighborhood
        int potentialHeight_;        // Height of the inhibition neighborhood
        int desiredLocalActivity_;   // Desired number of active columns within an inhibition neighborhood
        int minOverlap_;             // Minimum overlap score required for a column to be considered for activation
        bool centerInhib_;           // Whether the inhibition neighborhood is centered on each column

        std::vector<int> activeColumnsInd_;    // This is a list storing only the active columns indicies     
        std::vector<int> columnActive_;            // Active state of each column (1 for active, 0 for inactive)
        std::vector<int> inhibitedCols_;          // State indicating if a column is inhibited
        std::vector<int> numColsActInNeigh_;      // Number of active columns in each column's neighborhood
        std::vector<std::vector<int>> neighbourColsLists_; // Neighbors of each column
        std::vector<std::vector<int>> colInNeighboursLists_; // Columns that list each column as a neighbor
    };

} // namespace inhibition
