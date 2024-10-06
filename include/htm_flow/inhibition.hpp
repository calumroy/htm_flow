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
#include <taskflow/algorithm/for_each.hpp> // Ensure this is included
#include <htm_flow/inhibition_utils.hpp>
#include <htm_flow/overlap_utils.hpp> // For print_1d_vector

namespace inhibition
{

    // Define a class to calculate the inhibition for columns in an HTM layer
    class InhibitionCalculator
    {
    public:
        // Constructor
        InhibitionCalculator(int width, int height, int potentialInhibWidth, int potentialInhibHeight,
                             int desiredLocalActivity, int minOverlap, bool centerInhib, bool wrapMode);

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
        void calculate_inhibition(const std::vector<float>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                  const std::vector<float>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape);
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
        /// calculate_inhibition_for_column   Determines if a specific column should be active or inhibited based on its overlap score and the activity of its neighboring columns.
        ///
        ///           This function performs the following steps:
        ///           1. Iterates through the sorted list of column indices.
        ///           2. For each column, checks if it is already inhibited or active.
        ///           3. If the column is not inhibited or active and its overlap score meets the minimum threshold, it proceeds to evaluate its neighbors.
        ///           4. Counts the number of active neighboring columns.
        ///           5. If the number of active neighbors meets or exceeds the desired local activity, the column is marked as inhibited.
        ///           6. If the column is not inhibited and the number of active neighbors is less than the desired local activity, the column is marked as active.
        ///           7. Updates the active state and inhibition state of the column and its neighbors accordingly.
        ///
        /// This function ensures that the inhibition process respects the local activity constraints and prevents too many columns from becoming active within a neighborhood.
        /// @param[in] sortedIndices A vector of column indices sorted by their overlap scores.
        /// @param[in] overlapGrid A vector representing the overlap values for each column in a 2D grid (flattened).
        /// @param[in,out] inhibitedCols A vector indicating whether each column is inhibited (1 for inhibited, 0 for not).
        /// @param[in,out] columnActive A vector indicating whether each column is active (1 for active, 0 for not).
        /// @param[in,out] numColsActInNeigh A vector indicating the number of active columns in each column's neighborhood.
        /// @param[in,out] activeColumnsInd A vector storing the indices of active columns.
        /// @param[in] neighbourColsLists A vector of vectors, where each inner vector contains the indices of neighboring columns for each column.
        /// @param[in] colInNeighboursLists A vector of vectors, where each inner vector contains the indices of columns that list each column as a neighbor.
        /// @param[in] desiredLocalActivity The desired number of active columns within an inhibition neighborhood.
        /// @param[in] minOverlap The minimum overlap score required for a column to be considered for activation.
        /// @param[in] activeColumnsMutex A mutex to protect access to the active columns during concurrent execution.
        /// @param[in] taskflow A Taskflow object to manage parallel tasks.
        ///
        ///-----------------------------------------------------------------------------
        void calculate_inhibition_for_column(
                const std::vector<float>& overlapGrid,
                std::vector<int>& activeColumnsInd,
                const std::vector<std::vector<int>>& neighbourColsLists,
                const std::vector<std::vector<int>>& colInNeighboursLists,
                int desiredLocalActivity,
                int minOverlap,
                std::mutex& activeColumnsMutex,
                tf::Taskflow& taskflow);

        // Member variables
        int width_;                  // Width of the grid of columns
        int height_;                 // Height of the grid of columns
        int numColumns_;             // Total number of columns = width_ * height_
        int potentialWidth_;         // Width of the inhibition neighborhood
        int potentialHeight_;        // Height of the inhibition neighborhood
        int desiredLocalActivity_;   // Desired number of active columns within an inhibition neighborhood
        int minOverlap_;             // Minimum overlap score required for a column to be considered for activation
        bool centerInhib_;           // Whether the inhibition neighborhood is centered on each column
        bool wrapMode_;             // Whether the inhibition neighborhood wraps around the grid
        std::vector<int> activeColumnsInd_;    // This is a list storing only the active columns indicies     
        // We use atomic variables to ensure thread safety when updating the active and inhibited columns.
        // Use a pointer to an array of atomic variables to allow for dynamic memory allocation at runtime.
        // We cannot use a vector of atomic variables because the vector class does not support atomic types (atomic types cannot be copied or moved).
        // We use a unique_ptr to ensure that the memory is deallocated when the object is destroyed.
        std::unique_ptr<std::atomic<int>[]> columnActive_;      // Active state of each column
        std::unique_ptr<std::atomic<int>[]> inhibitedCols_;     // Inhibited state of each column
        std::unique_ptr<std::atomic<int>[]> numColsActInNeigh_; // Number of active neighbors
        std::vector<std::mutex> columnMutexes_;                 // Mutexes for each column

        std::vector<std::vector<int>> neighbourColsLists_; // Neighbors of each column
        std::vector<std::vector<int>> colInNeighboursLists_; // Columns that list each column as a neighbor
    };

} // namespace inhibition
