// inhibition.cpp
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <taskflow/taskflow.hpp>
#include "inhibition.hpp"
#include "utilities/logger.hpp"

namespace inhibition
{
    InhibitionCalculator::InhibitionCalculator(int width, int height, int potentialInhibWidth, int potentialInhibHeight,
                                               int desiredLocalActivity, int minOverlap, bool centerInhib)
        : width_(width), height_(height), numColumns_(width*height), potentialWidth_(potentialInhibWidth), potentialHeight_(potentialInhibHeight),
          desiredLocalActivity_(desiredLocalActivity), minOverlap_(minOverlap), centerInhib_(centerInhib),
          activeColumnsInd_(), columnActive_(width * height, 0), inhibitedCols_(width * height, 0), numColsActInNeigh_(width * height, 0)
    {
        // Initialize the neighbours list for each column
        neighbourColsLists_ = std::vector<std::vector<int>>(width * height);
        colInNeighboursLists_ = std::vector<std::vector<int>>(width * height);
        // Calculate neighbours
        for (int y = 0; y < height_; ++y)
        {
            for (int x = 0; x < width_; ++x)
            {
                int colIndex = x + width_ * y;
                neighbourColsLists_[colIndex] = neighbours(x, y);
            }
        }

        // Calculate columns in neighbours lists
        for (int y = 0; y < height_; ++y)
        {
            for (int x = 0; x < width_; ++x)
            {
                int colIndex = x + width_ * y;
                for (int neighbor : neighbourColsLists_[colIndex])
                {
                    if (neighbor >= 0)
                    {
                        colInNeighboursLists_[neighbor].push_back(colIndex);
                    }
                }
            }
        }
    }


    ///-----------------------------------------------------------------------------
    ///
    /// calculate_inhibition   Determines the active and inhibited columns in an HTM layer based on overlap and potential overlap scores.
    ///
    /// @param[in] colOverlapGrid A vector representing the overlap values for each column in a 2D grid (flattened).
    /// @param[in] colOverlapGridShape A pair representing the shape of the colOverlapGrid (rows, cols).
    /// @param[in] potColOverlapGrid A vector representing the potential overlap values for each column in a 2D grid (flattened).
    /// @param[in] potColOverlapGridShape A pair representing the shape of the potColOverlapGrid (rows, cols).
    ///
    /// This function performs the following steps:
    /// 1. Add Tie-Breakers:
    ///    - Apply a small tie-breaker value to each column's overlap score to resolve ties.
    /// 2. Sort Columns by Overlap Scores:
    ///    - Sort columns based on overlap scores using Taskflow for parallel processing.
    /// 3. Determine Active Columns from Overlap Scores:
    ///    - Iterate through sorted columns to mark them as active or inhibited based on overlap and neighbor constraints.
    /// 4. Repeat Process with Potential Overlap Scores (if necessary):
    ///    - Process potential overlap scores similarly to further refine column activation.
    /// 5. Finalize Active Columns:
    ///    - Store the final activation state of each column.
    ///-----------------------------------------------------------------------------
    void InhibitionCalculator::calculate_inhibition(const std::vector<int>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                    const std::vector<int>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape)
    {
        // Create a single Taskflow and Executor object
        tf::Taskflow taskflow;
        tf::Executor executor;

        // Prepare a vector of indices for sorting for the colOverlapGrid
        std::vector<int> sortedIndices_colOver(colOverlapGrid.size());
        std::iota(sortedIndices_colOver.begin(), sortedIndices_colOver.end(), 0);
        // Prepare a vector of indices for sorting for the potColOverlapGrid
        std::vector<int> sortedIndices_potOver(potColOverlapGrid.size());
        std::iota(sortedIndices_potOver.begin(), sortedIndices_potOver.end(), 0);

        // Define the taskflow structure for the inhibition calculation
        tf::Taskflow tf1, tf2, tf3, tf4, tf5, tf6;

        // Task to add a small tie breaker to the overlap grid
        add_tie_breaker(const_cast<std::vector<int>&>(colOverlapGrid), false, tf1);

        // Sort columns by overlap values using parallel_sort
        inhibition_utils::parallel_sort_ind(sortedIndices_colOver, colOverlapGrid, tf2);

        // Process columns from highest to lowest overlap based on the sorted indices
        tf3.emplace([&]() {
            calculate_inhibition_for_column(sortedIndices_colOver, colOverlapGrid, 
                                            inhibitedCols_, columnActive_, 
                                            numColsActInNeigh_, activeColumnsInd_, 
                                            neighbourColsLists_, colInNeighboursLists_, 
                                            desiredLocalActivity_, minOverlap_);
        }).name("ProcessOverlap");

        // Add a small tie breaker to the potential overlap grid
        add_tie_breaker(const_cast<std::vector<int>&>(potColOverlapGrid), false, tf4);

        // Sort columns by potential overlap values using parallel_sort
        inhibition_utils::parallel_sort_ind(sortedIndices_potOver, potColOverlapGrid, tf5);

        // Process columns with potential overlap values
        tf6.emplace([&]() {
            calculate_inhibition_for_column(sortedIndices_potOver, potColOverlapGrid, 
                                            inhibitedCols_, columnActive_, 
                                            numColsActInNeigh_, activeColumnsInd_, 
                                            neighbourColsLists_, colInNeighboursLists_, 
                                            desiredLocalActivity_, minOverlap_);
        }).name("ProcessPotOverlap");

        // Set the order of the tasks using tf::Task objects
        tf::Task f1_task = taskflow.composed_of(tf1).name("AddTieBreaker");
        tf::Task f2_task = taskflow.composed_of(tf2).name("SortOverlap");
        tf::Task f3_task = taskflow.composed_of(tf3).name("ProcessOverlap");
        tf::Task f4_task = taskflow.composed_of(tf4).name("AddTieBreakerPot");
        tf::Task f5_task = taskflow.composed_of(tf5).name("SortPotOverlap");
        tf::Task f6_task = taskflow.composed_of(tf6).name("ProcessPotOverlap");

        // Set the task dependencies
        f1_task.precede(f2_task);
        f2_task.precede(f3_task);
        f3_task.precede(f4_task);
        f4_task.precede(f5_task);
        f5_task.precede(f6_task);

        // Dump the graph to a DOT file through std::cout (optional for debugging)
        taskflow.dump(std::cout);

        // Run the constructed taskflow graph
        tf::Future<void> fu = executor.run(taskflow);
        fu.wait(); // Block until the execution completes
    
        // Print the results using LOG and overlap_utils functions
        LOG(INFO, "Final Results:");

        // Print the sorted indices
        LOG(INFO, "Sorted Indices:");
        overlap_utils::print_1d_vector(sortedIndices_colOver);

        // Print the overlap grid after tie-breakers
        LOG(INFO, "Overlap Grid (with Tie-Breakers):");
        overlap_utils::print_2d_vector(colOverlapGrid, colOverlapGridShape);

        // Print the inhibited columns
        LOG(INFO, "Inhibited Columns:");
        overlap_utils::print_1d_vector(inhibitedCols_);

        // Print the active columns
        LOG(INFO, "Active Columns:");
        overlap_utils::print_2d_vector(columnActive_, colOverlapGridShape);

        // Print the number of active neighbors in each column's neighborhood
        LOG(INFO, "Number of Active Neighbors:");
        overlap_utils::print_1d_vector(numColsActInNeigh_);

        // Print the list of active column indices
        LOG(INFO, "Active Columns Indices:");
        overlap_utils::print_1d_vector(activeColumnsInd_);
    }


    std::vector<int> InhibitionCalculator::get_active_columns()
    {
        return columnActive_;
    }

    void InhibitionCalculator::add_tie_breaker(std::vector<int>& overlapGrid, bool addColBias, tf::Taskflow &taskflow)
    {
        // Create a task in the taskflow for adding the tie breaker
        taskflow.emplace([this, &overlapGrid, addColBias]() {
            float normValue = 1.0f / float(2 * numColumns_ + 2);

            std::vector<float> tieBreaker(width_ * height_, 0.0f);
            for (int y = 0; y < height_; ++y)
            {
                for (int x = 0; x < width_; ++x)
                {
                    int index = y * width_ + x;
                    if ((y % 2) == 1)
                    {
                        // For odd rows, bias to the bottom left
                        tieBreaker[index] = ((y + 1) * width_ + (width_ - x - 1)) * normValue;
                    }
                    else
                    {
                        // For even rows, bias to the bottom right
                        tieBreaker[index] = (1 + x + y * width_) * normValue;
                    }
                }
            }

            if (addColBias)
            {
                // If previous columns were active, we can give them a small bias
                // to maintain stability across time steps (this part is optional)
                for (int i = 0; i < overlapGrid.size(); ++i)
                {
                    overlapGrid[i] += tieBreaker[i];
                }
            }
        }).name("AddTieBreaker");
    }


    std::vector<int> InhibitionCalculator::neighbours(int pos_x, int pos_y) const
    {
        std::vector<int> closeColumns;
        int topPos_y = centerInhib_ ? std::floor(potentialHeight_ / 2.0) : 0;
        int bottomPos_y = centerInhib_ ? std::ceil(potentialHeight_ / 2.0) - 1 : potentialHeight_ - 1;
        int leftPos_x = centerInhib_ ? std::floor(potentialWidth_ / 2.0) : 0;
        int rightPos_x = centerInhib_ ? std::ceil(potentialWidth_ / 2.0) - 1 : potentialWidth_ - 1;

        for (int i = pos_y - topPos_y; i <= pos_y + bottomPos_y; ++i)
        {
            if (i >= 0 && i < height_)
            {
                for (int j = pos_x - leftPos_x; j <= pos_x + rightPos_x; ++j)
                {
                    if (j >= 0 && j < width_)
                    {
                        closeColumns.push_back(i * width_ + j);
                    }
                }
            }
        }

        return closeColumns;
    }

    void InhibitionCalculator::calculate_inhibition_for_column(const std::vector<int>& sortedIndices,
                                                            const std::vector<int>& overlapGrid,
                                                            std::vector<int>& inhibitedCols, 
                                                            std::vector<int>& columnActive, 
                                                            std::vector<int>& numColsActInNeigh, 
                                                            std::vector<int>& activeColumnsInd, 
                                                            const std::vector<std::vector<int>>& neighbourColsLists, 
                                                            const std::vector<std::vector<int>>& colInNeighboursLists, 
                                                            int desiredLocalActivity,
                                                            int minOverlap)
    {
        for (int i : sortedIndices) {
            if (inhibitedCols[i] == 0 && columnActive[i] == 0 && overlapGrid[i] >= minOverlap) {
                std::vector<int> neighbourCols = neighbourColsLists[i];
                int numActiveNeighbours = 0;

                // Check neighbours for active columns
                for (int neighborIndex : neighbourCols)
                {
                    if (neighborIndex >= 0 && columnActive[neighborIndex] == 1)
                    {
                        numActiveNeighbours++;
                        if (numColsActInNeigh[neighborIndex] >= desiredLocalActivity)
                        {
                            inhibitedCols[i] = 1;
                        }
                    }
                }

                // Check if the column is in any neighbour lists of active columns
                for (int activeNeighbor : colInNeighboursLists[i])
                {
                    if (columnActive[activeNeighbor] == 1)
                    {
                        if (numColsActInNeigh[activeNeighbor] >= desiredLocalActivity)
                        {
                            inhibitedCols[i] = 1;
                        }
                    }
                }

                numColsActInNeigh[i] = numActiveNeighbours;

                // Activate column if not inhibited and the number of active neighbors is less than desired local activity
                if (inhibitedCols[i] != 1 && numColsActInNeigh[i] < desiredLocalActivity)
                {
                    activeColumnsInd.push_back(i);
                    columnActive[i] = 1;
                    for (int c : colInNeighboursLists[i])
                    {
                        if (c >= 0)
                        {
                            numColsActInNeigh[c]++;
                        }
                    }
                }
                else
                {
                    inhibitedCols[i] = 1;
                }
            }
        }
    }


} // namespace inhibition
