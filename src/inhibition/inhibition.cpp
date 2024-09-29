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
                                               int desiredLocalActivity, int minOverlap, bool centerInhib, bool wrapMode)
        : width_(width), height_(height), numColumns_(width * height),
          potentialWidth_(potentialInhibWidth), potentialHeight_(potentialInhibHeight),
          desiredLocalActivity_(desiredLocalActivity), minOverlap_(minOverlap),
          centerInhib_(centerInhib), wrapMode_(wrapMode), 
          activeColumnsInd_()
    {
        // Init the atomic vectors (these are not copyable or movable) so they need to be init a certain way.
        columnActive_ = std::vector<std::atomic<int>>(numColumns_);
        inhibitedCols_ = std::vector<std::atomic<int>>(numColumns_);
        numColsActInNeigh_ = std::vector<std::atomic<int>>(numColumns_);

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
    void InhibitionCalculator::calculate_inhibition(const std::vector<float>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                    const std::vector<float>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape)
    {
        // Create a single Taskflow and Executor object
        tf::Taskflow taskflow;
        tf::Executor executor;

        // TODO: move this into the constructor and only allow it to be called once.
        // THis will force the colOverlapGrid size to be statically determined and not change. 
        // Prepare a vector of indices for sorting for the colOverlapGrid
        std::vector<float> sortedIndices_colOver(colOverlapGrid.size());
        std::iota(sortedIndices_colOver.begin(), sortedIndices_colOver.end(), 0.0f);
        // Prepare a vector of indices for sorting for the potColOverlapGrid
        std::vector<float> sortedIndices_potOver(potColOverlapGrid.size());
        std::iota(sortedIndices_potOver.begin(), sortedIndices_potOver.end(), 0.0f);

        // Define the taskflow structure for the inhibition calculation
        tf::Taskflow tf1, tf2, tf3, tf4;

        // Sort columns by overlap values using parallel_sort
        inhibition_utils::parallel_sort_ind(sortedIndices_colOver, colOverlapGrid, tf1);

        // Define a mutex to protect access to activeColumnsInd_
        std::mutex activeColumnsMutex;
        
        // Process columns from highest to lowest overlap based on the sorted indices
        calculate_inhibition_for_column(sortedIndices_colOver, colOverlapGrid,
                                        inhibitedCols_, columnActive_,
                                        numColsActInNeigh_, activeColumnsInd_,
                                        neighbourColsLists_, colInNeighboursLists_,
                                        desiredLocalActivity_, minOverlap_, activeColumnsMutex, tf2);

        // Sort columns by potential overlap values using parallel_sort
        inhibition_utils::parallel_sort_ind(sortedIndices_potOver, potColOverlapGrid, tf3);

        // Process columns with potential overlap values
        calculate_inhibition_for_column(sortedIndices_potOver, potColOverlapGrid,
                                        inhibitedCols_, columnActive_,
                                        numColsActInNeigh_, activeColumnsInd_,
                                        neighbourColsLists_, colInNeighboursLists_,
                                        desiredLocalActivity_, minOverlap_, activeColumnsMutex, tf4);

        // Set the order of the tasks using tf::Task objects
        tf::Task f1_task = taskflow.composed_of(tf1).name("SortOverlap");
        tf::Task f2_task = taskflow.composed_of(tf2).name("ProcessOverlap");
        tf::Task f3_task = taskflow.composed_of(tf3).name("SortPotOverlap");
        tf::Task f4_task = taskflow.composed_of(tf4).name("ProcessPotOverlap");

        // Task dependencies to assign order of execution
        f1_task.precede(f2_task); // SortOverlap precedes ProcessOverlap
        f2_task.precede(f3_task); // ProcessOverlap precedes SortPotOverlap
        f3_task.precede(f4_task); // SortPotOverlap precedes ProcessPotOverlap

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

        // // Print the active columns
        // LOG(INFO, "Active Columns:");
        // overlap_utils::print_2d_vector(columnActive_, colOverlapGridShape);

        // Print the number of active neighbors in each column's neighborhood
        LOG(INFO, "Number of Active Neighbors:");
        overlap_utils::print_1d_vector(numColsActInNeigh_);

        // Print the list of active column indices
        LOG(INFO, "Active Columns Indices:");
        overlap_utils::print_1d_vector(activeColumnsInd_);
    }

    std::vector<int> InhibitionCalculator::get_active_columns()
    {
        // TODO:
        // Remove this this is not efficient to create a new vector each time.
        std::vector<int> activeColumns;
        for (size_t i = 0; i < columnActive_.size(); ++i)
        {
            if (columnActive_[i].load() == 1)
            {
                activeColumns.push_back(i);
            }
        }
        return activeColumns;
    }

    std::vector<int> InhibitionCalculator::neighbours(int pos_x, int pos_y) const
    {
        std::vector<int> closeColumns;
        int topPos_y = centerInhib_ ? std::floor(potentialHeight_ / 2.0) : 0;
        int bottomPos_y = centerInhib_ ? std::ceil(potentialHeight_ / 2.0) - 1 : potentialHeight_ - 1;
        int leftPos_x = centerInhib_ ? std::floor(potentialWidth_ / 2.0) : 0;
        int rightPos_x = centerInhib_ ? std::ceil(potentialWidth_ / 2.0) - 1 : potentialWidth_ - 1;

        for (int dy = -topPos_y; dy <= bottomPos_y; ++dy)
        {
            int i = pos_y + dy;
            int wrapped_i;

            if (wrapMode_)
            {
                // Wrap around the vertical boundaries
                wrapped_i = (i + height_) % height_;
            }
            else if (i >= 0 && i < height_)
            {
                wrapped_i = i;
            }
            else
            {
                continue; // Skip out-of-bounds indices when not wrapping
            }

            for (int dx = -leftPos_x; dx <= rightPos_x; ++dx)
            {
                int j = pos_x + dx;
                int wrapped_j;

                if (wrapMode_)
                {
                    // Wrap around the horizontal boundaries
                    wrapped_j = (j + width_) % width_;
                }
                else if (j >= 0 && j < width_)
                {
                    wrapped_j = j;
                }
                else
                {
                    continue; // Skip out-of-bounds indices when not wrapping
                }

                int neighborIndex = wrapped_i * width_ + wrapped_j;
                closeColumns.push_back(neighborIndex);
            }
        }

        return closeColumns;
    }

    void InhibitionCalculator::calculate_inhibition_for_column(
        const std::vector<float>& sortedIndices,
        const std::vector<float>& overlapGrid,
        std::vector<std::atomic<int>>& inhibitedCols,
        std::vector<std::atomic<int>>& columnActive,
        std::vector<std::atomic<int>>& numColsActInNeigh,
        std::vector<int>& activeColumnsInd,
        const std::vector<std::vector<int>>& neighbourColsLists,
        const std::vector<std::vector<int>>& colInNeighboursLists,
        int desiredLocalActivity,
        int minOverlap,
        std::mutex& activeColumnsMutex,
        tf::Taskflow& taskflow)
    {
        // Create a parallel_for task in the taskflow, 0ul is the start index, sortedIndices.size() is the end index, 1ul is the step size
        taskflow.for_each_index(0ul, sortedIndices.size(), 1ul, [&, this](size_t idx) {
            int i = sortedIndices[idx];
            if (inhibitedCols[i].load() == 0 && columnActive[i].load() == 0 && overlapGrid[i] >= minOverlap)
            {
                const std::vector<int>& neighbourCols = neighbourColsLists[i];
                int numActiveNeighbours = 0;
                bool inhibit = false;

                // Check neighbours for active columns
                for (int neighborIndex : neighbourCols)
                {
                    if (neighborIndex >= 0 && columnActive[neighborIndex].load() == 1)
                    {
                        numActiveNeighbours++;
                        if (numColsActInNeigh[neighborIndex].load() >= desiredLocalActivity)
                        {
                            inhibit = true;
                            inhibitedCols[i].store(1);
                            break; // No need to check further
                        }
                    }
                }

                if (!inhibit)
                {
                    // Check if the column is in any neighbour lists of active columns
                    const std::vector<int>& inNeighbours = colInNeighboursLists[i];
                    for (int activeNeighbor : inNeighbours)
                    {
                        if (columnActive[activeNeighbor].load() == 1)
                        {
                            if (numColsActInNeigh[activeNeighbor].load() >= desiredLocalActivity)
                            {
                                inhibit = true;
                                inhibitedCols[i].store(1);
                                break; // No need to check further
                            }
                        }
                    }
                }

                numColsActInNeigh[i].store(numActiveNeighbours);

                // Activate column if not inhibited and the number of active neighbors is less than desired local activity
                if (!inhibit && numColsActInNeigh[i].load() < desiredLocalActivity)
                {
                    // Add i to activeColumnsInd
                    {
                        std::lock_guard<std::mutex> lock(activeColumnsMutex);
                        activeColumnsInd.push_back(i);
                    }

                    columnActive[i].store(1);

                    // Increment numColsActInNeigh for inNeighbours
                    const std::vector<int>& inNeighbours = colInNeighboursLists[i];
                    for (int c : inNeighbours)
                    {
                        if (c >= 0)
                        {
                            numColsActInNeigh[c].fetch_add(1);
                        }
                    }
                }
                else
                {
                    inhibitedCols[i].store(1);
                }
            }
        }).name("ProcessOverlapOrPotential");
    }


} // namespace inhibition
