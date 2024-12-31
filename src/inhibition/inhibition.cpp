// inhibition.cpp

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
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
          activeColumnsInd_(),
          columnMutexes_(numColumns_)
    {
        // Init the atomic vectors (these are not copyable or movable) so they need to be init a certain way.
        // Initialize atomic variables. Allocate the arrays before using them
        columnActive_ = std::make_unique<std::atomic<int>[]>(numColumns_);
        inhibitedCols_ = std::make_unique<std::atomic<int>[]>(numColumns_);
        // Initialize atomic variables
        for (int i = 0; i < numColumns_; ++i)
        {
            columnActive_[i].store(0);
            inhibitedCols_[i].store(0);
        }

        // Initialize the neighbours list for each column
        neighbourColsLists_ = std::vector<std::vector<int>>(numColumns_);
        colInNeighboursLists_ = std::vector<std::vector<int>>(numColumns_);

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
        for (int i = 0; i < numColumns_; ++i)
        {
            for (int neighbor : neighbourColsLists_[i])
            {
                if (neighbor >= 0)
                {
                    colInNeighboursLists_[neighbor].push_back(i);
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
        tf::Taskflow taskflow;
        tf::Executor executor;

        // TODO: find a faster way to do this.
        // Reset atomic variables and active columns list
        for (int i = 0; i < numColumns_; ++i)
        {
            columnActive_[i].store(0);
            inhibitedCols_[i].store(0);
        }
        activeColumnsInd_.clear();

        // Define the taskflow structure for the inhibition calculation
        tf::Taskflow tf1, tf2;
        std::mutex activeColumnsMutex;

        // Process columns with overlap grid
        calculate_inhibition_for_column(colOverlapGrid,
                                        activeColumnsInd_,
                                        neighbourColsLists_, colInNeighboursLists_,
                                        desiredLocalActivity_, minOverlap_, activeColumnsMutex, tf1);

        // // Process columns with potential overlap grid
        // calculate_inhibition_for_column(potColOverlapGrid,
        //                                 activeColumnsInd_,
        //                                 neighbourColsLists_, colInNeighboursLists_,
        //                                 desiredLocalActivity_, minOverlap_, activeColumnsMutex, tf2);

        // Set the order of the tasks using tf::Task objects
        tf::Task f1_task = taskflow.composed_of(tf1).name("ProcessOverlap");
        // tf::Task f2_task = taskflow.composed_of(tf2).name("ProcessPotentialOverlap");
        // // Ensure t1 precedes t2
        // f1_task.precede(f2_task);    

        // Run the constructed taskflow graph
        tf::Future<void> fu = executor.run(taskflow);
        fu.wait(); // Block until the execution completes
    
        // // Print the results using LOG and overlap_utils functions
        // LOG(INFO, "Final Results:");

        // // Print the overlap grid after tie-breakers
        // LOG(INFO, "Overlap Grid (with Tie-Breakers):");
        // overlap_utils::print_2d_vector(colOverlapGrid, colOverlapGridShape);

        // // Print the inhibited columns
        // LOG(INFO, "Inhibited Columns:");
        // overlap_utils::print_1d_atomic_array(inhibitedCols_.get(), numColumns_);

        // // Print the active columns
        // LOG(INFO, "Active Columns:");
        // overlap_utils::print_2d_atomic_array(columnActive_.get(), colOverlapGridShape);

        // // Print the list of active column indices
        // LOG(INFO, "Active Columns Indices:");
        // overlap_utils::print_1d_vector(activeColumnsInd_);
    }

    std::vector<int> InhibitionCalculator::get_active_columns()
    {
        // TODO:
        // Remove this this is not efficient to create a new vector each time.
        std::vector<int> activeColumns(numColumns_, 0);
        for (size_t i = 0; i < numColumns_; ++i)
        {
            activeColumns[i] = columnActive_[i].load();
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
        const std::vector<float>& overlapGrid,
        std::vector<int>& activeColumnsInd,
        const std::vector<std::vector<int>>& neighbourColsLists,
        const std::vector<std::vector<int>>& colInNeighboursLists,
        int desiredLocalActivity,
        int minOverlap,
        std::mutex& activeColumnsMutex,
        tf::Taskflow& taskflow)
    {
        // Create a parallel_for task in the taskflow, 0ul is the start index, numColumns_ is the end index, 1ul is the step size.
        // Capture all required variables by reference using [&] also capture the var desiredLocalActivity and minOverlap by value as these are used in the lambda function.
        // We are making sure that if the lambda funciotn is called later on the variables are still valid.
        taskflow.for_each_index(0ul, static_cast<size_t>(numColumns_), 1ul, [&, this, desiredLocalActivity, minOverlap](size_t idx) {
            int i = idx;

            if (overlapGrid[i] < minOverlap)
            {
                // Do not consider columns with overlap below minOverlap
                return;
            }

            // Build the list of columns to lock: neighbor columns, columns that include i in their neighbor lists, and i itself
            std::vector<int> colsToLock = neighbourColsLists[i];
            colsToLock.insert(colsToLock.end(), colInNeighboursLists[i].begin(), colInNeighboursLists[i].end());
            colsToLock.push_back(i);

            // Remove duplicates and sort to prevent deadlocks
            std::sort(colsToLock.begin(), colsToLock.end());
            // Remove duplicate column indices and keep only unique ones
            colsToLock.erase(std::unique(colsToLock.begin(), colsToLock.end()), colsToLock.end());

            // Lock the mutexes in ascending order
            for (int colIndex : colsToLock)
            {
                columnMutexes_[colIndex].lock();
            }

            //LOG(INFO, "Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is being processed");

            // Begin critical section
            // Build list of active columns in the neighborhood (both neighbors and columns that include i as neighbor)
            std::vector<int> activeNeighbors;
            std::vector<float> activeNeighborsOverlap;

            // Collect active neighbors from neighbourColsLists[i]
            const std::vector<int>& neighborCols = neighbourColsLists[i];
            for (int neighborIndex : neighborCols)
            {
                if (columnActive_[neighborIndex].load() == 1)
                {
                    activeNeighbors.push_back(neighborIndex);
                    activeNeighborsOverlap.push_back(overlapGrid[neighborIndex]);
                }
            }

            // Collect active neighbors from colInNeighboursLists[i]
            const std::vector<int>& inNeighborCols = colInNeighboursLists[i];
            for (int neighborIndex : inNeighborCols)
            {
                if (columnActive_[neighborIndex].load() == 1)
                {
                    // Avoid duplicates
                    if (std::find(activeNeighbors.begin(), activeNeighbors.end(), neighborIndex) == activeNeighbors.end())
                    {
                        activeNeighbors.push_back(neighborIndex);
                        activeNeighborsOverlap.push_back(overlapGrid[neighborIndex]);
                    }
                }
            }

            // Check if the current column is already active
            if (columnActive_[i].load() == 1)
            {
                // Current column is already active
                for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
                {
                    columnMutexes_[*it].unlock();
                }
                //LOG(INFO, "Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is already active");
                return;
            }

            // Now see if we can become active
            if (static_cast<int>(activeNeighbors.size()) < desiredLocalActivity)
            {
                // EXTRA CHECK: Are there enough *higher-overlap* neighbors
                // still unprocessed that might displace us?

                int needed = desiredLocalActivity - static_cast<int>(activeNeighbors.size());
                if (needed > 0)
                {
                    // Gather neighbors that are neither active nor inhibited
                    // i.e., they "might still become active."
                    std::vector<int> candidateNeighbors;
                    {
                        // Combine both neighbor sets
                        std::vector<int> allNeighbors = neighbourColsLists[i];
                        allNeighbors.insert(allNeighbors.end(),
                                            colInNeighboursLists[i].begin(),
                                            colInNeighboursLists[i].end());

                        // Remove duplicates
                        std::sort(allNeighbors.begin(), allNeighbors.end());
                        allNeighbors.erase(std::unique(allNeighbors.begin(), allNeighbors.end()), allNeighbors.end());

                        // Filter: not inhibited, not active
                        for (int nb : allNeighbors)
                        {
                            if (columnActive_[nb].load() == 0 &&
                                inhibitedCols_[nb].load() == 0)
                            {
                                candidateNeighbors.push_back(nb);
                            }
                        }
                    }

                    // Sort candidates by overlap descending
                    std::sort(candidateNeighbors.begin(), candidateNeighbors.end(),
                              [&](int a, int b) {
                                  return overlapGrid[a] > overlapGrid[b];
                              });

                    // If there are enough neighbors with bigger overlap than me,
                    // they could fill up the local activity. So we skip activation.
                    if (static_cast<int>(candidateNeighbors.size()) >= needed)
                    {
                        float threshold = overlapGrid[candidateNeighbors[needed - 1]];
                        if (overlapGrid[i] < threshold)
                        {
                            // This means I'm strictly below
                            // the "needed-th best" candidate => I'd get displaced
                            inhibitedCols_[i].store(1);

                            // Unlock and return
                            for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
                            {
                                columnMutexes_[*it].unlock();
                            }
                            return;
                        }
                    }
                }

                // If we get here, we are "safe" to become active
                columnActive_[i].store(1);
                {
                    std::lock_guard<std::mutex> lock(activeColumnsMutex);
                    activeColumnsInd.push_back(i);
                }

                // LOG(INFO, "Column index: " + std::to_string(i) +
                //           " with overlap score: " + std::to_string(overlapGrid[i]) +
                //           " is activated as it has fewer active neighbors (" +
                //           std::to_string(activeNeighbors.size()) + ") than desiredLocalActivity (" +
                //           std::to_string(desiredLocalActivity) + ").");
            }
            else
            {
                // Find the active column with the lowest overlap score
                auto minIt = std::min_element(activeNeighborsOverlap.begin(), activeNeighborsOverlap.end());
                // Calculate the index of the minimum overlap score in the activeNeighborsOverlap vector
                int minIndex = std::distance(activeNeighborsOverlap.begin(), minIt);
                float minOverlapScore = *minIt;
                int minOverlapColIndex = activeNeighbors[minIndex];

                if (overlapGrid[i] > minOverlapScore)
                {
                    // Deactivate the column with the lowest overlap score
                    columnActive_[minOverlapColIndex].store(0);
                    inhibitedCols_[minOverlapColIndex].store(1);

                    // Remove it from activeColumnsInd
                    {
                        std::lock_guard<std::mutex> lock(activeColumnsMutex);
                        auto it = std::find(activeColumnsInd.begin(), activeColumnsInd.end(), minOverlapColIndex);
                        if (it != activeColumnsInd.end())
                        {
                            activeColumnsInd.erase(it);
                        }
                    }
                    //LOG(INFO, "    Deactivated column index: " + std::to_string(minOverlapColIndex) + " with overlap score: " + std::to_string(overlapGrid[minOverlapColIndex]));

                    // Activate the current column
                    columnActive_[i].store(1);

                    {
                        std::lock_guard<std::mutex> lock(activeColumnsMutex);
                        activeColumnsInd.push_back(i);
                        //LOG(INFO, "     Activated column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]));
                    }
                }
                else
                {
                    // Do not activate the current column
                    // Mark it as inhibited
                    inhibitedCols_[i].store(1);
                    //LOG(INFO, "    Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is inhibited as it has the lowest overlap score");
                    // Print out the active neighbors
                    //LOG(INFO, "        Active neighbors: ");
                    //overlap_utils::print_1d_vector(activeNeighbors);
                    // Print out the activeNeighborsOverlap vector
                    //LOG(INFO, "        Active neighbors overlap: ");
                    //overlap_utils::print_1d_vector(activeNeighborsOverlap);
                    // Print minOverlapColIndex
                    //LOG(INFO, "        Min overlap column index: " + std::to_string(minOverlapColIndex));
                    // Print minIt
                    //LOG(INFO, "        Min overlap score: " + std::to_string(minOverlapScore));
                    // Print minIndex
                    //LOG(INFO, "        Min index: " + std::to_string(minIndex));
                }
            }

            // End critical section
            // Unlock the mutexes in reverse order
            for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
            {
                columnMutexes_[*it].unlock();
            }

        }).name("ProcessColumns");
    }

} // namespace inhibition
