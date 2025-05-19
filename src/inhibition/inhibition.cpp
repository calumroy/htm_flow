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
                                               int desiredLocalActivity, int minOverlap, bool centerInhib, bool wrapMode, 
                                               bool strictLocalActivity, bool debug)
        : width_(width), height_(height), numColumns_(width * height),
          potentialWidth_(potentialInhibWidth), potentialHeight_(potentialInhibHeight),
          desiredLocalActivity_(desiredLocalActivity), minOverlap_(minOverlap),
          centerInhib_(centerInhib), wrapMode_(wrapMode), strictLocalActivity_(strictLocalActivity),
          activeColumnsInd_(),
          columnMutexes_(numColumns_),
          debug_(debug)
    {
        // If strictLocalActivity is enabled, ensure symmetrical inhibition areas
        if (strictLocalActivity_) {
            // Force odd values for width and height to ensure symmetry
            if (potentialWidth_ % 2 == 0) {
                potentialWidth_ += 1;
                LOG(INFO, "strictLocalActivity enabled: Adjusted potentialWidth to odd value so inhibition is symmetrical: " + std::to_string(potentialWidth_));
            }
            if (potentialHeight_ % 2 == 0) {
                potentialHeight_ += 1;
                LOG(INFO, "strictLocalActivity enabled: Adjusted potentialHeight to odd value so inhibition is symmetrical: " + std::to_string(potentialHeight_));
            }
            
            // Force center inhibition
            if (!centerInhib_) {
                centerInhib_ = true;
                LOG(INFO, "strictLocalActivity enabled: Forced centerInhib to true for symmetrical inhibition");
            }
            
            LOG(INFO, "Symmetrical inhibition areas enforced for strictLocalActivity");
        }

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

        // TODO: Reenable this.
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
    
        // Print the results using LOG and overlap_utils functions
        if (debug_) {
            LOG(DEBUG, "Final Results:");

            // Print the overlap grid after tie-breakers
            LOG(DEBUG, "Overlap Grid (with Tie-Breakers):");
            overlap_utils::print_2d_vector(colOverlapGrid, colOverlapGridShape);

            // Print the inhibited columns
            LOG(DEBUG, "Inhibited Columns:");
            overlap_utils::print_1d_atomic_array(inhibitedCols_.get(), numColumns_);

            // Print the active columns
            LOG(DEBUG, "Active Columns:");
            overlap_utils::print_2d_atomic_array(columnActive_.get(), colOverlapGridShape);

            // Print the list of active column indices
            LOG(DEBUG, "Active Columns Indices:");
            overlap_utils::print_1d_vector(activeColumnsInd_);
        }
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

    ///-----------------------------------------------------------------------------
    ///
    /// This function calculates inhibition for each column in parallel:
    ///
    /// 1. For each column with sufficient overlap (>= minOverlap):
    ///   a. Identify all relevant columns to lock (current column, neighbors, and 
    ///      columns that list this one as a neighbor)
    ///   b. Lock these columns in ascending order to prevent deadlocks
    ///
    /// 2. When strictLocalActivity is enabled:
    ///   a. Collect all columns in the neighborhood
    ///   b. Determine an activation threshold based on the top desiredLocalActivity overlap scores
    ///   c. Mark all columns below threshold as inhibited and deactivate them if they were active
    ///   d. Mark all columns above threshold as not inhibited to ensure consistency
    ///   e. This determines precisely which columns should be active/inhibited regardless of processing order
    ///
    /// 3. Ensure column state consistency:
    ///   a. Check for and resolve any inconsistent states (both active and inhibited)
    ///   b. If the column is already active, maintain its active state
    ///   c. If the column is already inhibited, skip further processing
    ///   d. If strictLocalActivity is enabled and column wasn't inhibited in step 2, activate it directly
    ///
    /// 4. Otherwise (strictLocalActivity not enabled):
    ///   a. Examine the neighborhood activity and count active neighbors
    ///   b. If active neighbors < desiredLocalActivity:
    ///     i. Mark column as active
    ///   c. If active neighbors >= desiredLocalActivity:
    ///     i. Find the active neighbor with the lowest overlap score
    ///     ii. If current column has higher overlap, deactivate that neighbor and 
    ///         activate current column instead
    ///     iii. Otherwise, inhibit the current column
    ///
    /// 5. Release all locks in reverse order
    ///
    ///-----------------------------------------------------------------------------
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

            // Step 1a: Check if column has sufficient overlap
            if (overlapGrid[i] < minOverlap)
            {
                // Do not consider columns with overlap below minOverlap
                return;
            }

            // Step 1b: Build the list of columns to lock and acquire locks
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

            if (debug_) {
                LOG(DEBUG, "Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is being processed");
            }

            // Begin critical section
            // Step 2: Determine activation threshold when strictLocalActivity is enabled
            if (strictLocalActivity_) {
                // Step 2a: Combine both neighbor sets to get all columns in the neighborhood
                std::vector<int> allNeighbors = neighbourColsLists[i];
                allNeighbors.insert(allNeighbors.end(), colInNeighboursLists[i].begin(), colInNeighboursLists[i].end());
                
                // Add the current column to consider it in the neighborhood
                allNeighbors.push_back(i);
                
                // Remove duplicates 
                std::sort(allNeighbors.begin(), allNeighbors.end());
                allNeighbors.erase(std::unique(allNeighbors.begin(), allNeighbors.end()), allNeighbors.end());
                
                // Filter neighbors with overlap >= minOverlap
                std::vector<int> eligibleNeighbors;
                std::vector<float> eligibleOverlaps;
                for (int nb : allNeighbors) {
                    if (overlapGrid[nb] >= minOverlap) {
                        eligibleNeighbors.push_back(nb);
                        eligibleOverlaps.push_back(overlapGrid[nb]);
                    }
                }
                
                // Step 2b: Sort neighbors by overlap score in descending order
                std::vector<int> indices(eligibleNeighbors.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                    return eligibleOverlaps[a] > eligibleOverlaps[b];
                });
                
                // Step 2c: Determine the activation threshold and mark columns below it as inhibited
                // If we have fewer eligible neighbors than desiredLocalActivity, all eligible columns can be active
                if (eligibleNeighbors.size() > desiredLocalActivity) {
                    float threshold = eligibleOverlaps[indices[desiredLocalActivity - 1]];
                    
                    // Step 2c: Mark all columns below threshold as inhibited and deactivate if needed
                    for (int nb : allNeighbors) {
                        if (overlapGrid[nb] < threshold && overlapGrid[nb] >= minOverlap) {
                            // If the column is also active, deactivate it first to maintain consistency
                            if (columnActive_[nb].load() == 1) {
                                columnActive_[nb].store(0);
                                
                                // Remove it from activeColumnsInd if present
                                {
                                    std::lock_guard<std::mutex> lock(activeColumnsMutex);
                                    auto it = std::find(activeColumnsInd.begin(), activeColumnsInd.end(), nb);
                                    if (it != activeColumnsInd.end()) {
                                        activeColumnsInd.erase(it);
                                        if (debug_) {
                                            LOG(DEBUG, "    Deactivated previously active column index: " + std::to_string(nb) + 
                                                  " due to being below threshold");
                                        }
                                    }
                                }
                            }
                            
                            inhibitedCols_[nb].store(1);
                            if (debug_) {
                                LOG(DEBUG, "    Column index: " + std::to_string(nb) + " with overlap score: " + 
                                      std::to_string(overlapGrid[nb]) + " inhibited by threshold " + 
                                      std::to_string(threshold) + " in neighborhood of column " + std::to_string(i));
                            }
                        }
                    }

                    // Step 2d: For columns above threshold, ensure they're not inhibited
                    // This is needed for deterministic behavior in parallel execution
                    for (int nb : allNeighbors) {
                        if (overlapGrid[nb] >= threshold && overlapGrid[nb] >= minOverlap) {
                            inhibitedCols_[nb].store(0);
                        }
                    }
                }
            }

            // Step 3a: Check for inconsistent state (both active and inhibited)
            // This shouldn't happen with our improved logic, but we'll check for safety
            if (columnActive_[i].load() == 1 && inhibitedCols_[i].load() == 1) {
                // Resolve by deactivating the column
                columnActive_[i].store(0);
                
                // Remove it from activeColumnsInd
                {
                    std::lock_guard<std::mutex> lock(activeColumnsMutex);
                    auto it = std::find(activeColumnsInd.begin(), activeColumnsInd.end(), i);
                    if (it != activeColumnsInd.end()) {
                        activeColumnsInd.erase(it);
                    }
                }
                
                if (debug_) {
                    LOG(DEBUG, "Column index: " + std::to_string(i) + 
                          " was both active and inhibited - resolved by deactivating");
                }
                
                // Unlock and return
                for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it) {
                    columnMutexes_[*it].unlock();
                }
                return;
            }

            // Step 3b: Check if the current column is already active
            if (columnActive_[i].load() == 1)
            {
                // Current column is already active
                for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
                {
                    columnMutexes_[*it].unlock();
                }
                if (debug_) {
                    LOG(DEBUG, "Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is already active");
                }
                return;
            }
            
            // Step 3c: Check if the column is already inhibited
            if (inhibitedCols_[i].load() == 1) {
                // Unlock and return
                for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
                {
                    columnMutexes_[*it].unlock();
                }
                if (debug_) {
                    LOG(DEBUG, "Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is already inhibited");
                }
                return;
            }

            // Step 3d: If strictLocalActivity is enabled, we can directly activate this column
            // since it passed the minOverlap check and wasn't inhibited in step 2
            if (strictLocalActivity_) {
                columnActive_[i].store(1);
                {
                    std::lock_guard<std::mutex> lock(activeColumnsMutex);
                    activeColumnsInd.push_back(i);
                }

                if (debug_) {
                    LOG(DEBUG, "Column index: " + std::to_string(i) +
                        " with overlap score: " + std::to_string(overlapGrid[i]) +
                        " is activated (strictLocalActivity enabled)");
                }
            }
            else {
                // Step 4: Process with standard inhibition when strictLocalActivity is disabled
                // Step 4a: Collect active neighbors to count them
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

                // Step 4b: If active neighbors < desiredLocalActivity, activate column
                if (static_cast<int>(activeNeighbors.size()) < desiredLocalActivity)
                {
                    // If we get here, we are "safe" to become active
                    columnActive_[i].store(1);
                    {
                        std::lock_guard<std::mutex> lock(activeColumnsMutex);
                        activeColumnsInd.push_back(i);
                    }

                    if (debug_) {
                        LOG(DEBUG, "Column index: " + std::to_string(i) +
                                  " with overlap score: " + std::to_string(overlapGrid[i]) +
                                  " is activated as it has fewer active neighbors (" +
                                  std::to_string(activeNeighbors.size()) + ") than desiredLocalActivity (" +
                                  std::to_string(desiredLocalActivity) + ").");
                    }
                }
                else
                {
                    // Step 4c: If active neighbors >= desiredLocalActivity, compare with lowest active neighbor
                    // Find the active column with the lowest overlap score
                    auto minIt = std::min_element(activeNeighborsOverlap.begin(), activeNeighborsOverlap.end());
                    // Calculate the index of the minimum overlap score in the activeNeighborsOverlap vector
                    int minIndex = std::distance(activeNeighborsOverlap.begin(), minIt);
                    float minOverlapScore = *minIt;
                    int minOverlapColIndex = activeNeighbors[minIndex];

                    // Step 4c-i: If current column has higher overlap, replace the lowest active neighbor
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
                        if (debug_) {
                            LOG(DEBUG, "    Deactivated column index: " + std::to_string(minOverlapColIndex) + " with overlap score: " + std::to_string(overlapGrid[minOverlapColIndex]));
                        }

                        // Activate the current column
                        columnActive_[i].store(1);

                        {
                            std::lock_guard<std::mutex> lock(activeColumnsMutex);
                            activeColumnsInd.push_back(i);
                            if (debug_) {
                                LOG(DEBUG, "     Activated column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]));
                            }
                        }
                    }
                    else
                    {
                        // Step 4c-ii: Otherwise, inhibit the current column
                        inhibitedCols_[i].store(1);
                        if (debug_) {
                            LOG(DEBUG, "    Column index: " + std::to_string(i) + " with overlap score: " + std::to_string(overlapGrid[i]) + " is inhibited as it has the lowest overlap score");
                            // Print out the active neighbors
                            LOG(DEBUG, "        Active neighbors: ");
                            overlap_utils::print_1d_vector(activeNeighbors);
                            // Print out the activeNeighborsOverlap vector
                            LOG(DEBUG, "        Active neighbors overlap: ");
                            overlap_utils::print_1d_vector(activeNeighborsOverlap);
                            // Print minOverlapColIndex
                            LOG(DEBUG, "        Min overlap column index: " + std::to_string(minOverlapColIndex));
                            // Print minIt
                            LOG(DEBUG, "        Min overlap score: " + std::to_string(minOverlapScore));
                            // Print minIndex
                            LOG(DEBUG, "        Min index: " + std::to_string(minIndex));
                        }
                    }
                }
            }

            // Step 5: End critical section and release locks
            // Unlock the mutexes in reverse order
            for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
            {
                columnMutexes_[*it].unlock();
            }

        }).name("ProcessColumns");
    }

} // namespace inhibition
