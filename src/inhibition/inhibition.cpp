// inhibition.cpp

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include "inhibition.hpp"
#include "utilities/logger.hpp"
#include "htm_flow/overlap_utils.hpp"

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
          debug_(debug),
          columnsToProcess_(numColumns_)
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
                                        desiredLocalActivity_, minOverlap_,
                                        activeColumnsMutex, tf1,
                                        inhibitionCounts_, minorlyInhibitedColumns_,
                                        minorlyInhibitedMutex_, needsReprocessing_,
                                        iterationCount_,
                                        columnsToProcess_);

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
    ///   d. Keep track of how many times each column is inhibited
    ///   e. Maintain a list of columns that have been inhibited less than desiredLocalActivity times
    ///   f. After initial processing, reprocess these minorly inhibited columns until the desired 
    ///      local activity is achieved or maximum iterations reached
    ///
    /// 3. Release all locks in reverse order
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
        tf::Taskflow& taskflow,
        std::vector<int>& inhibitionCounts,
        std::vector<int>& minorlyInhibitedColumns,
        std::mutex& minorlyInhibitedMutex,
        bool& needsReprocessing,
        int& iterationCount,
        std::vector<int>& columnsToProcess)
    {
        // Initialize with local variables (not class members)
        inhibitionCounts.resize(numColumns_);
        std::fill(inhibitionCounts.begin(), inhibitionCounts.end(), 0);
        minorlyInhibitedColumns.clear();
        needsReprocessing = false;
        iterationCount = 0;
        
        // Initialize task - does nothing
        tf::Task init = taskflow.emplace([](){}).name("Init");
        
        // Process all columns initially and check if reprocessing is needed
        tf::Task initial_pass = taskflow.for_each_index(
            0ul, static_cast<size_t>(numColumns_), 1ul,
            [&, this, desiredLocalActivity, minOverlap](size_t idx)
        {
            int i = static_cast<int>(idx);
            bool shouldProcess = true;
            bool shouldActivate = false;

            // Check if column has sufficient overlap
            if (overlapGrid[i] < minOverlap) {
                shouldProcess = false;
            }

            if (shouldProcess) {
                // Build the list of columns to lock
                std::vector<int> colsToLock = neighbourColsLists[i];
                colsToLock.insert(colsToLock.end(), colInNeighboursLists[i].begin(), colInNeighboursLists[i].end());
                colsToLock.push_back(i);

                // Remove duplicates and sort to prevent deadlocks
                std::sort(colsToLock.begin(), colsToLock.end());
                colsToLock.erase(std::unique(colsToLock.begin(), colsToLock.end()), colsToLock.end());

                // Lock the mutexes in ascending order
                for (int colIndex : colsToLock) {
                    if (colIndex >= 0 && colIndex < static_cast<int>(numColumns_)) {
                        columnMutexes_[colIndex].lock();
                    }
                }

                if (debug_) {
                    LOG(DEBUG, "Column index: " + std::to_string(i) + 
                          " with overlap score: " + std::to_string(overlapGrid[i]) + 
                          " is being processed in initial pass");
                }

                // Begin critical section
                
                // Check column state
                bool isActive = (columnActive_[i].load() == 1);
                bool isInhibited = (inhibitedCols_[i].load() == 1);

                // Ensure consistent state
                if (isActive && isInhibited) {
                    columnActive_[i].store(0);
                    
                    // Remove from active columns list
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
                    
                    isActive = false;
                }

                if (!isActive && !isInhibited) {
                    // Combine neighbor sets to get neighborhood
                    std::vector<int> allNeighbors = neighbourColsLists[i];
                    if (!colInNeighboursLists[i].empty()) {
                        allNeighbors.insert(allNeighbors.end(), colInNeighboursLists[i].begin(), colInNeighboursLists[i].end());
                    }
                    
                    // Add current column to neighborhood
                    allNeighbors.push_back(i);
                    
                    // Remove duplicates
                    std::sort(allNeighbors.begin(), allNeighbors.end());
                    allNeighbors.erase(std::unique(allNeighbors.begin(), allNeighbors.end()), allNeighbors.end());
                    
                    // Filter eligible neighbors (overlap >= minOverlap)
                    std::vector<int> eligibleNeighbors;
                    std::vector<float> eligibleOverlaps;
                    
                    for (int nb : allNeighbors) {
                        if (nb >= 0 && nb < static_cast<int>(overlapGrid.size()) && overlapGrid[nb] >= minOverlap) {
                            eligibleNeighbors.push_back(nb);
                            eligibleOverlaps.push_back(overlapGrid[nb]);
                        }
                    }
                    
                    // Sort by overlap score
                    std::vector<int> indices(eligibleNeighbors.size());
                    std::iota(indices.begin(), indices.end(), 0);
                    
                    if (!eligibleOverlaps.empty() && !indices.empty()) {
                        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                            if (a < 0 || a >= static_cast<int>(eligibleOverlaps.size()) || 
                                b < 0 || b >= static_cast<int>(eligibleOverlaps.size())) {
                                return false;
                            }
                            return eligibleOverlaps[a] > eligibleOverlaps[b];
                        });
                        
                        if (eligibleNeighbors.size() > static_cast<size_t>(desiredLocalActivity)) {
                            // Get threshold from Nth highest overlap
                            int thresholdIdx = desiredLocalActivity - 1;
                            if (thresholdIdx >= 0 && thresholdIdx < static_cast<int>(indices.size()) && 
                                indices[thresholdIdx] >= 0 && indices[thresholdIdx] < static_cast<int>(eligibleOverlaps.size())) {
                                float threshold = eligibleOverlaps[indices[thresholdIdx]];
                                
                                // Mark columns below threshold as inhibited
                                for (int nb : allNeighbors) {
                                    if (nb >= 0 && nb < static_cast<int>(overlapGrid.size()) && 
                                        overlapGrid[nb] < threshold && overlapGrid[nb] >= minOverlap) {
                                        // If active, deactivate first
                                        if (nb < static_cast<int>(numColumns_) && columnActive_[nb].load() == 1) {
                                            columnActive_[nb].store(0);
                                            
                                            // Remove from active columns list
                                            {
                                                std::lock_guard<std::mutex> lock(activeColumnsMutex);
                                                auto it = std::find(activeColumnsInd.begin(), activeColumnsInd.end(), nb);
                                                if (it != activeColumnsInd.end()) {
                                                    activeColumnsInd.erase(it);
                                                    if (debug_) {
                                                        LOG(DEBUG, "    Deactivated previously active column index: " + 
                                                              std::to_string(nb) + " due to being below threshold");
                                                    }
                                                }
                                            }
                                        }
                                        
                                        // Increment inhibition count
                                        if (nb >= 0 && nb < static_cast<int>(inhibitionCounts.size())) {
                                            inhibitionCounts[nb]++;
                                        }
                                        
                                        // Mark as inhibited
                                        if (nb < static_cast<int>(numColumns_)) {
                                            inhibitedCols_[nb].store(1);
                                        }
                                        
                                        // Add to minorly inhibited list if count is below threshold
                                        if (nb >= 0 && nb < static_cast<int>(inhibitionCounts.size()) && 
                                            inhibitionCounts[nb] < desiredLocalActivity) {
                                            std::lock_guard<std::mutex> lock(minorlyInhibitedMutex);
                                            if (std::find(minorlyInhibitedColumns.begin(), minorlyInhibitedColumns.end(), nb) 
                                                == minorlyInhibitedColumns.end()) {
                                                minorlyInhibitedColumns.push_back(nb);
                                                needsReprocessing = true;
                                            }
                                        }
                                        
                                        if (debug_) {
                                            LOG(DEBUG, "    Column index: " + std::to_string(nb) + 
                                                  " with overlap score: " + std::to_string(overlapGrid[nb]) + 
                                                  " inhibited by threshold " + std::to_string(threshold) + 
                                                  " in neighborhood of column " + std::to_string(i) + 
                                                  " (inhibition count: " + std::to_string(
                                                      nb < static_cast<int>(inhibitionCounts.size()) ? 
                                                      inhibitionCounts[nb] : -1) + ")");
                                        }
                                    }
                                }

                                // Mark columns above threshold as not inhibited and activate current column
                                for (int nb : allNeighbors) {
                                    if (nb >= 0 && nb < static_cast<int>(overlapGrid.size()) && 
                                        overlapGrid[nb] >= threshold && overlapGrid[nb] >= minOverlap) {
                                        if (nb < static_cast<int>(numColumns_)) {
                                            inhibitedCols_[nb].store(0);
                                        }
                                        
                                        // If this is current column, mark for activation
                                        if (nb == i) {
                                            shouldActivate = true;
                                        }
                                    }
                                }
                            }
                        } else {
                            // If fewer eligible columns than desired local activity, all are active
                            shouldActivate = true;
                        }
                    } else {
                        // No eligible neighbors, activate this column
                        shouldActivate = true;
                    }
                } else {
                    // Just log status for debugging
                    if (debug_) {
                        if (isActive) {
                            LOG(DEBUG, "Column index: " + std::to_string(i) + " with overlap score: " + 
                                  std::to_string(overlapGrid[i]) + " is already active");
                        } else if (isInhibited) {
                            LOG(DEBUG, "Column index: " + std::to_string(i) + " with overlap score: " + 
                                  std::to_string(overlapGrid[i]) + " is already inhibited");
                        }
                    }
                }

                // Apply activation if determined
                if (shouldActivate && !inhibitedCols_[i].load()) {
                    columnActive_[i].store(1);
                    {
                        std::lock_guard<std::mutex> lock(activeColumnsMutex);
                        if (std::find(activeColumnsInd.begin(), activeColumnsInd.end(), i) == activeColumnsInd.end()) {
                            activeColumnsInd.push_back(i);
                        }
                    }

                    if (debug_) {
                        LOG(DEBUG, "Column index: " + std::to_string(i) +
                            " with overlap score: " + std::to_string(overlapGrid[i]) +
                            " is activated (strictLocalActivity enabled, initial pass)");
                    }
                }

                // End critical section and release locks
                for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it) {
                    int colIndex = *it;
                    if (colIndex >= 0 && colIndex < static_cast<int>(numColumns_)) {
                        columnMutexes_[colIndex].unlock();
                    }
                }
            }
        }).name("ProcessColumns_Initial");
        
        // Process minorly inhibited columns and check if more reprocessing is needed
        tf::Task prepare_reprocess = taskflow.emplace([&]() -> int {
            ++iterationCount;

            {
                std::lock_guard<std::mutex> lk(minorlyInhibitedMutex);
                columnsToProcess.swap(minorlyInhibitedColumns);
            }
            needsReprocessing = !columnsToProcess.empty() &&
                                iterationCount < desiredLocalActivity;

            return needsReprocessing ? 0 : 1;   // 0 → keep looping, 1 → stop
        }).name("PrepareReprocess");
        
        // Process minorly inhibited columns and check if more reprocessing is needed
        tf::Task reprocess_pass = taskflow.for_each_index(
            0ul,
            static_cast<size_t>(numColumns_),  // iterate over the full column set; early-return if index is out of range for this iteration
            1ul,
            [&, this, desiredLocalActivity, minOverlap](size_t k)
        {
            if(k >= columnsToProcess.size()) return;
            int i = columnsToProcess[k];
            bool shouldActivate = false;
            
            // Safety check
            if (i < 0 || i >= static_cast<int>(numColumns_) || i >= static_cast<int>(overlapGrid.size())) {
                return;
            }
            
            // Skip if overlap is too low
            if (overlapGrid[i] < minOverlap) {
                return;
            }
            
            // Build the list of columns to lock
            std::vector<int> colsToLock;
            if (i < static_cast<int>(neighbourColsLists.size())) {
                colsToLock = neighbourColsLists[i];
                if (i < static_cast<int>(colInNeighboursLists.size()) && !colInNeighboursLists[i].empty()) {
                    colsToLock.insert(colsToLock.end(), colInNeighboursLists[i].begin(), colInNeighboursLists[i].end());
                }
            }
            colsToLock.push_back(i);
            
            // Remove duplicates and sort to prevent deadlocks
            std::sort(colsToLock.begin(), colsToLock.end());
            colsToLock.erase(std::unique(colsToLock.begin(), colsToLock.end()), colsToLock.end());
            
            // Lock the mutexes in ascending order - only for valid indices
            for (int colIndex : colsToLock) {
                if (colIndex >= 0 && colIndex < static_cast<int>(numColumns_)) {
                    columnMutexes_[colIndex].lock();
                }
            }
                            
            if (debug_) {
                LOG(DEBUG, "Column index: " + std::to_string(i) +
                        " with overlap score: " + std::to_string(overlapGrid[i]) +
                      " is being reprocessed in iteration " + std::to_string(iterationCount));
            }
            
            // Begin critical section
            
            // Check if column is already active
            bool isActive = (columnActive_[i].load() == 1);
            
            if (!isActive) {
                // Combine neighbor sets to get neighborhood
                std::vector<int> allNeighbors;
                if (i < static_cast<int>(neighbourColsLists.size())) {
                    allNeighbors = neighbourColsLists[i];
                    if (i < static_cast<int>(colInNeighboursLists.size())) {
                        allNeighbors.insert(allNeighbors.end(), colInNeighboursLists[i].begin(), colInNeighboursLists[i].end());
                    }
                }
                
                // Add current column to neighborhood
                allNeighbors.push_back(i);
                
                // Remove duplicates
                std::sort(allNeighbors.begin(), allNeighbors.end());
                allNeighbors.erase(std::unique(allNeighbors.begin(), allNeighbors.end()), allNeighbors.end());
                
                // Filter eligible neighbors (active or minorly inhibited)
                std::vector<int> eligibleNeighbors;
                std::vector<float> eligibleOverlaps;
                
                for (int nb : allNeighbors) {
                    if (nb < 0 || nb >= static_cast<int>(overlapGrid.size()) || nb >= static_cast<int>(numColumns_)) {
                        continue;
                    }
                    
                    bool eligible = overlapGrid[nb] >= minOverlap && 
                                  (columnActive_[nb].load() == 1 || 
                                   (inhibitedCols_[nb].load() == 1 && 
                                    nb < static_cast<int>(inhibitionCounts.size()) && 
                                    inhibitionCounts[nb] < desiredLocalActivity));
                                       
                    if (eligible) {
                        eligibleNeighbors.push_back(nb);
                        eligibleOverlaps.push_back(overlapGrid[nb]);
                    }
                }
                
                // Sort by overlap score
                std::vector<int> indices(eligibleNeighbors.size());
                std::iota(indices.begin(), indices.end(), 0);
                
                if (!eligibleOverlaps.empty() && !indices.empty()) {
                    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                        if (a < 0 || a >= static_cast<int>(eligibleOverlaps.size()) || 
                            b < 0 || b >= static_cast<int>(eligibleOverlaps.size())) {
                            return false;
                        }
                        return eligibleOverlaps[a] > eligibleOverlaps[b];
                    });
                    
                    if (eligibleNeighbors.size() > static_cast<size_t>(desiredLocalActivity)) {
                        // Get threshold from Nth highest overlap
                        int thresholdIdx = desiredLocalActivity - 1;
                        if (thresholdIdx >= 0 && thresholdIdx < static_cast<int>(indices.size()) && 
                            indices[thresholdIdx] >= 0 && indices[thresholdIdx] < static_cast<int>(eligibleOverlaps.size())) {
                            float threshold = eligibleOverlaps[indices[thresholdIdx]];
                            
                            // Mark columns below threshold as inhibited
                            for (int nb : allNeighbors) {
                                if (nb < 0 || nb >= static_cast<int>(overlapGrid.size()) || nb >= static_cast<int>(numColumns_)) {
                                    continue;
                                }
                                
                                if (overlapGrid[nb] < threshold && overlapGrid[nb] >= minOverlap) {
                                    // If active, deactivate first
                                    if (columnActive_[nb].load() == 1) {
                                        columnActive_[nb].store(0);
                                        
                                        // Remove from active columns list
                                {
                                    std::lock_guard<std::mutex> lock(activeColumnsMutex);
                                                auto it = std::find(activeColumnsInd.begin(), activeColumnsInd.end(), nb);
                                    if (it != activeColumnsInd.end()) {
                                        activeColumnsInd.erase(it);
                                                    if (debug_) {
                                                        LOG(DEBUG, "    Deactivated previously active column index: " + 
                                                              std::to_string(nb) + " due to being below threshold");
                                                    }
                                                }
                                            }
                                        }
                                        
                                        // Increment inhibition count
                                        if (nb >= 0 && nb < static_cast<int>(inhibitionCounts.size())) {
                                            inhibitionCounts[nb]++;
                                        }
                                        
                                        // Mark as inhibited
                                        inhibitedCols_[nb].store(1);
                                        
                                        // Add to minorly inhibited list if count is below threshold
                                        if (nb >= 0 && nb < static_cast<int>(inhibitionCounts.size()) && 
                                            inhibitionCounts[nb] < desiredLocalActivity) {
                                            std::lock_guard<std::mutex> lock(minorlyInhibitedMutex);
                                            if (std::find(minorlyInhibitedColumns.begin(), minorlyInhibitedColumns.end(), nb) 
                                                == minorlyInhibitedColumns.end()) {
                                                minorlyInhibitedColumns.push_back(nb);
                                                needsReprocessing = true;
                                            }
                                        }
                                    }
                                }
                                
                                // Mark columns above threshold as not inhibited and activate current column
                                for (int nb : allNeighbors) {
                                    if (nb >= 0 && nb < static_cast<int>(overlapGrid.size()) && 
                                        nb < static_cast<int>(numColumns_) &&
                                        overlapGrid[nb] >= threshold && overlapGrid[nb] >= minOverlap) {
                                        inhibitedCols_[nb].store(0);
                                        
                                        // If this is current column, mark for activation
                                        if (nb == i) {
                                shouldActivate = true;
                                        }
                                    }
                                }
                            }
                        } else {
                            // If fewer eligible columns than desired local activity, all are active
                            shouldActivate = true;
                        }
                    } else {
                        // No eligible neighbors, this column should be active
                        shouldActivate = true;
                    }

                    // Apply activation if determined
                    if (shouldActivate) {
                        columnActive_[i].store(1);
                        inhibitedCols_[i].store(0);
                        {
                            std::lock_guard<std::mutex> lock(activeColumnsMutex);
                            if (std::find(activeColumnsInd.begin(), activeColumnsInd.end(), i) == activeColumnsInd.end()) {
                            activeColumnsInd.push_back(i);
                            }
                        }

                        if (debug_) {
                                LOG(DEBUG, "Column index: " + std::to_string(i) +
                                    " with overlap score: " + std::to_string(overlapGrid[i]) +
                                  " is activated in reprocessing iteration " + std::to_string(iterationCount));
                        }
                    }
                }
                
                // End critical section and release locks
                for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it) {
                    int colIndex = *it;
                    if (colIndex >= 0 && colIndex < static_cast<int>(numColumns_)) {
                        columnMutexes_[colIndex].unlock();
                    }
                }
        }).name("ReprocessColumns");
        
        // Final task
        tf::Task done = taskflow.emplace([](){}).name("Done");
        
        // Set up the flow with conditional branching
        init.precede(initial_pass);
        initial_pass.precede(prepare_reprocess, done);
        prepare_reprocess.precede(reprocess_pass, done);
        reprocess_pass.precede(prepare_reprocess, done);
    }

} // namespace inhibition
