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
#include <cstring>  // for memset

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

        // Initialize serial implementation vectors
        serialColumnActive_.resize(numColumns_, 0);
        serialInhibitedCols_.resize(numColumns_, 0);
        serialNumColsActInNeigh_.resize(numColumns_, 0);

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
    /// @param[in] use_serial_sort_calc A boolean indicating whether to use the serial sorted implementation.
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
                                                    const std::vector<float>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape,
                                                    bool use_serial_sort_calc)
    {
        if (use_serial_sort_calc) {
            // Use the slower serial sorted implementation
            calculate_serial_sort_inhibition(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);
        } else {
            // Use the original parallel implementation
            tf::Taskflow taskflow;
            tf::Executor executor;

            // Define the taskflow structure for the inhibition calculation
            tf::Taskflow tf1, tf2;
            std::mutex activeColumnsMutex;

            // Add fast parallel clearing tasks to the main taskflow
            tf::Task clearColumnActive = taskflow.emplace([this]() {
                // Ultra-fast memory clearing using memset - much faster than individual atomic stores
                std::memset(reinterpret_cast<void*>(columnActive_.get()), 0, numColumns_ * sizeof(std::atomic<int>));
            }).name("FastClearColumnActive");
            
            tf::Task clearInhibitedCols = taskflow.emplace([this]() {
                std::memset(reinterpret_cast<void*>(inhibitedCols_.get()), 0, numColumns_ * sizeof(std::atomic<int>));
            }).name("FastClearInhibitedCols");
            
            tf::Task clearActiveColumnsInd = taskflow.emplace([this]() {
                activeColumnsInd_.clear();
            }).name("ClearActiveColumnsInd");

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

            // Process columns with potential overlap grid
            calculate_inhibition_for_column(potColOverlapGrid,
                                            activeColumnsInd_,
                                            neighbourColsLists_, colInNeighboursLists_,
                                            desiredLocalActivity_, minOverlap_,
                                            activeColumnsMutex, tf2,
                                            inhibitionCounts_, minorlyInhibitedColumns_,
                                            minorlyInhibitedMutex_, needsReprocessing_,
                                            iterationCount_,
                                            columnsToProcess_);

            // Set the order of the tasks using tf::Task objects
            tf::Task f1_task = taskflow.composed_of(tf1).name("ProcessOverlap");
            tf::Task f2_task = taskflow.composed_of(tf2).name("ProcessPotentialOverlap");
            
            // Establish task dependencies: clearing tasks must complete before processing tasks
            clearColumnActive.precede(f1_task);
            clearInhibitedCols.precede(f1_task);
            clearActiveColumnsInd.precede(f1_task);
            
            // Ensure f1 precedes f2 (existing dependency)
            f1_task.precede(f2_task);    

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
        //---------------------------------------------------------------------
        // 1.  Initial-isation (unchanged)
        //---------------------------------------------------------------------
        inhibitionCounts.assign(numColumns_, 0);
        minorlyInhibitedColumns.clear();
        iterationCount     = 0;
        needsReprocessing  = false;

        // Treat every column as "minorly inhibited" for the first pass
        columnsToProcess.resize(numColumns_);
        std::iota(columnsToProcess.begin(), columnsToProcess.end(), 0);

        //---------------------------------------------------------------------
        // 2.  Taskflow skeleton – only three tasks now
        //---------------------------------------------------------------------
        tf::Task init  = taskflow.emplace([]{}).name("Init");

        // Unified pass (re-used every iteration)
        tf::Task process_pass = taskflow.for_each_index(
            0ul,
            static_cast<size_t>(numColumns_),
            1ul,
            [&, this, desiredLocalActivity, minOverlap](size_t k)
        {
            LOG(DEBUG, "process_pass desiredLocalActivity="
                       + std::to_string(desiredLocalActivity)); 
            // --------------------------------------------------------------
            // --- Early exits ---------------------------------------------
            // --------------------------------------------------------------
            if (k >= columnsToProcess.size())                            return;
            int i = columnsToProcess[k];
            if (i < 0 || i >= numColumns_)                               return;
            if (overlapGrid[i] < minOverlap)                             return;
            if (inhibitionCounts[i] >= desiredLocalActivity)             return;

            //-----------------------------------------------------------------
            // Build neighbourhood and lock the related mutexes
            //-----------------------------------------------------------------
            std::vector<int> colsToLock = neighbourColsLists[i];
            colsToLock.insert(colsToLock.end(),
                              colInNeighboursLists[i].begin(),
                              colInNeighboursLists[i].end());
            colsToLock.push_back(i);
            std::sort(colsToLock.begin(), colsToLock.end());
            colsToLock.erase(std::unique(colsToLock.begin(),
                                         colsToLock.end()), colsToLock.end());

            for (int col : colsToLock)
                columnMutexes_[col].lock();

            //-----------------------------------------------------------------
            // Core logic  
            //-----------------------------------------------------------------
            
            // Compose the full neighbourhood, removing duplicates
            std::vector<int> allNeighbors = colsToLock;
            std::sort(allNeighbors.begin(), allNeighbors.end());
            allNeighbors.erase(std::unique(allNeighbors.begin(),
                                           allNeighbors.end()), allNeighbors.end());

            // Select eligible neighbours:
            //   1) overlap ≥ minOverlap
            //   2) neighbour has not yet reached its inhibition count limit
            std::vector<int> eligible;
            std::vector<float> eligibleOverlaps;
            for (int nb : allNeighbors)
            {
                if (overlapGrid[nb] >= minOverlap &&
                    inhibitionCounts[nb] < desiredLocalActivity)
                {
                    eligible.push_back(nb);
                    eligibleOverlaps.push_back(overlapGrid[nb]);
                }
            }

            //-----------------------------------------------------------------
            // Compute threshold from top-k overlaps (if we have enough)
            //-----------------------------------------------------------------
            float threshold = -1.0f;      // -1  ⇒  "everything is above threshold"
            if (eligible.size() >= static_cast<size_t>(desiredLocalActivity))
            {
                // indices sorted by overlap descending
                std::vector<size_t> idx(eligible.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(),
                                  idx.begin() + desiredLocalActivity,
                                  idx.end(),
                                  [&](size_t a, size_t b)
                                  { return eligibleOverlaps[a] > eligibleOverlaps[b]; });

                threshold = eligibleOverlaps[idx[desiredLocalActivity - 1]];
            }

            // Debug: threshold information
            if (this->debug_) {
                LOG(DEBUG, "Iteration " + std::to_string(iterationCount));
                LOG(DEBUG, "    Column: " + std::to_string(i));
                LOG(DEBUG, "       " + std::to_string(i) + " Overlap: " + std::to_string(overlapGrid[i]));
                LOG(DEBUG, "       " + std::to_string(i) + " Min Overlap: " + std::to_string(minOverlap));
                LOG(DEBUG, "       " + std::to_string(i) + " Eligible Neighbors: " + std::to_string(eligible.size()));
                LOG(DEBUG, "       " + std::to_string(i) + " Threshold: " + std::to_string(threshold));
            }

            //-----------------------------------------------------------------
            // Check if current column should be activated
            //-----------------------------------------------------------------
            bool shouldActivate = false;
            if (overlapGrid[i] >= minOverlap) {
                // Column is above threshold if threshold is -1 (no limit) or overlap >= threshold
                if (threshold < 0.0f || overlapGrid[i] >= threshold) {
                    shouldActivate = true;
                }
            }

            // Debug: activation decision
            if (this->debug_) {
                LOG(DEBUG, "    Col " + std::to_string(i) +
                           " shouldActivate=" + (shouldActivate ? "true" : "false") +
                           " overlap=" + std::to_string(overlapGrid[i]) +
                           " threshold=" + std::to_string(threshold));
            }

            //-----------------------------------------------------------------
            // If current column gets activated, process its neighbors
            //-----------------------------------------------------------------
            if (shouldActivate) {
                // Activate current column
                inhibitedCols_[i].store(0);
                if (!columnActive_[i].load()) {
                    columnActive_[i].store(1);
                    std::lock_guard<std::mutex> lk(activeColumnsMutex);
                    if (std::find(activeColumnsInd.begin(),
                                  activeColumnsInd.end(), i) == activeColumnsInd.end())
                        activeColumnsInd.push_back(i);
                }
                
                // Remove activated column from processing lists
                {
                    std::lock_guard<std::mutex> lk(minorlyInhibitedMutex);
                    minorlyInhibitedColumns.erase(
                        std::remove(minorlyInhibitedColumns.begin(),
                                    minorlyInhibitedColumns.end(), i),
                        minorlyInhibitedColumns.end());
                }

                if (this->debug_) {
                    LOG(DEBUG, "    Activated col " + std::to_string(i) +
                               " and removed from processing lists");
                }

                // Update inhibition counts of neighbors (only if current column got activated)
                for (int nb : allNeighbors)
                {
                    // Skip since we don't want to inhibit ourselves)
                    if (nb == i)  continue;

                    bool below = (threshold >= 0.0f) && (overlapGrid[nb] < threshold);
                    if (below)
                    {
                        // bump inhibition count
                        ++inhibitionCounts[nb];
                        inhibitedCols_[nb].store(1);

                        if (this->debug_) {
                            LOG(DEBUG, "    Inhibited col " + std::to_string(nb) +
                                       "  overlapGrid[nb]=" + std::to_string(overlapGrid[nb]) +
                                       "  newInhibCnt=" + std::to_string(inhibitionCounts[nb]));
                        }

                        // Queue again if still below the limit
                        if (inhibitionCounts[nb] < desiredLocalActivity)
                        {
                            std::lock_guard<std::mutex> lk(minorlyInhibitedMutex);
                            minorlyInhibitedColumns.push_back(nb);
                            needsReprocessing = true;
                        }

                        // deactivate if it had been active
                        if (columnActive_[nb].load())
                        {
                            columnActive_[nb].store(0);
                            std::lock_guard<std::mutex> lk(activeColumnsMutex);
                            activeColumnsInd.erase(
                                std::remove(activeColumnsInd.begin(),
                                            activeColumnsInd.end(), nb),
                                activeColumnsInd.end());
                        }
                    }
                    else // above threshold (but this neighbor didn't win since current column i won)
                    {
                        // Increment inhibition count since we didn't win but threshold is positive
                        // Also increment if current column has higher overlap than this neighbor
                        if (threshold >= 0.0f || overlapGrid[i] > overlapGrid[nb]) {
                            ++inhibitionCounts[nb];
                            
                            // Queue again if still below the limit
                            if (inhibitionCounts[nb] < desiredLocalActivity)
                            {
                                std::lock_guard<std::mutex> lk(minorlyInhibitedMutex);
                                minorlyInhibitedColumns.push_back(nb);
                                needsReprocessing = true;
                            }
                        }
                    }
                }
            }

            // Print the inhibition counts as 2d vector
            if (this->debug_) {
                LOG(DEBUG, "inhibitionCounts=");
                overlap_utils::print_2d_vector(inhibitionCounts, {height_, width_});
            }
            // Print active columns as 2d vector
            if (this->debug_) {
                LOG(DEBUG, "Active Columns (2D):");
                overlap_utils::print_2d_atomic_array(columnActive_.get(), {height_, width_});
            }

            // Print the minorlyInhibited columns
            if (this->debug_) {
                LOG(DEBUG, "Minorly Inhibited Columns:");
                overlap_utils::print_1d_vector(minorlyInhibitedColumns);
            }

            //-----------------------------------------------------------------
            // unlock in reverse order
            //-----------------------------------------------------------------
            for (auto it = colsToLock.rbegin(); it != colsToLock.rend(); ++it)
                columnMutexes_[*it].unlock();
        }).name("ProcessColumns");

        // Decide whether we need another iteration
        tf::Task prepare_next = taskflow.emplace([&, this, desiredLocalActivity]() {
            LOG(DEBUG, "prepare_next desiredLocalActivity="
                       + std::to_string(desiredLocalActivity)); 

            ++iterationCount;

            {
                std::lock_guard<std::mutex> lk(minorlyInhibitedMutex);
                columnsToProcess.swap(minorlyInhibitedColumns);
                minorlyInhibitedColumns.clear(); // Clear for next iteration
            }
            bool keepGoing = !columnsToProcess.empty() &&
                                iterationCount <= (desiredLocalActivity);
            needsReprocessing = keepGoing;

            if (this->debug_) {
                LOG(DEBUG, "PrepareNext – iteration " + std::to_string(iterationCount) +
                           "  columnsToProcess.size()=" + std::to_string(columnsToProcess.size()));
                LOG(DEBUG, "  columnsToProcess=");
                if (!columnsToProcess.empty()) {
                    overlap_utils::print_1d_vector(columnsToProcess);
                }
            }
            LOG(DEBUG, "iterationCount="
                       + std::to_string(iterationCount));
            return keepGoing ? 0 : 1;   // 0 → loop again, 1 → stop
        }).name("PrepareNext");

        tf::Task done = taskflow.emplace([&, this](){
            if (this->debug_) {
                LOG(DEBUG, "Done - Final state:");
                LOG(DEBUG, "Inhibition Counts:");
                overlap_utils::print_2d_vector(inhibitionCounts, {height_, width_});
                LOG(DEBUG, "Active Columns:");
                overlap_utils::print_2d_atomic_array(columnActive_.get(), {height_, width_});
                LOG(DEBUG, "columnsToProcess Columns:");
                overlap_utils::print_1d_vector(columnsToProcess);
            }
        }).name("Done");

        // Connect the tasks
        init.precede(process_pass);
        process_pass.precede(prepare_next, done);
        prepare_next.precede(process_pass, done);
    }

    void InhibitionCalculator::calculate_serial_sort_inhibition(const std::vector<float>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                               const std::vector<float>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape)
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

        // Clear the serial vectors
        std::fill(serialColumnActive_.begin(), serialColumnActive_.end(), 0);
        std::fill(serialInhibitedCols_.begin(), serialInhibitedCols_.end(), 0);
        std::fill(serialNumColsActInNeigh_.begin(), serialNumColsActInNeigh_.end(), 0);
        activeColumnsInd_.clear();

        // Define the taskflow structure for the inhibition calculation
        tf::Taskflow tf1, tf2, tf3, tf4, tf5, tf6;

        // Task to add a small tie breaker to the overlap grid
        add_tie_breaker(const_cast<std::vector<float>&>(colOverlapGrid), false, tf1);

        // Sort columns by overlap values using parallel_sort
        inhibition_utils::parallel_sort_ind(sortedIndices_colOver, colOverlapGrid, tf2);

        // Process columns from highest to lowest overlap based on the sorted indices
        tf3.emplace([&]() {
            calculate_serial_sort_inhibition_for_column(sortedIndices_colOver, colOverlapGrid, 
                                            serialInhibitedCols_, serialColumnActive_, 
                                            serialNumColsActInNeigh_, activeColumnsInd_, 
                                            neighbourColsLists_, colInNeighboursLists_, 
                                            desiredLocalActivity_, minOverlap_);
        }).name("ProcessOverlap");

        // Add a small tie breaker to the potential overlap grid
        add_tie_breaker(const_cast<std::vector<float>&>(potColOverlapGrid), false, tf4);

        // Sort columns by potential overlap values using parallel_sort
        inhibition_utils::parallel_sort_ind(sortedIndices_potOver, potColOverlapGrid, tf5);

        // Process columns with potential overlap values
        tf6.emplace([&]() {
            calculate_serial_sort_inhibition_for_column(sortedIndices_potOver, potColOverlapGrid, 
                                            serialInhibitedCols_, serialColumnActive_, 
                                            serialNumColsActInNeigh_, activeColumnsInd_, 
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

        // Run the constructed taskflow graph
        tf::Future<void> fu = executor.run(taskflow);
        fu.wait(); // Block until the execution completes

        // Copy results from serial vectors to atomic arrays for consistency
        for (int i = 0; i < numColumns_; ++i) {
            columnActive_[i].store(serialColumnActive_[i]);
            inhibitedCols_[i].store(serialInhibitedCols_[i]);
        }

        // Print the results using LOG and overlap_utils functions
        if (debug_) {
            LOG(DEBUG, "Final Results (Serial Sorted):");

            // Print the overlap grid after tie-breakers
            LOG(DEBUG, "Overlap Grid (with Tie-Breakers):");
            overlap_utils::print_2d_vector(colOverlapGrid, colOverlapGridShape);

            // Print the inhibited columns
            LOG(DEBUG, "Inhibited Columns:");
            overlap_utils::print_1d_vector(serialInhibitedCols_);

            // Print the active columns
            LOG(DEBUG, "Active Columns:");
            overlap_utils::print_2d_vector(serialColumnActive_, colOverlapGridShape);

            // Print the list of active column indices
            LOG(DEBUG, "Active Columns Indices:");
            overlap_utils::print_1d_vector(activeColumnsInd_);
        }
    }

    void InhibitionCalculator::calculate_serial_sort_inhibition_for_column(const std::vector<int>& sortedIndices,
                                                                          const std::vector<float>& overlapGrid,
                                                                          std::vector<int>& inhibitedCols, 
                                                                          std::vector<int>& columnActive, 
                                                                          std::vector<int>& numColsActInNeigh, 
                                                                          std::vector<int>& activeColumnsInd, 
                                                                          const std::vector<std::vector<int>>& neighbourColsLists, 
                                                                          const std::vector<std::vector<int>>& colInNeighboursLists, 
                                                                          int desiredLocalActivity,
                                                                          int minOverlap)
    {
        // Process columns in sorted order (highest overlap first)
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

    void InhibitionCalculator::add_tie_breaker(std::vector<float>& overlapGrid, bool addColBias, tf::Taskflow &taskflow)
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
                for (size_t i = 0; i < overlapGrid.size(); ++i)
                {
                    overlapGrid[i] += tieBreaker[i];
                }
            }
        }).name("AddTieBreaker");
    }

} // namespace inhibition
