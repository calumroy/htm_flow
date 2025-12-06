// inhibition.cpp

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
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
                                               bool strictLocalActivity, bool debug, bool useTieBreaker)
        : width_(width), height_(height), numColumns_(width * height),
          potentialWidth_(potentialInhibWidth), potentialHeight_(potentialInhibHeight),
          desiredLocalActivity_(desiredLocalActivity), minOverlap_(minOverlap),
          centerInhib_(centerInhib), wrapMode_(wrapMode), strictLocalActivity_(strictLocalActivity),
          activeColumnsInd_(),
          columnMutexes_(numColumns_),
          debug_(debug), useTieBreaker_(useTieBreaker),
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

        // Initialize tie breaker vector if enabled
        if (useTieBreaker_) {
            tieBreaker_.resize(numColumns_, 0.0f);
            // Generate tie breaker values similar to overlap calculator
            int numColumns = width_ * height_;
            float normValue = 1.0f / float(2 * numColumns + 2);
            std::vector<float> uniqueValues(numColumns);
            for (int i = 0; i < numColumns; ++i) {
                uniqueValues[i] = (i + 1) * normValue;
            }
            
            // Create a vector of indices to shuffle
            std::vector<int> indices(numColumns);
            std::iota(indices.begin(), indices.end(), 0);
            
            // Shuffle the indices using the same seed for reproducibility
            std::mt19937 gen(1);
            std::shuffle(indices.begin(), indices.end(), gen);

            // Assign the shuffled unique values to the tieBreaker
            for (int j = 0; j < numColumns; ++j) {
                int shuffledIndex = indices[j];
                tieBreaker_[j] = uniqueValues[shuffledIndex];
            }

            if (debug_) {
                LOG(DEBUG, "Tie-Breaker Values:");
                overlap_utils::print_1d_vector(tieBreaker_);
            }
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
    /// validate_input_parameters   Validates input parameters for inhibition calculation.
    ///
    /// @param[in] colOverlapGrid A vector representing the overlap values for each column.
    /// @param[in] colOverlapGridShape A pair representing the shape of the colOverlapGrid (rows, cols).
    /// @param[in] potColOverlapGrid A vector representing the potential overlap values for each column.
    /// @param[in] potColOverlapGridShape A pair representing the shape of the potColOverlapGrid (rows, cols).
    ///
    /// @throws std::invalid_argument if any validation check fails.
    ///-----------------------------------------------------------------------------
    void InhibitionCalculator::validate_input_parameters(const std::vector<float>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                         const std::vector<float>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape)
    {
        int expectedSize = colOverlapGridShape.first * colOverlapGridShape.second;
        int expectedPotSize = potColOverlapGridShape.first * potColOverlapGridShape.second;
        
        // Check colOverlapGrid dimensions
        if (colOverlapGrid.size() != static_cast<size_t>(expectedSize)) {
            std::string errorMsg = "colOverlapGrid size mismatch! Expected: " + std::to_string(expectedSize) + 
                                 " (rows=" + std::to_string(colOverlapGridShape.first) + 
                                 " × cols=" + std::to_string(colOverlapGridShape.second) + 
                                 "), but got: " + std::to_string(colOverlapGrid.size());
            LOG(ERROR, errorMsg);
            throw std::invalid_argument(errorMsg);
        }
        
        // Check potColOverlapGrid dimensions
        if (potColOverlapGrid.size() != static_cast<size_t>(expectedPotSize)) {
            std::string errorMsg = "potColOverlapGrid size mismatch! Expected: " + std::to_string(expectedPotSize) + 
                                 " (rows=" + std::to_string(potColOverlapGridShape.first) + 
                                 " × cols=" + std::to_string(potColOverlapGridShape.second) + 
                                 "), but got: " + std::to_string(potColOverlapGrid.size());
            LOG(ERROR, errorMsg);
            throw std::invalid_argument(errorMsg);
        }
        
        // Check that grid dimensions match the calculator's expected dimensions
        if (expectedSize != numColumns_) {
            std::string errorMsg = std::string("Grid dimensions don't match InhibitionCalculator configuration! ") +
                                 "Grid size: " + std::to_string(expectedSize) + 
                                 ", Calculator expects: " + std::to_string(numColumns_) + 
                                 " (width=" + std::to_string(width_) + " × height=" + std::to_string(height_) + ")";
            LOG(ERROR, errorMsg);
            throw std::invalid_argument(errorMsg);
        }
        
        // Check tie-breaker vector size if enabled
        if (useTieBreaker_ && tieBreaker_.size() != static_cast<size_t>(numColumns_)) {
            std::string errorMsg = "Tie-breaker vector size mismatch! Expected: " + std::to_string(numColumns_) + 
                                 ", but got: " + std::to_string(tieBreaker_.size());
            LOG(ERROR, errorMsg);
            throw std::invalid_argument(errorMsg);
        }
        
        if (debug_) {
            LOG(DEBUG, "Input validation passed:");
            LOG(DEBUG, "  colOverlapGrid size: " + std::to_string(colOverlapGrid.size()));
            LOG(DEBUG, "  colOverlapGridShape: " + std::to_string(colOverlapGridShape.first) + "×" + std::to_string(colOverlapGridShape.second));
            LOG(DEBUG, "  potColOverlapGrid size: " + std::to_string(potColOverlapGrid.size()));
            LOG(DEBUG, "  potColOverlapGridShape: " + std::to_string(potColOverlapGridShape.first) + "×" + std::to_string(potColOverlapGridShape.second));
            LOG(DEBUG, "  Calculator dimensions: " + std::to_string(width_) + "×" + std::to_string(height_) + " = " + std::to_string(numColumns_));
            if (useTieBreaker_) {
                LOG(DEBUG, "  Tie-breaker vector size: " + std::to_string(tieBreaker_.size()));
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
    /// 1. Validate Input Parameters:
    ///    - Check grid dimensions match expected sizes and calculator configuration.
    /// 2. Add Tie-Breakers (if enabled):
    ///    - Apply small tie-breaker values to resolve overlap score ties.
    /// 3. Process Columns:
    ///    - Serial mode: Sort by overlap scores, then process highest to lowest.
    ///    - Parallel mode: Process all columns concurrently with iterative refinement.
    /// 4. Apply Inhibition Rules:
    ///    - Enforce local activity constraints and neighbor competition.
    /// 5. Finalize Results:
    ///    - Store final activation state of each column.
    ///-----------------------------------------------------------------------------

    void InhibitionCalculator::calculate_inhibition(const std::vector<float>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                    const std::vector<float>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape,
                                                    bool use_serial_sort_calc)
    {
        // Validate input parameters
        validate_input_parameters(colOverlapGrid, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);

        // When strictLocalActivity is enabled, always use the serial implementation
        // as the parallel implementation cannot guarantee deterministic results
        // that strictly respect the local activity constraint.
        if (use_serial_sort_calc || strictLocalActivity_) {
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

            // Add tie breaker tasks if enabled
            tf::Taskflow tieBreakerTf1, tieBreakerTf2;
            tf::Task addTieBreakerCol, addTieBreakerPot;
            if (useTieBreaker_) {
                parallel_add_tie_breaker(const_cast<std::vector<float>&>(colOverlapGrid), tieBreakerTf1);
                addTieBreakerCol = taskflow.composed_of(tieBreakerTf1).name("AddTieBreakerCol");
                
                parallel_add_tie_breaker(const_cast<std::vector<float>&>(potColOverlapGrid), tieBreakerTf2);
                addTieBreakerPot = taskflow.composed_of(tieBreakerTf2).name("AddTieBreakerPot");
            }

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
            
            // Add tie breaker dependencies if enabled
            if (useTieBreaker_) {
                addTieBreakerCol.precede(f1_task);
                addTieBreakerPot.precede(f2_task);
            }
            
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

                // Print inhibition counts
                LOG(DEBUG, "Inhibition Counts:");
                overlap_utils::print_2d_vector(inhibitionCounts_, colOverlapGridShape);

                // Print the inhibited columns
                LOG(DEBUG, "Inhibited Columns:");
                overlap_utils::print_2d_atomic_array(inhibitedCols_.get(), colOverlapGridShape);

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
            // --------------------------------------------------------------
            // --- Early exits ---------------------------------------------
            // --------------------------------------------------------------
            if (k >= columnsToProcess.size())                            return;
            int i = columnsToProcess[k];
            if (i < 0 || i >= numColumns_)                               return;
            if (overlapGrid[i] < minOverlap)                             return;
            if (inhibitionCounts[i] >= desiredLocalActivity)             return;

            if (this->debug_) {
                LOG(DEBUG, "Iteration " + std::to_string(iterationCount) +
                           " process_pass i=" + std::to_string(i) +
                           " desiredLocalActivity=" + std::to_string(desiredLocalActivity)); 
            }

            //-----------------------------------------------------------------
            // Build neighbourhood and lock the related mutexes
            // For stricter local activity enforcement, we need to lock not just
            // immediate neighbors, but also neighbors-of-neighbors to safely
            // check if activation would violate the constraint.
            //-----------------------------------------------------------------
            std::vector<int> colsToLock = neighbourColsLists[i];
            colsToLock.insert(colsToLock.end(),
                              colInNeighboursLists[i].begin(),
                              colInNeighboursLists[i].end());
            colsToLock.push_back(i);
            
            // For stricter local activity, also include neighbors of columns that have
            // this column in their neighborhood (needed for the constraint check)
            for (int j : colInNeighboursLists[i]) {
                colsToLock.insert(colsToLock.end(),
                                  neighbourColsLists[j].begin(),
                                  neighbourColsLists[j].end());
            }
            
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

            // Additional check: verify that activating this column won't cause any
            // neighbor's neighborhood to exceed the desiredLocalActivity limit.
            // For each column j that has column i in its neighborhood (j in colInNeighboursLists[i]):
            // - If j is active and already has (desiredLocalActivity - 1) active neighbors,
            //   then activating i would make j's neighborhood have desiredLocalActivity + 1 active
            //   (j itself + its existing active neighbors + i), violating the constraint.
            if (shouldActivate) {
                for (int j : colInNeighboursLists[i]) {
                    if (columnActive_[j].load() == 1) {
                        // Count active neighbors of j (excluding i since i isn't active yet)
                        int activeNeighborsOfJ = 0;
                        for (int jNeighbor : neighbourColsLists[j]) {
                            if (jNeighbor != i && columnActive_[jNeighbor].load() == 1) {
                                activeNeighborsOfJ++;
                            }
                        }
                        // If j already has (desiredLocalActivity - 1) active neighbors,
                        // activating i would push j's neighborhood over the limit
                        // (j + activeNeighborsOfJ + i = 1 + (desiredLocalActivity-1) + 1 = desiredLocalActivity + 1)
                        if (activeNeighborsOfJ >= desiredLocalActivity - 1) {
                            shouldActivate = false;
                            if (this->debug_) {
                                LOG(DEBUG, "    Col " + std::to_string(i) +
                                           " blocked: neighbor " + std::to_string(j) +
                                           " already has " + std::to_string(activeNeighborsOfJ) +
                                           " active neighbors");
                            }
                            break;
                        }
                    }
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

                    bool shouldIncrementInhibition = false;
                    bool shouldDeactivate = false;
                    
                    bool below = (threshold >= 0.0f) && (overlapGrid[nb] < threshold);
                    if (below)
                    {
                        // This neighbor is below threshold, so it should be inhibited
                        shouldIncrementInhibition = true;

                        // If current column has overlap equal to threshold (weakest winner),
                        // permanently inhibit all neighbors below threshold
                        if (overlapGrid[i] == threshold)
                        {
                            inhibitionCounts[nb] = desiredLocalActivity;
                            shouldDeactivate = true;
                            inhibitedCols_[nb].store(1);
                            if (this->debug_) {
                                LOG(DEBUG, "    Permanently inhibited col " + std::to_string(nb) +
                                           " due to weak winner");
                            }
                        }
                        // Otherwise, check if neighbor has reached the desired local activity
                        else if (inhibitionCounts[nb] >= desiredLocalActivity)
                        {
                            shouldDeactivate = true;
                            inhibitedCols_[nb].store(1);
                        }

                        if (this->debug_) {
                            LOG(DEBUG, "    Inhibited col " + std::to_string(nb) +
                                       "  overlapGrid[nb]=" + std::to_string(overlapGrid[nb]) +
                                       "  below threshold=" + std::to_string(threshold));
                        }
                    }
                    else if (threshold >= 0.0f && overlapGrid[i] > overlapGrid[nb])
                    {
                        // This neighbor is above threshold but current column has higher overlap
                        // so this neighbor still gets inhibited (loses the competition)
                        shouldIncrementInhibition = true;
                        
                        if (this->debug_) {
                            LOG(DEBUG, "        Inhibited col " + std::to_string(nb) +
                                       "  overlapGrid[nb]=" + std::to_string(overlapGrid[nb]) +
                                       "  lost competition to col " + std::to_string(i) +
                                       " with overlap=" + std::to_string(overlapGrid[i]));
                        }
                    }

                    // Increment inhibition count if this neighbor should be inhibited
                    if (shouldIncrementInhibition)
                    {
                        ++inhibitionCounts[nb];
                        
                        if (this->debug_) {
                            LOG(DEBUG, "    Incremented inhibition count for col " + std::to_string(nb) +
                                       "  newInhibCnt=" + std::to_string(inhibitionCounts[nb]));
                        }

                        // Queue again if still below the limit
                        if (inhibitionCounts[nb] < desiredLocalActivity)
                        {
                            std::lock_guard<std::mutex> lk(minorlyInhibitedMutex);
                            minorlyInhibitedColumns.push_back(nb);
                            needsReprocessing = true;
                        }

                        // Deactivate if it had been active and should be deactivated
                        if (shouldDeactivate && columnActive_[nb].load())
                        {
                            columnActive_[nb].store(0);
                            std::lock_guard<std::mutex> lk(activeColumnsMutex);
                            activeColumnsInd.erase(
                                std::remove(activeColumnsInd.begin(),
                                            activeColumnsInd.end(), nb),
                                activeColumnsInd.end());
                            if (this->debug_) {
                                LOG(DEBUG, "    Deactivated col " + std::to_string(nb) +
                                           " and removed from active columns");
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
            if (this->debug_) {
                LOG(DEBUG, "prepare_next desiredLocalActivity="
                        + std::to_string(desiredLocalActivity)); 
            }

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
                LOG(DEBUG, "iterationCount=" + std::to_string(iterationCount));
                LOG(DEBUG, "keepGoing=" + std::to_string(keepGoing));
            }
            
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

        // Task to add a small tie breaker to the overlap grid (if enabled)
        if (useTieBreaker_) {
            add_tie_breaker(const_cast<std::vector<float>&>(colOverlapGrid), tf1);
        }

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

        // Add a small tie breaker to the potential overlap grid (if enabled)
        if (useTieBreaker_) {
            add_tie_breaker(const_cast<std::vector<float>&>(potColOverlapGrid), tf4);
        }

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
        tf::Task f2_task = taskflow.composed_of(tf2).name("SortOverlap");
        tf::Task f3_task = taskflow.composed_of(tf3).name("ProcessOverlap");
        tf::Task f5_task = taskflow.composed_of(tf5).name("SortPotOverlap");
        tf::Task f6_task = taskflow.composed_of(tf6).name("ProcessPotOverlap");

        // Set the task dependencies
        if (useTieBreaker_) {
            tf::Task f1_task = taskflow.composed_of(tf1).name("AddTieBreaker");
            tf::Task f4_task = taskflow.composed_of(tf4).name("AddTieBreakerPot");
            f1_task.precede(f2_task);
            f2_task.precede(f3_task);
            f3_task.precede(f4_task);
            f4_task.precede(f5_task);
            f5_task.precede(f6_task);
        } else {
            f2_task.precede(f3_task);
            f3_task.precede(f5_task);
            f5_task.precede(f6_task);
        }

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

                // Count active neighbors of column i
                for (int neighborIndex : neighbourCols)
                {
                    if (neighborIndex >= 0 && columnActive[neighborIndex] == 1)
                    {
                        numActiveNeighbours++;
                    }
                }

                numColsActInNeigh[i] = numActiveNeighbours;

                // Check if activating column i would violate the constraint for ANY neighborhood
                // that includes column i. A neighborhood centered on position j includes column i
                // if i is in neighbourColsLists[j], which means j is in colInNeighboursLists[i].
                // 
                // For each such position j, count active columns in j's neighborhood.
                // If count >= desiredLocalActivity, activating i would violate the constraint.
                bool wouldViolateConstraint = false;
                
                // Check i's own neighborhood first
                if (numActiveNeighbours >= desiredLocalActivity) {
                    wouldViolateConstraint = true;
                }
                
                // Check all positions that have i in their neighborhood
                if (!wouldViolateConstraint) {
                    for (int j : colInNeighboursLists[i]) {
                        // Count active columns in j's neighborhood (excluding i since i isn't active yet)
                        int activeInJsNeighborhood = 0;
                        for (int k : neighbourColsLists[j]) {
                            if (k != i && columnActive[k] == 1) {
                                activeInJsNeighborhood++;
                            }
                        }
                        // If j's neighborhood already has desiredLocalActivity active columns,
                        // activating i would make it exceed the limit
                        if (activeInJsNeighborhood >= desiredLocalActivity) {
                            wouldViolateConstraint = true;
                            break;
                        }
                    }
                }

                // Activate column if it won't violate the constraint
                if (!wouldViolateConstraint && numColsActInNeigh[i] < desiredLocalActivity)
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

    void InhibitionCalculator::add_tie_breaker(std::vector<float>& overlapGrid, tf::Taskflow &taskflow)
    {
        // Create a task in the taskflow for adding the tie breaker
        taskflow.emplace([this, &overlapGrid]() {
            // Validate that the overlap grid size matches the tie-breaker vector size
            if (overlapGrid.size() != tieBreaker_.size()) {
                std::string errorMsg = std::string("add_tie_breaker: Size mismatch! ") +
                                     "overlapGrid size: " + std::to_string(overlapGrid.size()) + 
                                     ", tieBreaker size: " + std::to_string(tieBreaker_.size()) + 
                                     ", expected numColumns: " + std::to_string(numColumns_);
                LOG(ERROR, errorMsg);
                throw std::invalid_argument(errorMsg);
            }
            
            if (debug_) {
                LOG(DEBUG, "add_tie_breaker: Processing " + std::to_string(overlapGrid.size()) + " elements");
            }
            
            // Add the pre-computed tie breaker values to the overlap grid
            for (size_t i = 0; i < overlapGrid.size(); ++i)
            {
                overlapGrid[i] += tieBreaker_[i];
            }
        }).name("AddTieBreaker");
    }

    void InhibitionCalculator::parallel_add_tie_breaker(std::vector<float>& overlapGrid, tf::Taskflow &taskflow)
    {
        // Validate that the overlap grid size matches the tie-breaker vector size
        if (overlapGrid.size() != tieBreaker_.size()) {
            std::string errorMsg = std::string("parallel_add_tie_breaker: Size mismatch! ") +
                                 "overlapGrid size: " + std::to_string(overlapGrid.size()) + 
                                 ", tieBreaker size: " + std::to_string(tieBreaker_.size()) + 
                                 ", expected numColumns: " + std::to_string(numColumns_);
            LOG(ERROR, errorMsg);
            throw std::invalid_argument(errorMsg);
        }
        
        if (debug_) {
            LOG(DEBUG, "parallel_add_tie_breaker: Processing " + std::to_string(overlapGrid.size()) + " elements");
        }

        // Create a task in the taskflow for adding the tie breaker in parallel
        taskflow.for_each_index(0, static_cast<int>(overlapGrid.size()), 1, [this, &overlapGrid](int i)
        {
            // Add the pre-computed tie breaker values to the overlap grid
            overlapGrid[i] += tieBreaker_[i];
        }).name("ParallelAddTieBreaker");
    }

} // namespace inhibition
