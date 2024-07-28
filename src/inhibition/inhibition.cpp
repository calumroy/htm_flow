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
        : width_(width), height_(height), potentialWidth_(potentialInhibWidth), potentialHeight_(potentialInhibHeight),
          desiredLocalActivity_(desiredLocalActivity), minOverlap_(minOverlap), centerInhib_(centerInhib),
          activeColumns_(), columnActive_(width * height, 0), inhibitedCols_(width * height, 0), numColsActInNeigh_(width * height, 0)
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

    void InhibitionCalculator::calculate_inhibition(const std::vector<int>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                    const std::vector<int>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape)
    {
        tf::Executor executor;
        tf::Taskflow taskflow;

        // Add a tie breaker to the overlap grid
        taskflow.emplace([this, &colOverlapGrid, colOverlapGridShape]() {
            add_tie_breaker(const_cast<std::vector<int>&>(colOverlapGrid), false);
        }).name("AddTieBreaker");

        // Calculate inhibition for each column
        taskflow.for_each_index(0, colOverlapGrid.size(), 1, [this, &colOverlapGrid](int i) {
            calculate_inhibition_for_column(i, colOverlapGrid[i]);
        }).name("CalculateInhibition");

        // Execute the taskflow
        executor.run(taskflow).wait();

        // Repeat the process with potential overlaps if necessary
        taskflow.emplace([this, &potColOverlapGrid, potColOverlapGridShape]() {
            add_tie_breaker(const_cast<std::vector<int>&>(potColOverlapGrid), false);
        }).name("AddTieBreakerPot");

        taskflow.for_each_index(0, potColOverlapGrid.size(), 1, [this, &potColOverlapGrid](int i) {
            if (inhibitedCols_[i] == 0 && columnActive_[i] == 0)
            {
                calculate_inhibition_for_column(i, potColOverlapGrid[i]);
            }
        }).name("CalculateInhibitionPot");

        executor.run(taskflow).wait();
    }

    std::vector<int> InhibitionCalculator::get_active_columns() const
    {
        return columnActive_;
    }

    void InhibitionCalculator::add_tie_breaker(std::vector<int>& overlapGrid, bool addColBias)
    {
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


    void InhibitionCalculator::calculate_inhibition_for_column(int colIndex, int overlapScore)
    {
        if (inhibitedCols_[colIndex] != 1)
        {
            std::vector<int> neighbourCols = neighbourColsLists_[colIndex];
            int numActiveNeighbours = 0;

            // Check neighbours for active columns
            for (int neighborIndex : neighbourCols)
            {
                if (neighborIndex >= 0 && columnActive_[neighborIndex] == 1)
                {
                    numActiveNeighbours++;
                    if (numColsActInNeigh_[neighborIndex] >= desiredLocalActivity_)
                    {
                        inhibitedCols_[colIndex] = 1;
                    }
                }
            }

            // Check if the column is in any neighbour lists of active columns
            for (int activeNeighbor : colInNeighboursLists_[colIndex])
            {
                if (columnActive_[activeNeighbor] == 1)
                {
                    if (numColsActInNeigh_[activeNeighbor] >= desiredLocalActivity_)
                    {
                        inhibitedCols_[colIndex] = 1;
                    }
                }
            }

            numColsActInNeigh_[colIndex] = numActiveNeighbours;

            // Activate column if not inhibited and the number of active neighbors is less than desired local activity
            if (inhibitedCols_[colIndex] != 1 && numColsActInNeigh_[colIndex] < desiredLocalActivity_)
            {
                activeColumns_.push_back(colIndex);
                columnActive_[colIndex] = 1;
                for (int c : colInNeighboursLists_[colIndex])
                {
                    if (c >= 0)
                    {
                        numColsActInNeigh_[c]++;
                    }
                }
            }
            else
            {
                inhibitedCols_[colIndex] = 1;
            }
        }
    }

    // Function: calculateWinningCols
    // This function determines which columns become active after applying the inhibition process.
    //
    // Inputs:
    // - colOverlapGrid: A vector representing the overlap values for each column in a 2D grid (flattened).
    // - colOverlapGridShape: The shape of the colOverlapGrid (rows, cols).
    // - potColOverlapGrid: A vector representing the potential overlap values for each column in a 2D grid (flattened).
    // - potColOverlapGridShape: The shape of the potColOverlapGrid (rows, cols).
    //
    // Outputs:
    // - A vector representing the active state of each column (1 for active, 0 for inactive).
    //
    // Function:
    // 1. Add a tie-breaker to the overlaps grid based on position and previous activity.
    // 2. Sort all columns by their overlap values and process from highest to lowest.
    // 3. Determine if each column should be active based on local inhibition criteria.
    // 4. Apply the same process to the potential overlaps if necessary.
    //
    void InhibitionCalculator::calculateWinningCols(const std::vector<int>& colOverlapGrid, const std::pair<int, int>& colOverlapGridShape,
                                                const std::vector<int>& potColOverlapGrid, const std::pair<int, int>& potColOverlapGridShape)
    {
        // Ensure the overlap grid dimensions match the expected size
        assert(colOverlapGrid.size() == colOverlapGridShape.first * colOverlapGridShape.second);
        assert(potColOverlapGrid.size() == potColOverlapGridShape.first * potColOverlapGridShape.second);

        // Initialize the active column array and inhibited columns array
        columnActive_.assign(colOverlapGrid.size(), 0);
        inhibitedCols_.assign(colOverlapGrid.size(), 0);
        numColsActInNeigh_.assign(colOverlapGrid.size(), 0);
        activeColumns_.clear();

        // Add tie-breaker to overlap grid
        add_tie_breaker(const_cast<std::vector<int>&>(colOverlapGrid), true);

        // Sort the columns by overlap values in descending order using parallel sorting
        std::vector<int> sortedIndices(colOverlapGrid.size());
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

        tf::Taskflow taskflow;
        tf::Executor executor;

        auto sort_task = taskflow.sort(sortedIndices.begin(), sortedIndices.end(), [&colOverlapGrid](int a, int b) {
            return colOverlapGrid[a] > colOverlapGrid[b];
        });

        // Execute the parallel sort
        executor.run(taskflow).wait();

        // Process columns from highest to lowest overlap
        for (int i : sortedIndices)
        {
            if (colOverlapGrid[i] >= minOverlap_)
            {
                calculate_inhibition_for_column(i, colOverlapGrid[i]);
            }
        }

        // Process potential overlaps for columns that are not already active
        add_tie_breaker(const_cast<std::vector<int>&>(potColOverlapGrid), false);

        auto sort_pot_task = taskflow.sort(sortedIndices.begin(), sortedIndices.end(), [&potColOverlapGrid](int a, int b) {
            return potColOverlapGrid[a] > potColOverlapGrid[b];
        });

        // Execute the parallel sort for potential overlaps
        executor.run(taskflow).wait();

        for (int i : sortedIndices)
        {
            if (inhibitedCols_[i] == 0 && columnActive_[i] == 0 && potColOverlapGrid[i] >= minOverlap_)
            {
                calculate_inhibition_for_column(i, potColOverlapGrid[i]);
            }
        }

        // Save the active column state for potential use in future time steps
        prevActiveColsGrid_ = columnActive_;
    }


} // namespace inhibition
