#include "spatiallearn.hpp"
#include <algorithm>
#include "utilities/logger.hpp"

namespace spatiallearn
{

    void SpatialLearnCalculator::updatePermanence(int c, int s,
                                                  const std::vector<std::vector<int>> &colPotInputs,
                                                  std::vector<std::vector<float>> &colSynPerm,
                                                  float incPerm, float decPerm)
    {
        if (colPotInputs[c][s] == 1)
        {
            colSynPerm[c][s] += incPerm;
            colSynPerm[c][s] = std::min(1.0f, colSynPerm[c][s]);
        }
        else
        {
            colSynPerm[c][s] -= decPerm;
            colSynPerm[c][s] = std::max(0.0f, colSynPerm[c][s]);
        }
    }

    void SpatialLearnCalculator::updatePermanenceValues(
        std::vector<std::vector<float>> &colSynPerm,
        const std::vector<std::vector<int>> &colPotInputs,
        const std::vector<int> &activeCols,
        std::vector<std::vector<int>> &prevColPotInputs,
        std::vector<int> &prevActiveCols,
        float spatialPermanenceInc,
        float spatialPermanenceDec,
        float activeColPermanenceDec)
    {
        tf::Taskflow taskflow;
        tf::Executor executor;

        // Create a vector of active column indices
        std::vector<int> activeColIndices;
        for (int c = 0; c < activeCols.size(); ++c)
        {
            if (activeCols[c] == 1)
            {
                activeColIndices.push_back(c);
            }
        }

        // Process each active column in parallel
        taskflow.for_each(activeColIndices.begin(), activeColIndices.end(),
                          [&prevActiveCols, &activeCols, &colSynPerm, &colPotInputs,
                           &prevColPotInputs, spatialPermanenceInc, spatialPermanenceDec, activeColPermanenceDec](int c)
                          {
                              if (prevActiveCols[c] != activeCols[c])
                              {
                                  // Column was newly activated
                                  for (int s = 0; s < colSynPerm[c].size(); ++s)
                                  {
                                      updatePermanence(c, s, colPotInputs, colSynPerm,
                                                       spatialPermanenceInc, spatialPermanenceDec);
                                  }
                              }
                              else
                              {
                                  // Column was previously active
                                  if (prevColPotInputs[c] != colPotInputs[c])
                                  {
                                      for (int s = 0; s < colSynPerm[c].size(); ++s)
                                      {
                                          updatePermanence(c, s, colPotInputs, colSynPerm,
                                                           spatialPermanenceInc, activeColPermanenceDec);
                                      }
                                  }
                              }
                          });

        // Run the taskflow
        executor.run(taskflow).wait();

        // Update previous inputs and active columns
        prevColPotInputs = colPotInputs;
        prevActiveCols = activeCols;
    }

    void SpatialLearnCalculator::calculate_spatiallearn(
        std::vector<std::vector<float>> &colSynPerm,
        const std::vector<std::vector<int>> &colPotInputs,
        const std::vector<int> &activeCols)
    {
        tf::Taskflow taskflow;
        tf::Executor executor;

        tf::Taskflow tf1;

        // Task to update permanence values
        tf1.emplace([&]()
                    {
                        updatePermanenceValues(colSynPerm, colPotInputs, activeCols,
                                               prevColPotInputs_, prevActiveCols_,
                                               spatialPermanenceInc_, spatialPermanenceDec_, activeColPermanenceDec_);
                    })
            .name("updatePermanenceValues");

        // Add the sub-taskflow to the main taskflow
        tf::Task f1_task = taskflow.composed_of(tf1).name("updatePermanenceValues");

        // Run the taskflow
        executor.run(taskflow).wait();

        LOG(INFO, "SpatialLearnCalculator calculate_spatiallearn Done.");
    }

} // namespace spatiallearn