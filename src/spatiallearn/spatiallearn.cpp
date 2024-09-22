#include "spatiallearn.hpp"
#include <algorithm>
// log
#include <utilities/logger.hpp>

namespace spatiallearn
{

    SpatialLearnCalculator::SpatialLearnCalculator(int numColumns,
                                                   int numPotSynapses,
                                                   float spatialPermanenceInc,
                                                   float spatialPermanenceDec,
                                                   float activeColPermanenceDec)
        : numColumns_(numColumns),
          numPotSynapses_(numPotSynapses),
          spatialPermanenceInc_(spatialPermanenceInc),
          spatialPermanenceDec_(spatialPermanenceDec),
          activeColPermanenceDec_(activeColPermanenceDec),
          prevColPotInputs_(numColumns, std::vector<int>(numPotSynapses, -1)),
          prevActiveCols_(numColumns, -1)
    {
    }

    void SpatialLearnCalculator::calculate_spatiallearn(
        std::vector<std::vector<float>> &colSynPerm,
        const std::vector<std::vector<int>> &colPotInputs,
        const std::vector<int> &activeCols,
        const std::vector<int> &activeColIndices)
    {
        tf::Taskflow taskflow;
        tf::Executor executor;

        // Prepare local copies of member variables
        std::vector<std::vector<int>> prevColPotInputs = prevColPotInputs_;
        std::vector<int> prevActiveCols = prevActiveCols_;
        float spatialPermanenceInc = spatialPermanenceInc_;
        float spatialPermanenceDec = spatialPermanenceDec_;
        float activeColPermanenceDec = activeColPermanenceDec_;

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
                                      if (colPotInputs[c][s] == 1)
                                      {
                                          colSynPerm[c][s] += spatialPermanenceInc;
                                          colSynPerm[c][s] = std::min(1.0f, colSynPerm[c][s]);
                                      }
                                      else
                                      {
                                          colSynPerm[c][s] -= spatialPermanenceDec;
                                          colSynPerm[c][s] = std::max(0.0f, colSynPerm[c][s]);
                                      }
                                  }
                              }
                              else
                              {
                                  // Column was previously active
                                  if (prevColPotInputs[c] != colPotInputs[c])
                                  {
                                      for (int s = 0; s < colSynPerm[c].size(); ++s)
                                      {
                                          if (colPotInputs[c][s] == 1)
                                          {
                                              colSynPerm[c][s] += spatialPermanenceInc;
                                              colSynPerm[c][s] = std::min(1.0f, colSynPerm[c][s]);
                                          }
                                          else
                                          {
                                              colSynPerm[c][s] -= activeColPermanenceDec;
                                              colSynPerm[c][s] = std::max(0.0f, colSynPerm[c][s]);
                                          }
                                      }
                                  }
                              }
                          });

        // Run the taskflow
        executor.run(taskflow).wait();

        // Update previous inputs and active columns
        prevColPotInputs_ = colPotInputs;
        prevActiveCols_ = activeCols;

        LOG(INFO, "SpatialLearnCalculator calculate_spatiallearn Done.");
    }

} // namespace spatiallearn
