#include "spatiallearn.hpp"
#include <algorithm>
// log
#include <utilities/logger.hpp>
#include <cassert>
#include <numeric>

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
          prevColPotInputs_(numColumns * numPotSynapses, -1),
          prevActiveCols_(numColumns, 0),
          prevActiveIndices_()
    {
    }

    void SpatialLearnCalculator::calculate_spatiallearn(
        std::vector<std::vector<float>> &colSynPerm,
        const std::vector<std::vector<int>> &colPotInputs,
        const std::vector<int> &activeCols,
        const std::vector<int> &activeColIndices)
    {
        // Legacy 2D API wrapper. Convert to 1D and delegate to calculate_spatiallearn_1d.
        assert(static_cast<int>(colSynPerm.size()) == numColumns_);
        assert(static_cast<int>(colPotInputs.size()) == numColumns_);

        std::vector<float> colSynPerm1D(static_cast<size_t>(numColumns_ * numPotSynapses_), 0.0f);
        std::vector<int> colPotInputs1D(static_cast<size_t>(numColumns_ * numPotSynapses_), 0);

        for (int c = 0; c < numColumns_; ++c)
        {
            assert(static_cast<int>(colSynPerm[c].size()) == numPotSynapses_);
            assert(static_cast<int>(colPotInputs[c].size()) == numPotSynapses_);
            const int base = c * numPotSynapses_;
            for (int s = 0; s < numPotSynapses_; ++s)
            {
                colSynPerm1D[static_cast<size_t>(base + s)] = colSynPerm[c][static_cast<size_t>(s)];
                colPotInputs1D[static_cast<size_t>(base + s)] = colPotInputs[c][static_cast<size_t>(s)];
            }
        }

        calculate_spatiallearn_1d(
            colSynPerm1D,
            std::make_pair(numColumns_, numPotSynapses_),
            colPotInputs1D,
            std::make_pair(numColumns_, numPotSynapses_),
            activeCols,
            activeColIndices);

        // Copy updated permanences back to 2D structure
        for (int c = 0; c < numColumns_; ++c)
        {
            const int base = c * numPotSynapses_;
            for (int s = 0; s < numPotSynapses_; ++s)
            {
                colSynPerm[c][static_cast<size_t>(s)] = colSynPerm1D[static_cast<size_t>(base + s)];
            }
        }
    }

    void SpatialLearnCalculator::calculate_spatiallearn_1d(
        std::vector<float> &colSynPerm,
        const std::pair<int, int> &colSynPerm_shape,
        const std::vector<int> &colPotInputs,
        const std::pair<int, int> &colPotInputs_shape,
        const std::vector<int> &activeCols,
        const std::vector<int> &activeColIndices)
    {
        // Keep the old API, but delegate to the indices-only version when possible.
        // We still sanity-check the active mask matches the provided indices.
        assert(static_cast<int>(activeCols.size()) == numColumns_);
        for (int c : activeColIndices) {
            if (c >= 0 && c < numColumns_) {
                assert(activeCols[c] > 0);
            }
        }

        calculate_spatiallearn_1d_active_indices(
            colSynPerm,
            colSynPerm_shape,
            colPotInputs,
            colPotInputs_shape,
            activeColIndices);
    }

    void SpatialLearnCalculator::calculate_spatiallearn_1d_active_indices(
        std::vector<float> &colSynPerm,
        const std::pair<int, int> &colSynPerm_shape,
        const std::vector<int> &colPotInputs,
        const std::pair<int, int> &colPotInputs_shape,
        const std::vector<int> &activeColIndices)
    {
        // Shape / bounds checks (cheap and catches bad pipeline wiring)
        assert(colSynPerm_shape.first == numColumns_);
        assert(colSynPerm_shape.second == numPotSynapses_);
        assert(colPotInputs_shape.first == numColumns_);
        assert(colPotInputs_shape.second == numPotSynapses_);
        assert(static_cast<int>(colSynPerm.size()) == numColumns_ * numPotSynapses_);
        assert(static_cast<int>(colPotInputs.size()) == numColumns_ * numPotSynapses_);
        assert(static_cast<int>(prevActiveCols_.size()) == numColumns_);
        assert(static_cast<int>(prevColPotInputs_.size()) == numColumns_ * numPotSynapses_);

        // We assume the producer (inhibition) provides a valid, duplicate-free list.
        // InhibitionCalculator now validates this invariant after each calculation.

        tf::Taskflow taskflow;
        tf::Executor executor;

        const float spatialPermanenceInc = spatialPermanenceInc_;
        const float spatialPermanenceDec = spatialPermanenceDec_;
        const float activeColPermanenceDec = activeColPermanenceDec_;

        // Read-only access to previous state during parallel region.
        const std::vector<int>& prevActiveCols = prevActiveCols_;
        const std::vector<int>& prevColPotInputs = prevColPotInputs_;

        taskflow.for_each(activeColIndices.begin(), activeColIndices.end(),
                          [&](int c)
                          {
                              if (c < 0 || c >= numColumns_) {
                                  return;
                              }

                              const int base = c * numPotSynapses_;
                              const bool was_active = (prevActiveCols[c] > 0);

                              if (!was_active)
                              {
                                  // Newly active
                                  for (int s = 0; s < numPotSynapses_; ++s)
                                  {
                                      const int pot = colPotInputs[static_cast<size_t>(base + s)];
                                      float &perm = colSynPerm[static_cast<size_t>(base + s)];

                                      if (pot == 1)
                                      {
                                          perm = std::min(1.0f, perm + spatialPermanenceInc);
                                      }
                                      else
                                      {
                                          perm = std::max(0.0f, perm - spatialPermanenceDec);
                                      }
                                  }
                                  return;
                              }

                              // Previously active: only update if the input patch changed.
                              bool inputs_changed = false;
                              for (int s = 0; s < numPotSynapses_; ++s)
                              {
                                  if (prevColPotInputs[static_cast<size_t>(base + s)] != colPotInputs[static_cast<size_t>(base + s)])
                                  {
                                      inputs_changed = true;
                                      break;
                                  }
                              }

                              if (!inputs_changed)
                              {
                                  return;
                              }

                              for (int s = 0; s < numPotSynapses_; ++s)
                              {
                                  const int pot = colPotInputs[static_cast<size_t>(base + s)];
                                  float &perm = colSynPerm[static_cast<size_t>(base + s)];

                                  if (pot == 1)
                                  {
                                      perm = std::min(1.0f, perm + spatialPermanenceInc);
                                  }
                                  else
                                  {
                                      perm = std::max(0.0f, perm - activeColPermanenceDec);
                                  }
                              }
                          });

        executor.run(taskflow).wait();

        // Update previous pot-inputs only for currently-active columns.
        for (int c : activeColIndices)
        {
            if (c < 0 || c >= numColumns_) {
                continue;
            }
            const int base = c * numPotSynapses_;
            for (int s = 0; s < numPotSynapses_; ++s)
            {
                prevColPotInputs_[static_cast<size_t>(base + s)] = colPotInputs[static_cast<size_t>(base + s)];
            }
        }

        // Update active state without scanning all columns:
        // - clear previously active columns
        // - set currently active columns
        for (int c : prevActiveIndices_)
        {
            if (c >= 0 && c < numColumns_)
            {
                prevActiveCols_[c] = 0;
            }
        }
        for (int c : activeColIndices)
        {
            if (c >= 0 && c < numColumns_)
            {
                prevActiveCols_[c] = 1;
            }
        }
        prevActiveIndices_ = activeColIndices;

        LOG(INFO, "SpatialLearnCalculator calculate_spatiallearn_1d_active_indices Done.");
    }

} // namespace spatiallearn
