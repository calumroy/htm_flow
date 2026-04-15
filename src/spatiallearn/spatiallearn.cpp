#include <htm_flow/spatiallearn.hpp>
#include <algorithm>
// log
#include <utilities/logger.hpp>
#include <cassert>
#include <numeric>

namespace spatiallearn
{

    ///-----------------------------------------------------------------------------
    ///
    /// SpatialLearnCalculator   Construct the spatial learning calculator and
    /// initialize cached state used between learning steps.
    ///
    /// Why this is needed:
    /// The calculator keeps previous active columns and input patches so later
    /// updates can avoid unnecessary relearning work.
    ///-----------------------------------------------------------------------------
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

    ///-----------------------------------------------------------------------------
    ///
    /// calculate_spatiallearn   Update spatial-learning permanences using the
    /// legacy 2D input format.
    ///
    /// Why this is needed:
    /// Preserves the older API while routing all learning through the newer 1D
    /// implementation so behavior stays consistent in one place.
    ///
    /// This function performs the following steps:
    /// 1. Validate the 2D input sizes against the configured layer shape.
    /// 2. Flatten the 2D permanence and input grids into 1D buffers.
    /// 3. Delegate learning to the shared 1D implementation.
    /// 4. Copy the updated permanences back into the 2D output structure.
    ///-----------------------------------------------------------------------------
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

    ///-----------------------------------------------------------------------------
    ///
    /// calculate_spatiallearn_1d   Update spatial-learning permanences using 1D
    /// buffers plus both active-column forms.
    ///
    /// Why this is needed:
    /// Keeps compatibility with callers that still provide an active mask while
    /// reusing the more direct active-index implementation underneath.
    ///
    /// This function performs the following steps:
    /// 1. Validate that the active mask size matches the configured column count.
    /// 2. Sanity-check that each provided active index is marked active in the mask.
    /// 3. Delegate the learning update to the indices-only 1D implementation.
    ///-----------------------------------------------------------------------------
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

    ///-----------------------------------------------------------------------------
    ///
    /// calculate_spatiallearn_1d_active_indices   Update synapse permanences for
    /// the active columns in the current step.
    ///
    /// Why this is needed:
    /// Reinforces synapses connected to active inputs and weakens the rest so
    /// each winning column gradually specializes on patterns it should represent.
    ///
    /// This function performs the following steps:
    /// 1. Validate the flattened input shapes and cached state buffers.
    /// 2. Iterate over the active columns in parallel.
    /// 3. For newly active columns, apply the standard permanence update.
    /// 4. For previously active columns, relearn only if the input patch changed.
    /// 5. Cache the latest active inputs and active-column state for the next step.
    ///-----------------------------------------------------------------------------
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

    }

} // namespace spatiallearn
