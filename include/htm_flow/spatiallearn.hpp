#pragma once

#include <iostream>
#include <vector>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <utility>
#include <cstddef>

namespace spatiallearn
{

    ///-----------------------------------------------------------------------------
    ///
    /// SpatialLearnCalculator
    /// A calculator class used to increase or decrease the permanence values of the potential synapses in a single HTM layer.
    /// This class uses standard std::vector arrays and is a CPU implementation.
    ///
    ///-----------------------------------------------------------------------------
    class SpatialLearnCalculator
    {
    public:
        ///-----------------------------------------------------------------------------
        ///
        /// SpatialLearnCalculator   Constructor for the SpatialLearnCalculator.
        ///
        /// @param[in] numColumns                Number of columns in the HTM layer.
        /// @param[in] numPotSynapses            Number of potential synapses per column.
        /// @param[in] spatialPermanenceInc      Amount to increment synapse permanence values.
        /// @param[in] spatialPermanenceDec      Amount to decrement synapse permanence values.
        /// @param[in] activeColPermanenceDec    Amount to decrement synapse permanence values for active columns.
        ///
        ///-----------------------------------------------------------------------------
        SpatialLearnCalculator(int numColumns,
                               int numPotSynapses,
                               float spatialPermanenceInc,
                               float spatialPermanenceDec,
                               float activeColPermanenceDec);

        ///-----------------------------------------------------------------------------
        ///
        /// calculate_spatiallearn   Performs the spatial learning calculation by updating the synapse permanence values.
        ///
        /// @param[in,out] colSynPerm        Matrix of synapse permanence values for each column.
        /// @param[in]     colPotInputs      Matrix of potential synapse inputs for each column.
        /// @param[in]     activeCols        Vector indicating active columns (1 for active, 0 for inactive).
        /// @param[in]     activeColIndices  Vector of indices for active columns.
        ///
        ///-----------------------------------------------------------------------------
        void calculate_spatiallearn(
            std::vector<std::vector<float>> &colSynPerm,
            const std::vector<std::vector<int>> &colPotInputs,
            const std::vector<int> &activeCols,
            const std::vector<int> &activeColIndices);

        ///-----------------------------------------------------------------------------
        ///
        /// calculate_spatiallearn_1d
        ///
        /// 1D version of spatial learning for pipeline integration.
        ///
        /// - `colSynPerm` is a 1D vector simulating a 2D matrix (numColumns x numPotSynapses)
        /// - `colPotInputs` is a 1D vector simulating a 2D matrix (numColumns x numPotSynapses)
        ///   (e.g. directly returned by `overlap::OverlapCalculator::get_col_pot_inputs()`).
        ///
        /// Only columns listed in `activeColIndices` are processed (in parallel).
        ///
        /// @param[in,out] colSynPerm          Flattened permanence matrix (numColumns x numPotSynapses)
        /// @param[in]     colSynPerm_shape    Shape of colSynPerm (numColumns, numPotSynapses)
        /// @param[in]     colPotInputs        Flattened potential input matrix (numColumns x numPotSynapses)
        /// @param[in]     colPotInputs_shape  Shape of colPotInputs (numColumns, numPotSynapses)
        /// @param[in]     activeCols          Active mask per column (size numColumns; 1 for active, 0 for inactive)
        /// @param[in]     activeColIndices    Indices of active columns (subset of [0, numColumns))
        ///
        ///-----------------------------------------------------------------------------
        void calculate_spatiallearn_1d(
            std::vector<float> &colSynPerm,
            const std::pair<int, int> &colSynPerm_shape,
            const std::vector<int> &colPotInputs,
            const std::pair<int, int> &colPotInputs_shape,
            const std::vector<int> &activeCols,
            const std::vector<int> &activeColIndices);

        ///-----------------------------------------------------------------------------
        ///
        /// calculate_spatiallearn_1d_active_indices
        ///
        /// Like `calculate_spatiallearn_1d`, but does NOT require the full `activeCols`
        /// mask. Instead, only `activeColIndices` is required, and the calculator tracks
        /// previous active indices internally to detect "newly active" columns without
        /// scanning all columns.
        ///
        /// This is the preferred API for pipeline usage where inhibition already
        /// produces the active index list.
        ///
        /// @param[in,out] colSynPerm          Flattened permanence matrix (numColumns x numPotSynapses)
        /// @param[in]     colSynPerm_shape    Shape of colSynPerm (numColumns, numPotSynapses)
        /// @param[in]     colPotInputs        Flattened potential input matrix (numColumns x numPotSynapses)
        /// @param[in]     colPotInputs_shape  Shape of colPotInputs (numColumns, numPotSynapses)
        /// @param[in]     activeColIndices    Indices of active columns
        ///
        ///-----------------------------------------------------------------------------
        void calculate_spatiallearn_1d_active_indices(
            std::vector<float> &colSynPerm,
            const std::pair<int, int> &colSynPerm_shape,
            const std::vector<int> &colPotInputs,
            const std::pair<int, int> &colPotInputs_shape,
            const std::vector<int> &activeColIndices);

    private:
        int numColumns_;                  // Number of columns in the HTM layer
        int numPotSynapses_;              // Number of potential synapses per column
        float spatialPermanenceInc_;      // Amount to increment synapse permanence values
        float spatialPermanenceDec_;      // Amount to decrement synapse permanence values
        float activeColPermanenceDec_;    // Amount to decrement synapse permanence values for active columns

        // Previous inputs to the potential synapses (flattened: numColumns x numPotSynapses).
        // Stored to detect when an already-active column's input patch changes.
        std::vector<int> prevColPotInputs_;
        std::vector<int> prevActiveCols_;                // Previous active columns (0/1-ish; non-positive treated as inactive)
        std::vector<int> prevActiveIndices_;             // Previous active column indices (for O(k) updates without scanning all columns)
    };

} // namespace spatiallearn
