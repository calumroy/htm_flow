#pragma once

#include <iostream>
#include <vector>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

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
        /// @param[in,out] colSynPerm    Matrix of synapse permanence values for each column.
        /// @param[in]     colPotInputs  Matrix of potential synapse inputs for each column.
        /// @param[in]     activeCols    Vector indicating active columns (1 for active, 0 for inactive).
        ///
        ///-----------------------------------------------------------------------------
        void calculate_spatiallearn(
            std::vector<std::vector<float>> &colSynPerm,
            const std::vector<std::vector<int>> &colPotInputs,
            const std::vector<int> &activeCols);

    private:
        ///-----------------------------------------------------------------------------
        ///
        /// updatePermanence   Updates the permanence value for a specific synapse.
        ///
        /// @param[in]     c             Column index.
        /// @param[in]     s             Synapse index within the column.
        /// @param[in]     colPotInputs  Matrix of potential synapse inputs for each column.
        /// @param[in,out] colSynPerm    Matrix of synapse permanence values for each column.
        /// @param[in]     incPerm       Amount to increment the permanence.
        /// @param[in]     decPerm       Amount to decrement the permanence.
        ///
        ///-----------------------------------------------------------------------------
        static void updatePermanence(int c, int s,
                                     const std::vector<std::vector<int>> &colPotInputs,
                                     std::vector<std::vector<float>> &colSynPerm,
                                     float incPerm, float decPerm);

        ///-----------------------------------------------------------------------------
        ///
        /// updatePermanenceValues   Updates the permanence values for all synapses based on the current inputs and active columns.
        ///
        /// @param[in,out] colSynPerm               Matrix of synapse permanence values for each column.
        /// @param[in]     colPotInputs             Matrix of potential synapse inputs for each column.
        /// @param[in]     activeCols               Vector indicating active columns (1 for active, 0 for inactive).
        /// @param[in,out] prevColPotInputs         Previous inputs to the potential synapses.
        /// @param[in,out] prevActiveCols           Previous active columns.
        /// @param[in]     spatialPermanenceInc     Amount to increment synapse permanence values.
        /// @param[in]     spatialPermanenceDec     Amount to decrement synapse permanence values.
        /// @param[in]     activeColPermanenceDec   Amount to decrement synapse permanence values for active columns.
        ///
        ///-----------------------------------------------------------------------------
        static void updatePermanenceValues(
            std::vector<std::vector<float>> &colSynPerm,
            const std::vector<std::vector<int>> &colPotInputs,
            const std::vector<int> &activeCols,
            std::vector<std::vector<int>> &prevColPotInputs,
            std::vector<int> &prevActiveCols,
            float spatialPermanenceInc,
            float spatialPermanenceDec,
            float activeColPermanenceDec);

        int numColumns_;                  // Number of columns in the HTM layer
        int numPotSynapses_;              // Number of potential synapses per column
        float spatialPermanenceInc_;      // Amount to increment synapse permanence values
        float spatialPermanenceDec_;      // Amount to decrement synapse permanence values
        float activeColPermanenceDec_;    // Amount to decrement synapse permanence values for active columns

        std::vector<std::vector<int>> prevColPotInputs_; // Previous inputs to the potential synapses
        std::vector<int> prevActiveCols_;                // Previous active columns
    };

} // namespace spatiallearn
