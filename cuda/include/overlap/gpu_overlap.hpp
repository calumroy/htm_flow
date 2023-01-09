
#pragma once

#include <vector>
#include <tuple>

namespace gpu_overlap
{
    ///-----------------------------------------------------------------------------
    ///
    /// flattenVector              Flattens a 2D vector into a 1D vector.
    ///
    /// @param[in] vec2D           A 2D vector.
    ///
    /// @return                    A 1D vector.
    ///-----------------------------------------------------------------------------
    std::vector<int> flattenVector(const std::vector<std::vector<int>> &vec2D);
    std::vector<int> flattenVector(const std::vector<std::vector<std::vector<std::vector<int>>>> &vec4D);
    std::vector<std::vector<int>> unflattenVector(const std::vector<int> &vec1D, size_t numRows, size_t numCols);
    std::vector<std::vector<std::vector<std::vector<int>>>> unflattenVector(const std::vector<int> &vec1D, size_t numLayers, size_t numChannels, size_t numRows, size_t numCols);

    std::vector<int> gpu_Images2Neibs(
        const std::vector<int> &input,
        const std::pair<int, int> &input_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

} // namespace gpu_overlap