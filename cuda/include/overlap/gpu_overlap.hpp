
#pragma once

#include <vector>
#include <tuple>

namespace gpu_overlap
{

    std::vector<std::vector<std::vector<std::vector<int>>>> gpu_Images2Neibs(
        const std::vector<std::vector<int>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

} // namespace gpu_overlap