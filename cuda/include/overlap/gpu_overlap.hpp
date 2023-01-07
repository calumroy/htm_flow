
#pragma once

#include <vector>
#include <tuple>

template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> gpu_Images2Neibs(
    const std::vector<std::vector<T>> &input,
    const std::pair<int, int> &neib_shape,
    const std::pair<int, int> &neib_step,
    bool wrap_mode,
    bool center_neigh);