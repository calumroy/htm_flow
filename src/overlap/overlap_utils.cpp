
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <taskflow/taskflow.hpp>

#include <utilities/logger.hpp>
#include <htm_flow/overlap_utils.hpp>

namespace overlap_utils
{

    // Define a function to convert 2D indices to 1D indices
    int flatten_index(int x, int y, int width)
    {
        return x + y * width;
    }

    // Define a function to convert 1D indices to 2D indices
    std::tuple<int, int> unflatten_index(int index, int width)
    {
        int x = index % width;
        int y = (index - x) / width;
        return std::make_tuple(x, y);
    }

    // Define a function to wrap 2D indices around the input dimensions
    std::tuple<int, int> wrap_indices(int x, int y, int input_width, int input_height)
    {
        x = x % input_width;
        if (x < 0)
            x += input_width;
        y = y % input_height;
        if (y < 0)
            y += input_height;
        return std::make_tuple(x, y);
    }

    // Multiply two tensors element-wise
    std::vector<float>
    multiple(const std::vector<float> &a, const std::vector<float> &b)
    {
        std::vector<float> result;
        result.reserve(a.size());

        for (size_t i = 0; i < a.size(); ++i)
        {
            result.push_back(a[i] * b[i]);
        }

        return result;
    }

    // Apply tie breaker values to an input grid by multiplying the grid by the tie breaker values and adding the result to the grid.
    std::vector<float> maskTieBreaker(const std::vector<float> &grid, const std::vector<float> &tieBreaker)
    {
        std::vector<float> multi_vals;
        multi_vals.reserve(grid.size());

        for (size_t i = 0; i < grid.size(); ++i)
        {
            multi_vals.push_back(grid[i] * tieBreaker[i]);
        }

        std::vector<float> result;
        result.reserve(grid.size());

        for (size_t i = 0; i < grid.size(); ++i)
        {
            result.push_back(grid[i] + multi_vals[i]);
        }

        return result;
    }
} // namespace overlap_utils