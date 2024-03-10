
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

    // Define a function to print out a 1D vector of uint32_t, where each bit in the uint32_t elements
    // represents a boolean value, simulating a 4D boolean vector. This function is used to visualize the bit-packed synapse connection states in a format that resembles a 4D vector, where each bit corresponds to the connected state of a synapse.
    void print_4d_bit_vector(const std::vector<uint32_t> &vec1D, const std::vector<int> &vec4D_shape) {
        int num_columns = vec4D_shape[0];
        int num_rows = vec4D_shape[1];
        int num_depth = vec4D_shape[2];
        int num_width = vec4D_shape[3];

        int bits_per_uint32 = 32;
        int total_bits = num_columns * num_rows * num_depth * num_width;

        for (int i = 0; i < num_columns; ++i) {
            for (int j = 0; j < num_rows; ++j) {
                for (int k = 0; k < num_depth; ++k) {
                    for (int l = 0; l < num_width; ++l) {
                        int bit_pos = ((i * num_rows + j) * num_depth + k) * num_width + l;
                        int vec_index = bit_pos / bits_per_uint32;
                        int bit_index = bit_pos % bits_per_uint32;

                        if (vec_index < vec1D.size()) {
                            bool bit_value = (vec1D[vec_index] & (1u << bit_index)) != 0;
                            std::cout << bit_value << " ";
                        }
                    }
                    std::cout << std::endl; // New line for each width layer
                }
                std::cout << std::endl; // New line for each depth layer
            }
            std::cout << std::endl; // New line for each row
        }
    }

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