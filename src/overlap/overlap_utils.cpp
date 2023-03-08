
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

    // Define a function to print out a 1D vector that is simulating a 2D vector.
    void print_2d_vector(const std::vector<int> &vec1D, const std::pair<int, int> &vec2D_shape)
    {
        std::vector<std::vector<int>> temp_2d_input = unflattenVector(vec1D, vec2D_shape.first, vec2D_shape.second);
        // Print out the temp_2d_input vector
        for (int i = 0; i < temp_2d_input.size(); i++)
        {
            for (int j = 0; j < temp_2d_input[i].size(); j++)
            {
                std::cout << temp_2d_input[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Define a function to print out a 1D vector that is simulating a 4D vector.
    // vec4D_shape is a vector of size 4 that contains the shape of the 4D vector.
    void print_4d_vector(const std::vector<int> &vec1D, std::vector<int> &vec4D_shape)
    {
        std::vector<std::vector<std::vector<std::vector<int>>>> temp_4d = unflattenVector(vec1D, vec4D_shape[0], vec4D_shape[1], vec4D_shape[2], vec4D_shape[3]);
        // Print out the temp_4d vector
        for (int i = 0; i < temp_4d.size(); i++)
        {
            for (int j = 0; j < temp_4d[i].size(); j++)
            {
                for (int k = 0; k < temp_4d[i][j].size(); k++)
                {
                    for (int l = 0; l < temp_4d[i][j][k].size(); l++)
                    {
                        std::cout << temp_4d[i][j][k][l] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::vector<int>> unflattenVector(const std::vector<int> &vec1D, size_t numRows, size_t numCols)
    {
        std::vector<std::vector<int>> vec2D(numRows, std::vector<int>(numCols));
        size_t index = 0;
        for (size_t i = 0; i < numRows; i++)
        {
            for (size_t j = 0; j < numCols; j++)
            {
                vec2D[i][j] = vec1D[index];
                index++;
            }
        }
        return vec2D;
    }

    std::vector<std::vector<std::vector<std::vector<int>>>> unflattenVector(const std::vector<int> &vec1D, size_t numLayers, size_t numChannels, size_t numRows, size_t numCols)
    {
        std::vector<std::vector<std::vector<std::vector<int>>>> vec4D(numLayers, std::vector<std::vector<std::vector<int>>>(numChannels, std::vector<std::vector<int>>(numRows, std::vector<int>(numCols))));
        size_t index = 0;
        for (size_t l = 0; l < numLayers; l++)
        {
            for (size_t c = 0; c < numChannels; c++)
            {
                for (size_t i = 0; i < numRows; i++)
                {
                    for (size_t j = 0; j < numCols; j++)
                    {
                        vec4D[l][c][i][j] = vec1D[index];
                        index++;
                    }
                }
            }
        }
        return vec4D;
    }

} // namespace overlap_utils