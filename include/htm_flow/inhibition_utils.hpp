
#pragma once

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace inhibition_utils
{
    ///-----------------------------------------------------------------------------
    ///
    /// parallel_sort   Sorts a vector of indices based on the corresponding values in the
    ///                 values input vector. Each value in indices should be sorted by the corresponding
    ///                 value in the values vector at the index in indicies. The sorting shoudl happen in
    ///                 parallel using Taskflow.
    ///
    /// @param[in,out] indices  The indices to be sorted.
    /// @param[in] values The values based on which the sorting is to be performed.
    /// @param[in,out] taskflow The Taskflow object for managing tasks.
    ///
    ///-----------------------------------------------------------------------------
    template <typename T>
    void parallel_sort_ind(std::vector<T> &indices, const std::vector<T> &values, tf::Taskflow &taskflow) {
        assert(!indices.empty());

        // Lambda function to perform sorting on a chunk of indices
        auto sort_indices = [&indices, &values](int begin, int end) {
            std::sort(indices.begin() + begin, indices.begin() + end, [&](T a, T b) {
                return values[a] > values[b]; // Sort in descending order.
            });
        };

        // Divide the work among available threads
        int num_threads = std::thread::hardware_concurrency();
        int chunk_size = indices.size() / num_threads;

        std::vector<tf::Task> tasks;
        tasks.reserve(num_threads);

        // Create sorting tasks for each chunk
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? indices.size() : (i + 1) * chunk_size;

            tasks.emplace_back(taskflow.emplace([start, end, sort_indices]() {
                sort_indices(start, end);
            }).name("sort_chunk_" + std::to_string(i)));
        }

        // Add a merge step to combine the sorted chunks
        taskflow.emplace([&indices, chunk_size, num_threads]() {
            for (int size = chunk_size; size < indices.size(); size *= 2) {
                for (int left = 0; left < indices.size() - size; left += 2 * size) {
                    int mid = left + size;
                    int right = std::min(left + 2 * size, static_cast<int>(indices.size()));
                    std::inplace_merge(indices.begin() + left, indices.begin() + mid, indices.begin() + right, [&](T a, T b) {
                        return a > b;
                    });
                }
            }
        }).name("merge_chunks").precede(tasks.back());
    }
    
} // namespace inhibition_utils