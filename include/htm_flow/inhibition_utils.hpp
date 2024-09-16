#pragma once

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace inhibition_utils
{
    ///-----------------------------------------------------------------------------
    ///
    /// parallel_sort   Sorts a vector of indices based on the corresponding values in the
    ///                 values input vector. Each value in indices should be sorted by the corresponding
    ///                 value in the values vector at the index in indicies. The sorting should happen in
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
        assert(indices.size() == values.size());

        size_t indices_size = indices.size();
        unsigned int num_threads = std::thread::hardware_concurrency();
        size_t chunk_size = std::max(size_t(1), indices_size / num_threads);

        // Calculate the number of chunks
        size_t num_chunks = (indices_size + chunk_size - 1) / chunk_size;

        std::vector<tf::Task> sort_tasks;
        sort_tasks.reserve(num_chunks);

        // Create sorting tasks for each chunk
        for (size_t i = 0; i < num_chunks; ++i) {
            sort_tasks.emplace_back(taskflow.emplace([&indices, &values, i, chunk_size, indices_size]() {
                size_t start = i * chunk_size;
                size_t end = std::min((i + 1) * chunk_size, indices_size);
                // Use std::stable_sort instead of std::sort to preserve relative order of equal values
                std::stable_sort(indices.begin() + start, indices.begin() + end, [&values](T a, T b) {
                    return values[a] > values[b]; // Sort indices based on corresponding values in descending order.
                });
            }).name("sort_chunk_" + std::to_string(i)));
        }

        // Add a merge step to combine the sorted chunks
        auto merge_task = taskflow.emplace([&indices, &values, chunk_size, indices_size]() {
            for (size_t size = chunk_size; size < indices_size; size *= 2) {
                for (size_t left = 0; left < indices_size - size; left += 2 * size) {
                    size_t mid = left + size;
                    size_t right = std::min(left + 2 * size, indices_size);
                    std::inplace_merge(indices.begin() + left, indices.begin() + mid, indices.begin() + right,
                        [&values](T a, T b) { return values[a] > values[b]; });
                }
            }
        }).name("merge_chunks");

        // Correct dependency setup: sort tasks should precede the merge task
        for (auto& task : sort_tasks) {
            task.precede(merge_task);
        }
    }
        
} // namespace inhibition_utils