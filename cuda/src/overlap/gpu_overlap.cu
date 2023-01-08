#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

#include <overlap/gpu_overlap.hpp>

namespace gpu_overlap
{

    // Define the kernel function.
    __global__ void sliding_window_kernel(int *input, int *output, int rows, int cols, int neib_rows, int neib_cols, int step_rows, int step_cols, bool wrap_mode, bool center_neigh)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < rows && j < cols)
        {
            for (int ii = 0; ii < neib_rows; ++ii)
            {
                for (int jj = 0; jj < neib_cols; ++jj)
                {
                    int x = i + ii;
                    int y = j + jj;

                    // If the "center_neigh" flag is set, center the neighbourhood over the current element in the input matrix.
                    if (center_neigh)
                    {
                        x = i + ii - neib_rows / 2;
                        y = j + jj - neib_cols / 2;
                    }

                    if (wrap_mode)
                    {
                        x = (x + rows) % rows;
                        y = (y + cols) % cols;
                    }
                    if (x >= 0 && x < rows && y >= 0 && y < cols)
                    {
                        output[i * cols + j * neib_rows * neib_cols + ii * neib_cols + jj] = input[x * cols + y];
                    }
                    else
                    {
                        output[i * cols + j * neib_rows * neib_cols + ii * neib_cols + jj] = 0;
                    }
                }
            }
        }
    }

    std::vector<std::vector<std::vector<std::vector<int>>>> gpu_Images2Neibs(
        const std::vector<std::vector<int>> &input,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh)
    {
        // Determine the dimensions of the input matrix.
        const int rows = input.size();
        const int cols = input[0].size();

        // Check that the neighbourhood shape is valid.
        if (neib_shape.first > rows || neib_shape.second > cols)
        {
            throw std::invalid_argument("Neighbourhood shape must not be larger than the input matrix");
        }

        // Set the default step size to the neighbourhood shape.
        std::pair<int, int> step = neib_step;
        if (step.first == 0 && step.second == 0)
        {
            step = neib_shape;
        }

        // Create the output matrix.
        std::vector<std::vector<std::vector<std::vector<int>>>> output;

        // Set up the GPU.
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            throw std::runtime_error("No CUDA devices found");
        }
        int device;
        cudaGetDevice(&device);
        cudaSetDevice(device);

        // Allocate memory on the GPU for the input matrix.
        int *d_input;
        cudaMalloc(&d_input, rows * cols * sizeof(int));

        // Copy the input matrix to the GPU.
        cudaMemcpy(d_input, input.data(), rows * cols * sizeof(int), cudaMemcpyHostToDevice);

        // Create a taskflow to parallelize the computation.
        tf::Taskflow taskflow;

        // Add tasks for each patch in the output matrix.
        for (int i = 0; i < rows; i += step.first)
        {
            std::vector<std::vector<std::vector<int>>> row_output;
            for (int j = 0; j < cols; j += step.second)
            {
                taskflow.emplace([&, i, j]()
                                 {
                std::vector<std::vector<int>> patch(neib_shape.first, std::vector<int>(neib_shape.second));
                int *d_output;
            // Allocate memory on the GPU for the output matrix.
                cudaMalloc(&d_output, neib_shape.first * neib_shape.second * sizeof(int));

                // Launch the kernel function on the GPU.
                dim3 grid((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
                dim3 block(16, 16);
                sliding_window_kernel<<<grid, block>>>(d_input, d_output, rows, cols, neib_shape.first, neib_shape.second, step.first, step.second, wrap_mode, center_neigh);

                // Copy the output matrix back to the host.
                cudaMemcpy(patch.data(), d_output, neib_shape.first * neib_shape.second * sizeof(int), cudaMemcpyDeviceToHost);

                // Free the memory on the GPU.
                cudaFree(d_output);

                row_output.push_back(patch); });
            }
            output.push_back(row_output);
        }

        // Run the taskflow and wait for it to complete.
        tf::Executor executor;
        executor.run(taskflow).wait();

        // Free the memory on the GPU.
        cudaFree(d_input);

        return output;
    }

} // namespace gpu_overlap