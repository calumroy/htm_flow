#include <vector>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

#include <overlap/gpu_overlap.hpp>

namespace gpu_overlap
{

    std::vector<int> flattenVector(const std::vector<std::vector<int>> &vec2D)
    {
        std::vector<int> vec1D;
        for (const auto &vec : vec2D)
        {
            vec1D.insert(vec1D.end(), vec.begin(), vec.end());
        }
        return vec1D;
    }

    std::vector<int> flattenVector(const std::vector<std::vector<std::vector<std::vector<int>>>> &vec4D)
    {
        std::vector<int> vec1D;
        for (const auto &vec3D : vec4D)
        {
            for (const auto &vec2D : vec3D)
            {
                for (const auto &vec : vec2D)
                {
                    vec1D.insert(vec1D.end(), vec.begin(), vec.end());
                }
            }
        }
        return vec1D;
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

    ///-----------------------------------------------------------------------------
    ///
    /// sliding_window_kernel      A kernel function that performs a sliding window operation on a matrix.
    ///                            This kernel function oerates on a simualted 2D matrix, but the matrix is
    ///                            actually stored as a 1D array. The kernel function is designed to be
    ///                            launched with a 2D grid of 2D blocks. Each thread in the block will
    ///                            perform the sliding window operation on a single element in the input
    ///                            matrix. The output matrix will also be a 1D vector simulating a 4D vector with dimensions
    ///                            rows x cols x neigh_rows x neigh_cols.
    ///                            Each element at the output[i * cols + j] will be a 2D matrix (simulated by a flattened 1D vector)
    ///                            containing the neighbourhood of the input matrix element input[i * cols + j].
    ///
    /// @param[in] input           A pointer to the input matrix on the GPU.
    /// @param[out] output         A pointer to the output matrix on the GPU.
    /// @param[in] rows            The number of rows in the input matrix.
    /// @param[in] cols            The number of columns in the input matrix.
    /// @param[in] neib_rows       The number of rows in the neighbourhood.
    /// @param[in] neib_cols       The number of columns in the neighbourhood.
    /// @param[in] step_rows       The number of rows to step the neighbourhood over the input for each iteration.
    /// @param[in] step_cols       The number of columns to step the neighbourhood over the input for each iteration.
    /// @param[in] wrap_mode       A flag indicating whether the neighbourhood should wrap around the input matrix.
    /// @param[in] center_neigh    A flag indicating whether the neighbourhood should be centered over the current element in the input matrix.
    ///-----------------------------------------------------------------------------
    __global__ void sliding_window_kernel(int *input, int *output, int rows, int cols, int neib_rows, int neib_cols, int step_rows, int step_cols, bool wrap_mode, bool center_neigh)
    {
        // The thread index is the index of the element in the input matrix that the current thread will operate on.
        int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of the thread index
        int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the thread index

        // The threads in the block that are outside the bounds of the input matrix do nothing.
        if (i < rows && j < cols)
        {
            // The output matrix is a 1D vector simulating a 4D vector with dimensions rows x cols x neigh_rows x neigh_cols.
            // ii and jj are the row and column indices of the current element in the neighbourhood.
            for (int ii = 0; ii < neib_rows; ++ii)
            {
                for (int jj = 0; jj < neib_cols; ++jj)
                {
                    // The indices of the current element in the neighbourhood.
                    int x = i + ii * step_rows; // Row index of the current element in the neighbourhood.
                    int y = j + jj * step_cols; // Column index of the current element in the neighbourhood.

                    // If the "center_neigh" flag is set, center the neighbourhood over the current element in the input matrix.
                    if (center_neigh)
                    {
                        x = i + (ii - neib_rows / 2) * step_rows;
                        y = j + (jj - neib_cols / 2) * step_cols;
                    }

                    // Wrap the indices around the bounds of the input matrix if "wrap_mode" is set.
                    if (wrap_mode)
                    {
                        x = (x + rows) % rows;
                        y = (y + cols) % cols;
                    }

                    // Set the element in the output matrix
                    if (x >= 0 && x < rows && y >= 0 && y < cols)
                    {
                        // Set output matrix element i,j,ii,jj to the input matrix element x,y.
                        int temp_idx = (i * cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;
                        int temp_out = input[x * cols + y];
                        output[temp_idx] = temp_out;
                    }
                    else
                    {
                        // Set the element in the output matrix to 0 if the indices are outside the bounds of the input matrix.
                        int temp_idx = (i * cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;
                        output[temp_idx] = 0;
                    }
                }
            }
        }
    }

    ///-----------------------------------------------------------------------------
    ///
    /// gpu_Images2Neibs           A function that performs a sliding window operation on a matrix.
    ///                            This function is designed to be called from the host. It allocates
    ///                            memory on the GPU, copies the input matrix to the GPU, launches the
    ///                            sliding_window_kernel kernel function, copies the output matrix from the GPU
    ///                            and frees the memory on the GPU.
    ///
    /// @param[in] input           A reference to the input matrix on the host. This is a 1D vector simulating a 2D matrix.
    /// @param[in] input_shape     A pair containing the number of rows and columns in the input matrix.
    /// @param[in] neib_shape      A pair containing the number of rows and columns in the neighbourhood.
    /// @param[in] neib_step       A pair containing the number of rows and columns to step the neighbourhood for each iteration.
    /// @param[in] wrap_mode       A flag indicating whether the neighbourhood should wrap around the input matrix.

    std::vector<int> gpu_Images2Neibs(
        const std::vector<int> &input,
        const std::pair<int, int> &input_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh)
    {
        // Determine the dimensions of the input matrix.
        const int rows = input_shape.first;
        const int cols = input_shape.second;

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

        int N = static_cast<int>(ceil(static_cast<float>(rows) / step.first));  // Number of rows in output matrix
        int M = static_cast<int>(ceil(static_cast<float>(cols) / step.second)); // Number of columns in output matrix
        int O = neib_shape.first;                                               // Number of rows in each patch
        int P = neib_shape.second;                                              // Number of columns in each patch

        // Create the output matrix. A 1D vector simulating a 4D vector with dimensions N x M x O x P.
        std::vector<int> output;

        // Allocate memory on the GPU for the input matrix.
        int *d_input, *d_output;

        tf::Taskflow taskflow("gpu_Images2Neibs");
        tf::Executor executor;

        // allocate device storage for the input matrix. The host (CPU) already has storage for the input.
        auto allocate_in = taskflow.emplace([&]()
                                            { TF_CHECK_CUDA(cudaMalloc(&d_input, rows * cols * sizeof(int)), "failed to allocate input"); })
                               .name("allocate_in");

        // // allocate the host and device storage for the ouput matrix.
        auto allocate_out = taskflow.emplace([&]()
                                             {
                                                // Host storage
                                                output.resize(N * M * O * P);
                                                TF_CHECK_CUDA(cudaMalloc(&d_output, N * M * O * P * sizeof(int)), "failed to allocate output"); })
                                .name("allocate_out");

        // create a cudaFlow to run the sliding_window_kernel.
        auto cudaFlow = taskflow.emplace([&](tf::cudaFlow &cf)
                                         {
                                            // copy the input matrix to the GPU. Copy from the first element in the multi dim vector.
                                            auto copy_in = cf.memcpy(d_input, input.data(), rows * cols * sizeof(int)).name("copy_in");

                                            // launch the kernel function on the GPU.
                                            int threadsPerBlock = 256;
                                            dim3 block(16, 16);   // 256 threads per block. A standard value this can be increased on some GPU models. 
                                            int noOfBlocks = cols * rows / 256;
                                            if ( (cols * rows) % threadsPerBlock) 
                                            {
                                                noOfBlocks++;
                                            }
                                            dim3 grid((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
                                            
                                            auto sliding_window = cf.kernel(grid, block, 0, sliding_window_kernel, d_input, d_output, rows, cols, neib_shape.first, neib_shape.second, step.first, step.second, wrap_mode, center_neigh)
                                                                        .name("sliding_window");

                                            // copy the output matrix back to the host. Copy to the pointer of the first element in the multi dim vector.
                                            auto copy_out = cf.memcpy(output.data(), d_output, N * M * O * P * sizeof(int) ).name("copy_out"); 
                                            sliding_window.succeed(copy_in)
                                                .precede(copy_out); })
                            .name("cudaFlow");

        auto free = taskflow.emplace([&]()
                                     {
                                         TF_CHECK_CUDA(cudaFree(d_input), "failed to free d_input");
                                         TF_CHECK_CUDA(cudaFree(d_output), "failed to free d_output"); })
                        .name("free");

        // create the dependency graph.
        cudaFlow.succeed(allocate_in, allocate_out)
            .precede(free);

        executor.run(taskflow)
            .wait();

        return output;
    }

} // namespace gpu_overlap