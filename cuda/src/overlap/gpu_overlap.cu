#include <vector>
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
    ///                            launched with a 2D grid of 2D blocks on a GPU. Each thread in the block will
    ///                            perform the sliding window operation on a single element in the input
    ///                            matrix. The output matrix will also be a 1D vector simulating a 4D vector with dimensions
    ///                            rows x cols x neigh_rows x neigh_cols (if the step size is (1,1)).
    ///                            Each element at the output[i * cols + j] will be a 2D matrix (simulated by a flattened 1D vector)
    ///                            containing the neighbourhood of the input matrix element at input[i * cols + j].
    ///                            Note that the output is actually a 1D vector simulating a 4D vector with dimensions
    ///                            rows x cols x neigh_rows x neigh_cols (if the step size is 1,1).
    ///                            If a step size is not 1,1 then the neighbourhood "patch" will be stepped over the input matrix
    ///                            by the step size in each dimension. E.g a step size of 2,2 will step the neighbourhood
    ///                            over the input matrix by 2 rows and 2 columns (in the input simulated 2D vector) for each iteration. 
    ///                            This means the output will have a size of ceil(rows/step_rows) x ceil(cols/step_cols) x neigh_rows x neigh_cols.
    ///                            The output is a 1D vector simulating a 4D vector where each output[i][j] is a 2D matrix (simulated by a flattened 1D vector)
    ///                            containing the neighbourhood of the input matrix element at input[i*2][j*2].
    ///
    /// @param[in] input           A pointer to the input matrix on the GPU.
    /// @param[out] output         A pointer to the output matrix on the GPU.
    /// @param[in] in_rows         The number of rows in the input matrix.
    /// @param[in] in_cols         The number of columns in the input matrix.
    /// @param[in] out_rows        The number of rows in the output matrix.
    ///                            Should be equal to ceil(in_rows / step_rows)
    /// @param[in] out_cols        The number of columns in the output matrix.
    ///                            Should be equal to ceil(in_cols / step_cols)
    /// @param[in] neib_rows       The number of rows in the neighbourhood.
    /// @param[in] neib_cols       The number of columns in the neighbourhood.
    /// @param[in] step_rows       The number of rows to step the neighbourhood over the input for each iteration.
    /// @param[in] step_cols       The number of columns to step the neighbourhood over the input for each iteration.
    /// @param[in] wrap_mode       A flag indicating whether the neighbourhood should wrap around the input matrix.
    /// @param[in] center_neigh    A flag indicating whether the neighbourhood should be centered over the current element in the input matrix.
    ///-----------------------------------------------------------------------------
    __global__ void sliding_window_kernel(int *input, int *output, int in_rows, int in_cols, int out_rows, int out_cols, int neib_rows, int neib_cols, int step_rows, int step_cols, bool wrap_mode, bool center_neigh)
    {
        // The thread index is the index of the element in the output matrix that the current thread will operate on.
        // Each thread calcualtes the neighbourhood of a single element (i,j) in the ouptut matrix.
        int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of the thread index
        int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the thread index

        // int out_rows = ceil(in_rows / step_rows); // Number of rows in output matrix
        // int out_cols = ceil(in_cols / step_cols); // Number of columns in output matrix
        // The threads in the block that are outside the bounds of the output matrix do nothing.
        if (i < out_rows && j < out_cols)
        {
            // The output matrix is a 1D vector simulating a 4D vector with dimensions:
            //  ceil(rows/step_rows) x ceil(cols/step_cols) x neigh_rows x neigh_cols.
            // ii and jj are the row and column indices of the current element in the neighbourhood.
            for (int ii = 0; ii < neib_rows; ++ii)
            {
                for (int jj = 0; jj < neib_cols; ++jj)
                {
                    // The indices into the input matrix of the current element in the neighbourhood.
                    int x = i * step_rows + ii; // Row index of the current element in the neighbourhood.
                    int y = j * step_cols + jj; // Column index of the current element in the neighbourhood.

                    // If the "center_neigh" flag is set, center the neighbourhood over the current element in the input matrix.
                    if (center_neigh)
                    {
                        x = i * step_rows + (ii - neib_rows / 2);
                        y = j * step_cols + (jj - neib_cols / 2);
                    }

                    // Wrap the indices around the bounds of the input matrix if "wrap_mode" is set.
                    if (wrap_mode)
                    {
                        x = (x + in_rows) % in_rows;
                        y = (y + in_cols) % in_cols;
                    }

                    // Set the element in the output matrix.
                    // Make sure the indicies x,y are within the bounds of the input matrix.
                    if (x >= 0 && x < in_rows && y >= 0 && y < in_cols)
                    {
                        // Set output matrix element i,j,ii,jj to the input matrix element x,y.
                        int temp_idx = (i * out_cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;
                        int temp_out = input[x * in_cols + y];
                        output[temp_idx] = temp_out;
                    }
                    else
                    {
                        // Set the element in the output matrix to 0 if the indices are outside the bounds of the input matrix.
                        int temp_idx = (i * out_cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;
                        output[temp_idx] = 0;
                    }
                }
            }
        }
    }

    
    ///-----------------------------------------------------------------------------
    ///
    /// overlap_kernel      A kernel function that performs the cortical overlap score calculation on an input matrix (2D grid).
    ///                     This kernel function oerates on a simualted 2D matrix, but the matrix is
    ///                     actually stored as a 1D array. The kernel function is designed to be
    ///                     launched with a 2D grid of 2D blocks on a GPU. Each thread in the block will
    ///                     perform the overlap calculation on a single element in the input
    ///                     matrix. The output matrix will also be a 1D vector simulating a 2D vector with dimensions
    ///                     num_cortical_rows x num_cortical_cols. The num_cortical_rows and num_cortical_cols depend on the
    ///                     step size and the input_grid size.
    ///                     Each element at the output[i * cols + j] will be a value that indicates how many "connected" synapses
    ///                     in the neighboroughood are attached in active inputs in the inputGrid.
    ///                     If a step size is not 1,1 then the neighbourhood "patch" will be stepped over the input matrix
    ///                     by the step size in each dimension. E.g a step size of 2,2 will step the neighbourhood
    ///                     over the input matrix by 2 rows and 2 columns (in the input simulated 2D vector) for each iteration. 
    ///                     This means the output will have a size of ceil(rows/step_rows) x ceil(cols/step_cols).
    ///                     The output is a 1D vector simulating a 2D vector where each output[i][j] is a single value of the overlap score 
    ///                     for the cortical column at i,j position in the 2d grid of cortical columns.
    ///
    /// @param[in] in_grid           A pointer to the input matrix on the GPU. 1D vector simulating a 2D vector size (in_rows * in_cols ).
    /// @param[in] in_pot_syn_tie_breaker  A pointer to the input matrix on the GPU, representing the pot synapse tie break values 
    ///                                    Potential synapse tie breaker matrix. It contains small values that help resolve any ties in 
    ///                                    potential overlap scores for columns. This is a 1D vector simulating a 4D vector with the 
    ///                                    size of the number of columns x number of potential synapses.
    ///                                    size = columns_height_ x columns_width_ x potential_height_ x potential_width_
    ///                                     (# TODO just calculate this instead).
    /// @param[out] out_overlap    A pointer to the output matrix on the GPU.
    ///                            The out_overlap matrix is a 1D vector simulating a 2D vector with dimensions:
    ///                            size = ceil(rows/step_rows) x ceil(cols/step_cols).
    ///                            
    /// @param[in] in_rows         The number of rows in the input matrix.
    /// @param[in] in_cols         The number of columns in the input matrix.
    /// @param[in] out_rows        The number of rows in the output matrix.
    ///                            Should be equal to ceil(in_rows / step_rows)
    /// @param[in] out_cols        The number of columns in the output matrix.
    ///                            Should be equal to ceil(in_cols / step_cols)
    /// @param[in] neib_rows       The number of rows in the neighbourhood.
    /// @param[in] neib_cols       The number of columns in the neighbourhood.
    /// @param[in] step_rows       The number of rows to step the neighbourhood over the input for each iteration.
    /// @param[in] step_cols       The number of columns to step the neighbourhood over the input for each iteration.
    /// @param[in] wrap_mode       A flag indicating whether the neighbourhood should wrap around the input matrix.
    /// @param[in] center_neigh    A flag indicating whether the neighbourhood should be centered over the current element in the input matrix.
    ///-----------------------------------------------------------------------------
    __global__ void overlap_kernel(int *in_grid, float *in_pot_syn_tie_breaker, float *out_overlap, 
                                   int in_rows, int in_cols, int out_rows, int out_cols, 
                                   int neib_rows, int neib_cols, 
                                   int step_rows, int step_cols, 
                                   bool wrap_mode, bool center_neigh)
    {
        // The thread index is the index of the element in the output matrix that the current thread will operate on.
        // Each thread calculates the neighbourhood of a single element (i,j) and then calcualtes the overlap score in 
        // the ouptut matrix from this neighbourhood.
        int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of the thread index
        int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the thread index

        // int out_rows = ceil(in_rows / step_rows); // Number of rows in output matrix
        // int out_cols = ceil(in_cols / step_cols); // Number of columns in output matrix
        // The threads in the block that are outside the bounds of the output matrix do nothing.
        if (i < out_rows && j < out_cols)
        {
            // The out_overlap matrix is a 1D vector simulating a 2D vector with dimensions:
            //  size = number of cortical cols = ceil(rows/step_rows) x ceil(cols/step_cols).
            
            float neib_and_tie_sum = 0.0f; // The sum of the elements in the neighbourhood and the corresponding tie breaker values for each neighbourhood element. 
            // ii and jj are the row and column indices of the current element in the neighbourhood.
            for (int ii = 0; ii < neib_rows; ++ii)
            {
                for (int jj = 0; jj < neib_cols; ++jj)
                {
                    // The indices into the in_grid matrix of the current element in the neighbourhood.
                    int x = i * step_rows + ii; // Row index of the current element in the neighbourhood.
                    int y = j * step_cols + jj; // Column index of the current element in the neighbourhood.

                    // If the "center_neigh" flag is set, center the neighbourhood over the current element in the in_grid matrix.
                    if (center_neigh)
                    {
                        x = i * step_rows + (ii - neib_rows / 2);
                        y = j * step_cols + (jj - neib_cols / 2);
                    }

                    // Wrap the indices around the bounds of the in_grid matrix if "wrap_mode" is set.
                    if (wrap_mode)
                    {
                        x = (x + in_rows) % in_rows;
                        y = (y + in_cols) % in_cols;
                    }

                    // Add the element in the in_pot_syn_tie_breaker matrix to the in_grid columns neirbourhood.
                    // Then find the sum of the elements in these neighbourhoods after the tie breaker value was added.
                    // Make sure the indicies x,y are within the bounds of the in_grid matrix.
                    
                    
                    if (x >= 0 && x < in_rows && y >= 0 && y < in_cols)
                    {
                        // Get out_overlap matrix element i,j,ii,jj to the in_grid matrix element x,y.
                        int temp_idx = (i * out_cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;
                        int temp_out = in_grid[x * in_cols + y];
                        // Add the corresponding tie breaker value to each of the neighbourhood elements.
                        // then sum the elements in the neighbourhood. 
                        neib_and_tie_sum += in_pot_syn_tie_breaker[temp_idx] + temp_out;
                    }
                    else
                    {
                        // Set the element in the out_overlap matrix to 0 if the indices are outside the bounds of the in_grid matrix.
                        // int temp_idx = (i * out_cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;
                        // neib_and_tie_sum += 0;  // Add nothing as this index is outside the bounds of the in_grid matrix.
                    }
                }
            }
            // Now that we have the sum of the neighbourhood and the corresponding tie breaker values for each neighbourhood element,
            // set the output = "overlap score" at each cortical column to this value.
            int cort_col_id = i * out_cols + j; // The index of the cortical column in the 2D grid of cortical columns.
            out_overlap[cort_col_id] = neib_and_tie_sum;
        }
    }

    // A function that performs a sliding window operation on an input 2D simulated matrix using a 1D input vector..
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

        // allocate the host and device storage for the ouput matrix.
        auto allocate_out = taskflow.emplace([&]()
                                             {
                                                // Host storage
                                                output.resize(N * M * O * P);
                                                TF_CHECK_CUDA(cudaMalloc(&d_output, N * M * O * P * sizeof(int)), "failed to allocate output"); })
                                .name("allocate_out");

        // create a cudaFlow to run the sliding_window_kernel.
        auto cudaFlow = taskflow.emplace([&]()
                                         {
                                            tf::cudaFlow cf;
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
                                            
                                            auto sliding_window = cf.kernel(grid, block, 0, sliding_window_kernel, d_input, d_output, rows, cols, N, M, neib_shape.first, neib_shape.second, step.first, step.second, wrap_mode, center_neigh)
                                                                        .name("sliding_window");

                                            // copy the output matrix back to the host. Copy to the pointer of the first element in the multi dim vector.
                                            auto copy_out = cf.memcpy(output.data(), d_output, N * M * O * P * sizeof(int) ).name("copy_out"); 
                                            sliding_window.succeed(copy_in)
                                                .precede(copy_out); 
                                                
                                        tf::cudaStream stream;
                                        cf.run(stream);
                                        stream.synchronize(); })
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

    // GPU implementation of the overlap calculation.
    // Try to minimise the number of memory allocations and copies to and from the GPU.
    // There are six main steps:
    //        1. Get the inputs for each column. 
    //               This used the gpu_Images2Neibs. We recreate this function to reduce memory allocations and copies.
    //        2. Calculate the potential overlap scores for every column.
    //               This uses the parallel_maskTieBreaker which is just a element wise multiplication and then addition.
    //        3. Calculate the connected synapse inputs for every column.
    //               parallel_calcOverlap which is just a sum of each row.
    //        4. Get the connected synapse inputs for every column.
    //               Simply apply an if statement to each element in a matrix if larger then a threshold return the corresponding element in another matrix.
    //        5. Calculate the actual overlap scores for every column.
    //               parallel_calcOverlap which is just a sum of each row.
    //        6. Add a small tie breaker value to each cortical column's actual overlap score so draws in overlap scores can be resolved.
    //               This uses the parallel_addVectors which is just a element wise addition.
    void calculate_overlap_gpu(const std::vector<float> &colSynPerm,
                               const std::pair<int, int> &colSynPerm_shape,
                               const std::vector<int> &inputGrid,
                               const std::pair<int, int> &inputGrid_shape,
                               const std::pair<int, int> &neib_shape,
                               const std::pair<int, int> &neib_step,
                               bool wrap_mode,
                               bool center_neigh,
                               const std::vector<float> &pot_syn_tie_breaker,
                               std::vector<int> overlap_output
                               )
    {
        // The original implementation of the overlap calculation.
        //
        //      get_col_inputs(inputGrid, inputGrid_shape, col_input_pot_syn_, tf1);
        //      parallel_maskTieBreaker(col_input_pot_syn_, pot_syn_tie_breaker_, col_input_pot_syn_tie_, tf2);
        //      parallel_calcOverlap(col_input_pot_syn_tie_, num_columns_, potential_height_ * potential_width_, col_pot_overlaps_, tf3);
        //      get_connected_syn_input(colSynPerm, col_input_pot_syn_, connected_perm_,
        //                                        num_columns_, potential_height_ * potential_width_,
        //                                        con_syn_input_, tf4);
        //      parallel_calcOverlap(con_syn_input_, num_columns_, potential_height_ * potential_width_, col_overlaps_, tf5);
        //      parallel_addVectors(col_overlaps_, col_tie_breaker_, col_overlaps_tie_, tf6);

        // Create a Taskflow object to manage tasks and their dependencies.
        // There should be one taskflow object for the entire program.
        tf::Taskflow taskflow("calculate_overlap_gpu");
        tf::Executor executor;

        // Step 1. Get the inputs for each column.
        // Determine the dimensions of the input matrix.
        const int rows = inputGrid_shape.first;
        const int cols = inputGrid_shape.second;

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

        int N = static_cast<int>(ceil(static_cast<float>(rows) / neib_step.first));  // Number of rows in output matrix
        int M = static_cast<int>(ceil(static_cast<float>(cols) / neib_step.second)); // Number of columns in output matrix
        int O = neib_shape.first;                                               // Number of rows in each patch
        int P = neib_shape.second;                                              // Number of columns in each patch

        // The output is a 1D vector simulating a 4D vector with dimensions N x M x O x P.
        //std::vector<int> output;
        // The output is a 1D vector simulating a 2D vector with dimensions N x M. This is the overlap score for each cortical column.
        std::vector<int> out_overlap;

        // Step2 inputs pot_syn_tie_breaker_
        // Make the pot_syn_tie_breaker_ matrix, this should have be done already and passed in as an input.
        // TODO

        

        // Allocate memory on the GPU for the input matrix "inputGrid" and "pot_syn_tie_breaker_" matrix
        int *d_in_grid;
        float *d_in_pot_syn_tie_breaker; 
        // Allocate memory on the GPU for putting the output, the overlap scores for each cortical column.
        float *d_out_overlap;

        // allocate device storage for the input matrix. The host (CPU) already has storage for the input.
        auto allocate_in = taskflow.emplace([&]()
                                            { TF_CHECK_CUDA(cudaMalloc(&d_in_grid, rows * cols * sizeof(int)), "failed to allocate d_in_grid"); 
                                              TF_CHECK_CUDA(cudaMalloc(&d_in_pot_syn_tie_breaker, N * M * sizeof(float)), "failed to allocate d_in_pot_syn_tie_breaker"); })
                               .name("allocate_in");

        // allocate the host and device storage for the ouput matrix.
        auto allocate_out = taskflow.emplace([&]()
                                             {
                                                // Host storage
                                                out_overlap.resize(N * M);
                                                TF_CHECK_CUDA(cudaMalloc(&d_out_overlap, N * M * sizeof(float)), "failed to allocate output"); })
                                .name("allocate_out");

        // create a cudaFlow to run the overlap_kernel. This kernel function runs
        // the overlap calculation on the GPU.
        auto cudaFlow = taskflow.emplace([&]()
                                         {
                                            tf::cudaFlow cf;
                                            // copy the input matrix to the GPU. Copy from the first element in the multi dim vector.
                                            auto copy_in = cf.memcpy(d_in_grid, inputGrid.data(), rows * cols * sizeof(int)).name("copy_in");
                                            auto copy_pot_syn_tie_breaker = cf.memcpy(d_in_pot_syn_tie_breaker, pot_syn_tie_breaker.data(), N * M * O * P * sizeof(float)).name("copy_pot_syn_tie_breaker");

                                            // launch the kernel function on the GPU.
                                            int threadsPerBlock = 256;
                                            dim3 block(16, 16);   // 256 threads per block. A standard value this can be increased on some GPU models. 
                                            int noOfBlocks = cols * rows / 256;
                                            if ( (cols * rows) % threadsPerBlock) 
                                            {
                                                noOfBlocks++;
                                            }
                                            dim3 grid((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
                                            
                                            auto overlap_calc = cf.kernel(grid, block, 0, overlap_kernel, d_in_grid, d_in_pot_syn_tie_breaker, d_out_overlap, rows, cols, N, M, neib_shape.first, neib_shape.second, step.first, step.second, wrap_mode, center_neigh)
                                                                        .name("overlap_calc");

                                            // copy the output matrix back to the host. Copy to the pointer of the first element in the multi dim vector.
                                            auto copy_out = cf.memcpy(out_overlap.data(), d_out_overlap, N * M * sizeof(float) ).name("copy_out"); 
                                            overlap_calc.succeed(copy_pot_syn_tie_breaker).succeed(copy_in)
                                                .precede(copy_out); 
                                                
                                        tf::cudaStream stream;
                                        cf.run(stream);
                                        stream.synchronize(); })
                            .name("cudaFlow");

        auto free = taskflow.emplace([&]()
                                     {
                                         TF_CHECK_CUDA(cudaFree(d_in_grid), "failed to free d_in_grid");
                                         TF_CHECK_CUDA(cudaFree(d_out_overlap), "failed to free d_out_overlap"); })
                        .name("free");

        // create the dependency graph.
        cudaFlow.succeed(allocate_in, allocate_out)
            .precede(free);

        executor.run(taskflow)
            .wait();

        
    }

    // TODO: Remove this function as it doesn't work!
    // Same function as above but different input and output parameters.
    // NOTE: passing in the tasflow and trying to run it outside of the function doesn't work
    // since many other vars in this function will go out of scope.
    // void gpu_Images2Neibs(
    //     std::vector<int> &output,
    //     std::vector<int> &output_shape,
    //     const std::vector<int> &input,
    //     const std::pair<int, int> &input_shape,
    //     const std::pair<int, int> &neib_shape,
    //     const std::pair<int, int> &neib_step,
    //     bool wrap_mode,
    //     bool center_neigh,
    //     tf::Taskflow &taskflow)
    // {
    //     // Determine the dimensions of the input matrix.
    //     const int rows = input_shape.first;
    //     const int cols = input_shape.second;

    //     // Check that the neighbourhood shape is valid.
    //     if (neib_shape.first > rows || neib_shape.second > cols)
    //     {
    //         throw std::invalid_argument("Neighbourhood shape must not be larger than the input matrix");
    //     }

    //     // Set the default step size to the neighbourhood shape.
    //     std::pair<int, int> step = neib_step;
    //     if (step.first == 0 && step.second == 0)
    //     {
    //         step = neib_shape;
    //     }

    //     int N = static_cast<int>(ceil(static_cast<float>(rows) / step.first));  // Number of rows in output matrix
    //     int M = static_cast<int>(ceil(static_cast<float>(cols) / step.second)); // Number of columns in output matrix
    //     int O = neib_shape.first;                                               // Number of rows in each patch
    //     int P = neib_shape.second;                                              // Number of columns in each patch

    //     // Assert that the output matrix has the correct size.
    //     assert(output.size() == N * M * O * P);

    //     // Allocate memory on the GPU for the input and output matrices.
    //     int *d_input, *d_output;
    //     TF_CHECK_CUDA(cudaMalloc(&d_input, rows * cols * sizeof(int)), "failed to allocate input");
    //     TF_CHECK_CUDA(cudaMalloc(&d_output, N * M * O * P * sizeof(int)), "failed to allocate output");

    //     // Copy the input matrix to the GPU.
    //     TF_CHECK_CUDA(cudaMemcpy(d_input, input.data(), rows * cols * sizeof(int), cudaMemcpyHostToDevice), "failed to copy input to device");

    //     // Launch the kernel function on the GPU using taskflow.
    //     auto kernel = taskflow.emplace([&]()
    //                                    {
    //         int threadsPerBlock = 256;
    //         dim3 block(16, 16);
    //         int noOfBlocks = cols * rows / 256;
    //         if ((cols * rows) % threadsPerBlock)
    //         {
    //             noOfBlocks++;
    //         }
    //         dim3 grid((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
    //         sliding_window_kernel<<<grid, block>>>(d_input, d_output, rows, cols, N, M, neib_shape.first, neib_shape.second, step.first, step.second, wrap_mode, center_neigh); });

    //     // Copy the output matrix from the GPU using taskflow.
    //     auto copy = taskflow.emplace([&]()
    //                                  {
    //         output.resize(N * M * O * P);
    //         TF_CHECK_CUDA(cudaMemcpy(output.data(), d_output, N * M * O * P * sizeof(int), cudaMemcpyDeviceToHost), "failed to copy output to host"); });

    //     // Free memory on the GPU using taskflow.
    //     auto free_memory = taskflow.emplace([&]()
    //                                         {
    //         TF_CHECK_CUDA(cudaFree(d_input), "failed to free d_input");
    //         TF_CHECK_CUDA(cudaFree(d_output), "failed to free d_output"); });

    //     // Set task dependencies and run the taskflow.
    //     kernel.precede(copy);
    //     copy.precede(free_memory);

    //     output_shape = {N,  // output_rows
    //                     M,  // output_cols
    //                     O,  // neib_shape.first
    //                     P}; // neib_shape.second
    // }

} // namespace gpu_overlap