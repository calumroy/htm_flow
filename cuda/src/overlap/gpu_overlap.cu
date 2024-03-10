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
    ///                     This kernel function operates on a simualted 2D matrix, but the matrix is
    ///                     actually stored as a 1D array. The kernel function is designed to be
    ///                     launched with a 2D grid of 2D blocks on a GPU. Each thread in the block will
    ///                     perform the overlap calculation on a single element in the input
    ///                     matrix. The output matrix will also be a 1D vector simulating a 2D vector with dimensions
    ///                     num_cortical_rows x num_cortical_cols. "Cortical columns" are the elements in the output matrix.
    ///                     These cortical columns have a group of synapses that connect to a neighbourhood of elements in the input matrix.
    ///                     Each element at the output[i * cols + j] will be a value that indicates how many "connected" synapses
    ///                     in the neighboroughood are attached in active inputs in the inputGrid.
    ///                     If a step size is not 1,1 then the neighbourhood "patch" will be stepped over the input matrix
    ///                     by the step size in each dimension. E.g a step size of 2,2 will step the neighbourhood
    ///                     over the input matrix by 2 columns and 2 rows (in the input simulated 2D vector) for each iteration. 
    ///                     This means the output will have a size of ceil(rows/step_rows) x ceil(cols/step_cols).
    ///                     The output is a 1D vector simulating a 2D vector where each output[i][j] is a single value of the overlap score 
    ///                     for the cortical column at i,j position in the 2d grid of cortical columns.
    ///                     
    ///                     The function calculates as needed a tiebreaker value for each element in the neighbourhood.
    ///                     These tie breaker values are used to resolve any ties in the overlap scores for each cortical column.
    ///                     The tie breaker values are added to the overlap scores so that the overlap scores are slightly different.
    ///                     The total of the sum of the tie breaker values for a neighbourhood is less then 0.5 so as to not affect the
    ///                     actual overlap scores (whole numbers) but only to break ties in the overlap scores.
    ///
    ///                     The function also calculates the potential overlap score for each cortical column.
    ///                     This is the sum of the elements in the neighbourhood connectted to a active input plus the tiebreaker values.
    ///                     The potential overlap score for each cortical column does not care about the synpase permance values where as the actual overlap
    ///                     score does.
    ///
    /// @param[in] in_grid        A pointer to the input matrix on the GPU. 1D vector simulating a 2D vector size (in_rows * in_cols ).
    ///                           This is a binary matrix where 1 indicates an active input and 0 indicates an inactive input.   
    /// @param[in] in_colSynPerm  A pointer to the input vector on the GPU simulating a 4D vector with dimensions:
    ///                           size = number of cortical columns = cortical col height x cortical col width = out_rows * out_cols = ceil(in_rows/step_rows) x ceil(in_cols/step_cols) x neib_rows x neib_cols.
    ///                           This is the permanence value for each synapse connected to each element in the neighbourhood of each cortical column.
    /// 
    /// @param[out] out_overlap    A pointer to the output matrix on the GPU.
    ///                            The out_overlap matrix is a 1D vector simulating a 2D vector with dimensions:
    ///                            size = number of cortical columns = cortical col height x cortical col width = out_rows * out_cols = ceil(in_rows/step_rows) x ceil(in_cols/step_cols).
    ///                           
    /// @param[out] out_potential_overlap A pointer to the output matrix on the GPU.
    ///                                   The out_potential_overlap matrix is a 1D vector simulating a 2D vector with dimensions:
    ///                                   size = number of cortical columns = cortical col height x cortical col width = out_rows * out_cols = ceil(in_rows/step_rows) x ceil(in_cols/step_cols).
    ///                            
    /// @param[in] in_rows         The number of rows in the input matrix.
    /// @param[in] in_cols         The number of columns in the input matrix.
    /// @param[in] out_rows        The number of rows in the output matrix.
    ///                            Should be equal to ceil(in_rows / step_rows)
    /// @param[in] out_cols        The number of columns in the output matrix.
    ///                            Should be equal to ceil(in_cols / step_cols)
    /// @param[in] neib_rows       The number of rows in the neighbourhood.
    /// @param[in] neib_cols       The number of columns in the neighbourhood.
    /// @param[in] step_cols       The step size in the x direction, the number of columns to step the neighbourhood over the input for each iteration.
    /// @param[in] step_rows       The step size in the y direction, the number of rows to step the neighbourhood over the input for each iteration.
    /// 
    /// @param[in] wrap_mode       A flag indicating whether the neighbourhood should wrap around the input matrix.
    /// @param[in] center_neigh    A flag indicating whether the neighbourhood should be centered over the current element in the input matrix.
    /// @param[in] connected_perm  A float of the permanence value threshold for a synapse to be considered "connected" to a cortical column.
    ///                            Synapses with a permanence below this aren't considered connected and don't contribute to the overlap score even when they "end/start" on an active input.
    ///-----------------------------------------------------------------------------
    __global__ void overlap_kernel(int *in_grid, float *in_colSynPerm, 
                               float *out_overlap, float *out_potential_overlap,
                               int in_rows, int in_cols, int out_rows, int out_cols, 
                               int neib_rows, int neib_cols, 
                               int step_cols, int step_rows, 
                               bool wrap_mode, bool center_neigh,
                               float connected_perm)
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
            // The out_potential_overlap matrix is a 1D vector simulating a 2D vector with dimensions:
            //  size = number of cortical cols = ceil(rows/step_rows) x ceil(cols/step_cols).

            // Get the norm value for the tie breaker used to calcualte a slightly different tie breaker value for each element in the neighbourhood.
            // THe sum of all these slightly different tie values is less then 0.5.
            // This is because they are all added together with the actual overlap scores so we don't want thte sum of the tie breaker values to be larger then 1 (we chose 0.5)
            // in order for the tie breaker to only break ties in overlap scores and not affect the actual overlap scores.
            float neib_and_tie_sum = 0.0f;
            float con_neib_and_tie_sum = 0.0f;  // THe sum of the neibourhood elements and tie breaker values for only the connected cortical "proxmial" synapses.
            float n = static_cast<float>(neib_cols*neib_rows);  // The number of elements in the neighbourhood.
            float norm_value = 0.5f / (n * (n + 1.0f) / 2.0f);  // The sum of the tie breaker values over a complete neighbourhood should be less then 0.5.

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
                    // Then find the sum of the elements in these neighbourhoods after the tie breaker value was added
                    // for only the active elements in the neighbourhood (the elements in the in_grid matrix that are active, > 0).
                    // The output of the above is the potential overlap score for each cortical column plus the tie breaker, "col_input_pot_syn_tie_".

                    // Next we want the actual overlap score for each cortical column.
                    // This takes the potential inputs and tiebreakers values and finds which of these has a connection to
                    // a synapse with a permanence value above the threshold. This is the connected synapse input.

                    // Take the connected synapse inputs and wum acorss the neibourhood to give the actual overlap score for each cortical column. 

                    // Make sure the indicies x,y are within the bounds of the in_grid matrix.
                    if (x >= 0 && x < in_rows && y >= 0 && y < in_cols)
                    {
                        float tie_breaker = (jj + 1) * norm_value; // Calculating the tie breaker value on-the-fly
                        int grid_value = in_grid[x * in_cols + y];
                        // Add the corresponding tie breaker value to each of the neighbourhood elements only if the grid input
                        // for this vlaue is not. THis is so tie breaker values only affect inputs that are active.
                        // Then sum the elements in the neighbourhood (including the active tie breakers) to give the
                        // potential overlap score for each cortical column.
                        if (grid_value > 0)
                        {
                            neib_and_tie_sum += tie_breaker + grid_value;
                            // Check if the proximal cortical column synapse is connected (its permanence value is above the threshold).
                            if (in_colSynPerm[(i * out_cols + j) * neib_rows * neib_cols + ii * neib_cols + jj] > connected_perm)
                            {
                                con_neib_and_tie_sum += tie_breaker + grid_value;
                            }
                        }
                        
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
            // set the output at each corresponding cortical column element to this calcualted value.
            int cort_col_id = i * out_cols + j;  // The index of the current cortical column.
            out_potential_overlap[cort_col_id] = neib_and_tie_sum;  // Potential overlap score.
            out_overlap[cort_col_id] = con_neib_and_tie_sum;   
        }
    }

    ///-----------------------------------------------------------------------------
    ///
    /// overlap_kernel_opt      A kernel function that performs the cortical overlap score calculation on an input matrix (2D grid).
    ///                         This kernel function is an optimised version of the overlap_kernel function.
    ///                         It differs from the overlap_kernel function in that it uses a 1D array of unsigned integers to store the
    ///                         connection bits for each synapse in the neighbourhood of each cortical column. This is done to reduce the
    ///                         amount of GPU memory required to store the connection bits. The connection bits are stored in a 1D array of 
    ///                         unsigned integers (32 bits long) where each bit in the integer represents the connection bit for a synapse in the neighbourhood.
    ///                         The connection bit indicates wether the synapses has a permanence strength above the threshold which allows it to be considered
    ///                         connected to the cortical column and if so used in the overlap score calculation. The connection bits are not used to calculate 
    ///                         the potential overlap score for each cortical column (potential overlap score is the sum of the elements in the neighbourhood connected 
    ///                         to a active input plus the tiebreaker values). The potential overlap score for each cortical column does not care about the synpase permance.
    ///
    /// params                    params are all the same as the overlap_kernel function except for the addition of the in_colConBits parameter and
    ///                           the removal of the in_colSynPerm parameter and connected_perm parameter.
    /// @param[in] in_colConBits  A pointer to the input vector on the GPU. Each bit in the array represents the connection bit for a synapse in the neighbourhood.
    ///                           The connection bit indicates wether the synapses has a permanence strength above the threshold which allows it to be considered
    ///                           connected to the cortical column and if so used in the overlap score calculation.
    /// ...
    ///
    ///-----------------------------------------------------------------------------
    __global__ void overlap_kernel_opt(int *in_grid, uint32_t *in_colConBits,
                                       float *out_overlap, float *out_potential_overlap,
                                       int in_rows, int in_cols, int out_rows, int out_cols, 
                                       int neib_rows, int neib_cols, 
                                       int step_cols, int step_rows, 
                                       bool wrap_mode, bool center_neigh)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
        int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

        if (i < out_rows && j < out_cols)
        {
            float neib_and_tie_sum = 0.0f;
            float con_neib_and_tie_sum = 0.0f;
            float norm_value = 0.5f / (neib_cols * neib_rows * (neib_cols * neib_rows + 1.0f) / 2.0f);

            for (int ii = 0; ii < neib_rows; ++ii)
            {
                for (int jj = 0; jj < neib_cols; ++jj)
                {
                    int x = i * step_rows + ii;
                    int y = j * step_cols + jj;

                    if (center_neigh)
                    {
                        x = i * step_rows + (ii - neib_rows / 2);
                        y = j * step_cols + (jj - neib_cols / 2);
                    }

                    if (wrap_mode)
                    {
                        x = (x + in_rows) % in_rows;
                        y = (y + in_cols) % in_cols;
                    }

                    if (x >= 0 && x < in_rows && y >= 0 && y < in_cols)
                    {
                        float tie_breaker = (jj + 1) * norm_value;
                        int grid_value = in_grid[x * in_cols + y];
                        
                        // Get the corresponding bit out of the in_colConBits array which indicates if this synapse has a permanence value above the threshold.
                        // If it does then add the tie breaker value and the grid value to the sum of the neighbourhood elements.
                        int bit_idx = (i * out_cols + j) * neib_rows * neib_cols + ii * neib_cols + jj;  // Get the index of the bit in the in_colConBits array.
                        unsigned int bit_mask = 1u << (bit_idx % 32); // Compute bit mask for the specific synapse
                        unsigned int con_bit = in_colConBits[bit_idx / 32] & bit_mask; // Extract the connection bit for the current synapse

                        if (grid_value > 0)
                        {
                            neib_and_tie_sum += tie_breaker + grid_value;

                            if (con_bit)
                            {
                                con_neib_and_tie_sum += tie_breaker + grid_value;
                            }
                        }
                    }
                }
            }

            int cort_col_id = i * out_cols + j;
            out_potential_overlap[cort_col_id] = neib_and_tie_sum;
            out_overlap[cort_col_id] = con_neib_and_tie_sum;
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
    std::vector<float> calculate_overlap_gpu(
                               const int width_cortical_cols, const int height_cortical_cols,
                               const std::vector<float> &colSynPerm,
                               const std::pair<int, int> &colSynPerm_shape,
                               const std::vector<int> &inputGrid,
                               const std::pair<int, int> &inputGrid_shape,
                               const std::pair<int, int> &neib_shape,
                               const std::pair<int, int> &neib_step,
                               bool wrap_mode,
                               bool center_neigh,
                               float connected_perm
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


        // Calculate the dimensions of the output matrix. This is the 2D size of the "cortical columns".
        int N = height_cortical_cols;  // Number of rows in output matrix. This is the height of the "cortical columns".
        int M = width_cortical_cols; // Number of columns in output matrix. This is the width of the "cortical columns".
        // Calculate the dimensions of the neighbourhood matrix (patch) that is stepped over the input matrix for each cortical column.
        int O = neib_shape.first;                                               // Number of rows in each patch connected to a cortical column.
        int P = neib_shape.second;                                              // Number of columns in each patch connected to a cortical column.

        // Their should be one permance value for every synapse connectted to a neighbourhood element in a cortical column.
        // This means the size of the colSynPerm vector should be equal to the size of the output matrix (num cortical columns) times
        // the size of the neighbourhood matrix (neib_shape.first * neib_shape.second). 
        assert(colSynPerm.size() == N * M * O * P);
        assert(colSynPerm_shape.first == N * M);
        assert(colSynPerm_shape.second == O * P);

        // The output is a 1D vector simulating a 4D vector with dimensions N x M x O x P.
        //std::vector<int> output;calculate_overlap_gpu a 2D vector with dimensions N x M. This is the overlap score for each cortical column.
        std::vector<float> out_overlap;

        // Allocate memory on the GPU for the input matrix "inputGrid" and "colSynPerm" matrix.
        int *d_in_grid;
        float *d_colSynPerm;

        // Allocate memory on the GPU for putting the output, the overlap scores for each cortical column.
        // Also allocate memory for the potential overlap scores for each cortical column.
        float *d_out_overlap;      // Overlap scores for each cortical column.
        float *d_out_pot_overlap;   // Potnetial overlap scores for each cortical column.

        // allocate device storage for the input matrix. The host (CPU) already has storage for the input.
        auto allocate_in = taskflow.emplace([&]()
                                            { TF_CHECK_CUDA(cudaMalloc(&d_in_grid, rows * cols * sizeof(int)), "failed to allocate d_in_grid"); 
                                              TF_CHECK_CUDA(cudaMalloc(&d_colSynPerm, N * M * O * P * sizeof(float)), "failed to allocate d_colSynPerm"); 
                                            })
                               .name("allocate_in");

        // allocate the host and device storage for the ouput matrix(s).
        auto allocate_out = taskflow.emplace([&]()
                                             {
                                                // Host storage
                                                out_overlap.resize(N * M);
                                                TF_CHECK_CUDA(cudaMalloc(&d_out_overlap, N * M * sizeof(float)), "failed to allocate GPU mem for the output"); 
                                                TF_CHECK_CUDA(cudaMalloc(&d_out_pot_overlap, N * M * sizeof(float)), "failed to allocateGPU mem for the second output"); })
                                .name("allocate_out");

        // create a cudaFlow to run the overlap_kernel. This kernel function runs
        // the overlap calculation on the GPU.
        auto cudaFlow = taskflow.emplace([&]()
                                         {
                                            tf::cudaFlow cf;
                                            // copy the input matrix to the GPU. Copy from the first element in the multi dim vector.
                                            auto copy_in = cf.memcpy(d_in_grid, inputGrid.data(), rows * cols * sizeof(int)).name("copy_in");
                                            auto copy_colSynPerm = cf.memcpy(d_colSynPerm, colSynPerm.data(), N * M * O * P * sizeof(float)).name("copy_colSynPerm");

                                            // launch the kernel function on the GPU.
                                            int threadsPerBlock = 256;
                                            dim3 block(16, 16);   // 256 threads per block. A standard value this can be increased on some GPU models. 
                                            int noOfBlocks = cols * rows / 256;
                                            if ( (cols * rows) % threadsPerBlock) 
                                            {
                                                noOfBlocks++;
                                            }
                                            dim3 grid((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
                                            
                                            // Setup the GPU kernel function to run the overlap calculation.
                                            // Pass in all the parameters needed for the kernel function including the input and output vector memory locations.
                                            // overlap_kernel = name of the kernel function to run, all paraemters after this are the paraemters required for this function.
                                            auto overlap_calc = cf.kernel(grid, block, 0, overlap_kernel, d_in_grid, d_colSynPerm, d_out_overlap, d_out_pot_overlap, rows, cols, N, M, neib_shape.first, neib_shape.second, step.first, step.second, wrap_mode, center_neigh, connected_perm)
                                                                        .name("overlap_calc");

                                            // copy the output matrix back to the host. Copy to the pointer of the first element in the multi dim vector.
                                            auto copy_out = cf.memcpy(out_overlap.data(), d_out_overlap, N * M * sizeof(float) ).name("copy_out"); 
                                            // Set the order of the flow tasks.
                                            overlap_calc.succeed(copy_colSynPerm, copy_in)
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

        return out_overlap;
    }

    //-----------------------------------------------------------------------------
    // STREAMING VERSION OF THE OVERLAP CALCULATION
    // 
    // A streaming version of the calculate_overlap_gpu funciton above.
    // We don't want to be allocating memory each call but efficiently use the same memory for multiple calls to this function
    // so we can process new inputs (of the same size) using the same allocated memory as the last call. 
    // Define GPU memory pointers globally or as class members to persist across multiple calls
    int *strm_d_in_grid = nullptr;
    float *strm_d_colSynPerm = nullptr;
    uint32_t *strm_d_colConBits = nullptr;  // Alternative to strm_d_colSynPerm uses an uint32_t to store a single bit for each synapses indicating if the syn is connected (has a permanence value above the threshold)
    float *strm_d_out_overlap = nullptr;
    float *strm_d_out_pot_overlap = nullptr;

    // Initialization function to allocate GPU memory
    void initialize_gpu_memory(int in_rows, int in_cols, int N, int M, int O, int P, bool optimised) 
    {
        if (!strm_d_in_grid) {
            TF_CHECK_CUDA(cudaMalloc(&strm_d_in_grid, in_rows * in_cols * sizeof(int)), "failed to allocate strm_d_in_grid");
            TF_CHECK_CUDA(cudaMalloc(&strm_d_out_overlap, N * M * sizeof(float)), "failed to allocate strm_d_out_overlap");
            TF_CHECK_CUDA(cudaMalloc(&strm_d_out_pot_overlap, N * M * sizeof(float)), "failed to allocate strm_d_out_pot_overlap");
            // Use either strm_d_colSynPerm or strm_d_colConBits depending on the optimised flag
            if (optimised)
            {
                // Calculate the required size for in_colConBits considering each uint32_t can hold 32 synapses' states
                int synapses_per_col = O * P;
                int total_synapses = N * M * synapses_per_col;
                int in_colConBits_size = (total_synapses + 31) / 32;  // Ensure enough space for all synapses
                TF_CHECK_CUDA(cudaMalloc(&strm_d_colConBits, in_colConBits_size * sizeof(uint32_t)), "failed to allocate strm_d_colConBits");
            } else {
                TF_CHECK_CUDA(cudaMalloc(&strm_d_colSynPerm, N * M * O * P * sizeof(float)), "failed to allocate strm_d_colSynPerm");
            }
        }
    }
    // Cleanup function to free GPU memory, call this after all calls to calculate_overlap_gpu are done
    void cleanup_gpu_memory(bool optimised) {
        TF_CHECK_CUDA(cudaFree(strm_d_in_grid), "failed to free strm_d_in_grid");
        TF_CHECK_CUDA(cudaFree(strm_d_out_overlap), "failed to free strm_d_out_overlap");
        TF_CHECK_CUDA(cudaFree(strm_d_out_pot_overlap), "failed to free strm_d_out_pot_overlap");
        if (optimised) {
            TF_CHECK_CUDA(cudaFree(strm_d_colConBits), "failed to free strm_d_colConBits");
        } else {
            TF_CHECK_CUDA(cudaFree(strm_d_colSynPerm), "failed to free strm_d_colSynPerm");
        }
        strm_d_in_grid = nullptr;
        strm_d_colSynPerm = nullptr;
        strm_d_out_overlap = nullptr;
        strm_d_out_pot_overlap = nullptr;
    }

    void calculate_overlap_gpu_stream(
                            const int width_cortical_cols, const int height_cortical_cols,
                            const std::vector<float> &colSynPerm,
                            const std::pair<int, int> &colSynPerm_shape,
                            const std::vector<int> &inputGrid,
                            const std::pair<int, int> &inputGrid_shape,
                            const std::pair<int, int> &neib_shape,
                            const std::pair<int, int> &neib_step,
                            bool wrap_mode,
                            bool center_neigh,
                            float connected_perm,
                            std::vector<float> &out_overlap, // Function output passed by reference to avoid allocating the output on each call
                            std::vector<float> &out_pot_overlap // Function output passed by reference to avoid allocating the output on each call
                            ) 
    {
        // Assume GPU memory is already allocated and pointers (strm_d_in_grid, etc.) are initialized

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

        // Calculate the dimensions of the output matrix. This is the 2D size of the "cortical columns".
        int N = height_cortical_cols;  // Number of rows in output matrix. This is the height of the "cortical columns".
        int M = width_cortical_cols; // Number of columns in output matrix. This is the width of the "cortical columns".
        // Calculate the dimensions of the neighbourhood matrix (patch) that is stepped over the input matrix for each cortical column.
        int O = neib_shape.first;                                               // Number of rows in each patch connected to a cortical column.
        int P = neib_shape.second;                                              // Number of columns in each patch connected to a cortical column.

        // Their should be one permance value for every synapse connectted to a neighbourhood element in a cortical column.
        // This means the size of the colSynPerm vector should be equal to the size of the output matrix (num cortical columns) times
        // the size of the neighbourhood matrix (neib_shape.first * neib_shape.second). 
        assert(colSynPerm.size() == N * M * O * P);
        assert(colSynPerm_shape.first == N * M);
        assert(colSynPerm_shape.second == O * P);

        // Ensure out_overlap is of the correct size to hold the output of this funciton.
        if (out_overlap.size() != N * M) {
            out_overlap.resize(N * M);
        }
        // Ensure out_pot_overlap is of the correct size to hold the output of this funciton.
        if (out_pot_overlap.size() != N * M) {
            out_pot_overlap.resize(N * M);
        }

        // Create a Taskflow object to manage tasks and their dependencies.
        // There should be one taskflow object for the entire program.
        tf::Taskflow taskflow("calculate_overlap_gpu_steam");
        tf::Executor executor;

        // Emplace tasks for GPU operations
        auto cudaFlow = taskflow.emplace([&]() {
            tf::cudaFlow cf;

            // Copy input data to GPU
            auto copy_in = cf.memcpy(strm_d_in_grid, inputGrid.data(), rows * cols * sizeof(int)).name("copy_in");
            auto copy_colSynPerm = cf.memcpy(strm_d_colSynPerm, colSynPerm.data(), N * M * O * P * sizeof(float)).name("copy_colSynPerm");

            // Launch kernel
            int threadsPerBlock = 256;
            dim3 block(16, 16);
            int noOfBlocks = (cols * rows + threadsPerBlock - 1) / threadsPerBlock;
            if ( (cols * rows) % threadsPerBlock) 
            {
                noOfBlocks++;
            }
            dim3 grid((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
            // Setup the GPU kernel function to run the overlap calculation.
            auto overlap_calc = cf.kernel(grid, block, 0, overlap_kernel, strm_d_in_grid, strm_d_colSynPerm, strm_d_out_overlap, strm_d_out_pot_overlap, rows, cols, N, M, O, P, step.first, step.second, wrap_mode, center_neigh, connected_perm);
            // Copy output data back to host
            auto copy_out = cf.memcpy(out_overlap.data(), strm_d_out_overlap, N * M * sizeof(float));
            auto copy_out_pot = cf.memcpy(out_pot_overlap.data(), strm_d_out_pot_overlap, N * M * sizeof(float));
            // Set the order of the flow tasks.
            overlap_calc.succeed(copy_colSynPerm, copy_in)
                .precede(copy_out, copy_out_pot);
            tf::cudaStream stream;
            cf.run(stream);
            stream.synchronize(); 
        }).name("cudaFlow");

        // Execute the taskflow
        executor.run(taskflow).wait();

    }

    // A different version of the calculate_overlap_gpu_stream function that uses a 1D array of 
    // unsigned integers to store the connection bits for each synapse in the neighbourhood of each cortical column.
    void calculate_overlap_gpu_stream_opt(
        const int width_cortical_cols, const int height_cortical_cols,
        const std::vector<uint32_t> &colConBits,  // This should be a vector of uint32_t where each bit in the array represents the connection bit for a synapse in the neighbourhood.
        const std::vector<int> &inputGrid,
        const std::pair<int, int> &inputGrid_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh,
        std::vector<float> &out_overlap,
        std::vector<float> &out_pot_overlap) 
    {
        // Assume GPU memory is already allocated and pointers (strm_d_in_grid, etc.) are initialized

        const int rows = inputGrid_shape.first;
        const int cols = inputGrid_shape.second;

        if (neib_shape.first > rows || neib_shape.second > cols) {
            throw std::invalid_argument("Neighbourhood shape must not be larger than the input matrix");
        }

        std::pair<int, int> step = neib_step;
        if (step.first == 0 && step.second == 0) {
            step = neib_shape;
        }

        int N = height_cortical_cols;
        int M = width_cortical_cols;
        int O = neib_shape.first;
        int P = neib_shape.second;

        // Check size of colConBits to ensure it's correct
        int expected_colConBits_size = (N * M * O * P + 31) / 32;
        assert(colConBits.size() == expected_colConBits_size);

        if (out_overlap.size() != N * M) {
            out_overlap.resize(N * M);
        }
        if (out_pot_overlap.size() != N * M) {
            out_pot_overlap.resize(N * M);
        }

        tf::Taskflow taskflow("calculate_overlap_gpu_steam");
        tf::Executor executor;

        auto cudaFlow = taskflow.emplace([&]() {
            tf::cudaFlow cf;

            auto copy_in = cf.memcpy(strm_d_in_grid, inputGrid.data(), rows * cols * sizeof(int)).name("copy_in");
            auto copy_colConBits = cf.memcpy(strm_d_colConBits, colConBits.data(), expected_colConBits_size * sizeof(uint32_t)).name("copy_colConBits");

            dim3 block(16, 16);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            auto overlap_calc = cf.kernel(grid, block, 0, overlap_kernel_opt, strm_d_in_grid, strm_d_colConBits, strm_d_out_overlap, strm_d_out_pot_overlap, rows, cols, N, M, O, P, step.first, step.second, wrap_mode, center_neigh);

            auto copy_out = cf.memcpy(out_overlap.data(), strm_d_out_overlap, N * M * sizeof(float)).name("copy_out");
            auto copy_out_pot = cf.memcpy(out_pot_overlap.data(), strm_d_out_pot_overlap, N * M * sizeof(float)).name("copy_out_pot");

            overlap_calc.succeed(copy_in, copy_colConBits)
                        .precede(copy_out, copy_out_pot);

            tf::cudaStream stream;
            cf.run(stream);
            stream.synchronize();
        }).name("cudaFlow");

        executor.run(taskflow).wait();
    }
    // END OF STREAMING VERSION OF THE OVERLAP CALCULATION
    //-----------------------------------------------------------------------------



} // namespace gpu_overlap