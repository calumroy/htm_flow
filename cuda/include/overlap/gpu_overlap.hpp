
#pragma once

#include <vector>
#include <tuple>

#include <taskflow/taskflow.hpp>

namespace gpu_overlap
{
    ///-----------------------------------------------------------------------------
    ///
    /// flattenVector              Flattens a 2D vector into a 1D vector.
    ///
    /// @param[in] vec2D           A 2D vector.
    ///
    /// @return                    A 1D vector.
    ///-----------------------------------------------------------------------------
    std::vector<int> flattenVector(const std::vector<std::vector<int>> &vec2D);
    std::vector<int> flattenVector(const std::vector<std::vector<std::vector<std::vector<int>>>> &vec4D);
    std::vector<std::vector<int>> unflattenVector(const std::vector<int> &vec1D, size_t numRows, size_t numCols);
    std::vector<std::vector<std::vector<std::vector<int>>>> unflattenVector(const std::vector<int> &vec1D, size_t numLayers, size_t numChannels, size_t numRows, size_t numCols);

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
    ///-----------------------------------------------------------------------------
    std::vector<int> gpu_Images2Neibs(
        const std::vector<int> &input,
        const std::pair<int, int> &input_shape,
        const std::pair<int, int> &neib_shape,
        const std::pair<int, int> &neib_step,
        bool wrap_mode,
        bool center_neigh);

    ///-----------------------------------------------------------------------------
    ///
    /// calculate_overlap_gpu   Calculate the overlap scores for a given input.
    ///                         This is how well each column connects to the active input.
    ///                         This is the main function of this class and its purpose.
    ///
    /// @param[in] columns_width        The width of the cortical columns (2D vector where each element is a cortical column that has a potential connection to a "neighbourhood" in the input 2D vector.).
    /// @param[in] columns_height       The height of the cortical columns (2D vector where each element is a cortical column that has a potential connection to a "neighbourhood" in the input 2D vector.).
    /// @param[in] colSynPerm           The synapse permanence values for each column. A 1D vector simulating a 2D vector of floats columns_width_ x columns_height_.
    /// @param[in] colSynPerm_shape     The shape of the colSynPerm vector height then width as a pair of ints. Should be (num_columns, num_pot_syn) = (cortical columns_width_ x cortical columns_height_, neib_shape(0) * neib_shape(1) )
    /// @param[in] inputGrid            The input grid as a 1D vector simulating a 2D vector of ints input_width_ x input_height_.
    /// @param[in] inputGrid_shape      The shape of the inputGrid vector height then width as a pair of ints.
    /// @param[in] neib_shape           The shape of the neighbourhood.
    /// @param[in] neib_step            The step size of the sliding window.
    /// @param[in] wrap_mode            Whether to wrap the patches around the edges if true or if false use padding of zero on the edges.
    /// @param[in] center_neigh         Whether to center the neighbourhood patch around the input element or not.
    /// @param[in] connected_perm       The connected permanence value threshold. If the permanence value is above this value then the synapse is connected.
    ///         
    /// @return                        The overlap scores for each column as a 1D vector simulating a 2D vector of ints columns_width_ x columns_height_.
    ///-----------------------------------------------------------------------------
    std::vector<float> calculate_overlap_gpu(
                               const int columns_width, const int columns_height,
                               const std::vector<float> &colSynPerm,
                               const std::pair<int, int> &colSynPerm_shape,
                               const std::vector<int> &inputGrid,
                               const std::pair<int, int> &inputGrid_shape,
                               const std::pair<int, int> &neib_shape,
                               const std::pair<int, int> &neib_step,
                               bool wrap_mode,
                               bool center_neigh,
                               float connected_perm
                               );

    ///-----------------------------------------------------------------------------
    /// 
    /// initialize_gpu_memory     Initialize the GPU memory for the input and output data for the calculate_overlap_gpu_steam function.
    /// 
    /// @param[in] in_rows          The number of rows in the input data.
    /// @param[in] in_cols          The number of columns in the input data.
    /// @param[in] N                The number of columns in the cortical columns.
    /// @param[in] M                The number of rows in the cortical columns.
    /// @param[in] O                The number of rows in the neighbourhood patch.
    /// @param[in] P                The number of columns in the neighbourhood patch.
    /// @param[in] optimised        Optional param, whether to use the optimised version of the kernel function or not.
    void initialize_gpu_memory(int in_rows, int in_cols, int N, int M, int O, int P, bool optimised = false);
    
    ///-----------------------------------------------------------------------------
    ///
    /// initialize_gpu_memory_sparse    Initialize the GPU memory for the input and output data for the calculate_overlap_gpu_stream_opt_sparse function.
    ///
    /// @param[in] num_active           The number of active elements in the input data.
    /// @param[in] N                    The number of columns in the cortical columns.
    /// @param[in] M                    The number of rows in the cortical columns.
    /// @param[in] O                    The number of rows in the neighbourhood patch.
    /// @param[in] P                    The number of columns in the neighbourhood patch.
    void initialize_gpu_memory_sparse(int num_active, int N, int M, int O, int P);


    ///-----------------------------------------------------------------------------
    ///
    /// cleanup_gpu_memory           Free the GPU memory for the input and output data for the calculate_overlap_gpu_steam function.
    ///
    /// @param[in] optimised          Optional param, whether to use the optimised version of the kernel function or not.
    void cleanup_gpu_memory(bool optimised = false);

    ///-----------------------------------------------------------------------------
    ///
    /// cleanup_gpu_memory_sparse    Free the GPU memory for the input and output data for the calculate_overlap_gpu_stream_opt_sparse function.
    void cleanup_gpu_memory_sparse();

    ///-----------------------------------------------------------------------------
    ///
    /// calculate_overlap_gpu_steam   Calculate the overlap scores for a given input.
    ///                               This is how well each column connects to the active input.
    ///                               This is the main function of this class and its purpose.
    ///                               This function is optimised for streaming data through it. 
    ///                               The input and output sizes should stay the same for each call.
    ///
    /// @param[in] columns_width        The width of the cortical columns (2D vector where each element is a cortical column that has a potential connection to a "neighbourhood" in the input 2D vector.).
    /// @param[in] columns_height       The height of the cortical columns (2D vector where each element is a cortical column that has a potential connection to a "neighbourhood" in the input 2D vector.).
    /// @param[in] colSynPerm           The synapse permanence values for each column. A 1D vector simulating a 2D vector of floats columns_width_ x columns_height_.
    /// @param[in] colSynPerm_shape     The shape of the colSynPerm vector height then width as a pair of ints. Should be (num_columns, num_pot_syn) = (cortical columns_width_ x cortical columns_height_, neib_shape(0) * neib_shape(1) )
    /// @param[in] inputGrid            The input grid as a 1D vector simulating a 2D vector of ints input_width_ x input_height_.
    /// @param[in] inputGrid_shape      The shape of the inputGrid vector height then width as a pair of ints.
    /// @param[in] neib_shape           The shape of the neighbourhood.
    /// @param[in] neib_step            The step size of the sliding window.
    /// @param[in] wrap_mode            Whether to wrap the patches around the edges if true or if false use padding of zero on the edges.
    /// @param[in] center_neigh         Whether to center the neighbourhood patch around the input element or not.
    /// @param[in] connected_perm       The connected permanence value threshold. If the permanence value is above this value then the synapse is connected.
    /// @param[out] out_overlap         The overlap scores for each column as a 1D vector simulating a 2D vector of ints columns_width_ x columns_height_.
    ///                                 This is the output of the function and is passed by reference to avoid allocating the output on each call.
    /// @param[out] out_pot_overlap     The potential overlap scores for each column as a 1D vector simulating a 2D vector of ints columns_width_ x columns_height_.
    ///                                 This is the output of the function and is passed by reference to avoid allocating the output on each call.
    ///                                 The potential overlap scores are the overlap scores before the connected_perm threshold is applied.
    ///         
    /// @return                         Void as the outputs (out_overlap and out_pot_overlap) are passed by reference.
    ///-----------------------------------------------------------------------------
    void calculate_overlap_gpu_stream(
                               const int columns_width, const int columns_height,
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
                               );
    
    // Struct to represent a 2D integer coordinate. 
    // Used as an input type in the calculate_overlap_gpu_stream_opt_sparse function.
    // This struct is equivalent to cuda::int2 struct and is converted to cuda::int2 in the function calculate_overlap_gpu_stream_opt_sparse.
    struct Int2 {
        int x;
        int y;
    };
    //-----------------------------------------------------------------------------
    ///
    /// calculate_overlap_gpu_stream_opt_sparse   Calculate the overlap scores for a given sparse input.
    ///                                          This function calculates how well each cortical column connects to the active inputs.
    ///                                          It's optimized for streaming sparse data, where active inputs are a small subset of the total.
    ///                                          The function is designed for efficient processing with varying active input patterns but
    ///                                          fixed structural parameters (like cortical column dimensions and neighborhood size).
    ///                                          It uses a list of active elements instead of a dense input grid to minimize memory usage
    ///                                          and computation on inactive elements. 
    ///                                          It also uses a colConBits input parameter. This is a vector of uint32_t where each bit in the array represents the connection bit for a synapse in the neighbourhood.
    ///                                          This input parameter is designed to reduce the GPU memory footprint of the connected state of each synpase.
    ///
    /// @param[in] columns_width           The width of the cortical columns (number of columns in the output matrix).
    /// @param[in] columns_height          The height of the cortical columns (number of rows in the output matrix).
    /// @param[in] colConBits              A vector of uint32_t where each bit represents the connection state of a synapse in the neighbourhood.
    /// @param[in] active_grid             A vector of int2 structures representing the active elements' coordinates in the input grid.
    /// @param[in] num_active              The number of active elements in the active_grid.
    /// @param[in] inputGrid_shape         The shape of the full input grid (height, width) as a pair of ints, for context in processing the sparse active elements.
    /// @param[in] neib_shape              The shape of the neighbourhood around each cortical column.
    /// @param[in] neib_step               The step size of the sliding window over the input grid.
    /// @param[in] wrap_mode               Whether to wrap the neighbourhood around the edges (toroidal topology) or not.
    /// @param[in] center_neigh            Whether to center the neighbourhood around each active element or start from the corner.
    /// @param[out] out_overlap            The overlap scores for each column, output as a 1D vector simulating a 2D matrix.
    /// @param[out] out_pot_overlap        The potential overlap scores for each column, output as a 1D vector simulating a 2D matrix.
    ///                                    The potential overlap scores are calculated without considering the synapse permanence values.
    ///        
    /// @return                            Void as the outputs (out_overlap and out_pot_overlap) are passed by reference.
    //-----------------------------------------------------------------------------
    void calculate_overlap_gpu_stream_opt_sparse(
                                            const int columns_width, const int columns_height,
                                            const std::vector<uint32_t> &colConBits,
                                            const std::vector<Int2> &active_grid, // List of active elements as int2 coordinates
                                            int num_active,
                                            const std::pair<int, int> &inputGrid_shape,
                                            const std::pair<int, int> &neib_shape,
                                            const std::pair<int, int> &neib_step,
                                            bool wrap_mode,
                                            bool center_neigh,
                                            std::vector<float> &out_overlap,
                                            std::vector<float> &out_pot_overlap
                                        );

} // namespace gpu_overlap