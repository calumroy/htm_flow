#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <taskflow/taskflow.hpp>

namespace overlap
{
    // Define a function to convert 2D indices to 1D indices
    int flatten_index(int x, int y, int width);

    // Define a function to wrap 2D indices around the input dimensions
    std::tuple<int, int> wrap_indices(int x, int y, int input_width, int input_height);

    // Define a class to calculate the overlap values for columns in a single HTM layer
    class OverlapCalculator
    {
    public:
        // Constructor
        OverlapCalculator(int potential_width, int potential_height, int columns_width, int columns_height,
                          int input_width, int input_height, bool center_pot_synapses, float connected_perm,
                          int min_overlap, bool wrap_input);

        // Calculate the overlap scores for a given input
        std::vector<int> calculate_overlap(const std::vector<std::vector<int>> &input);

    private:
        // Calculate the potential synapses for a given column
        inline std::vector<std::tuple<int, int>> calculate_pot_syn(int column,
                                                                   const std::vector<std::vector<float>> &input_tensor);
        // Calculate the potential overlap scores for a given set of potential synapses
        inline std::vector<float> calculate_pot_overlap(const std::vector<std::tuple<int, int>> &pot_syn,
                                                        const std::vector<std::vector<float>> &input_tensor);
        // Calculate the actual overlap score for a given column based on its potential overlap scores
        inline int calculate_overlap(const std::vector<float> &pot_overlap,
                                     const std::vector<std::tuple<int, int>> &pot_syn);
        // Get the step sizes for the potential synapses
        std::pair<int, int> get_step_sizes(int input_width, int input_height, int columns_width,
                                           int columns_height, int potential_width, int potential_height);
        // Make the potential synapse tie breaker matrix
        void make_pot_syn_tie_breaker(std::vector<std::vector<float>> &pot_syn_tie_breaker);

        // Member variables
        bool center_pot_synapses_;                              // Specifies if the potential synapses are centered over the columns
        bool wrap_input_;                                       // Use a wrap input function instead of padding the input to calcualte the overlap scores
        int potential_width_;                                   // Width of the potential synapses
        int potential_height_;                                  // Height of the potential synapses
        float connected_perm_;                                  // Probability that a column's potential synapses are connected to the input
        int min_overlap_;                                       // Minimum overlap required for a column to be considered for inhibition
        int input_width_;                                       // Width of the input
        int input_height_;                                      // Height of the input
        int columns_width_;                                     // Width of the columns
        int columns_height_;                                    // Height of the columns
        int num_columns_;                                       // Number of columns
        int step_x_;                                            // Step size in the x direction for the potential synapses
        int step_y_;                                            // Step size in the y direction for the potential synapses
        std::mt19937 rng_;                                      // Mersenne Twister random number generator
        std::vector<std::vector<float>> pot_syn_tie_breaker_;   // Potential synapse tie breaker matrix
        std::vector<std::vector<float>> col_input_pot_syn_tie_; // Potential synapse tie breaker matrix
        std::vector<float> col_tie_breaker_;                    // Potential synapse tie breaker matrix
    };

} // namespace overlap