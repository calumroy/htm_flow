#pragma once

#include <vector>

namespace htm_flow {

/// Configuration for a single HTM layer.
///
/// An HTM layer implements the core HTM algorithm consisting of:
/// 1. Spatial Pooling: Learns sparse distributed representations of inputs
///    - Overlap calculation: Measures input similarity to each column's receptive field
///    - Inhibition: Enforces sparse activation via local competition
///    - Spatial learning: Adapts proximal synapses to better match frequent patterns
///
/// 2. Sequence Memory (Temporal Memory):
///    - Active cells: Determines which cells in active columns should fire
///    - Predictive cells: Identifies cells predicting upcoming activity
///    - Sequence learning: Grows/reinforces distal synapses for temporal patterns
///
/// 3. Temporal Pooling: Maintains stable representations across sequence elements
///    - Keeps correctly-predicting cells active longer
///    - Creates invariant representations for entire sequences
///
/// The parameters are organized by which stage they affect.
struct HTMLayerConfig {
  // --------------------------------------------------------------------------
  // Overlap / proximal topology
  // --------------------------------------------------------------------------
  // These control how columns connect to the input space (proximal dendrites).
  // pot_width/height define the receptive field size for each column.
  // --------------------------------------------------------------------------
  int pot_width = 20;   ///< Width of each column's potential pool (receptive field)
  int pot_height = 1;   ///< Height of each column's potential pool
  bool center_pot_synapses = false;  ///< If true, center receptive field on column position

  int num_input_rows = 20;    ///< Height of the input grid
  int num_input_cols = 20;    ///< Width of the input grid
  int num_column_rows = 20;   ///< Height of the column grid
  int num_column_cols = 40;   ///< Width of the column grid

  /// Permanence threshold for a synapse to be considered "connected"
  float connected_perm = 0.3f;
  /// Minimum overlap score required for a column to participate in inhibition
  int min_overlap = 3;
  /// Minimum potential-overlap for bootstrapping (before any learning occurs).
  /// This fallback allows columns to compete even when all permanences start at 0.
  /// Set to 1 (not 0) to avoid degenerate ties when everyone has zero overlap.
  int min_potential_overlap = 1;
  /// Whether input wraps around edges (toroidal topology)
  bool wrap_input = true;

  // --------------------------------------------------------------------------
  // Inhibition
  // --------------------------------------------------------------------------
  // Controls local competition between columns. Only the most active columns
  // within each inhibition neighborhood become active, enforcing sparsity.
  // --------------------------------------------------------------------------
  int inhibition_width = 40;   ///< Width of the inhibition neighborhood
  int inhibition_height = 1;   ///< Height of the inhibition neighborhood
  int desired_local_activity = 1;  ///< Target number of active columns per neighborhood
  bool strict_local_activity = false;  ///< Strictly enforce desired_local_activity count

  // --------------------------------------------------------------------------
  // Spatial learning
  // --------------------------------------------------------------------------
  // Controls how proximal synapses adapt. Active columns strengthen synapses
  // to active inputs and weaken synapses to inactive inputs.
  // --------------------------------------------------------------------------
  float spatial_permanence_inc = 0.05f;  ///< Permanence increase for active synapses
  float spatial_permanence_dec = 0.05f;  ///< Permanence decrease for inactive synapses
  float active_col_permanence_dec = 0.05f;  ///< Extra decrease for already-active columns

  // --------------------------------------------------------------------------
  // Sequence pooler (active/predict/learn)
  // --------------------------------------------------------------------------
  // These parameters control the temporal memory / sequence learning.
  // Cells within columns learn to predict which cells will become active next,
  // enabling sequence recognition and prediction.
  // --------------------------------------------------------------------------
  int cells_per_column = 5;  ///< Number of cells per column (sequence capacity)
  int max_segments_per_cell = 2;  ///< Maximum distal dendrite segments per cell
  int max_synapses_per_segment = 20;  ///< Maximum synapses per segment
  int min_num_syn_threshold = 5;  ///< Minimum synapses for segment activation consideration
  float new_syn_permanence = 0.3f;  ///< Initial permanence for newly created synapses
  float connect_permanence = 0.2f;  ///< Threshold for distal synapse to be "connected"
  int activation_threshold = 6;  ///< Connected active synapses needed to activate a segment

  float sequence_permanence_inc = 0.05f;  ///< Distal synapse permanence increase
  float sequence_permanence_dec = 0.05f;  ///< Distal synapse permanence decrease

  // --------------------------------------------------------------------------
  // Temporal pooler
  // --------------------------------------------------------------------------
  // Controls how the layer maintains stable representations across time.
  // Correctly-predicting cells have their activity extended, creating
  // representations that are stable across entire learned sequences.
  // --------------------------------------------------------------------------
  bool temp_enabled = true;  ///< Enable/disable temporal pooling entirely
  int temp_delay_length = 4;  ///< How many timesteps to extend prediction persistence
  bool temp_enable_persistence = true;  ///< Enable temporal pooling persistence mechanism
  float temp_spatial_permanence_inc = 0.01f;  ///< Proximal learning rate for temporal pooling
  float temp_sequence_permanence_inc = 0.01f;  ///< Distal learning rate for temporal pooling
  float temp_sequence_permanence_dec = 0.01f;  ///< Distal decay rate for inactive synapses in temporal pooling

  // --------------------------------------------------------------------------
  // Runtime options
  // --------------------------------------------------------------------------
  bool log_timings = false;  ///< Log per-stage timing information for profiling
};

/// Configuration for an HTM region (a stack of layers).
///
/// A region represents a cortical column containing multiple layers that form
/// a processing hierarchy. Layer 0 receives external input, and each subsequent
/// layer receives the output (cell activations) of the layer below it.
///
/// This creates increasingly abstract representations:
/// - Layer 0: Learns spatial patterns directly from input
/// - Layer 1+: Learns patterns over the cell activations of lower layers,
///   effectively learning higher-order sequences and abstractions
struct HTMRegionConfig {
  /// Configuration for each layer (index 0 = bottom layer receiving input)
  std::vector<HTMLayerConfig> layers;
  /// Enable top-down feedback from higher layers (experimental)
  bool enable_feedback = false;
};

/// Configuration for an HTM network (multiple regions).
///
/// A network consists of multiple regions that can be organized hierarchically
/// or in parallel. This enables modeling of multiple sensory modalities or
/// more complex hierarchical structures.
struct HTMNetworkConfig {
  /// Configuration for each region
  std::vector<HTMRegionConfig> regions;
};

// ----------------------------------------------------------------------------
// Preset configurations for common scenarios
// ----------------------------------------------------------------------------
// These provide ready-to-use configurations for testing and experimentation.
// Use these as starting points and adjust parameters as needed.
// ----------------------------------------------------------------------------

/// Default layer configuration with reasonable defaults for most use cases.
/// Grid size: 20x20 input, 20x40 columns, 5 cells per column.
HTMLayerConfig default_layer_config();

/// Small/fast configuration optimized for quick unit and integration tests.
/// Uses smaller grids (10x10 input, 10x20 columns) and faster learning rates.
HTMLayerConfig small_test_config();

/// Configuration tuned for temporal pooling experiments.
/// Parameters match the Python temporal pooling test suite for comparison.
/// Smaller cells_per_column (3) and larger max_segments (10).
HTMLayerConfig temporal_pooling_test_config();

/// Helper to create a region with N layers, all using the same configuration.
/// The input dimensions of layer 1+ are automatically adjusted to match
/// the output dimensions of the layer below.
HTMRegionConfig uniform_region_config(int num_layers, const HTMLayerConfig& layer_cfg);

}  // namespace htm_flow
