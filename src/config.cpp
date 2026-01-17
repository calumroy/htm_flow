#include <htm_flow/config.hpp>

namespace htm_flow {

HTMLayerConfig default_layer_config() {
  // Returns the default configuration (all members already have default values)
  return HTMLayerConfig{};
}

HTMLayerConfig small_test_config() {
  HTMLayerConfig cfg;
  
  // Smaller grid sizes for faster tests
  cfg.num_input_rows = 10;
  cfg.num_input_cols = 10;
  cfg.num_column_rows = 10;
  cfg.num_column_cols = 20;
  
  // Smaller potential pool
  cfg.pot_width = 5;
  cfg.pot_height = 1;
  cfg.center_pot_synapses = true;
  
  // Smaller inhibition area
  cfg.inhibition_width = 10;
  cfg.inhibition_height = 1;
  cfg.desired_local_activity = 1;
  
  // Fewer cells/segments for speed
  cfg.cells_per_column = 3;
  cfg.max_segments_per_cell = 2;
  cfg.max_synapses_per_segment = 10;
  
  // Faster learning for quicker convergence in tests
  cfg.spatial_permanence_inc = 0.1f;
  cfg.spatial_permanence_dec = 0.05f;
  cfg.sequence_permanence_inc = 0.1f;
  cfg.sequence_permanence_dec = 0.05f;
  
  cfg.min_overlap = 2;
  cfg.min_potential_overlap = 1;
  
  return cfg;
}

HTMLayerConfig temporal_pooling_test_config() {
  // Configuration matching the Python temporal pooling test suite
  // See: HTM/tests/temporalPooling/test_temporalPoolingSuite2.py (layer 0 config)
  HTMLayerConfig cfg;
  
  // Grid sizes from Python test
  cfg.num_input_rows = 60;  // 2 * columnArrayHeight (30)
  cfg.num_input_cols = 30;  // columnArrayWidth (10) * cellsPerColumn (3)
  cfg.num_column_rows = 30;
  cfg.num_column_cols = 10;
  
  // Overlap / proximal topology
  cfg.pot_width = 4;
  cfg.pot_height = 4;
  cfg.center_pot_synapses = true;
  cfg.wrap_input = true;
  cfg.min_overlap = 3;
  cfg.min_potential_overlap = 1;
  cfg.connected_perm = 0.3f;
  
  // Inhibition
  cfg.inhibition_width = 4;
  cfg.inhibition_height = 2;
  cfg.desired_local_activity = 1;
  cfg.strict_local_activity = false;
  
  // Spatial learning
  cfg.spatial_permanence_inc = 0.1f;
  cfg.spatial_permanence_dec = 0.02f;
  cfg.active_col_permanence_dec = 0.02f;
  
  // Sequence pooler
  cfg.cells_per_column = 3;
  cfg.max_segments_per_cell = 10;
  cfg.max_synapses_per_segment = 10;  // newSynapseCount in Python
  cfg.min_num_syn_threshold = 5;      // minThreshold in Python
  cfg.min_score_threshold = 5;
  cfg.new_syn_permanence = 0.4f;      // cellSynPermanence in Python
  cfg.connect_permanence = 0.3f;
  cfg.activation_threshold = 6;
  cfg.sequence_permanence_inc = 0.1f;   // permanenceInc in Python
  cfg.sequence_permanence_dec = 0.02f;  // permanenceDec in Python
  
  // Temporal pooler
  cfg.temp_delay_length = 3;
  cfg.temp_enable_persistence = true;
  cfg.temp_spatial_permanence_inc = 0.1f;
  cfg.temp_sequence_permanence_inc = 0.1f;
  
  return cfg;
}

HTMRegionConfig uniform_region_config(int num_layers, const HTMLayerConfig& layer_cfg) {
  HTMRegionConfig cfg;
  cfg.layers.reserve(static_cast<std::size_t>(num_layers));
  for (int i = 0; i < num_layers; ++i) {
    cfg.layers.push_back(layer_cfg);
  }
  cfg.enable_feedback = false;
  return cfg;
}

}  // namespace htm_flow
