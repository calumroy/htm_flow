#include <htm_flow/config_loader.hpp>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace htm_flow {

namespace {

/// Helper to read a value from YAML node with a default fallback.
template <typename T>
T get_or(const YAML::Node& node, const std::string& key, T default_value) {
  if (node[key]) {
    return node[key].as<T>();
  }
  return default_value;
}

/// Parse a single layer configuration from a YAML node.
HTMLayerConfig parse_layer_node(const YAML::Node& node) {
  HTMLayerConfig cfg;

  // Input dimensions
  if (node["input"]) {
    cfg.num_input_rows = get_or(node["input"], "rows", cfg.num_input_rows);
    cfg.num_input_cols = get_or(node["input"], "cols", cfg.num_input_cols);
  }
  cfg.num_input_rows = get_or(node, "num_input_rows", cfg.num_input_rows);
  cfg.num_input_cols = get_or(node, "num_input_cols", cfg.num_input_cols);

  // Column dimensions
  if (node["columns"]) {
    cfg.num_column_rows = get_or(node["columns"], "rows", cfg.num_column_rows);
    cfg.num_column_cols = get_or(node["columns"], "cols", cfg.num_column_cols);
  }
  cfg.num_column_rows = get_or(node, "num_column_rows", cfg.num_column_rows);
  cfg.num_column_cols = get_or(node, "num_column_cols", cfg.num_column_cols);

  // Overlap / proximal topology
  if (node["overlap"]) {
    const auto& overlap = node["overlap"];
    cfg.pot_width = get_or(overlap, "pot_width", cfg.pot_width);
    cfg.pot_height = get_or(overlap, "pot_height", cfg.pot_height);
    cfg.center_pot_synapses = get_or(overlap, "center_pot_synapses", cfg.center_pot_synapses);
    cfg.connected_perm = get_or(overlap, "connected_perm", cfg.connected_perm);
    cfg.min_overlap = get_or(overlap, "min_overlap", cfg.min_overlap);
    cfg.min_potential_overlap = get_or(overlap, "min_potential_overlap", cfg.min_potential_overlap);
    cfg.wrap_input = get_or(overlap, "wrap_input", cfg.wrap_input);
  }
  // Also support flat keys
  cfg.pot_width = get_or(node, "pot_width", cfg.pot_width);
  cfg.pot_height = get_or(node, "pot_height", cfg.pot_height);
  cfg.center_pot_synapses = get_or(node, "center_pot_synapses", cfg.center_pot_synapses);
  cfg.connected_perm = get_or(node, "connected_perm", cfg.connected_perm);
  cfg.min_overlap = get_or(node, "min_overlap", cfg.min_overlap);
  cfg.min_potential_overlap = get_or(node, "min_potential_overlap", cfg.min_potential_overlap);
  cfg.wrap_input = get_or(node, "wrap_input", cfg.wrap_input);

  // Inhibition
  if (node["inhibition"]) {
    const auto& inh = node["inhibition"];
    cfg.inhibition_width = get_or(inh, "width", cfg.inhibition_width);
    cfg.inhibition_height = get_or(inh, "height", cfg.inhibition_height);
    cfg.desired_local_activity = get_or(inh, "desired_local_activity", cfg.desired_local_activity);
    cfg.strict_local_activity = get_or(inh, "strict_local_activity", cfg.strict_local_activity);
  }
  cfg.inhibition_width = get_or(node, "inhibition_width", cfg.inhibition_width);
  cfg.inhibition_height = get_or(node, "inhibition_height", cfg.inhibition_height);
  cfg.desired_local_activity = get_or(node, "desired_local_activity", cfg.desired_local_activity);
  cfg.strict_local_activity = get_or(node, "strict_local_activity", cfg.strict_local_activity);

  // Spatial learning
  if (node["spatial_learning"]) {
    const auto& sp = node["spatial_learning"];
    cfg.spatial_permanence_inc = get_or(sp, "permanence_inc", cfg.spatial_permanence_inc);
    cfg.spatial_permanence_dec = get_or(sp, "permanence_dec", cfg.spatial_permanence_dec);
    cfg.active_col_permanence_dec = get_or(sp, "active_col_permanence_dec", cfg.active_col_permanence_dec);
  }
  cfg.spatial_permanence_inc = get_or(node, "spatial_permanence_inc", cfg.spatial_permanence_inc);
  cfg.spatial_permanence_dec = get_or(node, "spatial_permanence_dec", cfg.spatial_permanence_dec);
  cfg.active_col_permanence_dec = get_or(node, "active_col_permanence_dec", cfg.active_col_permanence_dec);

  // Sequence pooler
  if (node["sequence_memory"]) {
    const auto& seq = node["sequence_memory"];
    cfg.cells_per_column = get_or(seq, "cells_per_column", cfg.cells_per_column);
    cfg.max_segments_per_cell = get_or(seq, "max_segments_per_cell", cfg.max_segments_per_cell);
    cfg.max_synapses_per_segment = get_or(seq, "max_synapses_per_segment", cfg.max_synapses_per_segment);
    cfg.min_num_syn_threshold = get_or(seq, "min_num_syn_threshold", cfg.min_num_syn_threshold);
    cfg.min_score_threshold = get_or(seq, "min_score_threshold", cfg.min_score_threshold);
    cfg.new_syn_permanence = get_or(seq, "new_syn_permanence", cfg.new_syn_permanence);
    cfg.connect_permanence = get_or(seq, "connect_permanence", cfg.connect_permanence);
    cfg.activation_threshold = get_or(seq, "activation_threshold", cfg.activation_threshold);
    cfg.sequence_permanence_inc = get_or(seq, "permanence_inc", cfg.sequence_permanence_inc);
    cfg.sequence_permanence_dec = get_or(seq, "permanence_dec", cfg.sequence_permanence_dec);
  }
  cfg.cells_per_column = get_or(node, "cells_per_column", cfg.cells_per_column);
  cfg.max_segments_per_cell = get_or(node, "max_segments_per_cell", cfg.max_segments_per_cell);
  cfg.max_synapses_per_segment = get_or(node, "max_synapses_per_segment", cfg.max_synapses_per_segment);
  cfg.min_num_syn_threshold = get_or(node, "min_num_syn_threshold", cfg.min_num_syn_threshold);
  cfg.min_score_threshold = get_or(node, "min_score_threshold", cfg.min_score_threshold);
  cfg.new_syn_permanence = get_or(node, "new_syn_permanence", cfg.new_syn_permanence);
  cfg.connect_permanence = get_or(node, "connect_permanence", cfg.connect_permanence);
  cfg.activation_threshold = get_or(node, "activation_threshold", cfg.activation_threshold);
  cfg.sequence_permanence_inc = get_or(node, "sequence_permanence_inc", cfg.sequence_permanence_inc);
  cfg.sequence_permanence_dec = get_or(node, "sequence_permanence_dec", cfg.sequence_permanence_dec);

  // Temporal pooling
  if (node["temporal_pooling"]) {
    const auto& tp = node["temporal_pooling"];
    cfg.temp_enable_persistence = get_or(tp, "enabled", cfg.temp_enable_persistence);
    cfg.temp_delay_length = get_or(tp, "delay_length", cfg.temp_delay_length);
    cfg.temp_spatial_permanence_inc = get_or(tp, "spatial_permanence_inc", cfg.temp_spatial_permanence_inc);
    cfg.temp_sequence_permanence_inc = get_or(tp, "sequence_permanence_inc", cfg.temp_sequence_permanence_inc);
  }
  cfg.temp_enable_persistence = get_or(node, "temp_enable_persistence", cfg.temp_enable_persistence);
  cfg.temp_delay_length = get_or(node, "temp_delay_length", cfg.temp_delay_length);
  cfg.temp_spatial_permanence_inc = get_or(node, "temp_spatial_permanence_inc", cfg.temp_spatial_permanence_inc);
  cfg.temp_sequence_permanence_inc = get_or(node, "temp_sequence_permanence_inc", cfg.temp_sequence_permanence_inc);

  // Runtime options
  cfg.log_timings = get_or(node, "log_timings", cfg.log_timings);

  return cfg;
}

/// Emit a layer configuration to a YAML emitter.
void emit_layer_node(YAML::Emitter& out, const HTMLayerConfig& cfg, int layer_index) {
  out << YAML::BeginMap;

  out << YAML::Key << "name" << YAML::Value << ("Layer" + std::to_string(layer_index));

  out << YAML::Key << "input" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "rows" << YAML::Value << cfg.num_input_rows;
  out << YAML::Key << "cols" << YAML::Value << cfg.num_input_cols;
  out << YAML::EndMap;

  out << YAML::Key << "columns" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "rows" << YAML::Value << cfg.num_column_rows;
  out << YAML::Key << "cols" << YAML::Value << cfg.num_column_cols;
  out << YAML::EndMap;

  out << YAML::Key << "overlap" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "pot_width" << YAML::Value << cfg.pot_width;
  out << YAML::Key << "pot_height" << YAML::Value << cfg.pot_height;
  out << YAML::Key << "center_pot_synapses" << YAML::Value << cfg.center_pot_synapses;
  out << YAML::Key << "connected_perm" << YAML::Value << cfg.connected_perm;
  out << YAML::Key << "min_overlap" << YAML::Value << cfg.min_overlap;
  out << YAML::Key << "min_potential_overlap" << YAML::Value << cfg.min_potential_overlap;
  out << YAML::Key << "wrap_input" << YAML::Value << cfg.wrap_input;
  out << YAML::EndMap;

  out << YAML::Key << "inhibition" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "width" << YAML::Value << cfg.inhibition_width;
  out << YAML::Key << "height" << YAML::Value << cfg.inhibition_height;
  out << YAML::Key << "desired_local_activity" << YAML::Value << cfg.desired_local_activity;
  out << YAML::Key << "strict_local_activity" << YAML::Value << cfg.strict_local_activity;
  out << YAML::EndMap;

  out << YAML::Key << "spatial_learning" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "permanence_inc" << YAML::Value << cfg.spatial_permanence_inc;
  out << YAML::Key << "permanence_dec" << YAML::Value << cfg.spatial_permanence_dec;
  out << YAML::Key << "active_col_permanence_dec" << YAML::Value << cfg.active_col_permanence_dec;
  out << YAML::EndMap;

  out << YAML::Key << "sequence_memory" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "cells_per_column" << YAML::Value << cfg.cells_per_column;
  out << YAML::Key << "max_segments_per_cell" << YAML::Value << cfg.max_segments_per_cell;
  out << YAML::Key << "max_synapses_per_segment" << YAML::Value << cfg.max_synapses_per_segment;
  out << YAML::Key << "min_num_syn_threshold" << YAML::Value << cfg.min_num_syn_threshold;
  out << YAML::Key << "min_score_threshold" << YAML::Value << cfg.min_score_threshold;
  out << YAML::Key << "new_syn_permanence" << YAML::Value << cfg.new_syn_permanence;
  out << YAML::Key << "connect_permanence" << YAML::Value << cfg.connect_permanence;
  out << YAML::Key << "activation_threshold" << YAML::Value << cfg.activation_threshold;
  out << YAML::Key << "permanence_inc" << YAML::Value << cfg.sequence_permanence_inc;
  out << YAML::Key << "permanence_dec" << YAML::Value << cfg.sequence_permanence_dec;
  out << YAML::EndMap;

  out << YAML::Key << "temporal_pooling" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "enabled" << YAML::Value << cfg.temp_enable_persistence;
  out << YAML::Key << "delay_length" << YAML::Value << cfg.temp_delay_length;
  out << YAML::Key << "spatial_permanence_inc" << YAML::Value << cfg.temp_spatial_permanence_inc;
  out << YAML::Key << "sequence_permanence_inc" << YAML::Value << cfg.temp_sequence_permanence_inc;
  out << YAML::EndMap;

  out << YAML::EndMap;
}

}  // namespace

HTMLayerConfig load_layer_config(const std::string& yaml_path) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    return parse_layer_node(root);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load layer config from '" + yaml_path + "': " + e.what());
  }
}

HTMRegionConfig load_region_config(const std::string& yaml_path) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    HTMRegionConfig cfg;

    cfg.enable_feedback = get_or(root, "enable_feedback", false);

    if (root["layers"] && root["layers"].IsSequence()) {
      for (const auto& layer_node : root["layers"]) {
        cfg.layers.push_back(parse_layer_node(layer_node));
      }
    } else {
      // Single layer config (no layers array)
      cfg.layers.push_back(parse_layer_node(root));
    }

    if (cfg.layers.empty()) {
      throw std::runtime_error("Config must have at least one layer");
    }

    return cfg;
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load region config from '" + yaml_path + "': " + e.what());
  }
}

void save_region_config(const HTMRegionConfig& cfg, const std::string& yaml_path) {
  YAML::Emitter out;
  out << YAML::BeginMap;

  out << YAML::Key << "enable_feedback" << YAML::Value << cfg.enable_feedback;

  out << YAML::Key << "layers" << YAML::Value << YAML::BeginSeq;
  for (std::size_t i = 0; i < cfg.layers.size(); ++i) {
    emit_layer_node(out, cfg.layers[i], static_cast<int>(i));
  }
  out << YAML::EndSeq;

  out << YAML::EndMap;

  std::ofstream fout(yaml_path);
  if (!fout) {
    throw std::runtime_error("Failed to open file for writing: " + yaml_path);
  }
  fout << out.c_str();
}

std::vector<std::string> list_config_files(const std::string& directory) {
  std::vector<std::string> files;
  if (!std::filesystem::exists(directory)) {
    return files;
  }
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      const auto& path = entry.path();
      if (path.extension() == ".yaml" || path.extension() == ".yml") {
        files.push_back(path.string());
      }
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

}  // namespace htm_flow
