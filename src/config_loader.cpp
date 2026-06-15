#include <htm_flow/config_loader.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
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

std::string join_lines(const std::vector<std::string>& lines) {
  std::ostringstream out;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i != 0) {
      out << '\n';
    }
    out << lines[i];
  }
  return out.str();
}

bool has_key(const std::vector<std::string>& allowed, const std::string& key) {
  return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
}

void reject_unknown_keys(const YAML::Node& node,
                         const std::vector<std::string>& allowed,
                         const std::string& path,
                         std::vector<std::string>& errors) {
  if (!node || !node.IsMap()) {
    return;
  }
  for (const auto& entry : node) {
    const std::string key = entry.first.as<std::string>();
    if (!has_key(allowed, key)) {
      errors.push_back(path + "." + key + " is not hot-swappable.");
    }
  }
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
  cfg.new_syn_permanence = get_or(node, "new_syn_permanence", cfg.new_syn_permanence);
  cfg.connect_permanence = get_or(node, "connect_permanence", cfg.connect_permanence);
  cfg.activation_threshold = get_or(node, "activation_threshold", cfg.activation_threshold);
  cfg.sequence_permanence_inc = get_or(node, "sequence_permanence_inc", cfg.sequence_permanence_inc);
  cfg.sequence_permanence_dec = get_or(node, "sequence_permanence_dec", cfg.sequence_permanence_dec);

  // Temporal pooling
  if (node["temporal_pooling"]) {
    const auto& tp = node["temporal_pooling"];
    cfg.temp_enabled = get_or(tp, "enabled", cfg.temp_enabled);
    cfg.temp_enable_persistence = get_or(tp, "enable_persistence", cfg.temp_enable_persistence);
    cfg.temp_delay_length = get_or(tp, "delay_length", cfg.temp_delay_length);
    cfg.temp_spatial_permanence_inc = get_or(tp, "spatial_permanence_inc", cfg.temp_spatial_permanence_inc);
    cfg.temp_active_predict_proximal_scale =
        get_or(tp, "active_predict_proximal_scale", cfg.temp_active_predict_proximal_scale);
    cfg.temp_post_active_proximal_scale =
        get_or(tp, "post_active_proximal_scale", cfg.temp_post_active_proximal_scale);
    cfg.temp_sequence_permanence_inc = get_or(tp, "sequence_permanence_inc", cfg.temp_sequence_permanence_inc);
    cfg.temp_sequence_permanence_dec = get_or(tp, "sequence_permanence_dec", cfg.temp_sequence_permanence_dec);
  }
  cfg.temp_enabled = get_or(node, "temp_enabled", cfg.temp_enabled);
  cfg.temp_enable_persistence = get_or(node, "temp_enable_persistence", cfg.temp_enable_persistence);
  cfg.temp_delay_length = get_or(node, "temp_delay_length", cfg.temp_delay_length);
  cfg.temp_spatial_permanence_inc = get_or(node, "temp_spatial_permanence_inc", cfg.temp_spatial_permanence_inc);
  cfg.temp_active_predict_proximal_scale =
      get_or(node, "temp_active_predict_proximal_scale", cfg.temp_active_predict_proximal_scale);
  cfg.temp_post_active_proximal_scale =
      get_or(node, "temp_post_active_proximal_scale", cfg.temp_post_active_proximal_scale);
  cfg.temp_sequence_permanence_inc = get_or(node, "temp_sequence_permanence_inc", cfg.temp_sequence_permanence_inc);
  cfg.temp_sequence_permanence_dec = get_or(node, "temp_sequence_permanence_dec", cfg.temp_sequence_permanence_dec);

  // Runtime options
  cfg.log_timings = get_or(node, "log_timings", cfg.log_timings);
  if (node["random_seed"]) {
    cfg.random_seed = node["random_seed"].as<std::uint32_t>();
  }

  return cfg;
}

HTMLayerRuntimePatch parse_runtime_layer_patch_node(const YAML::Node& node,
                                                    const std::string& path,
                                                    std::vector<std::string>& errors) {
  HTMLayerRuntimePatch patch;
  if (!node || node.IsNull()) {
    return patch;
  }
  if (!node.IsMap()) {
    errors.push_back(path + " must be a map.");
    return patch;
  }

  reject_unknown_keys(node,
                      {"overlap", "spatial_learning", "sequence_memory",
                       "temporal_pooling", "log_timings"},
                      path,
                      errors);

  if (node["overlap"]) {
    const auto& overlap = node["overlap"];
    if (!overlap.IsMap()) {
      errors.push_back(path + ".overlap must be a map.");
    } else {
      reject_unknown_keys(overlap,
                          {"connected_perm", "min_overlap", "min_potential_overlap"},
                          path + ".overlap",
                          errors);
      if (overlap["connected_perm"]) {
        patch.connected_perm = overlap["connected_perm"].as<float>();
      }
      if (overlap["min_overlap"]) {
        patch.min_overlap = overlap["min_overlap"].as<int>();
      }
      if (overlap["min_potential_overlap"]) {
        patch.min_potential_overlap = overlap["min_potential_overlap"].as<int>();
      }
    }
  }

  if (node["spatial_learning"]) {
    const auto& spatial_learning = node["spatial_learning"];
    if (!spatial_learning.IsMap()) {
      errors.push_back(path + ".spatial_learning must be a map.");
    } else {
      reject_unknown_keys(spatial_learning,
                          {"permanence_inc", "permanence_dec", "active_col_permanence_dec"},
                          path + ".spatial_learning",
                          errors);
      if (spatial_learning["permanence_inc"]) {
        patch.spatial_permanence_inc = spatial_learning["permanence_inc"].as<float>();
      }
      if (spatial_learning["permanence_dec"]) {
        patch.spatial_permanence_dec = spatial_learning["permanence_dec"].as<float>();
      }
      if (spatial_learning["active_col_permanence_dec"]) {
        patch.active_col_permanence_dec =
            spatial_learning["active_col_permanence_dec"].as<float>();
      }
    }
  }

  if (node["sequence_memory"]) {
    const auto& sequence_memory = node["sequence_memory"];
    if (!sequence_memory.IsMap()) {
      errors.push_back(path + ".sequence_memory must be a map.");
    } else {
      reject_unknown_keys(sequence_memory,
                          {"min_num_syn_threshold", "new_syn_permanence",
                           "connect_permanence", "activation_threshold",
                           "permanence_inc", "permanence_dec"},
                          path + ".sequence_memory",
                          errors);
      if (sequence_memory["min_num_syn_threshold"]) {
        patch.min_num_syn_threshold =
            sequence_memory["min_num_syn_threshold"].as<int>();
      }
      if (sequence_memory["new_syn_permanence"]) {
        patch.new_syn_permanence = sequence_memory["new_syn_permanence"].as<float>();
      }
      if (sequence_memory["connect_permanence"]) {
        patch.connect_permanence = sequence_memory["connect_permanence"].as<float>();
      }
      if (sequence_memory["activation_threshold"]) {
        patch.activation_threshold = sequence_memory["activation_threshold"].as<int>();
      }
      if (sequence_memory["permanence_inc"]) {
        patch.sequence_permanence_inc = sequence_memory["permanence_inc"].as<float>();
      }
      if (sequence_memory["permanence_dec"]) {
        patch.sequence_permanence_dec = sequence_memory["permanence_dec"].as<float>();
      }
    }
  }

  if (node["temporal_pooling"]) {
    const auto& temporal_pooling = node["temporal_pooling"];
    if (!temporal_pooling.IsMap()) {
      errors.push_back(path + ".temporal_pooling must be a map.");
    } else {
      reject_unknown_keys(temporal_pooling,
                          {"enabled", "enable_persistence", "delay_length",
                           "spatial_permanence_inc",
                           "active_predict_proximal_scale",
                           "post_active_proximal_scale",
                           "sequence_permanence_inc", "sequence_permanence_dec"},
                          path + ".temporal_pooling",
                          errors);
      if (temporal_pooling["enabled"]) {
        patch.temp_enabled = temporal_pooling["enabled"].as<bool>();
      }
      if (temporal_pooling["enable_persistence"]) {
        patch.temp_enable_persistence =
            temporal_pooling["enable_persistence"].as<bool>();
      }
      if (temporal_pooling["delay_length"]) {
        patch.temp_delay_length = temporal_pooling["delay_length"].as<int>();
      }
      if (temporal_pooling["spatial_permanence_inc"]) {
        patch.temp_spatial_permanence_inc =
            temporal_pooling["spatial_permanence_inc"].as<float>();
      }
      if (temporal_pooling["active_predict_proximal_scale"]) {
        patch.temp_active_predict_proximal_scale =
            temporal_pooling["active_predict_proximal_scale"].as<float>();
      }
      if (temporal_pooling["post_active_proximal_scale"]) {
        patch.temp_post_active_proximal_scale =
            temporal_pooling["post_active_proximal_scale"].as<float>();
      }
      if (temporal_pooling["sequence_permanence_inc"]) {
        patch.temp_sequence_permanence_inc =
            temporal_pooling["sequence_permanence_inc"].as<float>();
      }
      if (temporal_pooling["sequence_permanence_dec"]) {
        patch.temp_sequence_permanence_dec =
            temporal_pooling["sequence_permanence_dec"].as<float>();
      }
    }
  }

  if (node["log_timings"]) {
    patch.log_timings = node["log_timings"].as<bool>();
  }

  return patch;
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
  out << YAML::Key << "new_syn_permanence" << YAML::Value << cfg.new_syn_permanence;
  out << YAML::Key << "connect_permanence" << YAML::Value << cfg.connect_permanence;
  out << YAML::Key << "activation_threshold" << YAML::Value << cfg.activation_threshold;
  out << YAML::Key << "permanence_inc" << YAML::Value << cfg.sequence_permanence_inc;
  out << YAML::Key << "permanence_dec" << YAML::Value << cfg.sequence_permanence_dec;
  out << YAML::EndMap;

  out << YAML::Key << "temporal_pooling" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "enabled" << YAML::Value << cfg.temp_enabled;
  out << YAML::Key << "enable_persistence" << YAML::Value << cfg.temp_enable_persistence;
  out << YAML::Key << "delay_length" << YAML::Value << cfg.temp_delay_length;
  out << YAML::Key << "spatial_permanence_inc" << YAML::Value << cfg.temp_spatial_permanence_inc;
  out << YAML::Key << "active_predict_proximal_scale" << YAML::Value << cfg.temp_active_predict_proximal_scale;
  out << YAML::Key << "post_active_proximal_scale" << YAML::Value << cfg.temp_post_active_proximal_scale;
  out << YAML::Key << "sequence_permanence_inc" << YAML::Value << cfg.temp_sequence_permanence_inc;
  out << YAML::Key << "sequence_permanence_dec" << YAML::Value << cfg.temp_sequence_permanence_dec;
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

HTMRegionRuntimePatch load_runtime_patch(const std::string& yaml_path) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    HTMRegionRuntimePatch patch;
    std::vector<std::string> errors;

    if (!root || !root.IsMap()) {
      throw std::runtime_error("Runtime patch root must be a map.");
    }

    if (root["layers"]) {
      reject_unknown_keys(root, {"layers"}, "root", errors);
      if (!root["layers"].IsSequence()) {
        errors.push_back("root.layers must be a sequence.");
      } else {
        for (std::size_t i = 0; i < root["layers"].size(); ++i) {
          patch.layers.push_back(parse_runtime_layer_patch_node(
              root["layers"][i], "layers[" + std::to_string(i) + "]", errors));
        }
      }
    } else {
      patch.layers.push_back(parse_runtime_layer_patch_node(root, "layers[0]", errors));
    }

    if (!errors.empty()) {
      throw std::runtime_error(join_lines(errors));
    }
    if (patch.empty()) {
      throw std::runtime_error("Runtime patch does not contain any hot-swappable parameters.");
    }
    return patch;
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load runtime patch from '" + yaml_path + "': " +
                             e.what());
  }
}

std::vector<RuntimeParameterScheduleEntry> load_runtime_parameter_schedule(
    const std::string& yaml_path) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    std::vector<RuntimeParameterScheduleEntry> schedule;
    if (!root || !root.IsMap() || !root["runtime_parameter_schedule"]) {
      return schedule;
    }

    const auto& node = root["runtime_parameter_schedule"];
    if (!node.IsSequence()) {
      throw std::runtime_error("runtime_parameter_schedule must be a sequence.");
    }

    const std::filesystem::path base_dir =
        std::filesystem::path(yaml_path).parent_path();
    for (std::size_t i = 0; i < node.size(); ++i) {
      const auto& entry = node[i];
      std::vector<std::string> errors;
      if (!entry.IsMap()) {
        errors.push_back("runtime_parameter_schedule[" + std::to_string(i) +
                         "] must be a map.");
      } else {
        reject_unknown_keys(entry, {"at_timestep", "override"},
                            "runtime_parameter_schedule[" + std::to_string(i) + "]",
                            errors);
      }
      if (!errors.empty()) {
        throw std::runtime_error(join_lines(errors));
      }

      if (!entry["at_timestep"] || !entry["override"]) {
        throw std::runtime_error("runtime_parameter_schedule[" + std::to_string(i) +
                                 "] requires both at_timestep and override.");
      }

      RuntimeParameterScheduleEntry item;
      item.at_timestep = entry["at_timestep"].as<int>();
      if (item.at_timestep < 0) {
        throw std::runtime_error("runtime_parameter_schedule[" + std::to_string(i) +
                                 "].at_timestep must be >= 0.");
      }

      std::filesystem::path override_path = entry["override"].as<std::string>();
      if (override_path.is_relative()) {
        override_path = base_dir / override_path;
      }
      item.override_path = override_path.lexically_normal().string();
      schedule.push_back(std::move(item));
    }

    std::stable_sort(schedule.begin(), schedule.end(),
                     [](const RuntimeParameterScheduleEntry& a,
                        const RuntimeParameterScheduleEntry& b) {
                       return a.at_timestep < b.at_timestep;
                     });
    return schedule;
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load runtime schedule from '" + yaml_path +
                             "': " + e.what());
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

std::string format_runtime_patch_report(const RuntimePatchReport& report) {
  std::ostringstream out;
  if (report.applied.empty()) {
    out << "Applied: none";
  } else {
    out << "Applied: ";
    for (std::size_t i = 0; i < report.applied.size(); ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << report.applied[i];
    }
  }

  if (!report.rejected.empty()) {
    out << " | Rejected: ";
    for (std::size_t i = 0; i < report.rejected.size(); ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << report.rejected[i];
    }
  }
  return out.str();
}

}  // namespace htm_flow
