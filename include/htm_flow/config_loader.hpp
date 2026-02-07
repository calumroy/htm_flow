#pragma once

#include <htm_flow/config.hpp>
#include <string>
#include <vector>

namespace htm_flow {

/// Load an HTMLayerConfig from a YAML file.
/// The YAML should contain layer parameters directly (no "layers" array).
/// @param yaml_path Path to the YAML configuration file.
/// @return The loaded layer configuration.
/// @throws std::runtime_error if the file cannot be read or parsed.
HTMLayerConfig load_layer_config(const std::string& yaml_path);

/// Load an HTMRegionConfig from a YAML file.
/// The YAML should contain a "layers" array with per-layer configurations.
/// @param yaml_path Path to the YAML configuration file.
/// @return The loaded region configuration.
/// @throws std::runtime_error if the file cannot be read or parsed.
HTMRegionConfig load_region_config(const std::string& yaml_path);

/// Save an HTMRegionConfig to a YAML file.
/// Creates a human-readable YAML file with all layer parameters.
/// @param cfg The configuration to save.
/// @param yaml_path Path to write the YAML file.
/// @throws std::runtime_error if the file cannot be written.
void save_region_config(const HTMRegionConfig& cfg, const std::string& yaml_path);

/// List available YAML config files in a directory.
/// @param directory Path to the configs directory.
/// @return Vector of YAML file paths found.
std::vector<std::string> list_config_files(const std::string& directory);

}  // namespace htm_flow
