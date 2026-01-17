#pragma once

#include <memory>
#include <string>
#include <vector>

#include <htm_flow/config.hpp>
#include <htm_flow/htm_layer.hpp>

namespace htm_flow {

/// An HTM region consisting of multiple stacked layers forming a cortical hierarchy.
///
/// In biological terms, a region represents a vertical slice through the neocortex,
/// containing multiple layers (analogous to cortical layers L2-L6). Each layer
/// processes increasingly abstract representations of the input:
///
/// - Layer 0 (bottom): Receives raw sensory input and learns spatial patterns
/// - Higher layers: Receive the cell activations from the layer below, learning
///   more abstract/temporal patterns over larger receptive fields
///
/// This hierarchical structure enables:
/// 1. Temporal pooling: Higher layers maintain stable representations across
///    varying inputs, recognizing sequences rather than individual frames
/// 2. Abstraction: Each layer learns patterns over the output of the layer below,
///    creating increasingly invariant representations
/// 3. Prediction: Feedback connections (when enabled) allow higher layers to
///    prime lower layers with contextual expectations
///
/// Example usage:
/// @code
///   HTMRegionConfig cfg;
///   cfg.layers.push_back(layer0_config);  // Sensory layer
///   cfg.layers.push_back(layer1_config);  // Abstraction layer
///   HTMRegion region(cfg);
///
///   region.set_input(sensory_data);
///   region.step(1);
///   auto pooled_output = region.output();
/// @endcode
class HTMRegion {
public:
  explicit HTMRegion(const HTMRegionConfig& cfg);
  HTMRegion(const HTMRegionConfig& cfg, const std::string& name);

  /// Advance all layers by n steps.
  /// Each step: set external input -> step layer 0 -> pass output to layer 1 -> step layer 1 -> ...
  void step(int n = 1);

  /// Get the number of layers in this region.
  int num_layers() const { return static_cast<int>(layers_.size()); }

  /// Get a specific layer by index.
  HTMLayer& layer(int index);
  const HTMLayer& layer(int index) const;

  /// Set external input to the bottom layer (layer 0).
  void set_input(const std::vector<int>& input);
  void set_input(std::shared_ptr<const std::vector<int>> input);

  /// Get the output of the top layer (for passing to the next region or for analysis).
  std::vector<int> output() const;

  /// Get the region configuration.
  const HTMRegionConfig& config() const { return cfg_; }

  /// Get the region name.
  const std::string& name() const { return name_; }

  /// Get the current timestep (same across all layers).
  int timestep() const;

  /// Output dimensions (from the top layer).
  int output_rows() const;
  int output_cols() const;

private:
  void step_once();
  void propagate_inputs();

  HTMRegionConfig cfg_;
  std::string name_{"HTMRegion"};
  std::vector<std::unique_ptr<HTMLayer>> layers_;

  // Buffer for external input to layer 0
  std::shared_ptr<std::vector<int>> external_input_;
};

}  // namespace htm_flow
