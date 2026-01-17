#pragma once

#include <memory>
#include <string>
#include <vector>

#include <htm_gui/runtime.hpp>

#include <htm_flow/config.hpp>
#include <htm_flow/htm_region.hpp>
#include <utilities/sdr_inputs.hpp>

namespace htm_flow {

/// GUI adapter for visualizing and debugging multi-layer HTMRegion instances.
///
/// This class wraps an HTMRegion and implements the IHtmRuntime interface,
/// enabling the Qt debugger GUI to visualize any layer within the region.
///
/// Key features:
/// - **Layer selection**: The GUI can switch between visualizing different layers
///   using set_active_layer(). All IHtmRuntime queries (snapshot, synapses, etc.)
///   are delegated to the currently selected layer.
/// - **Unified stepping**: step() advances the entire region (all layers together),
///   maintaining proper data flow between layers.
/// - **Input generation**: Includes built-in MovingLineInputs for testing/demo,
///   with selectable patterns (left-right, right-left, top-bottom, bottom-top).
///
/// Use this when you want to visualize a multi-layer hierarchy in the GUI.
/// For single-layer visualization, you can use HTMLayer directly (it also
/// implements IHtmRuntime).
///
/// Example:
/// @code
///   HTMRegionConfig cfg = uniform_region_config(3, small_test_config());
///   HTMRegionRuntime runtime(cfg, "MyRegion");
///
///   runtime.set_active_layer(1);  // View layer 1
///   htm_gui::run_debugger(argc, argv, runtime);
/// @endcode
class HTMRegionRuntime : public htm_gui::IHtmRuntime {
public:
  /// Create a runtime with the given region configuration.
  explicit HTMRegionRuntime(const HTMRegionConfig& cfg);
  HTMRegionRuntime(const HTMRegionConfig& cfg, const std::string& name);

  /// Create a runtime from an existing region (takes ownership).
  explicit HTMRegionRuntime(std::unique_ptr<HTMRegion> region);

  // --- IHtmRuntime interface (delegates to active layer) ---
  htm_gui::Snapshot snapshot() const override;
  void step(int n = 1) override;
  htm_gui::ProximalSynapseQuery query_proximal(int column_x, int column_y) const override;
  int num_segments(int column_x, int column_y, int cell) const override;
  htm_gui::DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const override;
  std::vector<htm_gui::InputSequence> input_sequences() const override;
  int input_sequence() const override { return input_sequence_id_; }
  void set_input_sequence(int id) override;
  int activation_threshold() const override;
  std::string name() const override;

  // --- Layer selection for GUI ---
  int active_layer() const { return active_layer_idx_; }
  void set_active_layer(int idx);
  int num_layers() const;

  /// Get layer selection options for GUI dropdown.
  std::vector<htm_gui::InputSequence> layer_options() const;

  // --- Region access ---
  HTMRegion& region() { return *region_; }
  const HTMRegion& region() const { return *region_; }

private:
  void apply_input_sequence(int id);

  std::unique_ptr<HTMRegion> region_;
  std::string name_{"HTMRegion"};
  int active_layer_idx_{0};

  // Input generation
  int input_sequence_id_{1};
  utilities::MovingLineInputs line_inputs_;
  mutable std::mt19937 gen_;
};

}  // namespace htm_flow
