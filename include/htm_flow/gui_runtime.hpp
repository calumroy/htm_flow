#pragma once

#include <htm_flow/config.hpp>
#include <htm_flow/region_runtime.hpp>

namespace htm_flow {

/// Convenience wrapper for running a single HTM layer with the GUI.
///
/// This is a thin wrapper around HTMRegionRuntime with a single layer,
/// provided for backward compatibility with main.cpp.
///
/// For new code, prefer using HTMRegionRuntime directly:
/// @code
///   HTMRegionConfig cfg;
///   cfg.layers.push_back(my_layer_config);
///   HTMRegionRuntime runtime(cfg, "MyNetwork");
///   htm_gui::run_debugger(argc, argv, runtime);
/// @endcode
class HtmFlowRuntime final : public htm_gui::IHtmRuntime {
public:
  using Config = HTMLayerConfig;

  HtmFlowRuntime();
  explicit HtmFlowRuntime(const Config& cfg);

  // --- IHtmRuntime (delegates to runtime_) ---
  htm_gui::Snapshot snapshot() const override { return runtime_.snapshot(); }
  void step(int n = 1) override { runtime_.step(n); }
  htm_gui::ProximalSynapseQuery query_proximal(int column_x, int column_y) const override {
    return runtime_.query_proximal(column_x, column_y);
  }
  int num_segments(int column_x, int column_y, int cell) const override {
    return runtime_.num_segments(column_x, column_y, cell);
  }
  htm_gui::DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const override {
    return runtime_.query_distal(column_x, column_y, cell, segment);
  }
  std::vector<htm_gui::InputSequence> input_sequences() const override { return runtime_.input_sequences(); }
  int input_sequence() const override { return runtime_.input_sequence(); }
  void set_input_sequence(int id) override { runtime_.set_input_sequence(id); }
  int activation_threshold() const override { return runtime_.activation_threshold(); }
  std::string name() const override { return "htm_flow"; }

  // --- Access to underlying region ---
  HTMRegion& region() { return runtime_.region(); }
  const HTMRegion& region() const { return runtime_.region(); }

private:
  HTMRegionRuntime runtime_;
};

}  // namespace htm_flow
