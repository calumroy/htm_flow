#include <htm_flow/gui_runtime.hpp>

namespace htm_flow {

namespace {
/// Create a single-layer region config from a layer config.
HTMRegionConfig single_layer_region(const HTMLayerConfig& layer_cfg) {
  HTMRegionConfig cfg;
  cfg.layers.push_back(layer_cfg);
  return cfg;
}
}  // namespace

HtmFlowRuntime::HtmFlowRuntime(const Config& cfg)
    : runtime_(single_layer_region(cfg), "htm_flow") {}

HtmFlowRuntime::HtmFlowRuntime() : HtmFlowRuntime(Config{}) {}

}  // namespace htm_flow
