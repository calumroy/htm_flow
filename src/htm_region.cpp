#include <htm_flow/htm_region.hpp>

#include <stdexcept>

namespace htm_flow {

HTMRegion::HTMRegion(const HTMRegionConfig& cfg) : HTMRegion(cfg, "HTMRegion") {}

HTMRegion::HTMRegion(const HTMRegionConfig& cfg, const std::string& name)
    : cfg_(cfg), name_(name) {
  if (cfg_.layers.empty()) {
    throw std::invalid_argument("HTMRegion requires at least one layer configuration.");
  }

  layers_.reserve(cfg_.layers.size());

  // Create layers, adjusting input dimensions for stacking.
  for (std::size_t i = 0; i < cfg_.layers.size(); ++i) {
    HTMLayerConfig layer_cfg = cfg_.layers[i];

    // For layers beyond the first, the input dimensions must match
    // the output dimensions of the previous layer.
    if (i > 0) {
      const HTMLayer& prev_layer = *layers_[i - 1];
      layer_cfg.num_input_rows = prev_layer.output_rows();
      layer_cfg.num_input_cols = prev_layer.output_cols();
    }

    const std::string layer_name = name_ + "_Layer" + std::to_string(i);
    layers_.push_back(std::make_unique<HTMLayer>(layer_cfg, layer_name));
  }

  // Initialize external input buffer for layer 0
  const HTMLayerConfig& layer0_cfg = cfg_.layers[0];
  const std::size_t input_size =
      static_cast<std::size_t>(layer0_cfg.num_input_rows) * static_cast<std::size_t>(layer0_cfg.num_input_cols);
  external_input_ = std::make_shared<std::vector<int>>(input_size, 0);
}

HTMLayer& HTMRegion::layer(int index) {
  if (index < 0 || index >= static_cast<int>(layers_.size())) {
    throw std::out_of_range("Layer index out of range: " + std::to_string(index));
  }
  return *layers_[static_cast<std::size_t>(index)];
}

const HTMLayer& HTMRegion::layer(int index) const {
  if (index < 0 || index >= static_cast<int>(layers_.size())) {
    throw std::out_of_range("Layer index out of range: " + std::to_string(index));
  }
  return *layers_[static_cast<std::size_t>(index)];
}

void HTMRegion::set_input(const std::vector<int>& input) {
  external_input_ = std::make_shared<std::vector<int>>(input);
}

void HTMRegion::set_input(std::shared_ptr<const std::vector<int>> input) {
  if (input) {
    external_input_ = std::make_shared<std::vector<int>>(*input);
  }
}

std::vector<int> HTMRegion::output() const {
  if (layers_.empty()) {
    return {};
  }
  return layers_.back()->output();
}

int HTMRegion::timestep() const {
  if (layers_.empty()) {
    return 0;
  }
  return layers_[0]->timestep();
}

int HTMRegion::output_rows() const {
  if (layers_.empty()) {
    return 0;
  }
  return layers_.back()->output_rows();
}

int HTMRegion::output_cols() const {
  if (layers_.empty()) {
    return 0;
  }
  return layers_.back()->output_cols();
}

void HTMRegion::step(int n) {
  if (n <= 0) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    step_once();
  }
}

void HTMRegion::step_once() {
  if (layers_.empty()) {
    return;
  }

  // Set input to layer 0 from external input
  layers_[0]->set_input(external_input_);

  // Step layer 0
  layers_[0]->step(1);

  // For each subsequent layer: get output from previous layer, set as input, step
  for (std::size_t i = 1; i < layers_.size(); ++i) {
    std::vector<int> prev_output = layers_[i - 1]->output();
    layers_[i]->set_input(prev_output);
    layers_[i]->step(1);
  }
}

void HTMRegion::propagate_inputs() {
  // This is called internally if we need to propagate inputs without stepping.
  // Currently unused but kept for potential future use.
}

}  // namespace htm_flow
