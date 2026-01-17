#include <htm_flow/region_runtime.hpp>

#include <stdexcept>

namespace htm_flow {

HTMRegionRuntime::HTMRegionRuntime(const HTMRegionConfig& cfg)
    : HTMRegionRuntime(cfg, "HTMRegion") {}

HTMRegionRuntime::HTMRegionRuntime(const HTMRegionConfig& cfg, const std::string& name)
    : region_(std::make_unique<HTMRegion>(cfg, name)),
      name_(name),
      active_layer_idx_(0),
      input_sequence_id_(1),
      line_inputs_(cfg.layers.empty() ? 10 : cfg.layers[0].num_input_cols,
                   cfg.layers.empty() ? 10 : cfg.layers[0].num_input_rows),
      gen_(std::random_device{}()) {
  apply_input_sequence(input_sequence_id_);
}

HTMRegionRuntime::HTMRegionRuntime(std::unique_ptr<HTMRegion> region)
    : region_(std::move(region)),
      name_(region_ ? region_->name() : "HTMRegion"),
      active_layer_idx_(0),
      input_sequence_id_(1),
      line_inputs_(region_ && !region_->config().layers.empty() ? region_->config().layers[0].num_input_cols : 10,
                   region_ && !region_->config().layers.empty() ? region_->config().layers[0].num_input_rows : 10),
      gen_(std::random_device{}()) {
  if (!region_) {
    throw std::invalid_argument("HTMRegionRuntime requires a non-null region.");
  }
  apply_input_sequence(input_sequence_id_);
}

void HTMRegionRuntime::apply_input_sequence(int id) {
  input_sequence_id_ = id;
  switch (id) {
    case 1:
      line_inputs_.setPattern(utilities::MovingLineInputs::Pattern::LeftToRight);
      break;
    case 2:
      line_inputs_.setPattern(utilities::MovingLineInputs::Pattern::RightToLeft);
      break;
    case 3:
      line_inputs_.setPattern(utilities::MovingLineInputs::Pattern::TopToBottom);
      break;
    case 4:
      line_inputs_.setPattern(utilities::MovingLineInputs::Pattern::BottomToTop);
      break;
    default:
      input_sequence_id_ = 1;
      line_inputs_.setPattern(utilities::MovingLineInputs::Pattern::LeftToRight);
      break;
  }
  line_inputs_.setIndex(0);
}

void HTMRegionRuntime::set_input_sequence(int id) {
  if (id == input_sequence_id_) {
    return;
  }
  apply_input_sequence(id);
}

std::vector<htm_gui::InputSequence> HTMRegionRuntime::input_sequences() const {
  return {
      {1, "1: left→right line"},
      {2, "2: right→left line"},
      {3, "3: top→bottom line"},
      {4, "4: bottom→top line"},
  };
}

int HTMRegionRuntime::num_layers() const {
  return region_ ? region_->num_layers() : 0;
}

void HTMRegionRuntime::set_active_layer(int idx) {
  if (idx >= 0 && idx < num_layers()) {
    active_layer_idx_ = idx;
  }
}

std::vector<htm_gui::InputSequence> HTMRegionRuntime::layer_options() const {
  std::vector<htm_gui::InputSequence> options;
  const int n = num_layers();
  options.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    options.push_back({i, "Layer " + std::to_string(i)});
  }
  return options;
}

htm_gui::Snapshot HTMRegionRuntime::snapshot() const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return {};
  }
  return region_->layer(active_layer_idx_).snapshot();
}

void HTMRegionRuntime::step(int n) {
  if (!region_ || n <= 0) {
    return;
  }
  
  for (int i = 0; i < n; ++i) {
    // Generate input and feed to region
    std::vector<int> input = line_inputs_.next(gen_);
    region_->set_input(input);
    region_->step(1);
  }
}

htm_gui::ProximalSynapseQuery HTMRegionRuntime::query_proximal(int column_x, int column_y) const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return {};
  }
  return region_->layer(active_layer_idx_).query_proximal(column_x, column_y);
}

int HTMRegionRuntime::num_segments(int column_x, int column_y, int cell) const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return 0;
  }
  return region_->layer(active_layer_idx_).num_segments(column_x, column_y, cell);
}

htm_gui::DistalSynapseQuery HTMRegionRuntime::query_distal(int column_x, int column_y, int cell, int segment) const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return {};
  }
  return region_->layer(active_layer_idx_).query_distal(column_x, column_y, cell, segment);
}

int HTMRegionRuntime::activation_threshold() const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return 0;
  }
  return region_->layer(active_layer_idx_).activation_threshold();
}

std::string HTMRegionRuntime::name() const {
  return name_ + " (Layer " + std::to_string(active_layer_idx_) + "/" + std::to_string(num_layers()) + ")";
}

}  // namespace htm_flow
