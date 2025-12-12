#include <htm_flow/inhibition.hpp>
#include <htm_flow/overlap.hpp>
#include <htm_flow/overlap_utils.hpp>
#include <htm_flow/spatiallearn.hpp>
#include <htm_flow/temporal/temporal_memory.hpp>
#include <utilities/logger.hpp>
#include <utilities/stopwatch.hpp>

#ifdef HTM_FLOW_WITH_GUI
#include <htm_gui/debugger.hpp>
#endif

#include <cstdlib>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr int NUM_ITERATIONS = 3;

}  // namespace

#ifdef HTM_FLOW_WITH_GUI
namespace {

class HtmFlowRuntime final : public htm_gui::IHtmRuntime {
public:
  HtmFlowRuntime(int pot_width,
                 int pot_height,
                 bool center_pot_synapses,
                 int num_input_rows,
                 int num_input_cols,
                 int num_column_rows,
                 int num_column_cols,
                 float connected_perm,
                 int min_overlap,
                 bool wrap_input,
                 int inhibition_width,
                 int inhibition_height,
                 int desired_local_activity,
                 bool strict_local_activity,
                 float spatialPermanenceInc,
                 float spatialPermanenceDec,
                 float activeColPermanenceDec,
                 temporal::TMParams tm_params)
      : pot_width_(pot_width),
        pot_height_(pot_height),
        center_pot_synapses_(center_pot_synapses),
        input_rows_(num_input_rows),
        input_cols_(num_input_cols),
        col_rows_(num_column_rows),
        col_cols_(num_column_cols),
        connected_perm_(connected_perm),
        min_overlap_(min_overlap),
        wrap_input_(wrap_input),
        num_pot_syn_(pot_width * pot_height),
        num_columns_(num_column_rows * num_column_cols),
        col_syn_perm_(num_columns_ * num_pot_syn_),
        col_syn_perm_shape_{num_columns_, num_pot_syn_},
        input_(std::make_shared<std::vector<int>>(input_rows_ * input_cols_)),
        input_shape_{input_rows_, input_cols_},
        colOverlapGridShape_{col_rows_, col_cols_},
        potColOverlapGrid_(num_columns_, 0.0f),
        potColOverlapGridShape_{col_rows_, col_cols_},
        overlapCalc_(pot_width,
                     pot_height,
                     col_cols_,
                     col_rows_,
                     input_cols_,
                     input_rows_,
                     center_pot_synapses_,
                     connected_perm_,
                     min_overlap_,
                     wrap_input_),
        inhibitionCalc_(col_cols_,
                        col_rows_,
                        inhibition_width,
                        inhibition_height,
                        desired_local_activity,
                        min_overlap_,
                        center_pot_synapses_,
                        wrap_input_,
                        strict_local_activity),
        spatialLearnCalc_(num_columns_, num_pot_syn_, spatialPermanenceInc, spatialPermanenceDec, activeColPermanenceDec),
        tm_connected_perm_(tm_params.connected_perm),
        tm_(col_cols_, col_rows_, tm_params) {
    std::random_device rd;
    rng_ = std::mt19937(rd());

    // Initialize permanence randomly for now.
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : col_syn_perm_) {
      v = dis(rng_);
    }

    // Precompute step sizes for proximal synapse coordinate mapping.
    const auto step = overlap_utils::get_step_sizes(input_cols_, input_rows_, col_cols_, col_rows_, pot_width_, pot_height_);
    step_x_ = step.first;
    step_y_ = step.second;
  }

  std::string name() const override { return "htm_flow (Qt debugger)"; }

  htm_gui::Snapshot snapshot() const override {
    htm_gui::Snapshot s;
    s.timestep = iteration_;
    s.input_shape = htm_gui::GridShape{input_rows_, input_cols_};
    s.columns_shape = htm_gui::GridShape{col_rows_, col_cols_};
    s.cells_per_column = tm_.cells_per_column();
    s.input = std::static_pointer_cast<const std::vector<int>>(input_);
    s.active_column_indices = activeColIndices_;

    s.column_cell_masks.resize(num_columns_);
    const auto& tm_masks = tm_.column_masks();
    for (int i = 0; i < num_columns_; ++i) {
      s.column_cell_masks[i] = htm_gui::ColumnCellMasks{tm_masks[i].active, tm_masks[i].predictive, tm_masks[i].learning};
    }

    return s;
  }

  void step(int n = 1) override {
    for (int i = 0; i < n; ++i) {
      ++iteration_;

      // New random input each iteration (placeholder for real sensory input).
      std::uniform_int_distribution<int> dis2(0, 1);
      for (auto& v : *input_) {
        v = dis2(rng_);
      }

      // Overlap
      START_STOPWATCH();
      LOG(INFO, "Starting the overlap calculation.");
      overlapCalc_.calculate_overlap(col_syn_perm_, col_syn_perm_shape_, *input_, input_shape_);
      STOP_STOPWATCH();
      PRINT_ELAPSED_TIME();

      const std::vector<float> col_overlap_scores = overlapCalc_.get_col_overlaps();

      // If we don't have a true potential overlap buffer, use the tie-broken overlap as a proxy.
      potColOverlapGrid_ = col_overlap_scores;

      // Inhibition
      START_STOPWATCH();
      LOG(INFO, "Starting the inhibition calculation.");
      inhibitionCalc_.calculate_inhibition(col_overlap_scores, colOverlapGridShape_, potColOverlapGrid_, potColOverlapGridShape_);
      STOP_STOPWATCH();
      PRINT_ELAPSED_TIME();

      // Active columns
      const std::vector<int>& activeColIndices = inhibitionCalc_.get_active_column_indices();
      activeColIndices_ = activeColIndices;

      // Spatial learning
      START_STOPWATCH();
      LOG(INFO, "Starting the spatial learning calculation.");
      spatialLearnCalc_.calculate_spatiallearn_1d_active_indices(col_syn_perm_,
                                                                 col_syn_perm_shape_,
                                                                 overlapCalc_.get_col_pot_inputs(),
                                                                 overlapCalc_.get_col_pot_inputs_shape(),
                                                                 activeColIndices);
      STOP_STOPWATCH();
      PRINT_ELAPSED_TIME();

      // Temporal Memory consumes active columns to build cell/segment state.
      tm_.compute(activeColIndices);
    }
  }

  htm_gui::ProximalSynapseQuery query_proximal(int column_x, int column_y) const override {
    htm_gui::ProximalSynapseQuery q;
    q.column_x = column_x;
    q.column_y = column_y;

    if (column_x < 0 || column_x >= col_cols_ || column_y < 0 || column_y >= col_rows_) {
      return q;
    }

    const int col_idx = overlap_utils::flatten_index(column_x, column_y, col_cols_);
    const int base_row = column_y * step_y_;
    const int base_col = column_x * step_x_;

    q.synapses.reserve(num_pot_syn_);
    for (int ii = 0; ii < pot_height_; ++ii) {
      for (int jj = 0; jj < pot_width_; ++jj) {
        int row = base_row + ii;
        int col = base_col + jj;

        if (center_pot_synapses_) {
          row = base_row + ii - pot_height_ / 2;
          col = base_col + jj - pot_width_ / 2;
        }

        if (wrap_input_) {
          row = (row % input_rows_ + input_rows_) % input_rows_;
          col = (col % input_cols_ + input_cols_) % input_cols_;
        } else {
          if (row < 0 || row >= input_rows_ || col < 0 || col >= input_cols_) {
            continue;
          }
        }

        const int input_value = (*input_)[row * input_cols_ + col];
        const int syn_idx = ii * pot_width_ + jj;
        const float perm = col_syn_perm_[col_idx * num_pot_syn_ + syn_idx];

        htm_gui::ProximalSynapseInfo info;
        info.input_x = col;
        info.input_y = row;
        info.input_value = input_value;
        info.permanence = perm;
        info.connected = perm > connected_perm_;
        q.synapses.push_back(info);
      }
    }

    return q;
  }

  int num_segments(int column_x, int column_y, int cell) const override {
    if (column_x < 0 || column_x >= col_cols_ || column_y < 0 || column_y >= col_rows_) return 0;
    const int col_idx = overlap_utils::flatten_index(column_x, column_y, col_cols_);
    return tm_.num_segments(col_idx, cell);
  }

  htm_gui::DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const override {
    htm_gui::DistalSynapseQuery q;
    q.src_column_x = column_x;
    q.src_column_y = column_y;
    q.src_cell = cell;
    q.segment = segment;

    if (column_x < 0 || column_x >= col_cols_ || column_y < 0 || column_y >= col_rows_) return q;
    const int col_idx = overlap_utils::flatten_index(column_x, column_y, col_cols_);

    const temporal::Segment* seg = tm_.segment_ptr(col_idx, cell, segment);
    if (!seg) return q;

    q.synapses.reserve(seg->synapses.size());
    for (const auto& syn : seg->synapses) {
      const auto [dst_col_idx, dst_cell] = tm_.unflatten_cell(syn.src_cell);
      const int dst_x = dst_col_idx % col_cols_;
      const int dst_y = dst_col_idx / col_cols_;

      htm_gui::DistalSynapseInfo info;
      info.dst_column_x = dst_x;
      info.dst_column_y = dst_y;
      info.dst_cell = dst_cell;
      info.permanence = syn.permanence;
      info.connected = syn.permanence >= tm_connected_perm_;
      q.synapses.push_back(info);
    }

    return q;
  }

private:
  int pot_width_{0};
  int pot_height_{0};
  bool center_pot_synapses_{false};
  int input_rows_{0};
  int input_cols_{0};
  int col_rows_{0};
  int col_cols_{0};
  float connected_perm_{0.0f};
  int min_overlap_{0};
  bool wrap_input_{true};
  int num_pot_syn_{0};
  int num_columns_{0};

  int step_x_{1};
  int step_y_{1};

  int iteration_{0};

  mutable std::mt19937 rng_{};
  float tm_connected_perm_{0.5f};

  std::vector<float> col_syn_perm_;
  std::pair<int, int> col_syn_perm_shape_{};

  std::shared_ptr<std::vector<int>> input_;
  std::pair<int, int> input_shape_{};

  std::pair<int, int> colOverlapGridShape_{};
  std::vector<float> potColOverlapGrid_;
  std::pair<int, int> potColOverlapGridShape_{};

  overlap::OverlapCalculator overlapCalc_;
  inhibition::InhibitionCalculator inhibitionCalc_;
  spatiallearn::SpatialLearnCalculator spatialLearnCalc_;
  temporal::TemporalMemory tm_;

  std::vector<int> activeColIndices_;
};

}  // namespace
#endif

int main(int argc, char* argv[]) {
  using overlap::OverlapCalculator;
  using inhibition::InhibitionCalculator;
  using spatiallearn::SpatialLearnCalculator;

  // Overlap calculation parameters (similar to your existing setup)
  const int pot_width = 30;
  const int pot_height = 30;
  const bool center_pot_synapses = false;
  const int num_input_rows = 1200;
  const int num_input_cols = 1200;
  const int num_column_rows = 800;
  const int num_column_cols = 800;
  const float connected_perm = 0.3f;
  const int min_overlap = 3;
  const int num_pot_syn = pot_width * pot_height;
  const int num_columns = num_column_rows * num_column_cols;
  const bool wrap_input = true;
  const bool strict_local_activity = false;

  // Inhibition calculation parameters
  const int inhibition_width = 30;
  const int inhibition_height = 30;
  const int desired_local_activity = 10;

  // Spatial learning parameters
  const float spatialPermanenceInc = 0.05f;
  const float spatialPermanenceDec = 0.05f;
  const float activeColPermanenceDec = 0.01f;

#ifdef HTM_FLOW_WITH_GUI
  temporal::TMParams tm_params;
  tm_params.cells_per_column = 32;
  tm_params.connected_perm = 0.5f;
  tm_params.activation_threshold = 10;

  HtmFlowRuntime runtime(pot_width,
                         pot_height,
                         center_pot_synapses,
                         num_input_rows,
                         num_input_cols,
                         num_column_rows,
                         num_column_cols,
                         connected_perm,
                         min_overlap,
                         wrap_input,
                         inhibition_width,
                         inhibition_height,
                         desired_local_activity,
                         strict_local_activity,
                         spatialPermanenceInc,
                         spatialPermanenceDec,
                         activeColPermanenceDec,
                         tm_params);

  // Prime the runtime with an initial step so there's something to display immediately.
  runtime.step(1);

  htm_gui::DebuggerOptions opts;
  opts.window_title = "htm_flow";
  return htm_gui::run_debugger(argc, argv, runtime, opts);
#else
  // Headless fallback (original behavior): run a few iterations.
  std::vector<float> col_syn_perm(num_columns * num_pot_syn);
  const std::pair<int, int> col_syn_perm_shape = {num_columns, num_pot_syn};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (auto& v : col_syn_perm) v = dis(gen);

  std::vector<int> new_input_mat(num_input_rows * num_input_cols);
  const std::pair<int, int> new_input_mat_shape = {num_input_rows, num_input_cols};
  std::uniform_int_distribution<int> dis2(0, 1);

  std::pair<int, int> colOverlapGridShape = {num_column_rows, num_column_cols};
  std::vector<float> potColOverlapGrid(num_column_rows * num_column_cols, 1.0f);
  std::pair<int, int> potColOverlapGridShape = {num_column_rows, num_column_cols};

  OverlapCalculator overlapCalc(pot_width,
                                pot_height,
                                num_column_cols,
                                num_column_rows,
                                num_input_cols,
                                num_input_rows,
                                center_pot_synapses,
                                connected_perm,
                                min_overlap,
                                wrap_input);

  InhibitionCalculator inhibitionCalc(num_column_cols,
                                      num_column_rows,
                                      inhibition_width,
                                      inhibition_height,
                                      desired_local_activity,
                                      min_overlap,
                                      center_pot_synapses,
                                      wrap_input,
                                      strict_local_activity);

  SpatialLearnCalculator spatialLearnCalc(num_columns, num_pot_syn, spatialPermanenceInc, spatialPermanenceDec, activeColPermanenceDec);

  for (int t = 0; t < NUM_ITERATIONS; ++t) {
    LOG(INFO, "=== Iteration " + std::to_string(t) + " ===");

    for (auto& v : new_input_mat) v = dis2(gen);

    START_STOPWATCH();
    LOG(INFO, "Starting the overlap calculation.");
    overlapCalc.calculate_overlap(col_syn_perm, col_syn_perm_shape, new_input_mat, new_input_mat_shape);
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();

    const std::vector<float> col_overlap_scores = overlapCalc.get_col_overlaps();

    START_STOPWATCH();
    LOG(INFO, "Starting the inhibition calculation.");
    inhibitionCalc.calculate_inhibition(col_overlap_scores, colOverlapGridShape, potColOverlapGrid, potColOverlapGridShape);
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();

    const std::vector<int>& activeColIndices = inhibitionCalc.get_active_column_indices();

    START_STOPWATCH();
    LOG(INFO, "Starting the spatial learning calculation.");
    spatialLearnCalc.calculate_spatiallearn_1d_active_indices(col_syn_perm,
                                                             col_syn_perm_shape,
                                                             overlapCalc.get_col_pot_inputs(),
                                                             overlapCalc.get_col_pot_inputs_shape(),
                                                             activeColIndices);
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
  }

  return 0;
#endif
}
