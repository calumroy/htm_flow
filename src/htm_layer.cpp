#include <htm_flow/htm_layer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <utilities/logger.hpp>
#include <utilities/stopwatch.hpp>

namespace htm_flow {

namespace {

inline std::pair<int, int> unflatten_xy(int idx, int width) {
  const int x = idx % width;
  const int y = idx / width;
  return {x, y};
}

}  // namespace

HTMLayer::HTMLayer(const HTMLayerConfig& cfg) : HTMLayer(cfg, "HTMLayer") {}

HTMLayer::HTMLayer(const HTMLayerConfig& cfg, const std::string& name)
    : cfg_(cfg),
      name_(name),
      timestep_(0),
      num_pot_syn_(cfg_.pot_width * cfg_.pot_height),
      num_columns_(cfg_.num_column_rows * cfg_.num_column_cols),
      gen_(std::random_device{}()),
      input_(std::make_shared<std::vector<int>>(
          static_cast<std::size_t>(cfg_.num_input_rows) * static_cast<std::size_t>(cfg_.num_input_cols), 0)),
      col_syn_perm_(static_cast<std::size_t>(num_columns_) * static_cast<std::size_t>(num_pot_syn_)),
      col_overlap_grid_(static_cast<std::size_t>(num_columns_), 0.0f),
      pot_col_overlap_grid_(static_cast<std::size_t>(num_columns_), 1.0f),
      col_active01_(static_cast<std::size_t>(num_columns_), 0),
      overlap_calc_(cfg_.pot_width,
                    cfg_.pot_height,
                    cfg_.num_column_cols,
                    cfg_.num_column_rows,
                    cfg_.num_input_cols,
                    cfg_.num_input_rows,
                    cfg_.center_pot_synapses,
                    cfg_.connected_perm,
                    cfg_.min_overlap,
                    cfg_.wrap_input),
      inhibition_calc_(cfg_.num_column_cols,
                       cfg_.num_column_rows,
                       cfg_.inhibition_width,
                       cfg_.inhibition_height,
                       cfg_.desired_local_activity,
                       cfg_.min_overlap,
                       cfg_.min_potential_overlap,
                       cfg_.center_pot_synapses,
                       cfg_.wrap_input,
                       cfg_.strict_local_activity),
      spatial_learn_calc_(num_columns_,
                          num_pot_syn_,
                          cfg_.spatial_permanence_inc,
                          cfg_.spatial_permanence_dec,
                          cfg_.active_col_permanence_dec),
      active_cells_calc_(sequence_pooler::ActiveCellsCalculator::Config{
          num_columns_,
          cfg_.cells_per_column,
          cfg_.max_segments_per_cell,
          cfg_.max_synapses_per_segment,
          cfg_.min_num_syn_threshold,
          cfg_.new_syn_permanence,
          cfg_.connect_permanence,
      }),
      predict_cells_calc_(sequence_pooler::PredictCellsCalculator::Config{
          num_columns_,
          cfg_.cells_per_column,
          cfg_.max_segments_per_cell,
          cfg_.max_synapses_per_segment,
          cfg_.connect_permanence,
          cfg_.activation_threshold,
      }),
      seq_learn_calc_(sequence_pooler::SequenceLearningCalculator::Config{
          num_columns_,
          cfg_.cells_per_column,
          cfg_.max_segments_per_cell,
          cfg_.max_synapses_per_segment,
          cfg_.connect_permanence,
          cfg_.sequence_permanence_inc,
          cfg_.sequence_permanence_dec,
      }),
      temporal_pool_calc_(temporal_pooler::TemporalPoolerCalculator::Config{
          num_columns_,
          cfg_.cells_per_column,
          cfg_.max_segments_per_cell,
          cfg_.max_synapses_per_segment,
          num_pot_syn_,
          cfg_.temp_spatial_permanence_inc,
          cfg_.temp_sequence_permanence_inc,
          cfg_.temp_sequence_permanence_dec,
          cfg_.min_num_syn_threshold,
          cfg_.new_syn_permanence,
          cfg_.connect_permanence,
          cfg_.temp_delay_length,
          cfg_.temp_enable_persistence,
      }) {
  if (cfg_.num_input_rows <= 0 || cfg_.num_input_cols <= 0 || cfg_.num_column_rows <= 0 || cfg_.num_column_cols <= 0 ||
      cfg_.pot_width <= 0 || cfg_.pot_height <= 0) {
    throw std::invalid_argument("Invalid HTMLayer configuration (non-positive shape).");
  }

  // Compute step sizes using the same helper as overlap.
  const auto steps = overlap_utils::get_step_sizes(cfg_.num_input_cols,
                                                   cfg_.num_input_rows,
                                                   cfg_.num_column_cols,
                                                   cfg_.num_column_rows,
                                                   cfg_.pot_width,
                                                   cfg_.pot_height);
  step_x_ = steps.first;
  step_y_ = steps.second;

  // Initialize proximal permanence values to zero.
  std::fill(col_syn_perm_.begin(), col_syn_perm_.end(), 0.0f);

  // Initialize distal synapses randomly once.
  const std::size_t distal_syn_count =
      static_cast<std::size_t>(num_columns_) * static_cast<std::size_t>(cfg_.cells_per_column) *
      static_cast<std::size_t>(cfg_.max_segments_per_cell) * static_cast<std::size_t>(cfg_.max_synapses_per_segment);
  distal_synapses_.resize(distal_syn_count);

  std::uniform_int_distribution<int> col_dis(0, num_columns_ - 1);
  std::uniform_int_distribution<int> cell_dis(0, cfg_.cells_per_column - 1);
  for (std::size_t i = 0; i < distal_syn_count; ++i) {
    distal_synapses_[i] = sequence_pooler::DistalSynapse{col_dis(gen_), cell_dis(gen_), perm01_(gen_)};
  }
}

void HTMLayer::set_input(const std::vector<int>& input) {
  const std::size_t expected_size =
      static_cast<std::size_t>(cfg_.num_input_rows) * static_cast<std::size_t>(cfg_.num_input_cols);
  if (input.size() != expected_size) {
    throw std::invalid_argument("Input size mismatch: expected " + std::to_string(expected_size) + " got " +
                                std::to_string(input.size()));
  }
  // Copy into our internal buffer
  auto new_input = std::make_shared<std::vector<int>>(input);
  input_ = new_input;
}

void HTMLayer::set_input(std::shared_ptr<const std::vector<int>> input) {
  const std::size_t expected_size =
      static_cast<std::size_t>(cfg_.num_input_rows) * static_cast<std::size_t>(cfg_.num_input_cols);
  if (!input || input->size() != expected_size) {
    throw std::invalid_argument("Input size mismatch or null pointer");
  }
  // We need a mutable pointer for snapshot(), so copy if shared
  auto new_input = std::make_shared<std::vector<int>>(*input);
  input_ = new_input;
}

std::vector<int> HTMLayer::output() const {
  // Output is a 2D grid flattened: (num_column_rows, num_column_cols * cells_per_column)
  // Each cell's activation is 0 or 1.
  const int out_rows = cfg_.num_column_rows;
  const int out_cols = cfg_.num_column_cols * cfg_.cells_per_column;
  std::vector<int> result(static_cast<std::size_t>(out_rows) * static_cast<std::size_t>(out_cols), 0);

  const auto& active_time = active_cells_calc_.get_active_cells_time();

  for (int col = 0; col < num_columns_; ++col) {
    const int col_x = col % cfg_.num_column_cols;
    const int col_y = col / cfg_.num_column_cols;

    for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
      const int base = (col * cfg_.cells_per_column + cell) * 2;
      if (time_is_set(active_time, base, timestep_)) {
        // Output position: row = col_y, col = col_x * cells_per_column + cell
        const int out_idx = col_y * out_cols + col_x * cfg_.cells_per_column + cell;
        result[static_cast<std::size_t>(out_idx)] = 1;
      }
    }
  }

  return result;
}

bool HTMLayer::time_is_set(const std::vector<int>& time_tensor2, int idx0, int time_step) {
  return (time_tensor2[static_cast<std::size_t>(idx0)] == time_step) ||
         (time_tensor2[static_cast<std::size_t>(idx0 + 1)] == time_step);
}

void HTMLayer::step(int n) {
  if (n <= 0) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    step_once();
  }
}

void HTMLayer::step_once() {
  ++timestep_;

  if (cfg_.log_timings) {
    LOG(INFO, "=== " + name_ + " Iteration " + std::to_string(timestep_ - 1) + " ===");
  }

  const std::pair<int, int> col_syn_perm_shape = {num_columns_, num_pot_syn_};
  const std::pair<int, int> input_shape = {cfg_.num_input_rows, cfg_.num_input_cols};

  // Overlap
  if (cfg_.log_timings) {
    START_STOPWATCH();
    LOG(INFO, "Starting the overlap calculation.");
  }
  overlap_calc_.calculate_overlap(col_syn_perm_, col_syn_perm_shape, *input_, input_shape);
  if (cfg_.log_timings) {
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
  }
  const std::vector<float> col_overlap_scores = overlap_calc_.get_col_overlaps();
  col_overlap_grid_.assign(col_overlap_scores.begin(), col_overlap_scores.end());
  pot_col_overlap_grid_.assign(overlap_calc_.get_col_pot_overlaps().begin(), overlap_calc_.get_col_pot_overlaps().end());

  // Inhibition
  const std::pair<int, int> col_grid_shape = {cfg_.num_column_rows, cfg_.num_column_cols};
  if (cfg_.log_timings) {
    START_STOPWATCH();
    LOG(INFO, "Starting the inhibition calculation.");
  }
  inhibition_calc_.calculate_inhibition(
      col_overlap_scores, col_grid_shape, pot_col_overlap_grid_, col_grid_shape, /*use_serial_sort_calc=*/true);
  if (cfg_.log_timings) {
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
  }
  active_col_indices_ = inhibition_calc_.get_active_column_indices();
  if (cfg_.log_timings) {
    const float pot_max =
        pot_col_overlap_grid_.empty() ? 0.0f : *std::max_element(pot_col_overlap_grid_.begin(), pot_col_overlap_grid_.end());
    const float pot_min =
        pot_col_overlap_grid_.empty() ? 0.0f : *std::min_element(pot_col_overlap_grid_.begin(), pot_col_overlap_grid_.end());
    LOG(INFO, "Active columns: " + std::to_string(active_col_indices_.size()) + " / " + std::to_string(num_columns_) +
                  " (min_overlap=" + std::to_string(cfg_.min_overlap) +
                  ", min_potential_overlap=" + std::to_string(cfg_.min_potential_overlap) + ")");
    LOG(INFO, "Potential-overlap stats: min=" + std::to_string(pot_min) + " max=" + std::to_string(pot_max));
  }

  // Dense active columns bitfield
  for (int c : prev_active_col_indices_) {
    col_active01_[static_cast<std::size_t>(c)] = 0;
  }
  for (int c : active_col_indices_) {
    col_active01_[static_cast<std::size_t>(c)] = 1;
  }
  prev_active_col_indices_ = active_col_indices_;

  // Spatial learning
  if (cfg_.log_timings) {
    START_STOPWATCH();
    LOG(INFO, "Starting the spatial learning calculation.");
  }
  spatial_learn_calc_.calculate_spatiallearn_1d_active_indices(col_syn_perm_,
                                                               col_syn_perm_shape,
                                                               overlap_calc_.get_col_pot_inputs(),
                                                               overlap_calc_.get_col_pot_inputs_shape(),
                                                               active_col_indices_);
  if (cfg_.log_timings) {
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
    double sum = 0.0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    std::size_t nz = 0;
    for (float p : col_syn_perm_) {
      sum += static_cast<double>(p);
      mn = std::min(mn, p);
      mx = std::max(mx, p);
      nz += (p != 0.0f) ? 1u : 0u;
    }
    const double mean = col_syn_perm_.empty() ? 0.0 : (sum / static_cast<double>(col_syn_perm_.size()));
    LOG(INFO, "Proximal permanence stats: min=" + std::to_string(mn) + " max=" + std::to_string(mx) +
                  " mean=" + std::to_string(mean) + " nonzero=" + std::to_string(nz) + "/" +
                  std::to_string(col_syn_perm_.size()));
  }

  // Sequence pooler: active cells
  if (cfg_.log_timings) {
    START_STOPWATCH();
    LOG(INFO, "Starting the sequence-pooler active-cells calculation.");
  }
  active_cells_calc_.calculate_active_cells(timestep_,
                                            active_col_indices_,
                                            predict_cells_calc_.get_predict_cells_time(),
                                            predict_cells_calc_.get_active_segs_time(),
                                            distal_synapses_);
  if (cfg_.log_timings) {
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
  }

  // Sequence pooler: predict cells
  if (cfg_.log_timings) {
    START_STOPWATCH();
    LOG(INFO, "Starting the sequence-pooler predict-cells calculation.");
  }
  predict_cells_calc_.calculate_predict_cells(timestep_, active_cells_calc_.get_active_cells_time(), distal_synapses_);
  if (cfg_.log_timings) {
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
  }

  // Sequence learning
  if (cfg_.log_timings) {
    START_STOPWATCH();
    LOG(INFO, "Starting the sequence-pooler sequence-learning calculation.");
  }
  seq_learn_calc_.calculate_sequence_learning(timestep_,
                                              active_cells_calc_.get_active_cells_time(),
                                              active_cells_calc_.get_learn_cells_time(),
                                              predict_cells_calc_.get_predict_cells_time(),
                                              distal_synapses_,
                                              active_cells_calc_.get_seg_ind_update_active(),
                                              active_cells_calc_.get_seg_active_syn_active(),
                                              active_cells_calc_.get_seg_ind_new_syn_active(),
                                              active_cells_calc_.get_seg_new_syn_active(),
                                              predict_cells_calc_.get_seg_ind_update_mutable(),
                                              predict_cells_calc_.get_seg_active_syn_mutable());
  if (cfg_.log_timings) {
    STOP_STOPWATCH();
    PRINT_ELAPSED_TIME();
  }

  // Temporal pooler
  if (cfg_.temp_enabled) {
    if (cfg_.log_timings) {
      START_STOPWATCH();
      LOG(INFO, "Starting the temporal pooler calculation.");
    }
    temporal_pool_calc_.update_proximal(timestep_,
                                        overlap_calc_.get_col_pot_inputs(),
                                        col_active01_,
                                        col_syn_perm_,
                                        active_cells_calc_.get_burst_cols_time());
    temporal_pool_calc_.update_distal(timestep_,
                                      active_cells_calc_.get_current_learn_cells_list(),
                                      active_cells_calc_.get_learn_cells_time(),
                                      predict_cells_calc_.get_predict_cells_time_mutable(),
                                      active_cells_calc_.get_active_cells_time(),
                                      predict_cells_calc_.get_active_segs_time(),
                                      distal_synapses_);
    if (cfg_.log_timings) {
      STOP_STOPWATCH();
      PRINT_ELAPSED_TIME();
    }
  }
}

htm_gui::Snapshot HTMLayer::snapshot() const {
  htm_gui::Snapshot s;
  s.timestep = timestep_;
  s.input_shape = htm_gui::GridShape{cfg_.num_input_rows, cfg_.num_input_cols};
  s.columns_shape = htm_gui::GridShape{cfg_.num_column_rows, cfg_.num_column_cols};
  s.cells_per_column = cfg_.cells_per_column;
  s.input = std::static_pointer_cast<const std::vector<int>>(input_);
  s.active_column_indices = active_col_indices_;

  // Pack per-cell masks for every column.
  s.column_cell_masks.resize(static_cast<std::size_t>(num_columns_));

  const auto& active_time = active_cells_calc_.get_active_cells_time();
  const auto& learn_time = active_cells_calc_.get_learn_cells_time();
  const auto& pred_time = predict_cells_calc_.get_predict_cells_time();

  for (int col = 0; col < num_columns_; ++col) {
    htm_gui::ColumnCellMasks masks{};

    for (int cell = 0; cell < cfg_.cells_per_column && cell < 64; ++cell) {
      const int base = (col * cfg_.cells_per_column + cell) * 2;
      const std::uint64_t bit = (std::uint64_t(1) << cell);

      if (time_is_set(active_time, base, timestep_)) {
        masks.active |= bit;
      }
      if (time_is_set(pred_time, base, timestep_)) {
        masks.predictive |= bit;
      }
      if (time_is_set(learn_time, base, timestep_)) {
        masks.learning |= bit;
      }
    }

    s.column_cell_masks[static_cast<std::size_t>(col)] = masks;
  }

  return s;
}

htm_gui::ProximalSynapseQuery HTMLayer::query_proximal(int column_x, int column_y) const {
  htm_gui::ProximalSynapseQuery q;
  q.column_x = column_x;
  q.column_y = column_y;

  if (column_x < 0 || column_x >= cfg_.num_column_cols || column_y < 0 || column_y >= cfg_.num_column_rows) {
    return q;
  }

  const int col = flatten_col(column_x, column_y);
  if (col >= 0 && col < int(col_overlap_grid_.size())) {
    q.overlap = col_overlap_grid_[static_cast<std::size_t>(col)];
  }
  if (col >= 0 && col < int(pot_col_overlap_grid_.size())) {
    q.potential_overlap = pot_col_overlap_grid_[static_cast<std::size_t>(col)];
  }
  q.synapses.reserve(static_cast<std::size_t>(num_pot_syn_));

  const int start_row = column_y * step_y_;
  const int start_col = column_x * step_x_;
  const int row_off = cfg_.center_pot_synapses ? (cfg_.pot_height / 2) : 0;
  const int col_off = cfg_.center_pot_synapses ? (cfg_.pot_width / 2) : 0;

  for (int ii = 0; ii < cfg_.pot_height; ++ii) {
    for (int jj = 0; jj < cfg_.pot_width; ++jj) {
      int in_row = start_row + ii - row_off;
      int in_col = start_col + jj - col_off;

      if (cfg_.wrap_input) {
        in_row = (in_row % cfg_.num_input_rows + cfg_.num_input_rows) % cfg_.num_input_rows;
        in_col = (in_col % cfg_.num_input_cols + cfg_.num_input_cols) % cfg_.num_input_cols;
      }

      const int syn = ii * cfg_.pot_width + jj;
      const std::size_t perm_idx =
          static_cast<std::size_t>(col) * static_cast<std::size_t>(num_pot_syn_) + static_cast<std::size_t>(syn);
      const float perm = col_syn_perm_[perm_idx];
      const bool connected = perm > cfg_.connected_perm;

      htm_gui::ProximalSynapseInfo info;
      if (in_row < 0 || in_row >= cfg_.num_input_rows || in_col < 0 || in_col >= cfg_.num_input_cols) {
        info.input_x = -1;
        info.input_y = -1;
        info.input_value = 0;
      } else {
        info.input_x = in_col;
        info.input_y = in_row;
        info.input_value =
            (*input_)[static_cast<std::size_t>(in_row) * static_cast<std::size_t>(cfg_.num_input_cols) +
                      static_cast<std::size_t>(in_col)];
      }
      info.permanence = perm;
      info.connected = connected;
      q.synapses.push_back(info);
    }
  }

  return q;
}

int HTMLayer::num_segments(int column_x, int column_y, int cell) const {
  (void)column_x;
  (void)column_y;
  (void)cell;
  return cfg_.max_segments_per_cell;
}

htm_gui::DistalSynapseQuery HTMLayer::query_distal(int column_x, int column_y, int cell, int segment) const {
  htm_gui::DistalSynapseQuery q;
  q.src_column_x = column_x;
  q.src_column_y = column_y;
  q.src_cell = cell;
  q.segment = segment;

  if (column_x < 0 || column_x >= cfg_.num_column_cols || column_y < 0 || column_y >= cfg_.num_column_rows) {
    return q;
  }
  if (cell < 0 || cell >= cfg_.cells_per_column) {
    return q;
  }
  if (segment < 0 || segment >= cfg_.max_segments_per_cell) {
    return q;
  }

  const std::size_t origin_col = static_cast<std::size_t>(flatten_col(column_x, column_y));
  const std::size_t origin_cell = static_cast<std::size_t>(cell);
  const std::size_t origin_seg = static_cast<std::size_t>(segment);

  q.synapses.reserve(static_cast<std::size_t>(cfg_.max_synapses_per_segment));

  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx = sequence_pooler::idx_distal_synapse(origin_col,
                                                                origin_cell,
                                                                origin_seg,
                                                                static_cast<std::size_t>(syn),
                                                                static_cast<std::size_t>(cfg_.cells_per_column),
                                                                static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                                                static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const auto& ds = distal_synapses_[idx];

    const auto [dst_x, dst_y] = unflatten_xy(ds.target_col, cfg_.num_column_cols);

    htm_gui::DistalSynapseInfo info;
    info.dst_column_x = dst_x;
    info.dst_column_y = dst_y;
    info.dst_cell = ds.target_cell;
    info.permanence = ds.perm;
    info.connected = ds.perm > cfg_.connect_permanence;
    q.synapses.push_back(info);
  }

  return q;
}

}  // namespace htm_flow
