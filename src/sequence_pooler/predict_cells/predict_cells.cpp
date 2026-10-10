#include <htm_flow/sequence_pooler/predict_cells.hpp>

#include <cassert>
#include <taskflow/algorithm/for_each.hpp>

namespace sequence_pooler {

PredictCellsCalculator::PredictCellsCalculator(const Config& cfg) : cfg_(cfg) {
  assert(cfg_.num_columns > 0);
  assert(cfg_.cells_per_column > 0);
  assert(cfg_.max_segments_per_cell > 0);
  assert(cfg_.max_synapses_per_segment > 0);

  predict_cells_time_.assign(cfg_.num_columns * cfg_.cells_per_column * 2, -1);
  active_segs_time_.assign(cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell, -1);
  seg_ind_update_.assign(cfg_.num_columns * cfg_.cells_per_column, -1);
  seg_active_syn_.assign(cfg_.num_columns * cfg_.cells_per_column * cfg_.max_synapses_per_segment, 0);

  const int num_cells = cfg_.num_columns * cfg_.cells_per_column;
  const int num_segments = num_cells * cfg_.max_segments_per_cell;
  synapse_words_per_segment_ = (cfg_.max_synapses_per_segment + 63) / 64;
  prev_active_segment_syn_bits_.assign(
      static_cast<std::size_t>(num_segments) * synapse_words_per_segment_, 0);
  current_active_segment_syn_bits_.assign(
      static_cast<std::size_t>(num_segments) * synapse_words_per_segment_, 0);
  current_best_segments_.assign(num_cells, -1);
}

const std::vector<int>& PredictCellsCalculator::get_predict_cells_time() const {
  return predict_cells_time_;
}

const std::vector<int>& PredictCellsCalculator::get_active_segs_time() const {
  return active_segs_time_;
}

const std::vector<int>& PredictCellsCalculator::get_seg_ind_update() const {
  return seg_ind_update_;
}

const std::vector<int8_t>& PredictCellsCalculator::get_seg_active_syn() const {
  return seg_active_syn_;
}

const std::vector<std::uint64_t>& PredictCellsCalculator::get_prev_active_segment_syn_bits() const {
  return prev_active_segment_syn_bits_;
}

const std::vector<int>& PredictCellsCalculator::get_current_best_segments() const {
  return current_best_segments_;
}

const std::vector<std::uint64_t>& PredictCellsCalculator::get_current_active_segment_syn_bits() const {
  return current_active_segment_syn_bits_;
}

std::vector<int>& PredictCellsCalculator::get_seg_ind_update_mutable() {
  return seg_ind_update_;
}

std::vector<int8_t>& PredictCellsCalculator::get_seg_active_syn_mutable() {
  return seg_active_syn_;
}

bool PredictCellsCalculator::check_cell_active(const std::vector<int>& active_cells_time,
                                               int col,
                                               int cell,
                                               int time_step) const {
  const int a0 = active_cells_time[idx_cell_time(col, cell, 0)];
  const int a1 = active_cells_time[idx_cell_time(col, cell, 1)];
  return (a0 == time_step) || (a1 == time_step);
}

void PredictCellsCalculator::set_predict_cell(int col, int cell, int time_step) {
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  if (predict_cells_time_[i0] <= predict_cells_time_[i1]) {
    predict_cells_time_[i0] = time_step;
  } else {
    predict_cells_time_[i1] = time_step;
  }
}

void PredictCellsCalculator::set_active_seg(int col, int cell, int seg, int time_step) {
  active_segs_time_[idx_cell_seg(col, cell, seg)] = time_step;
}

int PredictCellsCalculator::count_active_connected_synapses(const std::vector<int>& active_cells_time,
                                                            const std::vector<DistalSynapse>& distal_synapses,
                                                            int time_step,
                                                            int col,
                                                            int cell,
                                                            int seg) const {
  int count = 0;
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(col),
                                               static_cast<std::size_t>(cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    if (s.perm > cfg_.connect_permanence) {
      if (check_cell_active(active_cells_time, s.target_col, s.target_cell, time_step)) {
        ++count;
      }
    }
  }
  return count;
}

void PredictCellsCalculator::fill_active_synapse_bits(
    const std::vector<int>& active_cells_time,
    const std::vector<DistalSynapse>& distal_synapses,
    int time_step,
    int col,
    int cell,
    int seg,
    std::uint64_t* out_bits) const {
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(col),
                                               static_cast<std::size_t>(cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    if (s.perm > cfg_.connect_permanence &&
        check_cell_active(active_cells_time, s.target_col, s.target_cell, time_step)) {
      out_bits[syn / 64] |= std::uint64_t{1} << (syn % 64);
    }
  }
}

void PredictCellsCalculator::calculate_predict_cells(int time_step,
                                                     const std::vector<int>& active_cells_time,
                                                     const std::vector<DistalSynapse>& distal_synapses) {
  // Basic parameter validation (shape checks).
  assert(static_cast<int>(active_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<std::size_t>(distal_synapses.size()) ==
         static_cast<std::size_t>(cfg_.num_columns) * cfg_.cells_per_column *
             cfg_.max_segments_per_cell * cfg_.max_synapses_per_segment);

  // Save the evidence produced on the previous predict pass before writing the
  // current pass. Sequence learning uses this snapshot after prediction runs.
  prev_active_segment_syn_bits_ = current_active_segment_syn_bits_;
  std::fill(current_active_segment_syn_bits_.begin(),
            current_active_segment_syn_bits_.end(),
            std::uint64_t{0});
  std::fill(current_best_segments_.begin(), current_best_segments_.end(), -1);

  // Implementation overview:
  // Step 1. For each column, scan all cells and segments and compute "prediction level"
  //         (= number of connected synapses ending on currently active cells).
  // Step 2. Mark segments with predictionLevel > activation_threshold as active segments.
  // Step 3. Set all cells with at least one active segment as predictive (multiple cells
  //         per column can be predictive simultaneously, matching standard HTM behavior).
  // Step 4. Save each active segment's synapse mask and each cell's best segment.

  tf::Taskflow taskflow;

  taskflow.for_each_index(
      0, cfg_.num_columns, 1,
      [&](int c) {
        for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
          int best_seg = -1;
          int best_count = 0;

          for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
            const int prediction_level =
                count_active_connected_synapses(active_cells_time, distal_synapses, time_step, c, cell, seg);

            if (prediction_level > cfg_.activation_threshold) {
              set_active_seg(c, cell, seg, time_step);
              const int segment_flat = idx_cell_seg(c, cell, seg);
              std::uint64_t* bits =
                  &current_active_segment_syn_bits_[static_cast<std::size_t>(segment_flat) *
                                                    synapse_words_per_segment_];
              fill_active_synapse_bits(
                  active_cells_time, distal_synapses, time_step, c, cell, seg, bits);
              if (prediction_level > best_count) {
                best_count = prediction_level;
                best_seg = seg;
              }
            }
          }

          if (best_seg >= 0) {
            // This cell has at least one active segment -- mark it predictive.
            set_predict_cell(c, cell, time_step);
            current_best_segments_[static_cast<std::size_t>(
                c * cfg_.cells_per_column + cell)] = best_seg;
          }
        }
      })
      .name("predict_cells_for_each_column");

  executor_.run(taskflow).wait();
}

} // namespace sequence_pooler

