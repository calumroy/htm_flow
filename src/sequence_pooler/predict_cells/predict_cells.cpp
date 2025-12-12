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

bool PredictCellsCalculator::check_cell_predicting(int col, int cell, int time_step) const {
  const int p0 = predict_cells_time_[idx_cell_time(col, cell, 0)];
  const int p1 = predict_cells_time_[idx_cell_time(col, cell, 1)];
  return (p0 == time_step) || (p1 == time_step);
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

void PredictCellsCalculator::fill_seg_active_syn_list(const std::vector<int>& active_cells_time,
                                                      const std::vector<DistalSynapse>& distal_synapses,
                                                      int time_step,
                                                      int col,
                                                      int cell,
                                                      int seg,
                                                      int8_t* out01) const {
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(col),
                                               static_cast<std::size_t>(cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    if (s.perm >= cfg_.connect_permanence &&
        check_cell_active(active_cells_time, s.target_col, s.target_cell, time_step)) {
      out01[syn] = 1;
    } else {
      out01[syn] = 0;
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

  // We overwrite seg update tensors each timestep (like the python version).
  std::fill(seg_ind_update_.begin(), seg_ind_update_.end(), -1);
  // seg_active_syn_ is only meaningful where seg_ind_update_ != -1; still clear for clarity.
  std::fill(seg_active_syn_.begin(), seg_active_syn_.end(), 0);

  // Implementation overview:
  // Step 1. For each column, scan all cells and segments and compute "prediction level"
  //         (= number of connected synapses ending on currently active cells).
  // Step 2. Mark segments with predictionLevel > activation_threshold as active segments.
  // Step 3. Choose the single most-predictive cell in each column and set it predictive.
  // Step 4. If it wasn't predicting last timestep, emit segment-update tensors for learning.

  tf::Taskflow taskflow;

  taskflow.for_each_index(
      0, cfg_.num_columns, 1,
      [&](int c) {
        int most_pred_syn_count = 0;
        int most_pred_cell = 0;
        int most_pred_seg = 0;
        bool column_predicting = false;

        for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
          for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
            const int prediction_level =
                count_active_connected_synapses(active_cells_time, distal_synapses, time_step, c, cell, seg);

            if (prediction_level > cfg_.activation_threshold) {
              set_active_seg(c, cell, seg, time_step);
              if (prediction_level > most_pred_syn_count) {
                most_pred_syn_count = prediction_level;
                most_pred_cell = cell;
                most_pred_seg = seg;
                column_predicting = true;
              }
            }
          }
        }

        if (column_predicting) {
          // Set the most predicting cell in the column as predictive.
          set_predict_cell(c, most_pred_cell, time_step);

          // Only emit update tensors if this cell wasn't already predicting at (time_step-1).
          if (!check_cell_predicting(c, most_pred_cell, time_step - 1)) {
            seg_ind_update_[c * cfg_.cells_per_column + most_pred_cell] = most_pred_seg;
            int8_t* out01 =
                &seg_active_syn_[(c * cfg_.cells_per_column + most_pred_cell) * cfg_.max_synapses_per_segment];
            fill_seg_active_syn_list(active_cells_time, distal_synapses, time_step, c, most_pred_cell, most_pred_seg,
                                     out01);
          }
        }
      })
      .name("predict_cells_for_each_column");

  executor_.run(taskflow).wait();
}

} // namespace sequence_pooler

