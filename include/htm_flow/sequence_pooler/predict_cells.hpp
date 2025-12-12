#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <taskflow/taskflow.hpp>

#include <htm_flow/sequence_pooler/sequence_types.hpp>

namespace sequence_pooler {

// Predict-cells stage of the sequence pooler.
//
// What this component does:
// - Looks at each column's distal segments and counts how many synapses connect to
//   cells that are active at `time_step`.
// - If a segment has more than `activation_threshold` active connected synapses, the
//   segment is considered active and will place the cell into the predictive state.
// - For each column, we pick the single "most predictive" cell (highest active-synapse
//   count across all segments/cells in that column).
//
// What this component returns:
// - `predictCellsTime`: last two timesteps each cell was predictive.
// - `activeSegsTime`  : last timestep each segment was active (sequence segment).
// - Segment-update tensors (`segIndUpdate`, `segActiveSyn`) for later learning.
class PredictCellsCalculator {
public:
  struct Config {
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;
    float connect_permanence = 0.0f;
    int activation_threshold = 0;
  };

  explicit PredictCellsCalculator(const Config& cfg);

  // Step 1. Read current active-cells time history (`active_cells_time`).
  // Step 2. Scan distal segments and mark predictive cells for this timestep.
  // Step 3. Produce update tensors for the learning stage (future component).
  void calculate_predict_cells(int time_step,
                               const std::vector<int>& active_cells_time,
                               const std::vector<DistalSynapse>& distal_synapses);

  // --- Outputs ---
  const std::vector<int>& get_predict_cells_time() const;
  const std::vector<int>& get_active_segs_time() const;
  const std::vector<int>& get_seg_ind_update() const;
  const std::vector<int8_t>& get_seg_active_syn() const;

  int num_columns() const { return cfg_.num_columns; }
  int cells_per_column() const { return cfg_.cells_per_column; }
  int max_segments_per_cell() const { return cfg_.max_segments_per_cell; }
  int max_synapses_per_segment() const { return cfg_.max_synapses_per_segment; }

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }
  inline int idx_cell_seg(int col, int cell, int seg) const {
    return (col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell + seg;
  }
  inline int idx_cell_syn_list(int col, int cell, int syn) const {
    return (col * cfg_.cells_per_column + cell) * cfg_.max_synapses_per_segment + syn;
  }

  bool check_cell_active(const std::vector<int>& active_cells_time,
                         int col,
                         int cell,
                         int time_step) const;

  bool check_cell_predicting(int col, int cell, int time_step) const;
  void set_predict_cell(int col, int cell, int time_step);
  void set_active_seg(int col, int cell, int seg, int time_step);

  int count_active_connected_synapses(const std::vector<int>& active_cells_time,
                                      const std::vector<DistalSynapse>& distal_synapses,
                                      int time_step,
                                      int col,
                                      int cell,
                                      int seg) const;

  // Build the segment-active synapse list (0/1) for a chosen segment.
  void fill_seg_active_syn_list(const std::vector<int>& active_cells_time,
                                const std::vector<DistalSynapse>& distal_synapses,
                                int time_step,
                                int col,
                                int cell,
                                int seg,
                                int8_t* out01) const;

  Config cfg_;

  // predictCellsTime: (num_columns, cells_per_column, 2)
  std::vector<int> predict_cells_time_;
  // activeSegsTime: (num_columns, cells_per_column, max_segments_per_cell)
  std::vector<int> active_segs_time_;

  // segIndUpdate: (num_columns, cells_per_column) => which segment to update, or -1
  std::vector<int> seg_ind_update_;
  // segActiveSyn: (num_columns, cells_per_column, max_synapses_per_segment) => 0/1 list
  std::vector<int8_t> seg_active_syn_;

  tf::Executor executor_;
};

} // namespace sequence_pooler

