#pragma once

#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include <taskflow/taskflow.hpp>

#include <htm_flow/sequence_pooler/sequence_types.hpp>

namespace temporal_pooler {

// Temporal pooler (ported from `HTM/src/HTM_calc/np_temporal.py`).
//
// Responsibilities:
// - Proximal temporal pooling: update proximal permanence values using *previous* and
//   *current* potential-input activity, skipping bursting columns.
// - Distal temporal pooling: track per-cell persistence and reinforce/create distal
//   segments for "active predictive" cells. Also extends predictive state for a
//   persistence window even without active segments.
class TemporalPoolerCalculator {
public:
  struct Config {
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;
    int num_pot_synapses = 0;

    float spatial_permanence_inc = 0.0f;
    float seq_permanence_inc = 0.0f;

    int min_num_syn_threshold = 0;
    float new_syn_permanence = 0.0f;
    float connect_permanence = 0.0f;

    int delay_length = 1;
  };

  explicit TemporalPoolerCalculator(const Config& cfg);

  // Proximal temporal pooling.
  //
  // Inputs:
  // - col_pot_inputs01: (num_columns, num_pot_synapses) 0/1 values (from overlap stage)
  // - col_active01    : (num_columns) 0/1 values (from inhibition stage)
  // - burst_cols_time : (num_columns, 2) last 2 timesteps column was bursting (from active-cells stage)
  //
  // Mutates:
  // - col_syn_perm: (num_columns, num_pot_synapses) permanence values
  void update_proximal(int time_step,
                       const std::vector<int>& col_pot_inputs01,
                       const std::vector<uint8_t>& col_active01,
                       std::vector<float>& col_syn_perm,
                       const std::vector<int>& burst_cols_time);

  // Distal temporal pooling.
  //
  // Inputs:
  // - new_learn_cells_list: variable-length list of (col, cell) learning cells at `time_step`
  // - learn_cells_time    : (num_columns, cells_per_column, 2) time-history
  // - active_cells_time   : (num_columns, cells_per_column, 2) time-history
  // - active_segs_time    : (num_columns, cells_per_column, max_segments_per_cell) time-history
  //
  // Mutates:
  // - predict_cells_time: (num_columns, cells_per_column, 2) time-history (persistence extension)
  // - distal_synapses   : (num_columns, cells_per_column, max_segments_per_cell, max_synapses_per_segment)
  void update_distal(int time_step,
                     const std::vector<std::pair<int, int>>& new_learn_cells_list,
                     const std::vector<int>& learn_cells_time,
                     std::vector<int>& predict_cells_time,
                     const std::vector<int>& active_cells_time,
                     const std::vector<int>& active_segs_time,
                     std::vector<sequence_pooler::DistalSynapse>& distal_synapses);

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }

  inline int idx_cell_seg(int col, int cell, int seg) const {
    return (col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell + seg;
  }

  inline int idx_cell_flat(int col, int cell) const { return col * cfg_.cells_per_column + cell; }

  bool check_col_bursting(const std::vector<int>& burst_cols_time, int col, int time_step) const;
  bool check_cell_time(const std::vector<int>& cells_time, int col, int cell, int time_step) const;

  bool check_cell_predict(const std::vector<int>& predict_cells_time, int col, int cell, int time_step) const;
  void set_predict_cell(std::vector<int>& predict_cells_time, int col, int cell, int time_step) const;

  bool check_cell_active_predict(const std::vector<int>& active_cells_time,
                                 const std::vector<int>& predict_cells_time,
                                 int col,
                                 int cell,
                                 int time_step) const;

  void update_avg_persist(int prev_tracking_num, float& avg_persist) const;

  void update_new_learn_cells_time(int time_step,
                                   const std::vector<std::pair<int, int>>& new_learn_cells_list,
                                   const std::vector<int>& learn_cells_time);

  std::vector<std::pair<int, int>> get_prev2_new_learn_cells(int time_step,
                                                             const std::vector<int>& active_cells_time,
                                                             int num_cells_needed) const;

  int segment_num_synapses_active_prev2(const std::vector<sequence_pooler::DistalSynapse>& distal_synapses,
                                        const std::unordered_set<int>& prev2_set,
                                        int origin_col,
                                        int origin_cell,
                                        int seg) const;

  int get_best_matching_segment_prev2(const std::vector<sequence_pooler::DistalSynapse>& distal_synapses,
                                      const std::unordered_set<int>& prev2_set,
                                      int origin_col,
                                      int origin_cell) const;

  void get_segment_active_synapses(const std::vector<sequence_pooler::DistalSynapse>& distal_synapses,
                                   const std::vector<int>& active_cells_time,
                                   int time_step,
                                   int origin_col,
                                   int origin_cell,
                                   int seg,
                                   int8_t* out01) const;

  void update_distal_synapses(int origin_col,
                              int origin_cell,
                              int seg,
                              const int8_t* seg_active_syn01,
                              std::vector<sequence_pooler::DistalSynapse>& distal_synapses) const;

  int find_least_used_seg(const std::vector<int>& active_segs_time, int origin_col, int origin_cell) const;

  void overwrite_segment_with_prev2(int time_step,
                                    int origin_col,
                                    int origin_cell,
                                    int seg,
                                    const std::vector<std::pair<int, int>>& prev2_cells,
                                    std::vector<sequence_pooler::DistalSynapse>& distal_synapses) const;

  static std::uint64_t splitmix64(std::uint64_t x);
  static std::size_t deterministic_pick(std::uint64_t seed, std::size_t mod);

  Config cfg_;

  // Proximal state (previous timestep buffers).
  std::vector<int8_t> prev_col_pot_inputs_; // (num_columns, num_pot_synapses)
  std::vector<uint8_t> prev_col_active_;   // (num_columns)

  // Distal persistence state.
  std::vector<int> cells_tracking_num_;   // (num_columns*cells_per_column)
  std::vector<float> cells_avg_persist_;  // (num_columns*cells_per_column)
  std::vector<int> cells_persistence_;    // (num_columns*cells_per_column)

  // Track last 2 timesteps when a cell first entered learning state.
  std::vector<int> new_learn_cells_time_; // (num_columns, cells_per_column, 2)

  // Append-only log of learning-entry events (time_step, col, cell).
  // Used to avoid O(N log N) sorting on huge grids each timestep.
  std::vector<std::tuple<int, int, int>> learn_entry_log_;

  tf::Executor executor_;
};

} // namespace temporal_pooler

