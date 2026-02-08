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
    // --- Shape / topology ---
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;
    int num_pot_synapses = 0;

    // --- Learning rates ---
    // The value by which proximal (column) synapse permanence values change by in the temporal pooler.
    float spatial_permanence_inc = 0.0f;
    // The value by which distal (cell) synapse permanence values are incremented for active synapses.
    float seq_permanence_inc = 0.0f;
    // The value by which distal (cell) synapse permanence values are decremented for inactive synapses
    // (synapses whose target cell is NOT active when the segment is reinforced).
    // When a synapse's permanence decays to 0, it is replaced with a new synapse targeting a recent
    // learning cell, keeping segments tuned to the current temporal context.
    float seq_permanence_dec = 0.0f;

    // --- Thresholds / permanence constants ---
    // More than this many synapses in a segment must be "active" for that segment to be considered
    // a best-matching segment (for reinforcement / selection).
    int min_num_syn_threshold = 0;
    // The starting permanence for newly created distal synapses.
    float new_syn_permanence = 0.0f;
    // The minimum permanence required for a distal synapse to be considered connected.
    float connect_permanence = 0.0f;

    // delay_length is a parameter for updating the average persistence count.
    // It determines how quickly the average persistence count changes (smoothing factor).
    //
    // Important:
    // - This does NOT directly enable/disable persistence. It only affects the smoothing of the average.
    // - To disable persistence-based *predictive extension* (while keeping distal reinforcement), use
    //   `enable_persistence = false`.
    int delay_length = 1;

    // If true, extend predictive state for a short "persistence window" even when no segment is active.
    //
    // Motivation / intent (matches python `np_temporal.py`):
    // - Cells that have been correctly predicted for several consecutive steps tend to stay active for a while.
    // - We track streak length ("active_predict" streak) and learn a smoothed average.
    // - When a streak ends, we allow the cell to remain predictive for a small number of steps ("coast"),
    //   which helps temporal continuity through brief gaps/noise.
    //
    // If false:
    // - The temporal pooler still reinforces / creates distal synapses for active-predict cells.
    // - But it will NOT mutate `predict_cells_time` to keep cells predicting just due to persistence.
    bool enable_persistence = true;
  };

  explicit TemporalPoolerCalculator(const Config& cfg);

  // -----------------------------------------------------------------------------
  // update_proximal (ported from python `updateProximalTempPool`)
  //
  // Update proximal synapses (column synapses) such that:
  //   a. For each currently active column, increment permanence values of potential
  //      synapses that were connected to an active input one timestep ago.
  //      Do not do this for bursting columns.
  //   b. For each column that was active one timestep ago, increment permanence values
  //      of potential synapses that are connected to an active input now.
  //
  // Inputs:
  //   1. col_pot_inputs01
  //      A 2D tensor (flattened) storing for each column's potential proximal synapses
  //      whether its end is connected to an active input.
  //      Shape: (num_columns, num_pot_synapses). Values: 0/1.
  //
  //   2. col_active01
  //      A 1D tensor storing if a column is active.
  //      Shape: (num_columns). Values: 0/1.
  //
  //   3. col_syn_perm
  //      A 2D tensor (flattened) storing the permanence values of every potential synapse
  //      for each column.
  //      Shape: (num_columns, num_pot_synapses). Values: [0,1].
  //
  //   4. time_step
  //      Current timestep (monotonically increasing int).
  //
  //   5. burst_cols_time
  //      A 2D tensor (flattened) storing the last two timesteps when a column was bursting.
  //      Shape: (num_columns, 2).
  //
  // State used:
  //   - prev_col_pot_inputs_ (previous timestep's potential-input activity)
  //   - prev_col_active_     (previous timestep's column active bits)
  //
  // Output:
  //   - Updates `col_syn_perm` in-place and refreshes the prev_* buffers for next timestep.
  // -----------------------------------------------------------------------------
  void update_proximal(int time_step,
                       const std::vector<int>& col_pot_inputs01,
                       const std::vector<uint8_t>& col_active01,
                       std::vector<float>& col_syn_perm,
                       const std::vector<int>& burst_cols_time);

  // -----------------------------------------------------------------------------
  // update_distal (ported from python `updateDistalTempPool`)
  //
  // Update distal synapses (cell synapses) such that:
  //   - Track per-cell "active-predict streaks" and maintain a smoothed average persistence.
  //   - If a cell was predicting/active last timestep and has remaining persistence, keep it
  //     in the predictive state even without active segments.
  //   - If a cell is "active predictive" (was predicting at t-1 and became active at t),
  //     then reinforce a best-matching segment (based on antepenultimate learning cells),
  //     otherwise create/overwrite a new segment.
  //
  // Inputs:
  //   1. time_step
  //      Current timestep (monotonically increasing int).
  //
  //   2. new_learn_cells_list
  //      Variable-length list of cells currently in the learning state for this timestep.
  //      Elements are (col, cell).
  //
  //   3. learn_cells_time
  //      Time-history tensor: last two timesteps each cell was in the learning state.
  //      Shape: (num_columns, cells_per_column, 2).
  //
  //   4. predict_cells_time
  //      Time-history tensor: last two timesteps each cell was in the predictive state.
  //      Shape: (num_columns, cells_per_column, 2).
  //      NOTE: this function MUTATES it to extend predictions via persistence.
  //
  //   5. active_cells_time
  //      Time-history tensor: last two timesteps each cell was active.
  //      Shape: (num_columns, cells_per_column, 2).
  //
  //   6. active_segs_time
  //      Segment time-history: last timestep each segment was active (sequence segment).
  //      Shape: (num_columns, cells_per_column, max_segments_per_cell).
  //
  //   7. distal_synapses
  //      Distal synapse tensor (flattened 5D):
  //        distal_synapses[col][cell][seg][syn] == {target_col, target_cell, perm}
  //      Shape: (num_columns, cells_per_column, max_segments_per_cell, max_synapses_per_segment).
  //      NOTE: this function MUTATES it (reinforcement / segment overwrite).
  //
  // State updated:
  //   - cells_tracking_num_, cells_avg_persist_, cells_persistence_
  //   - new_learn_cells_time_ / learn_entry_log_ (used to derive prev2 learning cells)
  // -----------------------------------------------------------------------------
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
  // These exist so `update_proximal` can look at what was active on the previous timestep
  // while processing the current timestep (matches python `prevColPotInputs`, `prevColActive`).
  std::vector<int8_t> prev_col_pot_inputs_; // (num_columns, num_pot_synapses)
  std::vector<uint8_t> prev_col_active_;   // (num_columns)

  // Distal persistence state.
  // These implement the "persistence" idea from the python temporal pooler:
  // - Track how long a cell continues to be correctly predicted (active_predict streak)
  // - Maintain a smoothed average streak length
  // - Convert that into a countdown that keeps cells predicting for a short window even
  //   when no segment is currently active (helps temporal continuity).
  std::vector<int> cells_tracking_num_;   // (num_columns*cells_per_column)
  std::vector<float> cells_avg_persist_;  // (num_columns*cells_per_column)
  std::vector<int> cells_persistence_;    // (num_columns*cells_per_column)

  // Track last 2 timesteps when a cell first entered learning state.
  // Important: this stores *entry* timesteps only (not subsequent timesteps where the cell
  // remains learning), matching the python comment for `newLearnCellsTime`.
  std::vector<int> new_learn_cells_time_; // (num_columns, cells_per_column, 2)

  // Append-only log of learning-entry events (time_step, col, cell).
  // Python implements `getPrev2NewLearnCells` by argsorting the entire `newLearnCellsTime` tensor
  // each timestep (O(N log N)). For large grids this is expensive.
  //
  // In C++ we keep an append-only log of learning-entry events and scan it backwards to obtain
  // the most recent entries. This preserves the intent (most-recent learning-entry cells) while
  // keeping the steady-state cost closer to O(#recent learning events).
  std::vector<std::tuple<int, int, int>> learn_entry_log_;

  tf::Executor executor_;
};

} // namespace temporal_pooler

