#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <taskflow/taskflow.hpp>

#include <htm_flow/sequence_pooler/sequence_types.hpp>

namespace sequence_pooler {

// Active-cells stage of the sequence pooler.
//
// What this component does:
// - Given the set of *active columns* at `time_step`, decide which *cells* inside those
//   columns become active and which become the learning cell.
//
// When a column becomes active:
// - If any cells were correctly predicted (predictive at t-1 with an active sequence
//   segment), ALL such cells become active and enter learning. Multiple cells per column
//   can be predicted simultaneously, which naturally handles overlapping sequences sharing
//   common column activations.
// - If no cell was predicted, the column "bursts" (all cells become active), representing
//   ambiguity about which temporal context the column belongs to.
//
// What this component returns:
// - A variable-length list of active cells: [(col, cell), ...]
// - A variable-length list of learning cells: [(col, cell), ...]
//   Note: there can be more than one learning cell per active column when multiple
//   cells correctly predicted via different active (sequence) segments.
// - Time-history tensors storing the last two timesteps a cell was active / learning.
//
// Notes:
// - This stage is fully wired to receive `predict_cells_time`, `active_segs_time`, and
//   `distal_synapses`.
// - The implementation includes the best-matching-cell logic used to select learning
//   cells and create update structures for the sequence-learning stage.
class ActiveCellsCalculator {
public:
  struct Config {
    // --- Shape / topology ---
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;

    // --- Thresholds used by segment matching ---
    // More than this many *connected* synapses in a segment must be active for that segment
    // to be considered a "match" (used when searching for a best-matching segment).
    int min_num_syn_threshold = 0;

    // --- Distal permanence parameters ---
    // Starting permanence used when proposing/creating new distal synapses.
    float new_syn_permanence = 0.0f;

    // Permanence threshold above which a distal synapse is considered connected.
    float connect_permanence = 0.0f;
  };

  explicit ActiveCellsCalculator(const Config& cfg);

  // Update active/learn states for the given timestep.
  //
  // Implementation outline:
  // - For each active column:
  //   - If the column was already active last timestep, preserve continuity (keep the same active/learn
  //     cell, or keep bursting if it was bursting).
  //   - If the column is newly active:
  //     - If any cell was correctly predicted via an active sequence segment, activate it (can be multiple).
  //     - Else burst (all cells active).
  // - If no learning cell was chosen, select a best-matching cell/segment and emit update structures for
  //   the sequence-learning stage to reinforce / grow distal connections.
  void calculate_active_cells(int time_step,
                              const std::vector<int>& active_col_indices,
                              const std::vector<int>& predict_cells_time,
                              const std::vector<int>& active_segs_time,
                              const std::vector<DistalSynapse>& distal_synapses);

  // --- Outputs / state accessors ---
  const std::vector<std::pair<int, int>>& get_current_active_cells_list() const;
  const std::vector<std::pair<int, int>>& get_current_learn_cells_list() const;

  // Flattened tensors of shape:
  // - active_cells_time: (num_columns, cells_per_column, 2)
  // - learn_cells_time : (num_columns, cells_per_column, 2)
  const std::vector<int>& get_active_cells_time() const;
  const std::vector<int>& get_learn_cells_time() const;
  // burst_cols_time: (num_columns, 2)
  const std::vector<int>& get_burst_cols_time() const;

  // --- Update-structure tensors for sequence learning (active-cells contribution) ---
  // segIndUpdateActive: (num_columns, cells_per_column) -> segment index to update, or -1
  // segActiveSynActive: (num_columns, cells_per_column, max_synapses_per_segment) -> 0/1 list
  // segIndNewSynActive: (num_columns, cells_per_column) -> segment index to overwrite/create synapses, or -1
  // segNewSynActive   : (num_columns, cells_per_column, max_synapses_per_segment) -> proposed synapses
  std::vector<int>& get_seg_ind_update_active();
  std::vector<int8_t>& get_seg_active_syn_active();
  std::vector<int>& get_seg_ind_new_syn_active();
  std::vector<DistalSynapse>& get_seg_new_syn_active();

  // Shape helpers
  int num_columns() const { return cfg_.num_columns; }
  int cells_per_column() const { return cfg_.cells_per_column; }

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }
  inline int idx_col_time(int col, int slot) const { return col * 2 + slot; }
  inline int idx_cell_seg(int col, int cell, int seg) const {
    return (col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell + seg;
  }

  bool check_cell_active(int col, int cell, int time_step) const;
  bool check_cell_learn(int col, int cell, int time_step) const;
  bool check_col_bursting(int col, int time_step) const;
  bool check_col_prev_active(int col) const;

  void set_active_cell(int col, int cell, int time_step);
  void set_learn_cell(int col, int cell, int time_step);
  void set_burst_col(int col, int time_step);

  int find_active_cell(int col, int time_step) const; // returns first active cell, or -1
  int find_learn_cell(int col, int time_step) const;  // returns first learn cell, or -1

  bool check_cell_predicting(const std::vector<int>& predict_cells_time,
                             int col,
                             int cell,
                             int time_step) const;
  bool check_cell_has_sequence_seg(const std::vector<int>& active_segs_time,
                                   int col,
                                   int cell,
                                   int time_step_minus_1) const;

  // --- Best-matching logic (used for learning cell selection) ---
  int segment_num_synapses_active(const std::vector<DistalSynapse>& distal_synapses,
                                  int origin_col,
                                  int origin_cell,
                                  int seg,
                                  int time_step,
                                  bool on_cell) const;
  int get_best_matching_segment(const std::vector<DistalSynapse>& distal_synapses,
                                int origin_col,
                                int origin_cell,
                                int time_step,
                                bool on_cell) const; // -1 if none active enough
  int find_num_segs(const std::vector<DistalSynapse>& distal_synapses,
                    int origin_col,
                    int origin_cell) const;
  int find_least_used_seg(const std::vector<int>& learn_segs_time,
                          int origin_col,
                          int origin_cell) const;
  // Returns: (cell, seg, bestCellFound)
  std::tuple<int, int, bool> get_best_matching_cell(const std::vector<DistalSynapse>& distal_synapses,
                                                    const std::vector<int>& active_segs_time,
                                                    int origin_col,
                                                    int time_step) const;
  void get_segment_active_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                   int origin_col,
                                   int origin_cell,
                                   int seg,
                                   int time_step,
                                   int8_t* out01) const;
  void new_random_prev_learn_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                      int origin_col,
                                      int origin_cell,
                                      int seg,
                                      int time_step,
                                      bool keep_connected_syn,
                                      DistalSynapse* out_new_syn) const;

  Config cfg_;

  // Previous timestep active-columns bitfield (0/1), as produced by inhibition.
  std::vector<uint8_t> prev_active_cols_; // shape: (num_columns)

  // Track last 2 timesteps a column was bursting (all cells active).
  std::vector<int> burst_cols_time_; // shape: (num_columns, 2)

  // Track last 2 timesteps each cell was active / learning.
  std::vector<int> active_cells_time_; // shape: (num_columns, cells_per_column, 2)
  std::vector<int> learn_cells_time_;  // shape: (num_columns, cells_per_column, 2)

  // Current timestep outputs as lists of (col, cell) pairs.
  std::vector<std::pair<int, int>> current_active_cells_list_;
  std::vector<std::pair<int, int>> current_learn_cells_list_;
  std::vector<std::pair<int, int>> prev_learn_cells_list_;
  // Last timestep each segment was selected as a learning target.
  std::vector<int> learn_segs_time_;

  // Update structures for the sequence learning stage (active-cells side).
  std::vector<int> seg_ind_update_active_;        // (num_columns*cells_per_column)
  std::vector<int8_t> seg_active_syn_active_;     // (num_columns*cells_per_column*max_synapses_per_segment)
  std::vector<int> seg_ind_new_syn_active_;       // (num_columns*cells_per_column)
  std::vector<DistalSynapse> seg_new_syn_active_; // (num_columns*cells_per_column*max_synapses_per_segment)

  // Executor reused across calls (like other calculators).
  tf::Executor executor_;
};

} // namespace sequence_pooler

