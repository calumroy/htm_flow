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

  ///-----------------------------------------------------------------------------
  ///
  /// get_current_active_cells_list - Returns the (col, cell) pairs active this timestep.
  ///
  /// Downstream stages (prediction, sequence learning) need to know exactly which
  /// cells fired so they can form / reinforce distal connections.
  ///
  ///-----------------------------------------------------------------------------
  const std::vector<std::pair<int, int>>& get_current_active_cells_list() const;

  ///-----------------------------------------------------------------------------
  ///
  /// get_current_learn_cells_list - Returns the (col, cell) pairs selected for
  ///                                learning this timestep.
  ///
  /// The sequence-learning stage reinforces distal segments only on learning
  /// cells, so this list drives all permanence updates.
  ///
  ///-----------------------------------------------------------------------------
  const std::vector<std::pair<int, int>>& get_current_learn_cells_list() const;

  ///-----------------------------------------------------------------------------
  ///
  /// get_active_cells_time - Flattened tensor (num_columns, cells_per_column, 2)
  ///                         storing the last two timesteps each cell was active.
  ///
  /// Two slots let the calculator distinguish "active now" from "active last step"
  /// without clearing the entire tensor every timestep.
  ///
  ///-----------------------------------------------------------------------------
  const std::vector<int>& get_active_cells_time() const;

  ///-----------------------------------------------------------------------------
  ///
  /// get_learn_cells_time - Same two-slot layout as active_cells_time, but for
  ///                        learning state.
  ///
  /// Used by best-matching-cell search and by the sequence-learning stage to
  /// identify which cells were learning at t-1 for synapse proposals.
  ///
  ///-----------------------------------------------------------------------------
  const std::vector<int>& get_learn_cells_time() const;

  ///-----------------------------------------------------------------------------
  ///
  /// get_burst_cols_time - Tensor (num_columns, 2) storing the last two timesteps
  ///                       a column was bursting (all cells active).
  ///
  /// Columns that remain active across timesteps use the burst flag to decide
  /// whether to preserve a single active cell or keep all cells firing.
  ///
  ///-----------------------------------------------------------------------------
  const std::vector<int>& get_burst_cols_time() const;

  ///-----------------------------------------------------------------------------
  ///
  /// get_seg_ind_update_active - Per-cell segment index whose existing synapses
  ///                             should be reinforced / punished by sequence learning.
  ///                             -1 means no update needed for that cell.
  ///
  /// Shape: (num_columns * cells_per_column)
  ///
  ///-----------------------------------------------------------------------------
  std::vector<int>& get_seg_ind_update_active();

  ///-----------------------------------------------------------------------------
  ///
  /// get_seg_active_syn_active - Binary mask (0/1) indicating which synapses on
  ///                             the update segment were active at the current step.
  ///
  /// Sequence learning uses this to selectively strengthen active synapses and
  /// weaken inactive ones on the chosen segment.
  ///
  /// Shape: (num_columns * cells_per_column * max_synapses_per_segment)
  ///
  ///-----------------------------------------------------------------------------
  std::vector<int8_t>& get_seg_active_syn_active();

  ///-----------------------------------------------------------------------------
  ///
  /// get_seg_ind_new_syn_active - Per-cell segment index where brand-new synapses
  ///                              should be written. -1 means no new synapses needed.
  ///
  /// Shape: (num_columns * cells_per_column)
  ///
  ///-----------------------------------------------------------------------------
  std::vector<int>& get_seg_ind_new_syn_active();

  ///-----------------------------------------------------------------------------
  ///
  /// get_seg_new_syn_active - Proposed new distal synapses (target col/cell + permanence)
  ///                          for the segment indicated by seg_ind_new_syn_active.
  ///
  /// Entries with perm < 0 are placeholders meaning "no proposal for this slot".
  ///
  /// Shape: (num_columns * cells_per_column * max_synapses_per_segment)
  ///
  ///-----------------------------------------------------------------------------
  std::vector<DistalSynapse>& get_seg_new_syn_active();

  ///-----------------------------------------------------------------------------
  ///
  /// num_columns / cells_per_column - Expose topology so callers can size
  ///                                  buffers or iterate without caching Config.
  ///
  ///-----------------------------------------------------------------------------
  int num_columns() const { return cfg_.num_columns; }
  int cells_per_column() const { return cfg_.cells_per_column; }
  void set_min_num_syn_threshold(int threshold) { cfg_.min_num_syn_threshold = threshold; }
  void set_new_syn_permanence(float permanence) { cfg_.new_syn_permanence = permanence; }
  void set_connect_permanence(float permanence) { cfg_.connect_permanence = permanence; }

private:
  // --- Flat-index helpers ------------------------------------------------
  // All state tensors are stored as 1D vectors; these convert (col, cell, slot/seg)
  // coordinates into flat offsets so the rest of the code stays readable.
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }
  inline int idx_col_time(int col, int slot) const { return col * 2 + slot; }
  inline int idx_cell_seg(int col, int cell, int seg) const {
    return (col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell + seg;
  }

  /// True if the given cell was marked active at @p time_step.
  /// Used throughout to test both current and previous-step activity.
  bool check_cell_active(int col, int cell, int time_step) const;

  /// True if the given cell was marked as a learning cell at @p time_step.
  bool check_cell_learn(int col, int cell, int time_step) const;

  /// True if the column was bursting (all cells active) at @p time_step.
  /// Continuity logic needs this to decide whether to preserve a single
  /// active cell or keep firing all cells.
  bool check_col_bursting(int col, int time_step) const;

  /// True if the column appeared in the previous timestep's active-columns set.
  /// Distinguishes "still active" columns from "newly active" ones, which
  /// follow different activation paths.
  bool check_col_prev_active(int col) const;

  /// Record a cell as active at @p time_step, writing into the older of
  /// the two time-history slots so the newer slot is preserved.
  void set_active_cell(int col, int cell, int time_step);

  /// Record a cell as the learning cell at @p time_step (same slot strategy).
  void set_learn_cell(int col, int cell, int time_step);

  /// Record a column as bursting at @p time_step.
  void set_burst_col(int col, int time_step);

  /// Return the first cell in the column that is active at @p time_step, or -1.
  /// Used by the continuity path to carry forward the previously active cell.
  int find_active_cell(int col, int time_step) const;

  /// Return the first learning cell in the column at @p time_step, or -1.
  int find_learn_cell(int col, int time_step) const;

  /// True if the cell was in the predictive state at @p time_step, meaning
  /// the prediction stage expected it to fire next.
  bool check_cell_predicting(const std::vector<int>& predict_cells_time,
                             int col,
                             int cell,
                             int time_step) const;

  /// True if any of the cell's distal segments was active at @p time_step_minus_1.
  /// Combined with check_cell_predicting this confirms the cell was
  /// *correctly* predicted via a sequence segment (not just any context).
  bool check_cell_has_sequence_seg(const std::vector<int>& active_segs_time,
                                   int col,
                                   int cell,
                                   int time_step_minus_1) const;

  // --- Best-matching logic (used for learning cell selection) -------------

  /// Count how many synapses with positive permanence on a segment end on cells
  /// (or columns) that were active at t-1. Uses perm > 0 (not connect_permanence)
  /// so sub-connected synapses contribute to matching for learning. The prediction
  /// path (predict_cells) uses the connected threshold separately.
  /// The @p on_cell flag switches between cell-level and column-level matching.
  int segment_num_synapses_active(const std::vector<DistalSynapse>& distal_synapses,
                                  int origin_col,
                                  int origin_cell,
                                  int seg,
                                  int time_step,
                                  bool on_cell) const;

  /// Return the segment on the cell with the most active connected synapses,
  /// provided it exceeds min_num_syn_threshold. Returns -1 otherwise.
  /// This identifies the segment that best "matches" the current context,
  /// which is needed to decide where to reinforce learning.
  int get_best_matching_segment(const std::vector<DistalSynapse>& distal_synapses,
                                int origin_col,
                                int origin_cell,
                                int time_step,
                                bool on_cell) const;

  /// Count how many segments the cell has ever used (learn_segs_time >= 0).
  /// Used as a tiebreaker: cells with fewer segments are preferred for new
  /// learning so that capacity is spread evenly.
  int find_num_segs(int origin_col, int origin_cell) const;

  /// Return the segment with the oldest learning timestamp. When all
  /// segments are occupied, the least-recently-used one is recycled.
  int find_least_used_seg(const std::vector<int>& learn_segs_time,
                          int origin_col,
                          int origin_cell) const;

  /// Select the best cell + segment in a column for learning.
  /// Prefers cells that already have a segment matching the current context;
  /// falls back to the cell with the fewest / least-recently-used segments.
  ///
  /// @return (cell, seg, bestCellFound) where bestCellFound is false when
  ///         no segment exceeded the matching threshold.
  std::tuple<int, int, bool> get_best_matching_cell(const std::vector<DistalSynapse>& distal_synapses,
                                                    const std::vector<int>& active_segs_time,
                                                    int origin_col,
                                                    int time_step) const;

  /// Write a 0/1 mask into @p out01 indicating which synapses on the segment
  /// have positive permanence and target a currently-active cell.
  /// Sequence learning uses this to know which synapses to reinforce.
  void get_segment_active_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                   int origin_col,
                                   int origin_cell,
                                   int seg,
                                   int time_step,
                                   int8_t* out01) const;

  /// Propose new distal synapses targeting cells that were learning at t-1.
  /// If @p keep_connected_syn is true, only weak (below threshold) synapse
  /// slots are overwritten; otherwise the entire segment is replaced.
  /// This is how the network grows new connections to represent context.
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

