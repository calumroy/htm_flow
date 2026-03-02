#include <htm_flow/sequence_pooler/active_cells.hpp>

#include <algorithm>
#include <cassert>
#include <tuple>
#include <taskflow/algorithm/for_each.hpp>

namespace sequence_pooler {

bool ActiveCellsCalculator::check_cell_predicting(const std::vector<int>& predict_cells_time,
                                                  int col,
                                                  int cell,
                                                  int time_step) const {
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  return (predict_cells_time[i0] == time_step) || (predict_cells_time[i1] == time_step);
}

bool ActiveCellsCalculator::check_cell_has_sequence_seg(const std::vector<int>& active_segs_time,
                                                        int col,
                                                        int cell,
                                                        int time_step_minus_1) const {
  // activeSegsTime is a 3D tensor (col, cell, seg) storing the last timestep a segment was active.
  const int base = (col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell;
  for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
    if (active_segs_time[base + seg] == time_step_minus_1) {
      return true;
    }
  }
  return false;
}

ActiveCellsCalculator::ActiveCellsCalculator(const Config& cfg) : cfg_(cfg) {
  assert(cfg_.num_columns > 0);
  assert(cfg_.cells_per_column > 0);

  prev_active_cols_.assign(cfg_.num_columns, 0);
  burst_cols_time_.assign(cfg_.num_columns * 2, -1);
  active_cells_time_.assign(cfg_.num_columns * cfg_.cells_per_column * 2, -1);
  learn_cells_time_.assign(cfg_.num_columns * cfg_.cells_per_column * 2, -1);
  learn_segs_time_.assign(cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell, -1);

  // Sequence-learning update structures (active-cells side).
  seg_ind_update_active_.assign(cfg_.num_columns * cfg_.cells_per_column, -1);
  seg_active_syn_active_.assign(cfg_.num_columns * cfg_.cells_per_column * cfg_.max_synapses_per_segment, 0);
  seg_ind_new_syn_active_.assign(cfg_.num_columns * cfg_.cells_per_column, -1);
  seg_new_syn_active_.assign(cfg_.num_columns * cfg_.cells_per_column * cfg_.max_synapses_per_segment,
                             DistalSynapse{0, 0, -1.0f});
}

const std::vector<std::pair<int, int>>& ActiveCellsCalculator::get_current_active_cells_list() const {
  return current_active_cells_list_;
}

const std::vector<std::pair<int, int>>& ActiveCellsCalculator::get_current_learn_cells_list() const {
  return current_learn_cells_list_;
}

const std::vector<int>& ActiveCellsCalculator::get_active_cells_time() const {
  return active_cells_time_;
}

const std::vector<int>& ActiveCellsCalculator::get_learn_cells_time() const {
  return learn_cells_time_;
}

const std::vector<int>& ActiveCellsCalculator::get_burst_cols_time() const {
  return burst_cols_time_;
}

std::vector<int>& ActiveCellsCalculator::get_seg_ind_update_active() {
  return seg_ind_update_active_;
}
std::vector<int8_t>& ActiveCellsCalculator::get_seg_active_syn_active() {
  return seg_active_syn_active_;
}
std::vector<int>& ActiveCellsCalculator::get_seg_ind_new_syn_active() {
  return seg_ind_new_syn_active_;
}
std::vector<DistalSynapse>& ActiveCellsCalculator::get_seg_new_syn_active() {
  return seg_new_syn_active_;
}

// Python: checkCellActive
// Check if the given cell was active at the given timestep. The activeCellsTime tensor
// holds the last 2 timesteps when each cell was active; we check both slots.
bool ActiveCellsCalculator::check_cell_active(int col, int cell, int time_step) const {
  const int a0 = active_cells_time_[idx_cell_time(col, cell, 0)];
  const int a1 = active_cells_time_[idx_cell_time(col, cell, 1)];
  return (a0 == time_step) || (a1 == time_step);
}

// Python: checkCellLearn
// Check if the given cell was in the learning state at the given timestep. The learnCellsTime
// tensor holds the last 2 timesteps when each cell was last in the learn state.
bool ActiveCellsCalculator::check_cell_learn(int col, int cell, int time_step) const {
  const int a0 = learn_cells_time_[idx_cell_time(col, cell, 0)];
  const int a1 = learn_cells_time_[idx_cell_time(col, cell, 1)];
  return (a0 == time_step) || (a1 == time_step);
}

// Python: checkColBursting
// Check whether the column was bursting (all cells active) at the given timestep.
// Returns false early if any cell is not active.
bool ActiveCellsCalculator::check_col_bursting(int col, int time_step) const {
  int count = 0;
  for (int i = 0; i < cfg_.cells_per_column; ++i) {
    if (check_cell_active(col, i, time_step)) {
      ++count;
    } else {
      return false;
    }
  }
  return count == cfg_.cells_per_column;
}

// Python: checkColPrevActive
// Check that the column was active one timestep ago using the saved prevActiveCols bitfield.
bool ActiveCellsCalculator::check_col_prev_active(int col) const {
  return prev_active_cols_[static_cast<size_t>(col)] == 1;
}

// Python: setBurstCol
// Record that a column was bursting at the given timestep. The burstColsTime tensor holds
// the last 2 timesteps when each column was bursting; we overwrite the oldest slot.
void ActiveCellsCalculator::set_burst_col(int col, int time_step) {
  const int i0 = idx_col_time(col, 0);
  const int i1 = idx_col_time(col, 1);
  if (burst_cols_time_[i0] <= burst_cols_time_[i1]) {
    burst_cols_time_[i0] = time_step;
  } else {
    burst_cols_time_[i1] = time_step;
  }
}

// Python: setActiveCell
// Set the given cell into an active state for the given timestep. The activeCellsTime tensor
// holds the last 2 timesteps when each cell was active; we overwrite the oldest slot.
void ActiveCellsCalculator::set_active_cell(int col, int cell, int time_step) {
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  if (active_cells_time_[i0] <= active_cells_time_[i1]) {
    active_cells_time_[i0] = time_step;
  } else {
    active_cells_time_[i1] = time_step;
  }
}

// Python: setLearnCell
// Set the given cell into a learn state for the given timestep. The learnCellsTime tensor
// holds the last 2 timesteps when each cell was in the learn state; we overwrite the oldest slot.
void ActiveCellsCalculator::set_learn_cell(int col, int cell, int time_step) {
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  if (learn_cells_time_[i0] <= learn_cells_time_[i1]) {
    learn_cells_time_[i0] = time_step;
  } else {
    learn_cells_time_[i1] = time_step;
  }
}

// Python: findActiveCell
// Return the first cell index that was active in the column at the given timestep.
// If all cells are active (bursting), returns the first cell. Returns -1 if none found.
int ActiveCellsCalculator::find_active_cell(int col, int time_step) const {
  for (int i = 0; i < cfg_.cells_per_column; ++i) {
    if (check_cell_active(col, i, time_step)) {
      return i;
    }
  }
  return -1;
}

// Python: findLearnCell
// Return the first cell index that was in the learn state in the column at the given timestep.
// Returns -1 if no cell was in the learn state.
int ActiveCellsCalculator::find_learn_cell(int col, int time_step) const {
  for (int i = 0; i < cfg_.cells_per_column; ++i) {
    if (check_cell_learn(col, i, time_step)) {
      return i;
    }
  }
  return -1;
}

// Python: segmentNumSynapsesActive
//
// Find the number of active synapses for the previous timestep. A synapse is "active" if
// its endpoint is on a cell (or column) that was active at time_step - 1. Each synapse
// stores [target_col, target_cell, permanence].
//
// If on_cell is true, count synapses whose endpoint cell was active at time_step - 1.
// If on_cell is false, count synapses whose endpoint column was active (using prevActiveCols).
//
// Threshold difference from Python: the Python version gates on connectPermanence, meaning
// only connected synapses count. This C++ version uses perm > 0 so that sub-connected
// synapses also contribute to segment matching for learning. The prediction path
// (predict_cells::count_active_connected_synapses) still uses the connected threshold.
// This distinction allows new synapses created below connectPermanence to be found by
// matching and subsequently reinforced toward the connected threshold.
int ActiveCellsCalculator::segment_num_synapses_active(const std::vector<DistalSynapse>& distal_synapses,
                                                       int origin_col,
                                                       int origin_cell,
                                                       int seg,
                                                       int time_step,
                                                       bool on_cell) const {
  int count = 0;
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx =
        idx_distal_synapse(static_cast<std::size_t>(origin_col),
                           static_cast<std::size_t>(origin_cell),
                           static_cast<std::size_t>(seg),
                           static_cast<std::size_t>(syn),
                           static_cast<std::size_t>(cfg_.cells_per_column),
                           static_cast<std::size_t>(cfg_.max_segments_per_cell),
                           static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    if (s.perm > 0.0f) {
      if (on_cell) {
        if (check_cell_active(s.target_col, s.target_cell, time_step - 1)) {
          ++count;
        }
      } else {
        if (check_col_prev_active(s.target_col)) {
          ++count;
        }
      }
    }
  }
  return count;
}

// Python: getBestMatchingSegment
//
// Find the segment whose synapses were most active for the previous timestep — i.e. the
// segment that was "most predicting" this column would become active. We iterate through
// all segments of the cell and find the one with the highest count of active synapses
// (via segment_num_synapses_active, which checks activity at time_step - 1).
//
// This routine is "aggressive": it allows synapses with permanence below connectPermanence
// to count (perm > 0), but requires the total active synapse count to exceed
// min_num_syn_threshold for the segment to be considered a match.
//
// Returns the index of the best matching segment, or -1 if no segment had enough active
// synapses to exceed the threshold.
int ActiveCellsCalculator::get_best_matching_segment(const std::vector<DistalSynapse>& distal_synapses,
                                                     int origin_col,
                                                     int origin_cell,
                                                     int time_step,
                                                     bool on_cell) const {
  int best_seg = 0;
  int most_active = -1;
  for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
    const int cnt = segment_num_synapses_active(distal_synapses, origin_col, origin_cell, seg, time_step, on_cell);
    if (cnt > most_active) {
      most_active = cnt;
      best_seg = seg;
    }
  }
  if (most_active > cfg_.min_num_syn_threshold) {
    return best_seg;
  }
  return -1;
}

// Python: findNumSegs
// Find the number of segments in a cell that have been used (i.e. have a learn_segs_time >= 0,
// meaning they were selected as a learning target at some point). Segments that have never
// been used still have their initial time of -1.
int ActiveCellsCalculator::find_num_segs(int origin_col, int origin_cell) const {
  const int base = (origin_col * cfg_.cells_per_column + origin_cell) * cfg_.max_segments_per_cell;
  int num = 0;
  for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
    if (learn_segs_time_[static_cast<size_t>(base + seg)] >= 0) {
      ++num;
    }
  }
  return num;
}

// Python: findLeastUsedSeg
// Find the most unused segment from the cell's list of previous learning times for each
// segment. Returns the index of the segment with the oldest (smallest) learning timestamp.
// When all segments are occupied, the least-recently-used one is recycled for new learning.
int ActiveCellsCalculator::find_least_used_seg(const std::vector<int>& learn_segs_time,
                                              int origin_col,
                                              int origin_cell) const {
  const int base = (origin_col * cfg_.cells_per_column + origin_cell) * cfg_.max_segments_per_cell;
  int least_seg = 0;
  int oldest = learn_segs_time[static_cast<size_t>(base)];
  for (int seg = 1; seg < cfg_.max_segments_per_cell; ++seg) {
    const int t = learn_segs_time[static_cast<size_t>(base + seg)];
    if (t < oldest) {
      oldest = t;
      least_seg = seg;
    }
  }
  return least_seg;
}

// Python: getBestMatchingCell
//
// Return the cell and segment that is "most matching" in the column. This finds the cell
// whose segment best matches the previous temporal context (cells active at time_step - 1).
//
// If a cell has a matching segment (one with more than min_num_syn_threshold active synapses),
// return that cell/segment with best_cell_found = true.
//
// If no cell has a matching segment, fall back to the cell with the fewest used segments.
// If there is a tie in segment count, prefer the cell whose least-used segment has the
// oldest learning timestamp — this spreads new learning across cells evenly. Return that
// cell's least-used segment with best_cell_found = false, signaling that the caller should
// create an entirely new set of synapses for this segment rather than reinforcing existing ones.
std::tuple<int, int, bool> ActiveCellsCalculator::get_best_matching_cell(const std::vector<DistalSynapse>& distal_synapses,
                                                                         const std::vector<int>& active_segs_time,
                                                                         int origin_col,
                                                                         int time_step) const {
  bool best_cell_found = false;
  int best_cell = 0;
  int best_seg = 0;
  int num_act_best = 0;

  // Fallback tracking: the cell with the fewest segments (and oldest least-used segment as tiebreaker).
  int cell_least_used_seg = 0;
  int fewest_segs = 0;
  int seg_least_used = 0;
  int least_used_time = time_step;

  for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
    // Track the cell with the fewest used segments for the fallback path.
    const int num_segs = find_num_segs(origin_col, cell);
    if (cell == 0 || num_segs < fewest_segs) {
      cell_least_used_seg = cell;
      fewest_segs = num_segs;
      seg_least_used = find_least_used_seg(learn_segs_time_, origin_col, cell);
      const int base = (origin_col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell;
      least_used_time = learn_segs_time_[static_cast<size_t>(base + seg_least_used)];
    } else if (num_segs == fewest_segs) {
      // Tie in segment count: prefer the cell whose least-used segment is older.
      const int candidate_seg = find_least_used_seg(learn_segs_time_, origin_col, cell);
      const int base = (origin_col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell;
      const int candidate_time = learn_segs_time_[static_cast<size_t>(base + candidate_seg)];
      if (candidate_time < least_used_time) {
        cell_least_used_seg = cell;
        seg_least_used = candidate_seg;
        least_used_time = candidate_time;
      }
    }

    // Try to find the segment that was most predicting for the previous input.
    const int h = get_best_matching_segment(distal_synapses, origin_col, cell, time_step, /*on_cell=*/true);
    if (h != -1) {
      // Use >= so that cell 0 segment 0 can be chosen as best (matching Python behavior).
      const int num_act = segment_num_synapses_active(distal_synapses, origin_col, cell, h, time_step, /*on_cell=*/true);
      if (num_act >= num_act_best) {
        best_cell = cell;
        best_seg = h;
        num_act_best = num_act;
        best_cell_found = true;
      }
    }
  }

  if (best_cell_found) {
    return {best_cell, best_seg, true};
  }
  return {cell_least_used_seg, seg_least_used, false};
}

// Python: getSegmentActiveSynapses
//
// Find which synapses in the segment are "active" for the given timestep. A synapse is
// active if its endpoint cell was active at the given timestep. Each synapse stores
// [target_col, target_cell, permanence].
//
// Returns a 0/1 mask in out01: 1 means the synapse's endpoint was active, 0 means inactive.
// Sequence learning uses this mask to decide which permanences to increment (active) vs
// decrement (inactive).
//
// Permanence threshold: uses perm > 0 (not connect_permanence) so that sub-connected
// synapses targeting active cells also get marked and reinforced by sequence learning.
// This matches the temporal pooler's get_segment_active_synapses behavior.
//
// Timestep note: the Python version passes timeStep (current) here, which works because
// the Python loop is sequential and earlier columns already have their cells set at T.
// The C++ version runs in parallel (taskflow), so the caller passes time_step - 1 to
// get a fully-computed, deterministic activity snapshot matching the context that
// segment_num_synapses_active matched against.
void ActiveCellsCalculator::get_segment_active_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                                        int origin_col,
                                                        int origin_cell,
                                                        int seg,
                                                        int time_step,
                                                        int8_t* out01) const {
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx =
        idx_distal_synapse(static_cast<std::size_t>(origin_col),
                           static_cast<std::size_t>(origin_cell),
                           static_cast<std::size_t>(seg),
                           static_cast<std::size_t>(syn),
                           static_cast<std::size_t>(cfg_.cells_per_column),
                           static_cast<std::size_t>(cfg_.max_segments_per_cell),
                           static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    if (s.perm > 0.0f && check_cell_active(s.target_col, s.target_cell, time_step)) {
      out01[syn] = 1;
    } else {
      out01[syn] = 0;
    }
  }
}

// Python: newRandomPrevActiveSynapses
//
// Fill out_new_syn with a random selection of new synapses that connect to cells which were
// in the learn state one timestep ago (prev_learn_cells_list_). Each proposed synapse stores
// [target_col, target_cell, new_syn_permanence]. A permanence of -1 means "no proposal for
// this slot" — sequence learning will skip it.
//
// If keep_connected_syn is false, create new synapses for ALL slots in the segment
// (overwrite everything). This is used when no matching segment was found and we are
// writing an entirely new segment.
//
// If keep_connected_syn is true, only overwrite slots whose existing synapse has
// permanence <= 0 (dead synapses). This preserves all live synapses (including
// sub-connected ones with 0 < perm < connect_permanence) so that sequence learning
// can reinforce them toward the connected threshold. The Python version uses
// connectPermanence as the threshold here; the C++ uses perm <= 0 to protect
// sub-connected synapses from being replaced before they can be reinforced.
//
// Note: the C++ uses deterministic hashing instead of random.sample() for thread safety
// under taskflow parallel execution.
void ActiveCellsCalculator::new_random_prev_learn_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                                           int origin_col,
                                                           int origin_cell,
                                                           int seg,
                                                           int time_step,
                                                           bool keep_connected_syn,
                                                           DistalSynapse* out_new_syn) const {
  if (prev_learn_cells_list_.empty()) {
    for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
      out_new_syn[syn] = DistalSynapse{0, 0, -1.0f};
    }
    return;
  }

  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    bool should_write = !keep_connected_syn;
    if (keep_connected_syn) {
      const std::size_t idx =
          idx_distal_synapse(static_cast<std::size_t>(origin_col),
                             static_cast<std::size_t>(origin_cell),
                             static_cast<std::size_t>(seg),
                             static_cast<std::size_t>(syn),
                             static_cast<std::size_t>(cfg_.cells_per_column),
                             static_cast<std::size_t>(cfg_.max_segments_per_cell),
                             static_cast<std::size_t>(cfg_.max_synapses_per_segment));
      // Protect all synapses with positive permanence (not just connected ones).
      // Sub-connected synapses need to survive so sequence learning can reinforce them.
      should_write = distal_synapses[idx].perm <= 0.0f;
    }

    if (!should_write) {
      out_new_syn[syn] = DistalSynapse{0, 0, -1.0f};
      continue;
    }

    // Deterministic selection (thread-safe): pick a "random" previous learning cell.
    const std::size_t pick =
        (static_cast<std::size_t>(origin_col) * 1315423911u +
         static_cast<std::size_t>(origin_cell) * 2654435761u +
         static_cast<std::size_t>(seg) * 97531u +
         static_cast<std::size_t>(syn) * 1013904223u +
         static_cast<std::size_t>(time_step)) %
        prev_learn_cells_list_.size();
    const auto& tgt = prev_learn_cells_list_[pick];
    out_new_syn[syn] = DistalSynapse{tgt.first, tgt.second, cfg_.new_syn_permanence};
  }
}

// Python: updateActiveCells
//
// Calculate which cells should be set as active and which should be put into a learning state.
// The learning state is used by the sequence learning stage to update synapse permanence values.
//
// Inputs:
//   1. time_step:           incrementing integer used to keep track of time.
//   2. active_col_indices:  indices of columns that are currently active (from inhibition).
//   3. predict_cells_time:  3D tensor (num_columns, cells_per_column, 2) storing the last 2
//                           timesteps when each cell was in a predictive state.
//   4. active_segs_time:    3D tensor (num_columns, cells_per_column, max_segments_per_cell)
//                           storing the last timestep each segment was active (i.e. was
//                           predicting the cell would become active — a "sequence segment").
//   5. distal_synapses:     5D tensor (num_columns, cells_per_column, max_segments_per_cell,
//                           max_synapses_per_segment) where each synapse stores
//                           [target_col, target_cell, permanence].
//
// Updates produced:
//   1. active_cells_time:   the last 2 timesteps when each cell was active.
//   2. learn_cells_time:    the last 2 timesteps when each cell was in the learn state.
//   3. current_active_cells_list / current_learn_cells_list: variable-length lists of
//      (col, cell) pairs for this timestep.
//   4. Four update-structure tensors consumed by sequence learning:
//      a. seg_ind_update_active:  per-cell segment index to reinforce (-1 = no update).
//      b. seg_active_syn_active:  0/1 mask of which synapses were active on that segment.
//      c. seg_ind_new_syn_active: per-cell segment index for new synapse proposals (-1 = none).
//      d. seg_new_syn_active:     proposed new synapses (perm < 0 means "no proposal").
//
// Algorithm (follows the CLA whitepaper / Python np_activeCells):
//   For each active column:
//   - If the column was already active last timestep, preserve continuity: keep the same
//     active/learn cell, or keep bursting if it was bursting.
//   - If the column is newly active:
//     - If any cell was predicted (predictive at t-1 with an active sequence segment at t-1),
//       activate and set it as learning. Multiple cells per column can be predicted.
//     - Otherwise the column "bursts" (all cells become active).
//     - If no learning cell was chosen, select the best matching cell/segment and emit
//       update structures for sequence learning to reinforce/grow distal connections.
//
// C++ parallelism note: the Python version processes columns sequentially in index order.
// This C++ version uses taskflow for_each_index over active columns. Each column writes
// only to its own cells, so there are no write conflicts. However, get_segment_active_synapses
// is called with time_step - 1 (not time_step as in Python) because the parallel execution
// means current-timestep activity is only partially computed at call time.
void ActiveCellsCalculator::calculate_active_cells(int time_step,
                                                   const std::vector<int>& active_col_indices,
                                                   const std::vector<int>& predict_cells_time,
                                                   const std::vector<int>& active_segs_time,
                                                   const std::vector<DistalSynapse>& distal_synapses) {
  assert(static_cast<int>(predict_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<int>(active_segs_time.size()) ==
         cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell);

  std::vector<uint8_t> active_cols(static_cast<size_t>(cfg_.num_columns), 0);
  for (int c : active_col_indices) {
    active_cols[static_cast<size_t>(c)] = 1;
  }

  // Save the previous timestep's learning cells list (used by new_random_prev_learn_synapses
  // to create new synapses connected to cells that were in the learn state one timestep ago).
  prev_learn_cells_list_ = current_learn_cells_list_;
  current_active_cells_list_.clear();
  current_learn_cells_list_.clear();

  // Reset update structures. These are populated per-cell below and consumed by sequence learning.
  std::fill(seg_ind_update_active_.begin(), seg_ind_update_active_.end(), -1);
  std::fill(seg_active_syn_active_.begin(), seg_active_syn_active_.end(), 0);
  std::fill(seg_ind_new_syn_active_.begin(), seg_ind_new_syn_active_.end(), -1);
  std::fill(seg_new_syn_active_.begin(), seg_new_syn_active_.end(), DistalSynapse{0, 0, -1.0f});

  tf::Taskflow taskflow;

  taskflow.for_each_index(
      0, static_cast<int>(active_col_indices.size()), 1,
      [&](int k) {
        const int c = active_col_indices[static_cast<size_t>(k)];

        // Columns that are still active from the last step keep the same state of cells:
        // the learning and active cells stay the same.
        if (prev_active_cols_[static_cast<size_t>(c)] == 1) {
          if (!check_col_bursting(c, time_step - 1)) {
            // Column had a single active cell last step — carry it forward.
            const int prev_active = find_active_cell(c, time_step - 1);
            const int chosen = (prev_active >= 0) ? prev_active : 0;
            set_active_cell(c, chosen, time_step);
            set_learn_cell(c, chosen, time_step);
          } else {
            // Column was bursting on the previous timestep.
            // Leave all cells in the column active by updating activeCellsTime.
            for (int i = 0; i < cfg_.cells_per_column; ++i) {
              set_active_cell(c, i, time_step);
            }
            // Leave the previous learn cell in the learn state.
            const int prev_learn = find_learn_cell(c, time_step - 1);
            const int learn = (prev_learn >= 0) ? prev_learn : 0;
            set_learn_cell(c, learn, time_step);
            // The column stays in the bursting state.
            set_burst_col(c, time_step);
          }
        } else {
          // For columns that have changed state from not-active to active, update their
          // cells by setting new active and learn states. Very similar to CLA paper method.
          bool active_cell_chosen = false;
          bool learning_cell_chosen = false;

          // Check if any cell was predicting it would become active on the previous timestep.
          // If a cell was predictive at (t-1) AND has a sequence segment active at (t-1),
          // then that segment was predicting the cell would be active now — activate it
          // and set it into learning.
          //
          // We do NOT break after the first match: multiple cells in the same column may
          // be set to learning if multiple sequence segments correctly predicted.
          for (int i = 0; i < cfg_.cells_per_column; ++i) {
            if (check_cell_predicting(predict_cells_time, c, i, time_step - 1) &&
                check_cell_has_sequence_seg(active_segs_time, c, i, time_step - 1)) {
              active_cell_chosen = true;
              set_active_cell(c, i, time_step);
              learning_cell_chosen = true;
              set_learn_cell(c, i, time_step);
            }
          }

          // No prediction so the column "bursts" — all cells become active.
          if (!active_cell_chosen) {
            set_burst_col(c, time_step);
            for (int i = 0; i < cfg_.cells_per_column; ++i) {
              set_active_cell(c, i, time_step);
            }
          }

          // If no learning cell was chosen, get the best matching cell whose segment was
          // most active (most predicting) for the previous input and set it as the learning cell.
          // Also emit update structures for sequence learning.
          if (!learning_cell_chosen) {
            // match_found indicates whether a cell had a segment with enough matching synapses.
            // If false, a new segment should be created (overwrite the least-used segment).
            const auto [cell, seg, match_found] =
                get_best_matching_cell(distal_synapses, active_segs_time, c, time_step);
            set_learn_cell(c, cell, time_step);

            const int flat = c * cfg_.cells_per_column + cell;

            // Store which segment to reinforce and which of its synapses were active.
            // Sequence learning will increment active synapses and decrement inactive ones.
            seg_ind_update_active_[static_cast<size_t>(flat)] = seg;
            int8_t* out01 =
                &seg_active_syn_active_[static_cast<size_t>(flat) * cfg_.max_synapses_per_segment];
            // The Python version passes timeStep here (current), which works because the
            // Python loop is sequential and earlier columns already have cells set at T.
            // The C++ runs in parallel, so we pass time_step - 1 to get a fully-computed,
            // deterministic snapshot matching the context that get_best_matching_segment
            // matched against (which also uses time_step - 1 internally).
            get_segment_active_synapses(distal_synapses, c, cell, seg, time_step - 1, out01);

            // Create new synapses connected to a random sample of learning cells from
            // one timestep ago. If match_found is false, create new synapses for ALL slots
            // in the segment (overwrite everything). If match_found is true, only overwrite
            // dead synapses (perm <= 0), preserving live ones for reinforcement.
            seg_ind_new_syn_active_[static_cast<size_t>(flat)] = seg;
            DistalSynapse* out_new =
                &seg_new_syn_active_[static_cast<size_t>(flat) * cfg_.max_synapses_per_segment];
            new_random_prev_learn_synapses(distal_synapses, c, cell, seg, time_step,
                                           /*keep_connected_syn=*/match_found, out_new);
            learn_segs_time_[static_cast<size_t>(idx_cell_seg(c, cell, seg))] = time_step;
          }
        }
      })
      .name("active_cells_for_each_active_column");

  executor_.run(taskflow).wait();

  // Build the currentActiveCellsList and currentLearnCellsList by scanning the per-cell
  // time tensors deterministically (in column index order). This avoids any ambiguity from
  // the parallel execution above if multiple cells end up active/learning in a column.
  current_active_cells_list_.reserve(active_col_indices.size() * static_cast<size_t>(cfg_.cells_per_column));
  current_learn_cells_list_.reserve(active_col_indices.size());
  for (int c : active_col_indices) {
    for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
      if (check_cell_active(c, cell, time_step)) {
        current_active_cells_list_.push_back({c, cell});
      }
    }
    for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
      if (check_cell_learn(c, cell, time_step)) {
        current_learn_cells_list_.push_back({c, cell});
      }
    }
  }

  // Save the previous active columns array (Python: self.prevActiveCols = activeColumns).
  prev_active_cols_ = active_cols;
}

} // namespace sequence_pooler

