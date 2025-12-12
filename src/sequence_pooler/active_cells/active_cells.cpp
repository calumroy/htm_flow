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

  cells_score_.assign(cfg_.num_columns * cfg_.cells_per_column, 0);
  col_highest_scored_cell_.assign(cfg_.num_columns, -1);

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

bool ActiveCellsCalculator::check_cell_active(int col, int cell, int time_step) const {
  const int a0 = active_cells_time_[idx_cell_time(col, cell, 0)];
  const int a1 = active_cells_time_[idx_cell_time(col, cell, 1)];
  return (a0 == time_step) || (a1 == time_step);
}

bool ActiveCellsCalculator::check_cell_learn(int col, int cell, int time_step) const {
  const int a0 = learn_cells_time_[idx_cell_time(col, cell, 0)];
  const int a1 = learn_cells_time_[idx_cell_time(col, cell, 1)];
  return (a0 == time_step) || (a1 == time_step);
}

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

bool ActiveCellsCalculator::check_col_prev_active(int col) const {
  return prev_active_cols_[static_cast<size_t>(col)] == 1;
}

void ActiveCellsCalculator::set_burst_col(int col, int time_step) {
  const int i0 = idx_col_time(col, 0);
  const int i1 = idx_col_time(col, 1);
  if (burst_cols_time_[i0] <= burst_cols_time_[i1]) {
    burst_cols_time_[i0] = time_step;
  } else {
    burst_cols_time_[i1] = time_step;
  }
}

void ActiveCellsCalculator::set_active_cell(int col, int cell, int time_step) {
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  if (active_cells_time_[i0] <= active_cells_time_[i1]) {
    active_cells_time_[i0] = time_step;
  } else {
    active_cells_time_[i1] = time_step;
  }
}

void ActiveCellsCalculator::set_learn_cell(int col, int cell, int time_step) {
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  if (learn_cells_time_[i0] <= learn_cells_time_[i1]) {
    learn_cells_time_[i0] = time_step;
  } else {
    learn_cells_time_[i1] = time_step;
  }
}

int ActiveCellsCalculator::find_active_cell(int col, int time_step) const {
  for (int i = 0; i < cfg_.cells_per_column; ++i) {
    if (check_cell_active(col, i, time_step)) {
      return i;
    }
  }
  return -1;
}

int ActiveCellsCalculator::find_learn_cell(int col, int time_step) const {
  for (int i = 0; i < cfg_.cells_per_column; ++i) {
    if (check_cell_learn(col, i, time_step)) {
      return i;
    }
  }
  return -1;
}

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
    if (s.perm > cfg_.connect_permanence) {
      if (on_cell) {
        // Match python behavior: count synapses ending on cells active at (time_step-1).
        if (check_cell_active(s.target_col, s.target_cell, time_step - 1)) {
          ++count;
        }
      } else {
        // Count synapses ending on columns that were active at (time_step-1).
        if (check_col_prev_active(s.target_col)) {
          ++count;
        }
      }
    }
  }
  return count;
}

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
  // Must exceed the threshold to count as "matching".
  if (most_active > cfg_.min_num_syn_threshold) {
    return best_seg;
  }
  return -1;
}

int ActiveCellsCalculator::segment_highest_score(const std::vector<DistalSynapse>& distal_synapses,
                                                 int origin_col,
                                                 int origin_cell,
                                                 int seg) const {
  int highest = 0;
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
    if (check_col_prev_active(s.target_col)) {
      const int flat = s.target_col * cfg_.cells_per_column + s.target_cell;
      const int score = cells_score_[static_cast<size_t>(flat)];
      if (score > highest) {
        highest = score;
      }
    }
  }
  return highest;
}

void ActiveCellsCalculator::update_active_cell_scores(const std::vector<uint8_t>& active_cols,
                                                      const std::vector<DistalSynapse>& distal_synapses,
                                                      int time_step) {
  for (int c = 0; c < cfg_.num_columns; ++c) {
    // Only update scores for columns that changed state from not-active to active.
    if (prev_active_cols_[static_cast<size_t>(c)] == 0 && active_cols[static_cast<size_t>(c)] == 1) {
      int highest_score = 0;
      col_highest_scored_cell_[static_cast<size_t>(c)] = -1;
      for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
        const int best_seg = get_best_matching_segment(distal_synapses, c, cell, time_step, /*on_cell=*/false);
        const int flat = c * cfg_.cells_per_column + cell;
        if (best_seg != -1) {
          const int score = 1 + segment_highest_score(distal_synapses, c, cell, best_seg);
          cells_score_[static_cast<size_t>(flat)] = score;
          if (score > highest_score) {
            highest_score = score;
            col_highest_scored_cell_[static_cast<size_t>(c)] = cell;
          }
        } else {
          cells_score_[static_cast<size_t>(flat)] = 0;
        }
      }
    }
  }
}

int ActiveCellsCalculator::find_num_segs(const std::vector<DistalSynapse>& distal_synapses,
                                        int origin_col,
                                        int origin_cell) const {
  int num = 0;
  for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
    for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
      const std::size_t idx =
          idx_distal_synapse(static_cast<std::size_t>(origin_col),
                             static_cast<std::size_t>(origin_cell),
                             static_cast<std::size_t>(seg),
                             static_cast<std::size_t>(syn),
                             static_cast<std::size_t>(cfg_.cells_per_column),
                             static_cast<std::size_t>(cfg_.max_segments_per_cell),
                             static_cast<std::size_t>(cfg_.max_synapses_per_segment));
      if (distal_synapses[idx].perm > 0.0f) {
        ++num;
        break;
      }
    }
  }
  return num;
}

int ActiveCellsCalculator::find_least_used_seg(const std::vector<int>& active_segs_time,
                                              int origin_col,
                                              int origin_cell) const {
  const int base = (origin_col * cfg_.cells_per_column + origin_cell) * cfg_.max_segments_per_cell;
  int least_seg = 0;
  int oldest = active_segs_time[static_cast<size_t>(base)];
  for (int seg = 1; seg < cfg_.max_segments_per_cell; ++seg) {
    const int t = active_segs_time[static_cast<size_t>(base + seg)];
    if (t < oldest) {
      oldest = t;
      least_seg = seg;
    }
  }
  return least_seg;
}

std::tuple<int, int, bool> ActiveCellsCalculator::get_best_matching_cell(const std::vector<DistalSynapse>& distal_synapses,
                                                                         const std::vector<int>& active_segs_time,
                                                                         int origin_col,
                                                                         int time_step) const {
  bool best_cell_found = false;
  int best_cell = 0;
  int best_seg = 0;
  int num_act_best = 0;

  int cell_least_used_seg = 0;
  int fewest_segs = 0;
  int seg_least_used = 0;
  int least_used_time = time_step;

  for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
    const int num_segs = find_num_segs(distal_synapses, origin_col, cell);
    if (cell == 0 || num_segs < fewest_segs) {
      cell_least_used_seg = cell;
      fewest_segs = num_segs;
      seg_least_used = find_least_used_seg(active_segs_time, origin_col, cell);
      const int base = (origin_col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell;
      least_used_time = active_segs_time[static_cast<size_t>(base + seg_least_used)];
    } else if (num_segs == fewest_segs) {
      const int candidate_seg = find_least_used_seg(active_segs_time, origin_col, cell);
      const int base = (origin_col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell;
      const int candidate_time = active_segs_time[static_cast<size_t>(base + candidate_seg)];
      if (candidate_time < least_used_time) {
        cell_least_used_seg = cell;
        seg_least_used = candidate_seg;
        least_used_time = candidate_time;
      }
    }

    const int h = get_best_matching_segment(distal_synapses, origin_col, cell, time_step, /*on_cell=*/true);
    if (h != -1) {
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
    if (s.perm >= cfg_.connect_permanence && check_cell_active(s.target_col, s.target_cell, time_step)) {
      out01[syn] = 1;
    } else {
      out01[syn] = 0;
    }
  }
}

void ActiveCellsCalculator::new_random_prev_learn_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                                           int origin_col,
                                                           int origin_cell,
                                                           int seg,
                                                           int time_step,
                                                           bool keep_connected_syn,
                                                           DistalSynapse* out_new_syn) const {
  // Fill `out_new_syn` with proposed new synapses (or perm < 0 to indicate "no proposal").
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
      should_write = distal_synapses[idx].perm < cfg_.connect_permanence;
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

void ActiveCellsCalculator::calculate_active_cells(int time_step,
                                                   const std::vector<int>& active_col_indices,
                                                   const std::vector<int>& predict_cells_time,
                                                   const std::vector<int>& active_segs_time,
                                                   const std::vector<DistalSynapse>& distal_synapses) {
  // Implementation overview (ported from the Python implementation):
  // Step 1. Compute per-cell scores for newly-active columns.
  // Step 2. Update active + learning states for active columns:
  //         - predicted activation path
  //         - alternative sequence path (highest score above threshold)
  //         - bursting fallback
  //         - best-matching-cell selection for learning cell + update structures
  // Step 3. Build output lists (active cells + learning cells) for this timestep.

  // Shape checks (predictive state is produced by PredictCellsCalculator).
  assert(static_cast<int>(predict_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<int>(active_segs_time.size()) ==
         cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell);

  // Step 0. Build a dense active-columns array (0/1) from indices.
  std::vector<uint8_t> active_cols(static_cast<size_t>(cfg_.num_columns), 0);
  for (int c : active_col_indices) {
    active_cols[static_cast<size_t>(c)] = 1;
  }

  // Step 1. Reset current timestep outputs.
  // Keep a copy of the previous timestep's learning cells list (used for new-synapse proposals).
  prev_learn_cells_list_ = current_learn_cells_list_;
  current_active_cells_list_.clear();
  current_learn_cells_list_.clear();

  // Reset active-cells update structures for this timestep.
  std::fill(seg_ind_update_active_.begin(), seg_ind_update_active_.end(), -1);
  std::fill(seg_active_syn_active_.begin(), seg_active_syn_active_.end(), 0);
  std::fill(seg_ind_new_syn_active_.begin(), seg_ind_new_syn_active_.end(), -1);
  std::fill(seg_new_syn_active_.begin(), seg_new_syn_active_.end(), DistalSynapse{0, 0, -1.0f});

  // Update the scores for the alternative-sequence selection.
  update_active_cell_scores(active_cols, distal_synapses, time_step);

  tf::Taskflow taskflow;

  taskflow.for_each_index(
      0, static_cast<int>(active_col_indices.size()), 1,
      [&](int k) {
        const int c = active_col_indices[static_cast<size_t>(k)];

        // Was this column active last timestep?
        if (prev_active_cols_[static_cast<size_t>(c)] == 1) {
          // Column remained active.
          if (!check_col_bursting(c, time_step - 1)) {
            const int prev_active = find_active_cell(c, time_step - 1);
            const int chosen = (prev_active >= 0) ? prev_active : 0;

            set_active_cell(c, chosen, time_step);
            set_learn_cell(c, chosen, time_step);
          } else {
            // Column was bursting previously; keep all cells active.
            for (int i = 0; i < cfg_.cells_per_column; ++i) {
              set_active_cell(c, i, time_step);
            }
            const int prev_learn = find_learn_cell(c, time_step - 1);
            const int learn = (prev_learn >= 0) ? prev_learn : 0;
            set_learn_cell(c, learn, time_step);
            set_burst_col(c, time_step);
          }
        } else {
          // Newly active column.
          bool active_cell_chosen = false;
          bool learning_cell_chosen = false;

          // Predicted activation path:
          // If a cell was predictive at (t-1) AND has a sequence segment active at (t-1),
          // activate it now and set it into learning.
          for (int i = 0; i < cfg_.cells_per_column; ++i) {
            if (check_cell_predicting(predict_cells_time, c, i, time_step - 1) &&
                check_cell_has_sequence_seg(active_segs_time, c, i, time_step - 1)) {
              active_cell_chosen = true;
              set_active_cell(c, i, time_step);
              learning_cell_chosen = true;
              set_learn_cell(c, i, time_step);
            }
          }

          // Alternative sequence path (score-based): if no predicted cell, try highest-scored cell.
          const int high_cell = col_highest_scored_cell_[static_cast<size_t>(c)];
          if (high_cell != -1 && !active_cell_chosen) {
            const int flat = c * cfg_.cells_per_column + high_cell;
            if (cells_score_[static_cast<size_t>(flat)] >= cfg_.min_score_threshold) {
              active_cell_chosen = true;
              set_active_cell(c, high_cell, time_step);
              learning_cell_chosen = true;
              set_learn_cell(c, high_cell, time_step);

              // Create a new segment by overwriting the least-used segment's synapses.
              const int seg = find_least_used_seg(active_segs_time, c, high_cell);
              seg_ind_new_syn_active_[static_cast<size_t>(flat)] = seg;
              DistalSynapse* out_new =
                  &seg_new_syn_active_[static_cast<size_t>(flat) * cfg_.max_synapses_per_segment];
              new_random_prev_learn_synapses(distal_synapses, c, high_cell, seg, time_step,
                                             /*keep_connected_syn=*/false, out_new);
            }
          }

          // Bursting fallback if still no active cell was chosen.
          if (!active_cell_chosen) {
            set_burst_col(c, time_step);
            for (int i = 0; i < cfg_.cells_per_column; ++i) {
              set_active_cell(c, i, time_step);
            }
          }

          // If no learning cell was chosen yet, choose the best matching cell and emit update structures.
          if (!learning_cell_chosen) {
            const auto [cell, seg, match_found] =
                get_best_matching_cell(distal_synapses, active_segs_time, c, time_step);
            set_learn_cell(c, cell, time_step);

            const int flat = c * cfg_.cells_per_column + cell;
            seg_ind_update_active_[static_cast<size_t>(flat)] = seg;
            int8_t* out01 =
                &seg_active_syn_active_[static_cast<size_t>(flat) * cfg_.max_synapses_per_segment];
            get_segment_active_synapses(distal_synapses, c, cell, seg, time_step, out01);

            seg_ind_new_syn_active_[static_cast<size_t>(flat)] = seg;
            DistalSynapse* out_new =
                &seg_new_syn_active_[static_cast<size_t>(flat) * cfg_.max_synapses_per_segment];
            // If a matching segment was found, only overwrite weak synapses; otherwise overwrite all.
            new_random_prev_learn_synapses(distal_synapses, c, cell, seg, time_step,
                                           /*keep_connected_syn=*/match_found, out_new);
          }
        }
      })
      .name("active_cells_for_each_active_column");

  // Step 2. Execute the per-active-column work.
  executor_.run(taskflow).wait();

  // Step 3. Build output lists deterministically by scanning the per-cell time tensors.
  // This avoids any ambiguity if multiple cells end up active/learning in a column.
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

  // Save the previous active columns bitfield for the next timestep.
  prev_active_cols_ = active_cols;
}

} // namespace sequence_pooler

