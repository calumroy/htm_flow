#include <htm_flow/sequence_pooler/active_cells.hpp>

#include <cassert>
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

  col_last_active_time_.assign(cfg_.num_columns, -1);
  burst_cols_time_.assign(cfg_.num_columns * 2, -1);
  active_cells_time_.assign(cfg_.num_columns * cfg_.cells_per_column * 2, -1);
  learn_cells_time_.assign(cfg_.num_columns * cfg_.cells_per_column * 2, -1);
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

int ActiveCellsCalculator::choose_learning_cell_stub(int /*col*/, int /*time_step*/) const {
  // v1 stub:
  // - With predictive state + distal synapses not yet plugged in, emulate the "no match"
  //   path by deterministically picking cell 0 as the learning cell.
  return 0;
}

void ActiveCellsCalculator::calculate_active_cells(int time_step,
                                                   const std::vector<int>& active_col_indices,
                                                   const std::vector<int>& predict_cells_time,
                                                   const std::vector<int>& active_segs_time,
                                                   const std::vector<DistalSynapse>& /*distal_synapses*/) {
  // Implementation overview:
  // Step 1. Clear the current output lists (we are computing a fresh timestep).
  // Step 2. In parallel over active columns:
  //         - If the column was active at (time_step-1), keep it stable (or keep bursting).
  //         - If the column is newly active:
  //           - If a cell was predicting at (time_step-1) AND has a segment that was active at (time_step-1),
  //             activate that cell (predicted activation path).
  //           - Otherwise, burst (all cells active).
  //         - Update the internal time-history tensors (active/learn/burst) for this timestep.
  // Step 3. Build the public output lists from the per-column decisions in a single thread
  //         (avoids concurrent push_back and keeps results deterministic).

  // Shape checks (predictive state is produced by PredictCellsCalculator).
  assert(static_cast<int>(predict_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<int>(active_segs_time.size()) ==
         cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell);

  // Step 1. Reset current timestep outputs.
  current_active_cells_list_.clear();
  current_learn_cells_list_.clear();

  struct ColDecision {
    int col = -1;
    bool burst = false;
    int active_cell = -1; // valid if !burst
    int learn_cell = -1;  // always valid
  };

  std::vector<ColDecision> decisions(active_col_indices.size());

  tf::Taskflow taskflow;

  taskflow.for_each_index(
      0, static_cast<int>(active_col_indices.size()), 1,
      [&](int k) {
        const int c = active_col_indices[static_cast<size_t>(k)];
        decisions[static_cast<size_t>(k)].col = c;

        // Was this column active last timestep?
        const bool col_was_active_prev = (col_last_active_time_[c] == time_step - 1);

        if (col_was_active_prev) {
          // Column remained active.
          if (!check_col_bursting(c, time_step - 1)) {
            const int prev_active = find_active_cell(c, time_step - 1);
            // Should exist; otherwise fall back to a deterministic choice.
            const int chosen = (prev_active >= 0) ? prev_active : 0;

            set_active_cell(c, chosen, time_step);
            set_learn_cell(c, chosen, time_step);

            decisions[static_cast<size_t>(k)].burst = false;
            decisions[static_cast<size_t>(k)].active_cell = chosen;
            decisions[static_cast<size_t>(k)].learn_cell = chosen;
          } else {
            // Column was bursting previously; keep all cells active.
            for (int i = 0; i < cfg_.cells_per_column; ++i) {
              set_active_cell(c, i, time_step);
            }
            const int prev_learn = find_learn_cell(c, time_step - 1);
            const int learn = (prev_learn >= 0) ? prev_learn : 0;
            set_learn_cell(c, learn, time_step);
            set_burst_col(c, time_step);

            decisions[static_cast<size_t>(k)].burst = true;
            decisions[static_cast<size_t>(k)].active_cell = -1;
            decisions[static_cast<size_t>(k)].learn_cell = learn;
          }
        } else {
          // Newly active column.
          bool active_cell_chosen = false;
          bool learning_cell_chosen = false;

          // Predicted activation path (matches the python logic at a high level):
          // If the cell was predictive at (t-1) AND had a sequence segment active at (t-1),
          // then activate it now and set it as the learning cell.
          for (int i = 0; i < cfg_.cells_per_column; ++i) {
            if (check_cell_predicting(predict_cells_time, c, i, time_step - 1) &&
                check_cell_has_sequence_seg(active_segs_time, c, i, time_step - 1)) {
              active_cell_chosen = true;
              set_active_cell(c, i, time_step);
              learning_cell_chosen = true;
              set_learn_cell(c, i, time_step);

              decisions[static_cast<size_t>(k)].burst = false;
              decisions[static_cast<size_t>(k)].active_cell = i;
              decisions[static_cast<size_t>(k)].learn_cell = i;
              break; // choose the first matching predicted cell for now (deterministic)
            }
          }

          if (!active_cell_chosen) {
            // No prediction -> burst.
            set_burst_col(c, time_step);
            for (int i = 0; i < cfg_.cells_per_column; ++i) {
              set_active_cell(c, i, time_step);
            }
            decisions[static_cast<size_t>(k)].burst = true;
            decisions[static_cast<size_t>(k)].active_cell = -1;
          }

          if (!learning_cell_chosen) {
            // v1 stub: until best-matching cell / scoring is implemented, pick deterministically.
            const int learn = choose_learning_cell_stub(c, time_step);
            set_learn_cell(c, learn, time_step);
            decisions[static_cast<size_t>(k)].learn_cell = learn;
          }
        }

        // Mark column active at this timestep.
        col_last_active_time_[c] = time_step;
      })
      .name("active_cells_for_each_active_column");

  // Step 2. Execute the per-active-column work.
  executor_.run(taskflow).wait();

  // Step 3. Build output lists deterministically (single-thread).
  //
  // Why do we do this in a single thread?
  // - The per-column computations above run in parallel. If each task appended directly
  //   to `current_active_cells_list_` / `current_learn_cells_list_`, we'd need locks (slow)
  //   or per-thread buffers (more complex).
  // - Instead, each parallel task writes one `ColDecision` into `decisions[k]`. Here we
  //   *materialize* those decisions into the public output lists in a deterministic order.
  //
  // What is `decisions`?
  // - One entry per active column for this timestep.
  // - Each entry records: which column it was, whether it burst, and which active/learning
  //   cell indices were chosen.
  //
  // Performance note:
  // - This pass is O(#activeColumns * cellsPerColumn) in the worst case (bursting adds all
  //   cells in a column). In typical HTM settings, #activeColumns is sparse, so `decisions`
  //   is small (usually thousands, not hundreds of thousands), and this loop is negligible
  //   compared to the heavy stages (overlap/inhibition/spatial learning). If #activeColumns
  //   ever becomes large, we can parallelize this step too by writing into a pre-sized
  //   output array instead of using push_back.
  current_active_cells_list_.reserve(active_col_indices.size() * static_cast<size_t>(cfg_.cells_per_column));
  current_learn_cells_list_.reserve(active_col_indices.size());

  for (const auto& d : decisions) {
    if (d.col < 0) {
      continue;
    }
    if (d.burst) {
      for (int i = 0; i < cfg_.cells_per_column; ++i) {
        current_active_cells_list_.push_back({d.col, i});
      }
    } else {
      current_active_cells_list_.push_back({d.col, d.active_cell});
    }
    current_learn_cells_list_.push_back({d.col, d.learn_cell});
  }
}

} // namespace sequence_pooler

