#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <taskflow/taskflow.hpp>

namespace sequence_pooler {

// Active-cells stage of the sequence pooler.
//
// What this component does:
// - Given the set of *active columns* at `time_step`, decide which *cells* inside those
//   columns become active and which become the learning cell.
//
// What this component returns:
// - A variable-length list of active cells: [(col, cell), ...]
// - A variable-length list of learning cells: [(col, cell), ...] (one per active column)
// - Time-history tensors storing the last two timesteps a cell was active / learning.
//
// Notes (current wiring stage):
// - The full temporal memory inputs (predictive state, active segments, distal synapses)
//   are not connected yet in this C++ pipeline.
// - For now, newly-active columns "burst" (all cells active) and the learning cell is
//   chosen deterministically. This keeps the plumbing correct and lets later calculators
//   plug in without changing the public interface.
class ActiveCellsCalculator {
public:
  struct Config {
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;
    int min_num_syn_threshold = 0;
    int min_score_threshold = 0;
    float new_syn_permanence = 0.0f;
    float connect_permanence = 0.0f;
  };

  explicit ActiveCellsCalculator(const Config& cfg);

  // Update active/learn states for the given timestep.
  //
  // Step 1. Accept the active column indices for this timestep.
  // Step 2. Update internal state (active/learn/burst history).
  // Step 3. Produce output lists for downstream stages.
  void calculate_active_cells(int time_step, const std::vector<int>& active_col_indices);

  // --- Outputs / state accessors ---
  const std::vector<std::pair<int, int>>& get_current_active_cells_list() const;
  const std::vector<std::pair<int, int>>& get_current_learn_cells_list() const;

  // Flattened tensors of shape:
  // - active_cells_time: (num_columns, cells_per_column, 2)
  // - learn_cells_time : (num_columns, cells_per_column, 2)
  const std::vector<int>& get_active_cells_time() const;
  const std::vector<int>& get_learn_cells_time() const;

  // Shape helpers
  int num_columns() const { return cfg_.num_columns; }
  int cells_per_column() const { return cfg_.cells_per_column; }

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }
  inline int idx_col_time(int col, int slot) const { return col * 2 + slot; }

  bool check_cell_active(int col, int cell, int time_step) const;
  bool check_cell_learn(int col, int cell, int time_step) const;
  bool check_col_bursting(int col, int time_step) const;

  void set_active_cell(int col, int cell, int time_step);
  void set_learn_cell(int col, int cell, int time_step);
  void set_burst_col(int col, int time_step);

  int find_active_cell(int col, int time_step) const; // returns first active cell, or -1
  int find_learn_cell(int col, int time_step) const;  // returns first learn cell, or -1

  // v1 stub: choose a deterministic learning cell for a newly-active column.
  int choose_learning_cell_stub(int col, int time_step) const;

  Config cfg_;

  // Track last timestep a column was active (replaces python prevActiveCols bit-array).
  std::vector<int> col_last_active_time_; // shape: (num_columns)

  // Track last 2 timesteps a column was bursting (all cells active).
  std::vector<int> burst_cols_time_; // shape: (num_columns, 2)

  // Track last 2 timesteps each cell was active / learning.
  std::vector<int> active_cells_time_; // shape: (num_columns, cells_per_column, 2)
  std::vector<int> learn_cells_time_;  // shape: (num_columns, cells_per_column, 2)

  // Current timestep outputs as lists of (col, cell) pairs.
  std::vector<std::pair<int, int>> current_active_cells_list_;
  std::vector<std::pair<int, int>> current_learn_cells_list_;

  // Executor reused across calls (like other calculators).
  tf::Executor executor_;
};

} // namespace sequence_pooler

