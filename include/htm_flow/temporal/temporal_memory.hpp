#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace temporal {

struct ColumnStateMasks {
  std::uint64_t active{0};
  std::uint64_t predictive{0};
  std::uint64_t learning{0};
};

struct Synapse {
  int src_cell{0};
  float permanence{0.0f};
};

struct Segment {
  std::vector<Synapse> synapses;
};

struct CellData {
  std::vector<Segment> segments;
};

struct TMParams {
  int cells_per_column{32};
  float connected_perm{0.5f};
  int activation_threshold{10};
  float perm_inc{0.05f};
  float perm_dec{0.05f};
  float initial_perm{0.21f};
  int max_segments_per_cell{16};
  int max_synapses_per_segment{32};
};

class TemporalMemory {
public:
  TemporalMemory(int columns_width, int columns_height, TMParams params);

  void reset();

  // active_columns are flattened column indices (x + y*columns_width).
  void compute(const std::vector<int>& active_columns);

  int time_step() const { return time_step_; }

  int columns_width() const { return columns_width_; }
  int columns_height() const { return columns_height_; }
  int num_columns() const { return columns_width_ * columns_height_; }
  int cells_per_column() const { return params_.cells_per_column; }

  const std::vector<ColumnStateMasks>& column_masks() const { return column_masks_; }

  // Segment inspection for GUI/debugging.
  int num_segments(int column_index, int cell) const;
  const Segment* segment_ptr(int column_index, int cell, int segment) const;

  // Convert a global cell index into (col_index, cell_in_column).
  std::pair<int, int> unflatten_cell(int global_cell) const;

private:
  int global_cell_index(int column_index, int cell) const;

  int columns_width_{0};
  int columns_height_{0};
  TMParams params_{};

  int time_step_{0};

  // Sparse per-cell segment storage.
  std::unordered_map<int, CellData> cells_;

  // State sets for matching.
  std::unordered_set<int> prev_active_cells_;
  std::vector<int> prev_winner_cells_;

  // Per-column masks for current timestep.
  std::vector<ColumnStateMasks> column_masks_;

  std::mt19937 rng_;
};

}  // namespace temporal
