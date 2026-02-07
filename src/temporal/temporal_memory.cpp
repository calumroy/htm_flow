#include <htm_flow/temporal/temporal_memory.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace temporal {

namespace {

float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

}  // namespace

TemporalMemory::TemporalMemory(int columns_width, int columns_height, TMParams params)
    : columns_width_(columns_width), columns_height_(columns_height), params_(params) {
  std::random_device rd;
  rng_ = std::mt19937(rd());

  if (params_.cells_per_column <= 0 || params_.cells_per_column > 64) {
    // GUI/debug code assumes <= 64 for bitmasking.
    params_.cells_per_column = std::max(1, std::min(64, params_.cells_per_column));
  }

  column_masks_.resize(num_columns());
}

void TemporalMemory::reset() {
  time_step_ = 0;
  cells_.clear();
  prev_active_cells_.clear();
  prev_winner_cells_.clear();
  std::fill(column_masks_.begin(), column_masks_.end(), ColumnStateMasks{});
}

int TemporalMemory::global_cell_index(int column_index, int cell) const {
  return column_index * params_.cells_per_column + cell;
}

std::pair<int, int> TemporalMemory::unflatten_cell(int global_cell) const {
  const int column_index = global_cell / params_.cells_per_column;
  const int cell = global_cell % params_.cells_per_column;
  return {column_index, cell};
}

int TemporalMemory::num_segments(int column_index, int cell) const {
  const int gc = global_cell_index(column_index, cell);
  auto it = cells_.find(gc);
  if (it == cells_.end()) return 0;
  return int(it->second.segments.size());
}

const Segment* TemporalMemory::segment_ptr(int column_index, int cell, int segment) const {
  const int gc = global_cell_index(column_index, cell);
  auto it = cells_.find(gc);
  if (it == cells_.end()) return nullptr;
  if (segment < 0 || segment >= int(it->second.segments.size())) return nullptr;
  return &it->second.segments[segment];
}

void TemporalMemory::compute(const std::vector<int>& active_columns) {
  ++time_step_;
  std::fill(column_masks_.begin(), column_masks_.end(), ColumnStateMasks{});

  // 1) Compute predictive mask based on prev_active_cells_ (from t-1).
  //    For scalability, only evaluate cells that actually have segments.
  std::vector<std::uint64_t> predictive_mask(num_columns(), 0);
  for (const auto& [gc, cell_data] : cells_) {
    const auto [col_idx, cell_in_col] = unflatten_cell(gc);
    if (col_idx < 0 || col_idx >= num_columns()) continue;

    for (const auto& seg : cell_data.segments) {
      int count = 0;
      for (const auto& syn : seg.synapses) {
        if (syn.permanence >= params_.connected_perm && prev_active_cells_.count(syn.src_cell)) {
          ++count;
          if (count >= params_.activation_threshold) break;
        }
      }
      if (count >= params_.activation_threshold) {
        predictive_mask[col_idx] |= (std::uint64_t(1) << cell_in_col);
        break;
      }
    }
  }

  // 2) Choose active/learning cells and winner cells.
  std::unordered_set<int> curr_active_cells;
  std::vector<int> curr_winner_cells;
  curr_winner_cells.reserve(active_columns.size());

  for (int col_idx : active_columns) {
    if (col_idx < 0 || col_idx >= num_columns()) continue;

    const std::uint64_t pred = predictive_mask[col_idx];
    std::uint64_t active_bits = 0;

    if (pred != 0) {
      active_bits = pred;  // correctly predicted: activate predicted cells
    } else {
      // burst
      active_bits = (params_.cells_per_column == 64) ? ~std::uint64_t(0)
                                                     : ((std::uint64_t(1) << params_.cells_per_column) - 1);
    }

    // Winner selection
    int winner_cell = 0;
    if (pred != 0) {
      // pick first predicted cell
      winner_cell = int(__builtin_ctzll(pred));
    } else {
      // pick cell with fewest segments (or 0)
      int best = 0;
      int best_count = 1'000'000;
      for (int c = 0; c < params_.cells_per_column; ++c) {
        const int nseg = num_segments(col_idx, c);
        if (nseg < best_count) {
          best_count = nseg;
          best = c;
        }
      }
      winner_cell = best;
    }

    const int winner_gc = global_cell_index(col_idx, winner_cell);

    // Update per-column masks.
    column_masks_[col_idx].active = active_bits;
    column_masks_[col_idx].predictive = pred;
    column_masks_[col_idx].learning = (std::uint64_t(1) << winner_cell);

    curr_winner_cells.push_back(winner_gc);

    for (int c = 0; c < params_.cells_per_column; ++c) {
      if (active_bits & (std::uint64_t(1) << c)) {
        curr_active_cells.insert(global_cell_index(col_idx, c));
      }
    }
  }

  // 3) Learning: update winner cell segments using prev_active_cells_.
  //    If no matching segment, create a new one using prev winner cells.
  for (int winner_gc : curr_winner_cells) {
    auto& cell_data = cells_[winner_gc];

    // Find best matching segment by overlap.
    int best_seg = -1;
    int best_score = -1;
    for (int si = 0; si < int(cell_data.segments.size()); ++si) {
      int score = 0;
      for (const auto& syn : cell_data.segments[si].synapses) {
        if (syn.permanence >= params_.connected_perm && prev_active_cells_.count(syn.src_cell)) {
          ++score;
        }
      }
      if (score > best_score) {
        best_score = score;
        best_seg = si;
      }
    }

    if (best_score >= params_.activation_threshold && best_seg >= 0) {
      // Reinforce best segment.
      auto& seg = cell_data.segments[best_seg];
      for (auto& syn : seg.synapses) {
        if (prev_active_cells_.count(syn.src_cell)) {
          syn.permanence = clamp01(syn.permanence + params_.perm_inc);
        } else {
          syn.permanence = clamp01(syn.permanence - params_.perm_dec);
        }
      }
    } else {
      // Create a new segment (bounded).
      if (int(cell_data.segments.size()) >= params_.max_segments_per_cell) {
        continue;
      }

      Segment seg;
      seg.synapses.reserve(std::min<int>(params_.max_synapses_per_segment, int(prev_winner_cells_.size())));

      // Sample from prev winner cells.
      std::vector<int> candidates = prev_winner_cells_;
      std::shuffle(candidates.begin(), candidates.end(), rng_);
      const int k = std::min<int>(params_.max_synapses_per_segment, int(candidates.size()));
      for (int i = 0; i < k; ++i) {
        seg.synapses.push_back(Synapse{candidates[i], params_.initial_perm});
      }

      cell_data.segments.push_back(std::move(seg));
    }
  }

  // 4) Advance time state.
  prev_active_cells_ = std::move(curr_active_cells);
  prev_winner_cells_ = std::move(curr_winner_cells);
}

}  // namespace temporal
