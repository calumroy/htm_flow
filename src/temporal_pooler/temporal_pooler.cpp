#include <htm_flow/temporal_pooler/temporal_pooler.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include <taskflow/algorithm/for_each.hpp>

namespace temporal_pooler {

using sequence_pooler::DistalSynapse;
using sequence_pooler::idx_distal_synapse;

TemporalPoolerCalculator::TemporalPoolerCalculator(const Config& cfg) : cfg_(cfg) {
  assert(cfg_.num_columns > 0);
  assert(cfg_.cells_per_column > 0);
  assert(cfg_.max_segments_per_cell > 0);
  assert(cfg_.max_synapses_per_segment > 0);
  assert(cfg_.num_pot_synapses > 0);
  assert(cfg_.delay_length > 0);

  const int num_cells = cfg_.num_columns * cfg_.cells_per_column;
  cells_tracking_num_.assign(num_cells, 0);
  cells_avg_persist_.assign(num_cells, -1.0f);
  cells_persistence_.assign(num_cells, 0);
  last_active_predict_support_by_column_.assign(cfg_.num_columns, 0);
  prev_active_predict_support_by_column_.assign(cfg_.num_columns, 0);

  new_learn_cells_time_.assign(num_cells * 2, -1);
}

int TemporalPoolerCalculator::update_proximal(
    int support_time,
    const std::vector<int>& col_pot_inputs01,
    std::vector<float>& col_syn_perm,
    const std::vector<uint8_t>* col_active01,
    const std::vector<int>* predict_cells_time,
    const std::vector<int>* active_segs_time) const {
  int reinforced_inputs = 0;
  if (support_time != last_active_predict_support_time_ ||
      cfg_.spatial_permanence_inc <= 0.0f) {
    return reinforced_inputs;
  }

  assert(static_cast<int>(col_pot_inputs01.size()) == cfg_.num_columns * cfg_.num_pot_synapses);
  assert(static_cast<int>(col_syn_perm.size()) == cfg_.num_columns * cfg_.num_pot_synapses);

  // The extra state is optional so older/unit callers can still exercise the
  // active-predict path. Without it, we cannot safely reinforce predicted
  // columns that failed to become active.
  const bool can_reinforce_predictive_non_active =
      col_active01 != nullptr && predict_cells_time != nullptr && active_segs_time != nullptr;
  if (can_reinforce_predictive_non_active) {
    assert(static_cast<int>(col_active01->size()) == cfg_.num_columns);
    assert(static_cast<int>(predict_cells_time->size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
    assert(static_cast<int>(active_segs_time->size()) ==
           cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell);
  }

  // Convert distal confidence into real proximal permanence so columns that TP
  // predicted can eventually win spatial overlap instead of only predicting.
  const int n = std::min(cfg_.num_columns,
                         static_cast<int>(last_active_predict_support_by_column_.size()));
  for (int col = 0; col < n; ++col) {
    const int support = last_active_predict_support_by_column_[static_cast<std::size_t>(col)];
    const bool active_now = can_reinforce_predictive_non_active &&
                            ((*col_active01)[static_cast<std::size_t>(col)] == 1);

    // Segment-backed prediction that lost inhibition: strengthen current inputs
    // so this column has a better chance to win this context next time.
    const bool predictive_non_active =
        can_reinforce_predictive_non_active && !active_now &&
        column_has_temporal_support(*predict_cells_time, *active_segs_time, col, support_time);

    // After a correct activation, also learn the next input only if the column
    // is still predicted by real segment evidence.
    const bool post_active_predict =
        predictive_non_active &&
        prev_active_predict_support_time_ == support_time - 1 &&
        col < static_cast<int>(prev_active_predict_support_by_column_.size()) &&
        prev_active_predict_support_by_column_[static_cast<std::size_t>(col)] > 0;
    if (support <= 0 && !predictive_non_active && !post_active_predict) {
      continue;
    }
    const int effective_support = support + (predictive_non_active ? 1 : 0) +
                                  (post_active_predict ? 1 : 0);
    const float inc = cfg_.spatial_permanence_inc * static_cast<float>(effective_support);
    const std::size_t base =
        static_cast<std::size_t>(col) * static_cast<std::size_t>(cfg_.num_pot_synapses);

    // Only strengthen inputs that are active now; no temporary overlap bonus or
    // global correction is introduced.
    for (int syn = 0; syn < cfg_.num_pot_synapses; ++syn) {
      const std::size_t idx = base + static_cast<std::size_t>(syn);
      if (col_pot_inputs01[idx] != 1) {
        continue;
      }
      ++reinforced_inputs;
      const float p = col_syn_perm[idx] + inc;
      const float after = (p > 1.0f) ? 1.0f : p;
      col_syn_perm[idx] = after;
    }
  }
  return reinforced_inputs;
}

std::uint64_t TemporalPoolerCalculator::splitmix64(std::uint64_t x) {
  // http://xorshift.di.unimi.it/splitmix64.c
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::size_t TemporalPoolerCalculator::deterministic_pick(std::uint64_t seed, std::size_t mod) {
  // We need deterministic "random-like" sampling that is:
  // - reproducible across runs
  // - thread-safe (no shared RNG state inside taskflow loops)
  // - cheap
  //
  // This replaces python's `random.sample(list, 1)` usage for synapse endpoints.
  if (mod == 0) {
    return 0;
  }
  return static_cast<std::size_t>(splitmix64(seed) % static_cast<std::uint64_t>(mod));
}

bool TemporalPoolerCalculator::check_cell_time(const std::vector<int>& cells_time,
                                               int col,
                                               int cell,
                                               int time_step) const {
  // Check if the given cell was in a particular state at `time_step`,
  // where the state-history tensor stores the last two timesteps for that state.
  const int t0 = cells_time[static_cast<std::size_t>(idx_cell_time(col, cell, 0))];
  const int t1 = cells_time[static_cast<std::size_t>(idx_cell_time(col, cell, 1))];
  return (t0 == time_step) || (t1 == time_step);
}

bool TemporalPoolerCalculator::column_has_temporal_support(const std::vector<int>& predict_cells_time,
                                                           const std::vector<int>& active_segs_time,
                                                           int col,
                                                           int time_step) const {
  for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
    if (!check_cell_predict(predict_cells_time, col, cell, time_step)) {
      continue;
    }

    const int base = idx_cell_seg(col, cell, 0);
    for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
      if (active_segs_time[static_cast<std::size_t>(base + seg)] == time_step) {
        return true;
      }
    }
  }
  return false;
}

bool TemporalPoolerCalculator::check_cell_predict(const std::vector<int>& predict_cells_time,
                                                  int col,
                                                  int cell,
                                                  int time_step) const {
  // Check if the given cell was predicting at the timestep given.
  // Mirrors python `checkCellPredict`.
  const int p0 = predict_cells_time[static_cast<std::size_t>(idx_cell_time(col, cell, 0))];
  const int p1 = predict_cells_time[static_cast<std::size_t>(idx_cell_time(col, cell, 1))];
  return (p0 == time_step) || (p1 == time_step);
}

void TemporalPoolerCalculator::set_predict_cell(std::vector<int>& predict_cells_time,
                                               int col,
                                               int cell,
                                               int time_step) const {
  // Set the given cell into a predictive state for the given timestep.
  // The tensor stores the last two predictive timesteps: we overwrite the older entry.
  // Mirrors python `setPredictCell`.
  const int i0 = idx_cell_time(col, cell, 0);
  const int i1 = idx_cell_time(col, cell, 1);
  if (predict_cells_time[static_cast<std::size_t>(i0)] <= predict_cells_time[static_cast<std::size_t>(i1)]) {
    predict_cells_time[static_cast<std::size_t>(i0)] = time_step;
  } else {
    predict_cells_time[static_cast<std::size_t>(i1)] = time_step;
  }
}

int TemporalPoolerCalculator::find_recent_active_segment(const std::vector<int>& active_segs_time,
                                                         int col,
                                                         int cell,
                                                         int time_step) const {
  const int base = idx_cell_seg(col, cell, 0);
  for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
    if (active_segs_time[static_cast<std::size_t>(base + seg)] == time_step) {
      return seg;
    }
  }
  return -1;
}

void TemporalPoolerCalculator::set_active_segment(std::vector<int>& active_segs_time,
                                                  int col,
                                                  int cell,
                                                  int seg,
                                                  int time_step) const {
  if (seg < 0 || seg >= cfg_.max_segments_per_cell) {
    return;
  }
  active_segs_time[static_cast<std::size_t>(idx_cell_seg(col, cell, seg))] = time_step;
}

bool TemporalPoolerCalculator::check_cell_active_predict(const std::vector<int>& active_cells_time,
                                                        const std::vector<int>& predict_cells_time,
                                                        int col,
                                                        int cell,
                                                        int time_step) const {
  // Check if a cell is active now AND was predicting one timestep before.
  // Mirrors python `checkCellActivePredict`.
  const bool cell_active = check_cell_time(active_cells_time, col, cell, time_step);
  const bool was_predict = check_cell_predict(predict_cells_time, col, cell, time_step - 1);
  return cell_active && was_predict;
}

void TemporalPoolerCalculator::update_avg_persist(int prev_tracking_num, float& avg_persist) const {
  // Update the average persistence count with an ARMA-style smoothing filter.
  // Mirrors python `updateAvgPesist`, but *fixes* the python bug where the value
  // was computed and then discarded.
  //
  // `delay_length` controls smoothing (larger => slower changes).
  // This average is only *used* to seed a finite persistence countdown when
  // `cfg_.enable_persistence` is true.
  if (avg_persist < 0.0f) {
    avg_persist = static_cast<float>(prev_tracking_num);
    return;
  }
  const float alpha = 1.0f - 1.0f / static_cast<float>(cfg_.delay_length);
  avg_persist = alpha * avg_persist + (1.0f - alpha) * static_cast<float>(prev_tracking_num);
}

void TemporalPoolerCalculator::update_new_learn_cells_time(
    int time_step,
    const std::vector<std::pair<int, int>>& new_learn_cells_list,
    const std::vector<int>& learn_cells_time) {
  // Update `new_learn_cells_time_` for cells that *just entered* learning state.
  //
  // Why track "entry" times separately?
  // The distal temporal pooler wants a set of cells that *most recently entered* learning
  // state (not cells that have merely stayed learning for many timesteps). This is used
  // to choose endpoints for new synapses (a proxy for "recently relevant context").
  //
  // Mirrors python logic in `getPrev2NewLearnCells` where `newLearnCellsTime` stores entry times.
  for (const auto& cc : new_learn_cells_list) {
    const int col = cc.first;
    const int cell = cc.second;
    // Only update if the cell was NOT learning at (time_step-1).
    if (!check_cell_time(learn_cells_time, col, cell, time_step - 1)) {
      const int i0 = idx_cell_time(col, cell, 0);
      const int i1 = idx_cell_time(col, cell, 1);
      if (new_learn_cells_time_[static_cast<std::size_t>(i0)] <= new_learn_cells_time_[static_cast<std::size_t>(i1)]) {
        new_learn_cells_time_[static_cast<std::size_t>(i0)] = time_step;
      } else {
        new_learn_cells_time_[static_cast<std::size_t>(i1)] = time_step;
      }
      learn_entry_log_.push_back({time_step, col, cell});
    }
  }
}

std::vector<std::pair<int, int>> TemporalPoolerCalculator::get_prev2_new_learn_cells(
    int time_step,
    const std::vector<int>& active_cells_time,
    int num_cells_needed) const {
  // Find cells that most recently entered learning state, but were NOT active at (time_step-1).
  //
  // This mirrors the intent of python `getPrev2NewLearnCells`:
  // - scan from most recent learning-entry events backwards
  // - collect at least `num_cells_needed` cells (if available)
  // - once you hit exactly `num_cells_needed` at some `latest_time_step`, also include any other
  //   cells that entered learning at that same timestep, then stop.
  //
  // Why exclude cells active at (t-1)?
  // The python comments call these "antepenultimate learning cells" (prev2). The goal is to bias
  // new synapses toward cells that *entered* learning recently but are not simply the immediately
  // previous active set, which helps create longer temporal links.
  std::vector<std::pair<int, int>> out;
  out.reserve(static_cast<std::size_t>(num_cells_needed));

  std::unordered_set<int> seen;
  seen.reserve(static_cast<std::size_t>(num_cells_needed) * 2u);

  bool found_num_cells = false;
  int latest_time_step = -1;

  for (std::size_t r = learn_entry_log_.size(); r-- > 0;) {
    const auto [t, col, cell] = learn_entry_log_[r];

    if (t < 0) {
      continue;
    }
    if (found_num_cells && t < latest_time_step) {
      break;
    }
    if (found_num_cells && t != latest_time_step) {
      continue;
    }

    // Must not have been active at (time_step-1).
    if (check_cell_time(active_cells_time, col, cell, time_step - 1)) {
      continue;
    }

    const int key = idx_cell_flat(col, cell);
    if (!seen.insert(key).second) {
      continue;
    }

    out.push_back({col, cell});
    if (static_cast<int>(out.size()) == num_cells_needed) {
      found_num_cells = true;
      latest_time_step = t;
    }
  }
  return out;
}

int TemporalPoolerCalculator::segment_num_synapses_active_prev2(
    const std::vector<DistalSynapse>& distal_synapses,
    const std::unordered_set<int>& prev2_set,
    int origin_col,
    int origin_cell,
    int seg) const {
  // Count how many connected synapses in this segment end on cells in `prev2_set`.
  // Mirrors python `segmentNumSynapsesActive`.
  int count = 0;
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(origin_col),
                                               static_cast<std::size_t>(origin_cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    // Match sub-connected synapses as well so freshly-created temporal-pooling
    // segments can be found and reinforced before they cross connect_permanence.
    if (s.perm > 0.0f) {
      const int key = idx_cell_flat(s.target_col, s.target_cell);
      if (prev2_set.find(key) != prev2_set.end()) {
        ++count;
      }
    }
  }
  return count;
}

int TemporalPoolerCalculator::get_best_matching_segment_prev2(const std::vector<DistalSynapse>& distal_synapses,
                                                             const std::unordered_set<int>& prev2_set,
                                                             int origin_col,
                                                             int origin_cell) const {
  // Find the segment with the most positive-permanence synapses ending on prev2 learning cells.
  // Mirrors python `getBestMatchingSegment`.
  //
  // We intentionally allow sub-connected synapses here (perm > 0) so temporal-pool segments
  // are reusable while they are still climbing toward connect_permanence.
  int best_seg = 0;
  int best_cnt = -1;
  for (int seg = 0; seg < cfg_.max_segments_per_cell; ++seg) {
    const int cnt = segment_num_synapses_active_prev2(distal_synapses, prev2_set, origin_col, origin_cell, seg);
    if (cnt > best_cnt) {
      best_cnt = cnt;
      best_seg = seg;
    }
  }
  if (best_cnt > cfg_.min_num_syn_threshold) {
    return best_seg;
  }
  return -1;
}

void TemporalPoolerCalculator::get_segment_active_synapses(const std::vector<DistalSynapse>& distal_synapses,
                                                          const std::vector<int>& active_cells_time,
                                                          int time_step,
                                                          int origin_col,
                                                          int origin_cell,
                                                          int seg,
                                                          int8_t* out01) const {
  // Build a 0/1 list of which synapses in the segment are active at `time_step`.
  // A synapse is active if its endpoint cell is active at that timestep.
  // Mirrors python `getSegmentActiveSynapses`.
  //
  // Note: the python routine does NOT require permanence>connect_permanence here; it checks
  // activity of endpoints regardless (we mirror that behavior).
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(origin_col),
                                               static_cast<std::size_t>(origin_cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    const DistalSynapse& s = distal_synapses[idx];
    out01[syn] = check_cell_time(active_cells_time, s.target_col, s.target_cell, time_step) ? 1 : 0;
  }
}

void TemporalPoolerCalculator::update_distal_synapses(int origin_col,
                                                     int origin_cell,
                                                     int seg,
                                                     const int8_t* seg_active_syn01,
                                                     std::vector<DistalSynapse>& distal_synapses) const {
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    if (seg_active_syn01[syn] != 1) {
      continue;
    }
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(origin_col),
                                               static_cast<std::size_t>(origin_cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    float p = distal_synapses[idx].perm + cfg_.seq_permanence_inc;
    distal_synapses[idx].perm = (p > 1.0f) ? 1.0f : p;
  }
}

int TemporalPoolerCalculator::find_least_used_seg(const std::vector<int>& active_segs_time,
                                                 int origin_col,
                                                 int origin_cell) const {
  const int base = idx_cell_seg(origin_col, origin_cell, 0);
  int least_seg = 0;
  int oldest = active_segs_time[static_cast<std::size_t>(base)];
  for (int seg = 1; seg < cfg_.max_segments_per_cell; ++seg) {
    const int t = active_segs_time[static_cast<std::size_t>(base + seg)];
    if (t < oldest) {
      oldest = t;
      least_seg = seg;
    }
  }
  return least_seg;
}

void TemporalPoolerCalculator::overwrite_segment_with_prev2(
    int time_step,
    int origin_col,
    int origin_cell,
    int seg,
    const std::vector<std::pair<int, int>>& prev2_cells,
    std::vector<DistalSynapse>& distal_synapses) const {
  // Fill the segment with a random-like selection of new synapses whose endpoints are
  // chosen from `prev2_cells` (antepenultimate learning cells).
  //
  // Mirrors python `newRandomPrevActiveSynapses` (in this temporal pooler file).
  //
  // Implementation detail:
  // - We use deterministic hashing instead of a shared RNG to remain thread-safe under taskflow.
  if (prev2_cells.empty()) {
    return;
  }
  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(time_step) << 32) ^
        (static_cast<std::uint64_t>(origin_col) * 0x9e3779b97f4a7c15ULL) ^
        (static_cast<std::uint64_t>(origin_cell) * 0xbf58476d1ce4e5b9ULL) ^
        (static_cast<std::uint64_t>(seg) * 0x94d049bb133111ebULL) ^
        static_cast<std::uint64_t>(syn);
    const std::size_t pick = deterministic_pick(seed, prev2_cells.size());
    const auto& tgt = prev2_cells[pick];
    const std::size_t idx = idx_distal_synapse(static_cast<std::size_t>(origin_col),
                                               static_cast<std::size_t>(origin_cell),
                                               static_cast<std::size_t>(seg),
                                               static_cast<std::size_t>(syn),
                                               static_cast<std::size_t>(cfg_.cells_per_column),
                                               static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                               static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    distal_synapses[idx] = DistalSynapse{tgt.first, tgt.second, cfg_.new_syn_permanence};
  }
}

void TemporalPoolerCalculator::update_distal(int time_step,
                                            const std::vector<std::pair<int, int>>& new_learn_cells_list,
                                            const std::vector<int>& learn_cells_time,
                                            std::vector<int>& predict_cells_time,
                                            const std::vector<int>& active_cells_time,
                                            std::vector<int>& active_segs_time,
                                            std::vector<DistalSynapse>& distal_synapses) {
  assert(static_cast<int>(learn_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<int>(predict_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<int>(active_cells_time.size()) == cfg_.num_columns * cfg_.cells_per_column * 2);
  assert(static_cast<int>(active_segs_time.size()) == cfg_.num_columns * cfg_.cells_per_column * cfg_.max_segments_per_cell);

  const std::size_t distal_size = static_cast<std::size_t>(cfg_.num_columns) * cfg_.cells_per_column *
                                  cfg_.max_segments_per_cell * cfg_.max_synapses_per_segment;
  assert(distal_synapses.size() == distal_size);

  prev_active_predict_support_by_column_ = last_active_predict_support_by_column_;
  prev_active_predict_support_time_ = last_active_predict_support_time_;

  if (cfg_.seq_permanence_inc <= 0.0f) {
    // Still update the learning-entry log to keep internal state consistent.
    update_new_learn_cells_time(time_step, new_learn_cells_list, learn_cells_time);
    std::fill(last_active_predict_support_by_column_.begin(),
              last_active_predict_support_by_column_.end(),
              0);
    last_active_predict_support_time_ = time_step;
    return;
  }

  // Step A: update learning-entry tracking and build the prev2 set/list.
  update_new_learn_cells_time(time_step, new_learn_cells_list, learn_cells_time);

  const int num_cells_needed = cfg_.max_synapses_per_segment;
  const std::vector<std::pair<int, int>> prev2_cells =
      get_prev2_new_learn_cells(time_step, active_cells_time, num_cells_needed);

  std::unordered_set<int> prev2_set;
  prev2_set.reserve(prev2_cells.size() * 2u);
  for (const auto& cc : prev2_cells) {
    prev2_set.insert(idx_cell_flat(cc.first, cc.second));
  }

  // Step B: per-cell updates (parallel and race-free).
  const int num_cells = cfg_.num_columns * cfg_.cells_per_column;
  std::vector<uint8_t> active_predict_cells(static_cast<std::size_t>(num_cells), 0);
  tf::Taskflow taskflow;

  taskflow.for_each_index(0, num_cells, 1, [&](int flat) {
    const int col = flat / cfg_.cells_per_column;
    const int cell = flat % cfg_.cells_per_column;

    const bool active_predict = check_cell_active_predict(active_cells_time, predict_cells_time, col, cell, time_step);
    if (active_predict) {
      active_predict_cells[static_cast<std::size_t>(flat)] = 1;
    }
    const bool was_active_prev = check_cell_time(active_cells_time, col, cell, time_step - 1);
    const bool was_predict_prev = check_cell_predict(predict_cells_time, col, cell, time_step - 1);

    // -------------------------------------------------------------------------
    // Persistence (optional): extend predictive state briefly by carrying forward
    // the most recent segment evidence for the same cell.
    //
    // This is deliberately decaying (countdown-based). Without a countdown, the condition
    // "(was_predict_prev || was_active_prev)" can become an absorbing state where once a cell
    // predicts it keeps predicting forever, even if distal support disappears.
    // -------------------------------------------------------------------------
    if (cfg_.enable_persistence) {
      if (active_predict) {
        cells_tracking_num_[static_cast<std::size_t>(flat)] += 1;
        // While the cell is in an active-predict streak, we are not "coasting" on persistence.
        cells_persistence_[static_cast<std::size_t>(flat)] = 0;
      } else {
        // If the prediction streak ended but the cell is still active, update avg persistence.
        if (cells_tracking_num_[static_cast<std::size_t>(flat)] > 0 &&
            check_cell_time(active_cells_time, col, cell, time_step)) {
          update_avg_persist(cells_tracking_num_[static_cast<std::size_t>(flat)],
                             cells_avg_persist_[static_cast<std::size_t>(flat)]);
        }

        // If a streak just ended, seed a finite "persistence countdown" based on the learned average.
        if (cells_tracking_num_[static_cast<std::size_t>(flat)] > 0) {
          const float avg = cells_avg_persist_[static_cast<std::size_t>(flat)];
          const int countdown = (avg > 0.0f) ? static_cast<int>(std::lround(avg)) : 0;
          cells_persistence_[static_cast<std::size_t>(flat)] = countdown;
        }

        cells_tracking_num_[static_cast<std::size_t>(flat)] = 0;
      }

      // Apply the countdown: while pers>0, keep the cell predicting at this timestep
      // only if there is a segment that was active on the previous timestep to carry
      // forward. This keeps predictive state aligned with what the burst gate expects.
      int& pers = cells_persistence_[static_cast<std::size_t>(flat)];
      if (pers > 0 && (was_predict_prev || was_active_prev)) {
        const int carried_seg = find_recent_active_segment(active_segs_time, col, cell, time_step - 1);
        if (carried_seg < 0) {
          pers = 0;
        } else {
          set_active_segment(active_segs_time, col, cell, carried_seg, time_step);
        if (!check_cell_predict(predict_cells_time, col, cell, time_step)) {
          set_predict_cell(predict_cells_time, col, cell, time_step);
        }
        pers -= 1;
        }
      }
    } else {
      // Persistence disabled: make sure internal counters can't accidentally influence state.
      cells_tracking_num_[static_cast<std::size_t>(flat)] = 0;
      cells_persistence_[static_cast<std::size_t>(flat)] = 0;
    }

    // If active predictive now, reinforce/create synapses connected to prev2 learning cells.
    if (active_predict) {
      const int best_seg = get_best_matching_segment_prev2(distal_synapses, prev2_set, col, cell);
      if (best_seg >= 0) {
        // Reinforce the best-matching segment:
        // - Increment permanence for synapses whose targets are in the prev2 learning-cell
        //   context used to select this segment.
        // - Decrement permanence for synapses outside that context (they are not
        //   contributing to this temporal-pooling match and should gradually weaken).
        // - Replace dead synapses (permanence <= 0) with new synapses targeting recent
        //   learning cells, so the segment adapts to the current temporal context.
        for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
          const std::size_t idx =
              idx_distal_synapse(static_cast<std::size_t>(col),
                                 static_cast<std::size_t>(cell),
                                 static_cast<std::size_t>(best_seg),
                                 static_cast<std::size_t>(syn),
                                 static_cast<std::size_t>(cfg_.cells_per_column),
                                 static_cast<std::size_t>(cfg_.max_segments_per_cell),
                                 static_cast<std::size_t>(cfg_.max_synapses_per_segment));
          const DistalSynapse s = distal_synapses[idx];
          const int target_key = idx_cell_flat(s.target_col, s.target_cell);
          if (prev2_set.find(target_key) != prev2_set.end()) {
            float p = distal_synapses[idx].perm + cfg_.seq_permanence_inc;
            distal_synapses[idx].perm = (p > 1.0f) ? 1.0f : p;
          } else if (s.perm > 0.0f && cfg_.seq_permanence_dec > 0.0f) {
            float p = distal_synapses[idx].perm - cfg_.seq_permanence_dec;
            distal_synapses[idx].perm = (p < 0.0f) ? 0.0f : p;
          }

          // Replace dead synapses with new connections to recent learning cells.
          if (distal_synapses[idx].perm <= 0.0f && !prev2_cells.empty()) {
            const std::uint64_t seed =
                static_cast<std::uint64_t>(col) * 1315423911u +
                static_cast<std::uint64_t>(cell) * 2654435761u +
                static_cast<std::uint64_t>(best_seg) * 97531u +
                static_cast<std::uint64_t>(syn) * 1013904223u +
                static_cast<std::uint64_t>(time_step);
            const std::size_t pick = deterministic_pick(splitmix64(seed), prev2_cells.size());
            const auto& tgt = prev2_cells[pick];
            distal_synapses[idx] = DistalSynapse{tgt.first, tgt.second, cfg_.new_syn_permanence};
          }
        }
      } else {
        const int seg = find_least_used_seg(active_segs_time, col, cell);
        overwrite_segment_with_prev2(time_step, col, cell, seg, prev2_cells, distal_synapses);
      }
    }
  }).name("temporal_pooler_update_distal_for_each_cell");

  executor_.run(taskflow).wait();

  std::fill(last_active_predict_support_by_column_.begin(),
            last_active_predict_support_by_column_.end(),
            0);
  for (int flat = 0; flat < num_cells; ++flat) {
    if (active_predict_cells[static_cast<std::size_t>(flat)] == 0) {
      continue;
    }
    const int col = flat / cfg_.cells_per_column;
    ++last_active_predict_support_by_column_[static_cast<std::size_t>(col)];
  }
  last_active_predict_support_time_ = time_step;
}

} // namespace temporal_pooler

