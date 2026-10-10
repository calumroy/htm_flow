#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <taskflow/taskflow.hpp>

#include <htm_flow/sequence_pooler/sequence_types.hpp>

namespace sequence_pooler {

// Predict-cells stage of the sequence pooler.
//
// What this component does:
// - Looks at each cell's distal segments and counts how many synapses connect to
//   cells that are active at `time_step`.
// - If a segment has more than `activation_threshold` active connected synapses, the
//   segment is considered active and will place the cell into the predictive state.
// - All cells with at least one active segment are set predictive. Multiple cells per
//   column can be predictive simultaneously, which naturally handles overlapping
//   sequences that share common column activations.
//
// What this component returns:
// - `predictCellsTime`: last two timesteps each cell was predictive.
// - `activeSegsTime`  : last timestep each segment was active (sequence segment).
// - Previous and current segment evidence for sequence learning.
class PredictCellsCalculator {
public:
  struct Config {
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;
    float connect_permanence = 0.0f;
    int activation_threshold = 0;
  };

  explicit PredictCellsCalculator(const Config& cfg);

  // Step 1. Read current active-cells time history (`active_cells_time`).
  // Step 2. Scan distal segments and mark predictive cells for this timestep.
  // Step 3. Save active-segment masks and each cell's best segment.
  void calculate_predict_cells(int time_step,
                               const std::vector<int>& active_cells_time,
                               const std::vector<DistalSynapse>& distal_synapses);

  // --- Outputs ---
  const std::vector<int>& get_predict_cells_time() const;
  const std::vector<int>& get_active_segs_time() const;
  const std::vector<int>& get_seg_ind_update() const;
  const std::vector<int8_t>& get_seg_active_syn() const;

  // Evidence from the previous and current prediction passes.
  // Each active-segment entry has a packed synapse mask. One bit identifies
  // one connected synapse whose target was active during that prediction.
  const std::vector<std::uint64_t>& get_prev_active_segment_syn_bits() const;
  const std::vector<int>& get_current_best_segments() const;
  const std::vector<std::uint64_t>& get_current_active_segment_syn_bits() const;
  int synapse_words_per_segment() const { return synapse_words_per_segment_; }

  // Sequence learning consumes the pending update, then queues the current best segment.
  std::vector<int>& get_seg_ind_update_mutable();
  std::vector<int8_t>& get_seg_active_syn_mutable();

  int num_columns() const { return cfg_.num_columns; }
  int cells_per_column() const { return cfg_.cells_per_column; }
  int max_segments_per_cell() const { return cfg_.max_segments_per_cell; }
  int max_synapses_per_segment() const { return cfg_.max_synapses_per_segment; }
  void set_connect_permanence(float permanence) { cfg_.connect_permanence = permanence; }
  void set_activation_threshold(int threshold) { cfg_.activation_threshold = threshold; }

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }
  inline int idx_cell_seg(int col, int cell, int seg) const {
    return (col * cfg_.cells_per_column + cell) * cfg_.max_segments_per_cell + seg;
  }
  bool check_cell_active(const std::vector<int>& active_cells_time,
                         int col,
                         int cell,
                         int time_step) const;

  void set_predict_cell(int col, int cell, int time_step);
  void set_active_seg(int col, int cell, int seg, int time_step);

  int count_active_connected_synapses(const std::vector<int>& active_cells_time,
                                      const std::vector<DistalSynapse>& distal_synapses,
                                      int time_step,
                                      int col,
                                      int cell,
                                      int seg) const;

  void fill_active_synapse_bits(const std::vector<int>& active_cells_time,
                                const std::vector<DistalSynapse>& distal_synapses,
                                int time_step,
                                int col,
                                int cell,
                                int seg,
                                std::uint64_t* out_bits) const;

  Config cfg_;

  // predictCellsTime: (num_columns, cells_per_column, 2)
  std::vector<int> predict_cells_time_;
  // activeSegsTime: (num_columns, cells_per_column, max_segments_per_cell)
  std::vector<int> active_segs_time_;

  // Pending best-segment update from the previous timestep. Sequence learning
  // consumes this before replacing it with the current best segment.
  std::vector<int> seg_ind_update_;
  std::vector<int8_t> seg_active_syn_;

  // Prediction evidence is kept for one timestep. Packed bits store the exact
  // synapses that caused each active segment without one byte per synapse.
  int synapse_words_per_segment_ = 0;
  std::vector<std::uint64_t> prev_active_segment_syn_bits_;
  std::vector<std::uint64_t> current_active_segment_syn_bits_;
  std::vector<int> current_best_segments_;

  tf::Executor executor_;
};

} // namespace sequence_pooler

