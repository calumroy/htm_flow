#pragma once

#include <cstdint>
#include <vector>

#include <taskflow/taskflow.hpp>

#include <htm_flow/sequence_pooler/sequence_types.hpp>

namespace sequence_pooler {

// Sequence-learning stage of the sequence pooler.
//
// This stage applies permanence updates to distal synapses based on:
// - which cells entered the learning state this timestep (positive reinforcement)
// - which segments predicted cells that did not become active (negative reinforcement)
//
// Failure is checked per segment. Another segment can keep the same cell
// predictive without hiding the first segment's failed prediction.
class SequenceLearningCalculator {
public:
  struct Config {
    int num_columns = 0;
    int cells_per_column = 0;
    int max_segments_per_cell = 0;
    int max_synapses_per_segment = 0;
    float connect_permanence = 0.0f;
    float permanence_inc = 0.0f;
    float permanence_dec = 0.0f;
  };

  explicit SequenceLearningCalculator(const Config& cfg);

  // -----------------------------------------------------------------------------
  // calculate_sequence_learning
  //
  // Update distal sequence synapses and move predictive evidence forward by one
  // timestep.
  //
  // The function applies three rules:
  //
  //   1. Positive learning
  //      When a cell enters learning at `time_step`, reinforce the active-cell
  //      update and the best predictive segment saved at `time_step - 1`.
  //      Apply any new synapses proposed by the active-cells stage.
  //
  //   2. Failed-segment learning
  //      Each segment active at `time_step - 1` predicted its origin cell for
  //      `time_step`. If that cell is not active now, decrement only the
  //      synapses that caused that segment prediction. Another segment can keep
  //      the cell predictive without hiding this failure.
  //
  //   3. Predictive handoff
  //      Replace the pending predictive update with the best segment from the
  //      current predict-cells pass. If support moves from segment A to segment
  //      B, a correct activation at the next timestep reinforces B.
  //
  // Packed synapse layout:
  //
  //   segment_flat =
  //       (column * cells_per_column + cell) * max_segments_per_cell + segment
  //
  //   word_index = segment_flat * synapse_words_per_segment + synapse / 64
  //   bit_index  = synapse % 64
  //
  // A set bit means that the connected synapse had an active target and
  // contributed to the segment prediction.
  //
  // Inputs:
  //   1. time_step
  //      Current timestep.
  //
  //   2. active_cells_time
  //      Last two timesteps each cell was active.
  //      Shape: (num_columns, cells_per_column, 2).
  //
  //   3. learn_cells_time
  //      Last two timesteps each cell was in the learning state.
  //      Shape: (num_columns, cells_per_column, 2).
  //
  //   4. prev_active_segment_syn_bits
  //      Packed causal-synapse masks from the predict-cells pass at
  //      `time_step - 1`. A segment with no set bits was not active.
  //      Shape: (num_columns, cells_per_column, max_segments_per_cell,
  //              synapse_words_per_segment).
  //
  //   5. current_best_segments
  //      Best segment from the current predict-cells pass for each cell.
  //      A value of -1 means that the cell has no active segment.
  //      Shape: (num_columns, cells_per_column).
  //
  //   6. current_active_segment_syn_bits
  //      Packed causal-synapse masks from the current predict-cells pass.
  //      Sequence learning copies the best segment's mask into the pending
  //      predictive update for the next timestep.
  //      Shape: (num_columns, cells_per_column, max_segments_per_cell,
  //              synapse_words_per_segment).
  //
  //   7. synapse_words_per_segment
  //      Number of 64-bit words used by one segment mask:
  //      ceil(max_synapses_per_segment / 64).
  //
  //   8. distal_synapses
  //      Distal synapse tensor. Permanence and proposed endpoints are updated
  //      in place.
  //      Shape: (num_columns, cells_per_column, max_segments_per_cell,
  //              max_synapses_per_segment).
  //
  //   9-12. seg_ind_update_active, seg_active_syn_active,
  //         seg_ind_new_syn_active, seg_new_syn_active
  //      Update structures produced by the active-cells stage. They identify
  //      existing synapses to adapt and new synapses to write when a cell
  //      enters learning.
  //
  //   13-14. seg_ind_update_predict, seg_active_syn_predict
  //      Pending best-segment update from `time_step - 1`. Positive learning
  //      consumes it first. The function then replaces it with the current
  //      best segment and its causal-synapse mask.
  //
  // Output:
  //   - Updates `distal_synapses` in place.
  //   - Marks consumed active-cell update indices as -1.
  //   - Queues the current best predictive segment for `time_step + 1`.
  // -----------------------------------------------------------------------------
  void calculate_sequence_learning(
      int time_step,
      const std::vector<int>& active_cells_time,
      const std::vector<int>& learn_cells_time,
      const std::vector<std::uint64_t>& prev_active_segment_syn_bits,
      const std::vector<int>& current_best_segments,
      const std::vector<std::uint64_t>& current_active_segment_syn_bits,
      int synapse_words_per_segment,
      std::vector<DistalSynapse>& distal_synapses,
      std::vector<int>& seg_ind_update_active,
      std::vector<int8_t>& seg_active_syn_active,
      std::vector<int>& seg_ind_new_syn_active,
      std::vector<DistalSynapse>& seg_new_syn_active,
      std::vector<int>& seg_ind_update_predict,
      std::vector<int8_t>& seg_active_syn_predict);

  void set_connect_permanence(float permanence) { cfg_.connect_permanence = permanence; }
  void set_learning_rates(float permanence_inc, float permanence_dec) {
    cfg_.permanence_inc = permanence_inc;
    cfg_.permanence_dec = permanence_dec;
  }

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }

  bool check_cell_time(const std::vector<int>& cells_time, int col, int cell, int time_step) const;

  // Update one segment after its cell enters learning.
  // Increase permanence for synapses whose target cells were active.
  // Decrease permanence for the other synapses.
  void apply_positive_segment_update(int origin_col,
                                     int origin_cell,
                                     int seg_index,
                                     const int8_t* active01,
                                     std::vector<DistalSynapse>& distal_synapses) const;

  // Copy new synapses proposed by the active-cells stage into one segment.
  // Skip entries whose permanence is less than zero.
  void apply_new_synapses(int origin_col,
                          int origin_cell,
                          int seg_index,
                          const DistalSynapse* new_syn_list,
                          std::vector<DistalSynapse>& distal_synapses) const;

  // Update a segment that predicted a cell which did not become active.
  // Decrease permanence only for synapses that helped make that prediction.
  // `active_synapse_bits` marks those synapses.
  void apply_failed_segment_update(int origin_col,
                                   int origin_cell,
                                   int seg,
                                   const std::uint64_t* active_synapse_bits,
                                   std::vector<DistalSynapse>& distal_synapses) const;

  Config cfg_;
  tf::Executor executor_;
};

} // namespace sequence_pooler

