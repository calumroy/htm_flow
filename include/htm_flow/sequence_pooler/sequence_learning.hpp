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
// - which cells made incorrect predictions (negative reinforcement)
//
// It consumes the "update structures" emitted by the active-cells and predict-cells stages.
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

  // Step 1. Identify which cells need positive/negative reinforcement at this timestep.
  // Step 2. Apply permanence updates to distal synapses using update structures.
  // Step 3. Optionally add proposed new synapses (from active-cells stage).
  //
  // NOTE: This function mutates:
  // - `distal_synapses` (permanence values and possibly endpoints)
  // - update-structure indices (sets used indices back to -1 to mark them consumed)
  void calculate_sequence_learning(
      int time_step,
      const std::vector<int>& active_cells_time,
      const std::vector<int>& learn_cells_time,
      const std::vector<int>& predict_cells_time,
      std::vector<DistalSynapse>& distal_synapses,
      std::vector<int>& seg_ind_update_active,
      std::vector<int8_t>& seg_active_syn_active,
      std::vector<int>& seg_ind_new_syn_active,
      std::vector<DistalSynapse>& seg_new_syn_active,
      std::vector<int>& seg_ind_update_predict,
      std::vector<int8_t>& seg_active_syn_predict);

private:
  inline int idx_cell_time(int col, int cell, int slot) const {
    return (col * cfg_.cells_per_column + cell) * 2 + slot;
  }

  bool check_cell_time(const std::vector<int>& cells_time, int col, int cell, int time_step) const;

  void apply_segment_update(int origin_col,
                            int origin_cell,
                            int seg_index,
                            const int8_t* active01,
                            bool positive_reinforcement,
                            std::vector<DistalSynapse>& distal_synapses) const;

  void apply_new_synapses(int origin_col,
                          int origin_cell,
                          int seg_index,
                          const DistalSynapse* new_syn_list,
                          std::vector<DistalSynapse>& distal_synapses) const;

  Config cfg_;
  tf::Executor executor_;
};

} // namespace sequence_pooler

