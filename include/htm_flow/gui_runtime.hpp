#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <htm_gui/runtime.hpp>

#include <utilities/sdr_inputs.hpp>

#include <htm_flow/inhibition.hpp>
#include <htm_flow/overlap.hpp>
#include <htm_flow/overlap_utils.hpp>
#include <htm_flow/spatiallearn.hpp>
#include <htm_flow/sequence_pooler/active_cells.hpp>
#include <htm_flow/sequence_pooler/predict_cells.hpp>
#include <htm_flow/sequence_pooler/sequence_learning.hpp>
#include <htm_flow/sequence_pooler/sequence_types.hpp>
#include <htm_flow/temporal_pooler/temporal_pooler.hpp>

namespace htm_flow {

// A concrete IHtmRuntime implementation backed by the existing htm_flow pipeline.
//
// This class owns the pipeline state (inputs, synapse permanences, distal synapses,
// time-history tensors) and exposes it to the Qt GUI via Snapshot + query_* methods.
class HtmFlowRuntime final : public htm_gui::IHtmRuntime {
public:
  struct Config {
    // Overlap / proximal topology
    int pot_width = 20;
    int pot_height = 1;
    bool center_pot_synapses = false;

    int num_input_rows = 20;
    int num_input_cols = 20;
    int num_column_rows = 20;
    int num_column_cols = 40;

    float connected_perm = 0.3f;
    int min_overlap = 3;
    // Minimum overlap threshold for the *potential-overlap* fallback pass.
    // This is what allows bootstrapping learning when all permanences start at 0
    // (so connected-overlap is ~0 everywhere).
    // NOTE: Setting this to 0 allows columns with *zero* potential overlap to be considered.
    // In the parallel inhibition implementation, that can create degenerate “everyone ties at 0”
    // behavior. A safer default is 1, which still enables bootstrapping from zero permanence
    // (because any column seeing at least one active bit in its potential pool can compete).
    int min_potential_overlap = 1;
    bool wrap_input = true;

    // Inhibition
    int inhibition_width = 40;
    int inhibition_height = 1;
    int desired_local_activity = 1;
    bool strict_local_activity = false;

    // Spatial learning
    float spatial_permanence_inc = 0.05f;
    float spatial_permanence_dec = 0.05f;
    float active_col_permanence_dec = 0.05f;

    // Sequence pooler (active/predict/learn)
    int cells_per_column = 5;
    int max_segments_per_cell = 2;
    int max_synapses_per_segment = 20;
    int min_num_syn_threshold = 5;
    int min_score_threshold = 5;
    float new_syn_permanence = 0.3f;
    float connect_permanence = 0.2f;
    int activation_threshold = 6;

    float sequence_permanence_inc = 0.05f;
    float sequence_permanence_dec = 0.05f;

    // Temporal pooler
    int temp_delay_length = 4;
    bool temp_enable_persistence = true;
    float temp_spatial_permanence_inc = 0.01f;
    float temp_sequence_permanence_inc = 0.01f;

    // Headless logging (matches the old `main.cpp` style output)
    bool log_timings = false;
  };

  HtmFlowRuntime();
  explicit HtmFlowRuntime(const Config& cfg);

  // --- IHtmRuntime ---
  htm_gui::Snapshot snapshot() const override;
  void step(int n = 1) override;
  htm_gui::ProximalSynapseQuery query_proximal(int column_x, int column_y) const override;
  int num_segments(int column_x, int column_y, int cell) const override;
  htm_gui::DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const override;
  std::vector<htm_gui::InputSequence> input_sequences() const override;
  int input_sequence() const override { return input_sequence_id_; }
  void set_input_sequence(int id) override;
  int activation_threshold() const override { return cfg_.activation_threshold; }
  std::string name() const override { return "htm_flow"; }

private:
  void step_once();
  void apply_input_sequence(int id);

  static bool time_is_set(const std::vector<int>& time_tensor2, int idx0, int time_step);

  inline int flatten_col(int x, int y) const { return x + y * cfg_.num_column_cols; }

  Config cfg_;
  int timestep_{0};

  // Derived sizes / step mapping
  int num_pot_syn_{0};
  int num_columns_{0};
  int step_x_{0};
  int step_y_{0};

  // Random sources
  mutable std::mt19937 gen_;
  mutable std::uniform_int_distribution<int> bit01_{0, 1};
  mutable std::uniform_real_distribution<float> perm01_{0.0f, 1.0f};

  // Input source (deterministic moving line; selectable patterns).
  int input_sequence_id_{1};
  utilities::MovingLineInputs line_inputs_;

  // Main state buffers
  std::shared_ptr<std::vector<int>> input_;       // size: num_input_rows*num_input_cols
  std::vector<float> col_syn_perm_;               // size: num_columns*num_pot_syn
  std::vector<float> col_overlap_grid_;           // size: num_columns (filled from OverlapCalculator each step)
  std::vector<float> pot_col_overlap_grid_;       // size: num_columns (filled from OverlapCalculator each step)
  std::vector<uint8_t> col_active01_;             // size: num_columns
  std::vector<int> prev_active_col_indices_;      // sparse list
  std::vector<int> active_col_indices_;           // sparse list (from inhibition)

  // Distal synapses (flattened)
  std::vector<sequence_pooler::DistalSynapse> distal_synapses_;

  // Pipeline calculators (owning, stateful where applicable)
  overlap::OverlapCalculator overlap_calc_;
  inhibition::InhibitionCalculator inhibition_calc_;
  spatiallearn::SpatialLearnCalculator spatial_learn_calc_;
  sequence_pooler::ActiveCellsCalculator active_cells_calc_;
  sequence_pooler::PredictCellsCalculator predict_cells_calc_;
  sequence_pooler::SequenceLearningCalculator seq_learn_calc_;
  temporal_pooler::TemporalPoolerCalculator temporal_pool_calc_;
};

}  // namespace htm_flow


