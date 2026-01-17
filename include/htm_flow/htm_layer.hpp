#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <htm_gui/runtime.hpp>

#include <htm_flow/config.hpp>
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

/// A single HTM layer implementing the complete Hierarchical Temporal Memory algorithm.
///
/// Each step processes input through the following pipeline:
///
/// 1. **Overlap**: Each column computes how well its proximal synapses match the
///    current input. Columns with more connected synapses to active input bits
///    have higher overlap scores.
///
/// 2. **Inhibition**: Local competition between columns ensures sparse activation.
///    Only the columns with the highest overlap in each local neighborhood become
///    active (typically ~2% sparsity).
///
/// 3. **Spatial Learning**: Active columns strengthen proximal synapses to active
///    inputs and weaken synapses to inactive inputs. This adapts the columns'
///    receptive fields to frequently occurring patterns.
///
/// 4. **Sequence Memory**:
///    - Active Cells: Determines which cells within active columns should fire.
///      If a cell was correctly predicted, only that cell fires (sparse).
///      If no cell predicted, all cells fire (bursting = novel/unexpected).
///    - Predictive Cells: Cells with distal segments matching currently active
///      cells enter the predictive state for the next timestep.
///    - Sequence Learning: Grows and reinforces distal synapses to learn
///      temporal sequences.
///
/// 5. **Temporal Pooling**: Maintains stable cell representations across sequence
///    elements. Correctly-predicting cells persist longer, creating invariant
///    representations for learned sequences.
///
/// The layer implements IHtmRuntime, allowing direct visualization in the GUI.
/// For multi-layer hierarchies, use HTMRegion to stack layers together.
///
/// Example:
/// @code
///   HTMLayerConfig cfg = small_test_config();
///   HTMLayer layer(cfg, "SensoryLayer");
///
///   layer.set_input(sensory_data);
///   layer.step(1);
///
///   auto snap = layer.snapshot();  // For GUI visualization
///   auto cell_output = layer.output();  // For feeding to next layer
/// @endcode
class HTMLayer : public htm_gui::IHtmRuntime {
public:
  explicit HTMLayer(const HTMLayerConfig& cfg);
  HTMLayer(const HTMLayerConfig& cfg, const std::string& name);

  // --- IHtmRuntime interface ---
  htm_gui::Snapshot snapshot() const override;
  void step(int n = 1) override;
  htm_gui::ProximalSynapseQuery query_proximal(int column_x, int column_y) const override;
  int num_segments(int column_x, int column_y, int cell) const override;
  htm_gui::DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const override;
  std::vector<htm_gui::InputSequence> input_sequences() const override { return {}; }
  int input_sequence() const override { return 0; }
  void set_input_sequence(int /*id*/) override {}
  int activation_threshold() const override { return cfg_.activation_threshold; }
  std::string name() const override { return name_; }

  // --- Layer-specific methods ---

  /// Set the input for the next step. The layer does NOT take ownership;
  /// it copies the data internally.
  void set_input(const std::vector<int>& input);

  /// Set the input via shared pointer (avoids copy if caller manages lifetime).
  void set_input(std::shared_ptr<const std::vector<int>> input);

  /// Get the output of this layer (cell activations).
  /// Output is a 2D grid: (num_column_rows, num_column_cols * cells_per_column)
  /// where each cell's activation is 0 or 1.
  std::vector<int> output() const;

  /// Get the current timestep.
  int timestep() const { return timestep_; }

  /// Get the layer configuration.
  const HTMLayerConfig& config() const { return cfg_; }

  /// Get the number of columns.
  int num_columns() const { return num_columns_; }

  /// Get active column indices (sparse).
  const std::vector<int>& active_column_indices() const { return active_col_indices_; }

  /// Output dimensions for use as input to the next layer.
  int output_rows() const { return cfg_.num_column_rows; }
  int output_cols() const { return cfg_.num_column_cols * cfg_.cells_per_column; }

private:
  void step_once();
  /// Check if a timestep value is stored in a 2-slot time history tensor.
  static bool time_is_set(const std::vector<int>& time_tensor2, int idx0, int time_step);
  /// Convert (x, y) column coordinates to flat index.
  inline int flatten_col(int x, int y) const { return x + y * cfg_.num_column_cols; }

  HTMLayerConfig cfg_;
  std::string name_{"HTMLayer"};
  int timestep_{0};

  // Derived sizes computed from configuration
  int num_pot_syn_{0};   ///< pot_width * pot_height (synapses per column)
  int num_columns_{0};   ///< num_column_rows * num_column_cols
  int step_x_{0};        ///< Column stride in x direction across input
  int step_y_{0};        ///< Column stride in y direction across input

  // Random number generators (mutable for use in const methods)
  mutable std::mt19937 gen_;
  mutable std::uniform_int_distribution<int> bit01_{0, 1};
  mutable std::uniform_real_distribution<float> perm01_{0.0f, 1.0f};

  // Input buffer: flattened 2D grid of binary values (0 or 1)
  std::shared_ptr<std::vector<int>> input_;

  // Proximal synapse permanences: [column][synapse] flattened to 1D
  // Each column has num_pot_syn proximal synapses connecting to its receptive field
  std::vector<float> col_syn_perm_;

  // Overlap scores computed each timestep
  std::vector<float> col_overlap_grid_;       ///< Connected overlap (permanence > threshold)
  std::vector<float> pot_col_overlap_grid_;   ///< Potential overlap (all synapses)

  // Column activity state
  std::vector<uint8_t> col_active01_;         ///< Dense: 1 if column is active, 0 otherwise
  std::vector<int> prev_active_col_indices_;  ///< Sparse: previous timestep's active columns
  std::vector<int> active_col_indices_;       ///< Sparse: current timestep's active columns

  // Distal synapses for sequence memory (lateral connections between cells)
  // Flattened 4D tensor: [column][cell][segment][synapse]
  std::vector<sequence_pooler::DistalSynapse> distal_synapses_;

  // Pipeline stage calculators - each owns its internal state
  overlap::OverlapCalculator overlap_calc_;
  inhibition::InhibitionCalculator inhibition_calc_;
  spatiallearn::SpatialLearnCalculator spatial_learn_calc_;
  sequence_pooler::ActiveCellsCalculator active_cells_calc_;
  sequence_pooler::PredictCellsCalculator predict_cells_calc_;
  sequence_pooler::SequenceLearningCalculator seq_learn_calc_;
  temporal_pooler::TemporalPoolerCalculator temporal_pool_calc_;
};

}  // namespace htm_flow
