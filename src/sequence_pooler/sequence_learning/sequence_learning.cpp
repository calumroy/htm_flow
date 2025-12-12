#include <htm_flow/sequence_pooler/sequence_learning.hpp>

#include <algorithm>
#include <cassert>
#include <taskflow/algorithm/for_each.hpp>

namespace sequence_pooler {

SequenceLearningCalculator::SequenceLearningCalculator(const Config& cfg) : cfg_(cfg) {
  assert(cfg_.num_columns > 0);
  assert(cfg_.cells_per_column > 0);
  assert(cfg_.max_segments_per_cell > 0);
  assert(cfg_.max_synapses_per_segment > 0);
}

bool SequenceLearningCalculator::check_cell_time(const std::vector<int>& cells_time,
                                                 int col,
                                                 int cell,
                                                 int time_step) const {
  const int t0 = cells_time[idx_cell_time(col, cell, 0)];
  const int t1 = cells_time[idx_cell_time(col, cell, 1)];
  return (t0 == time_step) || (t1 == time_step);
}

void SequenceLearningCalculator::apply_segment_update(int origin_col,
                                                      int origin_cell,
                                                      int seg_index,
                                                      const int8_t* active01,
                                                      bool positive_reinforcement,
                                                      std::vector<DistalSynapse>& distal_synapses) const {
  if (seg_index < 0) {
    return;
  }
  const int seg = seg_index % cfg_.max_segments_per_cell;

  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const std::size_t idx =
        idx_distal_synapse(static_cast<std::size_t>(origin_col),
                           static_cast<std::size_t>(origin_cell),
                           static_cast<std::size_t>(seg),
                           static_cast<std::size_t>(syn),
                           static_cast<std::size_t>(cfg_.cells_per_column),
                           static_cast<std::size_t>(cfg_.max_segments_per_cell),
                           static_cast<std::size_t>(cfg_.max_synapses_per_segment));

    float p = distal_synapses[idx].perm;
    const bool syn_was_active = (active01 != nullptr) ? (active01[syn] == 1) : false;

    if (positive_reinforcement) {
      // Standard HTM-style update:
      // - active synapses increase permanence
      // - inactive synapses decrease permanence
      p += syn_was_active ? cfg_.permanence_inc : -cfg_.permanence_dec;
    } else {
      // Negative reinforcement: decrement only synapses that were active for the prediction.
      if (syn_was_active) {
        p -= cfg_.permanence_dec;
      }
    }

    distal_synapses[idx].perm = std::clamp(p, 0.0f, 1.0f);
  }
}

void SequenceLearningCalculator::apply_new_synapses(int origin_col,
                                                    int origin_cell,
                                                    int seg_index,
                                                    const DistalSynapse* new_syn_list,
                                                    std::vector<DistalSynapse>& distal_synapses) const {
  if (seg_index < 0 || new_syn_list == nullptr) {
    return;
  }
  const int seg = seg_index % cfg_.max_segments_per_cell;

  for (int syn = 0; syn < cfg_.max_synapses_per_segment; ++syn) {
    const DistalSynapse proposal = new_syn_list[syn];
    // A permanence < 0 means "no synapse proposed for this slot".
    if (proposal.perm < 0.0f) {
      continue;
    }
    const std::size_t idx =
        idx_distal_synapse(static_cast<std::size_t>(origin_col),
                           static_cast<std::size_t>(origin_cell),
                           static_cast<std::size_t>(seg),
                           static_cast<std::size_t>(syn),
                           static_cast<std::size_t>(cfg_.cells_per_column),
                           static_cast<std::size_t>(cfg_.max_segments_per_cell),
                           static_cast<std::size_t>(cfg_.max_synapses_per_segment));
    distal_synapses[idx] = proposal;
  }
}

void SequenceLearningCalculator::calculate_sequence_learning(
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
    std::vector<int8_t>& seg_active_syn_predict) {
  // Shape checks.
  const int cells_time_size = cfg_.num_columns * cfg_.cells_per_column * 2;
  assert(static_cast<int>(active_cells_time.size()) == cells_time_size);
  assert(static_cast<int>(learn_cells_time.size()) == cells_time_size);
  assert(static_cast<int>(predict_cells_time.size()) == cells_time_size);

  const std::size_t distal_size = static_cast<std::size_t>(cfg_.num_columns) * cfg_.cells_per_column *
                                  cfg_.max_segments_per_cell * cfg_.max_synapses_per_segment;
  assert(distal_synapses.size() == distal_size);

  const int per_cell_size = cfg_.num_columns * cfg_.cells_per_column;
  assert(static_cast<int>(seg_ind_update_active.size()) == per_cell_size);
  assert(static_cast<int>(seg_ind_new_syn_active.size()) == per_cell_size);
  assert(static_cast<int>(seg_ind_update_predict.size()) == per_cell_size);

  const int per_cell_syn_size = per_cell_size * cfg_.max_synapses_per_segment;
  assert(static_cast<int>(seg_active_syn_active.size()) == per_cell_syn_size);
  assert(static_cast<int>(seg_active_syn_predict.size()) == per_cell_syn_size);
  assert(static_cast<int>(seg_new_syn_active.size()) == per_cell_syn_size);

  // Implementation overview:
  // - Parallelize over columns. Each column task updates only synapses that *originate*
  //   in that column, so there are no write conflicts.
  tf::Taskflow taskflow;
  taskflow.for_each_index(
      0, cfg_.num_columns, 1,
      [&](int c) {
        for (int cell = 0; cell < cfg_.cells_per_column; ++cell) {
          const bool learn_now = check_cell_time(learn_cells_time, c, cell, time_step);
          const bool learn_prev = check_cell_time(learn_cells_time, c, cell, time_step - 1);

          const bool pred_prev = check_cell_time(predict_cells_time, c, cell, time_step - 1);
          const bool pred_now = check_cell_time(predict_cells_time, c, cell, time_step);
          const bool active_now = check_cell_time(active_cells_time, c, cell, time_step);

          const int cell_flat = c * cfg_.cells_per_column + cell;

          // Positive reinforcement: cell entered learning state at this timestep.
          if (learn_now && !learn_prev) {
            const int seg_u_act = seg_ind_update_active[cell_flat];
            const int seg_u_pred = seg_ind_update_predict[cell_flat];
            const int seg_new = seg_ind_new_syn_active[cell_flat];

            const int8_t* act01 =
                &seg_active_syn_active[cell_flat * cfg_.max_synapses_per_segment];
            const int8_t* pred01 =
                &seg_active_syn_predict[cell_flat * cfg_.max_synapses_per_segment];
            const DistalSynapse* new_syn =
                &seg_new_syn_active[cell_flat * cfg_.max_synapses_per_segment];

            apply_segment_update(c, cell, seg_u_act, act01, true, distal_synapses);
            apply_segment_update(c, cell, seg_u_pred, pred01, true, distal_synapses);
            apply_new_synapses(c, cell, seg_new, new_syn, distal_synapses);

            // Mark update structures as consumed.
            seg_ind_update_active[cell_flat] = -1;
            seg_ind_new_syn_active[cell_flat] = -1;
            seg_ind_update_predict[cell_flat] = -1;
          }

          // Negative reinforcement: incorrect prediction.
          if (pred_prev && !pred_now && !active_now) {
            const int seg_u_act = seg_ind_update_active[cell_flat];
            const int seg_u_pred = seg_ind_update_predict[cell_flat];

            const int8_t* act01 =
                &seg_active_syn_active[cell_flat * cfg_.max_synapses_per_segment];
            const int8_t* pred01 =
                &seg_active_syn_predict[cell_flat * cfg_.max_synapses_per_segment];

            apply_segment_update(c, cell, seg_u_act, act01, false, distal_synapses);
            apply_segment_update(c, cell, seg_u_pred, pred01, false, distal_synapses);

            // Mark update structures as consumed.
            seg_ind_update_active[cell_flat] = -1;
            seg_ind_new_syn_active[cell_flat] = -1;
            seg_ind_update_predict[cell_flat] = -1;
          }
        }
      })
      .name("sequence_learning_for_each_column");

  executor_.run(taskflow).wait();
}

} // namespace sequence_pooler

