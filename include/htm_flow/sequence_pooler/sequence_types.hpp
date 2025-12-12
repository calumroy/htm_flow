#pragma once

#include <cstddef>

namespace sequence_pooler {

// A single distal synapse endpoint.
//
// Each synapse connects *from* a (col, cell, segment) in the current cell
// *to* a target (col, cell) in the layer, with a permanence value `perm`.
struct DistalSynapse {
  int target_col = 0;
  int target_cell = 0;
  float perm = 0.0f;
};

// Flattened 5D tensor shape (python-style):
//   distalSynapses[col][cell][seg][syn] == DistalSynapse{target_col, target_cell, perm}
//
// Flat storage is:
//   (((col * cells_per_col + cell) * max_seg + seg) * max_syn + syn)
inline std::size_t idx_distal_synapse(std::size_t col,
                                      std::size_t cell,
                                      std::size_t seg,
                                      std::size_t syn,
                                      std::size_t cells_per_col,
                                      std::size_t max_seg,
                                      std::size_t max_syn) {
  return (((col * cells_per_col + cell) * max_seg + seg) * max_syn + syn);
}

} // namespace sequence_pooler

