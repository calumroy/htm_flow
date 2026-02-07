#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace htm_gui {

struct GridShape {
  int rows{0};
  int cols{0};
};

inline int flatten_xy(int x, int y, int width) {
  return x + y * width;
}

struct ColumnCellMasks {
  // Bits [0..cells_per_column) indicate per-cell state within the column.
  // Only valid for cells_per_column <= 64.
  std::uint64_t active{0};
  std::uint64_t predictive{0};
  std::uint64_t learning{0};
};

struct Snapshot {
  int timestep{0};

  GridShape input_shape{};    // input SDR shape
  GridShape columns_shape{};  // columns grid shape
  int cells_per_column{0};

  // The input SDR grid (flattened row-major: y*cols + x). Shared to avoid huge copies.
  // The runtime guarantees this pointer remains valid until the next call to step().
  std::shared_ptr<const std::vector<int>> input;

  // Sparse active columns, flattened with flatten_xy(x,y,columns_shape.cols).
  std::vector<int> active_column_indices;

  // Per-column cell state masks (size == columns_shape.rows * columns_shape.cols).
  std::vector<ColumnCellMasks> column_cell_masks;
};

struct ProximalSynapseInfo {
  int input_x{0};
  int input_y{0};
  int input_value{0};
  float permanence{0.0f};
  bool connected{false};
};

struct ProximalSynapseQuery {
  int column_x{0};
  int column_y{0};
  // "Connected overlap" and "potential overlap" for this column (as computed by the runtime).
  // Typically:
  // - overlap: count of connected synapses with active input bits
  // - potential_overlap: count of potential synapses with active input bits
  float overlap{0.0f};
  float potential_overlap{0.0f};
  std::vector<ProximalSynapseInfo> synapses;  // potential synapses; connected flagged
};

struct DistalSynapseInfo {
  int dst_column_x{0};
  int dst_column_y{0};
  int dst_cell{0};
  float permanence{0.0f};
  bool connected{false};
};

struct DistalSynapseQuery {
  int src_column_x{0};
  int src_column_y{0};
  int src_cell{0};
  int segment{0};
  std::vector<DistalSynapseInfo> synapses;
};

}  // namespace htm_gui
