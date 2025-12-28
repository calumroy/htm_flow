#pragma once

#include <string>
#include <vector>

#include <htm_gui/snapshot.hpp>

namespace htm_gui {

// Optional input-sequence selection support.
//
// Runtimes that don't support this feature can ignore it and keep the defaults:
// - `input_sequences()` returns empty
// - `set_input_sequence()` is a no-op
struct InputSequence {
  int id{0};
  std::string name;
};

class IHtmRuntime {
public:
  virtual ~IHtmRuntime() = default;

  // Return a point-in-time snapshot for rendering.
  virtual Snapshot snapshot() const = 0;

  // Advance simulation by n steps.
  virtual void step(int n = 1) = 0;

  // Proximal (SP) synapses for a given column (potential + connected).
  virtual ProximalSynapseQuery query_proximal(int column_x, int column_y) const = 0;

  // Distal (TM) segment list size for a given cell.
  virtual int num_segments(int column_x, int column_y, int cell) const = 0;

  // Distal (TM) synapses for a specific segment.
  virtual DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const = 0;

  // Optional: input stimulus sequences (for GUI selection).
  // If empty, the GUI should hide/disable the sequence selector.
  virtual std::vector<InputSequence> input_sequences() const { return {}; }
  virtual int input_sequence() const { return 1; }
  virtual void set_input_sequence(int /*id*/) {}

  // Optional: segment activation threshold used by the predict-cells stage.
  // A segment is considered active if the number of *connected* synapses targeting
  // currently-active cells exceeds this threshold.
  virtual int activation_threshold() const { return 0; }

  // Optional: friendly title for window/status display.
  virtual std::string name() const { return "htm_flow"; }
};

}  // namespace htm_gui
