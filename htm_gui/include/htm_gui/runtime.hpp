#pragma once

#include <string>

#include <htm_gui/snapshot.hpp>

namespace htm_gui {

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

  // Optional: segment activation threshold used by the predict-cells stage.
  // A segment is considered active if the number of *connected* synapses targeting
  // currently-active cells exceeds this threshold.
  virtual int activation_threshold() const { return 0; }

  // Optional: friendly title for window/status display.
  virtual std::string name() const { return "htm_flow"; }
};

}  // namespace htm_gui
