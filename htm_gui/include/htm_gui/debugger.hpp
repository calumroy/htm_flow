#pragma once

#include <string>

#include <htm_gui/runtime.hpp>

namespace htm_gui {

struct DebuggerOptions {
  std::string window_title{};
};

// Runs a Qt event loop and blocks until the GUI exits.
int run_debugger(int argc, char** argv, IHtmRuntime& runtime, const DebuggerOptions& opts = {});

}  // namespace htm_gui
