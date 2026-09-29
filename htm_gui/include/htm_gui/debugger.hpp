#pragma once

#include <string>

#include <htm_gui/runtime.hpp>

namespace htm_gui {

struct DebuggerOptions {
  std::string window_title{};
  std::string theme{};  // "light" or "dark"; empty = platform default
};

// Runs a Qt event loop and blocks until the GUI exits.
int run_debugger(int argc, char** argv, IHtmRuntime& runtime, const DebuggerOptions& opts = {});

// Requests that a running debugger event loop exit. Safe to call from a worker thread.
void request_debugger_exit();

}  // namespace htm_gui
