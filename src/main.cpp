#include <htm_flow/gui_runtime.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#ifdef HTM_FLOW_WITH_GUI
#include <htm_gui/debugger.hpp>
#endif

#define NUM_ITERATIONS 3

namespace {

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " [--steps N] [--gui] [--log]\n\n"
            << "Options:\n"
            << "  --steps N   Run N steps in headless mode (default: " << NUM_ITERATIONS << ")\n"
            << "  --gui       Start the Qt debugger (requires building with -DHTM_FLOW_WITH_GUI=ON)\n"
            << "  --log       Print per-stage timing logs (useful with --gui)\n\n"
            << "Notes:\n"
            << "  Input is a deterministic moving vertical line (like the temporal pooling tests),\n"
            << "  not per-step random bits.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  bool use_gui = false;
  int steps = NUM_ITERATIONS;
  bool log = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (arg == "--gui") {
      use_gui = true;
      continue;
    }
    if (arg == "--log") {
      log = true;
      continue;
    }
    if (arg == "--steps") {
      if (i + 1 >= argc) {
        std::cerr << "--steps requires a value\n";
        usage(argv[0]);
        return 2;
      }
      steps = std::atoi(argv[++i]);
      continue;
    }

    std::cerr << "Unknown arg: " << arg << "\n";
    usage(argv[0]);
    return 2;
  }

  htm_flow::HtmFlowRuntime::Config cfg;
  // Preserve old headless log/timing output by default, but keep GUI quiet unless requested.
  cfg.log_timings = (!use_gui) || log;
  htm_flow::HtmFlowRuntime runtime(cfg);

  if (use_gui) {
#ifdef HTM_FLOW_WITH_GUI
    return htm_gui::run_debugger(argc, argv, runtime);
#else
    std::cerr << "This binary was built without GUI support.\n"
              << "Rebuild with: -DHTM_FLOW_WITH_GUI=ON\n";
    return 2;
#endif
  }

  runtime.step(steps);
  return 0;
}
