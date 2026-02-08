#include <htm_flow/config_loader.hpp>
#include <htm_flow/gui_runtime.hpp>
#include <htm_flow/region_runtime.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#ifdef HTM_FLOW_WITH_GUI
#include <htm_gui/debugger.hpp>
#endif

#define NUM_ITERATIONS 3

namespace {

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " [--steps N] [--gui] [--log] [--config FILE]\n\n"
            << "Options:\n"
            << "  --steps N       Run N steps in headless mode (default: " << NUM_ITERATIONS << ")\n"
            << "  --gui           Start the Qt debugger (requires building with -DHTM_FLOW_WITH_GUI=ON)\n"
            << "  --log           Print per-stage timing logs (useful with --gui)\n"
            << "  --config FILE   Load configuration from YAML file\n"
            << "  --list-configs  List available YAML configs in configs/\n\n"
            << "Notes:\n"
            << "  Input is a deterministic moving vertical line (like the temporal pooling tests),\n"
            << "  not per-step random bits.\n\n"
            << "Examples:\n"
            << "  " << prog << " --steps 100\n"
            << "  " << prog << " --config configs/small_test.yaml --gui\n"
            << "  " << prog << " --config configs/full_temporal_pooling.yaml --steps 50\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  bool use_gui = false;
  int steps = NUM_ITERATIONS;
  bool log = false;
  std::string config_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (arg == "--list-configs") {
      std::cout << "Available YAML configs in configs/:\n";
      auto files = htm_flow::list_config_files("configs");
      if (files.empty()) {
        std::cout << "  (none found)\n";
      } else {
        for (const auto& f : files) {
          std::cout << "  " << f << "\n";
        }
      }
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
    if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "--config requires a file path\n";
        usage(argv[0]);
        return 2;
      }
      config_file = argv[++i];
      continue;
    }

    std::cerr << "Unknown arg: " << arg << "\n";
    usage(argv[0]);
    return 2;
  }

  // Use HTMRegionRuntime if config file is provided, otherwise use HtmFlowRuntime
  std::unique_ptr<htm_gui::IHtmRuntime> runtime;
  std::string config_name = "htm_flow";

  if (!config_file.empty()) {
    // Load from YAML file
    try {
      auto cfg = htm_flow::load_region_config(config_file);
      config_name = std::filesystem::path(config_file).stem().string();
      
      // Apply logging settings
      for (auto& layer_cfg : cfg.layers) {
        layer_cfg.log_timings = (!use_gui) || log;
      }
      
      std::cout << "Loaded config from file: " << config_file << "\n"
                << "  " << cfg.layers.size() << " layer" << (cfg.layers.size() > 1 ? "s" : "");
      for (std::size_t li = 0; li < cfg.layers.size(); ++li) {
        const auto& lc = cfg.layers[li];
        std::cout << "\n  layer " << li << ": input=" << lc.num_input_rows << "x" << lc.num_input_cols
                  << " columns=" << lc.num_column_rows << "x" << lc.num_column_cols
                  << " cells_per_col=" << lc.cells_per_column;
      }
      std::cout << "\n";
      
      runtime = std::make_unique<htm_flow::HTMRegionRuntime>(cfg, config_name);
    } catch (const std::exception& e) {
      std::cerr << "Error loading config: " << e.what() << "\n";
      return 1;
    }
  } else {
    // Default: single layer with default config
    htm_flow::HtmFlowRuntime::Config cfg;
    cfg.log_timings = (!use_gui) || log;

    std::cout << "Using built-in default config (no --config file specified)\n"
              << "  1 layer | input=" << cfg.num_input_rows << "x" << cfg.num_input_cols
              << " columns=" << cfg.num_column_rows << "x" << cfg.num_column_cols
              << " cells_per_col=" << cfg.cells_per_column << "\n";

    runtime = std::make_unique<htm_flow::HtmFlowRuntime>(cfg);
  }

  if (use_gui) {
#ifdef HTM_FLOW_WITH_GUI
    return htm_gui::run_debugger(argc, argv, *runtime);
#else
    std::cerr << "This binary was built without GUI support.\n"
              << "Rebuild with: -DHTM_FLOW_WITH_GUI=ON\n";
    return 2;
#endif
  }

  runtime->step(steps);
  return 0;
}
