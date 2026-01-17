/// HTM Network Debugger - Run any HTM configuration with the GUI.
///
/// Usage:
///   ./debug_region [config] [options]
///
/// Examples:
///   ./debug_region --gui                    # Default single layer with GUI
///   ./debug_region 2layer --gui             # 2-layer region with GUI
///   ./debug_region 3layer --train 20 --gui  # Train 20 epochs, then debug
///
/// To add custom configurations, edit the `create_config()` function below.

#include <htm_flow/config.hpp>
#include <htm_flow/region_runtime.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#ifdef HTM_FLOW_WITH_GUI
#include <htm_gui/debugger.hpp>
#endif

namespace {

/// Create a region configuration by name.
/// Add your own configurations here for debugging.
htm_flow::HTMRegionConfig create_config(const std::string& name) {
  if (name == "1layer" || name == "single") {
    // Single layer - simplest case
    htm_flow::HTMRegionConfig cfg;
    cfg.layers.push_back(htm_flow::small_test_config());
    return cfg;
  }

  if (name == "2layer") {
    // Two layers for temporal pooling
    return htm_flow::uniform_region_config(2, htm_flow::small_test_config());
  }

  if (name == "3layer") {
    // Three layer hierarchy
    return htm_flow::uniform_region_config(3, htm_flow::small_test_config());
  }

  if (name == "temporal") {
    // Temporal pooling experiment config (from Python test suite)
    return htm_flow::uniform_region_config(2, htm_flow::temporal_pooling_test_config());
  }

  if (name == "default") {
    // Default config with standard parameters
    return htm_flow::uniform_region_config(2, htm_flow::default_layer_config());
  }

  // Unknown config - return single layer as fallback
  std::cerr << "Unknown config '" << name << "', using single layer.\n";
  htm_flow::HTMRegionConfig cfg;
  cfg.layers.push_back(htm_flow::small_test_config());
  return cfg;
}

void print_help(const char* prog) {
  std::cout << "HTM Network Debugger\n\n"
            << "Usage: " << prog << " [config] [options]\n\n"
            << "Configurations:\n"
            << "  1layer, single  Single layer (default)\n"
            << "  2layer          Two-layer hierarchy\n"
            << "  3layer          Three-layer hierarchy\n"
            << "  temporal        Temporal pooling experiment\n"
            << "  default         Default layer config (larger grid)\n\n"
            << "Options:\n"
            << "  --gui           Open Qt debugger GUI\n"
            << "  --train N       Run N training epochs first\n"
            << "  --steps N       Run N steps (headless mode)\n"
            << "  --log           Enable timing logs\n"
            << "  -h, --help      Show this help\n\n"
            << "Examples:\n"
            << "  " << prog << " --gui                    # Single layer with GUI\n"
            << "  " << prog << " 2layer --gui             # 2-layer with GUI\n"
            << "  " << prog << " 3layer --train 20 --gui  # Train then debug\n"
            << "  " << prog << " temporal --steps 100     # Headless run\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string config_name = "1layer";
  bool use_gui = false;
  int train_epochs = 0;
  int steps = 10;
  bool log_timings = false;

  // Parse args
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      return 0;
    }
    if (arg == "--gui") { use_gui = true; continue; }
    if (arg == "--log") { log_timings = true; continue; }
    if (arg == "--train" && i + 1 < argc) { train_epochs = std::atoi(argv[++i]); continue; }
    if (arg == "--steps" && i + 1 < argc) { steps = std::atoi(argv[++i]); continue; }
    // Assume config name
    config_name = arg;
  }

  // Create configuration
  auto cfg = create_config(config_name);
  for (auto& layer_cfg : cfg.layers) {
    layer_cfg.log_timings = log_timings || !use_gui;
  }

  std::cout << "Config: " << config_name << " (" << cfg.layers.size() << " layer"
            << (cfg.layers.size() > 1 ? "s" : "") << ")\n";

  // Create runtime
  htm_flow::HTMRegionRuntime runtime(cfg, config_name);

  // Training phase
  if (train_epochs > 0) {
    std::cout << "Training " << train_epochs << " epochs...\n";
    for (int epoch = 0; epoch < train_epochs; ++epoch) {
      // One epoch = one full sequence
      for (int i = 0; i < cfg.layers[0].num_input_cols; ++i) {
        runtime.step(1);
      }
    }
    std::cout << "Training complete (timestep " << runtime.region().timestep() << ")\n";
  }

  // Run
  if (use_gui) {
#ifdef HTM_FLOW_WITH_GUI
    std::cout << "Starting GUI... (use Layer dropdown to switch layers)\n";
    return htm_gui::run_debugger(argc, argv, runtime);
#else
    std::cerr << "Built without GUI. Rebuild with: ./build.sh Release GUI\n";
    return 1;
#endif
  }

  // Headless
  std::cout << "Running " << steps << " steps...\n";
  runtime.step(steps);
  std::cout << "Done (timestep " << runtime.region().timestep() << ")\n";
  return 0;
}
