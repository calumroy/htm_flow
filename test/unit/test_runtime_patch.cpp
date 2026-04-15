#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <htm_flow/config.hpp>
#include <htm_flow/config_loader.hpp>
#include <htm_flow/htm_layer.hpp>
#include <htm_flow/region_runtime.hpp>

namespace {

htm_flow::HTMLayerConfig make_layer_config() {
  auto cfg = htm_flow::small_test_config();
  cfg.temp_enabled = true;
  cfg.temp_delay_length = 4;
  return cfg;
}

std::vector<int> make_input(const htm_flow::HTMLayerConfig& cfg) {
  std::vector<int> input(static_cast<std::size_t>(cfg.num_input_rows * cfg.num_input_cols), 0);
  for (int y = 0; y < cfg.num_input_rows; ++y) {
    input[static_cast<std::size_t>(y * cfg.num_input_cols)] = 1;
  }
  return input;
}

std::filesystem::path write_temp_yaml(const std::string& stem, const std::string& contents) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    (stem + "_" + std::to_string(stamp) + ".yaml");
  std::ofstream out(path);
  out << contents;
  out.close();
  return path;
}

}  // namespace

TEST(RuntimePatchConfigLoader, ParsesHotSwappableFieldsAndSchedule) {
  const auto override_path = write_temp_yaml(
      "runtime_patch_override",
      R"(layers:
  - overlap:
      connected_perm: 0.45
      min_overlap: 4
    spatial_learning:
      permanence_inc: 0.02
    sequence_memory:
      activation_threshold: 7
    temporal_pooling:
      enabled: false
      delay_length: 6
)");
  const auto schedule_path = write_temp_yaml(
      "runtime_patch_schedule",
      "runtime_parameter_schedule:\n"
      "  - at_timestep: 2\n"
      "    override: " + override_path.filename().string() + "\n");

  const auto patch = htm_flow::load_runtime_patch(override_path.string());
  ASSERT_EQ(patch.layers.size(), 1u);
  ASSERT_TRUE(patch.layers[0].connected_perm.has_value());
  ASSERT_TRUE(patch.layers[0].min_overlap.has_value());
  ASSERT_TRUE(patch.layers[0].spatial_permanence_inc.has_value());
  ASSERT_TRUE(patch.layers[0].activation_threshold.has_value());
  ASSERT_TRUE(patch.layers[0].temp_enabled.has_value());
  ASSERT_TRUE(patch.layers[0].temp_delay_length.has_value());
  EXPECT_FLOAT_EQ(*patch.layers[0].connected_perm, 0.45f);
  EXPECT_EQ(*patch.layers[0].min_overlap, 4);
  EXPECT_FLOAT_EQ(*patch.layers[0].spatial_permanence_inc, 0.02f);
  EXPECT_EQ(*patch.layers[0].activation_threshold, 7);
  EXPECT_FALSE(*patch.layers[0].temp_enabled);
  EXPECT_EQ(*patch.layers[0].temp_delay_length, 6);

  const auto schedule = htm_flow::load_runtime_parameter_schedule(schedule_path.string());
  ASSERT_EQ(schedule.size(), 1u);
  EXPECT_EQ(schedule[0].at_timestep, 2);
  EXPECT_EQ(std::filesystem::path(schedule[0].override_path), override_path.lexically_normal());
}

TEST(RuntimePatchConfigLoader, RejectsUnsupportedStructuralFields) {
  const auto path = write_temp_yaml(
      "runtime_patch_invalid",
      R"(layers:
  - columns:
      cols: 99
)");

  try {
    (void)htm_flow::load_runtime_patch(path.string());
    FAIL() << "Expected runtime patch loader to reject structural fields.";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("not hot-swappable"), std::string::npos);
  }
}

TEST(RuntimePatchIntegration, LayerApplyPatchPreservesTimestepAndUpdatesConfig) {
  const auto cfg = make_layer_config();
  htm_flow::HTMLayer layer(cfg, "PatchLayer");
  layer.set_input(make_input(cfg));
  layer.step(1);
  ASSERT_EQ(layer.timestep(), 1);

  htm_flow::HTMLayerRuntimePatch patch;
  patch.spatial_permanence_inc = 0.17f;
  patch.activation_threshold = 9;
  patch.temp_delay_length = 6;

  const auto report = layer.apply_runtime_patch(patch);
  EXPECT_TRUE(report.ok());
  EXPECT_EQ(layer.timestep(), 1);
  EXPECT_FLOAT_EQ(layer.config().spatial_permanence_inc, 0.17f);
  EXPECT_EQ(layer.config().activation_threshold, 9);
  EXPECT_EQ(layer.config().temp_delay_length, 6);

  layer.step(1);
  EXPECT_EQ(layer.timestep(), 2);
}

TEST(RuntimePatchIntegration, ScheduledPatchAppliesAtExactTimestepBoundary) {
  const auto override_path = write_temp_yaml(
      "runtime_patch_scheduled_override",
      R"(layers:
  - sequence_memory:
      activation_threshold: 11
)");
  const auto schedule_path = write_temp_yaml(
      "runtime_patch_scheduled_main",
      "runtime_parameter_schedule:\n"
      "  - at_timestep: 2\n"
      "    override: " + override_path.filename().string() + "\n");
  const auto schedule = htm_flow::load_runtime_parameter_schedule(schedule_path.string());

  htm_flow::HTMRegionConfig cfg;
  cfg.layers.push_back(make_layer_config());
  htm_flow::HTMRegionRuntime runtime(cfg, "PatchRuntime");

  std::vector<int> thresholds_before_step;
  std::size_t next_override = 0;
  for (int i = 0; i < 3; ++i) {
    while (next_override < schedule.size() &&
           schedule[next_override].at_timestep == runtime.timestep()) {
      const auto result = runtime.apply_runtime_patch_file(schedule[next_override].override_path);
      ASSERT_TRUE(result.ok) << result.message;
      ++next_override;
    }
    thresholds_before_step.push_back(runtime.region().layer(0).config().activation_threshold);
    runtime.step(1);
  }

  ASSERT_EQ(thresholds_before_step.size(), 3u);
  EXPECT_EQ(thresholds_before_step[0], cfg.layers[0].activation_threshold);
  EXPECT_EQ(thresholds_before_step[1], cfg.layers[0].activation_threshold);
  EXPECT_EQ(thresholds_before_step[2], 11);
}
