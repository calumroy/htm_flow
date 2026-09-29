#include "test_gui.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace htm_test_gui {
namespace {

class CooperativeRuntime final : public htm_gui::IHtmRuntime {
public:
  class Controller {
  public:
    void advance(int count) {
      if (count <= 0) return;
      std::unique_lock<std::mutex> lock(mutex_);
      if (released_ || test_finished_) return;
      for (int step = 0; step < count && !released_ && !test_finished_;
           ++step) {
        const std::uint64_t target = completed_steps_ + 1;
        ++permits_;
        cv_.notify_all();
        cv_.wait(lock, [&] {
          return completed_steps_ >= target || released_ || test_finished_;
        });
      }
    }

    void before_step() {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return permits_ > 0 || released_; });
      if (!released_) --permits_;
    }

    void wait_until_advanced() {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return permits_ > 0 || released_; });
    }

    void after_step() {
      std::unique_lock<std::mutex> lock(mutex_);
      ++completed_steps_;
      cv_.notify_all();
      cv_.wait(lock, [&] { return permits_ > 0 || released_; });
    }

    void release() {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
      cv_.notify_all();
    }

    void finish() {
      std::lock_guard<std::mutex> lock(mutex_);
      test_finished_ = true;
      cv_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::uint64_t permits_{0};
    std::uint64_t completed_steps_{0};
    bool released_{false};
    bool test_finished_{false};
  };

  CooperativeRuntime(htm_gui::IHtmRuntime& target, Controller& controller)
      : target_(target),
        controller_(controller),
        cached_input_sequences_(target.input_sequences()),
        cached_input_sequence_(target.input_sequence()),
        cached_layer_options_(target.layer_options()),
        cached_num_layers_(target.num_layers()),
        cached_active_layer_(target.active_layer()),
        cached_activation_threshold_(target.activation_threshold()),
        cached_name_(target.name() + " (test stepping)") {
    capture();
  }

  void capture() {
    htm_gui::Snapshot snapshot = target_.snapshot();
    if (snapshot.input) {
      snapshot.input =
          std::make_shared<const std::vector<int>>(*snapshot.input);
    }
    cached_snapshot_ = std::move(snapshot);
  }

  void target_finished() {
    target_alive_.store(false, std::memory_order_release);
  }

  htm_gui::Snapshot snapshot() const override {
    if (!target_alive_.load(std::memory_order_acquire)) {
      return cached_snapshot_;
    }
    return target_.snapshot();
  }
  void step(int n = 1) override { controller_.advance(n); }
  htm_gui::ProximalSynapseQuery query_proximal(int x, int y) const override {
    if (!target_alive_.load(std::memory_order_acquire)) return {};
    return target_.query_proximal(x, y);
  }
  int num_segments(int x, int y, int cell) const override {
    if (!target_alive_.load(std::memory_order_acquire)) return 0;
    return target_.num_segments(x, y, cell);
  }
  htm_gui::DistalSynapseQuery query_distal(
      int x, int y, int cell, int segment) const override {
    if (!target_alive_.load(std::memory_order_acquire)) return {};
    return target_.query_distal(x, y, cell, segment);
  }
  std::vector<htm_gui::InputSequence> input_sequences() const override {
    return cached_input_sequences_;
  }
  int input_sequence() const override {
    if (!target_alive_.load(std::memory_order_acquire)) {
      return cached_input_sequence_;
    }
    return target_.input_sequence();
  }
  void set_input_sequence(int id) override {
    if (!target_alive_.load(std::memory_order_acquire)) return;
    target_.set_input_sequence(id);
    cached_input_sequence_ = target_.input_sequence();
  }
  std::vector<htm_gui::InputSequence> layer_options() const override {
    return cached_layer_options_;
  }
  int num_layers() const override { return cached_num_layers_; }
  int active_layer() const override {
    if (!target_alive_.load(std::memory_order_acquire)) {
      return cached_active_layer_;
    }
    return target_.active_layer();
  }
  void set_active_layer(int index) override {
    if (!target_alive_.load(std::memory_order_acquire)) return;
    target_.set_active_layer(index);
    cached_active_layer_ = target_.active_layer();
  }
  int activation_threshold() const override {
    return cached_activation_threshold_;
  }
  std::string name() const override { return cached_name_; }
  int timestep() const override { return snapshot().timestep; }
  htm_gui::RuntimePatchResult apply_runtime_patch_file(
      const std::string& path) override {
    if (!target_alive_.load(std::memory_order_acquire)) {
      return {false, "The test network has already been destroyed."};
    }
    return target_.apply_runtime_patch_file(path);
  }

private:
  htm_gui::IHtmRuntime& target_;
  Controller& controller_;
  std::atomic<bool> target_alive_{true};
  htm_gui::Snapshot cached_snapshot_;
  std::vector<htm_gui::InputSequence> cached_input_sequences_;
  int cached_input_sequence_{0};
  std::vector<htm_gui::InputSequence> cached_layer_options_;
  int cached_num_layers_{1};
  int cached_active_layer_{0};
  int cached_activation_threshold_{0};
  std::string cached_name_;
};

class RegionView final : public htm_gui::IHtmRuntime {
public:
  explicit RegionView(htm_flow::HTMRegion& region) : region_(region) {}

  htm_gui::Snapshot snapshot() const override {
    return activeLayer().snapshot();
  }
  void step(int /*n*/ = 1) override {}
  htm_gui::ProximalSynapseQuery query_proximal(int x, int y) const override {
    return activeLayer().query_proximal(x, y);
  }
  int num_segments(int x, int y, int cell) const override {
    return activeLayer().num_segments(x, y, cell);
  }
  htm_gui::DistalSynapseQuery query_distal(
      int x, int y, int cell, int segment) const override {
    return activeLayer().query_distal(x, y, cell, segment);
  }
  std::vector<htm_gui::InputSequence> layer_options() const override {
    std::vector<htm_gui::InputSequence> options;
    for (int index = 0; index < region_.num_layers(); ++index) {
      options.push_back({index, "Layer " + std::to_string(index)});
    }
    return options;
  }
  int num_layers() const override { return region_.num_layers(); }
  int active_layer() const override { return active_layer_; }
  void set_active_layer(int index) override {
    if (index >= 0 && index < region_.num_layers()) active_layer_ = index;
  }
  int activation_threshold() const override {
    return activeLayer().activation_threshold();
  }
  std::string name() const override { return region_.name(); }
  int timestep() const override { return region_.timestep(); }

private:
  htm_flow::HTMLayer& activeLayer() {
    return region_.layer(active_layer_);
  }
  const htm_flow::HTMLayer& activeLayer() const {
    return region_.layer(active_layer_);
  }

  htm_flow::HTMRegion& region_;
  int active_layer_{0};
};

class Session final : public htm_flow::StepObserver {
public:
  void enable() {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = true;
  }

  void registerRuntime(htm_gui::IHtmRuntime& target,
                       htm_flow::StepObservable& observable) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!enabled_) return;
      if (runtime_) {
        std::cerr << "Only one startGui() call is supported per GUI test run.\n";
        return;
      }
      observable.set_step_observer(this);
      runtime_ = std::make_unique<CooperativeRuntime>(target, controller_);
      cv_.notify_all();
    }
    controller_.wait_until_advanced();
  }

  void registerRegion(htm_flow::HTMRegion& region) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!enabled_) return;
      if (runtime_) {
        std::cerr << "Only one startGui() call is supported per GUI test run.\n";
        return;
      }
      region.set_step_observer(this);
      owned_view_ = std::make_unique<RegionView>(region);
      runtime_ =
          std::make_unique<CooperativeRuntime>(*owned_view_, controller_);
      cv_.notify_all();
    }
    controller_.wait_until_advanced();
  }

  htm_gui::IHtmRuntime* waitForRuntime() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return runtime_ || test_finished_; });
    return runtime_.get();
  }

  void markTestFinished() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      test_finished_ = true;
      if (runtime_) runtime_->target_finished();
      cv_.notify_all();
    }
    controller_.finish();
  }

  void release() { controller_.release(); }

  void before_step() override { controller_.before_step(); }
  void after_step() override {
    if (runtime_) runtime_->capture();
    controller_.after_step();
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  CooperativeRuntime::Controller controller_;
  std::unique_ptr<RegionView> owned_view_;
  std::unique_ptr<CooperativeRuntime> runtime_;
  bool enabled_{false};
  bool test_finished_{false};
};

Session& session() {
  static Session value;
  return value;
}

}  // namespace

namespace detail {

void register_runtime(htm_gui::IHtmRuntime& runtime,
                      htm_flow::StepObservable& observable) {
  session().registerRuntime(runtime, observable);
}

void register_region(htm_flow::HTMRegion& region) {
  session().registerRegion(region);
}

void enable() { session().enable(); }
void test_finished() { session().markTestFinished(); }
htm_gui::IHtmRuntime* wait_for_runtime() {
  return session().waitForRuntime();
}
void release_test() { session().release(); }

}  // namespace detail
}  // namespace htm_test_gui
