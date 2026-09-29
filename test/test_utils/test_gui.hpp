#pragma once

#include <type_traits>

#include <htm_gui/runtime.hpp>

#include <htm_flow/htm_region.hpp>
#include <htm_flow/step_observer.hpp>

namespace htm_test_gui {
namespace detail {

void register_runtime(htm_gui::IHtmRuntime& runtime,
                      htm_flow::StepObservable& observable);
void register_region(htm_flow::HTMRegion& region);

void enable();
void test_finished();
htm_gui::IHtmRuntime* wait_for_runtime();
void release_test();

}  // namespace detail

template <typename Runtime,
          typename std::enable_if_t<
              std::is_base_of_v<htm_gui::IHtmRuntime, Runtime> &&
                  std::is_base_of_v<htm_flow::StepObservable, Runtime>,
              int> = 0>
void startGui(Runtime& runtime) {
  detail::register_runtime(runtime, runtime);
}

inline void startGui(htm_flow::HTMRegion& region) {
  detail::register_region(region);
}

}  // namespace htm_test_gui
