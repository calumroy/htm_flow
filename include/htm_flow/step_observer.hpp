#pragma once

#include <atomic>

namespace htm_flow {

class StepObserver {
public:
  virtual ~StepObserver() = default;
  virtual void before_step() = 0;
  virtual void after_step() = 0;
};

class StepObservable {
public:
  void set_step_observer(StepObserver* observer) {
    step_observer_.store(observer, std::memory_order_release);
  }

protected:
  void notify_before_step() {
    if (StepObserver* observer = step_observer_.load(std::memory_order_acquire)) {
      observer->before_step();
    }
  }

  void notify_after_step() {
    if (StepObserver* observer = step_observer_.load(std::memory_order_acquire)) {
      observer->after_step();
    }
  }

private:
  std::atomic<StepObserver*> step_observer_{nullptr};
};

}  // namespace htm_flow
