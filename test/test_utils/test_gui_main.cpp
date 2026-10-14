#include "test_gui.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <thread>

#include <htm_gui/debugger.hpp>

int main(int argc, char** argv) {
  bool use_gui = false;
  std::string theme;
  for (int index = 1; index < argc;) {
    const std::string arg = argv[index];
    if (arg == "--gui") {
      use_gui = true;
      std::move(argv + index + 1, argv + argc, argv + index);
      --argc;
    } else if (arg == "--theme") {
      if (index + 1 >= argc) {
        std::cerr << "--theme requires a value: light|dark\n";
        return 2;
      }
      theme = argv[index + 1];
      std::move(argv + index + 2, argv + argc, argv + index);
      argc -= 2;
    } else {
      ++index;
    }
  }

  testing::InitGoogleTest(&argc, argv);
  if (!use_gui) {
    return RUN_ALL_TESTS();
  }

  htm_test_gui::detail::enable();
  int test_result = 1;
  std::thread test_thread([&] {
    test_result = RUN_ALL_TESTS();
    htm_test_gui::detail::test_finished();
    htm_gui::request_debugger_exit();
  });

  htm_gui::IHtmRuntime* runtime =
      htm_test_gui::detail::wait_for_runtime();
  if (runtime) {
    htm_gui::DebuggerOptions options;
    options.window_title = runtime->name();
    options.theme = theme;
    htm_gui::run_debugger(argc, argv, *runtime, options);
  }

  htm_test_gui::detail::release_test();
  test_thread.join();
  return test_result;
}
