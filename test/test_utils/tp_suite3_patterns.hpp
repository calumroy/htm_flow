#pragma once

#include <gtest/gtest.h>

#include <vector>

namespace temporal_pooling_test_utils {

// -----------------------------------------------------------------------------
// Suite3 (image-based in Python) hardcoded patterns for C++ tests.
//
// Python reference: HTM/tests/temporalPooling/test_temporalPoolingSuite3.py
//
// The Python suite reads PNGs from HTM/tests/temporalPooling/test_seqs_suite3/.
// Per your request, we do NOT do file IO in C++; instead we provide small,
// deterministic binary patterns that mimic the *relationships*:
// - ABC vs DEC share the 'C' element in the third position
// - DEF is a different 3-step sequence
// -----------------------------------------------------------------------------

inline std::vector<int> makeGrid(int w, int h, const std::vector<std::pair<int, int>>& ones) {
  std::vector<int> g(static_cast<std::size_t>(w * h), 0);
  for (auto [x, y] : ones) {
    EXPECT_GE(x, 0);
    EXPECT_LT(x, w);
    EXPECT_GE(y, 0);
    EXPECT_LT(y, h);
    g[static_cast<std::size_t>(y * w + x)] = 1;
  }
  return g;
}

struct Suite3Patterns {
  int w = 8;
  int h = 8;

  // Single-frame “letters”.
  std::vector<int> A() const {
    // A: two diagonals meeting + bar
    return makeGrid(w, h, {{1, 6}, {2, 5}, {3, 4}, {4, 5}, {5, 6}, {2, 4}, {3, 4}, {4, 4}});
  }
  std::vector<int> B() const { return makeGrid(w, h, {{2, 2}, {2, 3}, {2, 4}, {3, 2}, {3, 4}, {4, 3}}); }
  std::vector<int> C() const { return makeGrid(w, h, {{4, 2}, {3, 2}, {2, 3}, {2, 4}, {3, 5}, {4, 5}}); }
  std::vector<int> D() const { return makeGrid(w, h, {{2, 2}, {2, 3}, {2, 4}, {3, 2}, {4, 3}, {3, 4}}); }
  std::vector<int> E() const { return makeGrid(w, h, {{2, 2}, {2, 3}, {2, 4}, {3, 2}, {3, 3}, {3, 4}}); }
  std::vector<int> F() const { return makeGrid(w, h, {{2, 2}, {2, 3}, {2, 4}, {3, 2}, {3, 3}}); }

  // 3-step sequences.
  std::vector<std::vector<int>> ABC() const { return {A(), B(), C()}; }
  std::vector<std::vector<int>> DEC() const { return {D(), E(), C()}; }
  std::vector<std::vector<int>> DEF() const { return {D(), E(), F()}; }
};

} // namespace temporal_pooling_test_utils

