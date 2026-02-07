#pragma once

#include <cstdint>
#include <vector>

namespace spatial_pooling_test_utils {

// -----------------------------------------------------------------------------
// Metrics / helpers matching the legacy Python spatialPooling suites.
//
// Python reference:
// - HTM/tests/spatialPooling/test_spatialPooling*.py
//
// In those tests, "gridsSimilar" computes Jaccard similarity over binary grids:
//   totalActive1 = sum(grid1 != 0)
//   totalActive2 = sum(grid2 != 0)
//   totalAnd     = sum((grid1 != 0) & (grid2 != 0))
//   totalUnion   = totalActive1 + totalActive2 - totalAnd
//   percent      = totalAnd / totalUnion
// -----------------------------------------------------------------------------

inline double jaccardSimilarity01(const std::vector<uint8_t>& a01, const std::vector<uint8_t>& b01) {
  int a_active = 0;
  int b_active = 0;
  int both_active = 0;
  const std::size_t n = a01.size();
  for (std::size_t i = 0; i < n; ++i) {
    const bool a = (a01[i] != 0);
    const bool b = (b01[i] != 0);
    a_active += a ? 1 : 0;
    b_active += b ? 1 : 0;
    both_active += (a && b) ? 1 : 0;
  }
  const int uni = a_active + b_active - both_active;
  return (uni > 0) ? (static_cast<double>(both_active) / static_cast<double>(uni)) : 0.0;
}

inline std::vector<int> or01(const std::vector<int>& a01, const std::vector<int>& b01) {
  const std::size_t n = a01.size();
  std::vector<int> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) out[i] = (a01[i] != 0 || b01[i] != 0) ? 1 : 0;
  return out;
}

inline std::vector<int> and01(const std::vector<int>& a01, const std::vector<int>& b01) {
  const std::size_t n = a01.size();
  std::vector<int> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) out[i] = (a01[i] != 0 && b01[i] != 0) ? 1 : 0;
  return out;
}

inline std::vector<uint8_t> toU8(const std::vector<int>& v01) {
  std::vector<uint8_t> out(v01.size(), 0);
  for (std::size_t i = 0; i < v01.size(); ++i) out[i] = (v01[i] != 0) ? 1 : 0;
  return out;
}

} // namespace spatial_pooling_test_utils


