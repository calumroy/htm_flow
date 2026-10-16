#pragma once

#include <cstdint>
#include <vector>

namespace temporal_pooling_test_utils {

// -----------------------------------------------------------------------------
// Metrics (ported from the legacy Python suites)
//
// Python references:
// - HTM/utilities/measureTemporalPooling.py
// - HTM/utilities/sdrFunctions.py (similarInputGrids)
// -----------------------------------------------------------------------------

// Running-average temporal pooling metric:
// percent = |prev ∩ curr| / |prev|
class TemporalPoolingMeasure {
public:
  double temporalPoolingPercent(const std::vector<uint8_t>& grid01) {
    if (!prev_.empty()) {
      int prev_active = 0;
      int both_active = 0;
      for (std::size_t i = 0; i < grid01.size(); ++i) {
        const bool p = (prev_[i] != 0);
        const bool c = (grid01[i] != 0);
        prev_active += p ? 1 : 0;
        both_active += (p && c) ? 1 : 0;
      }
      const double percent = (prev_active > 0) ? (static_cast<double>(both_active) / static_cast<double>(prev_active))
                                               : 0.0;
      // Running average matches Python:
      // temporalAverage = (percentTemp + temporalAverage*(numInputGrids-1))/numInputGrids
      temporal_average_ =
          (percent + temporal_average_ * static_cast<double>(num_grids_ - 1)) / static_cast<double>(num_grids_);
    }
    prev_ = grid01;
    ++num_grids_;
    return temporal_average_;
  }

private:
  std::vector<uint8_t> prev_;
  double temporal_average_ = 0.0;
  int num_grids_ = 0;
};

// Similarity metric used heavily in the Python suites:
// similarity = |grid1 ∩ grid2| / |grid1|
inline double similarityPercent(const std::vector<uint8_t>& a01, const std::vector<uint8_t>& b01) {
  int a_active = 0;
  int both_active = 0;
  for (std::size_t i = 0; i < a01.size(); ++i) {
    const bool a = (a01[i] != 0);
    const bool b = (b01[i] != 0);
    a_active += a ? 1 : 0;
    both_active += (a && b) ? 1 : 0;
  }
  return (a_active > 0) ? (static_cast<double>(both_active) / static_cast<double>(a_active)) : 0.0;
}

// Fraction of active columns that are bursting at `time_step`.
// `burst_cols_time` is the (num_columns, 2) tensor from ActiveCellsCalculator.
// It is stamped for new bursts and for continuing bursts, so this reports the
// true bursting state. Do not infer bursting from "all cells active": several
// correctly predicted cells can also make every cell in a column active.
inline double burstFraction(const std::vector<uint8_t>& active_cols01,
                            const std::vector<int>& burst_cols_time,
                            int time_step) {
  int active = 0;
  int bursting = 0;
  for (std::size_t col = 0; col < active_cols01.size(); ++col) {
    if (active_cols01[col] == 0) {
      continue;
    }
    ++active;
    const int t0 = burst_cols_time[col * 2];
    const int t1 = burst_cols_time[col * 2 + 1];
    if (t0 == time_step || t1 == time_step) {
      ++bursting;
    }
  }
  return (active > 0) ? (static_cast<double>(bursting) / static_cast<double>(active)) : 0.0;
}

} // namespace temporal_pooling_test_utils

