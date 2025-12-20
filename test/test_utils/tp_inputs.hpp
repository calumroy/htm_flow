#pragma once

#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace temporal_pooling_test_utils {

// -----------------------------------------------------------------------------
// Input generators (inspired by the legacy Python suites)
//
// Python references:
// - HTM/utilities/simpleVerticalLineInputs.py
// - HTM/utilities/customSDRInputs.py
// -----------------------------------------------------------------------------

class VerticalLineInputs {
public:
  enum class Pattern {
    LeftToRight,
    RightToLeft,
    EvenPositions,
    OddPositions,
  };

  VerticalLineInputs(int width, int height, int seq_len)
      : width_(width), height_(height), seq_len_(seq_len), pat_(Pattern::LeftToRight), idx_(0) {
    EXPECT_GT(width_, 0);
    EXPECT_GT(height_, 0);
    EXPECT_GT(seq_len_, 0);
  }

  void setPattern(Pattern p) {
    pat_ = p;
    idx_ = 0;
  }

  void setIndex(int i) { idx_ = i % seq_len_; }

  int seqLen() const { return seq_len_; }

  // Deterministic "sequenceProbability" analogue:
  // - 1.0 => always in-sequence
  // - 0.0 => always random index (deterministic RNG)
  void setSequenceProbability(double p) { seq_prob_ = p; }

  std::vector<int> next(std::mt19937& rng) {
    int chosen = idx_;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) > seq_prob_) {
      std::uniform_int_distribution<int> pick(0, seq_len_ - 1);
      chosen = pick(rng);
    }

    std::vector<int> grid(static_cast<std::size_t>(width_ * height_), 0);
    const int x = x_for_index(chosen);
    for (int y = 0; y < height_; ++y) {
      grid[static_cast<std::size_t>(y * width_ + x)] = 1;
    }

    idx_ = (idx_ + 1) % seq_len_;
    return grid;
  }

private:
  int x_for_index(int t) const {
    switch (pat_) {
      case Pattern::LeftToRight:
        return t % width_;
      case Pattern::RightToLeft:
        return (width_ - 1 - (t % width_));
      case Pattern::EvenPositions:
        return (2 * (t % width_)) % width_;
      case Pattern::OddPositions:
        return (2 * (t % width_) + 1) % width_;
    }
    return 0;
  }

  int width_;
  int height_;
  int seq_len_;
  Pattern pat_;
  int idx_;
  double seq_prob_ = 1.0;
};

// A small helper to emulate Python's `customSDRInputs` API shape:
// - multiple named sequences ("patterns")
// - `changePattern`, `setIndex`, `sequenceProbability`
// - optional per-bit noise injection
class CustomSdrInputs {
public:
  explicit CustomSdrInputs(int width, int height) : width_(width), height_(height) {
    EXPECT_GT(width_, 0);
    EXPECT_GT(height_, 0);
  }

  int width() const { return width_; }
  int height() const { return height_; }

  // Add a new sequence (a vector of flattened 2D grids).
  // Returns the index of the newly added sequence.
  int appendSequence(const std::vector<std::vector<int>>& seq) {
    EXPECT_FALSE(seq.empty());
    for (const auto& g : seq) {
      EXPECT_EQ(static_cast<int>(g.size()), width_ * height_);
    }
    sequences_.push_back(seq);
    return static_cast<int>(sequences_.size() - 1);
  }

  int getNumInputsInSeq(int patIndex) const { return static_cast<int>(sequences_.at(static_cast<std::size_t>(patIndex)).size()); }

  void changePattern(int patternIndex) {
    patIndex_ = patternIndex;
    index_ = 0;
  }

  void setIndex(int newIndex) { index_ = newIndex; }

  void setNoise(double noise01) { noise_ = noise01; }
  void setSequenceProbability(double p) { seq_prob_ = p; }

  std::vector<int> next(std::mt19937& rng) {
    const auto& seq = sequences_.at(static_cast<std::size_t>(patIndex_));
    const int seqLen = static_cast<int>(seq.size());
    EXPECT_GT(seqLen, 0);

    int chosen = index_ % seqLen;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) > seq_prob_) {
      std::uniform_int_distribution<int> pick(0, seqLen - 1);
      chosen = pick(rng);
    }

    std::vector<int> out = seq[static_cast<std::size_t>(chosen)];

    if (noise_ > 0.0) {
      std::bernoulli_distribution noiseDist(noise_);
      for (auto& v : out) {
        if (noiseDist(rng)) {
          v = 1;
        }
      }
    }

    index_ = (index_ + 1) % seqLen;
    return out;
  }

private:
  int width_;
  int height_;

  std::vector<std::vector<std::vector<int>>> sequences_;
  int patIndex_ = 0;
  int index_ = 0;
  double noise_ = 0.0;
  double seq_prob_ = 1.0;
};

} // namespace temporal_pooling_test_utils

