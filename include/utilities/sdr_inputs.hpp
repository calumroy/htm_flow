#pragma once

#include <random>
#include <stdexcept>
#include <vector>

namespace utilities {

// -----------------------------------------------------------------------------
// Input generators (ported from htm_flow/test/test_utils/tp_inputs.hpp but with
// no gtest dependency so they can be reused by main/runtime code).
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
    if (width_ <= 0 || height_ <= 0 || seq_len_ <= 0) {
      throw std::invalid_argument("VerticalLineInputs: width/height/seq_len must be > 0");
    }
  }

  void setPattern(Pattern p) {
    pat_ = p;
    idx_ = 0;
  }

  void setIndex(int i) {
    if (seq_len_ <= 0) {
      idx_ = 0;
      return;
    }
    const int m = i % seq_len_;
    idx_ = (m < 0) ? (m + seq_len_) : m;
  }

  int seqLen() const { return seq_len_; }

  // Deterministic "sequenceProbability" analogue:
  // - 1.0 => always in-sequence
  // - 0.0 => always random index (RNG-driven)
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

// A moving line stimulus that can be either vertical (moves along X) or horizontal (moves along Y).
//
// Patterns used by the GUI runtime:
//  - LeftToRight:   vertical line, x=0..width-1
//  - RightToLeft:   vertical line, x=width-1..0
//  - TopToBottom:   horizontal line, y=0..height-1
//  - BottomToTop:   horizontal line, y=height-1..0
class MovingLineInputs {
public:
  enum class Pattern {
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
  };

  MovingLineInputs(int width, int height) : width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
      throw std::invalid_argument("MovingLineInputs: width/height must be > 0");
    }
  }

  void setPattern(Pattern p) {
    pat_ = p;
    idx_ = 0;
  }

  void setIndex(int i) {
    const int seq_len = seqLen();
    if (seq_len <= 0) {
      idx_ = 0;
      return;
    }
    const int m = i % seq_len;
    idx_ = (m < 0) ? (m + seq_len) : m;
  }

  int seqLen() const {
    switch (pat_) {
      case Pattern::LeftToRight:
      case Pattern::RightToLeft:
        return width_;
      case Pattern::TopToBottom:
      case Pattern::BottomToTop:
        return height_;
    }
    return width_;
  }

  void setSequenceProbability(double p) { seq_prob_ = p; }

  std::vector<int> next(std::mt19937& rng) {
    const int seq_len = seqLen();
    if (seq_len <= 0) {
      return {};
    }

    int chosen = idx_;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) > seq_prob_) {
      std::uniform_int_distribution<int> pick(0, seq_len - 1);
      chosen = pick(rng);
    }

    std::vector<int> grid(static_cast<std::size_t>(width_ * height_), 0);

    if (is_vertical()) {
      const int x = pos_for_index(chosen);
      for (int y = 0; y < height_; ++y) {
        grid[static_cast<std::size_t>(y * width_ + x)] = 1;
      }
    } else {
      const int y = pos_for_index(chosen);
      for (int x = 0; x < width_; ++x) {
        grid[static_cast<std::size_t>(y * width_ + x)] = 1;
      }
    }

    idx_ = (idx_ + 1) % seq_len;
    return grid;
  }

private:
  bool is_vertical() const {
    return pat_ == Pattern::LeftToRight || pat_ == Pattern::RightToLeft;
  }

  int pos_for_index(int t) const {
    // For vertical patterns, returns x; for horizontal patterns, returns y.
    switch (pat_) {
      case Pattern::LeftToRight:
      case Pattern::TopToBottom:
        return t;
      case Pattern::RightToLeft:
        return (width_ - 1 - t);
      case Pattern::BottomToTop:
        return (height_ - 1 - t);
    }
    return 0;
  }

  int width_;
  int height_;
  Pattern pat_{Pattern::LeftToRight};
  int idx_{0};
  double seq_prob_{1.0};
};

// A small helper to emulate Python's `customSDRInputs` API shape:
// - multiple named sequences ("patterns")
// - `changePattern`, `setIndex`, `sequenceProbability`
// - optional per-bit noise injection
class CustomSdrInputs {
public:
  explicit CustomSdrInputs(int width, int height) : width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
      throw std::invalid_argument("CustomSdrInputs: width/height must be > 0");
    }
  }

  int width() const { return width_; }
  int height() const { return height_; }

  // Add a new sequence (a vector of flattened 2D grids).
  // Returns the index of the newly added sequence.
  int appendSequence(const std::vector<std::vector<int>>& seq) {
    if (seq.empty()) {
      throw std::invalid_argument("CustomSdrInputs::appendSequence: sequence must not be empty");
    }
    for (const auto& g : seq) {
      if (static_cast<int>(g.size()) != width_ * height_) {
        throw std::invalid_argument("CustomSdrInputs::appendSequence: grid has wrong size");
      }
    }
    sequences_.push_back(seq);
    return static_cast<int>(sequences_.size() - 1);
  }

  int getNumInputsInSeq(int patIndex) const {
    return static_cast<int>(sequences_.at(static_cast<std::size_t>(patIndex)).size());
  }

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
    if (seqLen <= 0) {
      return {};
    }

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

}  // namespace utilities


