#pragma once

#include <gtest/gtest.h>

#include <utilities/sdr_inputs.hpp>

namespace temporal_pooling_test_utils {

// -----------------------------------------------------------------------------
// Input generators (inspired by the legacy Python suites)
//
// Python references:
// - HTM/utilities/simpleVerticalLineInputs.py
// - HTM/utilities/customSDRInputs.py
// -----------------------------------------------------------------------------

using VerticalLineInputs = utilities::VerticalLineInputs;

// A small helper to emulate Python's `customSDRInputs` API shape:
// - multiple named sequences ("patterns")
// - `changePattern`, `setIndex`, `sequenceProbability`
// - optional per-bit noise injection
using CustomSdrInputs = utilities::CustomSdrInputs;

} // namespace temporal_pooling_test_utils

