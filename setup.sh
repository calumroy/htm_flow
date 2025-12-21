#!/bin/sh
set -eu

# Idempotent/offline-friendly dependency bootstrap.
#
# - Prefer local vendored copies (taskflow/ and lib/googletest/) if present.
# - Only clone from the network if missing.

# -----------------------------------------------------------------------------
# Taskflow (header-only copy to include/taskflow)
# -----------------------------------------------------------------------------
if [ ! -d "taskflow" ]; then
  echo "taskflow/ not found; cloning (requires network)..."
  git clone https://github.com/taskflow/taskflow.git
fi

mkdir -p include/taskflow
cp -r taskflow/taskflow/* include/taskflow

# -----------------------------------------------------------------------------
# GoogleTest (only needed when BUILD_TESTS=ON)
# -----------------------------------------------------------------------------
mkdir -p lib
if [ ! -d "lib/googletest" ]; then
  echo "lib/googletest not found; cloning (requires network)..."
  (cd lib && git clone https://github.com/google/googletest/)
fi