#!/bin/sh
set -eu

# Idempotent/offline-friendly dependency bootstrap.
#
# - Prefer local vendored copies (taskflow/ and lib/googletest/) if present.
# - Only clone from the network if missing.

# -----------------------------------------------------------------------------
# Taskflow (header-only copy to include/taskflow)
# -----------------------------------------------------------------------------
TASKFLOW_VERSION="v3.5.0"  # Use this version as we are using C++17, and the latest taskflow requires C++20.
if [ ! -d "taskflow" ]; then
  echo "taskflow/ not found; cloning (requires network)..."
  git clone https://github.com/taskflow/taskflow.git
fi

# Pin Taskflow version (latest requires C++20 features).
if git -C taskflow rev-parse --verify -q "${TASKFLOW_VERSION}^{commit}" >/dev/null 2>&1; then
  git -C taskflow checkout -q "${TASKFLOW_VERSION}"
else
  echo "Taskflow tag ${TASKFLOW_VERSION} not found locally; fetching tags (may require network)..."
  git -C taskflow fetch -q --tags origin || true
  if git -C taskflow rev-parse --verify -q "${TASKFLOW_VERSION}^{commit}" >/dev/null 2>&1; then
    git -C taskflow checkout -q "${TASKFLOW_VERSION}"
  else
    echo "ERROR: Taskflow tag ${TASKFLOW_VERSION} not available. Connect to the network once, or vendor a taskflow/ checkout at ${TASKFLOW_VERSION}."
    exit 1
  fi
fi

mkdir -p include/taskflow
rm -rf include/taskflow/*
cp -r taskflow/taskflow/* include/taskflow

# -----------------------------------------------------------------------------
# GoogleTest (only needed when BUILD_TESTS=ON)
# -----------------------------------------------------------------------------
mkdir -p lib
if [ ! -d "lib/googletest" ]; then
  echo "lib/googletest not found; cloning (requires network)..."
  (cd lib && git clone https://github.com/google/googletest/)
fi

# -----------------------------------------------------------------------------
# yaml-cpp (for YAML config file loading)
# -----------------------------------------------------------------------------
if [ ! -d "lib/yaml-cpp" ]; then
  echo "lib/yaml-cpp not found; cloning (requires network)..."
  (cd lib && git clone https://github.com/jbeder/yaml-cpp.git)
fi