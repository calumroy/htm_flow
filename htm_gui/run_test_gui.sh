#!/usr/bin/env bash
# Build and run one GUI-enabled htm_flow test in the Qt container.
set -euo pipefail

if [[ $# -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./htm_gui/run_test_gui.sh TEST_FILTER

Example:
  ./htm_gui/run_test_gui.sh TemporalPoolingIntegrationSuite4.test_temporalDiff_patterns_remain_distinct

The selected test must call htm_test_gui::startGui(network).
EOF
  [[ $# -eq 1 ]] && exit 0
  exit 2
fi

IMAGE_NAME=${IMAGE_NAME:-htm_flow_gui:qt6}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
UID_NUM=$(id -u)
TEST_FILTER=$1

if ! podman image exists "$IMAGE_NAME" 2>/dev/null; then
  "$SCRIPT_DIR/build_image.sh"
fi

COMMON_ARGS=(
  --rm
  --userns=keep-id
  -v "$REPO_ROOT:/work/htm_flow:Z"
  -w /work/htm_flow
  -e QT_X11_NO_MITSHM=1
)

DISPLAY_ARGS=()
if [[ -n "${WAYLAND_DISPLAY-}" && -d "/run/user/$UID_NUM" ]]; then
  DISPLAY_ARGS+=(
    -e WAYLAND_DISPLAY="$WAYLAND_DISPLAY"
    -e XDG_RUNTIME_DIR="/run/user/$UID_NUM"
    -v "/run/user/$UID_NUM:/run/user/$UID_NUM:Z"
    -e QT_QPA_PLATFORM=wayland
  )
elif [[ -n "${DISPLAY-}" && -d /tmp/.X11-unix ]]; then
  DISPLAY_ARGS+=(
    -e DISPLAY="$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix:Z
    -e QT_QPA_PLATFORM=xcb
  )
else
  echo "No WAYLAND_DISPLAY or DISPLAY detected; cannot show GUI." >&2
  exit 1
fi

if [[ -d /dev/dri ]]; then
  DISPLAY_ARGS+=(--device /dev/dri)
fi

podman run "${COMMON_ARGS[@]}" "${DISPLAY_ARGS[@]}" \
  -e HTM_TEST_FILTER="$TEST_FILTER" "$IMAGE_NAME" bash -lc '
    set -euo pipefail
    if [[ ! -d include/taskflow ]]; then
      ./setup.sh
    fi
    cmake -S . -B build_qt_tests \
      -DHTM_FLOW_WITH_GUI=ON \
      -DBUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Debug
    cmake --build build_qt_tests -j2 --target htm_flow_tests
    ./build_qt_tests/htm_flow_tests --gui --gtest_filter="$HTM_TEST_FILTER"
  '
