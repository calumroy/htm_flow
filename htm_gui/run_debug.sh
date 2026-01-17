#!/usr/bin/env bash
# Run debug_region with GUI in a container.
# Usage: ./run_debug.sh [config] [options]
# Examples:
#   ./run_debug.sh                  # Single layer with GUI
#   ./run_debug.sh 2layer           # Two-layer with GUI
#   ./run_debug.sh 3layer --train 20  # Train then debug
#   ./run_debug.sh --help           # Show debug_region options
set -euo pipefail

# Handle --help locally for quick reference
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Run debug_region with GUI in a container.

Usage: ./run_debug.sh [config] [options]

Configurations:
  1layer, single  Single layer (default)
  2layer          Two-layer hierarchy
  3layer          Three-layer hierarchy
  temporal        Temporal pooling experiment
  default         Default layer config (larger grid)

Options:
  --train N       Run N training epochs first
  --help          Show this help

Examples:
  ./run_debug.sh                  # Single layer with GUI
  ./run_debug.sh 2layer           # Two-layer with GUI
  ./run_debug.sh 3layer --train 20  # Train then debug
EOF
  exit 0
fi

IMAGE_NAME=${IMAGE_NAME:-htm_flow_gui:qt6}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

UID_NUM=$(id -u)

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

# Pass all args to debug_region
DEBUG_ARGS="${*:-}"

podman run "${COMMON_ARGS[@]}" "${DISPLAY_ARGS[@]}" "$IMAGE_NAME" bash -lc "
  set -euo pipefail
  if [[ ! -d include/taskflow ]]; then
    echo 'Taskflow headers not found; running ./setup.sh'
    ./setup.sh
  fi

  cmake -S . -B build_qt -DHTM_FLOW_WITH_GUI=ON -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release 2>/dev/null
  cmake --build build_qt -j 2>/dev/null
  ./build_qt/debug_region $DEBUG_ARGS --gui
"
