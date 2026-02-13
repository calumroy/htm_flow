#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./htm_gui/run_gui.sh --config configs/small_test.yaml --theme dark
#   ./htm_gui/run_gui.sh --config configs/small_test.yaml --log
# Theme can also be specified in YAML as gui.theme: light|dark.

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

# Optional GPU accel (ignore if not present)
if [[ -d /dev/dri ]]; then
  DISPLAY_ARGS+=(--device /dev/dri)
fi

# Forward any extra arguments to htm_flow (e.g. --config configs/small_test.yaml --theme dark)
HTM_ARGS="${*:-}"

podman run "${COMMON_ARGS[@]}" "${DISPLAY_ARGS[@]}" "$IMAGE_NAME" bash -lc "
  set -euo pipefail
  if [[ ! -d include/taskflow ]]; then
    echo 'Taskflow headers not found; running ./setup.sh (needs network)'
    ./setup.sh
  fi

  cmake -S . -B build_qt -DHTM_FLOW_WITH_GUI=ON -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build build_qt -j
  ./build_qt/htm_flow --gui --log $HTM_ARGS
"
