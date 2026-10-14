#!/usr/bin/env bash
# Build and run one GUI-enabled htm_flow test in the Qt container.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./htm_gui/run_test_gui.sh [--theme light|dark] TEST_FILTER

Examples:
  ./htm_gui/run_test_gui.sh TemporalPoolingIntegrationSuite4.test_temporalDiff_patterns_remain_distinct
  ./htm_gui/run_test_gui.sh --theme dark TemporalPoolingIntegrationSuite4.test_temporalDiff_patterns_remain_distinct

The selected test must call htm_test_gui::startGui(network).
--theme changes window and text colors only. Column and cell colors stay the same.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -eq 1 ]] && exit 0
  exit 2
fi

THEME=""
TEST_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --theme)
      if [[ $# -lt 2 ]]; then
        echo "--theme requires a value: light|dark" >&2
        exit 2
      fi
      THEME=$2
      shift 2
      ;;
    --theme=*)
      THEME=${1#--theme=}
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$TEST_FILTER" ]]; then
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      TEST_FILTER=$1
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  if [[ -n "$TEST_FILTER" ]]; then
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 2
  fi
  TEST_FILTER=$1
  shift
  if [[ $# -gt 0 ]]; then
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 2
  fi
fi

if [[ -z "$TEST_FILTER" ]]; then
  usage >&2
  exit 2
fi

if [[ -n "$THEME" && "$THEME" != "light" && "$THEME" != "dark" ]]; then
  echo "Unsupported theme '$THEME'. Expected light or dark." >&2
  exit 2
fi

IMAGE_NAME=${IMAGE_NAME:-htm_flow_gui:qt6}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
UID_NUM=$(id -u)

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
  -e HTM_TEST_FILTER="$TEST_FILTER" \
  -e HTM_GUI_THEME="$THEME" "$IMAGE_NAME" bash -lc '
    set -euo pipefail
    if [[ ! -d include/taskflow ]]; then
      ./setup.sh
    fi
    cmake -S . -B build_qt_tests \
      -DHTM_FLOW_WITH_GUI=ON \
      -DBUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Debug
    cmake --build build_qt_tests -j2 --target htm_flow_tests
    THEME_ARGS=()
    if [[ -n "${HTM_GUI_THEME:-}" ]]; then
      THEME_ARGS=(--theme "$HTM_GUI_THEME")
    fi
    ./build_qt_tests/htm_flow_tests --gui "${THEME_ARGS[@]}" --gtest_filter="$HTM_TEST_FILTER"
  '
