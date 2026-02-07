#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-htm_flow_gui:qt6}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

podman build -f "$SCRIPT_DIR/Containerfile" -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "Built image: $IMAGE_NAME"
