#!/usr/bin/env bash
set -euo pipefail

# One-command runner for HIFI matcher on Raspberry Pi
# Usage:
#   bash run_matcher.sh
#
# Nucleo conveyor UART (wired Pi GPIO UART). Override before calling if needed:
#   HIFI_CONVEYOR_SERIAL=/dev/ttyACM0 bash run_matcher.sh   # ST-Link USB only
#   HIFI_CONVEYOR_SERIAL= bash run_matcher.sh               # disable conveyor serial
export HIFI_CONVEYOR_SERIAL="${HIFI_CONVEYOR_SERIAL:-/dev/serial0}"
export HIFI_CONVEYOR_BAUD="${HIFI_CONVEYOR_BAUD:-115200}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "Virtual environment .venv not found."
  echo "Run setup first: bash setup_pi.sh"
  exit 1
fi

source .venv/bin/activate

python "rayen mansali bellehy emchy.py"
