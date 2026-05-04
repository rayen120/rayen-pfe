#!/usr/bin/env bash
set -euo pipefail

# One-command setup for Raspberry Pi 5 (Bookworm Desktop)
# Usage:
#   bash setup_pi.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

echo "[1/6] Updating apt indexes..."
sudo apt update

echo "[2/6] Installing system dependencies..."
sudo apt install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-tk \
  tesseract-ocr \
  libtesseract-dev \
  libzbar0 \
  python3-opencv \
  libatlas-base-dev \
  libjpeg-dev \
  libpng-dev \
  libopenjp2-7 \
  libglib2.0-0

echo "[3/6] Creating/updating virtual environment..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip

echo "[5/6] Installing Python requirements..."
python -m pip install -r requirements.txt

echo "[6/6] Verifying key tools..."
python -c "import cv2, pytesseract, numpy, imutils, rapidfuzz; print('Python deps OK')"
tesseract --version >/dev/null
echo "Tesseract OK"

echo
echo "Setup complete."
echo "Run the app with:"
echo "  bash run_matcher.sh"
