#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -B compute_hex_langton_timed_frames.py \
  --target-steps 4398046511104 \
  --output-dir results/hex_langton_smooth_frames_2p40 \
  --doubling-interval 16
