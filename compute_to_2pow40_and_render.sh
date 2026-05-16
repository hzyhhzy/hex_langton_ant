#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -B compute_hex_langton_timed_frames.py \
  --target-steps 1099511627776 \
  --output-dir results/hex_langton_smooth_frames_2p40 \
  --doubling-interval 16

python -B render_hex_langton_timed_video.py \
  --frame-dir results/hex_langton_smooth_frames_2p40 \
  --output results/hex_langton_smooth_2p40_1920x1200_k16.mp4 \
  --fallback-gif-output results/hex_langton_smooth_2p40_1920x1200_k16.gif \
  --stride 16 \
  --width 1920 \
  --height 1200
