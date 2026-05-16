#!/usr/bin/env bash
set -euo pipefail

# Defaults can be overridden from the shell, for example:
#   STRIDES="64 32" WIDTH=1280 HEIGHT=720 bash render_4T_stride_videos.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
FRAME_DIR="${FRAME_DIR:-results/hex_langton_smooth_frames_2p40}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
VIDEO_TAG="${VIDEO_TAG:-4T}"
FPS="${FPS:-30}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1200}"
FINAL_HOLD_FRAMES="${FINAL_HOLD_FRAMES:-30}"
STRIDES=(${STRIDES:-64 32 16 8 4 2 1})

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$FRAME_DIR/metadata.csv" ]]; then
  echo "metadata.csv not found in: $FRAME_DIR" >&2
  exit 1
fi

export FRAME_DIR_FOR_CHECK="$FRAME_DIR"
"$PYTHON_BIN" -B -c 'import csv, os, pathlib
path = pathlib.Path(os.environ["FRAME_DIR_FOR_CHECK"]) / "metadata.csv"
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
last = rows[-1]
print(f"source frames: {len(rows)}")
print("last step: {:,}".format(int(last["step"])))
print(f"frame dir: {path.parent}")'

for stride in "${STRIDES[@]}"; do
  output="$OUTPUT_DIR/hex_langton_smooth_${VIDEO_TAG}_${WIDTH}x${HEIGHT}_k${stride}.mp4"
  fallback="$OUTPUT_DIR/hex_langton_smooth_${VIDEO_TAG}_${WIDTH}x${HEIGHT}_k${stride}.gif"
  echo
  echo "Rendering stride=$stride -> $output"
  "$PYTHON_BIN" -B render_hex_langton_timed_video.py \
    --frame-dir "$FRAME_DIR" \
    --output "$output" \
    --fallback-gif-output "$fallback" \
    --fps "$FPS" \
    --stride "$stride" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --final-hold-frames "$FINAL_HOLD_FRAMES"
done

echo
echo "Done."
