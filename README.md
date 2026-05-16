# Hexagonal Langton's Ant

[中文说明](README.zh-CN.md)

Suggested repository name: `hex_langton_ant`.

This repository contains Python scripts for simulating and rendering a hexagonal
Langton's ant rule:

- white cell: turn left by 60 degrees, flip to black
- black cell: turn right by 60 degrees, flip to white

The simulation uses axial hex coordinates and a dense NumPy grid. The hot update
loop is compiled with Numba. Long runs save resumable `.npz` snapshots and a
`metadata.csv` file; generated data and videos are intentionally ignored by git.

## Requirements

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Scripts

- `simulate_hex_langton.py`: simulate to one target step and render one PNG.
- `generate_power_images.py`: render still images at powers of two.
- `compute_hex_langton_timed_frames.py`: compute resumable timed frame data.
- `render_hex_langton_timed_video.py`: render video from timed frame data.
- `compute_to_2pow40_and_render.sh`: compute to `2^40` and render a stride-16 video.
- `continue_2p40_to_2p42.sh`: continue the same frame folder to `2^42`.
- `render_4T_stride_videos.sh`: render videos with strides `64, 32, 16, 8, 4, 2, 1`.

## Quick Examples

Render a single image:

```bash
python simulate_hex_langton.py --steps 1000000 --output results/hex_langton_1M.png
```

Render powers of two:

```bash
python generate_power_images.py --max-power 30 --output-dir results/hex_langton_powers
```

Compute timed frame data to `2^30`:

```bash
python compute_hex_langton_timed_frames.py \
  --target-steps 1073741824 \
  --output-dir results/hex_langton_smooth_frames_1G \
  --doubling-interval 16
```

Render a video from saved frame data:

```bash
python render_hex_langton_timed_video.py \
  --frame-dir results/hex_langton_smooth_frames_1G \
  --output results/hex_langton_smooth_1G_1920x1200_k16.mp4 \
  --stride 16 \
  --width 1920 \
  --height 1200
```

Compute and render the larger runs:

```bash
bash compute_to_2pow40_and_render.sh
bash continue_2p40_to_2p42.sh
bash render_4T_stride_videos.sh
```

`render_4T_stride_videos.sh` accepts environment overrides:

```bash
STRIDES="64 32" WIDTH=1280 HEIGHT=720 bash render_4T_stride_videos.sh
```

## Long Run Notes

`compute_hex_langton_timed_frames.py` writes:

- `metadata.csv`
- `frame_*.npz`
- `checkpoint.npz`

If the process is interrupted, rerun the same command with the same output
directory. It will resume from `checkpoint.npz`.

When the occupied span becomes large, snapshots are automatically downscaled
before saving to reduce disk usage. The coordinate scale is stored in each
frame's metadata and is handled by the renderer.

## Git Hygiene

Generated images, videos, `.npz` frame data, `.csv` metadata, archives, and
Python cache files are ignored by `.gitignore`.
