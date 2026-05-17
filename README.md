# Hexagonal Langton's Ant

[中文说明](README.zh-CN.md)

Python tools for simulating and rendering a hexagonal Langton's ant rule:

- white cell: turn left by 60 degrees, flip to black
- black cell: turn right by 60 degrees, flip to white

The simulation uses axial hex coordinates, NumPy arrays, and Numba-compiled hot
loops. Long runs are resumable through `checkpoint.npz`, with frame metadata
stored in `metadata.csv`.

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
- `checkpoint_viewer.py`: interactively inspect a checkpoint with mouse-wheel zoom.
- `compute_to_2pow40_and_render.sh`: compute to `2^40` and render a stride-16 video.
- `continue_2p40_to_2p42.sh`: continue the frame folder to `2^42`.
- `continue_2p42_to_2p44.sh`: continue the frame folder to `2^44`.
- `render_4T_stride_videos.sh`: render multiple stride versions from saved frames.

## Quick Examples

Render a single image:

```bash
python simulate_hex_langton.py --steps 1000000 --output results/hex_langton_1M.png
```

Render powers of two:

```bash
python generate_power_images.py --max-power 30 --output-dir results/hex_langton_powers
```

Compute timed frame data:

```bash
python compute_hex_langton_timed_frames.py \
  --target-steps 1073741824 \
  --output-dir results/hex_langton_smooth_frames_1G \
  --doubling-interval 16
```

Each computed frame prints its elapsed time and average speed, for example:

```text
frame 17509/18468  step=...  delta=...  black=...  time=12.345s  speed=123.456 Mstep/s
```

Render a video:

```bash
python render_hex_langton_timed_video.py \
  --frame-dir results/hex_langton_smooth_frames_1G \
  --output results/hex_langton_smooth_1G_1920x1200_k16.mp4 \
  --stride 16 \
  --width 1920 \
  --height 1200
```

Render with a light Cartesian grid:

```bash
python render_hex_langton_timed_video.py \
  --frame-dir results/hex_langton_smooth_frames_1G \
  --output results/hex_langton_grid.mp4 \
  --stride 16 \
  --show-cartesian-grid
```

The grid is drawn under the ant pattern. Its spacing follows the scale bar, and
the x camera center is locked to world `x=0`.

## Checkpoint Viewer

Open the interactive checkpoint viewer:

```bash
python checkpoint_viewer.py
```

Controls:

- mouse wheel: zoom around the mouse cursor
- left drag: pan
- `F`: fit full pattern
- `R`: reload checkpoint
- `+` / `-`: zoom around the window center
- `Esc`: quit

The viewer builds compressed LOD coordinate layers (`x2`, `x4`, `x8`, ...) from
the checkpoint and switches between them while zooming. The scale bar uses
nearest-neighbor hex spacing as unit `1`.

## Larger Runs

```bash
bash compute_to_2pow40_and_render.sh
bash continue_2p40_to_2p42.sh
bash continue_2p42_to_2p44.sh
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

If interrupted, rerun the same command with the same output directory. It will
resume from `checkpoint.npz`.

When the occupied span becomes large, snapshots are automatically downscaled
before saving to reduce disk usage. The coordinate scale is stored in each
frame's metadata and handled by the renderer.

## Git Hygiene

Generated images, videos, `.npz` frame data, `.csv` metadata, archives, and
Python cache files are ignored by `.gitignore`.
