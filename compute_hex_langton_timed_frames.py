import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit


# Edit these settings.
TARGET_STEPS = 1 << 30
OUTPUT_DIR = "hex_langton_smooth_frames_1G"
FPS = 30
INITIAL_SPEED_STEPS_PER_SECOND = 4
INITIAL_SPEED_DURATION_SECONDS = 16
DOUBLING_INTERVAL_SECONDS = 32
BACKUP_POWER_STEP = 0.25
AUTO_SCALE_SNAPSHOTS = True
SCALE_SPAN_HIGH = 8192
SCALE_SPAN_LOW = 4096
INITIAL_GRID_SIZE = 20000
MARGIN_CELLS = 256
CHUNK_STEPS = 100_000_000


METADATA_FILE = "metadata.csv"
CHECKPOINT_FILE = "checkpoint.npz"


@njit
def advance_until_margin(
    grid,
    steps,
    margin,
    offset,
    q,
    r,
    direction,
    black,
    minq,
    maxq,
    minr,
    maxr,
    mins,
    maxs,
):
    completed = 0
    for _ in range(steps):
        if grid[q, r]:
            direction = (direction + 1) % 6
            grid[q, r] = 0
            black -= 1
        else:
            direction = (direction + 5) % 6
            grid[q, r] = 1
            black += 1

        if direction == 0:
            q += 1
        elif direction == 1:
            q += 1
            r -= 1
        elif direction == 2:
            r -= 1
        elif direction == 3:
            q -= 1
        elif direction == 4:
            q -= 1
            r += 1
        else:
            r += 1

        aq = q - offset
        ar = r - offset
        s = -aq - ar
        if q < minq:
            minq = q
        if q > maxq:
            maxq = q
        if r < minr:
            minr = r
        if r > maxr:
            maxr = r
        if s < mins:
            mins = s
        if s > maxs:
            maxs = s

        completed += 1
        if q <= margin or r <= margin or q >= grid.shape[0] - margin - 1 or r >= grid.shape[1] - margin - 1:
            break

    return completed, q, r, direction, black, minq, maxq, minr, maxr, mins, maxs


@njit
def advance_until_margin_v2(
    grid,
    steps,
    margin,
    offset,
    q,
    r,
    direction,
    black,
    minq,
    maxq,
    minr,
    maxr,
    mins,
    maxs,
):
    completed = 0
    height = grid.shape[0]
    width = grid.shape[1]
    q_hi = height - margin - 1
    r_hi = width - margin - 1
    flat = grid.ravel()
    pos = q * width + r

    for _ in range(steps):
        if flat[pos]:
            direction += 1
            if direction == 6:
                direction = 0
            flat[pos] = 0
            black -= 1
        else:
            if direction == 0:
                direction = 5
            else:
                direction -= 1
            flat[pos] = 1
            black += 1

        if direction == 0:
            q += 1
            pos += width
        elif direction == 1:
            q += 1
            r -= 1
            pos += width - 1
        elif direction == 2:
            r -= 1
            pos -= 1
        elif direction == 3:
            q -= 1
            pos -= width
        elif direction == 4:
            q -= 1
            r += 1
            pos += 1 - width
        else:
            r += 1
            pos += 1

        aq = q - offset
        ar = r - offset
        s = -aq - ar
        if q < minq:
            minq = q
        if q > maxq:
            maxq = q
        if r < minr:
            minr = r
        if r > maxr:
            maxr = r
        if s < mins:
            mins = s
        if s > maxs:
            maxs = s

        completed += 1
        if q <= margin or r <= margin or q >= q_hi or r >= r_hi:
            break

    return completed, q, r, direction, black, minq, maxq, minr, maxr, mins, maxs


def speed_at_time(seconds):
    if seconds < INITIAL_SPEED_DURATION_SECONDS:
        return float(INITIAL_SPEED_STEPS_PER_SECOND)
    elapsed = seconds - INITIAL_SPEED_DURATION_SECONDS
    return INITIAL_SPEED_STEPS_PER_SECOND * (2 ** (elapsed / DOUBLING_INTERVAL_SECONDS))


def step_at_time(seconds):
    if seconds <= INITIAL_SPEED_DURATION_SECONDS:
        step = int(INITIAL_SPEED_STEPS_PER_SECOND * seconds)
    else:
        elapsed = seconds - INITIAL_SPEED_DURATION_SECONDS
        initial_steps = INITIAL_SPEED_STEPS_PER_SECOND * INITIAL_SPEED_DURATION_SECONDS
        growth_steps = (
            INITIAL_SPEED_STEPS_PER_SECOND
            * DOUBLING_INTERVAL_SECONDS
            / math.log(2)
            * (2 ** (elapsed / DOUBLING_INTERVAL_SECONDS) - 1)
        )
        step = int(initial_steps + growth_steps)
    return min(step, TARGET_STEPS)


def frame_schedule():
    rows = []
    frame = 0
    while True:
        seconds = frame / FPS
        step = step_at_time(seconds)
        rows.append(
            {
                "frame": frame,
                "time_seconds": f"{seconds:.6f}",
                "speed_steps_per_second": f"{speed_at_time(seconds):.6f}",
                "step": step,
            }
        )
        if step >= TARGET_STEPS:
            break
        frame += 1
    return rows


def backup_steps():
    steps = []
    power = 0.0
    while True:
        step = int(2**power)
        if step > TARGET_STEPS:
            break
        if not steps or step > steps[-1]:
            steps.append(step)
        power += BACKUP_POWER_STEP
    if not steps or steps[-1] != TARGET_STEPS:
        steps.append(TARGET_STEPS)
    return steps


def expand_grid(grid, state, offset, pad):
    q, r, direction, black, minq, maxq, minr, maxr, mins, maxs = state
    expanded = np.zeros((grid.shape[0] + 2 * pad, grid.shape[1] + 2 * pad), np.uint8)
    expanded[pad : pad + grid.shape[0], pad : pad + grid.shape[1]] = grid
    shifted_state = (
        q + pad,
        r + pad,
        direction,
        black,
        minq + pad,
        maxq + pad,
        minr + pad,
        maxr + pad,
        mins,
        maxs,
    )
    return expanded, shifted_state, offset + pad


def advance_dynamic(grid, state, offset, steps):
    done = 0
    expansions = 0
    while done < steps:
        delta = min(CHUNK_STEPS, steps - done)
        advanced = advance_until_margin_v2(grid, delta, MARGIN_CELLS, offset, *state)
        completed = advanced[0]
        state = advanced[1:]
        done += completed
        if completed < delta:
            pad = max(INITIAL_GRID_SIZE // 2, MARGIN_CELLS * 4, grid.shape[0] // 2)
            grid, state, offset = expand_grid(grid, state, offset, pad)
            expansions += 1
    return grid, state, offset, expansions


def exact_cells(grid, state, offset):
    _, _, _, _, minq, maxq, minr, maxr, _, _ = state
    q0 = max(0, minq - 2)
    q1 = min(grid.shape[0], maxq + 3)
    r0 = max(0, minr - 2)
    r1 = min(grid.shape[1], maxr + 3)
    coords = np.argwhere(grid[q0:q1, r0:r1] != 0)
    cells = np.empty((coords.shape[0], 2), dtype=np.int32)
    if coords.shape[0]:
        cells[:, 0] = coords[:, 0] + q0 - offset
        cells[:, 1] = coords[:, 1] + r0 - offset
    return cells


def scaled_cells(grid, state, offset, coord_scale):
    _, _, _, _, minq, maxq, minr, maxr, _, _ = state
    q_min = minq - offset
    q_max = maxq - offset
    r_min = minr - offset
    r_max = maxr - offset

    q_block_min = int(np.floor(q_min / coord_scale))
    q_block_max = int(np.floor(q_max / coord_scale))
    r_block_min = int(np.floor(r_min / coord_scale))
    r_block_max = int(np.floor(r_max / coord_scale))

    q0 = q_block_min * coord_scale + offset
    q1 = (q_block_max + 1) * coord_scale + offset
    r0 = r_block_min * coord_scale + offset
    r1 = (r_block_max + 1) * coord_scale + offset

    crop = grid[q0:q1, r0:r1]
    q_blocks = q_block_max - q_block_min + 1
    r_blocks = r_block_max - r_block_min + 1
    occupied = crop.reshape(q_blocks, coord_scale, r_blocks, coord_scale).any(axis=(1, 3))
    coords = np.argwhere(occupied)
    cells = np.empty((coords.shape[0], 2), dtype=np.int32)
    if coords.shape[0]:
        cells[:, 0] = coords[:, 0] + q_block_min
        cells[:, 1] = coords[:, 1] + r_block_min
    return cells


def snapshot_cells(grid, state, offset):
    coord_scale = snapshot_scale(state, offset) if AUTO_SCALE_SNAPSHOTS else 1
    if coord_scale > 1:
        cells = scaled_cells(grid, state, offset, coord_scale)
    else:
        cells = exact_cells(grid, state, offset)
    return cells, coord_scale


def snapshot_scale(state, offset):
    _, _, _, _, minq, maxq, minr, maxr, _, _ = state
    q_min = minq - offset
    q_max = maxq - offset
    r_min = minr - offset
    r_max = maxr - offset
    xs = [
        np.sqrt(3) * (q_min + r_min / 2),
        np.sqrt(3) * (q_min + r_max / 2),
        np.sqrt(3) * (q_max + r_min / 2),
        np.sqrt(3) * (q_max + r_max / 2),
    ]
    span = max(xs) - min(xs)
    coord_scale = 1
    while span >= SCALE_SPAN_HIGH:
        coord_scale *= 2
        span /= 2
    return coord_scale


def state_metadata(state, offset):
    q, r, direction, black, minq, maxq, minr, maxr, mins, maxs = state
    aq = q - offset
    ar = r - offset
    return {
        "q": int(aq),
        "r": int(ar),
        "s": int(-aq - ar),
        "direction": int(direction),
        "black_cells": int(black),
        "bbox_q_min": int(minq - offset),
        "bbox_q_max": int(maxq - offset),
        "bbox_r_min": int(minr - offset),
        "bbox_r_max": int(maxr - offset),
        "bbox_s_min": int(mins),
        "bbox_s_max": int(maxs),
    }


def save_frame(output_dir, schedule_row, state, offset, grid):
    step = int(schedule_row["step"])
    frame = int(schedule_row["frame"])
    cells, coord_scale = snapshot_cells(grid, state, offset)
    meta = state_metadata(state, offset)
    meta["coord_scale"] = int(coord_scale)
    name = f"frame_{frame:06d}_step_{step}.npz"
    np.savez_compressed(output_dir / name, step=np.int64(step), cells=cells, meta=json.dumps(meta))
    return {
        **schedule_row,
        **meta,
        "file": name,
    }


def save_checkpoint(output_dir, current_step, next_frame_index, next_backup_index, state, offset, grid):
    cells = exact_cells(grid, state, offset)
    meta = state_metadata(state, offset)
    meta["coord_scale"] = 1
    np.savez_compressed(
        output_dir / CHECKPOINT_FILE,
        current_step=np.int64(current_step),
        next_frame_index=np.int64(next_frame_index),
        next_backup_index=np.int64(next_backup_index),
        grid_size=np.int64(grid.shape[0]),
        cells=cells,
        meta=json.dumps(meta),
    )


def load_checkpoint(output_dir):
    path = output_dir / CHECKPOINT_FILE
    if not path.exists():
        offset = INITIAL_GRID_SIZE // 2
        grid = np.zeros((INITIAL_GRID_SIZE, INITIAL_GRID_SIZE), np.uint8)
        state = (offset, offset, 0, 0, offset, offset, offset, offset, 0, 0)
        return grid, state, offset, 0, 0, 0

    data = np.load(path, allow_pickle=False)
    grid_size = int(data["grid_size"])
    offset = grid_size // 2
    grid = np.zeros((grid_size, grid_size), np.uint8)
    cells = data["cells"]
    if len(cells):
        grid[cells[:, 0] + offset, cells[:, 1] + offset] = 1

    meta = json.loads(str(data["meta"]))
    state = (
        offset + meta["q"],
        offset + meta["r"],
        meta["direction"],
        meta["black_cells"],
        offset + meta["bbox_q_min"],
        offset + meta["bbox_q_max"],
        offset + meta["bbox_r_min"],
        offset + meta["bbox_r_max"],
        meta["bbox_s_min"],
        meta["bbox_s_max"],
    )
    return (
        grid,
        state,
        offset,
        int(data["current_step"]),
        int(data["next_frame_index"]),
        int(data["next_backup_index"]),
    )


def read_metadata(output_dir):
    path = output_dir / METADATA_FILE
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_metadata(output_dir, rows):
    if not rows:
        return
    with (output_dir / METADATA_FILE).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value):
    return int(value, 0)


def main():
    global TARGET_STEPS, OUTPUT_DIR, FPS, INITIAL_SPEED_STEPS_PER_SECOND
    global INITIAL_SPEED_DURATION_SECONDS, DOUBLING_INTERVAL_SECONDS, BACKUP_POWER_STEP
    global AUTO_SCALE_SNAPSHOTS, SCALE_SPAN_HIGH, SCALE_SPAN_LOW, INITIAL_GRID_SIZE, MARGIN_CELLS, CHUNK_STEPS

    parser = argparse.ArgumentParser(description="Compute timed snapshots for the hexagonal Langton ant.")
    parser.add_argument("--target-steps", type=parse_int, default=TARGET_STEPS, help="Final simulation step, e.g. 1073741824 or 0x40000000.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for frame npz files, metadata.csv, and checkpoint.npz.")
    parser.add_argument("--fps", type=int, default=FPS, help="Frame rate used when constructing the exponential time schedule.")
    parser.add_argument("--initial-speed", type=float, default=INITIAL_SPEED_STEPS_PER_SECOND, help="Initial video speed in steps/s.")
    parser.add_argument("--initial-speed-duration", type=float, default=INITIAL_SPEED_DURATION_SECONDS, help="Seconds to hold the initial speed.")
    parser.add_argument("--doubling-interval", type=float, default=DOUBLING_INTERVAL_SECONDS, help="Seconds per 2x speed increase after the initial hold.")
    parser.add_argument("--backup-power-step", type=float, default=BACKUP_POWER_STEP, help="Checkpoint spacing in powers of two.")
    parser.add_argument("--no-auto-scale", action="store_true", help="Save exact coordinates instead of downscaled snapshot coordinates.")
    parser.add_argument("--scale-span-high", type=int, default=SCALE_SPAN_HIGH, help="Begin snapshot downscaling when span reaches this value.")
    parser.add_argument("--scale-span-low", type=int, default=SCALE_SPAN_LOW, help="Target lower span bound after downscaling.")
    parser.add_argument("--initial-grid-size", type=int, default=INITIAL_GRID_SIZE, help="Initial square grid side length.")
    parser.add_argument("--margin-cells", type=int, default=MARGIN_CELLS, help="Grow the grid when the ant is within this margin.")
    parser.add_argument("--chunk-steps", type=parse_int, default=CHUNK_STEPS, help="Maximum steps per compiled advance call.")
    args = parser.parse_args()

    TARGET_STEPS = args.target_steps
    OUTPUT_DIR = args.output_dir
    FPS = args.fps
    INITIAL_SPEED_STEPS_PER_SECOND = args.initial_speed
    INITIAL_SPEED_DURATION_SECONDS = args.initial_speed_duration
    DOUBLING_INTERVAL_SECONDS = args.doubling_interval
    BACKUP_POWER_STEP = args.backup_power_step
    AUTO_SCALE_SNAPSHOTS = not args.no_auto_scale
    SCALE_SPAN_HIGH = args.scale_span_high
    SCALE_SPAN_LOW = args.scale_span_low
    INITIAL_GRID_SIZE = args.initial_grid_size
    MARGIN_CELLS = args.margin_cells
    CHUNK_STEPS = args.chunk_steps

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = frame_schedule()
    backups = backup_steps()
    rows = read_metadata(output_dir)
    grid, state, offset, current_step, next_frame_index, next_backup_index = load_checkpoint(output_dir)

    warm = np.zeros((20, 20), np.uint8)
    advance_until_margin_v2(warm, 10, 2, 10, 10, 10, 0, 0, 10, 10, 10, 10, 0, 0)

    rows = [row for row in rows if int(row["frame"]) < next_frame_index]
    start = perf_counter()
    print(f"frames: {len(schedule)}")
    print(f"target steps: {TARGET_STEPS:,}")
    print(f"resume step: {current_step:,}, next frame: {next_frame_index}")
    print(f"backup milestones: {len(backups)}")

    for index in range(next_frame_index, len(schedule)):
        schedule_row = schedule[index]
        target_step = int(schedule_row["step"])
        frame_start = perf_counter()
        previous_step = current_step
        frame_steps = target_step - previous_step
        grid, state, offset, _ = advance_dynamic(grid, state, offset, frame_steps)
        current_step = target_step
        row = save_frame(output_dir, schedule_row, state, offset, grid)
        rows.append(row)
        frame_elapsed = perf_counter() - frame_start
        frame_speed_msteps = frame_steps / frame_elapsed / 1_000_000 if frame_elapsed > 0 else 0.0
        print(
            f"frame {index + 1}/{len(schedule)}  "
            f"step={current_step:,}  "
            f"delta={frame_steps:,}  "
            f"black={row['black_cells']:,}  "
            f"time={frame_elapsed:.3f}s  "
            f"speed={frame_speed_msteps:.3f} Mstep/s"
        )
        if index % 30 == 0 or index == len(schedule) - 1:
            write_metadata(output_dir, rows)

        while next_backup_index < len(backups) and current_step >= backups[next_backup_index]:
            save_checkpoint(output_dir, current_step, index + 1, next_backup_index + 1, state, offset, grid)
            next_backup_index += 1

    write_metadata(output_dir, rows)
    save_checkpoint(output_dir, current_step, len(schedule), next_backup_index, state, offset, grid)
    print(f"done in {perf_counter() - start:.3f}s")
    print(f"metadata: {output_dir / METADATA_FILE}")
    print(f"checkpoint: {output_dir / CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
