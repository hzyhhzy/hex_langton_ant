import argparse
import csv
from math import cos, floor, log10, pi, sin, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont


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


def advance_dynamic(grid, state, offset, steps, initial_size, margin, chunk_steps):
    done = 0
    expansions = 0
    while done < steps:
        delta = min(chunk_steps, steps - done)
        advanced = advance_until_margin(grid, delta, margin, offset, *state)
        completed = advanced[0]
        state = advanced[1:]
        done += completed
        if completed < delta:
            pad = max(initial_size // 2, margin * 4, grid.shape[0] // 2)
            grid, state, offset = expand_grid(grid, state, offset, pad)
            expansions += 1
    return grid, state, offset, expansions


def axial_center(q, r):
    return sqrt(3) * (q + r / 2), 1.5 * r


def hex_points(cx, cy, side):
    return [
        (cx + side * cos(pi / 6 + k * pi / 3), cy + side * sin(pi / 6 + k * pi / 3))
        for k in range(6)
    ]


def nice_scale_cells(pixel_scale, target_pixels=180):
    raw_cells = max(1.0, target_pixels / (sqrt(3) * pixel_scale))
    exponent = floor(log10(raw_cells))
    base = 10**exponent
    for multiplier in (1, 2, 5, 10):
        value = multiplier * base
        if value >= raw_cells:
            return int(value)
    return int(10 * base)


def draw_scale_bar(draw, width, height, pixel_scale, font):
    cells = nice_scale_cells(pixel_scale)
    bar_pixels = sqrt(3) * cells * pixel_scale
    x0 = 48
    y0 = height - 58
    x1 = x0 + bar_pixels
    label = f"{cells:g} hex cells"

    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    bg_right = max(x1, x0 + text_w) + 18
    draw.rectangle((x0 - 12, y0 - 35, bg_right, y0 + 18), fill="white", outline=(210, 210, 210))
    draw.line((x0, y0, x1, y0), fill=(20, 20, 20), width=3)
    draw.line((x0, y0 - 8, x0, y0 + 8), fill=(20, 20, 20), width=3)
    draw.line((x1, y0 - 8, x1, y0 + 8), fill=(20, 20, 20), width=3)
    draw.text((x0, y0 - text_h - 12), label, fill=(20, 20, 20), font=font)


def render(grid, state, offset, step, power, output, width, height):
    q, r, direction, black, minq, maxq, minr, maxr, mins, maxs = state
    aq = q - offset
    ar = r - offset
    q0 = max(0, minq - 2)
    q1 = min(grid.shape[0], maxq + 3)
    r0 = max(0, minr - 2)
    r1 = min(grid.shape[1], maxr + 3)
    coords = np.argwhere(grid[q0:q1, r0:r1] != 0)
    cells = [(int(i + q0 - offset), int(j + r0 - offset)) for i, j in coords]

    points = cells + [(aq, ar)]
    xs = []
    ys = []
    for cell_q, cell_r in points:
        x, y = axial_center(cell_q, cell_r)
        xs.append(x)
        ys.append(y)

    margin_x = width * 0.05
    margin_y = height * 0.05
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span_x = max(maxx - minx, 1)
    span_y = max(maxy - miny, 1)
    scale = min(
        (width - 2 * margin_x) / (span_x + 2),
        (height - 2 * margin_y) / (span_y + 2),
    )
    side = max(0.45, scale * 0.54)

    def to_pixel(cell_q, cell_r):
        x, y = axial_center(cell_q, cell_r)
        return margin_x + (x - minx + 1) * scale, margin_y + (y - miny + 1) * scale

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for cell in cells:
        cx, cy = to_pixel(*cell)
        draw.polygon(hex_points(cx, cy, side), fill=(18, 18, 18))

    cx, cy = to_pixel(aq, ar)
    draw.polygon(hex_points(cx, cy, side * 1.35), fill=(214, 40, 40))

    title = f"2^{power} steps = {step:,}    black: {black:,}    ant: ({aq}, {ar}, {-aq - ar})    dir: {direction}"
    draw.rectangle((24, 20, width - 24, 48), fill="white")
    draw.text((34, 28), title, fill=(20, 20, 20), font=font)
    draw_scale_bar(draw, width, height, scale, font)
    image.save(output)

    return {
        "power": power,
        "steps": step,
        "black_cells": black,
        "q": aq,
        "r": ar,
        "s": -aq - ar,
        "direction": direction,
        "bbox_q_min": minq - offset,
        "bbox_q_max": maxq - offset,
        "bbox_r_min": minr - offset,
        "bbox_r_max": maxr - offset,
        "bbox_s_min": mins,
        "bbox_s_max": maxs,
        "image": output.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Render hex Langton ant snapshots at 2^n steps.")
    parser.add_argument("--max-power", type=int, default=30)
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=100_000_000)
    parser.add_argument("--output-dir", default="hex_langton_powers_of_two")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1050)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offset = args.size // 2
    grid = np.zeros((args.size, args.size), np.uint8)
    warm = np.zeros((20, 20), np.uint8)
    advance_until_margin(warm, 10, 2, 10, 10, 10, 0, 0, 10, 10, 10, 10, 0, 0)

    state = (offset, offset, 0, 0, offset, offset, offset, offset, 0, 0)
    current_step = 0
    rows = []
    start = perf_counter()
    total_expansions = 0

    for power in range(args.max_power + 1):
        target_step = 1 << power
        delta = target_step - current_step
        grid, state, offset, expansions = advance_dynamic(
            grid,
            state,
            offset,
            delta,
            args.size,
            args.margin,
            args.chunk_steps,
        )
        total_expansions += expansions
        current_step = target_step
        name = f"hex_langton_2^{power:02d}_{target_step}.png"
        output = output_dir / name
        row = render(grid, state, offset, target_step, power, output, args.width, args.height)
        rows.append(row)
        print(f"saved {output}  grid={grid.shape[0]}x{grid.shape[1]}  expansions={total_expansions}")

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"done in {perf_counter() - start:.3f}s")
    print(f"metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
