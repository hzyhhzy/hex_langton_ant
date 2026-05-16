import argparse
from math import cos, pi, sin, sqrt
from time import perf_counter

import numpy as np
from numba import njit
from PIL import Image, ImageDraw


@njit
def advance_until_margin(
    grid,
    steps,
    margin,
    offset,
    q,
    r,
    d,
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
            d = (d + 1) % 6
            grid[q, r] = 0
            black -= 1
        else:
            d = (d + 5) % 6
            grid[q, r] = 1
            black += 1

        if d == 0:
            q += 1
        elif d == 1:
            q += 1
            r -= 1
        elif d == 2:
            r -= 1
        elif d == 3:
            q -= 1
        elif d == 4:
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

    return completed, q, r, d, black, minq, maxq, minr, maxr, mins, maxs


def expand_grid(grid, state, offset, pad):
    q, r, d, black, minq, maxq, minr, maxr, mins, maxs = state
    expanded = np.zeros((grid.shape[0] + 2 * pad, grid.shape[1] + 2 * pad), np.uint8)
    expanded[pad : pad + grid.shape[0], pad : pad + grid.shape[1]] = grid
    shifted_state = (
        q + pad,
        r + pad,
        d,
        black,
        minq + pad,
        maxq + pad,
        minr + pad,
        maxr + pad,
        mins,
        maxs,
    )
    return expanded, shifted_state, offset + pad


def run_simulation(steps, initial_size, margin=256, chunk_steps=100_000_000):
    offset = initial_size // 2
    grid = np.zeros((initial_size, initial_size), np.uint8)
    state = (offset, offset, 0, 0, offset, offset, offset, offset, 0, 0)

    warm = np.zeros((20, 20), np.uint8)
    advance_until_margin(warm, 10, 2, 10, 10, 10, 0, 0, 10, 10, 10, 10, 0, 0)

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

    q, r, d, black, minq, maxq, minr, maxr, mins, maxs = state
    aq = q - offset
    ar = r - offset
    return (
        grid,
        offset,
        expansions,
        (
            aq,
            ar,
            -aq - ar,
            d,
            black,
            minq - offset,
            maxq - offset,
            minr - offset,
            maxr - offset,
            mins,
            maxs,
        ),
    )


def simulate(grid, steps, offset):
    state = (offset, offset, 0, 0, offset, offset, offset, offset, 0, 0)
    advanced = advance_until_margin(
        grid,
        steps,
        0,
        offset,
        *state,
    )
    q, r, d, black, minq, maxq, minr, maxr, mins, maxs = advanced[1:]
    aq = q - offset
    ar = r - offset
    return (
        aq,
        ar,
        -aq - ar,
        d,
        black,
        minq - offset,
        maxq - offset,
        minr - offset,
        maxr - offset,
        mins,
        maxs,
    )


def axial_center(q, r):
    return sqrt(3) * (q + r / 2), 1.5 * r


def hex_points(cx, cy, side):
    return [
        (cx + side * cos(pi / 6 + k * pi / 3), cy + side * sin(pi / 6 + k * pi / 3))
        for k in range(6)
    ]


def render(grid, result, offset, output, width=1800, height=1050):
    q, r, _, _, _, minq, maxq, minr, maxr, _, _ = result
    q0 = max(0, offset + minq - 2)
    q1 = min(grid.shape[0], offset + maxq + 3)
    r0 = max(0, offset + minr - 2)
    r1 = min(grid.shape[1], offset + maxr + 3)
    coords = np.argwhere(grid[q0:q1, r0:r1] != 0)
    cells = [(int(i + q0 - offset), int(j + r0 - offset)) for i, j in coords]

    xs = []
    ys = []
    for cell_q, cell_r in cells + [(q, r)]:
        x, y = axial_center(cell_q, cell_r)
        xs.append(x)
        ys.append(y)

    margin_x = width * 0.05
    margin_y = height * 0.05
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = min(
        (width - 2 * margin_x) / (maxx - minx + 2),
        (height - 2 * margin_y) / (maxy - miny + 2),
    )
    side = max(0.45, scale * 0.54)

    def to_pixel(cell_q, cell_r):
        x, y = axial_center(cell_q, cell_r)
        return margin_x + (x - minx + 1) * scale, margin_y + (y - miny + 1) * scale

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for cell in cells:
        cx, cy = to_pixel(*cell)
        draw.polygon(hex_points(cx, cy, side), fill=(18, 18, 18))

    cx, cy = to_pixel(q, r)
    draw.polygon(hex_points(cx, cy, side * 1.3), fill=(214, 40, 40))
    image.save(output)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Langton's ant on a hexagonal grid: white left 60, black right 60."
    )
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=100_000_000)
    parser.add_argument("--output", default="hex_langton.png")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    start = perf_counter()
    grid, offset, expansions, result = run_simulation(
        args.steps,
        args.size,
        margin=args.margin,
        chunk_steps=args.chunk_steps,
    )
    seconds = perf_counter() - start

    print(f"steps: {args.steps}")
    print(f"seconds: {seconds:.3f}")
    print(f"grid size: {grid.shape[0]} x {grid.shape[1]}")
    print(f"expansions: {expansions}")
    print(f"final axial position: {result[0:3]}")
    print(f"direction: {result[3]}")
    print(f"black cells: {result[4]}")
    print(f"bbox q: {result[5]}..{result[6]}")
    print(f"bbox r: {result[7]}..{result[8]}")
    print(f"bbox s: {result[9]}..{result[10]}")

    if not args.no_render:
        render(grid, result, offset, args.output)
        print(f"saved image: {args.output}")


if __name__ == "__main__":
    main()
