import argparse
import json
from math import ceil, floor, sqrt
from pathlib import Path
from time import perf_counter
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import cv2
except ImportError:  # pragma: no cover - optional display polish
    cv2 = None


DEFAULT_CHECKPOINT = "results/hex_langton_smooth_frames_2p40/checkpoint.npz"
SQRT3 = sqrt(3)
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900
BACKGROUND = (255, 255, 255)
CELL_COLOR = (18, 18, 18)
ANT_COLOR = (214, 40, 40)
CHUNK_ROWS = 1_000_000
ZOOM_STEP = 1.25
LOD_TARGET_SPACING_PX = 1.0
MAX_LOD_FACTOR = 65536
MIN_LOD_CELLS = 20_000
SCALE_BAR_TARGET_PX = 180
SCALE_BAR_X = 36
SCALE_BAR_BOTTOM = 38


def axial_center(q, r):
    return SQRT3 * (q + r / 2), 1.5 * r


def direction_vector(direction):
    vectors = [
        (SQRT3, 0.0),
        (SQRT3 / 2, -1.5),
        (-SQRT3 / 2, -1.5),
        (-SQRT3, 0.0),
        (-SQRT3 / 2, 1.5),
        (SQRT3 / 2, 1.5),
    ]
    dx, dy = vectors[direction]
    length = (dx * dx + dy * dy) ** 0.5
    return dx / length, dy / length


def load_font(size):
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def format_value(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value:.3e}"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def nice_scale_bar_units(pixel_per_unit, target_pixels=SCALE_BAR_TARGET_PX):
    raw_units = max(1.0, target_pixels / pixel_per_unit)
    exponent = floor(np.log10(raw_units))
    base = 10**exponent
    for multiplier in (1, 2, 5, 10):
        value = multiplier * base
        if value >= raw_units:
            return int(value)
    return int(10 * base)


def compress_cells_by_two(cells):
    started = perf_counter()
    q = np.floor_divide(cells[:, 0], 2).astype(np.int64, copy=False)
    r = np.floor_divide(cells[:, 1], 2).astype(np.int64, copy=False)
    q_min = int(q.min())
    r_min = int(r.min())
    r_range = int(r.max() - r_min + 1)
    keys = (q - q_min) * r_range + (r - r_min)
    keys = np.unique(keys)
    compressed = np.empty((len(keys), 2), dtype=np.int32)
    compressed[:, 0] = keys // r_range + q_min
    compressed[:, 1] = keys % r_range + r_min
    print(f"  compressed {len(cells):,} -> {len(compressed):,} in {perf_counter() - started:.3f}s")
    return compressed


class CheckpointViewer:
    def __init__(self, root, checkpoint_path, width, height, show_overlay=True):
        self.root = root
        self.checkpoint_path = Path(checkpoint_path)
        self.width = width
        self.height = height
        self.show_overlay = show_overlay
        self.photo = None
        self.pending_render = None
        self.drag_start = None
        self.rendering = False
        self.font = load_font(18)

        self.current_step, self.cells, self.meta = self.load_checkpoint(self.checkpoint_path)
        self.lod_levels = self.build_lod_levels(self.cells)

        self.center_x, self.center_y, self.scale = self.initial_view()

        self.canvas = tk.Canvas(root, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        root.bind("f", lambda _event: self.fit_view())
        root.bind("F", lambda _event: self.fit_view())
        root.bind("r", lambda _event: self.reload_checkpoint())
        root.bind("R", lambda _event: self.reload_checkpoint())
        root.bind("+", lambda _event: self.zoom_at(self.width / 2, self.height / 2, ZOOM_STEP))
        root.bind("=", lambda _event: self.zoom_at(self.width / 2, self.height / 2, ZOOM_STEP))
        root.bind("-", lambda _event: self.zoom_at(self.width / 2, self.height / 2, 1 / ZOOM_STEP))
        root.bind("<Escape>", lambda _event: root.destroy())

        root.title(f"Hex Langton checkpoint viewer - step {self.current_step:,}")
        self.schedule_render()

    @staticmethod
    def load_checkpoint(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        started = perf_counter()
        data = np.load(path, allow_pickle=False)
        cells = data["cells"].astype(np.int32, copy=False)
        current_step = int(data["current_step"])
        meta = json.loads(str(data["meta"]))
        data.close()
        print(f"loaded {path}  cells={len(cells):,}  step={current_step:,}  time={perf_counter() - started:.3f}s")
        return current_step, cells, meta

    def initial_view(self):
        corners = [
            (self.meta["bbox_q_min"], self.meta["bbox_r_min"]),
            (self.meta["bbox_q_min"], self.meta["bbox_r_max"]),
            (self.meta["bbox_q_max"], self.meta["bbox_r_min"]),
            (self.meta["bbox_q_max"], self.meta["bbox_r_max"]),
        ]
        xs = []
        ys = []
        for q, r in corners:
            x, y = axial_center(q, r)
            xs.append(x)
            ys.append(y)
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        scale = min(self.width / max(span_x * 1.08, 1), self.height / max(span_y * 1.08, 1))
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, scale

    def fit_view(self):
        self.center_x, self.center_y, self.scale = self.initial_view()
        self.schedule_render()

    def reload_checkpoint(self):
        self.current_step, self.cells, self.meta = self.load_checkpoint(self.checkpoint_path)
        self.lod_levels = self.build_lod_levels(self.cells)
        self.fit_view()

    def build_lod_levels(self, cells):
        print("building LOD levels")
        levels = [{"factor": 1, "cells": cells, "q_values": cells[:, 0]}]
        factor = 1
        current = cells
        while factor < MAX_LOD_FACTOR and len(current) > MIN_LOD_CELLS:
            factor *= 2
            print(f"LOD x{factor}:")
            current = compress_cells_by_two(current)
            levels.append({"factor": factor, "cells": current, "q_values": current[:, 0]})
            if len(current) == len(levels[-2]["cells"]):
                break
        print("LOD levels:", ", ".join(f"x{level['factor']}={len(level['cells']):,}" for level in levels))
        return levels

    def select_lod_level(self):
        selected = self.lod_levels[0]
        for level in self.lod_levels:
            spacing_px = level["factor"] * SQRT3 * self.scale
            if spacing_px <= LOD_TARGET_SPACING_PX:
                selected = level
            else:
                break
        return selected

    def on_resize(self, event):
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self.schedule_render()

    def on_mouse_wheel(self, event):
        if hasattr(event, "delta") and event.delta:
            factor = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        else:
            factor = ZOOM_STEP if event.num == 4 else 1 / ZOOM_STEP
        self.zoom_at(event.x, event.y, factor)

    def zoom_at(self, screen_x, screen_y, factor):
        world_x, world_y = self.screen_to_world(screen_x, screen_y)
        max_scale = self.width / (3 * SQRT3)
        self.scale = min(max(self.scale * factor, 1e-9), max_scale)
        self.center_x = world_x - (screen_x - self.width / 2) / self.scale
        self.center_y = world_y - (screen_y - self.height / 2) / self.scale
        self.schedule_render()

    def on_drag_start(self, event):
        self.drag_start = (event.x, event.y)

    def on_drag_move(self, event):
        if self.drag_start is None:
            return
        last_x, last_y = self.drag_start
        dx = event.x - last_x
        dy = event.y - last_y
        self.center_x -= dx / self.scale
        self.center_y -= dy / self.scale
        self.drag_start = (event.x, event.y)
        self.schedule_render()

    def on_drag_end(self, _event):
        self.drag_start = None

    def screen_to_world(self, screen_x, screen_y):
        return (
            self.center_x + (screen_x - self.width / 2) / self.scale,
            self.center_y + (screen_y - self.height / 2) / self.scale,
        )

    def world_to_screen(self, world_x, world_y):
        return (
            self.width / 2 + (world_x - self.center_x) * self.scale,
            self.height / 2 + (world_y - self.center_y) * self.scale,
        )

    def visible_axial_bounds(self):
        min_x, min_y = self.screen_to_world(0, 0)
        max_x, max_y = self.screen_to_world(self.width, self.height)
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        r_min = floor(min_y / 1.5) - 2
        r_max = ceil(max_y / 1.5) + 2
        q_candidates = [
            min_x / SQRT3 - r_min / 2,
            min_x / SQRT3 - r_max / 2,
            max_x / SQRT3 - r_min / 2,
            max_x / SQRT3 - r_max / 2,
        ]
        return floor(min(q_candidates)) - 2, ceil(max(q_candidates)) + 2, r_min, r_max

    def schedule_render(self):
        if self.pending_render is not None:
            self.root.after_cancel(self.pending_render)
        self.pending_render = self.root.after(25, self.render)

    def render(self):
        self.pending_render = None
        if self.rendering:
            self.schedule_render()
            return
        self.rendering = True
        started = perf_counter()
        image, visible_count = self.render_image()
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        elapsed = perf_counter() - started
        self.rendering = False
        print(
            f"render {self.width}x{self.height}  visible={visible_count:,}  "
            f"scale={self.scale:.6g} px/unit  time={elapsed:.3f}s"
        )

    def render_image(self):
        pixels = np.full((self.height, self.width, 3), BACKGROUND, dtype=np.uint8)
        level = self.select_lod_level()
        factor = level["factor"]
        cells = level["cells"]
        q_values = level["q_values"]
        q_min, q_max, r_min, r_max = self.visible_axial_bounds()
        q_min_lod = floor(q_min / factor) - 2
        q_max_lod = ceil(q_max / factor) + 2
        r_min_lod = floor(r_min / factor) - 2
        r_max_lod = ceil(r_max / factor) + 2
        lo = int(np.searchsorted(q_values, q_min_lod, side="left"))
        hi = int(np.searchsorted(q_values, q_max_lod, side="right"))

        mask_image = np.zeros((self.height, self.width), dtype=np.uint8)
        visible_count = 0
        for start in range(lo, hi, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, hi)
            chunk = cells[start:end]
            r_values = chunk[:, 1]
            r_mask = (r_values >= r_min_lod) & (r_values <= r_max_lod)
            if not np.any(r_mask):
                continue

            block_offset = (factor - 1) / 2
            q = (chunk[r_mask, 0].astype(np.float64) * factor) + block_offset
            r = (r_values[r_mask].astype(np.float64) * factor) + block_offset
            x = SQRT3 * (q + r / 2)
            y = 1.5 * r
            px = np.rint(self.width / 2 + (x - self.center_x) * self.scale).astype(np.int32)
            py = np.rint(self.height / 2 + (y - self.center_y) * self.scale).astype(np.int32)
            screen_mask = (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
            if not np.any(screen_mask):
                continue
            px = px[screen_mask]
            py = py[screen_mask]
            mask_image[py, px] = 255
            visible_count += len(px)

        unit_px = SQRT3 * self.scale
        if cv2 is not None:
            diameter = max(2.0, 0.8 * unit_px)
            radius = int(round(diameter / 2))
            if radius >= 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                mask_image = cv2.dilate(mask_image, kernel)
        pixels[mask_image > 0] = CELL_COLOR

        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        self.draw_ant(draw)
        self.draw_scale_bar(draw)
        if self.show_overlay:
            self.draw_overlay(draw, visible_count, level)
        return image, visible_count

    def draw_scale_bar(self, draw):
        pixel_per_unit = SQRT3 * self.scale
        units = nice_scale_bar_units(pixel_per_unit)
        bar_pixels = units * pixel_per_unit
        x0 = SCALE_BAR_X
        y0 = self.height - SCALE_BAR_BOTTOM
        x1 = x0 + bar_pixels
        label = format_value(units)
        text_box = draw.textbbox((0, 0), label, font=self.font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        box_right = max(x1, x0 + text_w) + 14
        box_top = y0 - text_h - 24
        box_bottom = y0 + 18
        draw.rectangle((x0 - 12, box_top, box_right, box_bottom), fill=(255, 255, 255), outline=(220, 220, 220))
        draw.line((x0, y0, x1, y0), fill=(20, 20, 20), width=2)
        draw.line((x0, y0 - 8, x0, y0 + 8), fill=(20, 20, 20), width=2)
        draw.line((x1, y0 - 8, x1, y0 + 8), fill=(20, 20, 20), width=2)
        draw.text((x0, y0 - text_h - 12), label, fill=(20, 20, 20), font=self.font)


    def draw_ant(self, draw):
        q = self.meta["q"]
        r = self.meta["r"]
        x, y = axial_center(q, r)
        px, py = self.world_to_screen(x, y)
        if not (-40 <= px <= self.width + 40 and -40 <= py <= self.height + 40):
            return
        radius = max(4, SQRT3 * self.scale * 0.7)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=ANT_COLOR)
        dx, dy = direction_vector(self.meta["direction"])
        length = radius * 1.8
        tip = (px + dx * length, py + dy * length)
        left = (px - dx * radius * 0.4 - dy * radius * 0.65, py - dy * radius * 0.4 + dx * radius * 0.65)
        right = (px - dx * radius * 0.4 + dy * radius * 0.65, py - dy * radius * 0.4 - dx * radius * 0.65)
        draw.polygon((tip, left, right), fill=ANT_COLOR)

    def draw_overlay(self, draw, visible_count, level):
        lines = [
            f"Step: {format_value(self.current_step)}",
            f"Black: {format_value(self.meta['black_cells'])}",
            f"Visible: {format_value(visible_count)}",
            f"LOD: x{level['factor']} ({len(level['cells']):,})",
            f"Zoom: {SQRT3 * self.scale:.4g} px/cell",
            "Wheel: zoom   Drag: pan   F: fit   R: reload   Esc: quit",
        ]
        line_h = 24
        width = max(draw.textbbox((0, 0), line, font=self.font)[2] for line in lines) + 22
        height = line_h * len(lines) + 18
        draw.rectangle((12, 12, 12 + width, 12 + height), fill=(255, 255, 255), outline=(220, 220, 220))
        for i, line in enumerate(lines):
            draw.text((24, 22 + i * line_h), line, fill=(20, 20, 20), font=self.font)


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for a hex Langton ant checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="checkpoint.npz path.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Initial window width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Initial window height.")
    parser.add_argument("--no-overlay", action="store_true", help="Hide the status overlay.")
    args = parser.parse_args()

    root = tk.Tk()
    CheckpointViewer(root, args.checkpoint, args.width, args.height, show_overlay=not args.no_overlay)
    root.mainloop()


if __name__ == "__main__":
    main()
