import argparse
import csv
import json
from math import ceil, cos, floor, log10, pi, sin, sqrt
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Edit these settings.
FRAME_DIR = "results/hex_langton_smooth_frames_2p40"
OUTPUT = "results/hex_langton_smooth_2p40_1920x1200_k16.mp4"
FALLBACK_GIF_OUTPUT = "results/hex_langton_smooth_2p40_1920x1200_k16.gif"
FPS = 30
FRAME_STRIDE = 16
WIDTH = 1920
HEIGHT = 1200
DRAW_SCALE_BAR = True
MIN_VIEW_WIDTH_CELLS = 25
MIN_VIEW_HEIGHT_CELLS = 17
MARGIN_FRACTION = 0.05
HEX_RADIUS_TO_CENTER_SPACING = 0.45
LABEL_FONT_SIZE = 34
SCALE_BAR_FONT_SIZE = 56
VIEW_SMOOTHING_SECONDS = 8
HEX_RENDER_MIN_RADIUS_PX = 0.5
ARROW_MIN_RADIUS_PX = 6.0
CAMERA_SCREEN_OFFSET_X_FRACTION = 1 / 8
SCALE_LABEL_EXTRA_UP_PX = 18
CAMERA_ANCHOR_STEP = 1000
CAMERA_MIN_RADIUS = 15
FINAL_HOLD_FRAMES = 30
HUD_X = 48
HUD_Y = 44
SCALE_BAR_X = 76
SCALE_BAR_BOTTOM_MARGIN = 82


METADATA_FILE = "metadata.csv"


def axial_center(q, r):
    return sqrt(3) * (q + r / 2), 1.5 * r


def hex_points(cx, cy, side):
    return [
        (cx + side * cos(pi / 6 + k * pi / 3), cy + side * sin(pi / 6 + k * pi / 3))
        for k in range(6)
    ]


def direction_vector(direction):
    vectors = [
        (sqrt(3), 0.0),
        (sqrt(3) / 2, -1.5),
        (-sqrt(3) / 2, -1.5),
        (-sqrt(3), 0.0),
        (-sqrt(3) / 2, 1.5),
        (sqrt(3) / 2, 1.5),
    ]
    dx, dy = vectors[direction]
    length = (dx * dx + dy * dy) ** 0.5
    return dx / length, dy / length


def draw_arrow(draw, cx, cy, direction, length):
    dx, dy = direction_vector(direction)
    x1 = cx + dx * length
    y1 = cy + dy * length
    tail_x = cx - dx * length * 0.45
    tail_y = cy - dy * length * 0.45
    draw.line((tail_x, tail_y, x1, y1), fill="white", width=max(2, int(length * 0.18)))

    left_x = x1 - dx * length * 0.35 - dy * length * 0.18
    left_y = y1 - dy * length * 0.35 + dx * length * 0.18
    right_x = x1 - dx * length * 0.35 + dy * length * 0.18
    right_y = y1 - dy * length * 0.35 - dx * length * 0.18
    draw.polygon([(x1, y1), (left_x, left_y), (right_x, right_y)], fill="white")


def load_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def nice_scale_cells(pixel_scale, target_pixels=180):
    raw_cells = max(1.0, target_pixels / (sqrt(3) * pixel_scale))
    exponent = floor(log10(raw_cells))
    base = 10**exponent
    for multiplier in (1, 2, 5, 10):
        value = multiplier * base
        if value >= raw_cells:
            return int(value)
    return int(10 * base)


def format_value(value):
    value = float(value)
    if abs(value) > 1_000_000:
        return f"{value:.2e}"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}"


def draw_scale_bar(draw, width, height, pixel_scale, font):
    cells = nice_scale_cells(pixel_scale)
    bar_pixels = sqrt(3) * cells * pixel_scale
    x0 = SCALE_BAR_X
    y0 = height - SCALE_BAR_BOTTOM_MARGIN
    x1 = x0 + bar_pixels
    label = format_value(cells)
    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    bg_right = max(x1, x0 + text_w) + 18
    draw.rectangle((x0 - 12, y0 - 35, bg_right, y0 + 18), fill="white")
    draw.line((x0, y0, x1, y0), fill=(20, 20, 20), width=3)
    draw.line((x0, y0 - 8, x0, y0 + 8), fill=(20, 20, 20), width=3)
    draw.line((x1, y0 - 8, x1, y0 + 8), fill=(20, 20, 20), width=3)
    draw.text((x0, y0 - text_h - 12 - SCALE_LABEL_EXTRA_UP_PX), label, fill=(20, 20, 20), font=font)


class AnimationWriter:
    def __init__(self, output, fps, width, height):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.video = None
        self.gif = None
        self.actual_output = self.output

    def __enter__(self):
        if self.output.suffix.lower() in (".mp4", ".avi", ".mov"):
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if self.output.suffix.lower() == ".mp4" else "MJPG"))
            self.video = cv2.VideoWriter(str(self.output), fourcc, self.fps, (self.width, self.height))
            if self.video.isOpened():
                return self
            self.video.release()
            self.video = None
            self.actual_output = Path(FALLBACK_GIF_OUTPUT)
            self.actual_output.parent.mkdir(parents=True, exist_ok=True)

        self.gif = imageio.get_writer(self.actual_output, mode="I", duration=1 / self.fps)
        return self

    def append_data(self, frame):
        if self.video is not None:
            self.video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            self.gif.append_data(frame)

    def __exit__(self, exc_type, exc, tb):
        if self.video is not None:
            self.video.release()
        if self.gif is not None:
            self.gif.close()


def read_metadata(frame_dir):
    with (frame_dir / METADATA_FILE).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: int(row["frame"]))


def smoothed_views(rows):
    if not rows:
        return []
    margin_x = WIDTH * MARGIN_FRACTION
    margin_y = HEIGHT * MARGIN_FRACTION
    available_h = HEIGHT - 2 * margin_y
    window = max(1, int(round(FPS * VIEW_SMOOTHING_SECONDS)))

    step = np.array([int(row["step"]) for row in rows], dtype=np.float64)
    xmin = np.array([int(row["bbox_q_min"]) for row in rows], dtype=np.float64)
    xmax = np.array([int(row["bbox_q_max"]) for row in rows], dtype=np.float64)
    ymin = np.array([int(row["bbox_r_min"]) for row in rows], dtype=np.float64)
    ymax = np.array([int(row["bbox_r_max"]) for row in rows], dtype=np.float64)
    centers_q = (xmax + xmin) / 2
    centers_r = (ymax + ymin) / 2

    anchor_index = int(np.argmin(np.abs(step - CAMERA_ANCHOR_STEP)))
    centers_q_zeroed = centers_q - centers_q[anchor_index]
    centers_r_zeroed = centers_r - centers_r[anchor_index]

    kernel = np.ones(window, dtype=np.float64) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left

    def smooth(values):
        if len(values) < pad_left + 2:
            padded = np.pad(values, (pad_left, pad_right), mode="edge")
        else:
            left_slope = np.mean(np.diff(values[: min(len(values), pad_left + 1)]))
            right_slope = np.mean(np.diff(values[max(0, len(values) - pad_right - 1) :]))
            left = values[0] - left_slope * np.arange(pad_left, 0, -1)
            right = values[-1] + right_slope * np.arange(1, pad_right + 1)
            padded = np.concatenate([left, values, right])
        return np.convolve(padded, kernel, mode="valid")

    smooth_q = smooth(centers_q_zeroed) + centers_q[anchor_index]
    smooth_r = smooth(centers_r_zeroed) + centers_r[anchor_index]
    rmax = np.maximum.reduce(
        [
            xmax - smooth_q,
            smooth_q - xmin,
            ymax - smooth_r,
            smooth_r - ymin,
            np.ones_like(xmax),
        ]
    )
    log_radius = smooth(np.log(rmax))
    radius = np.exp(np.maximum(np.log(CAMERA_MIN_RADIUS), log_radius))

    views = []
    for i in range(len(rows)):
        center_x, center_y = axial_center(smooth_q[i], smooth_r[i])
        scale = available_h / (3 * radius[i])
        views.append(
            {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "scale": scale,
                "size_width": 2 * float(radius[i]),
                "size_height": 2 * float(radius[i]),
            }
        )
    return views


def load_frame(frame_dir, row):
    data = np.load(frame_dir / row["file"], allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["cells"], meta


def raster_cells_fast(image, cells, coord_scale, view, side):
    if len(cells) == 0:
        return
    q = cells[:, 0].astype(np.float64) * coord_scale
    r = cells[:, 1].astype(np.float64) * coord_scale
    x = np.sqrt(3) * (q + r / 2)
    y = 1.5 * r
    px = np.rint(WIDTH * (0.5 + CAMERA_SCREEN_OFFSET_X_FRACTION) + (x - view["center_x"]) * view["scale"]).astype(np.int32)
    py = np.rint(HEIGHT / 2 + (y - view["center_y"]) * view["scale"]).astype(np.int32)
    mask = (px >= 0) & (px < WIDTH) & (py >= 0) & (py < HEIGHT)
    px = px[mask]
    py = py[mask]
    if len(px) == 0:
        return

    pixels = np.array(image)
    effective_radius = side * max(1, coord_scale)
    counts = np.zeros((HEIGHT, WIDTH), np.uint16)
    np.add.at(counts, (py, px), 1)
    mask_image = (counts > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_image = cv2.dilate(mask_image, kernel)
    pixels[mask_image > 0] = (18, 18, 18)
    image.paste(Image.fromarray(pixels))


def render_frame(row, cells, meta, view, speed_multiplier=1):
    q = meta["q"]
    r = meta["r"]
    coord_scale = int(meta.get("coord_scale", 1))
    scale = view["scale"]
    side = sqrt(3) * scale * HEX_RADIUS_TO_CENTER_SPACING
    effective_side = side * max(1, coord_scale)

    def to_pixel(cell_q, cell_r):
        x, y = axial_center(cell_q, cell_r)
        return (
            WIDTH * (0.5 + CAMERA_SCREEN_OFFSET_X_FRACTION) + (x - view["center_x"]) * scale,
            HEIGHT / 2 + (y - view["center_y"]) * scale,
        )

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    label_font = load_font(LABEL_FONT_SIZE)
    scale_font = load_font(SCALE_BAR_FONT_SIZE)

    if effective_side >= HEX_RENDER_MIN_RADIUS_PX:
        for cell_q, cell_r in cells:
            cx, cy = to_pixel(int(cell_q) * coord_scale, int(cell_r) * coord_scale)
            draw.polygon(hex_points(cx, cy, effective_side), fill=(18, 18, 18))
    else:
        raster_cells_fast(image, cells, coord_scale, view, side)

    cx, cy = to_pixel(q, r)
    ant_side = side * 1.35
    draw.polygon(hex_points(cx, cy, ant_side), fill=(214, 40, 40))
    if ant_side >= ARROW_MIN_RADIUS_PX:
        draw_arrow(draw, cx, cy, meta["direction"], max(7, side * 1.05))

    lines = [
        f"Step: {format_value(int(row['step']))}",
        f"Speed: {format_value(float(row['speed_steps_per_second']) * speed_multiplier)} step/s",
        f"Black: {format_value(meta['black_cells'])}",
        f"Camera: {format_value(view['size_width'])} x {format_value(view['size_height'])}",
    ]
    line_height = LABEL_FONT_SIZE + 8
    box_w = max(draw.textbbox((0, 0), line, font=label_font)[2] for line in lines) + 28
    box_h = line_height * len(lines) + 22
    draw.rectangle((HUD_X, HUD_Y, HUD_X + box_w, HUD_Y + box_h), fill="white", outline=(220, 220, 220))
    for i, line in enumerate(lines):
        draw.text((HUD_X + 14, HUD_Y + 12 + i * line_height), line, fill=(20, 20, 20), font=label_font)

    if DRAW_SCALE_BAR:
        draw_scale_bar(draw, WIDTH, HEIGHT, scale, scale_font)
    return np.asarray(image)


def main():
    global FRAME_DIR, OUTPUT, FALLBACK_GIF_OUTPUT, FPS, FRAME_STRIDE, WIDTH, HEIGHT, FINAL_HOLD_FRAMES

    parser = argparse.ArgumentParser(description="Render timed hex Langton ant frames into a video.")
    parser.add_argument("--frame-dir", default=FRAME_DIR, help="Directory containing metadata.csv and frame npz files.")
    parser.add_argument("--output", default=OUTPUT, help="Output video path.")
    parser.add_argument("--fallback-gif-output", default=FALLBACK_GIF_OUTPUT, help="Fallback GIF path if video writing fails.")
    parser.add_argument("--fps", type=int, default=FPS, help="Output frames per second.")
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE, help="Render every Nth saved source frame.")
    parser.add_argument("--width", type=int, default=WIDTH, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Output height in pixels.")
    parser.add_argument("--final-hold-frames", type=int, default=FINAL_HOLD_FRAMES, help="Extra copies of the final frame.")
    args = parser.parse_args()

    FRAME_DIR = args.frame_dir
    OUTPUT = args.output
    FALLBACK_GIF_OUTPUT = args.fallback_gif_output
    FPS = args.fps
    FRAME_STRIDE = args.stride
    WIDTH = args.width
    HEIGHT = args.height
    FINAL_HOLD_FRAMES = args.final_hold_frames
    if FRAME_STRIDE < 1:
        raise ValueError("--stride must be >= 1")
    if FPS < 1:
        raise ValueError("--fps must be >= 1")

    frame_dir = Path(FRAME_DIR)
    rows = read_metadata(frame_dir)
    views = smoothed_views(rows)
    selected = list(range(0, len(rows), FRAME_STRIDE))
    if selected[-1] != len(rows) - 1:
        selected.append(len(rows) - 1)
    print(f"frames: {len(rows)} source, {len(selected)} rendered with stride {FRAME_STRIDE}")
    print(f"output: {OUTPUT}")

    with AnimationWriter(OUTPUT, FPS, WIDTH, HEIGHT) as writer:
        final_frame = None
        for out_index, index in enumerate(selected):
            row = rows[index]
            cells, meta = load_frame(frame_dir, row)
            final_frame = render_frame(row, cells, meta, views[index], FRAME_STRIDE)
            writer.append_data(final_frame)
            if out_index % 100 == 0 or out_index == len(selected) - 1:
                print(f"frame {out_index + 1}/{len(selected)}  source={index + 1}/{len(rows)}  step={int(row['step']):,}")
        if final_frame is not None and FINAL_HOLD_FRAMES > 0:
            for _ in range(FINAL_HOLD_FRAMES):
                writer.append_data(final_frame)
            print(f"held final frame for {FINAL_HOLD_FRAMES} frames")
        print(f"animation file: {writer.actual_output}")


if __name__ == "__main__":
    main()
