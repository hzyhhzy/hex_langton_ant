# 六边形网格上的兰顿蚂蚁

[English](README.md)

这个项目包含一组用于仿真和渲染六边形网格兰顿蚂蚁的 Python 脚本。规则是：

- 白色格子：左转 60 度，并翻转为黑色
- 黑色格子：右转 60 度，并翻转为白色

程序使用轴坐标表示六边形网格，用 NumPy 保存网格和坐标数据，并用 Numba 编译核心更新循环。长时间仿真可以通过 `checkpoint.npz` 续跑，关键帧信息记录在 `metadata.csv` 中。

## 依赖

建议使用 Python 3.10 或更新版本。

```bash
pip install -r requirements.txt
```

## 脚本说明

- `simulate_hex_langton.py`：仿真到指定步数，并渲染一张 PNG 图片。
- `generate_power_images.py`：生成 `2^n` 步时的静态图片。
- `compute_hex_langton_timed_frames.py`：按视频时间轴计算关键帧数据，支持中断后续跑。
- `render_hex_langton_timed_video.py`：根据关键帧数据渲染视频。
- `checkpoint_viewer.py`：交互式查看 checkpoint，支持鼠标滚轮缩放。
- `compute_to_2pow40_and_render.sh`：计算到 `2^40` 步，并生成 stride=16 的视频。
- `continue_2p40_to_2p42.sh`：在同一个关键帧目录中继续计算到 `2^42` 步。
- `continue_2p42_to_2p44.sh`：在同一个关键帧目录中继续计算到 `2^44` 步。
- `render_4T_stride_videos.sh`：根据已保存关键帧生成多个 stride 版本的视频。

## 快速示例

生成单张图片：

```bash
python simulate_hex_langton.py --steps 1000000 --output results/hex_langton_1M.png
```

生成 `2^n` 步的图片序列：

```bash
python generate_power_images.py --max-power 30 --output-dir results/hex_langton_powers
```

计算视频关键帧数据：

```bash
python compute_hex_langton_timed_frames.py \
  --target-steps 1073741824 \
  --output-dir results/hex_langton_smooth_frames_1G \
  --doubling-interval 16
```

每完成一帧会打印这一帧的总耗时和平均速度，例如：

```text
frame 17509/18468  step=...  delta=...  black=...  time=12.345s  speed=123.456 Mstep/s
```

根据关键帧数据渲染视频：

```bash
python render_hex_langton_timed_video.py \
  --frame-dir results/hex_langton_smooth_frames_1G \
  --output results/hex_langton_smooth_1G_1920x1200_k16.mp4 \
  --stride 16 \
  --width 1920 \
  --height 1200
```

渲染带浅灰直角坐标系的视频：

```bash
python render_hex_langton_timed_video.py \
  --frame-dir results/hex_langton_smooth_frames_1G \
  --output results/hex_langton_grid.mp4 \
  --stride 16 \
  --show-cartesian-grid
```

坐标网格画在图案下方，网格间距和 scale bar 同步，横向相机中心锁定在世界坐标 `x=0`。

## Checkpoint Viewer

打开交互式查看器：

```bash
python checkpoint_viewer.py
```

操作：

- 鼠标滚轮：以鼠标指向的位置为中心缩放
- 左键拖拽：平移
- `F`：回到全图视图
- `R`：重新读取 checkpoint
- `+` / `-`：以窗口中心缩放
- `Esc`：退出

viewer 会从 checkpoint 预先生成压缩 LOD 坐标层，例如 `x2`、`x4`、`x8`，缩放时自动切换。scale bar 的单位定义为最近两个六边形中心的距离等于 `1`。

## 较大规模运行

```bash
bash compute_to_2pow40_and_render.sh
bash continue_2p40_to_2p42.sh
bash continue_2p42_to_2p44.sh
bash render_4T_stride_videos.sh
```

`render_4T_stride_videos.sh` 可以用环境变量临时覆盖参数：

```bash
STRIDES="64 32" WIDTH=1280 HEIGHT=720 bash render_4T_stride_videos.sh
```

## 长时间仿真说明

`compute_hex_langton_timed_frames.py` 会在输出目录中写入：

- `metadata.csv`
- `frame_*.npz`
- `checkpoint.npz`

如果运行中断，用同样的命令重新运行即可。程序会从 `checkpoint.npz` 恢复，不会从头开始。

当图案跨度变大时，关键帧会默认自动降采样保存，以减少磁盘占用。每一帧的坐标缩放比例会写入该帧的元数据，渲染脚本会自动读取并还原显示尺度。

## Git 说明

`.gitignore` 已经忽略生成的图片、视频、`.npz` 仿真数据、`.csv` 元数据、压缩包、`results/` 和 Python 缓存文件。
