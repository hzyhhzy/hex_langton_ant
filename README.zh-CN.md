# 六边形网格上的兰顿蚂蚁

[English](README.md)

建议仓库名：`hex_langton_ant`。

这个项目包含一组用于仿真和渲染六边形网格兰顿蚂蚁的 Python 脚本。规则是：

- 白色格子：左转 60 度，并翻转为黑色
- 黑色格子：右转 60 度，并翻转为白色

程序使用轴坐标表示六边形网格，用 NumPy 保存稠密网格，并用 Numba 编译核心更新循环。长时间仿真会保存可续跑的 `.npz` 快照和 `metadata.csv`，生成的数据、图片和视频默认不加入 git。

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
- `compute_to_2pow40_and_render.sh`：计算到 `2^40` 步，并生成一个 stride=16 的视频。
- `continue_2p40_to_2p42.sh`：在同一个关键帧目录中继续计算到 `2^42` 步。
- `render_4T_stride_videos.sh`：根据已经算好的 `2^42` 数据，生成 stride 为 `64, 32, 16, 8, 4, 2, 1` 的多版本视频。

## 快速示例

生成单张图片：

```bash
python simulate_hex_langton.py --steps 1000000 --output results/hex_langton_1M.png
```

生成 `2^n` 步的图片序列：

```bash
python generate_power_images.py --max-power 30 --output-dir results/hex_langton_powers
```

计算到 `2^30` 步的视频关键帧数据：

```bash
python compute_hex_langton_timed_frames.py \
  --target-steps 1073741824 \
  --output-dir results/hex_langton_smooth_frames_1G \
  --doubling-interval 16
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

计算并渲染较大的运行：

```bash
bash compute_to_2pow40_and_render.sh
bash continue_2p40_to_2p42.sh
bash render_4T_stride_videos.sh
```

`render_4T_stride_videos.sh` 可以用环境变量临时覆盖参数，例如只渲染 stride=64 和 stride=32：

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

## 视频时间轴

当前视频时间轴设置为：

- 30 fps
- 初始速度 4 step/s
- 初始速度保持 16 秒
- 之后每 16 秒速度翻倍

渲染脚本支持 `--stride` 参数。比如 `--stride 16` 表示每 16 个已保存关键帧渲染一帧，用来快速预览；`--stride 1` 会渲染全部关键帧，质量最高但最慢。

## Git 说明

`.gitignore` 已经忽略以下生成物：

- 图片和视频文件
- `.npz` 仿真数据
- `.csv` 元数据
- 压缩包
- `results/`
- Python 缓存文件

因此可以直接把这个目录作为 GitHub 仓库使用，不会误提交大体积仿真结果。
