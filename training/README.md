# Training

Training consists of two stages: Stage 1 trains the Camera ControlNet, and Stage 2 trains the History ControlNet.

| Stage | Model | Purpose |
| --- | --- | --- |
| Stage 1 | `WanControlnet` | Control the generation of the current video segment using camera trajectories. |
| Stage 2 | `WanHistoryControlnet` | Use historical frames and their camera parameters to improve spatial consistency when revisiting scenes. |


## Data Format

Each video sample contains the following fields, where `T` is the number of video frames.

| Field | Type / Shape | Description |
| --- | --- | --- |
| `video_path` | `str` | Video path relative to `--video_root_dir`, such as `video_clips/scene_0001/clip_0001.mp4`. |
| `long_caption` | `str` | Text description of the video. |
| `camera_extrinsics` | NumPy array, `(T, 4, 4)`; `float32` recommended | Matrices that transform world coordinates to camera coordinates (w2c), aligned with the video frames. |
| `camera_intrinsics` | NumPy array, `(4,)`; `float32` recommended | Normalized intrinsic parameters `[fx, fy, cx, cy]`, shared across the entire video. |

Example directory structure:

```text
RealEstate10K/
├── video_clips/
│   ├── scene_0001/
│   │   └── clip_0001.mp4
│   └── scene_0002/
│       └── clip_0002.mp4
└── annotations/
    ├── train.npz
    └── val.npz
```

## Single GPU Training

Activate your Python environment and set the dataset and output paths in the launch scripts.

```bash
cd /path/to/Memorize-When-Needed
GPU_ID=0 bash training/train_single_gpu.sh
```

## Training on Multiple Nodes and GPUs

Launch training with [train_multigpu.sh](train_multigpu.sh). Set the number of nodes and the total number of GPUs in [configs/accelerate_multigpu.yaml](configs/accelerate_multigpu.yaml):

```yaml
num_machines: 2    # Number of nodes
num_processes: 16  # Total GPUs across all nodes, with one process per GPU
```

The current configuration uses 2 nodes with 8 GPUs per node. 

## Hardware Requirements

Our experiments were trained on NVIDIA A800 GPUs. The provided training configuration causes out-of-memory (OOM) errors on GPUs with 48 GB of VRAM.