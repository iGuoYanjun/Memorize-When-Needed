#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Activate your Python environment before running this script.
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_ROOT}/checkpoints}"
export MODEL_NAME="${MODEL_NAME:-${CHECKPOINTS_DIR}/Wan2.1-I2V-14B-480P}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
video_root_dir="${VIDEO_ROOT_DIR:-/path/to/RealEstate10K}"
npz_path="${NPZ_PATH:-${video_root_dir}/annotations/train.npz}"
test_npz_path="${TEST_NPZ_PATH:-${video_root_dir}/annotations/val.npz}"
output_dir="${OUTPUT_DIR:-/path/to/output/camera_controlnet}"
log_file="${LOG_FILE:-${SCRIPT_DIR}/train_multigpu.log}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/configs/accelerate_single_gpu.yaml" \
  "${SCRIPT_DIR}/train.py" \
  --config_path="${PROJECT_ROOT}/config/wan2.1/wan_civitai.yaml" \
  --controlnet_config_path="${SCRIPT_DIR}/configs/controlnet.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --video_root_dir="$video_root_dir" \
  --npz_path="$npz_path" \
  --height=480 \
  --width=832 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=1\
  --num_train_epochs=50 \
  --checkpointing_steps=400 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=250 \
  --seed=42 \
  --output_dir="$output_dir" \
  --gradient_checkpointing \
  --adam_weight_decay=1e-4 \
  --adam_epsilon=1e-8 \
  --max_grad_norm=1.0 \
  --num_validation_samples=8 \
  --num_inference_steps=40 \
  --validation_steps=200 \
  --low_vram \
  --do_validation \
  --test_npz_path="$test_npz_path" > "$log_file" 2>&1
