#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate your Python environment before running this script.
PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${SCRIPT_DIR}/infer.py"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${SCRIPT_DIR}/checkpoints}"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-${CHECKPOINTS_DIR}/Wan2.1-I2V-14B-480P}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/wan2.1/wan_civitai.yaml}"
METADATA_PATH="${METADATA_PATH:-${SCRIPT_DIR}/assets/prompt.json}"
CONTROLNET_CKPT_DIR="${CONTROLNET_CKPT_DIR:-${CHECKPOINTS_DIR}/camera_controlnet}"
HISTORY_CONTROLNET_CKPT="${HISTORY_CONTROLNET_CKPT:-${CHECKPOINTS_DIR}/history_controlnet}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

GPU_ID="${GPU_ID:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference entry not found: ${INFER_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Launching open-assets inference on GPU=${GPU_ID}, output=${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON_BIN}" "${INFER_SCRIPT}" \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --metadata_path "${METADATA_PATH}" \
  --controlnet_ckpt_dir "${CONTROLNET_CKPT_DIR}" \
  --history_controlnet_ckpt "${HISTORY_CONTROLNET_CKPT}" \
  --output_path "${OUTPUT_DIR}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --video_sample_n_frames 49 \
  --guidance_scale 6.5 \
  --shift 3.0
