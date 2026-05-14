#!/usr/bin/env bash
set -euo pipefail

# Default VSCDNet evaluation script.
# Example:
#   DATA_ROOT=/path/to/vscd_dataset SAM_ROOT=/path/to/segment-anything SAM_CKPT=/path/to/sam_vit_b_01ec64.pth MODEL_CKPT=./runs/vscd_default/ckpt_best.pt bash scripts/eval_default.sh

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_ROOT=${DATA_ROOT:-/path/to/vscd_dataset}
SAM_ROOT=${SAM_ROOT:-/path/to/segment-anything}
SAM_CKPT=${SAM_CKPT:-/path/to/sam_vit_b_01ec64.pth}
MODEL_CKPT=${MODEL_CKPT:-./runs/vscd_default/ckpt_best.pt}
SAVE_DIR=${SAVE_DIR:-./outputs/eval_default}

python -u eval.py \
  --data_root "${DATA_ROOT}" \
  --sam_root "${SAM_ROOT}" \
  --sam_ckpt "${SAM_CKPT}" \
  --model_ckpt "${MODEL_CKPT}" \
  --split test \
  --device cuda \
  --batch_size 1 \
  --num_workers 2 \
  --num_frames_fixed 32 \
  --backbone_chunk 2 \
  --amp \
  --topk_ref_per_t 4 \
  --max_ref_cands_per_t 6 \
  --local_k 5 \
  --max_msp_len 5 \
  --thr 0.5 \
  --softmax_temp 0.5 \
  --save_dir "${SAVE_DIR}"
