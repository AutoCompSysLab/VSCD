#!/usr/bin/env bash
set -euo pipefail

# Default VSCDNet training script.
# Override paths from the command line if needed, e.g.:
#   DATA_ROOT=/path/to/vscd_dataset SAM_ROOT=/path/to/segment-anything SAM_CKPT=/path/to/sam_vit_b_01ec64.pth bash scripts/train_default.sh

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_ROOT=${DATA_ROOT:-/path/to/vscd_dataset}
SAM_ROOT=${SAM_ROOT:-/path/to/segment-anything}
SAM_CKPT=${SAM_CKPT:-/path/to/sam_vit_b_01ec64.pth}
OUT_BASE=${OUT_BASE:-./runs}

python -u train.py \
  --data_root "${DATA_ROOT}" \
  --sam_root "${SAM_ROOT}" \
  --sam_ckpt "${SAM_CKPT}" \
  --out_dir "${OUT_BASE}/vscd_default" \
  --split_train train \
  --split_val test \
  --device cuda \
  --epochs 80 \
  --seed 0 \
  --batch_size 2 \
  --num_workers 2 \
  --num_frames_fixed 32 \
  --backbone_chunk 2 \
  --amp \
  --topk_ref_per_t 4 \
  --max_ref_cands_per_t 6 \
  --local_k 5 \
  --max_msp_len 5 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --save_best_last_only \
  --val_thr 0.5 \
  --softmax_temp 0.5
