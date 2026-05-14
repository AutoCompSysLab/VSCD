#!/usr/bin/env bash
set -euo pipefail

# Hyper-parameter ablations corresponding to the paper appendix.
# This script launches one training run per setting. It can take a long time.

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_ROOT=${DATA_ROOT:-/path/to/vscd_dataset}
SAM_ROOT=${SAM_ROOT:-/path/to/segment-anything}
SAM_CKPT=${SAM_CKPT:-/path/to/sam_vit_b_01ec64.pth}
OUT_BASE=${OUT_BASE:-./runs/hparam_ablations}

run_exp() {
  local name="$1"
  shift
  python -u train.py \
    --data_root "${DATA_ROOT}" \
    --sam_root "${SAM_ROOT}" \
    --sam_ckpt "${SAM_CKPT}" \
    --out_dir "${OUT_BASE}/${name}" \
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
    --softmax_temp 0.5 \
    "$@"
}

run_exp "K2" --topk_ref_per_t 2
run_exp "K6" --topk_ref_per_t 6
run_exp "Smax4" --max_ref_cands_per_t 4
run_exp "Smax8" --max_ref_cands_per_t 8
run_exp "k3" --local_k 3
run_exp "k7" --local_k 7
run_exp "Lmax3" --max_msp_len 3
run_exp "Lmax7" --max_msp_len 7
run_exp "Tkey16" --num_frames_fixed 16
run_exp "Tkey64" --num_frames_fixed 64
