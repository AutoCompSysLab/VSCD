#!/usr/bin/env bash
set -euo pipefail

# Run Table-4 style module ablations sequentially.
# Set DATA_ROOT, SAM_ROOT, SAM_CKPT, OUT_BASE, and CUDA_VISIBLE_DEVICES before running if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/train_wo_at.sh"
bash "${SCRIPT_DIR}/train_wo_csp.sh"
bash "${SCRIPT_DIR}/train_wo_cf.sh"
bash "${SCRIPT_DIR}/train_wo_cf_csp.sh"
