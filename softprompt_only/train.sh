#!/usr/bin/env bash
set -euo pipefail

SOFTPROMPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SOFTPROMPT_ROOT"
SOFTPROMPT_PYTHON="${SOFTPROMPT_PYTHON:-$SOFTPROMPT_ROOT/.venv/bin/python}"
SOFTPROMPT_DATASET="${1:-DB15K}"
if [[ $# -gt 0 ]]; then shift; fi

exec "$SOFTPROMPT_PYTHON" main.py \
  -dataset "$SOFTPROMPT_DATASET" \
  -pretrained_model ./model_path/bert-base-uncased \
  -batch_size 112 -val_batch_size 128 -desc_max_length 40 \
  -lr 5e-4 -prompt_length 10 -label_smoothing 0.1 \
  -alpha 0.1 -n_lar 8 -alpha_step 0.00001 \
  -embed_dim 256 -k_w 16 -k_h 16 -epoch 30 \
  -check_val_every_n_epoch 1 -eval_split valid \
  "$@"
