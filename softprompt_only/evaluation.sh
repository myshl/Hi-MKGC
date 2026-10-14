#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash evaluation.sh DATASET CHECKPOINT [main.py arguments]" >&2
  exit 2
fi
SOFTPROMPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOFTPROMPT_PYTHON="${SOFTPROMPT_PYTHON:-$SOFTPROMPT_ROOT/.venv/bin/python}"
SOFTPROMPT_DATASET="$1"
# Resolve the supplied checkpoint relative to the caller, before changing cwd.
SOFTPROMPT_CHECKPOINT="$(realpath -- "$2")"
shift 2
cd "$SOFTPROMPT_ROOT"

exec "$SOFTPROMPT_PYTHON" main.py \
  -dataset "$SOFTPROMPT_DATASET" \
  -pretrained_model ./model_path/bert-base-uncased \
  -batch_size 112 -val_batch_size 128 -desc_max_length 40 \
  -prompt_length 10 -label_smoothing 0.1 \
  -embed_dim 256 -k_w 16 -k_h 16 \
  -model_path "$SOFTPROMPT_CHECKPOINT" -eval_split test \
  "$@"
