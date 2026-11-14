#!/usr/bin/env bash
# One-click SemiBench.   BASE_URL=http://host:30000/v1 MODEL=Micius-Flash bash scripts/run_semibench.sh
set -euo pipefail
BASE_URL=${BASE_URL:-http://127.0.0.1:30000/v1}
MODEL=${MODEL:-Micius-Flash}
OUT=${OUT:-outputs/semibench/${MODEL//\//_}}
python -m micius_eval.semibench.run --base-url $BASE_URL --model "$MODEL" --output-dir "$OUT" \
  --reasoning-effort "${REASONING_EFFORT:-xhigh}" --workers "${WORKERS:-32}" "$@"
