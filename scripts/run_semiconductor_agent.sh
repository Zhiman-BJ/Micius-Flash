#!/usr/bin/env bash
# One-click Semiconductor Agent: generation, then frozen-rubric verification.
#
#   export MICIUS_DATA=/path/to/data          # after scripts/download_data.sh semiconductor_agent
#   export OPENAI_API_KEY=...                 # key for the verifier model (Codex CLI)
#   BASE_URL=http://host:30000/v1 MODEL=Micius-Flash bash scripts/run_semiconductor_agent.sh
#
# SKIP_VERIFY=1 runs generation only.
set -euo pipefail
BASE_URL=${BASE_URL:-http://127.0.0.1:30000/v1}
MODEL=${MODEL:-Micius-Flash}
OUT=${OUT:-outputs/semiconductor_agent/${MODEL//\//_}}
python -m micius_eval.semiconductor_agent.run --base-url $BASE_URL --model "$MODEL" --output-dir "$OUT" \
  --reasoning-effort "${REASONING_EFFORT:-xhigh}" --workers "${WORKERS:-16}" "$@"
if [ "${SKIP_VERIFY:-0}" != 1 ]; then
  python -m micius_eval.semiconductor_agent.verify --output-dir "$OUT" --workers "${VERIFY_WORKERS:-8}" \
    ${VERIFIER_BASE_URL:+--verifier-base-url $VERIFIER_BASE_URL}
fi
