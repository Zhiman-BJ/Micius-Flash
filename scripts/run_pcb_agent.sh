#!/usr/bin/env bash
# One-click PCB Agent: containerized generation, then three-axis review.
#
#   export OPENAI_API_KEY=...                 # key for the verifier model (Codex CLI)
#   BASE_URL=http://host:30000/v1 MODEL=Micius-Flash bash scripts/run_pcb_agent.sh
#
# The model server must be reachable from Docker containers (127.0.0.1 is
# rewritten to host.docker.internal automatically). SKIP_VERIFY=1 runs generation only.
set -euo pipefail
BASE_URL=${BASE_URL:-http://127.0.0.1:30000/v1}
MODEL=${MODEL:-Micius-Flash}
OUT=${OUT:-outputs/pcb_agent/${MODEL//\//_}}
python -m micius_eval.pcb_agent.run --base-url "$BASE_URL" --model "$MODEL" --output-dir "$OUT" \
  --concurrency "${CONCURRENCY:-10}" "$@"
if [ "${SKIP_VERIFY:-0}" != 1 ]; then
  python -m micius_eval.pcb_agent.verify --output-dir "$OUT" --workers "${VERIFY_WORKERS:-4}"
fi
