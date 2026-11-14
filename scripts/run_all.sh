#!/usr/bin/env bash
# Evaluate one served model on all three Micius benchmarks.
#   BASE_URL=http://host:30000/v1 MODEL=Micius-Flash bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_semibench.sh
bash scripts/run_semiconductor_agent.sh
bash scripts/run_pcb_agent.sh
python -m micius_eval.report outputs
