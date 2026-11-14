#!/usr/bin/env bash
# Download benchmark assets that are too large for git.
#
#   bash scripts/download_data.sh semiconductor_agent      # environments + tool runtimes + rubrics (large; see docs/semiconductor_agent.md)
#   MICIUS_DATA=/data/micius bash scripts/download_data.sh semiconductor_agent
#
# SemiBench and PCB Agent inputs ship with this repository (data/).
set -euo pipefail

WHAT=${1:-semiconductor_agent}
MICIUS_DATA=${MICIUS_DATA:-$(pwd)/data}
HF_DATASET=${MICIUS_HF_DATASET:-zhiman-ai/Micius-Flash-Bench}

case "$WHAT" in
  semiconductor_agent)
    command -v hf >/dev/null || pip install -q "huggingface_hub>=1.0"
    command -v unzstd >/dev/null || { echo "zstd is required (apt install zstd / conda install zstd)"; exit 1; }
    mkdir -p "$MICIUS_DATA/semiconductor_agent"
    hf download "$HF_DATASET" --repo-type dataset \
      --include "semiconductor_agent/*" --local-dir "$MICIUS_DATA"
    python -m micius_eval.semiconductor_agent.setup_data --data-root "$MICIUS_DATA/semiconductor_agent"
    ;;
  *)
    echo "unknown dataset: $WHAT (expected: semiconductor_agent)"; exit 2 ;;
esac
echo "Done. Export MICIUS_DATA=$MICIUS_DATA before running the evaluation scripts."
