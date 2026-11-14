#!/usr/bin/env bash
# Serve Micius-Flash with SGLang behind an OpenAI-compatible API.
#
#   MODEL_PATH=/path/to/Micius-Flash TP=8 bash inference/serve_sglang.sh
#
# 27B in BF16 needs ~55 GB of weights: TP=1 fits on one 80 GB GPU with a short
# context; TP=4/8 is recommended for the full 256K context and agentic workloads.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-zhiman-ai/Micius-Flash}
SERVED_NAME=${SERVED_NAME:-Micius-Flash}
TP=${TP:-8}
PORT=${PORT:-30000}
CONTEXT=${CONTEXT:-262144}
MEM_FRACTION=${MEM_FRACTION:-0.85}

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tp "$TP" \
  --dtype bfloat16 \
  --context-length "$CONTEXT" \
  --mem-fraction-static "$MEM_FRACTION" \
  --chunked-prefill-size 8192 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  "$@"
