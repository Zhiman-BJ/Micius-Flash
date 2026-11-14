#!/usr/bin/env bash
# Serve Micius-Flash with vLLM behind an OpenAI-compatible API.
#
#   MODEL_PATH=/path/to/Micius-Flash TP=8 bash inference/serve_vllm.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-zhiman-ai/Micius-Flash}
SERVED_NAME=${SERVED_NAME:-Micius-Flash}
TP=${TP:-8}
PORT=${PORT:-30000}
CONTEXT=${CONTEXT:-262144}

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len "$CONTEXT" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "$@"
