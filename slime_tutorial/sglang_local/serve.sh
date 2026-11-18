#!/usr/bin/env bash
set -euo pipefail
: "${SGLANG_MODEL:?Set SGLANG_MODEL to a local model directory}"
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit or runtime directory}"
export TOKENIZERS_PARALLELISM=false
exec "${SGLANG_PYTHON:-/opt/sglang-venv/bin/python}" -m sglang.launch_server \
  --model-path "$SGLANG_MODEL" --host 127.0.0.1 --port "${SGLANG_PORT:-18080}" \
  --dtype bfloat16 --context-length 256 --mem-fraction-static 0.4 \
  --attention-backend torch_native --sampling-backend pytorch \
  --disable-cuda-graph --max-running-requests 4
