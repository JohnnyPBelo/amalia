#!/usr/bin/env bash
# amalia-v1 launcher — starts the local Qwen2.5-7B orchestrator (llama.cpp Vulkan)
# and the amalia-v1 OpenAI-compatible server.
#
# Usage:  ./run.sh [config.yaml|config.e2e.yaml]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-config.yaml}"
LLAMA="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${ORCH_GGUF:-$HOME/models/qwen2.5-7b-instruct/Qwen2.5-7B-Instruct-Q5_K_M.gguf}"
ORCH_PORT="${ORCH_PORT:-8901}"

# 1) orchestrator (Qwen2.5-7B) on :8901 if not already up
if ! curl -sf "http://localhost:${ORCH_PORT}/health" >/dev/null 2>&1; then
  echo "[amalia] starting Qwen2.5-7B orchestrator on :${ORCH_PORT} ..."
  "$LLAMA" -m "$MODEL" --host 0.0.0.0 --port "$ORCH_PORT" \
           -ngl 99 -fa 1 -c 8192 --alias qwen2.5-7b-instruct \
           >/tmp/amalia-orchestrator.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf "http://localhost:${ORCH_PORT}/health" >/dev/null 2>&1 && break
    sleep 1
  done
  echo "[amalia] orchestrator ready."
else
  echo "[amalia] orchestrator already running on :${ORCH_PORT}."
fi

# 2) amalia-v1 server
echo "[amalia] starting amalia-v1 server with ${CONFIG} ..."
cd "$REPO"
source .venv/bin/activate
exec env AMALIA_CONFIG="$CONFIG" AMALIA_DEBUG="${AMALIA_DEBUG:-0}" \
     python -m uvicorn amalia.server:app --host 0.0.0.0 --port 8900
