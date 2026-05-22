#!/usr/bin/env bash
# Launch 4 CPU substitute servers (used by the notebook when no GPU is available).
# Each runs `python serve_hf_model.py` in its own tmux session.
set -euo pipefail

HF_HOME="${HF_HOME:-$PWD/hf_cache}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "$LOG_DIR" "$HF_HOME"

TMUX_BIN=$(command -v tmux)
TMUX_CONF="${TMUX_CONF:-/exec-daemon/tmux.portal.conf}"
if [[ ! -f "$TMUX_CONF" ]]; then TMUX_CONF=""; fi
tmux_cmd() {
  if [[ -n "$TMUX_CONF" ]]; then "$TMUX_BIN" -f "$TMUX_CONF" "$@"; else "$TMUX_BIN" "$@"; fi
}

start_one() {
  local port="$1" model="$2"
  local name="judge-${port}"
  local log="$LOG_DIR/${name}.log"
  echo "starting ${name} ${model} -> ${log}"
  tmux_cmd kill-session -t "${name}" 2>/dev/null || true
  tmux_cmd new-session -d -s "${name}" -c "$PWD" -- bash -l
  tmux_cmd send-keys -t "${name}:0.0" \
    "HF_HOME='${HF_HOME}' DTYPE=bfloat16 OMP_NUM_THREADS=1 HOST=127.0.0.1 MODEL_ID='${model}' PORT=${port} python3 serve_hf_model.py 2>&1 | tee '${log}'" C-m
}

start_one 8001 "Qwen/Qwen2.5-1.5B-Instruct"
start_one 8002 "Qwen/Qwen2.5-0.5B-Instruct"
start_one 8003 "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
start_one 8004 "HuggingFaceTB/SmolLM2-360M-Instruct"

echo
echo "All 4 CPU servers launched. Health-check with:"
for p in 8001 8002 8003 8004; do echo "  curl -fsS http://127.0.0.1:${p}/health"; done
