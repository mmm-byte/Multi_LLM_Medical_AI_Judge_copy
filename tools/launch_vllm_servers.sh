#!/usr/bin/env bash
# Launch 4 vLLM servers on 4 GPUs — recommended panel for the EMNLP paper.
#
# Each server is started inside its own tmux session so the notebook can
# tail its log and the user can attach for debugging.
#
# Requires:
#   * `vllm` on PATH
#   * 4x GPUs (default: CUDA 0,1,2,3)
#   * HUGGING_FACE_HUB_TOKEN exported (medgemma is gated)
#
# Usage:
#   bash tools/launch_vllm_servers.sh
#
# Customize via env:
#   GPU_IDS="0,1,2,3"          GPUs to use (comma separated, 4 IDs)
#   HF_HOME=./hf_cache         model cache directory
#   VLLM_BIN=$(which vllm)     path to vllm CLI
#   MAX_MODEL_LEN=4096         vLLM context-length flag value
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<<"$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain 4 IDs (got: $GPU_IDS)" >&2
  exit 2
fi

VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
if [[ -z "${VLLM_BIN}" ]]; then
  echo "vllm not found on PATH — pip install vllm first" >&2
  exit 2
fi

if [[ -z "${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]]; then
  echo "WARNING: HUGGING_FACE_HUB_TOKEN not set — medgemma download will 401" >&2
fi

HF_HOME="${HF_HOME:-$PWD/hf_cache}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "$LOG_DIR" "$HF_HOME"

TMUX_BIN="${TMUX_BIN:-$(command -v tmux)}"
TMUX_CONF="${TMUX_CONF:-/exec-daemon/tmux.portal.conf}"
if [[ ! -f "$TMUX_CONF" ]]; then TMUX_CONF=""; fi
tmux_cmd() {
  if [[ -n "$TMUX_CONF" ]]; then "$TMUX_BIN" -f "$TMUX_CONF" "$@"; else "$TMUX_BIN" "$@"; fi
}

start_one() {
  local name="$1" model="$2" port="$3" gpu="$4"
  local log="$LOG_DIR/vllm-${name}.log"
  echo "starting $name on port $port (GPU $gpu) -> $log"
  tmux_cmd kill-session -t "vllm-${name}" 2>/dev/null || true
  tmux_cmd new-session -d -s "vllm-${name}" -c "$PWD" -- bash -l
  tmux_cmd send-keys -t "vllm-${name}:0.0" \
    "CUDA_VISIBLE_DEVICES=${gpu} HF_HOME='${HF_HOME}' HUGGING_FACE_HUB_TOKEN='${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}' '${VLLM_BIN}' serve '${model}' --host 0.0.0.0 --port ${port} --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization 0.85 --dtype bfloat16 2>&1 | tee '${log}'" C-m
}

start_one medgemma         google/medgemma-4b-it           8001 "${GPUS[0]}"
start_one biomistral_dare  BioMistral/BioMistral-7B-DARE   8002 "${GPUS[1]}"
start_one medicine_chat    AdaptLLM/medicine-chat          8003 "${GPUS[2]}"
start_one biomistral_base  BioMistral/BioMistral-7B        8004 "${GPUS[3]}"

echo
echo "All 4 vLLM servers launched. Wait for 'Application startup complete' in each log:"
for n in medgemma biomistral_dare medicine_chat biomistral_base; do
  echo "  tail -f $LOG_DIR/vllm-${n}.log"
done
