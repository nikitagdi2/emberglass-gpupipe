#!/usr/bin/env bash
# Точка входа пода: тянем веса, поднимаем ComfyUI.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH="${PIPE:-/opt/pipe}:${PYTHONPATH:-}"

mkdir -p /workspace/{models,inputs,output,hf}

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi недоступен"

echo "=== веса ==="
python /opt/docker/fetch_weights.py --comfy "${COMFY:-/opt/ComfyUI}" --models /workspace/models

echo "=== ComfyUI :8188 ==="
exec python "${COMFY:-/opt/ComfyUI}/main.py" --listen 0.0.0.0 --port 8188 "$@"
