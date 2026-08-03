#!/usr/bin/env bash
# Точка входа пода: тянем веса, поднимаем ComfyUI.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH="${PIPE:-/opt/pipe}:${PYTHONPATH:-}"

mkdir -p /workspace/models /workspace/inputs /workspace/output /workspace/hf

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi недоступен"

echo "=== веса ==="
# Неудача загрузки НЕ должна ронять контейнер: мёртвый под на оплаченной карте
# хуже живого с неполными весами. Поднимаем ComfyUI в любом случае, чтобы
# `run_session.py doctor` мог сказать, чего именно не хватает, и чтобы можно
# было починить токен и дозагрузить веса не пересоздавая под.
WEIGHTS_OK=1
python /opt/docker/fetch_weights.py --comfy "${COMFY:-/opt/ComfyUI}" --models /workspace/models \
  || WEIGHTS_OK=0

if [ "$WEIGHTS_OK" != "1" ]; then
  echo
  echo "!!! ВЕСА ЗАГРУЖЕНЫ НЕ ПОЛНОСТЬЮ."
  echo "!!! Частая причина: не задан HF_TOKEN либо не принято соглашение на"
  echo "!!!   huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8"
  echo "!!! ComfyUI поднимается всё равно. Диагностика:"
  echo "!!!   python /opt/pipe/run_session.py doctor"
  echo "!!! Повторная загрузка после починки токена:"
  echo "!!!   python /opt/docker/fetch_weights.py"
  echo
fi

echo "=== ComfyUI :8188 ==="
exec python "${COMFY:-/opt/ComfyUI}/main.py" --listen 0.0.0.0 --port 8188 "$@"
