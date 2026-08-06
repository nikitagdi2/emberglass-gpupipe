#!/usr/bin/env bash
# Точка входа пода: тянем веса, поднимаем ComfyUI.
set -euo pipefail

# hf_transfer больше не используется, huggingface_hub ругается на него как на
# устаревший и просит взамен ускорение через Xet.
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=${HF_HOME:-/workspace/hf}

# PYTHONPATH сюда НЕ добавляем. Наши модули лежат в /opt/pipe, и попадание
# этого каталога на общий путь импорта затеняет одноимённые пакеты ComfyUI.
# run_session.py сам подкладывает свой каталог в sys.path, ему это не нужно.

mkdir -p /workspace/models /workspace/inputs /workspace/output /workspace/hf

# RunPod прокидывает публичный ключ аккаунта переменной PUBLIC_KEY и ожидает,
# что образ сам поднимет sshd. Без него ни `ssh`, ни `rsync` до пода не дойдут:
# слушать некому, а результаты сессии забирать чем-то надо.
setup_ssh() {
  mkdir -p /root/.ssh && chmod 700 /root/.ssh

  if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
  else
    echo "PUBLIC_KEY не задан: sshd поднимется, но пускать будет некого."
    echo "  Ключ добавляется в RunPod -> Settings -> SSH Public Keys."
  fi

  ssh-keygen -A >/dev/null 2>&1 || true   # хостовые ключи, если их ещё нет
  mkdir -p /run/sshd
  /usr/sbin/sshd || echo "sshd не стартовал; передача файлов только через runpodctl"
}

echo "=== SSH ==="
setup_ssh

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
# Без exec: если ComfyUI упадёт, контейнер не должен умирать вместе с ним.
# Иначе RunPod перезапускает его по кругу, sshd умирает вместе с контейнером,
# и подключиться для диагностики некуда — карта при этом тарифицируется.
cd "${COMFY:-/opt/ComfyUI}"
python "${COMFY:-/opt/ComfyUI}/main.py" --listen 0.0.0.0 --port 8188 "$@" || COMFY_EXIT=$?

echo
echo "!!! ComfyUI завершился с кодом ${COMFY_EXIT:-0}."
echo "!!! Контейнер оставлен живым намеренно: sshd работает, можно зайти и"
echo "!!! посмотреть, вместо бесконечного перезапуска вслепую."
echo "!!!   python /opt/pipe/run_session.py doctor"
echo "!!! Когда закончишь — TERMINATE пода, карта тарифицируется."
echo

# Держим контейнер живым ради sshd.
tail -f /dev/null
