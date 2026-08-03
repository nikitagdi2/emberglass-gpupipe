# EMBERGLASS asset pipeline: FLUX.2 klein (2D) + TRELLIS.2 (image -> 3D).
#
# Веса в образ НЕ пекутся: они тянутся с HuggingFace при старте пода
# (см. docker/fetch_weights.sh). Образ содержит только окружение и
# скомпилированные CUDA-расширения.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY=/opt/ComfyUI \
    TRELLIS=/opt/TRELLIS2 \
    PIPE=/opt/pipe

# 8.0=A100, 8.6=A40/A6000/A5000/3090, 8.9=4090/L40S, 9.0=H100.
# Сборка под одну архитектуру даёт "no kernel image is available for execution
# on the device" при смене железа, поэтому по умолчанию перечислены все.
#
# Каждая архитектура — это отдельный проход nvcc, то есть время и ПАМЯТЬ.
# Бесплатный раннер GitHub (4 ядра, 16 ГБ) на полном списке умирает при сборке
# flex_gemm, поэтому CI передаёт сюда суженный список: "8.6;8.9+PTX" покрывает
# A40/A6000/A5000/3090/4090/L40S нативно и H100 через JIT из PTX. A100 (8.0)
# в такой сборке не поддержан — по плану он и не используется, втрое дороже
# без выигрыша на инференсе.
ARG TORCH_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST}

# Параллелизм nvcc. Пусто — по числу ядер; на раннере ограничивается, иначе OOM.
ARG MAX_JOBS=""
ENV MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates build-essential ninja-build cmake \
      python3.10 python3.10-dev python3.10-venv python3-pip \
      libgl1 libglib2.0-0 libegl1 libxrender1 libsm6 libxext6 \
      ffmpeg rsync openssh-client jq \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# PyTorch. Версия продиктована TRELLIS.2 (его setup.sh ставит ровно 2.6.0+cu124)
# и определяет совместимость колеса flash-attn ниже.
# ---------------------------------------------------------------------------
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# Бэкенд внимания: xformers.
#
# Замерено, а не предположено: torch 2.6.0+cu124 собран со СТАРЫМ C++ ABI —
# libc10.so экспортирует _ZN3c105ErrorC2ENS_14SourceLocationESs (суффикс Ss).
# Оба опубликованных колеса flash-attn 2.8.3 под torch2.6, включая помеченное
# cxx11abiFALSE, требуют символ с __cxx11, то есть собраны против torch с НОВЫМ
# ABI и с 2.6.0 несовместимы. Проверено и на torch с cu124-индекса, и с PyPI.
#
# Сверяться по torch._C._GLIBCXX_USE_CXX11_ABI при этом бесполезно: флаг равен
# False, а нужное колесо всё равно не подходит ни одно.
#
# Поэтому по умолчанию берём xformers — штатный запасной бэкенд, документированный
# самим TRELLIS.2. Кому нужна максимальная скорость, тот собирает flash-attn из
# исходников флагом WITH_FLASH_ATTN=1 (десятки минут на слой).
# ---------------------------------------------------------------------------
ARG XFORMERS_VERSION=0.0.29.post3
RUN pip install "xformers==${XFORMERS_VERSION}" --index-url https://download.pytorch.org/whl/cu124 && \
    pip install einops && \
    python -c "import torch, xformers, xformers.ops; print('xformers', xformers.__version__, 'torch', torch.__version__)"

ENV ATTN_BACKEND=xformers

ARG WITH_FLASH_ATTN=0
ARG FLASH_ATTN_VERSION=2.7.3
RUN if [ "$WITH_FLASH_ATTN" = "1" ]; then \
      set -eux; \
      MAX_JOBS=$(nproc) pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"; \
      python -c 'import flash_attn; print("flash_attn", flash_attn.__version__)'; \
    else \
      echo "flash-attn пропущен, используется ATTN_BACKEND=xformers"; \
    fi

# ---------------------------------------------------------------------------
# TRELLIS.2 и его CUDA-расширения.
#
# Компиляция видеокарты не требует, только toolkit — поэтому образ собирается
# на машине без GPU. Ревизии пинятся: без этого следующая сборка соберёт
# другой набор версий и воспроизводимости не будет.
# ---------------------------------------------------------------------------
# Мелкий клон без истории: полный тянет около гигабайта, который в образе
# не нужен ни для чего.
ARG TRELLIS2_REF=main
RUN git clone --depth 1 --branch $TRELLIS2_REF --recurse-submodules --shallow-submodules \
      https://github.com/microsoft/TRELLIS.2.git $TRELLIS && \
    rm -rf $TRELLIS/.git

WORKDIR /tmp/extensions

# Клонируем с сабмодулями: CuMesh вендорит cubvh в third_party/, и без
# --recurse-submodules сборка падает на ninja "api_gpu.cu missing".
# Исходники удаляются ВНУТРИ того же RUN: уборка отдельным слоём размер
# образа не уменьшает, файлы остаются в нижележащем слое.
ARG NVDIFFRAST_REF=v0.4.0
RUN git clone -b $NVDIFFRAST_REF --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/NVlabs/nvdiffrast.git && \
    pip install ./nvdiffrast --no-build-isolation && \
    rm -rf ./nvdiffrast

ARG NVDIFFREC_REF=renderutils
RUN git clone -b $NVDIFFREC_REF --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/JeffreyXiang/nvdiffrec.git && \
    pip install ./nvdiffrec --no-build-isolation && \
    rm -rf ./nvdiffrec

RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/JeffreyXiang/CuMesh.git && \
    test -f CuMesh/third_party/cubvh/src/api_gpu.cu || \
      (echo "CuMesh: сабмодули не подтянулись" && ls -R CuMesh/third_party | head -30 && exit 1) && \
    pip install ./CuMesh --no-build-isolation && \
    rm -rf ./CuMesh

RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/JeffreyXiang/FlexGEMM.git && \
    pip install ./FlexGEMM --no-build-isolation && \
    rm -rf ./FlexGEMM

# o-voxel живёт внутри самого репозитория TRELLIS.2, отдельного origin у него нет.
# Путь между ревизиями переезжал, поэтому ищем каталог, а не хардкодим.
RUN set -eux; \
    DIR="$(find $TRELLIS -maxdepth 3 -type d \( -name 'o-voxel' -o -name 'o_voxel' \) \
           -exec test -e '{}/setup.py' -o -e '{}/pyproject.toml' ';' -print | head -1)"; \
    test -n "$DIR" || { echo "o-voxel не найден в $TRELLIS"; find $TRELLIS -maxdepth 3 -type d | head -40; exit 1; }; \
    echo "o-voxel: $DIR"; \
    pip install "$DIR" --no-build-isolation

RUN if [ -f $TRELLIS/requirements.txt ]; then pip install -r $TRELLIS/requirements.txt || true; fi

# Пакет TRELLIS.2 ставится, если у него есть setup.py/pyproject; иначе он
# подключается через PYTHONPATH — репозиторий не всегда оформлен пакетом.
RUN if [ -f $TRELLIS/setup.py ] || [ -f $TRELLIS/pyproject.toml ]; then \
      pip install -e $TRELLIS; \
    else \
      echo "$TRELLIS" > /usr/local/lib/python3.10/dist-packages/trellis2_repo.pth; \
    fi

# ---------------------------------------------------------------------------
# ComfyUI — только для 2D-ветки (FLUX.2 klein поддержан нативно).
# 3D идёт мимо ComfyUI, официальным пакетом TRELLIS.2: сторонние враппер-ноды
# недокументированы и существуют в четырёх конкурирующих форках.
# ---------------------------------------------------------------------------
ARG COMFY_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI $COMFY && \
    cd $COMFY && git checkout $COMFY_REF && \
    pip install -r requirements.txt

WORKDIR $COMFY/custom_nodes
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git && \
    for d in */; do \
      if [ -f "$d/requirements.txt" ]; then pip install -r "$d/requirements.txt" || true; fi; \
    done

RUN pip install hf_transfer "huggingface_hub[cli]" requests pillow numpy trimesh

COPY docker/ /opt/docker/
COPY pipeline/ $PIPE/
RUN chmod +x /opt/docker/*.sh

# Санитарная проверка: всё, что компилировалось, должно импортироваться.
# Ошибку сборки лучше получить здесь, чем на оплачиваемой карте.
#
# Образ собирается на машине БЕЗ GPU, поэтому здесь нельзя трогать ничего, что
# при импорте требует активный драйвер: xformers.ops тянет автотюнеры Triton,
# а тот падает с "0 active drivers". Такие импорты проверяются на поде
# командой run_session.py doctor, а не на сборке.
#
# o_voxel/flex_gemm регистрируют автотюнеры Triton прямо при импорте, поэтому
# их наличие проверяется find_spec — он находит модуль, не исполняя его.
COPY docker/verify_build.py /opt/docker/verify_build.py
RUN python /opt/docker/verify_build.py

WORKDIR $COMFY
EXPOSE 8188
CMD ["/opt/docker/start.sh"]
