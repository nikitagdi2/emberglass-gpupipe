"""Реестр бэкендов этапа image -> 3D.

Бэкенды отличаются способом запуска: TRELLIS.2 вызывается как python-пакет,
Pixal3D — как CLI собственного репозитория. Поэтому семья описывается данными,
а запуск диспетчеризуется по виду.

Лицензия здесь не украшение: она решает, можно ли вообще класть выход в
коммерческую игру, и проверяется до, а не после генерации ростера.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Экономит память на длинных батчах: без этого фрагментация выделений
# отъедает больше, чем сама модель.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class MeshBackend:
    """Описание бэкенда image -> 3D."""

    id: str
    title: str
    kind: str            # "python" | "cli"
    license: str
    repo: str = ""
    model_id: str = ""
    root_env: str = ""   # переменная с путём к репозиторию, для kind="cli"
    vram_gb: float = 24.0
    notes: str = ""


BACKENDS: dict[str, MeshBackend] = {
    "trellis2": MeshBackend(
        id="trellis2",
        title="TRELLIS.2-4B",
        kind="python",
        license="MIT",
        model_id="microsoft/TRELLIS.2-4B",
        vram_gb=24.0,
        notes="Базовый бэкенд, скомпилирован в образе. Экспорт через "
              "o_voxel.postprocess.to_glb.",
    ),
    "pixal3d": MeshBackend(
        id="pixal3d",
        title="Pixal3D",
        kind="cli",
        license="MIT",
        repo="https://github.com/TencentARC/Pixal3D",
        root_env="PIXAL3D",
        vram_gb=24.0,
        notes="Построен поверх TRELLIS.2 и требует его окружения. Ставится "
              "только в образе с WITH_PIXAL3D=1: его requirements пинят "
              "pillow/transformers/diffusers и способны сломать ComfyUI. "
              "Поддерживает --low_vram и ATTN_BACKEND=sdpa.",
    ),
}

# Отклонено до генерации, чтобы не искать заново.
EXCLUDED = {
    "hunyuan3d-2.1": "tencent/Hunyuan3D-2.1 — лицензия tencent-hunyuan-community "
                     "запрещает использование в ЕС, Великобритании и Южной Корее, "
                     "плюс порог по MAU для крупного коммерческого использования. "
                     "Исключено 2026-08-09 по решению владельца. Технически также "
                     "дороже: пинит torch 2.5.1 против нашего 2.6.0 и требует "
                     "компиляции двух собственных расширений.",
}


def get(backend_id: str) -> MeshBackend:
    if backend_id not in BACKENDS:
        hint = EXCLUDED.get(backend_id)
        if hint:
            raise SystemExit(f"бэкенд '{backend_id}' исключён: {hint}")
        raise SystemExit(f"нет бэкенда '{backend_id}'. Доступны: {', '.join(sorted(BACKENDS))}")
    return BACKENDS[backend_id]


@dataclass
class MeshSettings:
    """Параметры выгрузки меша. Всё настраиваемое, без зашитых констант."""

    decimation_target: int = 200_000
    texture_size: int = 2048
    resolution: int = 1024
    low_vram: bool = False
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0
    aabb: tuple = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    seed: int = 0


class Trellis2Runner:
    """Ленивая обёртка: модель грузится при первом запуске, не при импорте."""

    def __init__(self, model_id: str, attn_backend: str | None = None) -> None:
        self.model_id = model_id
        # На картах без flash-attention нужен запасной бэкенд внимания.
        if attn_backend:
            os.environ["ATTN_BACKEND"] = attn_backend
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
            pipeline.cuda()
            self._pipeline = pipeline
        return self._pipeline

    def run(self, image_path: Path, out_glb: Path, settings: MeshSettings) -> Path:
        from PIL import Image
        import o_voxel

        pipeline = self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        mesh = pipeline.run(image, seed=settings.seed)[0]

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[list(settings.aabb[0]), list(settings.aabb[1])],
            decimation_target=settings.decimation_target,
            texture_size=settings.texture_size,
            remesh=settings.remesh,
            remesh_band=settings.remesh_band,
            remesh_project=settings.remesh_project,
            verbose=False,
        )
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(out_glb), extension_webp=True)
        return out_glb


class CliRunner:
    """Бэкенд, который запускается собственным скриптом из своего репозитория."""

    def __init__(self, backend: MeshBackend, attn_backend: str | None = None) -> None:
        root = os.environ.get(backend.root_env or "", "")
        if not root or not Path(root).exists():
            raise SystemExit(
                f"{backend.title}: каталог не найден (переменная {backend.root_env}). "
                f"Образ собран без него? Нужен --build-arg WITH_PIXAL3D=1"
            )
        self.backend = backend
        self.root = Path(root)
        self.attn_backend = attn_backend

    def run(self, image_path: Path, out_glb: Path, settings: MeshSettings) -> Path:
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "inference.py",
                   "--image", str(image_path),
                   "--output", str(out_glb),
                   "--resolution", str(settings.resolution)]
        if settings.low_vram:
            command.append("--low_vram")

        env = dict(os.environ)
        if self.attn_backend:
            # У Pixal3D свой набор бэкендов; xformers он не знает, sdpa знает.
            env["ATTN_BACKEND"] = "sdpa" if self.attn_backend == "xformers" else self.attn_backend

        result = subprocess.run(command, cwd=str(self.root), env=env,
                                capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
            raise SystemExit(f"{self.backend.title} упал (код {result.returncode}):\n  "
                             + "\n  ".join(tail))
        return out_glb


def make_runner(backend: MeshBackend, attn_backend: str | None = None):
    if backend.kind == "python":
        return Trellis2Runner(backend.model_id, attn_backend)
    return CliRunner(backend, attn_backend)
