"""Этап image -> 3D на TRELLIS.2.

Идёт мимо ComfyUI, официальным пакетом: сторонние враппер-ноды недокументированы
и существуют в четырёх конкурирующих форках, а это самый тяжёлый этап пайплайна.

API взят из README TRELLIS.2, не выдуман: Trellis2ImageTo3DPipeline.from_pretrained,
экспорт через o_voxel.postprocess.to_glb.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Экономит память на длинных батчах: без этого фрагментация выделений
# отъедает больше, чем сама модель.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class MeshSettings:
    """Параметры выгрузки меша. Всё настраиваемое — без зашитых констант."""

    decimation_target: int = 200_000
    texture_size: int = 2048
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0
    aabb: tuple = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    seed: int = 0


class Trellis2Runner:
    """Ленивая обёртка: модель грузится при первом запуске, не при импорте."""

    def __init__(self, model_id: str = "microsoft/TRELLIS.2-4B",
                 attn_backend: str | None = None) -> None:
        self.model_id = model_id
        # На картах без поддержки flash-attention (например V100) нужен xformers.
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

        if out_glb.exists() and out_glb.stat().st_size > 0:
            print(f"  [skip] {out_glb.name} уже есть")
            return out_glb

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
        print(f"  [ok  ] {out_glb.name}")
        return out_glb
