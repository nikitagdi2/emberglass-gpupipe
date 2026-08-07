"""Клиент ComfyUI и построитель графа FLUX.2 klein.

Граф собирается кодом и проверяется против живой схемы `/object_info`, а не
пишется руками: угаданный API-граф падает уже на арендованной карте, а
проверка схемы стоит секунду и делается на CPU.

Модуль НЕ должен называться `comfy`: у ComfyUI есть каталог `comfy/` без
`__init__.py`, то есть namespace package. Обычный модуль-однофамилец Python
предпочитает namespace-пакету НЕЗАВИСИМО от порядка путей, поэтому даже
каталог самого main.py в sys.path[0] не защищает: `import comfy.options`
падает с «'comfy' is not a package», и контейнер уходит в перезапуск.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests


class ComfyError(RuntimeError):
    pass


@dataclass
class SamplerSettings:
    """Контракт сэмплера для базового klein.

    cfg строго больше 1.0: у дистиллированного варианта cfg=1.0, и тогда
    ComfyUI пропускает безусловный проход — негативный промпт не действует
    вообще, молча. Все промпты арт-библии держатся на негативах.
    """

    steps: int = 22
    cfg: float = 4.0
    sampler: str = "euler"
    scheduler: str = "simple"
    # Осознанное разрешение cfg<=1 для дистиллированных сборок. Проверка
    # живёт здесь, а не только в вызывающем коде: build_txt2img_graph зовёт
    # validate() сам, и обход, сделанный снаружи, до него не доходил.
    allow_low_cfg: bool = False

    def validate(self) -> None:
        if self.cfg <= 1.0 and not self.allow_low_cfg:
            raise ComfyError(
                f"cfg={self.cfg}: при cfg<=1.0 негативный промпт не действует. "
                "Нужен базовый klein и cfg>1, либо явный --allow-cfg1."
            )


@dataclass
class NodeSpec:
    """Узел графа: класс, входы и человекочитаемая роль."""

    class_type: str
    inputs: dict
    title: str = ""


@dataclass
class GraphPlan:
    """Именованные кандидаты классов нод.

    ComfyUI переименовывает ноды между релизами, поэтому для каждой роли
    держим список кандидатов, а конкретный выбирается по живой схеме.
    """

    diffusion_loader: list[str] = field(default_factory=lambda: ["UNETLoader", "DiffusionModelLoader"])
    clip_loader: list[str] = field(default_factory=lambda: ["CLIPLoader"])
    vae_loader: list[str] = field(default_factory=lambda: ["VAELoader"])
    text_encode: list[str] = field(default_factory=lambda: ["CLIPTextEncode"])
    latent: list[str] = field(default_factory=lambda: ["EmptySD3LatentImage", "EmptyLatentImage"])
    sampler: list[str] = field(default_factory=lambda: ["KSampler"])
    decode: list[str] = field(default_factory=lambda: ["VAEDecode"])
    save: list[str] = field(default_factory=lambda: ["SaveImage"])


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self._schema: dict | None = None

    # --- схема -------------------------------------------------------------

    def object_info(self) -> dict:
        if self._schema is None:
            response = requests.get(f"{self.base_url}/object_info", timeout=self.timeout)
            response.raise_for_status()
            self._schema = response.json()
        return self._schema

    def pick_class(self, candidates: list[str], role: str) -> str:
        schema = self.object_info()
        for candidate in candidates:
            if candidate in schema:
                return candidate
        raise ComfyError(
            f"роль '{role}': ни один из классов {candidates} не найден в /object_info. "
            f"Похожие: {sorted(k for k in schema if any(c.lower() in k.lower() for c in candidates))[:10]}"
        )

    def input_names(self, class_type: str) -> set[str]:
        info = self.object_info()[class_type]["input"]
        names: set[str] = set()
        for section in ("required", "optional"):
            names.update(info.get(section, {}).keys())
        return names

    def enum_values(self, class_type: str, input_name: str) -> list:
        info = self.object_info()[class_type]["input"]
        for section in ("required", "optional"):
            if input_name in info.get(section, {}):
                spec = info[section][input_name][0]
                return list(spec) if isinstance(spec, list) else []
        return []

    def validate_graph(self, graph: dict) -> list[str]:
        """Расхождения графа с живой схемой. Пустой список — граф исполним."""
        problems: list[str] = []
        schema = self.object_info()
        for node_id, node in graph.items():
            class_type = node.get("class_type")
            if class_type not in schema:
                problems.append(f"узел {node_id}: неизвестный класс '{class_type}'")
                continue
            allowed = self.input_names(class_type)
            for key in node.get("inputs", {}):
                if key not in allowed:
                    problems.append(
                        f"узел {node_id} ({class_type}): вход '{key}' отсутствует; "
                        f"есть {sorted(allowed)}"
                    )
        return problems

    # --- исполнение --------------------------------------------------------

    def submit(self, graph: dict) -> str:
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": graph, "client_id": self.client_id},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise ComfyError(f"/prompt вернул {response.status_code}: {response.text[:800]}")
        return response.json()["prompt_id"]

    def wait(self, prompt_id: str, poll_seconds: float = 2.0, timeout_seconds: float = 1800.0) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout)
            response.raise_for_status()
            history = response.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(f"выполнение упало: {json.dumps(status)[:1200]}")
                return entry.get("outputs", {})
            time.sleep(poll_seconds)
        raise ComfyError(f"{prompt_id}: не дождались за {timeout_seconds:.0f} с")

    def download_image(self, image: dict, dest: Path) -> Path:
        query = urllib.parse.urlencode({
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        })
        response = requests.get(f"{self.base_url}/view?{query}", timeout=self.timeout)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return dest


def build_txt2img_graph(client: ComfyClient, *, positive: str, negative: str, width: int,
                        height: int, seed: int, sampler: SamplerSettings,
                        diffusion_name: str, clip_name: str, vae_name: str,
                        filename_prefix: str, clip_type: str = "flux2",
                        plan: GraphPlan | None = None) -> dict:
    """API-граф text2img под фактическую схему сервера.

    Форма графа одинакова для всех поддерживаемых семей моделей; различается
    только тип текстового энкодера, поэтому он и вынесен в параметр.
    """
    sampler.validate()
    plan = plan or GraphPlan()

    cls_diffusion = client.pick_class(plan.diffusion_loader, "загрузчик диффузионной модели")
    cls_clip = client.pick_class(plan.clip_loader, "загрузчик текстового энкодера")
    cls_vae = client.pick_class(plan.vae_loader, "загрузчик VAE")
    cls_encode = client.pick_class(plan.text_encode, "кодирование текста")
    cls_latent = client.pick_class(plan.latent, "пустой латент")
    cls_sampler = client.pick_class(plan.sampler, "сэмплер")
    cls_decode = client.pick_class(plan.decode, "декодер")
    cls_save = client.pick_class(plan.save, "сохранение")

    diffusion_inputs = {"unet_name": diffusion_name}
    if "weight_dtype" in client.input_names(cls_diffusion):
        options = client.enum_values(cls_diffusion, "weight_dtype")
        diffusion_inputs["weight_dtype"] = "default" if "default" in options else (options[0] if options else "default")

    clip_inputs = {"clip_name": clip_name}
    if "type" in client.input_names(cls_clip):
        options = [str(v) for v in client.enum_values(cls_clip, "type")]
        if clip_type not in options:
            raise ComfyError(
                f"{cls_clip}.type не знает '{clip_type}' в этой ревизии ComfyUI. "
                f"Доступно: {sorted(options)}"
            )
        clip_inputs["type"] = clip_type

    latent_inputs = {"width": width, "height": height, "batch_size": 1}

    graph = {
        "1": {"class_type": cls_diffusion, "inputs": diffusion_inputs},
        "2": {"class_type": cls_clip, "inputs": clip_inputs},
        "3": {"class_type": cls_vae, "inputs": {"vae_name": vae_name}},
        "4": {"class_type": cls_encode, "inputs": {"text": positive, "clip": ["2", 0]}},
        "5": {"class_type": cls_encode, "inputs": {"text": negative, "clip": ["2", 0]}},
        "6": {"class_type": cls_latent, "inputs": latent_inputs},
        "7": {"class_type": cls_sampler, "inputs": {
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": seed,
            "steps": sampler.steps,
            "cfg": sampler.cfg,
            "sampler_name": sampler.sampler,
            "scheduler": sampler.scheduler,
            "denoise": 1.0,
        }},
        "8": {"class_type": cls_decode, "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": cls_save, "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}},
    }
    return graph
