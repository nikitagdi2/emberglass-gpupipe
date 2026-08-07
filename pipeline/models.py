"""Реестр моделей генерации концептов.

Каждая семья моделей отличается тремя вещами: файлом диффузии, текстовым
энкодером (и его типом в ComfyUI) и VAE. Форма графа при этом одна и та же,
поэтому семья описывается данными, а не отдельным построителем.

Тип энкодера обязан существовать в живой схеме `CLIPLoader.type` — проверяется
на поде командой `doctor`, потому что набор типов зависит от ревизии ComfyUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WeightFile:
    """Один файл весов: откуда взять и в какой каталог ComfyUI положить."""

    repo: str
    patterns: list[str]
    dest: str          # относительно models/ внутри ComfyUI
    gated: bool = False


@dataclass(frozen=True)
class ModelPreset:
    """Семья модели целиком: веса, тип энкодера и контракт сэмплера."""

    id: str
    title: str
    clip_type: str
    files: list[WeightFile]
    steps: int
    cfg: float
    sampler: str = "euler"
    scheduler: str = "simple"
    negatives_work: bool = True
    notes: str = ""
    # Подсказки для выбора конкретного файла среди уже загруженных.
    diffusion_hint: str = ""
    clip_hint: str = ""
    vae_hint: str = ""


_FLUX2_ENCODER = WeightFile(
    repo="Comfy-Org/flux2-klein-9B",
    patterns=[r"text_encoders/.*qwen_3_8b.*fp8.*\.safetensors$",
              r"text_encoders/.*qwen_3_8b.*\.safetensors$"],
    dest="text_encoders",
)

_FLUX2_VAE = WeightFile(
    repo="Comfy-Org/flux2-dev",
    patterns=[r"vae/flux2-vae\.safetensors$"],
    dest="vae",
)


PRESETS: dict[str, ModelPreset] = {
    "klein-base-9b": ModelPreset(
        id="klein-base-9b",
        title="FLUX.2 klein base 9B fp8",
        clip_type="flux2",
        files=[
            WeightFile(repo="black-forest-labs/FLUX.2-klein-base-9b-fp8",
                       patterns=[r"klein.*base.*9b.*fp8.*\.safetensors$", r".*\.safetensors$"],
                       dest="diffusion_models", gated=True),
            _FLUX2_ENCODER, _FLUX2_VAE,
        ],
        steps=22, cfg=4.0,
        diffusion_hint="klein-base-9b", clip_hint="qwen_3_8b", vae_hint="flux2-vae",
        notes="Базовая сборка: cfg>1, негативные промпты действуют.",
    ),

    "klein-9b": ModelPreset(
        id="klein-9b",
        title="FLUX.2 klein 9B fp8 (дистиллированная)",
        clip_type="flux2",
        files=[
            WeightFile(repo="black-forest-labs/FLUX.2-klein-9b-fp8",
                       patterns=[r"klein-9b.*fp8.*\.safetensors$", r".*\.safetensors$"],
                       dest="diffusion_models", gated=True),
            _FLUX2_ENCODER, _FLUX2_VAE,
        ],
        steps=4, cfg=1.0, negatives_work=False,
        diffusion_hint="klein-9b", clip_hint="qwen_3_8b", vae_hint="flux2-vae",
        notes="Примерно вшестеро быстрее базовой, но при cfg=1 негативы не "
              "действуют: в кадре проступает то, что негатив запрещает "
              "(замечено: тень под объектом).",
    ),

    "krea2-turbo": ModelPreset(
        id="krea2-turbo",
        title="Krea 2 Turbo fp8 (12B DiT)",
        clip_type="krea2",
        files=[
            WeightFile(repo="Comfy-Org/Krea-2",
                       patterns=[r"diffusion_models/krea2_turbo_fp8_scaled\.safetensors$"],
                       dest="diffusion_models"),
            WeightFile(repo="Comfy-Org/Krea-2",
                       patterns=[r"text_encoders/qwen3vl_4b_fp8_scaled\.safetensors$"],
                       dest="text_encoders"),
            WeightFile(repo="Comfy-Org/Krea-2",
                       patterns=[r"vae/qwen_image_vae\.safetensors$"],
                       dest="vae"),
        ],
        steps=8, cfg=1.0, negatives_work=False,
        diffusion_hint="krea2_turbo", clip_hint="qwen3vl_4b", vae_hint="qwen_image_vae",
        notes="Turbo-сборка: энкодер Qwen3-VL-4B, VAE от Qwen Image. "
              "Как и всякая turbo, идёт на низком cfg — негативы под вопросом.",
    ),

    "qwen-image": ModelPreset(
        id="qwen-image",
        title="Qwen-Image fp8 (20B MMDiT)",
        clip_type="qwen_image",
        files=[
            WeightFile(repo="Comfy-Org/Qwen-Image_ComfyUI",
                       patterns=[r"split_files/diffusion_models/qwen_image_fp8_e4m3fn\.safetensors$"],
                       dest="diffusion_models"),
            WeightFile(repo="Comfy-Org/Qwen-Image_ComfyUI",
                       patterns=[r"split_files/text_encoders/qwen_2\.5_vl_7b_fp8_scaled\.safetensors$"],
                       dest="text_encoders"),
            WeightFile(repo="Comfy-Org/Qwen-Image_ComfyUI",
                       patterns=[r"split_files/vae/qwen_image_vae\.safetensors$"],
                       dest="vae"),
        ],
        steps=20, cfg=3.5,
        diffusion_hint="qwen_image_fp8", clip_hint="qwen_2.5_vl_7b", vae_hint="qwen_image_vae",
        notes="Полноразмерная Qwen-Image. ТРЕБУЕТ ЧИСТОГО ПРОЦЕССА ComfyUI: "
              "поверх уже загруженных klein и krea2 она валит его нехваткой "
              "VRAM без внятного сообщения (снаружи виден только Connection "
              "refused посреди батча). На свежем процессе занимает 28 ГБ из 46 "
              "на A40. Варианта nvidia Qwen-Image-Flash в формате ComfyUI не "
              "существует, он опубликован только как diffusers; ближайший "
              "быстрый аналог — qwen-image-distill.",
    ),

    "qwen-image-distill": ModelPreset(
        id="qwen-image-distill",
        title="Qwen-Image distill fp8 (быстрый вариант)",
        clip_type="qwen_image",
        files=[
            WeightFile(repo="Comfy-Org/Qwen-Image_ComfyUI",
                       patterns=[r"non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn\.safetensors$"],
                       dest="diffusion_models"),
            WeightFile(repo="Comfy-Org/Qwen-Image_ComfyUI",
                       patterns=[r"split_files/text_encoders/qwen_2\.5_vl_7b_fp8_scaled\.safetensors$"],
                       dest="text_encoders"),
            WeightFile(repo="Comfy-Org/Qwen-Image_ComfyUI",
                       patterns=[r"split_files/vae/qwen_image_vae\.safetensors$"],
                       dest="vae"),
        ],
        steps=8, cfg=1.0, negatives_work=False,
        diffusion_hint="qwen_image_distill", clip_hint="qwen_2.5_vl_7b", vae_hint="qwen_image_vae",
        notes="Неофициальная дистилляция Qwen-Image из того же репозитория.",
    ),
}

# Не поддержано нашей ревизией ComfyUI — перечислено, чтобы не искать заново.
UNSUPPORTED = {
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo (Comfy-Org/z_image_turbo): "
                     "CLIPLoader.type не знает 'z_image' в текущей ревизии "
                     "ComfyUI. Нужна более свежая ревизия — отдельная пересборка.",
    "qwen-image-flash": "nvidia/Qwen-Image-Flash: опубликован только в формате "
                        "diffusers, переупаковки под ComfyUI нет. Нужна конвертация "
                        "либо использовать qwen-image-distill.",
}


def get(preset_id: str) -> ModelPreset:
    if preset_id not in PRESETS:
        hint = UNSUPPORTED.get(preset_id)
        if hint:
            raise SystemExit(f"модель '{preset_id}' не поддержана: {hint}")
        raise SystemExit(f"нет пресета '{preset_id}'. Доступны: {', '.join(sorted(PRESETS))}")
    return PRESETS[preset_id]
