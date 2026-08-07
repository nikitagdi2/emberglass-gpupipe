#!/usr/bin/env python3
"""Загрузка весов при старте пода.

Веса не пекутся в образ: 50+ ГБ раздули бы его так, что он тянулся бы дольше,
чем те же файлы с HuggingFace. hf_transfer даёт сотни МБ/с, то есть весь набор
приезжает за несколько минут GPU-времени.

Имена файлов у зеркал Comfy-Org расходятся между документацией и реальным
деревом репозитория, поэтому конкретный файл выбирается по факту из листинга,
а не задаётся константой.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


class Target:
    """Один файл, который надо достать: репозиторий, предпочтения, куда положить."""

    def __init__(self, name: str, repo: str, patterns: list[str], dest: Path,
                 gated: bool = False) -> None:
        self.name = name
        self.repo = repo
        self.patterns = patterns
        self.dest = dest
        self.gated = gated


def resolve(target: Target, token: str | None) -> str:
    """Первый файл репозитория, совпавший с самым приоритетным шаблоном."""
    try:
        files = list_repo_files(target.repo, token=token)
    except Exception as error:
        raise RuntimeError(
            f"{target.name}: не удалось прочитать {target.repo}: {error}"
            + ("\n  Репозиторий gated — примите условия на huggingface.co и задайте HF_TOKEN."
               if target.gated else "")
        ) from error

    for pattern in target.patterns:
        matched = sorted(f for f in files if re.search(pattern, f))
        if matched:
            return matched[0]

    raise RuntimeError(
        f"{target.name}: в {target.repo} нет файла под шаблоны {target.patterns}.\n"
        f"  Доступно: {sorted(f for f in files if f.endswith('.safetensors'))[:20]}"
    )


def fetch(target: Target, token: str | None) -> Path:
    target.dest.mkdir(parents=True, exist_ok=True)
    remote = resolve(target, token)
    local = target.dest / Path(remote).name

    if local.exists() and local.stat().st_size > 0:
        print(f"  [skip] {target.name}: {local.name} уже на месте")
        return local

    print(f"  [get ] {target.name}: {target.repo}/{remote}")
    path = hf_hub_download(repo_id=target.repo, filename=remote, token=token,
                           local_dir=str(target.dest))
    resolved = Path(path)
    if resolved != local and not local.exists():
        # hf_hub_download сохраняет вложенный путь репозитория; кладём файл плоско,
        # потому что ComfyUI ищет модели ровно в своей папке, без подкаталогов.
        local.parent.mkdir(parents=True, exist_ok=True)
        os.replace(resolved, local)
    return local


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", default=os.environ.get("COMFY", "/opt/ComfyUI"))
    parser.add_argument("--models", default="/workspace/models")
    parser.add_argument("--skip-trellis", action="store_true")
    parser.add_argument("--skip-flux", action="store_true")
    parser.add_argument("--preset", action="append", default=[],
                        help="пресет модели из pipeline/models.py; можно повторять")
    parser.add_argument("--pipe", default=os.environ.get("PIPE", "/opt/pipe"))
    args = parser.parse_args()

    if args.preset:
        sys.path.insert(0, args.pipe)
        import models as model_registry

        token_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        comfy_root = Path(args.comfy)
        problems: list[str] = []
        for preset_id in args.preset:
            preset = model_registry.get(preset_id)
            print(f"== {preset.title} ==")
            for spec in preset.files:
                target = Target(f"{preset.id}/{spec.dest}", spec.repo, spec.patterns,
                                comfy_root / "models" / spec.dest, gated=spec.gated)
                try:
                    fetch(target, token_env)
                except RuntimeError as error:
                    print(f"  [FAIL] {error}", file=sys.stderr)
                    problems.append(target.name)
        if problems:
            print(f"\nНЕ загружено: {', '.join(problems)}", file=sys.stderr)
            return 1
        print("\nвеса пресетов готовы")
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    comfy = Path(args.comfy)
    models = Path(args.models)

    # Базовый 9B, а НЕ дистиллированный. У дистиллированного CFG обязан быть 1.0,
    # а при CFG=1 ComfyUI пропускает безусловный проход, и негативные промпты
    # не действуют вообще. Все промпты арт-библии на негативах и построены.
    flux_targets = [
        Target("FLUX.2 klein base 9B fp8",
               os.environ.get("FLUX_REPO", "black-forest-labs/FLUX.2-klein-base-9b-fp8"),
               [r"klein.*base.*9b.*fp8.*\.safetensors$", r".*\.safetensors$"],
               comfy / "models" / "diffusion_models",
               gated=True),
        Target("Qwen3-8B text encoder",
               "Comfy-Org/flux2-klein-9B",
               [r"text_encoders/.*qwen_3_8b.*fp8.*\.safetensors$",
                r"text_encoders/.*qwen_3_8b.*\.safetensors$",
                r".*qwen_3_8b.*\.safetensors$"],
               comfy / "models" / "text_encoders"),
        Target("FLUX.2 VAE",
               "Comfy-Org/flux2-dev",
               [r"vae/flux2-vae\.safetensors$", r".*flux2-vae\.safetensors$"],
               comfy / "models" / "vae"),
    ]

    failures: list[str] = []

    if not args.skip_flux:
        print("== FLUX.2 klein ==")
        for target in flux_targets:
            try:
                fetch(target, token)
            except RuntimeError as error:
                print(f"  [FAIL] {error}", file=sys.stderr)
                failures.append(target.name)

    if not args.skip_trellis:
        print("== TRELLIS.2-4B ==")
        trellis_dir = models / "trellis2"
        if trellis_dir.exists() and any(trellis_dir.iterdir()):
            print(f"  [skip] уже на месте: {trellis_dir}")
        else:
            try:
                snapshot_download(repo_id=os.environ.get("TRELLIS_REPO", "microsoft/TRELLIS.2-4B"),
                                  local_dir=str(trellis_dir), token=token)
            except Exception as error:
                print(f"  [FAIL] TRELLIS.2: {error}", file=sys.stderr)
                failures.append("TRELLIS.2-4B")

    if failures:
        print(f"\nНЕ загружено: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("\nвеса готовы")
    return 0


if __name__ == "__main__":
    sys.exit(main())
