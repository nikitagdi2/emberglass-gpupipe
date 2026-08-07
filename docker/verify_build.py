#!/usr/bin/env python3
"""Проверка образа на этапе сборки, где GPU нет по замыслу.

Часть пакетов (o_voxel, flex_gemm, xformers.ops) регистрирует автотюнеры Triton
прямо при импорте, а Triton без активного драйвера падает с
"0 active drivers". Поэтому такие модули проверяются наличием через find_spec —
он находит модуль, не исполняя его, — а фактические импорты выполняются уже на
поде командой `run_session.py doctor`.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

# Импортируются на любой машине: драйвер им не нужен.
# torchaudio здесь не для полноты: он подгружает нативную библиотеку и падает,
# если приехал сборкой под другую CUDA, чем torch. Именно так ComfyUI однажды
# и слёг на поде — проверка обязана ловить это на сборке.
IMPORTABLE = ("torch", "torchvision", "torchaudio", "xformers", "trimesh",
              "nvdiffrast", "PIL", "requests")

# Требуют драйвер при импорте — проверяем только наличие.
# Кортеж = допустимые имена модуля: пакеты ставятся под разными именами
# (nvdiffrec с ветки renderutils регистрируется как nvdiffrec_render).
PRESENT_ONLY: tuple[tuple[str, ...], ...] = (
    ("o_voxel",),
    ("cumesh",),
    ("flex_gemm",),
    ("nvdiffrec_render", "nvdiffrec", "renderutils"),
)


def main() -> int:
    failures: list[str] = []

    for name in IMPORTABLE:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "—")
            print(f"  [import] {name:<12} {version}")
        except Exception as error:
            print(f"  [FAIL  ] {name}: {error}", file=sys.stderr)
            failures.append(name)

    for candidates in PRESENT_ONLY:
        found = None
        for name in candidates:
            try:
                spec = importlib.util.find_spec(name)
            except Exception:
                spec = None
            if spec is not None:
                found = (name, spec.origin)
                break
        if found is None:
            print(f"  [FAIL  ] {' | '.join(candidates)}: модуль не найден", file=sys.stderr)
            failures.append(candidates[0])
        else:
            print(f"  [present] {found[0]:<11} {found[1]}")

    try:
        import torch
        print(f"\n  torch {torch.__version__}, cuda {torch.version.cuda}")
        print(f"  arch list: {torch.cuda.get_arch_list()}")
    except Exception as error:
        print(f"  torch не отдал сведения: {error}", file=sys.stderr)

    if failures:
        print(f"\nПРОВЕРКА НЕ ПРОЙДЕНА: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("\nобраз собран, все компоненты на месте")
    return 0


if __name__ == "__main__":
    sys.exit(main())
