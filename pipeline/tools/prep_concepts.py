#!/usr/bin/env python3
"""Подготовка концептов ко входу в image -> 3D.

Концепты приходят извне (внешний генератор), а реконструктор берёт картинку
как ОДИН объект. Лист с двумя проекциями он слепит в одну фигуру, поэтому
такие листы режутся на одиночные виды. Заодно кадр доводится до квадрата с
полями: объект, упирающийся в край, реконструируется хуже.

    python prep_concepts.py --src artwork/concepts --dst artwork/concepts3d
    python prep_concepts.py --src ... --dst ... --two-view mrc_trooper kln_trooper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops

SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def content_columns(image: Image.Image, tolerance: int, step: int = 4) -> list[int]:
    """Колонки, занятые объектом.

    Фон у концептов ровный, поэтому эталон берётся из угла кадра, а не
    угадывается порогом яркости: тёмный объект на тёмном фоне порог обманул бы.
    """
    small = image.convert("RGB").resize((max(1, image.width // step),
                                         max(1, image.height // step)))
    background = small.getpixel((1, 1))
    diff = ImageChops.difference(small, Image.new("RGB", small.size, background))
    mask = diff.convert("L").point(lambda v: 255 if v > tolerance else 0)
    pixels = mask.load()

    columns = []
    for x in range(mask.width):
        if any(pixels[x, y] for y in range(0, mask.height, 2)):
            columns.append(x * step)
    return columns


def split_two_views(image: Image.Image, tolerance: int,
                    min_gap_fraction: float) -> tuple[int, int, int] | None:
    """Граница между двумя фигурами — самый широкий разрыв занятых колонок.

    Разрыв должен быть заметным, иначе это просто щель внутри одной фигуры,
    например между рукой и корпусом.
    """
    columns = content_columns(image, tolerance)
    if not columns:
        return None

    widest, cut = 0, None
    for left, right in zip(columns, columns[1:]):
        if right - left > widest:
            widest, cut = right - left, (left + right) // 2

    if cut is None or widest < image.width * min_gap_fraction:
        return None
    return cut, columns[0], columns[-1]


def to_square(image: Image.Image, margin: float) -> Image.Image:
    side = int(max(image.size) * (1.0 + margin))
    canvas = Image.new("RGB", (side, side), image.getpixel((1, 1)))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--two-view", nargs="*", default=[],
                        help="имена файлов (без расширения) с двумя проекциями в кадре")
    parser.add_argument("--auto-two-view", action="store_true",
                        help="резать любой кадр, где нашёлся достаточно широкий разрыв")
    parser.add_argument("--tolerance", type=int, default=18,
                        help="порог отличия от фона")
    parser.add_argument("--min-gap", type=float, default=0.04,
                        help="минимальная ширина разрыва в долях кадра")
    parser.add_argument("--margin", type=float, default=0.08,
                        help="поля вокруг объекта в долях большей стороны")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"нет каталога с концептами: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    wanted = set(args.two_view)
    made = 0
    for path in sorted(p for p in src.iterdir() if p.suffix.lower() in SUFFIXES):
        image = Image.open(path).convert("RGB")
        crops: dict[str, Image.Image] = {"": image}

        if path.stem in wanted or args.auto_two_view:
            split = split_two_views(image, args.tolerance, args.min_gap)
            if split is None:
                print(f"{path.name}: разрыв не найден, оставляю целиком")
            else:
                cut, left, right = split
                pad = int(image.width * 0.02)
                crops = {
                    "_front": image.crop((max(0, left - pad), 0, cut, image.height)),
                    "_side": image.crop((cut, 0, min(image.width, right + pad), image.height)),
                }
                print(f"{path.name}: разрез на x={cut} (контент {left}..{right})")

        for suffix, crop in crops.items():
            out = dst / f"{path.stem}{suffix}.png"
            to_square(crop, args.margin).save(out)
            print(f"  -> {out.name}  {out.stat().st_size // 1024} КБ")
            made += 1

    print(f"\nподготовлено входов: {made}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
