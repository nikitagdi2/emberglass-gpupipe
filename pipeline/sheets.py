"""Контактные листы для отбора.

Смысл этапа: домой уезжает не сорок отдельных картинок и мешей, которые надо
открывать по одному, а сетка превью. Отбор превращается в разглядывание одного
файла вместо часа возни. Рендерится на поде, пока карта всё равно оплачена.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def contact_sheet(images: list[tuple[str, Path]], out_path: Path, *, columns: int = 4,
                  cell: int = 320, background: tuple = (58, 58, 62),
                  label_height: int = 18) -> Path:
    """Сетка превью с подписями. Пропущенные файлы отмечаются, а не роняют лист."""
    if not images:
        raise ValueError("нечего собирать: пустой список")

    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), background)
    draw = ImageDraw.Draw(sheet)

    for index, (label, path) in enumerate(images):
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)

        if path.exists():
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((cell, cell), Image.LANCZOS)
                sheet.paste(thumb, (x + (cell - thumb.width) // 2,
                                    y + label_height + (cell - thumb.height) // 2))
        else:
            draw.text((x + 8, y + label_height + cell // 2), "— нет файла —", fill=(200, 90, 90))

        draw.text((x + 6, y + 4), label[:48], fill=(230, 230, 230))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
