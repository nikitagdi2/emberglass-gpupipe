#!/usr/bin/env python3
"""Переносит дословные промпты из GDD в файлы промптов как альтернативный стиль.

Нужен для честного A/B: промпты §11.2/§11.3 писались под CLIP (SDXL), рабочие
переписаны под текстовый энкодер Qwen3. Сравнивать их имеет смысл только если
исходный текст взят буквально, поэтому он извлекается из GDD скриптом, а не
переписывается руками.

    python sync_gdd_prompts.py --gdd ../../../docs/game/design/gdd.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Строка таблицы: | `id` | тип | лист | кадры | `позитив` | `негатив` |
ROW = re.compile(r"^\|\s*`(?P<id>[a-z0-9_]+)`\s*\|")


def parse_section(text: str, heading: str) -> dict[str, dict[str, str]]:
    """Промпты из markdown-таблицы указанного раздела GDD."""
    start = text.find(heading)
    if start < 0:
        raise SystemExit(f"в GDD нет раздела {heading!r}")

    # До следующего заголовка того же уровня.
    rest = text[start + len(heading):]
    end = rest.find("\n### ")
    body = rest if end < 0 else rest[:end]

    found: dict[str, dict[str, str]] = {}
    for line in body.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7:
            continue
        positive = cells[5].strip().strip("`").strip()
        negative = cells[6].strip().strip("`").strip()
        if not positive:
            continue
        found[match.group("id")] = {"positive": positive, "negative": negative}
    return found


def apply(prompts_path: Path, key: str, extracted: dict[str, dict[str, str]]) -> tuple[int, list[str]]:
    with open(prompts_path, encoding="utf-8") as handle:
        data = json.load(handle)

    updated = 0
    missing: list[str] = []
    for entry in data[key]:
        source = extracted.get(entry["id"])
        if source is None:
            # Юниты, дописанные нами: в GDD их нет, сравнивать не с чем.
            if entry.get("origin") == "gdd":
                missing.append(entry["id"])
            continue
        entry["positive_gdd"] = source["positive"]
        entry["negative_gdd"] = source["negative"]
        updated += 1

    with open(prompts_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return updated, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdd", required=True, help="путь к docs/game/design/gdd.md")
    parser.add_argument("--prompts-dir", default=str(Path(__file__).resolve().parent.parent / "prompts"))
    args = parser.parse_args()

    text = Path(args.gdd).read_text(encoding="utf-8")
    prompts_dir = Path(args.prompts_dir)

    plan = [
        ("### 11.2 Юниты", prompts_dir / "units.json", "units"),
        ("### 11.3 Здания", prompts_dir / "buildings.json", "buildings"),
    ]

    problems = 0
    for heading, path, key in plan:
        extracted = parse_section(text, heading)
        updated, missing = apply(path, key, extracted)
        print(f"{path.name}: извлечено из GDD {len(extracted)}, проставлено {updated}")
        if missing:
            print(f"  ВНИМАНИЕ, помечены origin=gdd, но в таблице не найдены: {missing}")
            problems += 1

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
