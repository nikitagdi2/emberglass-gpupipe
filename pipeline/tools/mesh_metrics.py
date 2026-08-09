#!/usr/bin/env python3
"""Объективные метрики GLB для сравнения бэкендов image -> 3D.

Смотреть меши глазами в вьюере — вкусовщина. Здесь считается то, что
определяет пригодность для нашего пайплайна: сколько геометрии, есть ли UV
и текстура, замкнута ли поверхность, и какие пропорции габарита — последнее
важно, потому что запекание масштабирует модель по длине.

    python mesh_metrics.py artwork/mesh/*.glb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh


def analyse(path: Path) -> dict:
    scene = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        return {"file": path.name, "error": "в файле нет полигональной геометрии"}

    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    extents = mesh.extents
    order = sorted(extents, reverse=True)

    texture_side = 0
    has_uv = False
    for part in meshes:
        visual = getattr(part, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is not None and len(uv):
            has_uv = True
        material = getattr(visual, "material", None)
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if image is not None:
            texture_side = max(texture_side, max(image.size))

    return {
        "file": path.name,
        "parts": len(meshes),
        "tris": len(mesh.faces),
        "verts": len(mesh.vertices),
        "watertight": bool(mesh.is_watertight),
        "uv": has_uv,
        "texture": texture_side,
        # Пропорции в порядке убывания: запекание масштабирует по длинной оси,
        # поэтому важно соотношение, а не абсолютный размер.
        "proportions": tuple(round(float(v) / float(order[0]), 2) for v in order),
        "mb": round(path.stat().st_size / 2**20, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    # glob из stdlib, а не Path().glob: последний не принимает абсолютные
    # шаблоны, а пути к мешам приходят абсолютными.
    import glob as globmod

    rows = []
    for pattern in args.files:
        matches = sorted(globmod.glob(pattern)) or ([pattern] if Path(pattern).exists() else [])
        for name in matches:
            rows.append(analyse(Path(name)))

    if not rows:
        raise SystemExit("файлов не нашлось")

    print(f"{'файл':<40} {'частей':>6} {'треуг.':>9} {'замкн.':>7} {'UV':>4} "
          f"{'текстура':>9} {'пропорции':>18} {'МБ':>6}")
    for r in rows:
        if "error" in r:
            print(f"{r['file']:<40} {r['error']}")
            continue
        print(f"{r['file']:<40} {r['parts']:>6} {r['tris']:>9,} "
              f"{'да' if r['watertight'] else 'НЕТ':>7} {'да' if r['uv'] else 'НЕТ':>4} "
              f"{(str(r['texture']) + 'px') if r['texture'] else '—':>9} "
              f"{str(r['proportions']):>18} {r['mb']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
