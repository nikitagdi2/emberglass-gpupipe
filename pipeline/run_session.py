#!/usr/bin/env python3
"""Драйвер сессии на поде.

Батч готовится заранее и запускается одной командой: карта не должна ждать,
пока человек думает. Все этапы идемпотентны — уже существующие файлы
пропускаются, поэтому прерванный под (Community Cloud) продолжается с того же
места, а не начинается заново.

    python run_session.py doctor
    python run_session.py concepts --seeds 4
    python run_session.py sheets
    python run_session.py mesh --pick picks.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import models  # noqa: E402
from comfy_client import ComfyClient, ComfyError, SamplerSettings, build_txt2img_graph  # noqa: E402
from sheets import contact_sheet  # noqa: E402


def load_prompts(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def select_units(data: dict, only: list[str] | None, faction: str | None,
                 unit_class: str | None) -> list[dict]:
    units = data["units"]
    if only:
        wanted = set(only)
        units = [u for u in units if u["id"] in wanted]
        missing = wanted - {u["id"] for u in units}
        if missing:
            raise SystemExit(f"нет таких юнитов в промптах: {sorted(missing)}")
    if faction:
        units = [u for u in units if u["faction"].upper() == faction.upper()]
    if unit_class:
        units = [u for u in units if u["class"] == unit_class]
    return units


def compose_positive(entry: dict, data: dict) -> str:
    """Позитивный промпт, собранный из частей в порядке parts_order.

    Порядок задан в файле промптов один раз: у диффузии ранние токены весят
    больше, поэтому первыми идут предмет и жёсткие требования, а декоративные
    детали — последними. Общие части (кадрирование, стиль) берутся из shared,
    причём кадрирование зависит от класса: пехоте нужны два вида, технике один.
    """
    shared = data.get("shared", {})
    order = data.get("meta", {}).get("parts_order",
                                     ["subject", "hard", "framing", "style", "palette", "detail"])

    # Кадрирование зависит от вида записи: пехоте нужны два вида, технике один,
    # зданиям свой. У зданий поля class нет — они описаны футпринтом, поэтому
    # ключ подбирается по нескольким признакам, а не по одному.
    candidates = []
    if entry.get("class"):
        candidates.append(f"framing_{entry['class']}")
    if entry.get("footprint"):
        candidates.append("framing_building")
    candidates += ["framing_vehicle", "framing_building"]

    framing = next((shared[k] for k in candidates if shared.get(k)), "")
    if not framing:
        framing = next((v for k, v in shared.items() if k.startswith("framing")), "")

    parts = {
        "framing": framing,
        "style": shared.get("style", ""),
    }
    for key in ("subject", "hard", "palette", "detail"):
        parts[key] = entry.get(key, "")

    return " ".join(filter(None, (parts.get(name, "") for name in order)))


def resolve_prompt(entry: dict, data: dict, style: str) -> tuple[str, str] | None:
    """Промпт в запрошенном стиле; None — если такого варианта у записи нет.

    Негатив не используется никогда и всегда пуст. Причина установлена
    экспериментом: у моделей с cfg<=1 ComfyUI пропускает безусловный проход и
    негатив не действует вообще, а у qwen-image он формально действует, но
    приор модели оказывается сильнее запрета. Всё, что раньше было негативом,
    живёт в части `hard` утвердительной формулировкой.

    Стиль `gdd` — дословный текст §11.2, который восстанавливает
    sync_gdd_prompts.py; он существует только для сравнения и содержит
    исходную SDXL-редакцию.
    """
    if style == "gdd":
        positive = entry.get("positive_gdd")
        if not positive:
            return None
        return positive, ""

    return compose_positive(entry, data), ""


def select_presets(ids: list[str]) -> list[models.ModelPreset]:
    """Пресеты в порядке запуска, с проверкой отклонённых и порядка загрузки."""
    presets = [models.get(i) for i in ids]

    for preset in presets:
        if preset.retired:
            raise SystemExit(f"пресет '{preset.id}' отклонён: {preset.retired}")

    return presets


def resolve_model_names(client: ComfyClient, preset: models.ModelPreset) -> dict:
    """Имена файлов берём из живой схемы по подсказкам пресета, а не из констант."""
    def pick(class_type: str, input_name: str, needle: str, role: str) -> str:
        options = [str(v) for v in client.enum_values(class_type, input_name)]
        if not options:
            raise ComfyError(f"{role}: {class_type}.{input_name} пуст — веса не загружены?")
        match = next((o for o in options if needle.lower() in o.lower()), None)
        if match is None:
            raise ComfyError(
                f"{role}: файла с '{needle}' нет среди {options}. "
                f"Загрузить веса пресета: fetch_weights.py --preset {preset.id}"
            )
        return match

    diffusion_cls = client.pick_class(["UNETLoader", "DiffusionModelLoader"], "диффузия")
    clip_cls = client.pick_class(["CLIPLoader"], "энкодер")
    vae_cls = client.pick_class(["VAELoader"], "vae")

    return {
        "diffusion": pick(diffusion_cls, "unet_name", preset.diffusion_hint, "диффузия"),
        "clip": pick(clip_cls, "clip_name", preset.clip_hint, "текстовый энкодер"),
        "vae": pick(vae_cls, "vae_name", preset.vae_hint, "vae"),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    print("== окружение ==")
    try:
        import torch
        print(f"  torch {torch.__version__}, cuda={torch.version.cuda}, "
              f"доступна={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  карта: {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} ГБ")
    except Exception as error:
        print(f"  torch недоступен: {error}")
    print(f"  ATTN_BACKEND={os.environ.get('ATTN_BACKEND', '(не задан)')}")

    print("== ComfyUI ==")
    client = ComfyClient(args.comfy_url)
    try:
        schema = client.object_info()
        print(f"  схема получена, классов нод: {len(schema)}")
    except Exception as error:
        print(f"  НЕДОСТУПЕН: {error}")
        return 1

    preset = select_presets(args.model)[0]
    print(f"== пресет {preset.id}: {preset.title} ==")
    if not preset.negatives_work:
        print("  cfg<=1 по контракту модели — негативные промпты не действуют")

    try:
        names = resolve_model_names(client, preset)
        for role, value in names.items():
            print(f"  {role}: {value}")
    except ComfyError as error:
        print(f"  веса: {error}")
        return 1

    print("== граф ==")
    try:
        graph = build_txt2img_graph(
            client, positive="probe", negative="probe", width=512, height=512, seed=0,
            sampler=SamplerSettings(steps=preset.steps, cfg=preset.cfg,
                                    allow_low_cfg=not preset.negatives_work),
            diffusion_name=names["diffusion"], clip_name=names["clip"],
            vae_name=names["vae"], clip_type=preset.clip_type, filename_prefix="probe")
    except ComfyError as error:
        print(f"  собрать не удалось: {error}")
        return 1

    problems = client.validate_graph(graph)
    if problems:
        print("  РАСХОЖДЕНИЯ СО СХЕМОЙ:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  граф валиден")
    print("\nготов к батчу")
    return 0


def cmd_concepts(args: argparse.Namespace) -> int:
    data = load_prompts(Path(args.prompts))
    units = select_units(data, args.only, args.faction, args.unit_class)
    if not units:
        raise SystemExit("под фильтры не попал ни один юнит")

    presets = select_presets(args.model)
    client = ComfyClient(args.comfy_url)
    out_root = Path(args.out)
    styles = args.prompt_style

    resolved = [(unit, style, resolve_prompt(unit, data, style))
                for unit in units for style in styles]
    skipped = [f"{u['id']}/{s}" for u, s, p in resolved if p is None]
    work = [(u, s, p) for u, s, p in resolved if p is not None]
    if skipped:
        print(f"без варианта промпта, пропущены: {', '.join(skipped)}")

    planned = len(work) * args.seeds * len(presets)
    print(f"юнитов {len(units)} x стилей {len(styles)} x сидов {args.seeds} "
          f"x моделей {len(presets)} = {planned} кадров")

    done = 0
    for index, preset in enumerate(presets):
        # Перед сменой семейства освобождаем VRAM. Без этого тяжёлая модель,
        # загруженная поверх предыдущих, роняет ComfyUI без сообщения об OOM.
        if index > 0:
            if client.free():
                print("  VRAM освобождена перед сменой модели")
            else:
                print("  ВНИМАНИЕ: /free недоступен, VRAM не освобождена. "
                      "Если упадёт — ставь тяжёлую модель первой "
                      f"(порядок по весу: "
                      f"{' '.join(p.id for p in sorted(presets, key=lambda p: -p.vram_gb))})")

        names = resolve_model_names(client, preset)

        # Параметры сэмплера берутся из пресета; явные ключи их перекрывают.
        steps = args.steps if args.steps is not None else preset.steps
        cfg = args.cfg if args.cfg is not None else preset.cfg
        sampler = SamplerSettings(steps=steps, cfg=cfg,
                                  allow_low_cfg=args.allow_cfg1 or not preset.negatives_work)
        sampler.validate()

        print(f"\n== {preset.title} [{preset.clip_type}] "
              f"cfg={sampler.cfg} steps={sampler.steps} ==")
        print(f"  файлы: {names['diffusion']} / {names['clip']} / {names['vae']}")

        for unit, style, (positive, negative) in work:
            width, height = unit["sheet_px"]

            for seed_index in range(args.seeds):
                seed = args.seed_base + seed_index
                # Метка модели и стиля — в имени файла и в префиксе ComfyUI:
                # иначе в его выводе не отличить, чем сгенерирован кадр.
                dest = (out_root / unit["id"] /
                        f"{unit['id']}_{preset.id}_{style}_s{seed}.png")
                if dest.exists() and dest.stat().st_size > 0:
                    print(f"  [skip] {dest.name}")
                    done += 1
                    continue

                graph = build_txt2img_graph(
                    client, positive=positive, negative=negative,
                    width=width, height=height, seed=seed, sampler=sampler,
                    diffusion_name=names["diffusion"], clip_name=names["clip"],
                    vae_name=names["vae"], clip_type=preset.clip_type,
                    filename_prefix=f"emberglass/{unit['id']}_{preset.id}_{style}")

                problems = client.validate_graph(graph)
                if problems:
                    raise SystemExit("граф разошёлся со схемой:\n  " + "\n  ".join(problems))

                print(f"  [gen ] {unit['id']} [{style}] seed={seed} {width}x{height}")
                outputs = client.wait(client.submit(graph), timeout_seconds=args.timeout)

                images = [img for node in outputs.values() for img in node.get("images", [])]
                if not images:
                    print(f"  [WARN] {unit['id']} [{style}] seed={seed}: изображений нет")
                    continue
                client.download_image(images[0], dest)
                done += 1

    print(f"\nготово: {done}/{planned}")
    return 0


def cmd_sheets(args: argparse.Namespace) -> int:
    data = load_prompts(Path(args.prompts))
    out_root = Path(args.out)
    sheet_dir = Path(args.sheets)

    made = 0
    for unit in data["units"]:
        unit_dir = out_root / unit["id"]
        images = sorted(unit_dir.glob("*.png")) if unit_dir.exists() else []
        if not images:
            continue
        contact_sheet([(p.stem, p) for p in images], sheet_dir / f"{unit['id']}.png",
                      columns=args.columns)
        made += 1

    everything = []
    for unit in data["units"]:
        unit_dir = out_root / unit["id"]
        first = sorted(unit_dir.glob("*.png"))[:1] if unit_dir.exists() else []
        if first:
            everything.append((unit["id"], first[0]))
    if everything:
        contact_sheet(everything, sheet_dir / "_roster.png", columns=args.columns)

    print(f"листов собрано: {made} (+ сводный _roster.png)" if made else "нечего собирать")
    return 0


def short_error(error: BaseException) -> str:
    """Суть ошибки одной строкой.

    Трейсбеки моделей уходят на десятки строк и топят собой протокол прогона,
    а различимая часть у них обычно в конце.
    """
    text = " ".join(str(error).split())
    if "gated repo" in text or "not in the authorized list" in text:
        import re
        match = re.search(r"huggingface\.co/([\w\-./]+)", text)
        repo = match.group(1).split("/resolve")[0] if match else "?"
        return (f"репозиторий {repo} gated — примите условия на "
                f"huggingface.co/{repo} тем аккаунтом, чей HF_TOKEN на поде")
    return text[:300]


def cmd_mesh(args: argparse.Namespace) -> int:
    import mesh3d

    # Картинки не обязаны рождаться на поде: концепты может принести человек,
    # отобрав их снаружи. Поэтому источник задаётся явно.
    picks: list[Path] = []
    if args.pick:
        for line in Path(args.pick).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                picks.append(Path(line))
    else:
        source = Path(args.source) if args.source else Path(args.out)
        if not source.exists():
            raise SystemExit(f"каталог с картинками не найден: {source}")
        picks = sorted(p for p in source.rglob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))

    if not picks:
        raise SystemExit("нет входных изображений")

    backends = [mesh3d.get(b) for b in args.backend]
    settings = mesh3d.MeshSettings(decimation_target=args.decimation,
                                   texture_size=args.texture,
                                   resolution=args.resolution,
                                   low_vram=args.low_vram,
                                   seed=args.seed_base)

    mesh_dir = Path(args.mesh_out)
    print(f"изображений {len(picks)} x бэкендов {len(backends)} = "
          f"{len(picks) * len(backends)} мешей")

    failures = 0
    for backend in backends:
        print(f"\n== {backend.title} [{backend.license}] ==")
        # Падение одного бэкенда не должно уносить остальные: сравнительный
        # прогон идёт на оплачиваемой карте, и потерять из-за него второй
        # бэкенд — значит платить за сессию дважды.
        try:
            runner = mesh3d.make_runner(backend, os.environ.get("ATTN_BACKEND"))
        except BaseException as error:  # noqa: BLE001 - см. выше
            print(f"  [FAIL] бэкенд не поднялся: {short_error(error)}")
            failures += 1
            continue

        for image_path in picks:
            # Метка бэкенда в имени: иначе сравнить два меша от одной картинки
            # будет нечем.
            out_glb = mesh_dir / f"{image_path.stem}__{backend.id}.glb"
            if out_glb.exists() and out_glb.stat().st_size > 0:
                print(f"  [skip] {out_glb.name}")
                continue
            try:
                runner.run(image_path, out_glb, settings)
                print(f"  [ok  ] {out_glb.name}  {out_glb.stat().st_size / 2**20:.1f} МБ")
            except BaseException as error:  # noqa: BLE001 - см. выше
                print(f"  [FAIL] {image_path.name}: {short_error(error)}")
                failures += 1

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Сессия генерации ассетов EMBERGLASS")
    parser.add_argument("--comfy-url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    parser.add_argument("--prompts", default=str(Path(__file__).parent / "prompts" / "units.json"))
    parser.add_argument("--out", default="/workspace/output/concepts")
    parser.add_argument("--sheets", default="/workspace/output/sheets")
    parser.add_argument("--mesh-out", default="/workspace/output/mesh")
    parser.add_argument("--seed-base", type=int, default=20260803)
    parser.add_argument("--model", nargs="+", default=["klein-9b"],
                        choices=sorted(models.PRESETS),
                        help="пресеты моделей; несколько — прогон по каждому, "
                             "VRAM освобождается между ними")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="проверить окружение, веса и валидность графа")

    concepts = sub.add_parser("concepts", help="батч концептов")
    concepts.add_argument("--seeds", type=int, default=4)
    concepts.add_argument("--steps", type=int, default=None, help="перекрывает значение пресета")
    concepts.add_argument("--cfg", type=float, default=None, help="перекрывает значение пресета")
    concepts.add_argument("--only", nargs="*")
    concepts.add_argument("--faction", choices=["MRC", "KLN"])
    concepts.add_argument("--unit-class", dest="unit_class")
    concepts.add_argument("--timeout", type=float, default=1800.0)
    concepts.add_argument("--allow-cfg1", action="store_true",
                          help="разрешить cfg<=1 вручную; негативы при этом не действуют")
    concepts.add_argument(
        "--prompt-style", nargs="+", choices=["flux", "gdd"], default=["flux"],
        help="flux — переписанный под Qwen3 (по умолчанию); gdd — дословный §11.2/§11.3. "
             "Указать оба — прогон A/B на одних сидах")

    sheets = sub.add_parser("sheets", help="контактные листы для отбора")
    sheets.add_argument("--columns", type=int, default=4)

    mesh = sub.add_parser("mesh", help="image -> 3D через TRELLIS.2")
    mesh.add_argument("--pick", help="файл со списком путей к отобранным картинкам")
    mesh.add_argument("--source", help="каталог с картинками; по умолчанию вывод concepts")
    # Имя --backend, а не --model: верхнеуровневый --model уже занят пресетами
    # 2D-моделей, и одинаковые имена argparse разруливает неочевидно.
    mesh.add_argument("--backend", nargs="+", default=["trellis2"],
                      help="бэкенды image->3D; несколько — сравнительный прогон")
    mesh.add_argument("--decimation", type=int, default=200_000)
    mesh.add_argument("--texture", type=int, default=2048)
    mesh.add_argument("--resolution", type=int, default=1024)
    mesh.add_argument("--low-vram", action="store_true", dest="low_vram")

    args = parser.parse_args()
    return {
        "doctor": cmd_doctor,
        "concepts": cmd_concepts,
        "sheets": cmd_sheets,
        "mesh": cmd_mesh,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
