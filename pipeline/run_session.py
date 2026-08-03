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

from comfy import ComfyClient, ComfyError, SamplerSettings, build_flux2_graph  # noqa: E402
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


def resolve_prompt(entry: dict, style: str, shared_negative: str) -> tuple[str, str] | None:
    """Промпт в запрошенном стиле; None — если такого варианта у записи нет.

    Стиль `gdd` — дословный текст §11.2/§11.3, извлечённый скриптом
    sync_gdd_prompts.py. У юнитов, дописанных нами, исходника в GDD нет,
    поэтому в режиме сравнения они молча не участвуют — сравнивать не с чем.
    """
    if style == "gdd":
        positive = entry.get("positive_gdd")
        negative = entry.get("negative_gdd", "")
        if not positive:
            return None
    else:
        positive = entry["positive"]
        negative = entry.get("negative", "")

    return positive, ", ".join(filter(None, [negative, shared_negative]))


def resolve_model_names(client: ComfyClient) -> dict:
    """Имена файлов моделей берём из живой схемы, а не из констант."""
    def pick(class_type: str, input_name: str, needles: list[str], role: str) -> str:
        options = [str(v) for v in client.enum_values(class_type, input_name)]
        if not options:
            raise ComfyError(f"{role}: {class_type}.{input_name} пуст — веса не загружены?")
        for needle in needles:
            match = next((o for o in options if needle in o.lower()), None)
            if match:
                return match
        raise ComfyError(f"{role}: среди {options} нет ничего похожего на {needles}")

    diffusion_cls = client.pick_class(["UNETLoader", "DiffusionModelLoader"], "диффузия")
    clip_cls = client.pick_class(["CLIPLoader"], "энкодер")
    vae_cls = client.pick_class(["VAELoader"], "vae")

    return {
        "diffusion": pick(diffusion_cls, "unet_name", ["klein-base", "klein", "flux"], "диффузия"),
        "clip": pick(clip_cls, "clip_name", ["qwen_3_8b", "qwen"], "текстовый энкодер"),
        "vae": pick(vae_cls, "vae_name", ["flux2-vae", "flux2"], "vae"),
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

    try:
        names = resolve_model_names(client)
        for role, value in names.items():
            print(f"  {role}: {value}")
    except ComfyError as error:
        print(f"  веса: {error}")
        return 1

    print("== граф FLUX.2 ==")
    try:
        graph = build_flux2_graph(
            client, positive="probe", negative="probe", width=512, height=512, seed=0,
            sampler=SamplerSettings(), diffusion_name=names["diffusion"],
            clip_name=names["clip"], vae_name=names["vae"], filename_prefix="probe")
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

    client = ComfyClient(args.comfy_url)
    names = resolve_model_names(client)
    sampler = SamplerSettings(steps=args.steps, cfg=args.cfg)
    sampler.validate()

    out_root = Path(args.out)
    shared_negative = data.get("shared_negative", "")

    styles = args.prompt_style
    resolved = [(unit, style, resolve_prompt(unit, style, shared_negative))
                for unit in units for style in styles]
    skipped = [f"{u['id']}/{s}" for u, s, p in resolved if p is None]
    work = [(u, s, p) for u, s, p in resolved if p is not None]

    planned = len(work) * args.seeds
    print(f"юнитов {len(units)} x стилей {len(styles)} x сидов {args.seeds} = {planned} кадров")
    print(f"модель: {names['diffusion']}, cfg={sampler.cfg}, steps={sampler.steps}")
    if skipped:
        print(f"без варианта промпта, пропущены: {', '.join(skipped)}")

    done = 0
    for unit, style, (positive, negative) in work:
        width, height = unit["sheet_px"]

        for seed_index in range(args.seeds):
            seed = args.seed_base + seed_index
            # Стиль в имени файла: оба варианта ложатся в одну папку и
            # попадают на общий контактный лист рядом — так их и сравнивают.
            dest = out_root / unit["id"] / f"{unit['id']}_{style}_s{seed}.png"
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  [skip] {dest.name}")
                done += 1
                continue

            graph = build_flux2_graph(
                client, positive=positive, negative=negative,
                width=width, height=height, seed=seed, sampler=sampler,
                diffusion_name=names["diffusion"], clip_name=names["clip"],
                vae_name=names["vae"], filename_prefix=f"emberglass/{unit['id']}")

            problems = client.validate_graph(graph)
            if problems:
                raise SystemExit("граф разошёлся со схемой:\n  " + "\n  ".join(problems))

            print(f"  [gen ] {unit['id']} [{style}] seed={seed} {width}x{height}")
            outputs = client.wait(client.submit(graph), timeout_seconds=args.timeout)

            images = [img for node in outputs.values() for img in node.get("images", [])]
            if not images:
                print(f"  [WARN] {unit['id']} [{style}] seed={seed}: сервер не вернул изображений")
                continue
            client.download_image(images[0], dest)
            done += 1

    print(f"готово: {done}/{planned}")
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


def cmd_mesh(args: argparse.Namespace) -> int:
    from trellis3d import MeshSettings, Trellis2Runner

    picks: list[Path] = []
    if args.pick:
        for line in Path(args.pick).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                picks.append(Path(line))
    else:
        picks = sorted(Path(args.out).rglob("*.png"))

    if not picks:
        raise SystemExit("нет входных изображений")

    runner = Trellis2Runner(model_id=args.model,
                            attn_backend=os.environ.get("ATTN_BACKEND"))
    settings = MeshSettings(decimation_target=args.decimation,
                            texture_size=args.texture,
                            seed=args.seed_base)

    mesh_dir = Path(args.mesh_out)
    print(f"изображений на вход: {len(picks)}")
    for image_path in picks:
        runner.run(image_path, mesh_dir / f"{image_path.stem}.glb", settings)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Сессия генерации ассетов EMBERGLASS")
    parser.add_argument("--comfy-url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    parser.add_argument("--prompts", default=str(Path(__file__).parent / "prompts" / "units.json"))
    parser.add_argument("--out", default="/workspace/output/concepts")
    parser.add_argument("--sheets", default="/workspace/output/sheets")
    parser.add_argument("--mesh-out", default="/workspace/output/mesh")
    parser.add_argument("--seed-base", type=int, default=20260803)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="проверить окружение, веса и валидность графа")

    concepts = sub.add_parser("concepts", help="батч концептов через FLUX.2 klein")
    concepts.add_argument("--seeds", type=int, default=4)
    concepts.add_argument("--steps", type=int, default=22)
    concepts.add_argument("--cfg", type=float, default=4.0)
    concepts.add_argument("--only", nargs="*")
    concepts.add_argument("--faction", choices=["MRC", "KLN"])
    concepts.add_argument("--unit-class", dest="unit_class")
    concepts.add_argument("--timeout", type=float, default=1800.0)
    concepts.add_argument(
        "--prompt-style", nargs="+", choices=["flux", "gdd"], default=["flux"],
        help="flux — переписанный под Qwen3 (по умолчанию); gdd — дословный §11.2/§11.3. "
             "Указать оба — прогон A/B на одних сидах")

    sheets = sub.add_parser("sheets", help="контактные листы для отбора")
    sheets.add_argument("--columns", type=int, default=4)

    mesh = sub.add_parser("mesh", help="image -> 3D через TRELLIS.2")
    mesh.add_argument("--pick", help="файл со списком путей к отобранным картинкам")
    mesh.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    mesh.add_argument("--decimation", type=int, default=200_000)
    mesh.add_argument("--texture", type=int, default=2048)

    args = parser.parse_args()
    return {
        "doctor": cmd_doctor,
        "concepts": cmd_concepts,
        "sheets": cmd_sheets,
        "mesh": cmd_mesh,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
