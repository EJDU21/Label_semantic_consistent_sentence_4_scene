#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

# Reuse the single-scene implementation
from label_scene import (
    DEFAULT_PROMPT,
    compute_clip_scores,
    generate_caption_openai,
    infer_scene_content_dir,
    iter_rgb_images,
    read_captions_jsonl,
    append_caption_jsonl,
    CaptionItem,
)


def iter_scene_dirs(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.glob("scene_*") if p.is_dir()])


def parse_skip_scenes(value: str) -> set[str]:
    # comma-separated: "scene_12,scene_03"
    items = [v.strip() for v in value.split(",")]
    return {v for v in items if v}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch label multiple scenes under a dataset root (scene_*)."
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing scene_* directories (e.g. /media/iverson/KINGSTON/dataset).",
    )
    ap.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Optional comma-separated list of scene ids to run (e.g. scene_01,scene_02). Default: all scene_* under dataset-root.",
    )
    ap.add_argument(
        "--skip-scenes",
        type=str,
        default="scene_12",
        help='Comma-separated list of scenes to skip. Default: "scene_12". Use empty string to skip none.',
    )

    ap.add_argument("--openai-model", type=str, default="gpt-5.5-pro-2026-04-23")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--skip-gpt", action="store_true", help="Skip GPT captioning; requires per-scene captions cache exists.")

    ap.add_argument("--clip-model", type=str, default="ViT-L-14")
    ap.add_argument("--clip-pretrained", type=str, default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--write-details", action="store_true")

    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    all_scenes = iter_scene_dirs(dataset_root)
    if not all_scenes:
        raise SystemExit(f"no scene_* directories found under: {dataset_root}")

    selected = None
    if args.scenes:
        want = parse_skip_scenes(args.scenes)
        selected = [p for p in all_scenes if p.name in want]
    else:
        selected = all_scenes

    skip = parse_skip_scenes(args.skip_scenes) if args.skip_scenes else set()
    selected = [p for p in selected if p.name not in skip]

    for scene_root in tqdm(selected, desc="Scenes"):
        scene_name = scene_root.name
        out_dir = infer_scene_content_dir(scene_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        captions_path = out_dir / f"{scene_name}.captions.jsonl"
        captions = read_captions_jsonl(captions_path)

        image_paths = iter_rgb_images(scene_root)
        if not image_paths:
            print(f"[skip] no rgb images: {scene_root}")
            continue

        # 1) GPT captions (cached)
        if not args.skip_gpt:
            for p in tqdm(image_paths, desc=f"{scene_name} GPT", leave=False):
                key = str(p)
                if key in captions and captions[key].strip():
                    continue
                text = generate_caption_openai(p, model=args.openai_model, prompt=args.prompt)
                if not text:
                    text = "(caption failed)"
                captions[key] = text
                append_caption_jsonl(captions_path, CaptionItem(image_path=key, text=text))
        else:
            if not captions:
                print(f"[skip] captions cache missing/empty: {captions_path}")
                continue

        # Build candidates
        candidate_texts: list[str] = []
        candidate_meta: list[dict[str, str]] = []
        for p in image_paths:
            key = str(p)
            if key not in captions:
                continue
            t = captions[key].strip()
            if not t:
                continue
            candidate_texts.append(t)
            candidate_meta.append({"source_image": key, "text": t})

        if not candidate_texts:
            print(f"[skip] no candidate texts: {scene_root}")
            continue

        # 2) CLIP scoring
        scores = compute_clip_scores(
            image_paths,
            candidate_texts,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            device=args.device,
            batch_size=args.batch_size,
        )
        best_idx = int(scores["best_index"])
        best_text = candidate_texts[best_idx] if best_idx >= 0 else None
        best_score = float(scores["avg_scores"][best_idx]) if best_idx >= 0 else None

        best_path = out_dir / f"{scene_name}.json"
        best_path.write_text(
            __import__("json").dumps(
                {"best_text": best_text, "best_avg_clip_score": best_score},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if args.write_details:
            detail_path = out_dir / f"{scene_name}.clip_scores.json"
            detail_path.write_text(
                __import__("json").dumps(
                    {
                        "scene": scene_name,
                        "scene_root": str(scene_root),
                        "images": [str(p) for p in image_paths],
                        "candidates": candidate_meta,
                        "scores": scores,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        print(f"[ok] {scene_name} -> {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

