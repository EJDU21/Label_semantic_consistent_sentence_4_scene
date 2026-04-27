#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


def iter_rgb_images(scene_root: Path) -> list[Path]:
    # Expected structure (nested ok):
    # scene_root/**/{training_view,testing_view}/rgb/*.png
    imgs: list[Path] = []
    for view_name in ("training_view", "testing_view"):
        for view_root in scene_root.rglob(view_name):
            rgb_dir = view_root / "rgb"
            if rgb_dir.exists():
                imgs.extend(sorted(rgb_dir.glob("*.png")))
    # de-dup while preserving sort-ish stability
    uniq = sorted({p.resolve() for p in imgs})
    return [Path(p) for p in uniq]


def load_image_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    # Unreal exports may be RGBA; CLIP expects RGB
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _openai_client():
    from openai import OpenAI  # type: ignore

    return OpenAI()


DEFAULT_PROMPT = (
    "Write ONE concise English sentence describing the scene in the image.\n"
    "Requirements:\n"
    "- Describe only visible objects, layout, and environment; avoid speculation.\n"
    "- Keep it short (about 12–25 words).\n"
    "- Do not mention camera parameters, azimuth/elevation, or filenames.\n"
    "- Do not use bullet points.\n"
)


def generate_caption_openai(image_path: Path, *, model: str, prompt: str) -> str:
    client = _openai_client()
    with load_image_rgb(image_path) as im:
        # Encode to PNG bytes for API
        import io

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    # Responses API (image + text)
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{__import__('base64').b64encode(png_bytes).decode('utf-8')}",
                    },
                ],
            }
        ],
    )
    text = (resp.output_text or "").strip()
    return text


@dataclass(frozen=True)
class CaptionItem:
    image_path: str
    text: str


def read_captions_jsonl(path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    if not path.exists():
        return captions
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        img = str(obj["image_path"])
        captions[img] = str(obj["text"])
    return captions


def append_caption_jsonl(path: Path, item: CaptionItem) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"image_path": item.image_path, "text": item.text}, ensure_ascii=False) + "\n")


def compute_clip_scores(
    image_paths: list[Path],
    candidate_texts: list[str],
    *,
    clip_model: str,
    clip_pretrained: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model)

    model.eval()
    model.to(device)

    # 1) Encode all images once
    img_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP encode images"):
            batch = image_paths[i : i + batch_size]
            ims = [preprocess(load_image_rgb(p)) for p in batch]
            ims_t = torch.stack(ims).to(device)
            feats = model.encode_image(ims_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            img_embs.append(feats.cpu())
    img_emb = torch.cat(img_embs, dim=0)  # [N, D] on CPU

    # 2) For each candidate text: encode + similarity + average
    avg_scores: list[float] = []
    per_image_scores: list[list[float]] = []
    with torch.no_grad():
        for t in tqdm(candidate_texts, desc="CLIP score texts"):
            tok = tokenizer([t])
            tok = tok.to(device)
            txt = model.encode_text(tok)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            txt_cpu = txt.cpu()  # [1, D]
            sims = (img_emb @ txt_cpu.T).squeeze(1)  # cosine similarity, [N]
            sims_np = sims.numpy().astype(np.float32)
            per_image_scores.append([float(x) for x in sims_np.tolist()])
            avg_scores.append(float(sims_np.mean()))

    best_idx = int(np.argmax(np.array(avg_scores, dtype=np.float32))) if avg_scores else -1
    return {
        "clip": {"model": clip_model, "pretrained": clip_pretrained, "device": device},
        "num_images": len(image_paths),
        "num_candidates": len(candidate_texts),
        "avg_scores": avg_scores,
        "per_image_scores": per_image_scores,
        "best_index": best_idx,
    }


def infer_scene_content_dir(scene_root: Path) -> Path:
    # scene_root is like: /.../dataset/scene_01
    # We want: /.../dataset/scene_01/<scene_content_name>/
    children = [p for p in scene_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(children) == 1:
        return children[0]
    if len(children) == 0:
        raise SystemExit(f"no scene content folder found under: {scene_root}")
    names = ", ".join(sorted(p.name for p in children))
    raise SystemExit(
        "multiple scene content folders found; please pass --out-dir explicitly. "
        f"scene_root={scene_root} candidates=[{names}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate 1 caption per image for a scene, then pick best caption by avg CLIP.")
    ap.add_argument("--scene-root", type=Path, required=True, help="Path to one scene directory (e.g. /.../scene_01).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <scene-root>/<scene_content_name>/ (auto inferred).",
    )
    ap.add_argument(
        "--captions-jsonl",
        type=Path,
        default=None,
        help="Captions cache file (jsonl). Default: out-dir/<scene>.captions.jsonl",
    )

    ap.add_argument("--openai-model", type=str, default="gpt-5.5-pro-2026-04-23", help="OpenAI model for image captioning.")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for captioning.")
    ap.add_argument("--skip-gpt", action="store_true", help="Skip GPT captioning; requires captions cache already exists.")

    ap.add_argument("--clip-model", type=str, default="ViT-B-32", help="CLIP model name (open_clip).")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k", help="Pretrained weights id (open_clip).")
    ap.add_argument("--device", type=str, default='cuda', help="cpu or cuda. Default: auto.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP image encoding.")
    ap.add_argument(
        "--write-details",
        action="store_true",
        help="If set, also write <scene>.clip_scores.json for debugging.",
    )

    args = ap.parse_args()

    scene_root: Path = args.scene_root
    if not scene_root.exists():
        raise SystemExit(f"scene root not found: {scene_root}")

    scene_name = scene_root.name
    out_dir: Path = args.out_dir or infer_scene_content_dir(scene_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions_path = args.captions_jsonl or (out_dir / f"{scene_name}.captions.jsonl")
    captions = read_captions_jsonl(captions_path)

    image_paths = iter_rgb_images(scene_root)
    if not image_paths:
        raise SystemExit(f"no rgb images found under: {scene_root}")

    # 1) GPT caption each image (1 sentence) with caching
    if not args.skip_gpt:
        for p in tqdm(image_paths, desc="GPT captions"):
            key = str(p)
            if key in captions and captions[key].strip():
                continue
            text = generate_caption_openai(p, model=args.openai_model, prompt=args.prompt)
            if not text:
                text = "（描述失敗）"
            captions[key] = text
            append_caption_jsonl(captions_path, CaptionItem(image_path=key, text=text))
    else:
        if not captions:
            raise SystemExit("--skip-gpt was set but captions cache is empty/missing.")

    # Only keep captions for the images we will score
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
        raise SystemExit("no candidate texts available (captions empty).")

    # Device auto
    if args.device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    # 2) CLIP average scoring per candidate over all images in the scene
    scores = compute_clip_scores(
        image_paths,
        candidate_texts,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
        batch_size=args.batch_size,
    )

    best_idx = int(scores["best_index"])
    best = candidate_meta[best_idx] if best_idx >= 0 else None

    # 3) Write outputs
    best_path = out_dir / f"{scene_name}.json"

    if best is None:
        best_obj = {"best_text": None, "best_avg_clip_score": None}
    else:
        best_obj = {"best_text": best["text"], "best_avg_clip_score": float(scores["avg_scores"][best_idx])}
    best_path.write_text(json.dumps(best_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.write_details:
        detail_path = out_dir / f"{scene_name}.clip_scores.json"
        detail = {
            "scene": scene_name,
            "scene_root": str(scene_root),
            "images": [str(p) for p in image_paths],
            "candidates": candidate_meta,
            "scores": scores,
        }
        detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"captions_cache={captions_path}")
    print(f"best={best_path}")
    if args.write_details:
        print(f"detail_scores={detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

