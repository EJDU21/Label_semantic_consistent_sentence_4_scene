#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImgStats:
    shape: tuple[int, ...]
    mode_frac: float
    std_mean: float


def _img_to_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    # Common in Unreal exports: RGBA. Ignore alpha for statistics.
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def compute_stats(path: Path) -> ImgStats:
    arr = _img_to_array(path)
    if arr.ndim == 2:
        flat = arr.reshape(-1)
        _, cnt = np.unique(flat, return_counts=True)
        mode_frac = float(cnt.max() / cnt.sum())
        std_mean = float(flat.astype(np.float32).std())
        return ImgStats(shape=tuple(arr.shape), mode_frac=mode_frac, std_mean=std_mean)

    flat = arr.reshape(-1, arr.shape[-1])
    _, cnt = np.unique(flat, axis=0, return_counts=True)
    mode_frac = float(cnt.max() / cnt.sum())
    std_mean = float(arr.reshape(-1, arr.shape[-1]).astype(np.float32).std(axis=0).mean())
    return ImgStats(shape=tuple(arr.shape), mode_frac=mode_frac, std_mean=std_mean)


def is_invalid_by_stats(
    stats: ImgStats,
    *,
    mode_frac_threshold: float,
    std_mean_threshold: float,
) -> bool:
    # Skybox / missing geometry frequently produces nearly-constant outputs.
    return (stats.mode_frac >= mode_frac_threshold) or (stats.std_mean <= std_mean_threshold)


def iter_view_roots(dataset_root: Path) -> Iterable[Path]:
    # Support nested structure like:
    # dataset/scene_01/InsideOut/{training_view,testing_view}/...
    for name in ("training_view", "testing_view"):
        yield from dataset_root.rglob(name)


def _basename(p: Path) -> str:
    return p.stem


def _pair_paths(view_root: Path, basename: str) -> dict[str, Path]:
    return {
        "rgb": view_root / "rgb" / f"{basename}.png",
        "depth": view_root / "depth" / f"{basename}.png",
        "normal": view_root / "normal" / f"{basename}.png",
        "camera_info": view_root / "camera_info" / f"{basename}.json",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Filter Unreal capture views where depth/normal are likely invalid "
            "(e.g., skybox-only). Default is dry-run."
        )
    )
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Dataset root directory.")
    ap.add_argument(
        "--mode-frac-threshold",
        type=float,
        default=0.995,
        help="Mark invalid if the most frequent pixel value ratio >= this threshold.",
    )
    ap.add_argument(
        "--std-mean-threshold",
        type=float,
        default=0.5,
        help="Mark invalid if mean pixel std <= this threshold.",
    )
    ap.add_argument(
        "--require-both",
        action="store_true",
        help="If set, only mark invalid when BOTH depth and normal are invalid. Default is either one invalid.",
    )
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. If omitted, only prints what would be deleted.",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("filter_report.json"),
        help="Write a JSON report to this path.",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "mode_frac_threshold": args.mode_frac_threshold,
        "std_mean_threshold": args.std_mean_threshold,
        "require_both": bool(args.require_both),
        "delete": bool(args.delete),
        "views": [],
        "summary": {
            "invalid_views": 0,
            "files_deleted": 0,
            "files_missing_in_pair": 0,
        },
    }

    invalid_view_count = 0
    files_deleted = 0
    files_missing_in_pair = 0

    for view_root in sorted(iter_view_roots(dataset_root)):
        rgb_dir = view_root / "rgb"
        depth_dir = view_root / "depth"
        normal_dir = view_root / "normal"
        cam_dir = view_root / "camera_info"

        if not (rgb_dir.exists() and depth_dir.exists() and normal_dir.exists() and cam_dir.exists()):
            # Not a view root we can operate on.
            continue

        basenames = sorted({_basename(p) for p in depth_dir.glob("*.png")})
        view_entry = {
            "view_root": str(view_root),
            "candidates": len(basenames),
            "invalid": [],
        }

        for bn in basenames:
            depth_path = depth_dir / f"{bn}.png"
            normal_path = normal_dir / f"{bn}.png"

            # If any modality is missing, we won't delete (safer), but we do report it.
            pair = _pair_paths(view_root, bn)
            missing = [k for k, p in pair.items() if not p.exists()]
            if missing:
                files_missing_in_pair += len(missing)
                continue

            depth_stats = compute_stats(depth_path)
            normal_stats = compute_stats(normal_path)

            depth_bad = is_invalid_by_stats(
                depth_stats, mode_frac_threshold=args.mode_frac_threshold, std_mean_threshold=args.std_mean_threshold
            )
            normal_bad = is_invalid_by_stats(
                normal_stats, mode_frac_threshold=args.mode_frac_threshold, std_mean_threshold=args.std_mean_threshold
            )

            invalid = (depth_bad and normal_bad) if args.require_both else (depth_bad or normal_bad)
            if not invalid:
                continue

            invalid_view_count += 1
            item = {
                "basename": bn,
                "depth": {"mode_frac": depth_stats.mode_frac, "std_mean": depth_stats.std_mean, "shape": depth_stats.shape},
                "normal": {"mode_frac": normal_stats.mode_frac, "std_mean": normal_stats.std_mean, "shape": normal_stats.shape},
                "paths": {k: str(p) for k, p in pair.items()},
            }
            view_entry["invalid"].append(item)

            if args.delete:
                for p in pair.values():
                    try:
                        os.remove(p)
                        files_deleted += 1
                    except FileNotFoundError:
                        pass

        report["views"].append(view_entry)

    report["summary"]["invalid_views"] = invalid_view_count
    report["summary"]["files_deleted"] = files_deleted
    report["summary"]["files_missing_in_pair"] = files_missing_in_pair

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Console summary
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if not args.delete:
        print(f"[dry-run] report written to: {args.report}")
    else:
        print(f"[delete] report written to: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

