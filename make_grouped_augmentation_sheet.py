from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    fitted = Image.new("RGB", size, (245, 245, 242))
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    left = (target_w - copy.width) // 2
    top = (target_h - copy.height) // 2
    fitted.paste(copy, (left, top))
    return fitted


def _draw_label(canvas: Image.Image, xy: tuple[int, int], text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(xy, text, fill=(25, 25, 25), font=font)


def _make_group_row(
    original_path: Path,
    augmentations: list[dict[str, Any]],
    tile_size: tuple[int, int],
    pad: int,
    label_h: int,
) -> Image.Image:
    tiles: list[tuple[str, Image.Image]] = [("original", Image.open(original_path))]
    for idx, aug in enumerate(augmentations, start=1):
        tiles.append((f"aug {idx}", Image.open(aug["augmented"])))

    width = len(tiles) * tile_size[0] + (len(tiles) + 1) * pad
    height = tile_size[1] + label_h + pad * 2
    row = Image.new("RGB", (width, height), (236, 236, 232))

    x = pad
    for label, image in tiles:
        _draw_label(row, (x, pad), label)
        row.paste(_fit_image(image, tile_size), (x, pad + label_h))
        x += tile_size[0] + pad
    return row


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.records.read_text(encoding="utf-8"))

    rows: list[tuple[str, Image.Image]] = []
    for image_name, item in records.items():
        original_path = args.input_dir / image_name
        if not original_path.exists():
            print(f"Skipping missing original: {original_path}")
            continue
        augmentations = item.get("augmentations", [])
        if not augmentations:
            continue
        row = _make_group_row(
            original_path,
            augmentations,
            tile_size=(args.tile_size, args.tile_size),
            pad=args.pad,
            label_h=args.label_height,
        )
        group_path = args.output_dir / f"{original_path.stem}_group.jpg"
        row.save(group_path, quality=94)
        print(group_path)
        rows.append((image_name, row))

    if not rows:
        raise SystemExit("No grouped rows were created.")

    sheet_w = max(row.width for _, row in rows) + args.pad * 2
    sheet_h = sum(row.height + args.label_height + args.pad for _, row in rows) + args.pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (226, 226, 222))
    y = args.pad
    for image_name, row in rows:
        _draw_label(sheet, (args.pad, y), image_name)
        y += args.label_height
        sheet.paste(row, (args.pad, y))
        y += row.height + args.pad

    sheet_path = args.output_dir / "grouped_augmentation_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    print(sheet_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create horizontal original-plus-augment groups from augmentation records.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the original images.")
    parser.add_argument("--records", type=Path, required=True, help="detections_and_augmentations.json from auto_detect_person_face_augment.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--pad", type=int, default=14)
    parser.add_argument("--label-height", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
