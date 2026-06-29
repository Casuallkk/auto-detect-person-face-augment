from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass
import math

from PIL import Image, ImageDraw, ImageFont

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class AugmentResult:
    image: Image.Image
    angle: float
    crop_box: Box
    anchor_kind: str
    tries: int

def _box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersect(a: Box, b: Box) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _clamp_crop(left: float, top: float, crop_w: float, crop_h: float, img_w: int, img_h: int) -> Box:
    left = min(max(0.0, left), max(0.0, img_w - crop_w))
    top = min(max(0.0, top), max(0.0, img_h - crop_h))
    return (left, top, left + crop_w, top + crop_h)


def _rotate_points(points: Iterable[tuple[float, float]], w: float, h: float, angle_degrees: float) -> list[tuple[float, float]]:
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = w * 0.5, h * 0.5

    corners = [(-cx, -cy), (w - cx, -cy), (w - cx, h - cy), (-cx, h - cy)]
    rotated_corners = [(cos_a * x - sin_a * y, sin_a * x + cos_a * y) for x, y in corners]
    min_x = min(x for x, _ in rotated_corners)
    min_y = min(y for _, y in rotated_corners)

    rotated = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        rx = cos_a * dx - sin_a * dy - min_x
        ry = sin_a * dx + cos_a * dy - min_y
        rotated.append((rx, ry))
    return rotated

def _transform_box_from_crop(box: Box, crop: Box, angle: float, crop_w: float, crop_h: float, inner_left: float, inner_top: float) -> Box:
    local = [
        (box[0] - crop[0], box[1] - crop[1]),
        (box[2] - crop[0], box[1] - crop[1]),
        (box[2] - crop[0], box[3] - crop[1]),
        (box[0] - crop[0], box[3] - crop[1]),
    ]
    rotated = _rotate_points(local, crop_w, crop_h, angle)
    xs = [p[0] - inner_left for p in rotated]
    ys = [p[1] - inner_top for p in rotated]
    return (min(xs), min(ys), max(xs), max(ys))


def _center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    target_w, target_h = size
    aspect = target_w / target_h
    if w / h > aspect:
        crop_h = h
        crop_w = int(h * aspect)
    else:
        crop_w = w
        crop_h = int(w / aspect)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h)).resize(size, Image.Resampling.LANCZOS)



def _largest_rotated_inner_rect(w: float, h: float, angle_degrees: float) -> tuple[int, int]:
    """Largest centered axis-aligned rectangle inside a rotated w x h rectangle."""
    if w <= 0 or h <= 0:
        return 1, 1

    angle = math.radians(angle_degrees % 180)
    if angle > math.pi / 2:
        angle = math.pi - angle

    sin_a = abs(math.sin(angle))
    cos_a = abs(math.cos(angle))
    if sin_a < 1e-9:
        return max(1, int(w)), max(1, int(h))

    width_is_longer = w >= h
    side_long = w if width_is_longer else h
    side_short = h if width_is_longer else w

    if side_short <= 2.0 * sin_a * cos_a * side_long:
        x = 0.5 * side_short
        inner_long = x / sin_a
        inner_short = x / cos_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        if abs(cos_2a) < 1e-9:
            inner_long = inner_short = side_short / math.sqrt(2.0)
        else:
            inner_long = (side_long * cos_a - side_short * sin_a) / cos_2a
            inner_short = (side_short * cos_a - side_long * sin_a) / cos_2a

    if width_is_longer:
        inner_w, inner_h = inner_long, inner_short
    else:
        inner_w, inner_h = inner_short, inner_long
    return max(1, int(inner_w)), max(1, int(inner_h))




def person_centered_random_crop_rotate(
    image: Image.Image,
    target_size: tuple[int, int] = (512, 512),
    face_boxes: list[Box] | None = None,
    person_boxes: list[Box] | None = None,
    max_angle: float = 15.0,
    min_face_visible: float = 0.72,
    attempts: int = 80,
    rng: random.Random | None = None,
) -> AugmentResult:
    """Random crop/rotate without mirroring or black borders.

    If face boxes are provided, at least one face is kept visible after the
    rotation-safe crop. Person boxes are used as softer anchors.
    """
    rng = rng or random.Random()
    image = image.convert("RGB")
    img_w, img_h = image.size
    target_w, target_h = target_size
    aspect = target_w / target_h
    face_boxes = face_boxes or []
    person_boxes = person_boxes or []

    if face_boxes:
        required_box = max(face_boxes, key=_box_area)
        anchor = _box_center(required_box)
        anchor_kind = "face"
    elif person_boxes:
        required_box = None
        anchor = _box_center(max(person_boxes, key=_box_area))
        anchor_kind = "person"
    else:
        required_box = None
        anchor = (img_w * 0.5, img_h * 0.5)
        anchor_kind = "center"

    min_side = min(img_w, img_h)
    for idx in range(1, attempts + 1):
        angle = rng.uniform(-max_angle, max_angle)
        base = rng.uniform(0.56, 0.96) * min_side
        if aspect >= 1.0:
            crop_w = min(float(img_w), base * aspect)
            crop_h = crop_w / aspect
        else:
            crop_h = min(float(img_h), base / aspect)
            crop_w = crop_h * aspect

        if crop_w > img_w:
            crop_w = float(img_w)
            crop_h = crop_w / aspect
        if crop_h > img_h:
            crop_h = float(img_h)
            crop_w = crop_h * aspect

        jitter_x = rng.uniform(-0.22, 0.22) * crop_w
        jitter_y = rng.uniform(-0.22, 0.22) * crop_h
        crop = _clamp_crop(anchor[0] + jitter_x - crop_w * 0.5, anchor[1] + jitter_y - crop_h * 0.5, crop_w, crop_h, img_w, img_h)

        if required_box:
            face_in_crop = _intersect(required_box, crop)
            if _box_area(face_in_crop) / max(1.0, _box_area(required_box)) < 0.98:
                continue

        patch = image.crop(tuple(round(v) for v in crop))
        patch_w, patch_h = patch.size
        rotated = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        inner_w, inner_h = _largest_rotated_inner_rect(patch_w, patch_h, angle)
        inner_w = min(inner_w, rotated.size[0])
        inner_h = min(inner_h, rotated.size[1])
        inner_left = (rotated.size[0] - inner_w) * 0.5
        inner_top = (rotated.size[1] - inner_h) * 0.5

        if required_box:
            face_after = _transform_box_from_crop(required_box, crop, angle, patch_w, patch_h, inner_left, inner_top)
            visible = _box_area(_intersect(face_after, (0, 0, inner_w, inner_h))) / max(1.0, _box_area(face_after))
            cx, cy = _box_center(face_after)
            if visible < min_face_visible or not (0 <= cx <= inner_w and 0 <= cy <= inner_h):
                continue

        safe = rotated.crop((round(inner_left), round(inner_top), round(inner_left + inner_w), round(inner_top + inner_h)))
        safe = safe.resize(target_size, Image.Resampling.LANCZOS)
        return AugmentResult(safe, angle, crop, anchor_kind, idx)

    fallback = _center_crop(image, target_size)
    return AugmentResult(fallback, 0.0, (0, 0, img_w, img_h), "fallback", attempts)

def _make_preview(original: Image.Image, augmented: Image.Image, caption: str) -> Image.Image:
    preview_h = 512
    left = original.convert("RGB")
    left.thumbnail((512, preview_h), Image.Resampling.LANCZOS)
    right = augmented.convert("RGB")
    right.thumbnail((512, preview_h), Image.Resampling.LANCZOS)

    pad = 14
    label_h = 34
    width = left.width + right.width + pad * 3
    height = max(left.height, right.height) + pad * 2 + label_h
    canvas = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    left_xy = (pad, pad + label_h)
    right_xy = (pad * 2 + left.width, pad + label_h)
    canvas.paste(left, left_xy)
    canvas.paste(right, right_xy)
    draw.text((pad, pad), "original", fill=(30, 30, 30), font=font)
    draw.text((right_xy[0], pad), f"augmented | {caption}", fill=(30, 30, 30), font=font)
    return canvas

def _list_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _clip_box(box: Box, width: int, height: int) -> Box:
    return (
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    )


def _load_ultralytics_model(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install it, then rerun with "
            f"--yolo-model {model_name}. Example: pip install ultralytics"
        ) from exc
    return YOLO(model_name)


def detect_person_boxes_yolo(image_path: Path, model: Any, conf: float) -> list[Box]:
    results = model.predict(str(image_path), classes=[0], conf=conf, verbose=False)
    if not results:
        return []

    boxes: list[Box] = []
    result = results[0]
    if result.boxes is None:
        return boxes

    xyxy = result.boxes.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    for row in xyxy:
        box = tuple(float(v) for v in row[:4])
        if _box_area(box) > 16:
            boxes.append(box)  # type: ignore[arg-type]
    return boxes


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: opencv-contrib-python is needed for YuNet. "
            "Install it and pass --yunet-model path\\to\\face_detection_yunet_2023mar.onnx."
        ) from exc
    return cv2


def detect_face_boxes_yunet(image_path: Path, model_path: Path, score_threshold: float) -> list[Box]:
    cv2 = _load_cv2()
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError("Your OpenCV build does not include FaceDetectorYN. Install opencv-contrib-python.")
    if not model_path.exists():
        raise RuntimeError(f"YuNet model was not found: {model_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return []
    height, width = bgr.shape[:2]
    detector = cv2.FaceDetectorYN_create(str(model_path), "", (width, height), score_threshold, 0.3, 5000)
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)
    if faces is None:
        return []

    boxes: list[Box] = []
    for face in faces:
        x, y, w, h = [float(v) for v in face[:4]]
        box = _clip_box((x, y, x + w, y + h), width, height)
        if _box_area(box) > 16:
            boxes.append(box)
    return boxes


def detect_face_boxes_retinaface(image_path: Path, score_threshold: float) -> list[Box]:
    try:
        from retinaface import RetinaFace
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: retinaface. Install a RetinaFace package or use --face-backend yunet."
        ) from exc

    image = Image.open(image_path)
    width, height = image.size
    detections = RetinaFace.detect_faces(str(image_path), threshold=score_threshold)
    if not isinstance(detections, dict):
        return []

    boxes: list[Box] = []
    for item in detections.values():
        area = item.get("facial_area")
        if not area or len(area) != 4:
            continue
        box = _clip_box(tuple(float(v) for v in area), width, height)  # type: ignore[arg-type]
        if _box_area(box) > 16:
            boxes.append(box)
    return boxes


def detect_face_boxes(
    image_path: Path,
    backend: str,
    yunet_model: Path | None,
    score_threshold: float,
    allow_unavailable: bool,
) -> tuple[list[Box], str]:
    if backend == "none":
        if not allow_unavailable:
            raise RuntimeError(
                "Face detection is disabled. This is unsafe for your requirement because images with faces "
                "could be cropped until the face disappears. Pass --allow-no-face-detector only for debugging."
            )
        return [], "none"
    if backend == "yunet":
        if yunet_model is None:
            raise RuntimeError("--yunet-model is required when --face-backend yunet.")
        return detect_face_boxes_yunet(image_path, yunet_model, score_threshold), "yunet"
    if backend == "retinaface":
        return detect_face_boxes_retinaface(image_path, score_threshold), "retinaface"

    errors: list[str] = []
    if yunet_model is not None:
        try:
            return detect_face_boxes_yunet(image_path, yunet_model, score_threshold), "yunet"
        except RuntimeError as exc:
            errors.append(str(exc))
    try:
        return detect_face_boxes_retinaface(image_path, score_threshold), "retinaface"
    except RuntimeError as exc:
        errors.append(str(exc))

    if errors:
        message = "Face detector unavailable:\n" + "\n".join(f"  - {error}" for error in errors)
        if not allow_unavailable:
            raise RuntimeError(
                message
                + "\nInstall/configure YuNet or RetinaFace before generating training augmentations, "
                "or pass --allow-no-face-detector only for debugging previews."
            )
        print(message)
        print("Continuing without face boxes because --allow-no-face-detector was set.")
    return [], "none"


def _draw_boxes(image: Image.Image, face_boxes: list[Box], person_boxes: list[Box]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for box in person_boxes:
        draw.rectangle(box, outline=(30, 144, 255), width=4)
        draw.text((box[0] + 3, box[1] + 3), "person", fill=(30, 144, 255), font=font)
    for box in face_boxes:
        draw.rectangle(box, outline=(255, 80, 80), width=4)
        draw.text((box[0] + 3, box[1] + 3), "face", fill=(255, 80, 80), font=font)
    return out


def _save_detection_preview(output_path: Path, image: Image.Image, face_boxes: list[Box], person_boxes: list[Box]) -> None:
    preview = _draw_boxes(image, face_boxes, person_boxes)
    preview.thumbnail((900, 900), Image.Resampling.LANCZOS)
    preview.save(output_path, quality=94)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir = args.output_dir / "augmented"
    preview_dir = args.output_dir / "previews"
    detection_dir = args.output_dir / "detections"
    augmented_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)
    detection_dir.mkdir(exist_ok=True)

    yolo_model = None
    if args.person_backend == "yolo":
        yolo_model = _load_ultralytics_model(args.yolo_model)

    rng = random.Random(args.seed)
    records: dict[str, Any] = {}
    for image_path in _list_images(args.input_dir)[: args.limit]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if yolo_model is None:
            person_boxes: list[Box] = []
            person_backend = "none"
        else:
            person_boxes = [
                _clip_box(box, width, height)
                for box in detect_person_boxes_yolo(image_path, yolo_model, args.person_conf)
            ]
            person_backend = "yolo"

        face_boxes, face_backend = detect_face_boxes(
            image_path,
            args.face_backend,
            args.yunet_model,
            args.face_conf,
            args.allow_no_face_detector,
        )
        face_boxes = [_clip_box(box, width, height) for box in face_boxes]

        _save_detection_preview(detection_dir / f"{image_path.stem}_detections.jpg", image, face_boxes, person_boxes)

        records[image_path.name] = {
            "face_backend": face_backend,
            "person_backend": person_backend,
            "face_boxes": face_boxes,
            "person_boxes": person_boxes,
            "augmentations": [],
        }

        for aug_idx in range(1, args.num_aug + 1):
            result = person_centered_random_crop_rotate(
                image,
                target_size=(args.size, args.size),
                face_boxes=face_boxes,
                person_boxes=person_boxes,
                max_angle=args.max_angle,
                min_face_visible=args.min_face_visible,
                attempts=args.attempts,
                rng=rng,
            )
            aug_path = augmented_dir / f"{image_path.stem}_aug{aug_idx:02d}.jpg"
            result.image.save(aug_path, quality=94)

            caption = f"{result.anchor_kind}, aug={aug_idx}, angle={result.angle:.1f}, tries={result.tries}"
            preview = _make_preview(_draw_boxes(image, face_boxes, person_boxes), result.image, caption)
            preview_path = preview_dir / f"{image_path.stem}_aug{aug_idx:02d}_preview.jpg"
            preview.save(preview_path, quality=94)

            records[image_path.name]["augmentations"].append(
                {
                    "augmented": str(aug_path),
                    "preview": str(preview_path),
                    "anchor_kind": result.anchor_kind,
                    "angle": result.angle,
                    "tries": result.tries,
                    "crop_box": result.crop_box,
                }
            )
            print(preview_path)

    (args.output_dir / "detections_and_augmentations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-detect people/faces, then create person-centered no-flip no-black-border augmentations.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument("--num-aug", type=int, default=3)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-angle", type=float, default=15.0)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--min-face-visible", type=float, default=0.72)
    parser.add_argument("--person-backend", choices=["yolo", "none"], default="yolo")
    parser.add_argument("--yolo-model", default="yolo26n.pt", help="Ultralytics YOLO model name or local .pt path.")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--face-backend", choices=["auto", "yunet", "retinaface", "none"], default="auto")
    parser.add_argument("--yunet-model", type=Path, default=None, help="Path to YuNet ONNX model, e.g. face_detection_yunet_2023mar.onnx.")
    parser.add_argument("--face-conf", type=float, default=0.65)
    parser.add_argument(
        "--allow-no-face-detector",
        action="store_true",
        help="Debug only: allow augmentation without face boxes. Do not use this for face-preserving training data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
