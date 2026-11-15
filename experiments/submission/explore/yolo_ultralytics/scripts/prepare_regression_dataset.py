import argparse
import json
import random
from pathlib import Path

from PIL import Image

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def object_bbox(obj: dict, image_w: int, image_h: int):
    x_vertex = float(obj.get("x_vertex", 0.0))
    y_vertex = float(obj.get("y_vertex", 0.0))
    width = max(2.0, float(obj.get("width", 2.0)))
    height = max(1.0, float(obj.get("height", 1.0)))
    thickness = max(1.0, float(obj.get("thickness", 1.0)))

    half_w = width / 2.0
    x_min = x_vertex - half_w
    x_max = x_vertex + half_w
    y_min = y_vertex - thickness / 2.0
    y_max = y_vertex + height + thickness / 2.0

    x_min = clamp(x_min, 0.0, float(image_w))
    x_max = clamp(x_max, 0.0, float(image_w))
    y_min = clamp(y_min, 0.0, float(image_h))
    y_max = clamp(y_max, 0.0, float(image_h))

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def create_sample(obj: dict, bbox: tuple):
    x_min, y_min, x_max, y_max = bbox
    crop_w = max(1.0, x_max - x_min)
    crop_h = max(1.0, y_max - y_min)

    x_vertex = float(obj["x_vertex"])
    y_vertex = float(obj["y_vertex"])
    width = float(obj["width"])
    height = float(obj["height"])
    thickness = float(obj["thickness"])

    x_vertex_n = clamp((x_vertex - x_min) / crop_w, 0.0, 1.0)
    y_vertex_n = clamp((y_vertex - y_min) / crop_h, 0.0, 1.0)
    width_n = clamp(width / crop_w, 0.0, 1.0)
    height_n = clamp(height / crop_h, 0.0, 1.0)
    thickness_n = clamp(thickness / crop_h, 0.0, 1.0)

    return {
        "target": [x_vertex_n, y_vertex_n, width_n, height_n, thickness_n],
        "bbox": [x_min, y_min, x_max, y_max],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for direct hyperbola-parameter regression.")
    parser.add_argument("--data-dir", type=str, default="../../", help="Directory containing images and annotations.json")
    parser.add_argument("--annotations", type=str, default="", help="Path to annotations.json (default: <data-dir>/annotations.json)")
    parser.add_argument("--output-dir", type=str, default="../reg_dataset", help="Output dataset folder")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    annotations_path = Path(args.annotations).resolve() if args.annotations else data_dir / "annotations.json"
    output_dir = Path(args.output_dir).resolve()

    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {annotations_path}")

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    samples = []
    for image_name, objects in annotations.items():
        image_path = data_dir / image_name
        if Path(image_name).suffix.lower() not in SUPPORTED_EXTS:
            continue
        if not image_path.exists():
            continue

        if not objects:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_w, image_h = image.size

            for obj_idx, obj in enumerate(objects):
                bbox = object_bbox(obj, image_w, image_h)
                if bbox is None:
                    continue

                x_min, y_min, x_max, y_max = bbox
                crop_box = (int(x_min), int(y_min), int(x_max), int(y_max))
                crop = image.crop(crop_box)

                crop_name = f"{Path(image_name).stem}_{obj_idx:03d}.jpg"
                crop_path = crops_dir / crop_name
                crop.save(crop_path)

                sample = create_sample(obj, bbox)
                sample["crop_path"] = str(crop_path.as_posix())
                sample["image_name"] = image_name
                sample["obj_index"] = obj_idx
                samples.append(sample)

    if len(samples) < 2:
        raise RuntimeError("Not enough samples for train/val split. Need at least 2 labeled objects.")

    random.seed(args.seed)
    random.shuffle(samples)

    split_idx = max(1, min(len(samples) - 1, int(len(samples) * args.train_ratio)))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    (output_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in train_samples),
        encoding="utf-8",
    )
    (output_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in val_samples),
        encoding="utf-8",
    )

    meta = {
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "target_order": ["x_vertex_n", "y_vertex_n", "width_n", "height_n", "thickness_n"],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Regression dataset prepared.")
    print(f"- Output: {output_dir}")
    print(f"- Total samples: {len(samples)}")
    print(f"- Train samples: {len(train_samples)}")
    print(f"- Val samples: {len(val_samples)}")


if __name__ == "__main__":
    main()
