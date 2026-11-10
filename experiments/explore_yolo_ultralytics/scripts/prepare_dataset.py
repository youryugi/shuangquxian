import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def hyperbola_to_bbox(obj: dict, image_w: int, image_h: int):
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

    bw = x_max - x_min
    bh = y_max - y_min
    cx = x_min + bw / 2.0
    cy = y_min + bh / 2.0

    return cx, cy, bw, bh


def build_yolo_label(objects: list, image_w: int, image_h: int, class_id: int = 0):
    lines = []
    for obj in objects:
        bbox = hyperbola_to_bbox(obj, image_w, image_h)
        if bbox is None:
            continue
        cx, cy, bw, bh = bbox
        cx_n = cx / image_w
        cy_n = cy / image_h
        bw_n = bw / image_w
        bh_n = bh / image_h
        lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")
    return lines


def write_yaml(dataset_root: Path, class_name: str):
    yaml_path = dataset_root / "dataset.yaml"
    content = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert hyperbola annotations.json to YOLO dataset format.")
    parser.add_argument("--data-dir", type=str, default="../", help="Directory containing images and annotations.json")
    parser.add_argument("--annotations", type=str, default="", help="Path to annotations.json (default: <data-dir>/annotations.json)")
    parser.add_argument("--output-dir", type=str, default="../yolo_dataset", help="Output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class-name", type=str, default="hyperbola", help="Single class name")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    annotations_path = Path(args.annotations).resolve() if args.annotations else data_dir / "annotations.json"
    output_dir = Path(args.output_dir).resolve()

    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {annotations_path}")

    output_images_train = output_dir / "images" / "train"
    output_images_val = output_dir / "images" / "val"
    output_labels_train = output_dir / "labels" / "train"
    output_labels_val = output_dir / "labels" / "val"

    for p in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    all_image_names = []
    for name in sorted(annotations.keys()):
        ext = Path(name).suffix.lower()
        if ext in SUPPORTED_EXTS and (data_dir / name).exists():
            all_image_names.append(name)

    if not all_image_names:
        raise RuntimeError("No valid images found from annotations.json keys.")

    random.seed(args.seed)
    random.shuffle(all_image_names)
    split_idx = max(1, min(len(all_image_names) - 1, int(len(all_image_names) * args.train_ratio)))

    train_names = set(all_image_names[:split_idx])
    val_names = set(all_image_names[split_idx:])

    def process_split(names: set, split: str):
        image_out = output_images_train if split == "train" else output_images_val
        label_out = output_labels_train if split == "train" else output_labels_val

        count_images = 0
        count_objects = 0

        for image_name in sorted(names):
            image_path = data_dir / image_name
            objects = annotations.get(image_name, [])

            with Image.open(image_path) as img:
                w, h = img.size

            yolo_lines = build_yolo_label(objects, w, h, class_id=0)
            count_objects += len(yolo_lines)

            shutil.copy2(image_path, image_out / image_name)
            label_path = label_out / f"{Path(image_name).stem}.txt"
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")
            count_images += 1

        return count_images, count_objects

    n_train_img, n_train_obj = process_split(train_names, "train")
    n_val_img, n_val_obj = process_split(val_names, "val")

    write_yaml(output_dir, args.class_name)

    print("Dataset prepared.")
    print(f"- Output: {output_dir}")
    print(f"- Train images: {n_train_img}, objects: {n_train_obj}")
    print(f"- Val images: {n_val_img}, objects: {n_val_obj}")
    print(f"- YAML: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
