import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from e2e_detector_model import HyperbolaE2EDetector, decode_predictions

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_source_images(source: Path):
    if source.is_file():
        if source.suffix.lower() in SUPPORTED_EXTS:
            return [source]
        return []
    out = []
    for p in sorted(source.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out.append(p)
    return out


def hyperbola_band_polygon(obj: dict, n_points: int = 120):
    x_vertex = float(obj["x_vertex"])
    y_vertex = float(obj["y_vertex"])
    width = max(2.0, float(obj["width"]))
    height = max(1.0, float(obj["height"]))
    thickness = max(1.0, float(obj["thickness"]))

    half_w = width / 2.0
    x_left = x_vertex - half_w
    x_right = x_vertex + half_w

    upper_pts = []
    lower_pts = []
    centerline = []

    for i in range(n_points + 1):
        t = i / n_points
        x = x_left + (x_right - x_left) * t
        dx = (x - x_vertex) / half_w
        y_center = y_vertex + height * (dx**2)
        upper_pts.append((x, y_center - thickness / 2.0))
        lower_pts.append((x, y_center + thickness / 2.0))
        centerline.append((x, y_center))

    polygon = upper_pts + list(reversed(lower_pts))
    return polygon, centerline


def draw_hyperbolas_on_image(image_path: Path, objects, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    for obj in objects:
        polygon, centerline = hyperbola_band_polygon(obj)
        draw.polygon(polygon, fill=(0, 255, 0, 55))
        draw.line(centerline, fill=(0, 255, 0, 160), width=2)
        xv = float(obj["x_vertex"])
        yv = float(obj["y_vertex"])
        r = 4
        draw.ellipse((xv - r, yv - r, xv + r, yv + r), fill=(255, 255, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def to_annotation(decoded_item, image_w: int, image_h: int, img_size: int):
    sx = image_w / img_size
    sy = image_h / img_size

    x_vertex = decoded_item["x_vertex_n"] * image_w
    y_vertex = decoded_item["y_vertex_n"] * image_h
    width = decoded_item["width_n"] * image_w
    height = decoded_item["height_n"] * image_h
    thickness = decoded_item["thickness_n"] * image_h

    x_vertex = max(0.0, min(float(image_w), x_vertex))
    y_vertex = max(0.0, min(float(image_h), y_vertex))
    width = max(2.0, min(float(image_w), width))
    height = max(1.0, min(float(image_h), height))
    thickness = max(1.0, min(float(image_h), thickness))

    box = decoded_item["bbox"]
    bbox = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]

    return {
        "label": "hyperbola",
        "score": round(float(decoded_item["score"]), 4),
        "x_vertex": round(x_vertex, 2),
        "y_vertex": round(y_vertex, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "thickness": round(thickness, 2),
        "bbox": [round(float(v), 2) for v in bbox],
    }


def main():
    parser = argparse.ArgumentParser(description="Infer end-to-end hyperbola detector.")
    parser.add_argument("--weights", type=str, required=True, help="Path to e2e best.pt")
    parser.add_argument("--source", type=str, required=True, help="Image path or folder path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--conf", type=float, default=0.35, help="Objectness threshold")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--project", type=str, default="../runs_e2e", help="Output root")
    parser.add_argument("--name", type=str, default="predict_e2e", help="Run name")
    parser.add_argument("--save-json", action="store_true", help="Save annotation JSON")
    parser.add_argument("--save-hyperbola-vis", action="store_true", help="Save overlay visualization")
    args = parser.parse_args()

    ckpt = torch.load(str(Path(args.weights).resolve()), map_location="cpu")
    img_size = int(ckpt.get("img_size", 640))
    stride = int(ckpt.get("stride", 16))

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = HyperbolaE2EDetector(width=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    source = Path(args.source).resolve()
    image_paths = iter_source_images(source)

    if not image_paths:
        raise RuntimeError(f"No images found in source: {source}")

    run_dir = Path(args.project).resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    out = {}

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size

            x = tf(image).unsqueeze(0).to(device)
            pred = model(x)[0].cpu()
            decoded = decode_predictions(pred, img_size=img_size, stride=stride, conf_thr=args.conf, nms_iou=args.nms_iou)

            ann_list = [to_annotation(d, image_w=image_w, image_h=image_h, img_size=img_size) for d in decoded]
            out[image_path.name] = ann_list

    if args.save_json:
        json_path = run_dir / "predictions_annotations.json"
        json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {json_path}")

    if args.save_hyperbola_vis:
        vis_dir = run_dir / "hyperbola_vis"
        for image_path in image_paths:
            objs = out.get(image_path.name, [])
            plain_objs = [
                {
                    "label": obj["label"],
                    "x_vertex": obj["x_vertex"],
                    "y_vertex": obj["y_vertex"],
                    "width": obj["width"],
                    "height": obj["height"],
                    "thickness": obj["thickness"],
                }
                for obj in objs
            ]
            draw_hyperbolas_on_image(image_path, plain_objs, vis_dir / image_path.name)
        print(f"Saved visualization: {vis_dir}")

    print("Inference finished.")


if __name__ == "__main__":
    main()
