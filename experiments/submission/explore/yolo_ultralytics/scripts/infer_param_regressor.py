import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
from ultralytics import YOLO


def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 5),
        torch.nn.Sigmoid(),
    )
    return model


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


def draw_hyperbolas_on_image(image_path: Path, objects: list, output_path: Path):
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


def decode_params(pred, x_min, y_min, x_max, y_max):
    x_vertex_n, y_vertex_n, width_n, height_n, thickness_n = pred

    crop_w = max(1.0, x_max - x_min)
    crop_h = max(1.0, y_max - y_min)

    x_vertex = x_min + x_vertex_n * crop_w
    y_vertex = y_min + y_vertex_n * crop_h
    width = max(2.0, width_n * crop_w)
    height = max(1.0, height_n * crop_h)
    thickness = max(1.0, thickness_n * crop_h)

    return {
        "label": "hyperbola",
        "x_vertex": round(float(x_vertex), 2),
        "y_vertex": round(float(y_vertex), 2),
        "width": round(float(width), 2),
        "height": round(float(height), 2),
        "thickness": round(float(thickness), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Infer hyperbola parameters with detector + direct parameter regressor.")
    parser.add_argument("--det-weights", type=str, required=True, help="YOLO detector weights (best.pt)")
    parser.add_argument("--reg-weights", type=str, required=True, help="Parameter regressor weights (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image path or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence")
    parser.add_argument("--iou", type=float, default=0.7, help="Detector IoU")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--project", type=str, default="../runs_reg", help="Output project directory")
    parser.add_argument("--name", type=str, default="predict_param_reg", help="Output run name")
    parser.add_argument("--save-json", action="store_true", help="Save predictions to JSON")
    parser.add_argument("--save-hyperbola-vis", action="store_true", help="Save hyperbola overlay images")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    det_model = YOLO(str(Path(args.det_weights).resolve()))

    reg_ckpt = torch.load(str(Path(args.reg_weights).resolve()), map_location=device)
    reg_model = build_model().to(device)
    reg_model.load_state_dict(reg_ckpt["model_state_dict"])
    reg_model.eval()

    image_size = int(reg_ckpt.get("image_size", 224))
    reg_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    run_dir = Path(args.project).resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    det_results = det_model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        save_txt=True,
        project=str(Path(args.project).resolve()),
        name=args.name,
    )

    output = {}

    for r in det_results:
        image_path = Path(r.path)
        image_name = image_path.name
        output[image_name] = []

        if r.boxes is None or len(r.boxes) == 0:
            continue

        image = Image.open(image_path).convert("RGB")

        for box in r.boxes.xyxy.cpu().numpy():
            x_min, y_min, x_max, y_max = [float(v) for v in box]
            if x_max <= x_min or y_max <= y_min:
                continue

            crop = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            crop_tensor = reg_tf(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = reg_model(crop_tensor).squeeze(0).cpu().numpy().tolist()

            obj = decode_params(pred, x_min, y_min, x_max, y_max)
            output[image_name].append(obj)

    if args.save_json:
        json_path = run_dir / "predictions_annotations.json"
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {json_path}")

    if args.save_hyperbola_vis:
        vis_dir = run_dir / "hyperbola_vis"
        for image_name, objects in output.items():
            image_path = Path(args.source) / image_name if Path(args.source).is_dir() else Path(args.source)
            if image_path.exists():
                draw_hyperbolas_on_image(image_path, objects, vis_dir / image_name)
        print(f"Saved hyperbola visualization images to: {vis_dir}")

    print("Inference finished.")


if __name__ == "__main__":
    main()
