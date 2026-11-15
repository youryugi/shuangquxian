import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


def det_to_hyperbola(box_xywh, thickness_ratio: float = 0.25):
    x_center, y_center, width, bbox_height = [float(v) for v in box_xywh]
    thickness_ratio = max(0.05, min(0.8, thickness_ratio))

    thickness = max(1.0, bbox_height * thickness_ratio)
    height = max(1.0, bbox_height - thickness)
    y_vertex = y_center - (height / 2.0)

    return {
        "label": "hyperbola",
        "x_vertex": round(x_center, 2),
        "y_vertex": round(y_vertex, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "thickness": round(thickness, 2),
    }


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


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference and optionally export annotations-like JSON.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image path, folder path, or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="Device: 0/cpu")
    parser.add_argument("--project", type=str, default="../runs", help="Inference output folder")
    parser.add_argument("--name", type=str, default="predict_hyperbola", help="Inference experiment name")
    parser.add_argument("--thickness-ratio", type=float, default=0.25, help="Convert bbox->hyperbola thickness ratio")
    parser.add_argument("--save-json", action="store_true", help="Export predictions as annotations-like JSON")
    parser.add_argument("--save-hyperbola-vis", action="store_true", help="Save images with hyperbola-shaped overlays")
    args = parser.parse_args()

    model = YOLO(str(Path(args.weights).resolve()))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        save_txt=True,
        project=str(Path(args.project).resolve()),
        name=args.name,
    )

    print("Inference finished.")

    out = {}
    for r in results:
        image_name = Path(r.path).name
        out[image_name] = []
        if r.boxes is None:
            continue
        xywh = r.boxes.xywh.cpu().numpy()
        for one_box in xywh:
            out[image_name].append(det_to_hyperbola(one_box, args.thickness_ratio))

    run_dir = Path(args.project).resolve() / args.name

    if args.save_json:
        output_json = run_dir / "predictions_annotations.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {output_json}")

    if args.save_hyperbola_vis:
        vis_dir = run_dir / "hyperbola_vis"
        for r in results:
            image_name = Path(r.path).name
            draw_hyperbolas_on_image(Path(r.path), out.get(image_name, []), vis_dir / image_name)
        print(f"Saved hyperbola visualization images to: {vis_dir}")


if __name__ == "__main__":
    main()
