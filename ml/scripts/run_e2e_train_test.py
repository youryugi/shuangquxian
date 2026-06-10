import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from e2e_detector_model import HyperbolaE2EDetector, compute_loss, decode_predictions, hyperbola_to_bbox_from_params


# ===========================
# One-click configuration
# ===========================
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = (SCRIPT_DIR / "../../").resolve()
ANNOTATIONS_PATH = DATA_DIR / "annotations.json"

IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.5
SEED = 2

DEVICE_PREFERENCE = "cuda"  # "cuda" or "cpu"

RUN_TRAIN = True
RUN_TEST = True

RUN_NAME = "e2e_oneclick"
OUTPUT_ROOT = (SCRIPT_DIR / "../runs_e2e").resolve()

TEST_SOURCE = DATA_DIR  # single image path or folder path
TEST_CONF = 0.35
TEST_NMS_IOU = 0.5
SAVE_JSON = True
SAVE_HYPERBOLA_VIS = True

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class HyperbolaE2EDataset(Dataset):
    def __init__(self, image_items: List[Dict], img_size: int, stride: int):
        self.image_items = image_items
        self.img_size = img_size
        self.stride = stride
        self.grid_h = img_size // stride
        self.grid_w = img_size // stride
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx: int):
        item = self.image_items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image_w, image_h = image.size
        x = self.tf(image)

        target_obj = torch.zeros((1, self.grid_h, self.grid_w), dtype=torch.float32)
        target_bbox = torch.zeros((4, self.grid_h, self.grid_w), dtype=torch.float32)
        target_param = torch.zeros((5, self.grid_h, self.grid_w), dtype=torch.float32)
        area_map = torch.zeros((self.grid_h, self.grid_w), dtype=torch.float32)

        for obj in item["objects"]:
            x_vertex = float(obj["x_vertex"])
            y_vertex = float(obj["y_vertex"])
            width = max(2.0, float(obj["width"]))
            height = max(1.0, float(obj["height"]))
            thickness = max(1.0, float(obj["thickness"]))

            x_min, y_min, x_max, y_max = hyperbola_to_bbox_from_params(
                x_vertex,
                y_vertex,
                width,
                height,
                thickness,
                float(image_w),
                float(image_h),
            )
            bw = max(1.0, x_max - x_min)
            bh = max(1.0, y_max - y_min)
            area = bw * bh

            cx_n = ((x_min + x_max) / 2.0) / image_w
            cy_n = ((y_min + y_max) / 2.0) / image_h
            bw_n = bw / image_w
            bh_n = bh / image_h

            cx = cx_n * self.img_size
            cy = cy_n * self.img_size
            gx = int(cx // self.stride)
            gy = int(cy // self.stride)

            if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
                continue

            if area <= area_map[gy, gx]:
                continue

            area_map[gy, gx] = area
            target_obj[0, gy, gx] = 1.0

            tx = (cx / self.stride) - gx
            ty = (cy / self.stride) - gy
            target_bbox[0, gy, gx] = float(max(0.0, min(1.0, tx)))
            target_bbox[1, gy, gx] = float(max(0.0, min(1.0, ty)))
            target_bbox[2, gy, gx] = float(max(0.0, min(1.0, bw_n)))
            target_bbox[3, gy, gx] = float(max(0.0, min(1.0, bh_n)))

            xv_n = x_vertex / image_w
            yv_n = y_vertex / image_h
            w_n = width / image_w
            h_n = height / image_h
            t_n = thickness / image_h
            target_param[0, gy, gx] = float(max(0.0, min(1.0, xv_n)))
            target_param[1, gy, gx] = float(max(0.0, min(1.0, yv_n)))
            target_param[2, gy, gx] = float(max(0.0, min(1.0, w_n)))
            target_param[3, gy, gx] = float(max(0.0, min(1.0, h_n)))
            target_param[4, gy, gx] = float(max(0.0, min(1.0, t_n)))

        return x, target_obj, target_bbox, target_param


def pick_device() -> torch.device:
    if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_items(data_dir: Path, annotations_path: Path) -> List[Dict]:
    with annotations_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    items = []
    for image_name, objects in ann.items():
        image_path = data_dir / image_name
        if not image_path.exists() or image_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        items.append({"image_path": str(image_path), "objects": objects})
    return items


def split_items(items: List[Dict], train_ratio: float, seed: int):
    random.seed(seed)
    random.shuffle(items)
    split_idx = max(1, min(len(items) - 1, int(len(items) * train_ratio)))
    return items[:split_idx], items[split_idx:]


def run_eval(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, t_obj, t_bbox, t_param in loader:
            x = x.to(device)
            t_obj = t_obj.to(device)
            t_bbox = t_bbox.to(device)
            t_param = t_param.to(device)

            pred = model(x)
            losses = compute_loss(pred, t_obj, t_bbox, t_param)
            bs = x.shape[0]
            total += float(losses["loss"].item()) * bs
            count += bs
    return total / max(1, count)


def train_e2e(run_dir: Path, device: torch.device) -> Path:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"annotations.json not found: {ANNOTATIONS_PATH}")

    model = HyperbolaE2EDetector(width=64)
    stride = model.stride
    if IMG_SIZE % stride != 0:
        raise ValueError(f"IMG_SIZE must be divisible by stride={stride}")

    items = load_items(DATA_DIR, ANNOTATIONS_PATH)
    if len(items) < 2:
        raise RuntimeError("Need at least 2 images to split train/val.")

    train_items, val_items = split_items(items, TRAIN_RATIO, SEED)

    train_ds = HyperbolaE2EDataset(train_items, img_size=IMG_SIZE, stride=stride)
    val_ds = HyperbolaE2EDataset(val_items, img_size=IMG_SIZE, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_path = run_dir / "best.pt"
    best_val = float("inf")

    print("=== Train Config ===")
    print(f"Device: {device}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Train/Val images: {len(train_items)}/{len(val_items)}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, ImgSize: {IMG_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for x, t_obj, t_bbox, t_param in train_loader:
            x = x.to(device)
            t_obj = t_obj.to(device)
            t_bbox = t_bbox.to(device)
            t_param = t_param.to(device)

            pred = model(x)
            losses = compute_loss(pred, t_obj, t_bbox, t_param)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            bs = x.shape[0]
            total_loss += float(losses["loss"].item()) * bs
            n += bs

        train_loss = total_loss / max(1, n)
        val_loss = run_eval(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": IMG_SIZE,
                    "stride": stride,
                },
                best_path,
            )

    print("Training finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best val loss: {best_val:.6f}")
    return best_path


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
        y_center = y_vertex + height * (dx ** 2)
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


def to_annotation(decoded_item, image_w: int, image_h: int):
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

    return {
        "label": "hyperbola",
        "score": round(float(decoded_item["score"]), 4),
        "x_vertex": round(x_vertex, 2),
        "y_vertex": round(y_vertex, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "thickness": round(thickness, 2),
    }


def test_e2e(weights_path: Path, run_dir: Path, device: torch.device):
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    ckpt = torch.load(str(weights_path), map_location="cpu")
    img_size = int(ckpt.get("img_size", IMG_SIZE))
    stride = int(ckpt.get("stride", 16))

    model = HyperbolaE2EDetector(width=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    source = TEST_SOURCE.resolve()
    image_paths = iter_source_images(source)
    if not image_paths:
        raise RuntimeError(f"No test images found in: {source}")

    print("=== Test Config ===")
    print(f"Test source: {source}")
    print(f"Images: {len(image_paths)}")
    print(f"Conf: {TEST_CONF}, NMS IoU: {TEST_NMS_IOU}")

    out = {}
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size

            x = tf(image).unsqueeze(0).to(device)
            pred = model(x)[0].cpu()
            decoded = decode_predictions(pred, img_size=img_size, stride=stride, conf_thr=TEST_CONF, nms_iou=TEST_NMS_IOU)
            out[image_path.name] = [to_annotation(d, image_w=image_w, image_h=image_h) for d in decoded]

    if SAVE_JSON:
        json_path = run_dir / "predictions_annotations.json"
        json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {json_path}")

    if SAVE_HYPERBOLA_VIS:
        vis_dir = run_dir / "hyperbola_vis"
        for image_path in image_paths:
            objs = out.get(image_path.name, [])
            draw_hyperbolas_on_image(image_path, objs, vis_dir / image_path.name)
        print(f"Saved visualization: {vis_dir}")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = pick_device()

    run_dir = OUTPUT_ROOT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"

    if RUN_TRAIN:
        best_path = train_e2e(run_dir, device)

    if RUN_TEST:
        test_e2e(best_path, run_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()
