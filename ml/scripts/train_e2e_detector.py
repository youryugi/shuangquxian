import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from e2e_detector_model import HyperbolaE2EDetector, compute_loss, hyperbola_to_bbox_from_params

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class HyperbolaE2EDataset(Dataset):
    def __init__(self, image_items: List[Dict], img_size: int, stride: int):
        self.image_items = image_items
        self.img_size = img_size
        self.stride = stride
        self.grid_h = img_size // stride
        self.grid_w = img_size // stride
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

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

            # Keep larger object when multiple centers fall into the same grid cell.
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


def load_items(data_dir: Path, annotations_path: Path):
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


def main():
    parser = argparse.ArgumentParser(description="Train one-stage end-to-end hyperbola detector.")
    parser.add_argument("--data-dir", type=str, default="../../", help="Directory with images and annotations.json")
    parser.add_argument("--annotations", type=str, default="", help="Path to annotations.json")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--output-dir", type=str, default="../runs_e2e", help="Output root")
    parser.add_argument("--name", type=str, default="e2e_detector", help="Run name")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    annotations_path = Path(args.annotations).resolve() if args.annotations else data_dir / "annotations.json"

    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {annotations_path}")

    model = HyperbolaE2EDetector(width=64)
    stride = model.stride
    if args.img_size % stride != 0:
        raise ValueError(f"img-size must be divisible by stride={stride}")

    items = load_items(data_dir, annotations_path)
    if len(items) < 2:
        raise RuntimeError("Need at least 2 images to split train/val.")

    train_items, val_items = split_items(items, args.train_ratio, args.seed)

    train_ds = HyperbolaE2EDataset(train_items, img_size=args.img_size, stride=stride)
    val_ds = HyperbolaE2EDataset(val_items, img_size=args.img_size, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    run_dir = Path(args.output_dir).resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
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
                    "img_size": args.img_size,
                    "stride": stride,
                },
                best_path,
            )

    print("Training finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
