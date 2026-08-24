"""
Cross-framework reproduction of the invert_aug (true polarity-inversion
augmentation) ablation on torchvision's Faster R-CNN (ResNet-50-FPN) --
a two-stage CNN detector, architecturally and code-path unrelated to both
the YOLO CNN family and RT-DETR's transformer decoder. Both the backbone
and detection head are randomly initialized (weights=None,
weights_backbone=None) -- no ImageNet/COCO pretraining -- matching the
no-pretraining regime used in run_invert_ablation_multi_model.py, since
that's where the invert_aug effect was found to be largest and
statistically significant.

Reuses train_yolo_bbox.build_yolo_dataset() for the actual data
prep/split/invert_aug-copy logic, so the train/val composition for a
given seed is identical to the YOLO/RT-DETR runs -- only the detector
framework differs. Everything downstream (Dataset, model, training loop,
mAP evaluation) is plain torchvision + torchmetrics, since Faster R-CNN
isn't part of the ultralytics API and can't reuse train_yolo_bbox.train().

Trains with INVERT_AUG on and off, 5 random seeds each (10 runs total).
mAP50 / mAP50-95 are computed with torchmetrics' MeanAveragePrecision
(COCO-style, pycocotools-backed) to stay comparable to the ultralytics
numbers in the other ablation scripts.

Caveat: Faster R-CNN (ResNet-50-FPN, ~41M params) trained fully from
scratch on ~186-370 images (depending on invert_aug) is data-starved and
likely to converge much more slowly / to a much lower absolute mAP than
the YOLO models -- treat absolute numbers with caution, the relevant
comparison is ON vs OFF *within* this same framework, not against YOLO's
absolute mAP.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation_fasterrcnn.py
"""
import csv
import os
import random
import statistics

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from train_yolo_bbox import (
    DATA_DIR, WORK_DIR, EPOCHS, IMGSZ, BATCH, VAL_RATIO,
    POLARITY_DIR, INVERT_DIR, build_yolo_dataset,
)

SEEDS = [0, 1, 2, 3, 4]
WORK_ROOT = os.path.join(WORK_DIR, "invert_ablation_fasterrcnn")
RESULTS_CSV = os.path.join(WORK_ROOT, "results.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # background + hyperbola
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4


class YoloBoxDataset(Dataset):
    """Reads the images/{split} + labels/{split} layout build_yolo_dataset()
    produces (YOLO-format normalized `class_id xc yc w h` .txt files) and
    converts boxes to absolute-pixel x1y1x2y2, torchvision's Faster R-CNN
    convention (label 0 reserved for background, so class_id 0 -> label 1)."""

    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.names = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        img = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        w, h = img.size

        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes, labels = [], []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)  # 0 is background in torchvision's convention

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    return fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES,
    ).to(DEVICE)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        preds = model(images)
        preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


def run_one(invert_aug, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    work_dir = os.path.join(WORK_ROOT, f"invert_{'on' if invert_aug else 'off'}_seed{seed}")
    yaml_path = build_yolo_dataset(
        DATA_DIR, work_dir, VAL_RATIO, seed,
        polarity_dir=POLARITY_DIR, polarity_aug=False,
        invert_dir=INVERT_DIR, invert_aug=invert_aug,
    )
    dataset_dir = os.path.dirname(yaml_path)
    train_ds = YoloBoxDataset(os.path.join(dataset_dir, "images", "train"),
                               os.path.join(dataset_dir, "labels", "train"))
    val_ds = YoloBoxDataset(os.path.join(dataset_dir, "images", "val"),
                             os.path.join(dataset_dir, "labels", "val"))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(EPOCHS * 0.7), int(EPOCHS * 0.9)], gamma=0.1,
    )

    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}")

    map50, map5095 = evaluate(model, val_loader)
    print(f"[val] mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
    return map50, map5095


def main():
    os.makedirs(WORK_ROOT, exist_ok=True)
    rows = []
    for invert_aug in (True, False):
        for seed in SEEDS:
            print(f"\n=== [fasterrcnn, scratch] invert_aug={invert_aug} seed={seed} ===")
            map50, map5095 = run_one(invert_aug, seed)
            rows.append({"invert_aug": invert_aug, "seed": seed, "map50": map50, "map50_95": map5095})
            with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["invert_aug", "seed", "map50", "map50_95"])
                writer.writeheader()
                writer.writerows(rows)

    print("\n=== summary (Faster R-CNN, scratch) ===")
    for invert_aug in (True, False):
        group = [r for r in rows if r["invert_aug"] == invert_aug]
        map50s = [r["map50"] for r in group]
        map5095s = [r["map50_95"] for r in group]
        label = "invert_aug=ON " if invert_aug else "invert_aug=OFF"
        print(f"{label}  mAP50={statistics.mean(map50s):.4f}+/-{statistics.pstdev(map50s):.4f}  "
              f"mAP50-95={statistics.mean(map5095s):.4f}+/-{statistics.pstdev(map5095s):.4f}")
    print(f"\n[ablation] per-run results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
