"""Faster R-CNN (torchvision, ResNet-50-FPN v2) on the same prepared dataset, same evaluation protocol.

    python train_frcnn.py --data datasets/ds2_f0_s100 --exp frcnn_r50_pre --seed 0            # COCO-pretrained
    python train_frcnn.py --data datasets/ds2_f0_s100 --exp frcnn_r50_scratch --seed 0 --scratch

Augmentation: horizontal flip (p=0.5) + torchvision's internal resize (min_size=max_size=imgsz). This is the
usual Faster R-CNN recipe; it is weaker than YOLO's mosaic pipeline and that is reported as such.
Model selection: best val AP50 (evaluated every --eval-every epochs). Predictions exported at score >= 0.001.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402
from torchvision.transforms.functional import to_tensor  # noqa: E402

from evaluate import ap50, collect, evaluate_run  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class YoloDirDataset(Dataset):
    """Reads <root>/images/<phase>/*.jpg + <root>/labels/<phase>/*.txt (YOLO format)."""

    def __init__(self, root: Path, phase: str, train: bool):
        self.img_dir = root / "images" / phase
        self.lbl_dir = root / "labels" / phase
        self.files = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        boxes = []
        lp = self.lbl_dir / f"{p.stem}.txt"
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        x = to_tensor(img)
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
            if len(boxes):
                boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
        target = {"boxes": boxes, "labels": torch.ones((len(boxes),), dtype=torch.int64)}
        return x, target, p.stem


def collate(b):
    return tuple(zip(*b))


def build_model(pretrained: bool, imgsz: int):
    if pretrained:
        m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=imgsz, max_size=imgsz,
                                       box_score_thresh=0.001, box_detections_per_img=300)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    else:
        m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=2, trainable_backbone_layers=5,
                                       min_size=imgsz, max_size=imgsz, box_score_thresh=0.001, box_detections_per_img=300)
    return m


@torch.no_grad()
def predict(model, ds, device, batch=4):
    model.eval()
    preds = {}
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=0)
    for xs, _, stems in dl:
        outs = model([x.to(device) for x in xs])
        for s, o in zip(stems, outs):
            b = o["boxes"].cpu().numpy(); sc = o["scores"].cpu().numpy()
            preds[s] = [[float(a), float(bb), float(c), float(d), float(e)] for (a, bb, c, d), e in zip(b, sc)]
    return preds


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None, help="sub-directory under <runs>/<dataset>/<exp>/ (default seed<k>; "
                    "the 5-fold runner uses f<fold>_seed<k>, the layout basic_ds2/aggregate_cv.py reads)")
    ap.add_argument("--weights", default="best", choices=["best", "last"],
                    help="best = highest val AP50 (default); last = final epoch, no val-based selection (5-fold protocol)")
    ap.add_argument("--scratch", action="store_true", help="no COCO / ImageNet weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    manifest = json.load(open(data_dir / "manifest.json", encoding="utf-8"))
    dataset = manifest["dataset"]
    run_dir = Path(args.runs).resolve() / dataset / args.exp / (args.run_name or f"seed{args.seed}")
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pretrained = not args.scratch

    train_ds = YoloDirDataset(data_dir, "train", train=True)
    val_ds = YoloDirDataset(data_dir, "val", train=False)
    test_ds = YoloDirDataset(data_dir, "test", train=False)
    gt = json.load(open(data_dir / "gt_boxes.json", encoding="utf-8"))
    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.workers,
                    generator=g, drop_last=False, persistent_workers=args.workers > 0)

    model = build_model(pretrained, args.imgsz).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    iters = args.epochs * len(dl)
    warm = min(200, max(1, len(dl)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, iters - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_ap, best_ep, log = -1.0, -1, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xs, ts, _ in dl:
            xs = [x.to(device) for x in xs]
            ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                losses = model(xs, ts)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
        rec = {"epoch": ep, "train_loss": tot / max(1, len(dl)), "lr": opt.param_groups[0]["lr"]}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            pv = predict(model, val_ds, device)
            s, t, n = collect(pv, gt["val"])
            rec["val_AP50"] = ap50(s, t, n)
            if rec["val_AP50"] > best_ap:
                best_ap, best_ep = rec["val_AP50"], ep
                torch.save(model.state_dict(), run_dir / "train" / "best.pt")
        log.append(rec)
        print(json.dumps(rec))
    torch.save(model.state_dict(), run_dir / "train" / "last.pt")
    with open(run_dir / "train" / "results.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    train_minutes = (time.time() - t0) / 60

    if args.weights == "best":
        model.load_state_dict(torch.load(run_dir / "train" / "best.pt", map_location=device))
    for phase, ds in (("val", val_ds), ("test", test_ds)):
        with open(run_dir / f"preds_{phase}.json", "w", encoding="utf-8") as f:
            json.dump(predict(model, ds, device), f)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        x = [torch.zeros(3, args.imgsz, args.imgsz, device=device)]
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x)
        gflops = round(fc.get_total_flops() / 1e9, 2)
    except Exception as e:
        print("FLOPs count failed:", e)
    info = dict(dataset=dataset, exp=args.exp, seed=args.seed, fold=manifest.get("fold"),
                box_source="hbat" if manifest.get("box_source") == "hbat" else "official",
                model_selection=("last epoch (no val-based selection)" if args.weights == "last"
                                 else "best val AP50"),
                model="fasterrcnn_resnet50_fpn_v2",
                pretrained=pretrained, band=False, framework=f"torchvision {torchvision.__version__}",
                torch=torch.__version__, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                train_minutes=round(train_minutes, 2), params_M=round(n_params, 3), GFLOPs=gflops,
                best_epoch=best_ep, best_val_AP50=best_ap, data_dir=str(data_dir), manifest=manifest, args=vars(args),
                pred_conf=0.001, pred_nms_iou=0.5)
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    m = evaluate_run(run_dir, data_dir)
    t = m["test"]["at_selected"]
    print(f"[{dataset}/{args.exp}/seed{args.seed}] val-selected conf {m['val']['selected_conf']:.2f} -> "
          f"test F1 {t['F1']:.4f} (P {t['P']:.3f} R {t['R']:.3f}) AP50 {m['test']['AP50']:.4f} | "
          f"{n_params:.2f} M params, {gflops} GFLOPs, {train_minutes:.1f} min")


if __name__ == "__main__":
    main()
