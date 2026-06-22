"""
小样本对比实验主驱动：固定 seed=0，train 按百分比，4 种方法同 train/test，统一评估。
方法：ours(你的方法) / cnn(朴素CNN) / yolo_scratch / yolo_pretrained。

跑：  C:/Users/79152/.conda/envs/gpr/python.exe few_shot_exp/run_fewshot.py
（训练较久，建议后台跑。先用默认 20% 单点跑通；扫描改 TRAIN_RATIOS。）
"""
import os
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

import fewshot_common as fc

exp    = fc.exp
device = exp.device
cnn    = fc.load_mod("bbox_cnn_baseline.py", "cnn_base")
yc     = fc.load_mod("yolo_common.py", "yc")

# ── 配置 ──────────────────────────────────────────────────────────────────────
TRAIN_RATIOS   = [0.2]                       # 先单点；扫描改 [0.1, 0.2, 0.3]
METHODS        = ["ours", "cnn", "yolo_scratch", "yolo_pretrained"]
FEWSHOT_EPOCHS = exp.num_epochs              # 想快速验证可临时调小
HM_THRESH      = exp.HM_THRESH               # 统一置信阈值（4 方法一致）


# ── 你的方法 ──────────────────────────────────────────────────────────────────
def run_ours(full, train_idx, val_idx, test_idx, work):
    tl = DataLoader(exp.AugWrapper(Subset(full, train_idx)), batch_size=exp.batch_size,
                    shuffle=True, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    vl = DataLoader(Subset(full, val_idx), batch_size=exp.batch_size,
                    shuffle=False, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    model = exp.HyperbolaNet(in_ch=1, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=exp.LR)
    best, bp = float("inf"), os.path.join(work, "ours_best.pth")
    for ep in range(1, FEWSHOT_EPOCHS + 1):
        exp.train_one_epoch(model, tl, opt, device)
        va = exp.validate_one_epoch(model, vl, device)
        if va < best:
            best = va; torch.save(model.state_dict(), bp)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()

    preds = []
    for i in test_idx:
        meta = full[i][-1]
        _, mask, dets = exp.predict_single_image(
            model, meta["image_path"], exp.input_size, device,
            obj_thresh=HM_THRESH, nms_k=exp.nms_kernel, max_det=exp.max_det)
        preds.append({
            "boxes":    [fc.hyperbola_to_bbox(d) for d in dets],
            "scores":   [d["score"] for d in dets],
            "vertices": [(d["x_vertex"], d["y_vertex"]) for d in dets],
            "mask":     mask,
        })
    return preds


# ── 朴素 CNN ──────────────────────────────────────────────────────────────────
def run_cnn(train_idx, val_idx, test_paths, work):
    rect = os.path.join(os.path.dirname(cnn.exp.data_sources[0]["annotation_json"]), "annotations_rect.json")
    full = cnn.BBoxDataset(cnn.exp.data_sources[0]["image_dir"],
                           cnn.exp.data_sources[0]["annotation_json"], rect,
                           cnn.input_size, cnn.HM_STRIDE, cnn.HM_SIGMA)
    tl = DataLoader(cnn.AugWrapper(Subset(full, train_idx)), batch_size=cnn.batch_size,
                    shuffle=True, num_workers=0, collate_fn=cnn.collate)
    vl = DataLoader(Subset(full, val_idx), batch_size=cnn.batch_size,
                    shuffle=False, num_workers=0, collate_fn=cnn.collate)
    model = cnn.BBoxNet(in_ch=1, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cnn.LR)
    best, bp = float("inf"), os.path.join(work, "cnn_best.pth")
    for ep in range(1, FEWSHOT_EPOCHS + 1):
        cnn.train_one_epoch(model, tl, opt)
        va = cnn.validate(model, vl)
        if va < best:
            best = va; torch.save(model.state_dict(), bp)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()

    H, W = fc.input_size
    preds = []
    for p in test_paths:
        boxes, scores = cnn.predict_boxes(model, p)
        preds.append({
            "boxes": boxes, "scores": scores,
            "vertices": [fc.bbox_to_vertex(b) for b in boxes],
            "mask": fc.solid_bbox_mask(boxes, H, W),
        })
    return preds


# ── YOLO（scratch / pretrained）───────────────────────────────────────────────
def run_yolo(model_spec, tag, train_idx, val_idx, test_paths, work):
    from ultralytics import YOLO
    yc._patch_torch_save_for_bytesio()
    names, paths, rect = yc.prepare_yolo_data(work)
    tl = os.path.join(work, f"{tag}_train.txt"); open(tl, "w", encoding="utf-8").write("\n".join(paths[i] for i in train_idx))
    vl = os.path.join(work, f"{tag}_val.txt");   open(vl, "w", encoding="utf-8").write("\n".join(paths[i] for i in val_idx))
    yaml = os.path.join(work, f"{tag}.yaml")
    open(yaml, "w", encoding="utf-8").write(f"path: {work}\ntrain: {tl}\nval: {vl}\nnames:\n  0: hyperbola\n")

    model = YOLO(model_spec)
    model.train(data=yaml, epochs=FEWSHOT_EPOCHS, imgsz=yc.IMGSZ, batch=exp.batch_size,
                seed=fc.SEED, project=work, name=tag, exist_ok=True, verbose=False, workers=2)
    bm = YOLO(os.path.join(work, tag, "weights", "best.pt"))

    H, W = fc.input_size
    preds = []
    for p in test_paths:
        res = bm.predict(p, imgsz=yc.IMGSZ, conf=HM_THRESH, max_det=exp.max_det, verbose=False)
        b = res[0].boxes
        boxes = b.xyxy.cpu().numpy().tolist() if len(b) else []
        scores = b.conf.cpu().numpy().tolist() if len(b) else []
        preds.append({
            "boxes": boxes, "scores": scores,
            "vertices": [fc.bbox_to_vertex(bx) for bx in boxes],
            "mask": fc.solid_bbox_mask(boxes, H, W),
        })
    return preds


def main():
    now = datetime.now()
    root = os.path.join(fc._HERE, f"run_{now.strftime('%m%d_%H%M')}")
    os.makedirs(root, exist_ok=True)
    full = exp.HyperbolaDataset(
        image_dir=exp.data_sources[0]["image_dir"],
        annotation_json=exp.data_sources[0]["annotation_json"],
        input_size=exp.input_size, hm_stride=exp.HM_STRIDE, sigma=exp.HM_SIGMA)
    n_total = len(full)
    print(f"Total samples: {n_total}  | methods={METHODS}  | ratios={TRAIN_RATIOS}")

    all_rows = []
    for ratio in TRAIN_RATIOS:
        train_idx, val_idx, test_idx = fc.make_fewshot_split(n_total, ratio)
        metas = [full[i][-1] for i in test_idx]
        test_paths = [m["image_path"] for m in metas]
        work = os.path.join(root, f"ratio{int(ratio*100):02d}")
        os.makedirs(work, exist_ok=True)
        print(f"\n{'='*70}\nratio={ratio}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n{'='*70}")

        results = {}
        if "ours" in METHODS:
            results["ours"] = fc.evaluate_method(run_ours(full, train_idx, val_idx, test_idx, work), metas)
        if "cnn" in METHODS:
            results["cnn"] = fc.evaluate_method(run_cnn(train_idx, val_idx, test_paths, work), metas)
        if "yolo_scratch" in METHODS:
            results["yolo_scratch"] = fc.evaluate_method(
                run_yolo("yolov8n.yaml", "yolo_scratch", train_idx, val_idx, test_paths, work), metas)
        if "yolo_pretrained" in METHODS:
            results["yolo_pretrained"] = fc.evaluate_method(
                run_yolo("yolov8n.pt", "yolo_pretrained", train_idx, val_idx, test_paths, work), metas)

        keys = ["bbox_P", "bbox_R", "bbox_F1", "vertex_recall", "mask_iou"]
        print(f"\n[ratio={ratio}]  ({len(train_idx)} train imgs)")
        print(f"{'method':>16}" + "".join(f"{k:>14}" for k in keys))
        for m in METHODS:
            r = results.get(m)
            if r:
                print(f"{m:>16}" + "".join(f"{r[k]:>14.4f}" for k in keys))
                row = {"ratio": ratio, "n_train": len(train_idx), "method": m}; row.update(r)
                all_rows.append(row)

    csv_path = os.path.join(root, "fewshot_results.csv")
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
