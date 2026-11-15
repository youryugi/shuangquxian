"""Train an ultralytics detector (YOLOv8 / YOLO11 / RT-DETR), optionally with the band plug-in, then export
raw predictions on val/test and evaluate with the unified protocol.

    python train_ultralytics.py --data datasets/ds2_f0_s100 --model yolov8s.pt --exp yolov8s_pre --seed 0
    python train_ultralytics.py --data datasets/ds2_f0_s100 --model yolov8s.pt --exp yolov8s_pre_band --seed 0 --band
    python train_ultralytics.py --data datasets/ds2_f0_s100 --model yolov8s.yaml --exp yolov8s_scratch --seed 0
    python train_ultralytics.py --data datasets/ds2_f0_s100 --model rtdetr-l.pt --exp rtdetr_l_pre --seed 0

model ending in .pt  -> COCO-pretrained weights (downloaded by ultralytics on first use)
model ending in .yaml -> random initialisation ("scratch")

Outputs in --runs/<dataset>/<exp>/seed<k>/ :
    ultralytics training directory (weights/best.pt, results.csv, train_batch*_band.jpg alignment plots for --band)
    preds_val.json, preds_test.json   raw detections, conf >= 0.001, original pixel coordinates
    run_info.json                     model / params / GFLOPs / timing / all arguments
    metrics.json                      from evaluate.evaluate_run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from evaluate import evaluate_run  # noqa: E402


def export_predictions(model, data_dir: Path, phase: str, out_path: Path, imgsz: int, conf=0.001, iou=0.7,
                       max_det=300, batch=16, device=0):
    gt = json.load(open(data_dir / "gt_boxes.json", encoding="utf-8"))[phase]
    img_dir = data_dir / "images" / phase
    files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
    stems = sorted(gt.keys())
    preds = {}
    for i in range(0, len(stems), batch):
        chunk = [str(files[s]) for s in stems[i:i + batch]]
        results = model.predict(source=chunk, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False,
                                device=device, half=False, augment=False)
        for s, r in zip(stems[i:i + batch], results):
            if r.boxes is None or len(r.boxes) == 0:
                preds[s] = []
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            preds[s] = [[float(a), float(b), float(c), float(d), float(e)] for (a, b, c, d), e in zip(xyxy, sc)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset directory from prepare_data.py")
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8s.pt | yolov8s.yaml | yolo11s.pt | rtdetr-l.pt ...")
    ap.add_argument("--exp", required=True, help="experiment name (one directory per exp, one sub-dir per seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=100, help="early-stopping patience (epochs)")
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--pred-conf", type=float, default=0.001)
    ap.add_argument("--pred-iou", type=float, default=0.7, help="NMS IoU used when exporting predictions")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra ultralytics overrides, e.g. '{\"lr0\":0.01}'")
    # band plug-in
    ap.add_argument("--band", action="store_true", help="enable hyperbola-band auxiliary supervision (YOLO only)")
    ap.add_argument("--band-gain", type=float, default=1.0)
    ap.add_argument("--band-layer", type=int, default=15)
    ap.add_argument("--band-stride", type=int, default=4)
    ap.add_argument("--band-target", default="soft", choices=["soft", "hard"])
    ap.add_argument("--band-warmup", type=int, default=3)
    ap.add_argument("--band-hidden", type=int, default=64)
    ap.add_argument("--mask-dir", default="masks_band", help="masks_band | masks_box | any sibling of images/")
    ap.add_argument("--mask-missing", default="error", choices=["error", "zero"])
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    manifest = json.load(open(data_dir / "manifest.json", encoding="utf-8"))
    dataset = manifest["dataset"]
    # absolute path: ultralytics prefixes a *relative* project with its own runs_dir/<task>/
    run_dir = Path(args.runs).resolve() / dataset / args.exp / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    is_rtdetr = "rtdetr" in args.model.lower()
    pretrained = args.model.endswith(".pt")
    if args.band and is_rtdetr:
        raise SystemExit("--band is implemented for YOLO (DetectionTrainer) only")

    overrides = dict(
        model=args.model, data=str(data_dir / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, device=args.device, workers=args.workers, project=str(run_dir),
        name="train", exist_ok=True, patience=args.patience, close_mosaic=args.close_mosaic, plots=not args.no_plots,
        verbose=False, pretrained=pretrained,
    )
    overrides.update(json.loads(args.extra))

    t0 = time.time()
    if args.band:
        from band_yolo import train_band
        band = dict(gain=args.band_gain, layer=args.band_layer, stride=args.band_stride, target=args.band_target,
                    warmup_epochs=args.band_warmup, hidden=args.band_hidden, mask_dir=args.mask_dir,
                    mask_missing=args.mask_missing)
        trainer = train_band(overrides, band)
    elif is_rtdetr:
        from ultralytics.models.rtdetr.train import RTDETRTrainer
        from ultralytics.data.augment import Compose, RandomPerspective

        class RTDETRTrainerFixed(RTDETRTrainer):
            """ultralytics 8.4.70: RTDETRDataset builds v8_transforms(stretch=True), whose RandomPerspective keeps
            the 2*imgsz mosaic canvas instead of cropping back to imgsz (YOLO's path does crop). That trains
            RT-DETR at 1280 px and quadruples memory. Pin the affine output size to imgsz, as for YOLO."""

            def build_dataset(self, img_path, mode="val", batch=None):
                ds = super().build_dataset(img_path, mode, batch)
                if mode == "train":
                    def pin(t):
                        if isinstance(t, Compose):
                            for x in t.transforms:
                                pin(x)
                        elif isinstance(t, RandomPerspective):
                            t.size = (self.args.imgsz, self.args.imgsz)
                        elif getattr(t, "pre_transform", None) is not None:
                            pin(t.pre_transform)
                    pin(ds.transforms)
                return ds

        trainer = RTDETRTrainerFixed(overrides=overrides)
        trainer.train()
    else:
        from ultralytics.models.yolo.detect import DetectionTrainer
        trainer = DetectionTrainer(overrides=overrides)
        trainer.train()
    train_minutes = (time.time() - t0) / 60
    best = trainer.best if Path(trainer.best).exists() else trainer.last

    # predictions with the trained weights
    if args.band:
        import band_yolo  # noqa: F401  (needed to unpickle BandAuxDetectionModel)
    if is_rtdetr:
        from ultralytics import RTDETR
        model = RTDETR(str(best))
    else:
        from ultralytics import YOLO
        model = YOLO(str(best))
    device = 0 if str(args.device) not in ("cpu",) else "cpu"
    for phase in ("val", "test"):
        export_predictions(model, data_dir, phase, run_dir / f"preds_{phase}.json", args.imgsz,
                           conf=args.pred_conf, iou=args.pred_iou, device=device)

    # bookkeeping
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    net = model.model
    n_aux = sum(p.numel() for p in net.aux_head.parameters()) if hasattr(net, "aux_head") else 0
    info = dict(
        # inference-time size: the band head is training-only, so it is excluded (GFLOPs already exclude it)
        params_M_inference=round((get_num_params(net) - n_aux) / 1e6, 3), params_M_aux_head=round(n_aux / 1e6, 4),
        dataset=dataset, exp=args.exp, seed=args.seed, model=args.model, pretrained=pretrained, band=args.band,
        band_cfg=(dict(gain=args.band_gain, layer=args.band_layer, stride=args.band_stride, target=args.band_target,
                       warmup=args.band_warmup, mask_dir=args.mask_dir) if args.band else None),
        framework=f"ultralytics {__import__('ultralytics').__version__}", torch=torch.__version__,
        imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, train_minutes=round(train_minutes, 2),
        params_M=round(get_num_params(net) / 1e6, 3), GFLOPs=round(float(get_flops(net, args.imgsz)), 2),
        best_weights=str(best), data_dir=str(data_dir), manifest=manifest, args=vars(args),
        pred_conf=args.pred_conf, pred_nms_iou=args.pred_iou,
    )
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    m = evaluate_run(run_dir, data_dir)
    t = m["test"]["at_selected"]
    print(f"[{dataset}/{args.exp}/seed{args.seed}] val-selected conf {m['val']['selected_conf']:.2f} -> "
          f"test F1 {t['F1']:.4f} (P {t['P']:.3f} R {t['R']:.3f}) AP50 {m['test']['AP50']:.4f} | "
          f"{info['params_M']} M params, {info['GFLOPs']} GFLOPs, {train_minutes:.1f} min")


if __name__ == "__main__":
    main()
