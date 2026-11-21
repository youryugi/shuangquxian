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


# ----------------------------------------------------------------------------- paired-initialisation check
def _sha(arrays):
    import hashlib
    h = hashlib.sha1()
    for a in arrays:
        h.update(a.detach().float().cpu().numpy().tobytes() if torch.is_tensor(a) else bytes(a))
    return h.hexdigest()[:16]


def with_fingerprint(base, seed_data_order=True):
    """Subclass a trainer so that it records (a) a hash of the detection weights right before training starts and
    (b) a hash of the first training batch (images after augmentation, boxes). Two arms of a paired comparison must
    have identical fingerprints; otherwise the Δ mixes the method with initialisation / data-order noise
    (lesson of basic_ds2 Round 2–4, init_fingerprint). The aux band head is excluded from (a)."""

    class _FP(base):
        seed_data_order = True

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            """ultralytics 8.4.70 seeds the train DataLoader's generator with a constant (build.py), so shuffling and
            the worker seeds (mosaic / affine / flip draws) are the SAME for every --seed; only the re-initialised
            detection head changes. Re-seed that generator from --seed so that a seed also changes data order and
            augmentation, as in basic_ds2. All arms with the same seed still get the same stream (paired)."""
            dl = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode == "train" and self.seed_data_order and getattr(dl, "generator", None) is not None:
                dl.generator.manual_seed(6148914691236517205 + 1000003 * int(self.args.seed))  # sampler shares it
                old = dl.iterator
                if hasattr(old, "_shutdown_workers"):
                    old._shutdown_workers()
                dl.iterator = dl._get_iterator()   # redraw the worker base seed from the re-seeded generator
            return dl

        def _setup_train(self):
            super()._setup_train()
            sd = self.model.state_dict()
            keys = [k for k in sd if not k.startswith("aux_head.")]
            self.fingerprint = {"init_det_weights": _sha(sd[k] for k in keys), "n_det_tensors": len(keys)}

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            fp = getattr(self, "fingerprint", None)
            if fp is not None and "first_batch" not in fp:
                fp["first_batch"] = _sha([batch["img"][:, :3], batch["bboxes"], batch["cls"], batch["batch_idx"]])
            return batch

    _FP.seed_data_order = seed_data_order
    _FP.__name__ = base.__name__ + "FP"
    return _FP


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
    ap.add_argument("--run-name", default=None, help="sub-directory under <runs>/<dataset>/<exp>/ (default seed<k>; "
                    "the 5-fold runner uses f<fold>_seed<k>, the layout basic_ds2/aggregate_cv.py reads)")
    ap.add_argument("--weights", default="best", choices=["best", "last"],
                    help="checkpoint used for predictions. best = ultralytics fitness on val (8.4.70: mAP50-95 only); "
                         "last = fixed epoch budget, no val-based selection (basic_ds2 protocol)")
    ap.add_argument("--fixed-data-order", action="store_true",
                    help="keep ultralytics' constant DataLoader seed (data order / augmentation identical for every "
                         "--seed). Default: derive it from --seed")
    ap.add_argument("--save-band", action="store_true",
                    help="band arms: also export the aux head's band map on val/test (PNG) and pixel counts for band IoU")
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
    run_dir = Path(args.runs).resolve() / dataset / args.exp / (args.run_name or f"seed{args.seed}")
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
        from band_yolo import BandTrainer, train_band
        band = dict(gain=args.band_gain, layer=args.band_layer, stride=args.band_stride, target=args.band_target,
                    warmup_epochs=args.band_warmup, hidden=args.band_hidden, mask_dir=args.mask_dir,
                    mask_missing=args.mask_missing)
        trainer = train_band(overrides, band, trainer_cls=with_fingerprint(BandTrainer, not args.fixed_data_order))
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
        trainer = with_fingerprint(DetectionTrainer, not args.fixed_data_order)(overrides=overrides)
        trainer.train()
    train_minutes = (time.time() - t0) / 60
    if args.weights == "last" or not Path(trainer.best).exists():
        best = trainer.last
    else:
        best = trainer.best

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

    # band output of the aux head (band arms only): PNG per image + pooled pixel counts for band IoU
    band_quality = None
    if args.save_band and hasattr(model.model, "aux_head"):
        import cv2
        import numpy as np
        from band_yolo import band_counts, predict_band
        dev = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        band_quality = {}
        for phase in ("val", "test"):
            gt = json.load(open(data_dir / "gt_boxes.json", encoding="utf-8"))[phase]
            img_dir = data_dir / "images" / phase
            files = {p.stem: p for p in img_dir.iterdir()}
            maps = predict_band(model.model, [files[s] for s in sorted(gt)], args.imgsz, dev)
            out_dir = run_dir / f"band_{phase}"
            out_dir.mkdir(exist_ok=True)
            tot = {"inter": 0, "union": 0, "pred": 0, "gt": 0}
            for s, m in maps.items():
                cv2.imwrite(str(out_dir / f"{s}.png"), m)
                g = cv2.imread(str(data_dir / "masks_band" / phase / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                for k, v in band_counts(m, g).items():
                    tot[k] += v
            band_quality[phase] = {**tot, "iou": tot["inter"] / max(tot["union"], 1),
                                   "precision": tot["inter"] / max(tot["pred"], 1),
                                   "recall": tot["inter"] / max(tot["gt"], 1), "threshold": 0.5,
                                   "note": "vs masks_band (HBAT band) even for the box-mask arm"}

    # bookkeeping
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    net = model.model
    n_aux = sum(p.numel() for p in net.aux_head.parameters()) if hasattr(net, "aux_head") else 0
    info = dict(
        # inference-time size: the band head is training-only, so it is excluded (GFLOPs already exclude it)
        params_M_inference=round((get_num_params(net) - n_aux) / 1e6, 3), params_M_aux_head=round(n_aux / 1e6, 4),
        dataset=dataset, exp=args.exp, seed=args.seed, fold=manifest.get("fold"),
        box_source="hbat" if manifest.get("box_source") == "hbat" else "official",
        model_selection=("last epoch (no val-based selection)" if best == trainer.last
                         else "best.pt by ultralytics fitness on val (mAP50-95)"),
        fingerprint=getattr(trainer, "fingerprint", None), band_quality=band_quality,
        data_order=("constant (ultralytics default)" if args.fixed_data_order else "seeded by --seed"),
        model=args.model, pretrained=pretrained, band=args.band,
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
