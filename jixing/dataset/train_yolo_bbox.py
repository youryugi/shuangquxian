"""
Train / run a real YOLO (ultralytics) detector on the boxes produced by
biaozhu-yolo-bbox.py: a folder of images, each with a same-name .txt label
file (class_id xc yc w h, normalized 0..1) and a shared classes.txt.

Run inside the `gpr` conda env (already has ultralytics + torch + CUDA):

    conda activate gpr
    python train_yolo_bbox.py train  --data-dir jixing-zhengchang --epochs 100
    python train_yolo_bbox.py detect --weights yolo_runs/train/weights/best.pt --source jixing-zhengchang
"""
import os
import glob
import random
import shutil
import argparse

import yaml

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (default, help). Defaults are all 0 (every built-in augmentation
# off) so the baseline run is clean; pass a flag to turn a specific one back
# on for an ablation. (ultralytics' own out-of-the-box defaults are noted in
# each help string for reference.)
AUG_PARAMS = {
    "hsv_h":      (0.0, "HSV hue augmentation fraction (ultralytics default 0.015)"),
    "hsv_s":      (0.0, "HSV saturation augmentation fraction (ultralytics default 0.7)"),
    "hsv_v":      (0.0, "HSV value/brightness augmentation fraction (ultralytics default 0.4)"),
    "degrees":    (0.0, "rotation degrees (+/-)"),
    "translate":  (0.0, "translation fraction (+/-) (ultralytics default 0.1)"),
    "scale":      (0.0, "scale gain (+/-) (ultralytics default 0.5)"),
    "shear":      (0.0, "shear degrees (+/-)"),
    "perspective":(0.0, "perspective fraction (0-0.001 typical)"),
    "flipud":     (0.0, "vertical flip probability (radar depth axis: keep 0 unless testing otherwise)"),
    "fliplr":     (0.0, "horizontal flip probability (ultralytics default 0.5)"),
    "mosaic":     (0.0, "mosaic augmentation probability (ultralytics default 1.0)"),
    "mixup":      (0.0, "mixup augmentation probability"),
    "copy_paste": (0.0, "copy-paste augmentation probability"),
}


def read_classes(data_dir):
    classes_path = os.path.join(data_dir, "classes.txt")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"classes.txt not found in {data_dir}. Annotate at least one box with "
            f"biaozhu-yolo-bbox.py first."
        )
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def collect_labeled_images(data_dir):
    """Images that have a matching non-empty YOLO .txt label file."""
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not img_path.lower().endswith(SUPPORTED_EXTS):
            continue
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            pairs.append((img_path, label_path))
    return pairs


def build_yolo_dataset(data_dir, work_dir, val_ratio=0.2, seed=0, polarity_dir=None, polarity_aug=False):
    """Copy images/labels into images/{train,val} + labels/{train,val} and
    write data.yaml, the layout ultralytics expects.

    If polarity_aug is set, each *train*-split image that has a matching
    pixel-aligned "<base>_polarity.png" in polarity_dir gets that variant
    added as an extra training sample reusing the same box labels (the
    polarity attribute map is the same size/geometry as the source image,
    so the annotations carry over unchanged). Val is left untouched so
    train/val never contain two representations of the same underlying scan.
    """
    classes = read_classes(data_dir)
    pairs = collect_labeled_images(data_dir)
    if not pairs:
        raise RuntimeError(f"No labeled images found in {data_dir} (need image + non-empty .txt).")

    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    if not train_pairs:
        train_pairs = val_pairs  # tiny dataset: reuse the same pair(s) for train too

    if polarity_aug and not polarity_dir:
        raise ValueError("--polarity-aug requires --polarity-dir")

    dataset_dir = os.path.join(work_dir, "yolo_dataset")
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_polarity_added = 0
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, os.path.join(img_dir, os.path.basename(img_path)))
            shutil.copy2(label_path, os.path.join(lbl_dir, os.path.basename(label_path)))

            if split == "train" and polarity_aug:
                base = os.path.splitext(os.path.basename(img_path))[0]
                pol_img_path = os.path.join(polarity_dir, f"{base}_polarity.png")
                if os.path.exists(pol_img_path):
                    pol_name = os.path.basename(pol_img_path)
                    shutil.copy2(pol_img_path, os.path.join(img_dir, pol_name))
                    shutil.copy2(label_path, os.path.join(lbl_dir, os.path.splitext(pol_name)[0] + ".txt"))
                    n_polarity_added += 1
        if split == "train" and polarity_aug:
            print(f"[dataset] +{n_polarity_added} polarity-attribute train images "
                  f"(of {len(train_pairs)} base train images)")

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": dataset_dir,
            "train": "images/train",
            "val": "images/val",
            "names": {i: c for i, c in enumerate(classes)},
        }, f, allow_unicode=True, sort_keys=False)

    print(f"[dataset] train={len(train_pairs)} val={len(val_pairs)} classes={classes}")
    print(f"[dataset] -> {yaml_path}")
    return yaml_path


def train(args):
    from ultralytics import YOLO

    # Ultralytics resolves relative `project` paths against its own cwd
    # bookkeeping, not the caller's — a relative "../.." work-dir silently
    # nests under the wrong directory. Absolute paths avoid that entirely.
    args.data_dir = os.path.abspath(args.data_dir)
    args.work_dir = os.path.abspath(args.work_dir)
    if args.polarity_aug:
        args.polarity_dir = os.path.abspath(args.polarity_dir)

    yaml_path = build_yolo_dataset(
        args.data_dir, args.work_dir, args.val_ratio, args.seed,
        polarity_dir=args.polarity_dir, polarity_aug=args.polarity_aug,
    )
    model = YOLO(args.model)
    aug_kwargs = {name: getattr(args, name) for name in AUG_PARAMS}
    # model.train() runs a final validation pass on best.pt internally and
    # returns those metrics directly, so a separate model.val() call would
    # just duplicate that work (and scatter an extra runs/detect/val dir).
    metrics = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project=args.work_dir,
        name="train",
        exist_ok=True,
        **aug_kwargs,
    )
    print(f"[val] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    best = os.path.join(args.work_dir, "train", "weights", "best.pt")
    print(f"[train] best weights -> {best}")
    return best


def detect(args):
    from ultralytics import YOLO

    args.source = os.path.abspath(args.source)
    args.work_dir = os.path.abspath(args.work_dir)
    args.weights = os.path.abspath(args.weights)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=args.work_dir,
        name="detect",
        exist_ok=True,
    )
    total_boxes = sum(len(r.boxes) for r in results)
    print(f"[detect] {len(results)} images, {total_boxes} boxes detected")
    print(f"[detect] annotated images -> {os.path.join(args.work_dir, 'detect')}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train a YOLO detector on the annotated boxes.")
    pt.add_argument("--data-dir", default=os.path.join(_HERE, "jixing-zhengchang"),
                     help="Folder with images + matching .txt labels + classes.txt (annotated by biaozhu-yolo-bbox.py).")
    pt.add_argument("--work-dir", default=os.path.join(_HERE, "yolo_runs"),
                     help="Output folder for the copied dataset + training runs.")
    pt.add_argument("--model", default="yolov8n.pt", help="Ultralytics base model/checkpoint to start from.")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--imgsz", type=int, default=640)
    pt.add_argument("--batch", type=int, default=16)
    pt.add_argument("--val-ratio", type=float, default=0.2)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--polarity-dir", default=os.path.join(_HERE, "polarity"),
                     help="Folder with pixel-aligned <base>_polarity.png attribute maps (same size as the source images).")
    pt.add_argument("--polarity-aug", action="store_true",
                     help="Add each train image's polarity-attribute variant as an extra training sample sharing the same box labels.")
    aug_group = pt.add_argument_group("augmentation (all off by default; override for ablations)")
    for name, (default, help_text) in AUG_PARAMS.items():
        aug_group.add_argument(f"--{name.replace('_', '-')}", type=float, default=default, help=help_text)

    pd = sub.add_parser("detect", help="Run a trained YOLO detector on images.")
    pd.add_argument("--weights", required=True, help="Path to trained .pt weights (e.g. yolo_runs/train/weights/best.pt).")
    pd.add_argument("--source", default=os.path.join(_HERE, "jixing-zhengchang"),
                     help="Image file or folder to run detection on.")
    pd.add_argument("--work-dir", default=os.path.join(_HERE, "yolo_runs"), help="Output folder for detection results.")
    pd.add_argument("--conf", type=float, default=0.25)
    pd.add_argument("--imgsz", type=int, default=640)

    return p


def main():
    args = build_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "detect":
        detect(args)


if __name__ == "__main__":
    main()
