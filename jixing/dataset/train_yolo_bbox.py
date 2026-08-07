"""
Train / run a real YOLO (ultralytics) detector on the boxes produced by
biaozhu-yolo-bbox.py: a folder of images, each with a same-name .txt label
file (class_id xc yc w h, normalized 0..1) and a shared classes.txt.

Run inside the `gpr` conda env (already has ultralytics + torch + CUDA):

    conda activate gpr
    python train_yolo_bbox.py train  --data-dir jixing-merged --epochs 100
    python train_yolo_bbox.py detect --weights yolo_runs/train/weights/best.pt --source jixing-merged
"""
import os
import sys
import glob
import random
import shutil
import argparse
from pathlib import Path

import yaml

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── 参数配置（直接改这里；点击 Run 不带命令行参数时生效。命令行参数可覆盖同名值）──
# --- train ---
DATA_DIR     = os.path.join(_HERE, "jixing-merged")  # 图片 + 同名 .txt 标签 + classes.txt（build_merged_dataset.py 合并 vis+vis-all 产出，单类 hyperbola）
WORK_DIR     = os.path.join(_HERE, "yolo_runs")           # 数据集拷贝 + 训练产出目录
MODEL        = "yolov8n.pt"                                # ultralytics 基础模型/权重
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 16
VAL_RATIO    = 0.2
SEED         = 0

POLARITY_DIR = os.path.join(_HERE, "polarity")             # 极性属性图（Hilbert 变换派生通道，<base>_polarity.png）
POLARITY_AUG = False

INVERT_DIR   = os.path.join(_HERE, "..", "..", "dataset2", "vis_inverted_all")  # 真极性反转渲染图（data -> -data）
INVERT_AUG   = True

# --- detect ---
DETECT_WEIGHTS = os.path.join(WORK_DIR, "train", "weights", "best.pt")
DETECT_SOURCE  = os.path.join(_HERE, "jixing-merged")
DETECT_CONF    = 0.25
# ─────────────────────────────────────────────────────────

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


def build_invert_lookup(invert_dir):
    """dataset2/polarity_features.py flattens the raw dataset tree with
    flat_output_name(): "_".join(relative_parts) + ".png", relative to the
    *whole* dataset root. That leaves one extra leading "<top_folder>_"
    segment versus the flat names used here (produced from a single
    top-folder subtree), so exact-basename lookup fails. Stripping the
    first "_"-segment off each inverted filename recovers a key that
    matches this project's basenames directly.
    """
    lookup = {}
    for path in Path(invert_dir).glob("*.png"):
        _, _, rest = path.name.partition("_")
        if rest:
            lookup[rest] = str(path)
    return lookup


def build_yolo_dataset(data_dir, work_dir, val_ratio=0.2, seed=0, polarity_dir=None, polarity_aug=False,
                        invert_dir=None, invert_aug=False):
    """Copy images/labels into images/{train,val} + labels/{train,val} and
    write data.yaml, the layout ultralytics expects.

    If polarity_aug is set, each *train*-split image that has a matching
    pixel-aligned "<base>_polarity.png" in polarity_dir gets that variant
    added as an extra training sample reusing the same box labels (the
    polarity attribute map is the same size/geometry as the source image,
    so the annotations carry over unchanged). Val is left untouched so
    train/val never contain two representations of the same underlying scan.

    If invert_aug is set, each *train*-split image gets its true
    sign-inverted render (data -> -data, from dataset2/polarity_features.py,
    looked up via build_invert_lookup) added as an extra training sample
    with the same box labels — the hyperbola geometry is unchanged, only
    the amplitude sign flips. This is a different augmentation from
    polarity_aug: that one adds a derived attribute-map channel, this one
    adds a genuinely re-rendered, sign-flipped radargram.
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
    if invert_aug and not invert_dir:
        raise ValueError("--invert-aug requires --invert-dir")
    invert_lookup = build_invert_lookup(invert_dir) if invert_aug else {}

    dataset_dir = os.path.join(work_dir, "yolo_dataset")
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_polarity_added = 0
        n_invert_added = 0
        for img_path, label_path in split_pairs:
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(img_dir, img_name))
            shutil.copy2(label_path, os.path.join(lbl_dir, os.path.basename(label_path)))

            if split == "train" and polarity_aug:
                base = os.path.splitext(img_name)[0]
                pol_img_path = os.path.join(polarity_dir, f"{base}_polarity.png")
                if os.path.exists(pol_img_path):
                    pol_name = os.path.basename(pol_img_path)
                    shutil.copy2(pol_img_path, os.path.join(img_dir, pol_name))
                    shutil.copy2(label_path, os.path.join(lbl_dir, os.path.splitext(pol_name)[0] + ".txt"))
                    n_polarity_added += 1

            if split == "train" and invert_aug:
                inv_img_path = invert_lookup.get(img_name)
                if inv_img_path:
                    base, ext = os.path.splitext(img_name)
                    inv_name = f"{base}_inverted{ext}"
                    shutil.copy2(inv_img_path, os.path.join(img_dir, inv_name))
                    shutil.copy2(label_path, os.path.join(lbl_dir, os.path.splitext(inv_name)[0] + ".txt"))
                    n_invert_added += 1
        if split == "train" and polarity_aug:
            print(f"[dataset] +{n_polarity_added} polarity-attribute train images "
                  f"(of {len(train_pairs)} base train images)")
        if split == "train" and invert_aug:
            print(f"[dataset] +{n_invert_added} true-polarity-inverted train images "
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
    if args.invert_aug:
        args.invert_dir = os.path.abspath(args.invert_dir)

    yaml_path = build_yolo_dataset(
        args.data_dir, args.work_dir, args.val_ratio, args.seed,
        polarity_dir=args.polarity_dir, polarity_aug=args.polarity_aug,
        invert_dir=args.invert_dir, invert_aug=args.invert_aug,
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
    pt.add_argument("--data-dir", default=DATA_DIR,
                     help="Folder with images + matching .txt labels + classes.txt (annotated by biaozhu-yolo-bbox.py).")
    pt.add_argument("--work-dir", default=WORK_DIR,
                     help="Output folder for the copied dataset + training runs.")
    pt.add_argument("--model", default=MODEL, help="Ultralytics base model/checkpoint to start from.")
    pt.add_argument("--epochs", type=int, default=EPOCHS)
    pt.add_argument("--imgsz", type=int, default=IMGSZ)
    pt.add_argument("--batch", type=int, default=BATCH)
    pt.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    pt.add_argument("--seed", type=int, default=SEED)
    pt.add_argument("--polarity-dir", default=POLARITY_DIR,
                     help="Folder with pixel-aligned <base>_polarity.png attribute maps (same size as the source images).")
    pt.add_argument("--polarity-aug", dest="polarity_aug", action="store_true", default=POLARITY_AUG,
                     help="Add each train image's polarity-attribute variant as an extra training sample sharing the same box labels.")
    pt.add_argument("--no-polarity-aug", dest="polarity_aug", action="store_false",
                     help="Disable the polarity-attribute augmentation.")
    pt.add_argument("--invert-dir", default=INVERT_DIR,
                     help="Folder with true sign-inverted (data -> -data) re-rendered radargrams, "
                          "produced by dataset2/polarity_features.py.")
    pt.add_argument("--invert-aug", dest="invert_aug", action="store_true", default=INVERT_AUG,
                     help="Add each train image's true sign-inverted render as an extra training sample "
                          "sharing the same box labels (distinct from --polarity-aug's attribute map).")
    pt.add_argument("--no-invert-aug", dest="invert_aug", action="store_false",
                     help="Disable the true sign-inverted-render augmentation.")
    aug_group = pt.add_argument_group("augmentation (all off by default; override for ablations)")
    for name, (default, help_text) in AUG_PARAMS.items():
        aug_group.add_argument(f"--{name.replace('_', '-')}", type=float, default=default, help=help_text)

    pd = sub.add_parser("detect", help="Run a trained YOLO detector on images.")
    pd.add_argument("--weights", default=DETECT_WEIGHTS, help="Path to trained .pt weights (e.g. yolo_runs/train/weights/best.pt).")
    pd.add_argument("--source", default=DETECT_SOURCE,
                     help="Image file or folder to run detection on.")
    pd.add_argument("--work-dir", default=WORK_DIR, help="Output folder for detection results.")
    pd.add_argument("--conf", type=float, default=DETECT_CONF)
    pd.add_argument("--imgsz", type=int, default=IMGSZ)

    return p


def main():
    # Clicking "Run" in an IDE invokes this with no argv at all, which would
    # otherwise hit argparse's required subcommand error. Default to `train`
    # (with all its own defaults, e.g. --invert-aug on) so that just works.
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("train", "detect", "-h", "--help"):
        argv = ["train"] + argv
    args = build_argparser().parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "detect":
        detect(args)


if __name__ == "__main__":
    main()
