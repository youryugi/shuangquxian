"""Shared helpers for the detector-comparison / band-plugin experiments.

Everything that touches the repository's data layout lives here, so the
training scripts only see a prepared YOLO-format dataset directory.

Data sources (see data/README.md and paper/revision/revision_map.md):
  * splits        data/splits/dataset{1,2}_*.json   (grouped folds, nested train subsets)
  * official GT   ds1: Roboflow YOLOv5-OBB labelTxt (axis-aligned rectangles as 4 points)
                  ds2: data/additional_utilities/gt_boxes_official_aug6/<id>.txt (YOLO, normalized)
  * band masks    rasterized with tools/annotation/hbat.py (the same code the annotator sees)
  * ds1 crop      data/primary_roboflow/hbat/crop.json: the screenshot axis margins are cropped away and the
                  plot area resized to 640x640 (tools/annotation/d1_crop.py). HBAT annotations live in that
                  working frame, so every copy of an item is rendered in the same frame here (image_frame /
                  load_image) and the official boxes are mapped into it (official_boxes).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]   # experiments/revision/shared -> repo
TOOLS_ANNOTATION = REPO / "tools" / "annotation"
if str(TOOLS_ANNOTATION) not in sys.path:
    sys.path.insert(0, str(TOOLS_ANNOTATION))

import hbat  # noqa: E402  (tools/annotation/hbat.py; imports tkinter but creates no window)
import d1_crop  # noqa: E402  (tools/annotation/d1_crop.py; ds1 crop frame)

def _first_existing(*rel):
    """The data folder is being reorganised (2026-11); accept the new location first, old ones as fallback."""
    for r in rel:
        p = REPO / r
        if p.exists():
            return p
    return REPO / rel[0]


DATASETS = {
    "ds1": {
        "name": "dataset1_roboflow",
        "split": REPO / "data/splits/dataset1_roboflow.json",
        # canonical HBAT file keyed by the official v1 export filenames (seeded by tools/annotation/seed_from_legacy.py)
        "hbat_json": _first_existing("data/primary_roboflow/hbat/annotations.json",
                                     "data/primary_roboflow/selected_318_hbat/annotations.json"),
        "box_source": "roboflow_obb",
        "crop_json": REPO / "data/primary_roboflow/hbat/crop.json",
    },
    "ds2": {
        "name": "dataset2_mendeley",
        "split": REPO / "data/splits/dataset2_mendeley.json",
        # canonical HBAT file keyed by the original filenames (001.jpg ...)
        "hbat_json": _first_existing("data/additional_utilities/hbat/annotations.json",
                                     "data/additional_utilities/legacy/augmented_247_hbat/annotations.json",
                                     "data/additional_utilities/augmented_247_hbat/annotations.json"),
        "box_source": "official_aug6",
        "box_dir": _first_existing("data/additional_utilities/original/gt_boxes_official_aug6",
                                   "data/additional_utilities/gt_boxes_official_aug6"),
    },
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------- io
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=1):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def image_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


# ----------------------------------------------------------------------------- working frame (ds1 crop)
_CROPS = {}


def image_frame(ds, item_id):
    """(M, out_size) mapping official-image pixels to the annotation working frame, or None (identity).

    Uses the crop *applied* in the HBAT folder (crop.json 'applied'), i.e. the frame the HBAT annotations of
    this item are stored in; items whose crop is not yet confirmed stay in the official frame."""
    path = DATASETS[ds].get("crop_json")
    if path is None or not Path(path).exists():
        return None
    if ds not in _CROPS:
        _CROPS[ds] = load_json(path)
    c = _CROPS[ds]["crops"].get(item_id)
    if not c or not c.get("applied"):
        return None
    return d1_crop.frame_matrix(c["applied"]), tuple(_CROPS[ds]["output_size"])


def load_image(ds, item_id, file):
    """Image of one copy in the working frame -> (BGR array or None if unchanged, size (w, h))."""
    fr = image_frame(ds, item_id)
    if fr is None:
        return None, image_size(REPO / file)
    c = _CROPS[ds]["crops"][item_id]
    return d1_crop.render(d1_crop.imread(str(REPO / file)), c["applied"], fr[1]), fr[1]


# ----------------------------------------------------------------------------- splits
def load_split(ds):
    return load_json(DATASETS[ds]["split"])


def resolve_split(split, fold=0, subset="100"):
    """Return {'train': [item_id...], 'val': [...], 'test': [...]} for a fold / nested subset."""
    f = split["folds"][fold]
    groups = split["groups"]

    def items_of(group_ids):
        out = []
        for g in group_ids:
            out.extend(groups[g])
        return sorted(out)

    train_groups = f["train_subsets"][str(subset)] if str(subset) != "pool" else f["train_pool"]
    return {"train": items_of(train_groups), "val": items_of(f["val"]), "test": items_of(f["test"])}


def item_files(item, phase):
    """Training uses every copy of an item; val/test evaluate the single designated file."""
    if phase == "train":
        return list(item["files"])
    return [item["eval_file"]]


# ----------------------------------------------------------------------------- official boxes
def _parse_obb_txt(path):
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            xs = [float(v) for v in parts[0:8:2]]
            ys = [float(v) for v in parts[1:8:2]]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes


def _parse_yolo_txt(path, img_w, img_h):
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h])
    return boxes


def official_boxes(ds, item_id, item, file, img_size):
    """Axis-aligned official GT boxes in pixels (x1, y1, x2, y2), clipped to the image.

    img_size is the size in the working frame; for cropped ds1 items the boxes are mapped into it."""
    w, h = img_size
    cfg = DATASETS[ds]
    if cfg["box_source"] == "roboflow_obb":
        idx = item["files"].index(file) if file in item["files"] else 0
        boxes = _parse_obb_txt(REPO / item["labels"][idx])
        fr = image_frame(ds, item_id)
        if fr is not None:
            M = fr[0]
            boxes = [[M[0, 0] * x1 + M[0, 2], M[1, 1] * y1 + M[1, 2], M[0, 0] * x2 + M[0, 2], M[1, 1] * y2 + M[1, 2]]
                     for x1, y1, x2, y2 in boxes]
    elif cfg["box_source"] == "official_aug6":
        boxes = _parse_yolo_txt(cfg["box_dir"] / f"{item_id}.txt", w, h)
    else:
        raise ValueError(cfg["box_source"])
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            out.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
    return out


def boxes_to_yolo_lines(boxes, img_size, cls=0):
    w, h = img_size
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ----------------------------------------------------------------------------- HBAT band annotations
def _candidate_keys(item_id, item, file):
    """Annotation-file keys to try, most trustworthy first.

    ds1: the three photometric copies share identical geometry. The designated eval_file copy is the one
         annotated / reviewed in HBAT (tools/annotation/d1_todo.py), so it is used for every copy; the other
         copies' keys only hold stale legacy_init seeds.
    ds2: HBAT annotations were made on augmented images; aug_6 (and aug_1) keep the geometry of the original.
    """
    keys = [os.path.basename(item["eval_file"]), os.path.basename(file)]
    legacy = list(item.get("legacy_hbat_files", []))
    legacy.sort(key=lambda p: (0 if p.endswith("_aug_6.jpg") else 1 if p.endswith("_aug_1.jpg") else 2, p))
    keys += [os.path.basename(p) for p in legacy]
    keys += [f"{item_id}_aug_6.jpg", f"{item_id}_aug_1.jpg", f"{item_id}.jpg", f"{item_id}.png"]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def hbat_objects(hbat_ann, item_id, item, file):
    """Return (objects, key) for the first matching annotation key, or (None, None)."""
    for k in _candidate_keys(item_id, item, file):
        if k in hbat_ann:
            return hbat_ann[k], k
    return None, None


def render_band_mask(objects, img_size, **kw):
    """Union of HBAT band masks for one image -> uint8 array (0/255). Uses hbat.band_mask verbatim."""
    w, h = img_size
    union = np.zeros((h, w), dtype=np.uint8)
    for obj in objects or []:
        m = np.array(hbat.band_mask(obj, (w, h), **kw), dtype=np.uint8)
        union = np.maximum(union, m)
    return union


def render_box_mask(boxes, img_size):
    """Control mask: official boxes filled solid (shape prior degenerates to a rectangle)."""
    w, h = img_size
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[int(math.floor(y1)):int(math.ceil(y2)), int(math.floor(x1)):int(math.ceil(x2))] = 255
    return m


# ----------------------------------------------------------------------------- misc
def stem_of(path):
    return Path(path).stem


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
