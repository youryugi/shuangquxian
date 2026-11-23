"""Build a YOLO-format dataset directory (images / labels / masks) from a grouped split.

    python prepare_data.py --dataset ds2 --fold 0 --subset 100 --missing-mask skip

Output  (default: experiments/revision/shared/datasets/<ds>_f<fold>_s<subset>/)
    images/{train,val,test}/<stem>.jpg      copied images (train: every copy; val/test: eval_file only)
    labels/{train,val,test}/<stem>.txt      official GT boxes, YOLO format
    masks_band/{train,val,test}/<stem>.png  HBAT hyperbola-band mask, union over targets (0/255)
    masks_box/{...}                         control mask: official boxes filled solid
    data.yaml                               ultralytics data file (train/val/test)
    gt_boxes.json                           {phase: {stem: {file, item, boxes[[x1,y1,x2,y2]], category, stratum}}}
    manifest.json                           counts, parameters, items without band annotation

The SAME directory is used by every detector and by both arms of the plug-in experiment,
so the training images are identical across arms by construction.

--missing-mask decides what to do with TRAIN images that have official boxes but no HBAT band yet:
    error  stop and list them (default; use once annotation is complete)
    skip   drop these items from train for every arm (smoke tests while annotation is incomplete)
    zero   all-black mask (band loss treats the image as pure background: wrong supervision, avoid)
    box    fall back to the filled official box (contaminates the band-vs-box ablation, avoid)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from common import (DATASETS, REPO, HERE, boxes_to_yolo_lines, hbat_objects, image_size, item_files,
                    load_json, load_split, official_boxes, render_band_mask, render_box_mask, resolve_split,
                    save_json, stem_of)
import hbat  # noqa: E402  (path inserted by common)


def hbat_boxes(band_objs, boxes_off, img_size):
    """One box per HBAT band (hbat.hyperbola_to_bbox, clipped to the image).

    Objects without band parameters (param_status 'unparam') fall back to their official box
    (by gt_box_id) when no parametrised object in the same image claims that official box.
    Returns (boxes, n_fallback)."""
    out, n_fb = [], 0
    claimed = {o.get("gt_box_id") for o in band_objs if hbat.has_band(hbat.normalize_obj(o))}
    for o in band_objs:
        b = hbat.hyperbola_to_bbox(o, img_size)
        if b is not None:
            out.append([b["x1"], b["y1"], round(b["x1"] + b["width"], 2), round(b["y1"] + b["height"], 2)])
            continue
        gid = o.get("gt_box_id")
        if gid is not None and gid not in claimed and isinstance(gid, int) and 0 <= gid < len(boxes_off):
            out.append(list(boxes_off[gid])); n_fb += 1
    return out, n_fb


def write_yaml(out, names=("hyperbola",)):
    txt = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        f"names: {list(names)}\n"
        "channels: 3\n"
    )
    (out / "data.yaml").write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--subset", default="100", help="25 / 50 / 75 / 100 (nested train subsets) or 'pool'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hbat-json", default=None, help="HBAT annotations.json (default: per-dataset)")
    ap.add_argument("--missing-mask", default="error", choices=["error", "skip", "zero", "box"])
    ap.add_argument("--no-copy", action="store_true", help="symlink instead of copying images")
    ap.add_argument("--box-source", default="official", choices=["official", "hbat"],
                    help="detection targets: official boxes (default) or one box per HBAT band (hbat). "
                         "Both box sets are always written (gt_boxes_official.json / gt_boxes_hbat.json) so the same "
                         "predictions can be scored against either.")
    args = ap.parse_args()

    ds = args.dataset
    cfg = DATASETS[ds]
    split = load_split(ds)
    phases = resolve_split(split, args.fold, args.subset)
    hbat_ann = load_json(args.hbat_json or cfg["hbat_json"])
    suffix = "_hbatbox" if args.box_source == "hbat" else ""
    out = Path(args.out) if args.out else HERE / "datasets" / f"{ds}_f{args.fold}_s{args.subset}{suffix}"
    n_fallback = 0
    if out.exists():
        shutil.rmtree(out)
    for ph in ("train", "val", "test"):
        for sub in ("images", "labels", "masks_band", "masks_box"):
            (out / sub / ph).mkdir(parents=True, exist_ok=True)

    items = split["items"]
    gt = {"train": {}, "val": {}, "test": {}}
    missing_band = {"train": [], "val": [], "test": []}
    n_img = {"train": 0, "val": 0, "test": 0}
    n_box = {"train": 0, "val": 0, "test": 0}
    n_band_obj = 0
    skipped_items = []

    for ph in ("train", "val", "test"):
        for item_id in phases[ph]:
            item = items[item_id]
            files = item_files(item, ph)
            # decide once per item whether a band annotation exists (geometry is shared by all copies)
            objs, key = hbat_objects(hbat_ann, item_id, item, files[0])
            band_objs = [o for o in (objs or []) if o.get("x_vertex") is not None]
            has_band = bool(band_objs)
            if not has_band:
                missing_band[ph].append(item_id)
                if ph == "train" and args.missing_mask == "skip":
                    skipped_items.append(item_id)
                    continue
            for f in files:
                src = REPO / f
                size = image_size(src)
                boxes_off = official_boxes(ds, item_id, item, f, size)
                boxes_hb, n_fb = hbat_boxes(band_objs, boxes_off, size) if has_band else ([], 0)
                n_fallback += n_fb
                boxes = boxes_hb if args.box_source == "hbat" else boxes_off
                stem = stem_of(f)
                dst = out / "images" / ph / (stem + src.suffix.lower())
                if args.no_copy:
                    try:
                        dst.symlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
                (out / "labels" / ph / f"{stem}.txt").write_text("\n".join(boxes_to_yolo_lines(boxes, size)) + ("\n" if boxes else ""), encoding="utf-8")
                # masks
                if has_band:
                    band = render_band_mask(band_objs, size)
                elif ph == "train" and args.missing_mask == "box":
                    band = render_box_mask(boxes, size)
                else:
                    band = np.zeros((size[1], size[0]), dtype=np.uint8)
                Image.fromarray(band).save(out / "masks_band" / ph / f"{stem}.png")
                Image.fromarray(render_box_mask(boxes, size)).save(out / "masks_box" / ph / f"{stem}.png")
                gt[ph][stem] = {
                    "file": f, "item": item_id, "group": item.get("group"),
                    "category": item.get("category"), "stratum": item.get("stratum"),
                    "width": size[0], "height": size[1], "boxes": boxes,
                    "boxes_official": boxes_off, "boxes_hbat": boxes_hb,
                    "band_key": key if has_band else None, "n_band": len(band_objs) if has_band else 0,
                }
                n_img[ph] += 1
                n_box[ph] += len(boxes)
                if has_band:
                    n_band_obj += len(band_objs)

    if args.missing_mask == "error" and missing_band["train"]:
        raise SystemExit(
            f"[prepare_data] {len(missing_band['train'])} train items have official boxes but no HBAT band: "
            f"{missing_band['train']}\n  -> finish annotation, or re-run with --missing-mask skip (smoke test only)."
        )

    write_yaml(out)
    save_json(gt, out / "gt_boxes.json")
    # the alternative box set, same structure, for scoring the same predictions against the other GT
    for alt in ("official", "hbat"):
        gt_alt = {ph: {s: {**r, "boxes": r[f"boxes_{alt}"]} for s, r in d.items()} for ph, d in gt.items()}
        save_json(gt_alt, out / f"gt_boxes_{alt}.json")
    n_box_alt = {alt: {ph: sum(len(r[f"boxes_{alt}"]) for r in gt[ph].values()) for ph in gt} for alt in ("official", "hbat")}
    manifest = {
        "dataset": ds, "fold": args.fold, "subset": args.subset,
        "split_file": str(cfg["split"].relative_to(REPO)), "hbat_json": str(Path(args.hbat_json or cfg["hbat_json"])),
        "box_source": args.box_source if args.box_source == "hbat" else cfg["box_source"],
        "official_box_source": cfg["box_source"], "missing_mask": args.missing_mask,
        "n_items": {ph: len(phases[ph]) for ph in phases}, "n_images": n_img, "n_boxes": n_box,
        "n_boxes_by_source": n_box_alt, "n_hbat_box_fallback_to_official": n_fallback,
        "n_band_objects": n_band_obj,
        "items_without_band": missing_band, "train_items_skipped": skipped_items,
    }
    save_json(manifest, out / "manifest.json")

    print(f"[prepare_data] -> {out}")
    for ph in ("train", "val", "test"):
        print(f"  {ph:5s}: items {len(phases[ph]):3d}  images {n_img[ph]:3d}  boxes {n_box[ph]:3d}  "
              f"items without band {len(missing_band[ph])}")
    if skipped_items:
        print(f"  train items dropped (no band, --missing-mask skip): {len(skipped_items)}")
    print(f"  band objects rendered: {n_band_obj}")
    print(f"  boxes by source: {n_box_alt}  (targets = {manifest['box_source']}; "
          f"unparam objects falling back to official box: {n_fallback})")


if __name__ == "__main__":
    main()
