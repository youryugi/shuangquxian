"""Seed HBAT annotation files from the old self-made annotations, as *initial values* only.

Every seeded object gets  "source": "legacy_init"  and an auto-assigned gt_box_id (None when the
band overlaps no official box, or when two boxes overlap about equally). Updating an object in HBAT
rewrites it from the controls, which drops the "source" field: objects that still carry
"source": "legacy_init" have not been reviewed yet.

    python tools/annotation/seed_from_legacy.py <repo root>

Dataset 2: raw_131/annotations.json (old width/height schema) -> data/additional_utilities/hbat/annotations.json
Dataset 1: selected_318_hbat/annotations.json -> data/primary_roboflow/hbat/annotations.json (train+valid+test in
           one file), mapped onto the official export by group name (<id>_jpg): the three Roboflow copies of one
           image share geometry, so the old objects apply to all of them.
"""
import os, sys, json, collections
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbat  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "."


def seed(objs_by_image, img_dir, gt_dir, out_path, out=None):
    """Convert and append into `out` (a dict); write out_path when out is None."""
    gt = hbat.GTSource(gt_dir)
    write = out is None
    out = {} if out is None else out
    stats = collections.Counter()
    for name, objs in sorted(objs_by_image.items()):
        p = os.path.join(img_dir, name)
        if not os.path.exists(p):
            stats["image_missing"] += 1
            continue
        with Image.open(p) as im:
            size = im.size
        boxes = gt.boxes_for(name, size)
        seeded = []
        for o in objs:
            o = hbat.normalize_obj(o)
            for k in ("width", "height", "a", "b"):
                o.pop(k, None)
            best, ambiguous = hbat.assign_gt_box(o, boxes, size)
            o["gt_box_id"] = None if ambiguous else best
            o["source"] = "legacy_init"
            stats["objects"] += 1
            stats["gt_none" if o["gt_box_id"] is None else "gt_assigned"] += 1
            stats["gt_ambiguous"] += ambiguous
            seeded.append(o)
        out[name] = seeded
        stats["images"] += 1
    print(f"  {dict(stats)}")
    if not write:
        return out
    if os.path.exists(out_path):
        print(f"  已存在，不覆盖: {out_path}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  写入 {out_path}")
    return out


def write_json(d, out_path):
    if os.path.exists(out_path):
        print(f"  已存在，不覆盖: {out_path}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"  写入 {out_path}: {len(d)} 张图")


# ---- dataset 2 -----------------------------------------------------------------
print("数据集 2 (Mendeley)")
raw = json.load(open(os.path.join(R, "data/additional_utilities/legacy/raw_131/annotations.json"), encoding="utf-8"))
seed(raw,
     os.path.join(R, "data/additional_utilities/original/original_Utilities"),
     os.path.join(R, "data/additional_utilities/gt_boxes_official_aug6"),
     os.path.join(R, "data/additional_utilities/hbat/annotations.json"))

# ---- dataset 1 -----------------------------------------------------------------
print("数据集 1 (Roboflow)")
sel = json.load(open(os.path.join(R, "data/primary_roboflow/selected_318_hbat/annotations.json"), encoding="utf-8"))
by_group = {}
for name, objs in sel.items():
    by_group.setdefault(hbat.HyperbolaAnnotator.get_group_name(name), objs)
export = os.path.join(R, "data/primary_roboflow/original_GPR.v1i.yolov5-obb")
merged = {}   # one file for all three splits: file names are unique, HBAT keeps the other splits' entries when saving
for split in ("train", "valid", "test"):
    img_dir = os.path.join(export, split, "images")
    mapped = {}
    for f in sorted(os.listdir(img_dir)):
        g = hbat.HyperbolaAnnotator.get_group_name(f)
        if g in by_group:
            mapped[f] = by_group[g]
    print(f"  {split}: 官方图 {len(os.listdir(img_dir))} 张，其中 {len(mapped)} 张有旧标注可作初值")
    seed(mapped, img_dir, os.path.join(export, split, "labelTxt"), None, out=merged)
write_json(merged, os.path.join(R, "data/primary_roboflow/hbat/annotations.json"))
