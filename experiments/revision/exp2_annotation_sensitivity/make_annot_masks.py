"""第 21 轮：标注敏感性（回应 R1-4、R2-4）。把 HBAT 标注参数按档位扰动，导出新的 band 监督掩码。

    python make_annot_masks.py            # 全部档位、5 折
    python make_annot_masks.py --levels t050 rep1 --check-only

只改 attention 的监督掩码：检测框（gt_boxes.json，HBAT 派生框）、测试集掩码（真实 band 上限、band IoU 用）都不动。
训练集和验证集的掩码换成扰动后的版本（同一位标注员标了这两部分），测试集掩码原样复制，保证 PreparedDataset 能找到文件。
扰动按目标定义在整个数据集上（与折无关）：同一个目标在任何一折里都是同一条扰动后的 band，等于"换了一份标注"。

档位
  第 1 档：系统偏差（R2-4 点名的厚度、跨度）
    t025 … t200   每个目标的厚度 × 0.25 / 0.5 / 0.75 / 1.25 / 1.5 / 2（中心线不变，下限 2 px）
    tuni          统一厚度：全部目标取数据集厚度中位数（"厚度要不要逐个标"）
    sp075 / sp125 跨度 × 0.75 / 1.25
  第 2 档：随机标注误差（每个目标独立抽一次，固定随机种子；同一类的两档共用同一组随机数，只差倍数）
    ap2 / ap5     顶点沿随机方向移 2 / 5 px（同一标注员重复标注的顶点误差中位数 2.6 px、90% 分位 5.4 px）
    sl10 / sl20   slope × (1 ± 10% / 20%)，符号随机（重复标注 log slope 差的 std 0.09）
    rep1 / rep2   重复标注差值重抽样：从 68 对"同一目标标两次"（legacy augmented_247_hbat 的 aug_1 vs aug_6）的
                  (Δx, Δy, Δlog slope, Δlog thickness, Δlog span) 中整组抽一对、随机取正负号，× 1 / × 2 加到目标上

输出
  <dataset>/masks_ann_<档位>/{train,val,test}/<stem>.png
  results/levels.json        每个档位的定义、逐目标扰动后的参数
  results/mask_stats.md      每个档位相对原始掩码的像素 IoU、面积比（全部 131 张图，逐图并集）
  results/overlay_<档位>.png 前 8 张图：白 = 原始 band，红 = 扰动后 band（目检用）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATASETS = HERE.parent / "shared" / "datasets"
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import hbat  # noqa: E402

HBAT_JSON = REPO / "data/additional_utilities/hbat/annotations.json"
REPEAT_JSON = REPO / "data/additional_utilities/legacy/augmented_247_hbat/annotations.json"
OUT = HERE / "results"
FOLDS = range(5)
MIN_THICK = 2.0
NOISE_SEED = 21

LEVELS = {  # 名称: (类型, 参数)
    "t025": ("thick_scale", 0.25), "t050": ("thick_scale", 0.5), "t075": ("thick_scale", 0.75),
    "t125": ("thick_scale", 1.25), "t150": ("thick_scale", 1.5), "t200": ("thick_scale", 2.0),
    "tuni": ("thick_uniform", None),
    "sp075": ("span_scale", 0.75), "sp125": ("span_scale", 1.25),
    "ap2": ("apex", 2.0), "ap5": ("apex", 5.0),
    "sl10": ("slope", 0.10), "sl20": ("slope", 0.20),
    "rep1": ("repeat", 1.0), "rep2": ("repeat", 2.0),
}
TIER1 = ("t025", "t050", "t075", "t125", "t150", "t200", "tuni", "sp075", "sp125")
TIER2 = ("ap2", "ap5", "sl10", "sl20", "rep1", "rep2")


def obj_rng(key, j, kind):
    """每个 (图, 目标序号, 扰动类型) 一个独立、可复现的随机数发生器；同一类型的两档共用。"""
    h = hashlib.sha256(f"{NOISE_SEED}|{key}|{j}|{kind}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def repeat_pairs():
    """同一标注员对同一目标标两次的参数差（aug_1 vs aug_6，几何不变的两个增强版本），顶点距离 > 20 px 的视为不同目标。"""
    d = json.load(open(REPEAT_JSON, encoding="utf-8"))
    by = {}
    for key, objs in d.items():
        m = re.match(r"(.+)_aug_(\d)", key)
        if m:
            by.setdefault(m.group(1), {})[int(m.group(2))] = objs
    diffs = []
    for img in sorted(by):
        v = by[img]
        if 1 not in v or 6 not in v:
            continue
        B = [q for q in v[6] if q.get("x_vertex") is not None]
        for o in v[1]:
            if o.get("x_vertex") is None or not B:
                continue
            q = min(B, key=lambda q: (q["x_vertex"] - o["x_vertex"]) ** 2 + (q["y_vertex"] - o["y_vertex"]) ** 2)
            if math.hypot(q["x_vertex"] - o["x_vertex"], q["y_vertex"] - o["y_vertex"]) > 20:
                continue
            diffs.append((q["x_vertex"] - o["x_vertex"], q["y_vertex"] - o["y_vertex"], math.log(q["slope"] / o["slope"]),
                          math.log(q["thickness"] / o["thickness"]), math.log(q["span"] / o["span"])))
    return np.array(diffs, float)


def perturb(obj, key, j, level, ctx):
    o = hbat.normalize_obj(obj)
    if not hbat.has_band(o):
        return o
    kind, p = LEVELS[level]
    if kind == "thick_scale":
        o["thickness"] = max(MIN_THICK, o["thickness"] * p)
    elif kind == "thick_uniform":
        o["thickness"] = ctx["median_thickness"]
    elif kind == "span_scale":
        o["span"] = o["span"] * p
    elif kind == "apex":
        th = obj_rng(key, j, "apex").uniform(0, 2 * math.pi)
        o["x_vertex"] += p * math.cos(th); o["y_vertex"] += p * math.sin(th)
    elif kind == "slope":
        sgn = 1.0 if obj_rng(key, j, "slope").random() < 0.5 else -1.0
        o["slope"] = o["slope"] * (1.0 + sgn * p)
    elif kind == "repeat":
        r = obj_rng(key, j, "repeat")
        d = ctx["repeat"][r.integers(len(ctx["repeat"]))] * (1.0 if r.random() < 0.5 else -1.0) * p
        o["x_vertex"] += d[0]; o["y_vertex"] += d[1]
        o["slope"] *= math.exp(d[2]); o["thickness"] = max(MIN_THICK, o["thickness"] * math.exp(d[3])); o["span"] *= math.exp(d[4])
    return o


def render(objs, size):
    w, h = size
    u = np.zeros((h, w), np.uint8)
    for o in objs:
        if hbat.has_band(hbat.normalize_obj(o)):
            u = np.maximum(u, np.array(hbat.band_mask(o, (w, h)), np.uint8))   # 厚度下限已在 perturb 里施加
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--check-only", action="store_true", help="只算统计和叠加图，不写掩码目录")
    args = ap.parse_args()
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    # 所有图（131 张）：stem → (band_key, size)，从 fold 0 的三个 phase 里收集
    items = {}
    for ph in ("train", "val", "test"):
        for stem, g in json.load(open(DATASETS / "ds2_f0_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))[ph].items():
            items[stem] = (g["band_key"], (g["width"], g["height"]))
    banded = [hbat.normalize_obj(o) for k, _ in items.values() for o in ann.get(k, []) if hbat.has_band(hbat.normalize_obj(o))]
    ctx = {"median_thickness": float(np.median([o["thickness"] for o in banded])), "repeat": repeat_pairs()}
    print(f"{len(items)} 张图、{len(banded)} 个有 band 的目标；厚度中位数 {ctx['median_thickness']:.2f} px；"
          f"重复标注差值 {len(ctx['repeat'])} 对", flush=True)

    # 自检：不扰动时（厚度下限 1）逐像素复现数据集里的 masks_band
    for f in FOLDS:
        d = DATASETS / f"ds2_f{f}_s100_hbatbox"
        for ph in ("train", "val", "test"):
            for stem in json.load(open(d / "gt_boxes.json", encoding="utf-8"))[ph]:
                k, size = items[stem]
                ref = np.array(Image.open(d / "masks_band" / ph / f"{stem}.png"))
                w, h = size
                mine = np.zeros((h, w), np.uint8)
                for o in ann.get(k, []):
                    mine = np.maximum(mine, np.array(hbat.band_mask(o, size), np.uint8))
                assert np.array_equal(ref, mine), (f, ph, stem)
    print("自检通过：原始参数重画的掩码与 masks_band 逐像素相同", flush=True)

    record = json.load(open(OUT / "levels.json", encoding="utf-8")) if (OUT / "levels.json").exists() else {}
    record["_meta"] = {"median_thickness": ctx["median_thickness"], "n_repeat_pairs": len(ctx["repeat"]),
                       "repeat_sd": {n: float(s) for n, s in zip(("dx", "dy", "dlog_slope", "dlog_thickness", "dlog_span"),
                                                                ctx["repeat"].std(axis=0))},
                       "min_thickness": MIN_THICK, "noise_seed": NOISE_SEED}
    stats = {}
    for level in args.levels:
        pert = {k: [perturb(o, k, j, level, ctx) for j, o in enumerate(ann.get(k, []))] for k, _ in items.values()}
        masks = {stem: render(pert[k], size) for stem, (k, size) in items.items()}
        # 统计（全部 131 张，按图）
        ious, area = [], []
        tot_i = tot_u = tot_o = tot_p = 0
        for stem, (k, size) in items.items():
            orig = render([hbat.normalize_obj(o) for o in ann.get(k, [])], size) > 0
            new = masks[stem] > 0
            i, u = (orig & new).sum(), (orig | new).sum()
            tot_i += i; tot_u += u; tot_o += orig.sum(); tot_p += new.sum()
            if u:
                ious.append(i / u)
        stats[level] = {"pixel_iou": tot_i / tot_u, "median_image_iou": float(np.median(ious)), "area_ratio": tot_p / tot_o}
        print(f"{level}: IoU {stats[level]['pixel_iou']:.3f}  图中位 IoU {stats[level]['median_image_iou']:.3f}  "
              f"面积比 {stats[level]['area_ratio']:.2f}", flush=True)
        record[level] = {"kind": LEVELS[level][0], "param": LEVELS[level][1], "stats": stats[level],
                         "objects": {k: [{a: o.get(a) for a in ("x_vertex", "y_vertex", "slope", "span", "thickness")} for o in v]
                                     for k, v in pert.items()}}
        # 叠加图
        tiles = []
        for stem in sorted(items)[:8]:
            k, size = items[stem]
            orig = render([hbat.normalize_obj(o) for o in ann.get(k, [])], size) > 0
            t = np.zeros(orig.shape + (3,), np.uint8)
            t[orig] = 255
            t[masks[stem] > 0, 0] = 255; t[(masks[stem] > 0) & ~orig, 1:] = 0
            t[(masks[stem] > 0) & orig] = (255, 150, 150)
            tiles.append(np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=60))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(OUT / f"overlay_{level}.png")
        if args.check_only:
            continue
        for f in FOLDS:
            d = DATASETS / f"ds2_f{f}_s100_hbatbox"
            gt = json.load(open(d / "gt_boxes.json", encoding="utf-8"))
            out = d / f"masks_ann_{level}"
            if out.exists():
                shutil.rmtree(out)
            for ph in ("train", "val", "test"):
                (out / ph).mkdir(parents=True)
                for stem in gt[ph]:
                    if ph == "test":
                        shutil.copy(d / "masks_band" / ph / f"{stem}.png", out / ph / f"{stem}.png")
                    else:
                        Image.fromarray(masks[stem]).save(out / ph / f"{stem}.png")
        print(f"  写入 5 折 masks_ann_{level}/", flush=True)
    json.dump(record, open(OUT / "levels.json", "w", encoding="utf-8"), ensure_ascii=False)
    allstats = {k: v["stats"] for k, v in record.items() if k != "_meta"}
    L = ["# 第 21 轮：扰动后的 band 掩码相对原始掩码（全部 131 张图）", "",
         f"厚度中位数 {ctx['median_thickness']:.2f} px；重复标注差值 {len(ctx['repeat'])} 对，std："
         + "，".join(f"{n} {s:.3f}" for n, s in record["_meta"]["repeat_sd"].items()), "",
         "| 档位 | 类型 | 参数 | 像素 IoU（合计） | 逐图 IoU 中位数 | 面积比 |", "|---|---|---|---|---|---|"]
    for level in LEVELS:
        if level in allstats:
            s = allstats[level]
            L.append(f"| {level} | {LEVELS[level][0]} | {LEVELS[level][1]} | {s['pixel_iou']:.3f} | {s['median_image_iou']:.3f} | {s['area_ratio']:.2f} |")
    (OUT / "mask_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
