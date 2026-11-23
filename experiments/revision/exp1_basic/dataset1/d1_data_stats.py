"""数据集 1 与数据集 2 的标注分布对比（只读 HBAT 标注和划分文件，不训练）。

回答"ds2 上调好的实验 1 管线能不能原样搬到 ds1"：目标密度、框大小、图边截断、相邻重叠、
weak / single-wing 比例、slope 范围、相邻目标间距（换算到 224 输入），以及 CenterNet 热图上
多个目标中心落进同一格子（训练目标被覆盖）的数量。

    python d1_data_stats.py            # 打印并写 data_stats.md

ds1 按 item 的 category 分 pipe / rebar；ds2 全部合在一起。框 = HBAT band 多边形的外接框（裁到图内）。
"""
from __future__ import annotations

import collections
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import common as C  # noqa: E402
import hbat  # noqa: E402

FIELDS = ("visibility", "completeness", "distortion", "param_status")
CELL_CONFIGS = ((224, 8), (224, 4), (448, 8), (640, 8))   # (输入边长, 热图步长)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def frame_size(ds, item_id, item):
    """标注所在坐标系的 (W, H)：ds1 已应用裁剪的 item = 裁剪后的工作坐标系，其余 = 原图。"""
    fr = C.image_frame(ds, item_id) if C.DATASETS[ds].get("crop_json") else None
    return fr[1] if fr else C.image_size(C.REPO / item["eval_file"])


def collect(ds):
    split = C.load_split(ds)
    ann = C.load_json(C.DATASETS[ds]["hbat_json"])
    S = collections.defaultdict(lambda: collections.defaultdict(list))
    missing = 0
    for iid, item in split["items"].items():
        objs, _ = C.hbat_objects(ann, iid, item, item["eval_file"])
        if objs is None:
            missing += 1
            continue
        cat = item.get("category") if ds == "ds1" else "ds2"
        W, H = frame_size(ds, iid, item)
        s = S[cat]
        s["n_per_img"].append(len(objs))
        boxes, apex, cells = [], [], {cfg: collections.Counter() for cfg in CELL_CONFIGS}
        for o in objs:
            for f in FIELDS:
                s[f].append(o.get(f))
            if o.get("slope"):
                s["slope"].append(o["slope"])
            if o.get("thickness"):
                s["thick224"].append(o["thickness"] * 224 / H)
            if o.get("span"):
                s["span224"].append(o["span"] * 224 / W)
            if o.get("x_vertex") is not None:
                apex.append((o["x_vertex"] * 224 / W, o["y_vertex"] * 224 / H))
            poly, _ = hbat.hyperbola_band_polygon(o)
            if poly is None:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            b = (max(0.0, min(xs)), max(0.0, min(ys)), min(W, max(xs)), min(H, max(ys)))
            boxes.append(b)
            s["w_rel"].append((b[2] - b[0]) / W)
            s["h_rel"].append((b[3] - b[1]) / H)
            s["clipped"].append(min(xs) < 0 or max(xs) > W or max(ys) > H)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for size, stride in CELL_CONFIGS:
                cells[(size, stride)][(round(cx * size / W / stride), round(cy * size / H / stride))] += 1
        for i, a in enumerate(boxes):
            s["max_iou"].append(max([box_iou(a, b) for j, b in enumerate(boxes) if j != i] or [0.0]))
        for i, p in enumerate(apex):
            d = [math.dist(p, q) for j, q in enumerate(apex) if j != i]
            if d:
                s["nn224"].append(min(d))
        for cfg, cnt in cells.items():
            s[f"cell_lost_{cfg[0]}_{cfg[1]}"].append(sum(n - 1 for n in cnt.values()))
    return S, len(split["items"]), missing


def pct(v, qs):
    return "/".join(f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}" for x in np.percentile(v, qs))


def fmt_counts(vals):
    c = collections.Counter(vals)
    return ", ".join(f"{k} {n}" for k, n in sorted(c.items(), key=lambda kv: -kv[1]))


def main():
    out = ["# 数据集 1 / 数据集 2 标注分布（d1_data_stats.py 生成）", ""]
    rows = []
    for ds in ("ds1", "ds2"):
        S, n_items, missing = collect(ds)
        out.append(f"- {ds}：{n_items} 张，缺标注 {missing} 张")
        for cat, s in S.items():
            n = sum(s["n_per_img"])
            nb = len(s["w_rel"])
            r = {
                "组": f"{ds} {cat}",
                "图 / 目标": f"{len(s['n_per_img'])} / {n}",
                "每图目标 中位 / 最大": f"{np.median(s['n_per_img']):.0f} / {max(s['n_per_img'])}",
                "框宽/图宽 中位": f"{np.median(s['w_rel']):.3f}",
                "框高/图高 中位": f"{np.median(s['h_rel']):.3f}",
                "图边截断": f"{np.mean(s['clipped']):.2f}",
                "与邻框重叠 >0 / >0.3": f"{np.mean(np.array(s['max_iou']) > 0):.2f} / {np.mean(np.array(s['max_iou']) > 0.3):.2f}",
                "thickness @224 中位": f"{np.median(s['thick224']):.1f}",
                "span @224 中位": f"{np.median(s['span224']):.1f}",
                "slope 5/50/95%": pct(s["slope"], [5, 50, 95]),
                "slope > 6": f"{np.mean(np.array(s['slope']) > 6):.2f}",
                "最近顶点间距 @224 10/50/90%": pct(s["nn224"], [10, 50, 90]) if s["nn224"] else "-",
                "最近顶点间距 < 16 px": f"{np.mean(np.array(s['nn224']) < 16):.2f}" if s["nn224"] else "-",
            }
            for size, stride in CELL_CONFIGS:
                r[f"同格被覆盖 {size}/s{stride}"] = f"{sum(s[f'cell_lost_{size}_{stride}'])}/{nb}"
            for f in FIELDS:
                r[f] = fmt_counts(s[f])
            rows.append(r)
    out.append("")
    keys = list(rows[0])
    out.append("| 指标 | " + " | ".join(r["组"] for r in rows) + " |")
    out.append("|---|" + "---|" * len(rows))
    for k in keys[1:]:
        out.append(f"| {k} | " + " | ".join(r[k] for r in rows) + " |")
    out.append("")
    out.append("说明：@224 = 换算到 224×224 输入后的像素；\"同格被覆盖 S/sK\" = 输入 S、热图步长 K 时，"
               "中心落进已被别的目标占用的热图格子、训练目标被覆盖的目标数（只数有 band 的目标）。")
    text = "\n".join(out) + "\n"
    print(text)
    (HERE / "data_stats.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
