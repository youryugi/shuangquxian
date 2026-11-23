"""第 20 轮（想法 6 + 7）：attention 损失的像素权重图，只重新分配 band 像素内部的权重，不改 band 和背景的总权重。

Round 18 的诊断：重叠目标里约三成有一条 band 整条没画出来，翼尖覆盖率只有顶点的一半。监督是所有目标 band 的并集，
一条又小又弱的 band 整条漏掉，在像素损失里占比很小；翼尖像素少，也占比很小。

每张训练图：对每个目标 i（band 臂 = HBAT band 多边形，box-mask 臂 = 同一目标的 HBAT 框填实），
    raw_i(p) = (1 / A_i) · (1 + β · min(1, |x_p − x_c,i| / hw_i))       p ∈ M_i
    A_i = 目标 i 的掩码面积；x_c = 顶点 x（band）/ 框中心 x（box）；hw = 半跨度（band）/ 框半宽（box）
    raw(p) = Σ_i raw_i(p)；再在并集 U 上归一化，使 Σ_{p∈U} w(p) = |U|（band 像素平均权重 1）；背景 w = 1。
β = 0 时只做"每个目标的总权重相同"（想法 6）；β > 0 再加上翼尖加权（想法 7）。

输出 <dataset>/<out>/{train,val,test}/<stem>.npy（float32，原图分辨率）。train_basic.py --attn-weight-dir 读取。
同时检查：各目标掩码的并集和数据集里已有的 masks_band / masks_box 逐像素相同。

    python make_weight_maps.py --beta 2

第 24 轮：按 HBAT 的 visibility 加权（--vis clear=1,medium=2,weak=4）。和上面不同，**不在每张图内归一化**：
    w(p) = max_{i: p ∈ M_i} v_i（band 像素），背景 1
即 weak 目标的 band 像素在所有图里都是 ×4（只含一个 weak 目标的图也起作用）。box 臂用同一个目标的 visibility
（框按 diag_strata.label_gt_boxes 对回 HBAT 目标；对不上的按 clear）。输出 wmaps_{band,box}_vis<c><m><w>。

    python make_weight_maps.py --vis clear=1,medium=2,weak=4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATASETS = HERE.parent / "shared" / "datasets"
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import hbat  # noqa: E402

HBAT_JSON = REPO / "data" / "additional_utilities" / "hbat" / "annotations.json"


def weight_map(masks, xc, hw, beta):
    """masks: [N, H, W] bool；xc, hw: [N]。返回 (w, union)。"""
    H, W = masks.shape[1:]
    union = masks.any(0)
    if not union.any():
        return np.ones((H, W), np.float32), union
    xs = np.arange(W, dtype=np.float64)[None, :]
    raw = np.zeros((H, W), np.float64)
    for m, c, h in zip(masks, xc, hw):
        a = m.sum()
        if a == 0:
            continue
        u = np.minimum(1.0, np.abs(xs - c) / max(h, 1.0))
        raw += m * (1.0 + beta * u) / a
    w = np.ones((H, W), np.float64)
    w[union] = raw[union] * union.sum() / raw[union].sum()
    return w.astype(np.float32), union


def vis_map(masks, vis, vw):
    """masks: [N, H, W] bool；vis: [N] 字符串。返回 (w, union)：band 像素取覆盖它的目标的最大权重，背景 1。"""
    H, W = masks.shape[1:]
    w = np.ones((H, W), np.float32)
    union = masks.any(0) if len(masks) else np.zeros((H, W), bool)
    for m, v in zip(masks, vis):
        w[m] = np.maximum(w[m], vw.get(v, 1.0))
    return w, union


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--vis", default=None, help="第 24 轮：按 visibility 加权，如 clear=1,medium=2,weak=4（不做每图归一化，忽略 --beta）")
    args = ap.parse_args()
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    vw = None
    if args.vis:
        vw = {k: float(v) for k, v in (kv.split("=") for kv in args.vis.split(","))}
        sys.path.insert(0, str(HERE))
        from diag_strata import label_gt_boxes  # noqa: E402
    tag = f"b{args.beta:g}" if vw is None else "vis" + "".join(f"{vw.get(k, 1):g}" for k in ("clear", "medium", "weak"))
    stats = {"band": [], "box": []}
    for fold in args.folds:
        dd = DATASETS / f"ds2_f{fold}_s100_hbatbox"
        gt = json.load(open(dd / "gt_boxes.json", encoding="utf-8"))
        for ph, items in gt.items():
            for kind in ("band", "box"):
                (dd / f"wmaps_{kind}_{tag}" / ph).mkdir(parents=True, exist_ok=True)
            for stem, item in items.items():
                w_img, h_img = item["width"], item["height"]
                objs = [o for o in ann.get(item.get("band_key") or f"{stem}.jpg", []) if o.get("x_vertex") is not None]
                # band 臂：每个 HBAT band 单独画（和 common.render_band_mask 同一个 hbat.band_mask）
                bm = np.stack([np.array(hbat.band_mask(o, (w_img, h_img)), np.uint8) > 127 for o in objs]) if objs else np.zeros((0, h_img, w_img), bool)
                bxc = np.array([o["x_vertex"] for o in objs], float)
                bhw = np.array([(hbat.normalize_obj(dict(o)).get("span") or 2.0) / 2.0 for o in objs], float)   # 没有 band 的对象掩码为空，不参与
                # box 臂：gt 框（hbatbox 数据集里就是 HBAT 派生框），和 common.render_box_mask 相同的取整
                boxes = item["boxes"]
                xm = np.zeros((len(boxes), h_img, w_img), bool)
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    xm[i, int(math.floor(y1)):int(math.ceil(y2)), int(math.floor(x1)):int(math.ceil(x2))] = True
                xxc = np.array([(b[0] + b[2]) / 2 for b in boxes], float)
                xhw = np.array([(b[2] - b[0]) / 2 for b in boxes], float)
                if vw is not None:
                    bvis = [str(o.get("visibility", "clear")) for o in objs]
                    lab = label_gt_boxes({stem: item})[0]
                    xvis = [lab[(stem, j)]["visibility"] for j in range(len(boxes))]
                for kind, masks, xc, hw in (("band", bm, bxc, bhw), ("box", xm, xxc, xhw)):
                    if vw is None:
                        w, union = weight_map(masks, xc, hw, args.beta)
                    else:
                        w, union = vis_map(masks, bvis if kind == "band" else xvis, vw)
                    ref = np.array(Image.open(dd / f"masks_{kind}" / ph / f"{stem}.png")) > 127
                    if not np.array_equal(ref, union):
                        print(f"[warn] f{fold} {ph} {stem} {kind}: union of instance masks != masks_{kind} ({(ref != union).sum()} px)")
                    np.save(dd / f"wmaps_{kind}_{tag}" / ph / f"{stem}.npy", w)
                    if ph == "train" and union.any():
                        stats[kind].append((w[union].min(), w[union].max(), len(masks)))
    for kind, s in stats.items():
        s = np.array(s)
        print(f"{kind}: band-pixel weight min {s[:, 0].min():.2f}, median of per-image max {np.median(s[:, 1]):.2f}, max {s[:, 1].max():.2f}")


if __name__ == "__main__":
    main()
