"""第 24 轮准备：band 在 weak / grouped 的候选上有没有可用的信号（只读缓存，不训练）。

在原 fit 系统（Round 18 选中的 (0.7, 0.1, union, 0.5)，验证集选的阈值 conf）下看两类候选，算 AUC（真 vs 假），band 臂和 box-mask 臂对照：

W  阈值以下的检测（0.1 ≤ score < conf）："提上来能不能变成 TP"
   正例 = 它的框（fit 后）和一个系统漏掉的 GT IoU ≥ 0.5；weak 正例 = 那个 GT 是 visibility = weak
   特征：score；低阈值 band 证据 = 在 t = 0.3 的 band 上自己的拟合曲线 support / 框宽（没有拟合 = 0）；框内平均 attention
G  阈值以上、拟合到了别人（分数更高的检测）曲线上的检测（顶点相距 < 8 px）："真邻居还是重复检测"
   正例 = 它的原框和一个"没被系统其他框匹配"的 GT IoU ≥ 0.5（真邻居），否则是重复
   特征：score；和占用者原框的 IoU（越小越像真邻居，取负）；解释掉占用者曲线后在 t = 0.3 上重新拟合的 support / 框宽
输出 runs_cv/r24/evidence.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_bandbox as ib  # noqa: E402
import diag_strata as ds  # noqa: E402
from evaluate import box_iou, clip_box  # noqa: E402
from r22_assign import refit  # noqa: E402

OUT = HERE / "runs_cv" / "r24"
SRCS = ("hbatbox_flipfix_abs", "hbatbox_flipfix_boxmask")
T_LOW = 0.3


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort").astype(float) + 1
    # 平均秩处理并列
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def one(job):
    src, seed = job
    t, s_seed, mode, nms_thr = ib.base_fit_cfg(src)
    conf = ds.pooled_val_conf(ds.load_seed(src + "_fitbox", seed, ib.HERE / "runs_cv")[0])
    gt_all = {}
    for f in ib.FOLDS:
        gt_all.update(json.load(open(ib.data_dir(f) / "gt_boxes.json"))["test"])
    labels = ds.label_gt_boxes(gt_all)[0]
    W, G = defaultdict(list), defaultdict(list)
    for f in ib.FOLDS:
        dets, metas, attn = ib.load_cache(src, f, seed, "test")
        fits = ib.fits_for(src, f, seed, "test", t)
        fits_low = ib.fits_for(src, f, seed, "test", T_LOW)
        for stem, item in json.load(open(ib.data_dir(f) / "gt_boxes.json"))["test"].items():
            Wd, Hd = item["width"], item["height"]
            d = dets[stem]; sc = metas[stem]["scale"]; sx, sy = sc
            ref = [list(clip_box(p, Wd, Hd)) for p in ib.apply_fits(d, fits[stem], sc, s_seed, mode)]
            post = ib.nms(ref, nms_thr)
            m = ds.match_detail(post, item["boxes"], conf)[0]
            missed = [j for j in range(len(item["boxes"])) if m[j] is None]
            matched_boxes = {tuple(m[j][0]) for j in range(len(item["boxes"])) if m[j] is not None}
            A = cv2.resize(attn[stem].astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
            for i, p in enumerate(d):
                bw = max((p[2] - p[0]) * sx, 8.0)
                if not (0.1 <= p[4] < conf):
                    continue
                fl = fits_low[stem].get(str(i))
                ev = (fl["support"] / bw) if fl else 0.0
                x1, y1, x2, y2 = [int(round(v)) for v in (p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy)]
                inbox = float(A[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].mean()) if x2 > x1 and y2 > y1 else 0.0
                hits = [j for j in missed if box_iou(ref[i], item["boxes"][j]) >= 0.5]
                lab = "pos" if hits else "neg"
                W[(lab, "score")].append(p[4]); W[(lab, "band_ev")].append(ev); W[(lab, "inbox")].append(inbox)
                if hits and any(labels[(stem, j)]["visibility"] == "weak" for j in hits):
                    W[("weak", "score")].append(p[4]); W[("weak", "band_ev")].append(ev); W[("weak", "inbox")].append(inbox)
            # G：撞曲线
            fi = {i: fits[stem].get(str(i)) for i in range(len(d))}
            order = sorted(range(len(d)), key=lambda i: -d[i][4])
            for r_, i in enumerate(order):
                if d[i][4] < conf or fi[i] is None:
                    continue
                occ = [k for k in order[:r_] if fi[k] is not None and abs(fi[k]["xv"] - fi[i]["xv"]) < 8 and abs(fi[k]["yv"] - fi[i]["yv"]) < 8]
                if not occ:
                    continue
                o = occ[0]
                own = [j for j in range(len(item["boxes"])) if box_iou(d[i], item["boxes"][j]) >= 0.5]
                occ_gt = [j for j in range(len(item["boxes"])) if box_iou(ref[o], item["boxes"][j]) >= 0.5]
                lab = "pos" if any(j not in occ_gt for j in own) else "neg"
                raw_i = (d[i][0] * sx, d[i][1] * sy, d[i][2] * sx, d[i][3] * sy)
                nf = refit((src, f, seed, stem, i), attn[stem], T_LOW, raw_i, [fi[k] for k in occ])
                ev = 0.0
                if nf is not None:
                    fl = ib.fit_seed_support if hasattr(ib, "fit_seed_support") else None
                    ev = (nf["box"][2] - nf["box"][0]) / max(raw_i[2] - raw_i[0], 8.0)   # 重新拟合出的曲线覆盖了原框多宽
                G[(lab, "score")].append(d[i][4]); G[(lab, "neg_iou_occ")].append(-box_iou(d[i], d[o])); G[(lab, "refit_ev")].append(ev)
                if lab == "pos" and any(labels[(stem, j)]["grouped"].startswith("grouped") for j in own):
                    G[("grp", "n")].append(1)
    return src, seed, {"W": {"|".join(k): v for k, v in W.items()}, "G": {"|".join(k): v for k, v in G.items()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(20) as pool:
        res = pool.map(one, [(s, k) for s in SRCS for k in range(10)])
    L = ["# band 在 weak / grouped 候选上的信号（测试集，10 seeds；AUC 每个 seed 算后取均值）", ""]
    for part, feats, title in (("W", ("score", "band_ev", "inbox"), "W：阈值以下的检测，能不能变成 TP"),
                               ("G", ("score", "neg_iou_occ", "refit_ev"), "G：撞曲线的检测，真邻居 vs 重复")):
        L += [f"## {title}", "", "| 来源 | 每 seed 正例 / 负例 | " + " | ".join(f"AUC {x}" for x in feats) +
              (" | weak 正例数 | " + " | ".join(f"weak AUC {x}" for x in feats) if part == "W" else "") + " |",
              "|---|---|" + "---|" * (len(feats) * (2 if part == "W" else 1) + (1 if part == "W" else 0))]
        for src in SRCS:
            rs = [r for s, _, r in res if s == src]
            npos = np.mean([len(r[part].get("pos|score", [])) for r in rs]); nneg = np.mean([len(r[part].get("neg|score", [])) for r in rs])
            row = f"| {src.replace('hbatbox_flipfix_', '')} | {npos:.1f} / {nneg:.1f} | " + " | ".join(
                f"{np.nanmean([auc(r[part].get(f'pos|{x}', []), r[part].get(f'neg|{x}', [])) for r in rs]):.3f}" for x in feats)
            if part == "W":
                nw = np.mean([len(r["W"].get("weak|score", [])) for r in rs])
                row += f" | {nw:.1f} | " + " | ".join(
                    f"{np.nanmean([auc(r['W'].get(f'weak|{x}', []), r['W'].get(f'neg|{x}', [])) for r in rs]):.3f}" for x in feats)
            L.append(row + " |")
        L.append("")
    (OUT / "evidence.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
