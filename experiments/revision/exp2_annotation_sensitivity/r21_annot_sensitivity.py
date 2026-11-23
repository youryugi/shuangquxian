"""第 21 轮：标注敏感性汇总（回应 R1-4、R2-4）。只读已有结果，不训练。

    python ../exp1_basic/aggregate_cv.py && python r21_annot_sensitivity.py

读 ../exp1_basic/runs_cv/cv_summary.json（各档位 hbatbox_flipfix_abs_ann_<档位>{,_bandbox_nms,_fitbox,_r18_fitbox}，
标准档 = Round 18 的 hbatbox_flipfix_abs*，baseline / box-mask = Round 18 的 hbatbox_flipfix_{none,boxmask}*），
../exp1_basic/runs_cv/bandbox_cache/<exp>/f*_seed*/attn_test.npz（预测的 band）、results/levels.json（掩码统计）。

表 1：每个档位的 F1（5 折合并，10 seeds 均值 ± std）：band fusion、Voronoi + NMS、band + fit（各自选参）、band + fit（固定 Round 18 参数）
表 2：配对 Δ（95% CI，胜/平/负）：baseline → 本档 band + fit；标准档 band + fit → 本档 band + fit；
      box-mask 臂最好的后处理（Voronoi + NMS）→ 本档 band + fit；baseline → 本档 fusion
表 3：机制：预测 band 相对**原始**测试集 band 的像素 IoU、原始中心线覆盖率（t = 0.5，224×224），grouped / alone recall（band + fit）
输出 results/summary.md、summary.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "exp1_basic"     # 训练、后处理、配对聚合都在实验 1 的管线里（run_seeds.py / infer_bandbox.py / aggregate_cv.py）
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP1))
from aggregate_cv import paired  # noqa: E402
from diag_strata import analyse_seed, hbat, HBAT_JSON, label_gt_boxes, load_seed  # noqa: E402
from make_annot_masks import LEVELS, TIER1, TIER2  # noqa: E402

RUNS = EXP1 / "runs_cv"
CACHE = RUNS / "bandbox_cache"
DATASETS = HERE.parent / "shared" / "datasets"
OUT = HERE / "results"
SEEDS, FOLDS = range(10), range(5)
STD = "hbatbox_flipfix_abs"
THICK_ORDER = ("t025", "t050", "t075", "STD", "t125", "t150", "t200", "tuni")
LABEL = {"STD": "标准（×1）", "t025": "厚度 ×0.25", "t050": "厚度 ×0.5", "t075": "厚度 ×0.75", "t125": "厚度 ×1.25",
         "t150": "厚度 ×1.5", "t200": "厚度 ×2", "tuni": "统一厚度（中位数）", "sp075": "跨度 ×0.75", "sp125": "跨度 ×1.25",
         "ap2": "顶点随机移 2 px", "ap5": "顶点随机移 5 px", "sl10": "slope ±10%", "sl20": "slope ±20%",
         "rep1": "重复标注差值 ×1", "rep2": "重复标注差值 ×2"}
ORDER = list(THICK_ORDER) + ["sp075", "sp125"] + ["STD"] + list(TIER2)


def src(lv):
    return STD if lv == "STD" else f"{STD}_ann_{lv}"


def centerlines(stem_key, size):
    ann = json.load(open(HBAT_JSON, encoding="utf-8")) if not hasattr(centerlines, "ann") else centerlines.ann
    centerlines.ann = ann
    w, h = size
    m = np.zeros((h, w), np.uint8)
    for o in ann.get(stem_key, []):
        _, cl = hbat.hyperbola_band_polygon(o)
        if cl:
            pts = np.array([[round(x), round(y)] for x, y in cl], np.int32)
            cv2.polylines(m, [pts], False, 1, 1)
    return m > 0


def band_quality(exp, t=0.5):
    """每个 seed：5 折测试集像素合计的 IoU（对原始 band）、原始中心线被预测 band 覆盖的比例。缓存缺失返回 None。"""
    ious, covs = [], []
    for s in SEEDS:
        I = U = C = N = 0
        for f in FOLDS:
            p = CACHE / exp / f"f{f}_seed{s}" / "attn_test.npz"
            if not p.exists():
                return None
            z = np.load(p)
            gt = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"]
            for stem in z.files:
                g = gt[stem]
                pred = cv2.resize(z[stem].astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR) > t
                ref = cv2.imread(str(DATASETS / f"ds2_f{f}_s100_hbatbox" / "masks_band" / "test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                ref = cv2.resize(ref, (224, 224), interpolation=cv2.INTER_NEAREST) > 127
                cl = centerlines(g["band_key"], (g["width"], g["height"]))
                if cl.shape != (224, 224):
                    cl = cv2.resize(cl.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST) > 0
                I += (pred & ref).sum(); U += (pred | ref).sum(); C += (pred & cl).sum(); N += cl.sum()
        ious.append(I / max(U, 1)); covs.append(C / max(N, 1))
    return ious, covs


def main():
    cv = json.load(open(RUNS / "cv_summary.json", encoding="utf-8"))["per_exp"]

    def f1s(exp):
        if exp not in cv or len(cv[exp]) < 10:
            return None
        return [cv[exp][str(s)]["primary"]["test"]["F1"] for s in SEEDS]

    chosen = json.load(open(RUNS / "bandbox" / "chosen.json", encoding="utf-8"))
    stats = json.load(open(OUT / "levels.json", encoding="utf-8"))
    base, bm_best, bm_fit, std_fit = (f1s("hbatbox_flipfix_none"), f1s("hbatbox_flipfix_boxmask_bandbox_nms"),
                                      f1s("hbatbox_flipfix_boxmask_fitbox"), f1s(f"{STD}_fitbox"))
    res = {"levels": {}}
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if v else "—"

    def pfmt(p):
        sig = "" if p["ci95"][0] > 0 or p["ci95"][1] < 0 else "（n.s.）"
        return f"{p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]，{p['wins']}/{p['ties']}/{p['losses']}{sig}"

    # 分层 recall 需要的标签（全部 131 张测试图）
    gt_all = {}
    for f in FOLDS:
        gt_all.update(json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"])
    labels, _ = label_gt_boxes(gt_all)

    def strata(exp):
        g, a = [], []
        for s in SEEDS:
            r = load_seed(exp, s, RUNS)
            if r is None:
                return None
            o = analyse_seed(*r, labels, ["grouped"])["recall"]
            gm = [v for k, v in o.items() if k.startswith("grouped|grouped")][0]
            am = o["grouped|alone"]
            g.append(gm[0] / gm[1]); a.append(am[0] / am[1])
        return g, a

    levels = ["STD"] + list(TIER1) + list(TIER2)
    for lv in levels:
        e = src(lv)
        r = {"fusion": f1s(e), "vnms": f1s(f"{e}_bandbox_nms"), "fit": f1s(f"{e}_fitbox"),
             "fit_r18": f1s(f"{e}_fitbox" if lv == "STD" else f"{e}_r18_fitbox"),
             "fit_config": chosen.get(f"{e}_fitbox", {}).get("config")}
        if r["fit"] is None:
            print(f"[skip] {lv}: 结果不全"); continue
        r["pairs"] = {"baseline→fit": paired(base, r["fit"]), "std_fit→fit": paired(std_fit, r["fit"]),
                      "std_fit→fit_r18": paired(std_fit, r["fit_r18"]) if r["fit_r18"] else None,
                      "boxmask_vnms→fit": paired(bm_best, r["fit"]), "boxmask_fit→fit": paired(bm_fit, r["fit"]),
                      "baseline→fusion": paired(base, r["fusion"])}
        bq = band_quality(e)
        r["band_iou"], r["cl_cov"] = (bq if bq else (None, None))
        st = strata(f"{e}_fitbox")
        r["grouped_recall"], r["alone_recall"] = (st if st else (None, None))
        res["levels"][lv] = r
        print(f"{lv}: fit {np.mean(r['fit']):.3f}", flush=True)

    L = ["# 第 21 轮：标注敏感性（数据集 2，5 折合并，10 seeds）", "",
         "只换训练 / 验证集的 band 监督掩码（make_annot_masks.py），检测框、测试集、baseline 和 box-mask 臂都与 Round 18 相同（配对初始化，`--workers 1`）。",
         f"参考：baseline {ms(base)}；box-mask + Voronoi + NMS（box-mask 臂最好的后处理）{ms(bm_best)}；box-mask + fit {ms(bm_fit)}。", "",
         "## 表 1：F1（均值 ± std）", "",
         "| 档位 | 掩码 IoU（对原始） | band fusion | band + Voronoi + NMS | **band + fit**（各自选参） | band + fit（固定 R18 参数） | 选中参数 |",
         "|---|---|---|---|---|---|---|"]
    for lv in ORDER:
        if lv not in res["levels"]:
            continue
        r = res["levels"][lv]
        miou = "1.000" if lv == "STD" else f"{stats[lv]['stats']['pixel_iou']:.3f}"
        L.append(f"| {LABEL[lv]} | {miou} | {ms(r['fusion'])} | {ms(r['vnms'])} | **{ms(r['fit'])}** | {ms(r['fit_r18'])} | {r['fit_config']} |")
    L += ["", "## 表 2：配对 ΔF1（B − A，95% CI，胜/平/负）", "",
          "| 档位 | baseline → band + fit | 标准档 band + fit → 本档 | 同上，固定 R18 参数 | box-mask 最好的后处理 → band + fit | box-mask + fit → band + fit | baseline → fusion |",
          "|---|---|---|---|---|---|---|"]
    for lv in ORDER:
        if lv not in res["levels"]:
            continue
        p = res["levels"][lv]["pairs"]
        same = lv == "STD"
        L.append(f"| {LABEL[lv]} | {pfmt(p['baseline→fit'])} | {'—' if same else pfmt(p['std_fit→fit'])} | "
                 f"{'—' if same or not p['std_fit→fit_r18'] else pfmt(p['std_fit→fit_r18'])} | {pfmt(p['boxmask_vnms→fit'])} | "
                 f"{pfmt(p['boxmask_fit→fit'])} | {pfmt(p['baseline→fusion'])} |")
    L += ["", "## 表 3：机制（预测 band 对原始测试 band；t = 0.5；band + fit 的分层 recall@IoU0.5）", "",
          "| 档位 | band IoU | 中心线覆盖率 | grouped recall | alone recall |", "|---|---|---|---|---|"]
    for lv in ORDER:
        if lv not in res["levels"]:
            continue
        r = res["levels"][lv]
        L.append(f"| {LABEL[lv]} | {ms(r['band_iou'])} | {ms(r['cl_cov'])} | {ms(r['grouped_recall'])} | {ms(r['alone_recall'])} |")
    L += ["", "说明：厚度档的 band IoU 对原始（标准厚度）band 计算，细带 / 粗带本来就会拉低它；中心线覆盖率不受厚度影响，看的是曲线位置有没有画对。",
          "掩码 IoU 是扰动后的训练掩码和原始掩码的像素 IoU（全部 131 张图，见 mask_stats.md）。"]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
