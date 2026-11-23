"""漏检分类：hbatbox_aug_{none,abs}，每个 seed 用 5 折合并验证集选阈值，然后对测试集每个漏检 GT 分类。"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from evaluate import collect, select_conf, box_iou  # noqa

RUNS = ROOT / "exp1_basic" / "runs_cv" / "ds2"
DS = ROOT / "shared" / "datasets"
EXPS = sys.argv[1:] or ["hbatbox_aug_none", "hbatbox_aug_abs"]
# 传 "refine:<exp>" 读 runs_cv/refine/<exp>/f*_seed*/preds_test_refined.json（阈值仍在原 preds_val 上选；修正只改框不改分数，验证集修正后阈值会略不同，这里取近似）
REFINE = ROOT / "exp1_basic" / "runs_cv" / "refine"



def greedy(preds, gts, thr=0.5):
    """返回每个 GT 是否被匹配（score 降序贪心，同 evaluate.match_image 逻辑）。"""
    preds = sorted(preds, key=lambda p: -p[4])
    used = [False] * len(gts)
    for p in preds:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            i = box_iou(p[:4], g)
            if i >= best:
                best, bj = i, j
        if bj >= 0:
            used[bj] = True
    return used


summary = {}
missed_sets = {}
for exp in EXPS:
    cats = Counter(); n_gt_total = 0; attrs = defaultdict(list)
    missed_count = Counter()
    for seed in range(10):
        sv, tv, nv = [], [], 0
        per_fold = {}
        for f in range(5):
            base = exp.split(":", 1)[1] if exp.startswith("refine:") else exp
            rd = RUNS / base / f"f{f}_seed{seed}"
            gt = json.load(open(DS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))
            pv = json.load(open(rd / "preds_val.json"))
            pt = json.load(open((REFINE / base / f"f{f}_seed{seed}" / "preds_test_refined.json") if exp.startswith("refine:") else (rd / "preds_test.json")))
            s, t, n = collect(pv, gt["val"]); sv.append(s); tv.append(t); nv += n
            per_fold[f] = (pt, gt["test"])
        conf, _ = select_conf(np.concatenate(sv), np.concatenate(tv), nv)
        for f, (pt, gtt) in per_fold.items():
            for stem, g in gtt.items():
                gts = g["boxes"]; preds = pt.get(stem, [])
                kept = [p for p in preds if p[4] >= conf]
                matched = greedy(kept, gts)
                n_gt_total += len(gts)
                for j, gb in enumerate(gts):
                    if matched[j]:
                        continue
                    missed_count[(stem, j)] += 1
                    ious_all = [(box_iou(p[:4], gb), p[4]) for p in preds]
                    ious_kept = [box_iou(p[:4], gb) for p in kept]
                    best_any = max([i for i, _ in ious_all], default=0)
                    best_kept = max(ious_kept, default=0)
                    if best_kept >= 0.5:
                        c = "A 框对了但被相邻目标抢走"
                    elif best_any >= 0.5:
                        # 有位置正确的候选，只是分数低于阈值
                        sc = max(s for i, s in ious_all if i >= 0.5)
                        c = "B 位置对、分数不够(<阈值)"
                        attrs["B_score_ratio"].append(sc / conf)
                    elif best_kept >= 0.1:
                        c = "C 附近有框但定位/尺寸不准(IoU0.1-0.5)"
                    elif best_any >= 0.1:
                        c = "D 只有低分且不准的候选"
                    else:
                        c = "E 完全没有候选"
                    cats[c] += 1
                    w, h = gb[2] - gb[0], gb[3] - gb[1]
                    attrs[c[0] + "_size"].append((w * h) ** 0.5)
                    attrs[c[0] + "_nbox"].append(len(gts))
    summary[exp] = (cats, n_gt_total, attrs)
    missed_sets[exp] = missed_count
    tot = sum(cats.values())
    print(f"== {exp}: GT 实例(10 seed 累计) {n_gt_total}, 漏检 {tot} (召回 {1 - tot / n_gt_total:.3f})")
    for c in sorted(cats):
        print(f"   {c:34s} {cats[c]:5d}  {cats[c] / tot:6.1%}  size中位 {np.median(attrs[c[0] + '_size']):.0f}px  同图框数中位 {np.median(attrs[c[0] + '_nbox']):.0f}")
    if attrs["B_score_ratio"]:
        r = np.array(attrs["B_score_ratio"])
        print(f"   B 类候选分数/阈值: 中位 {np.median(r):.2f}, >0.5 的占 {(r > 0.5).mean():.0%}")

# 两臂漏的是不是同一批目标
if len(EXPS) == 2:
    a, b = missed_sets[EXPS[0]], missed_sets[EXPS[1]]
    keys = set(a) | set(b)
    hard_a = {k for k in keys if a[k] >= 8}; hard_b = {k for k in keys if b[k] >= 8}
    print(f"\n 10 seed 中 >=8 次被漏的 GT: {EXPS[0]} {len(hard_a)}, {EXPS[1]} {len(hard_b)}, 交集 {len(hard_a & hard_b)}")
    va = np.array([a[k] for k in keys]); vb = np.array([b[k] for k in keys])
    print(f" 每个 GT 漏检次数的两臂相关 r = {np.corrcoef(va, vb)[0, 1]:.2f}  (共 {len(keys)} 个至少被漏过一次的 GT)")
    allgt = 0
    for f in range(5):
        gt = json.load(open(DS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))
        allgt += sum(len(g["boxes"]) for g in gt["test"].values())
    print(f" 测试 GT 总数 {allgt}；两臂都从未漏过的 {allgt - len(keys)}")

# 难目标(两臂都 >=8/10 被漏) vs 易目标(两臂都 <=2/10) 的特征
import cv2
feat = defaultdict(list)
for f in range(5):
    gt = json.load(open(DS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))
    for stem, g in gt["test"].items():
        img = cv2.imread(str(ROOT.parents[1] / g["file"]), 0).astype(np.float32)
        bg = np.median(img); sd = img.std() + 1e-6
        for j, gb in enumerate(g["boxes"]):
            k = (stem, j); ma, mb = a.get(k, 0), b.get(k, 0)
            grp = "hard" if (ma >= 8 and mb >= 8) else ("easy" if (ma <= 2 and mb <= 2) else None)
            if grp is None: continue
            x0, y0, x1, y1 = [int(round(v)) for v in gb]
            patch = img[max(y0,0):y1, max(x0,0):x1]
            contrast = (np.percentile(patch, 95) - bg) / sd if patch.size else 0
            maxov = max([box_iou(gb, o) for jj, o in enumerate(g["boxes"]) if jj != j], default=0)
            feat[grp + "_size"].append(((x1-x0)*(y1-y0))**0.5)
            feat[grp + "_contrast"].append(abs(contrast))
            feat[grp + "_nbox"].append(len(g["boxes"]))
            feat[grp + "_overlap"].append(maxov)
            feat[grp + "_unused"].append(g.get("stratum") == "unused")
            feat[grp + "_edge"].append(x0 <= 2 or y0 <= 2 or x1 >= 222 or y1 >= 222)
for grp in ["hard", "easy"]:
    n = len(feat[grp + "_size"])
    print(f"{grp:4s} n={n:3d}  尺寸中位 {np.median(feat[grp+'_size']):.0f}px  对比度中位 {np.median(feat[grp+'_contrast']):.2f}σ  "
          f"同图框数中位 {np.median(feat[grp+'_nbox']):.0f}  与相邻框IoU>0.1占 {np.mean(np.array(feat[grp+'_overlap'])>0.1):.0%}  "
          f"贴图像边缘 {np.mean(feat[grp+'_edge']):.0%}  unused图 {np.mean(feat[grp+'_unused']):.0%}")
