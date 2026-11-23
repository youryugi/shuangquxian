"""第 22 轮：fit 之后检测和曲线怎么配对（想法 1–3，只改推理；用 bandbox_cache 里缓存的检测和拟合，不重训）。

Round 20 定位到：band 整条没画出来的那个目标，它的检测会被拟合到邻居的曲线上（"撞曲线"），变成重复框后被 NMS 删掉。
这里把"拟合之后"这一步换成显式的配对：

  1. 曲线：分数 ≥ s_seed 且拟合有效的检测，按分数从高到低，顶点相距 ≤ delta px（输入分辨率）的归成同一条曲线。
  2. 配对（assign）
       greedy  每条曲线归分数最高的那个检测（Round 20 曲线归属的做法）
       hung    全图一起配（想法 3）：候选 = 分数 ≥ s_seed 的检测 × 曲线，代价 = 检测原框的顶边中点到曲线顶点的归一化距离
               hypot(dx / w, dy / h)，超过 gate 的不许配；匈牙利算法取"配上的对数最多、总距离最小"
     配上的检测：原框和这条曲线的拟合框按该来源已选的 mode 合成（与 apply_fits 相同）。
  3. 没配上、但自己的拟合曲线被别的检测占了的检测（撞曲线），判断是重复还是另一个目标（dup）
       all     全当另一个目标
       iou     和占用者的原框 IoU < thr 才算另一个目标（Round 20 的做法）
       geo     自己原框的顶边中点离占用者曲线顶点的归一化距离 > thr 才算另一个目标（想法 2）
     另一个目标：保留原框，分数 × alpha（想法 1；alpha = 1 不降分，alpha = 0 等于全删）；重复：删掉。
  5. （第 2 阶段，想法 5）撞曲线的检测先试一次"解释掉再拟合"：attention 按 t_low 二值化，把已配出去的曲线抹掉，从这个检测的原框出发
     重新拟合；新曲线不和已配出去的曲线重合、且顶点离原框顶边中点的归一化距离 ≤ refit gate，就用新曲线定框（分数不变）；
     否则按第 3 步处理。
  4. 最后按分数做贪心 NMS（阈值 = 该来源已选的 NMS）；涉及保留原框的检测时比原框，且不和自己的占用者比（第 3 步已经判过）。

(t, s_seed, mode, NMS) 固定用该来源 *_fitbox 已选中的那组（chosen.json）。新参数在 5 折合并验证集上选：每组参数对 10 个 seed
各算一次验证 F1（置信度阈值也在验证集选），取 10 个 seed 均值最大的一组。测试集不参与选择。

    python r22_assign.py --jobs 8             # 第 1 阶段（想法 1–3）：runs_cv/r22/grid_<src>.json + summary.md
    python r22_assign.py --jobs 8 --stage 2   # 第 2 阶段（想法 5）：在第 1 阶段每类选中的参数上加重拟合，grid2_<src>.json，重写 summary.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_bandbox as ib  # noqa: E402
import diag_strata as ds  # noqa: E402
from evaluate import box_iou, clip_box, prf_at, select_conf  # noqa: E402
from extend_fit import curve_y  # noqa: E402
from fit_hyperbola import fit_box, fit_seed  # noqa: E402
from aggregate_cv import paired  # noqa: E402
from r20_compare import per_seed  # noqa: E402

OUT = HERE / "runs_cv" / "r22"
SRCS = ("hbatbox_flipfix_abs", "hbatbox_flipfix_boxmask", "hbatbox_flipfix_abs_unet", "hbatbox_flipfix_boxmask_unet")
DELTAS = (8, 12)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
IOU_THR = (0.3, 0.5, 0.7)
GEO_THR = (0.1, 0.2, 0.3, 0.5)
GATES = (0.3, 0.6, 1.0)


def grid():
    """(delta, assign, gate, dup, thr, alpha)；None = 原来的流程（apply_fits + NMS）。"""
    cfgs = [None]
    for dl in DELTAS:
        for assign, gate in [("greedy", None)] + [("hung", g) for g in GATES]:
            cfgs.append((dl, assign, gate, "all", None, 0.0))
            for a in ALPHAS:
                cfgs.append((dl, assign, gate, "all", None, a))
                cfgs += [(dl, assign, gate, "geo", r, a) for r in GEO_THR]
                if assign == "greedy":
                    cfgs += [(dl, assign, gate, "iou", r, a) for r in IOU_THR]
    return cfgs




def topc_dist(r, f):
    x1, y1, x2, y2 = r
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return float(np.hypot((f["xv"] - (x1 + x2) / 2) / w, (f["yv"] - y1) / h))


def combine(r, fb, mode):
    x1, y1, x2, y2 = r
    bx1, by1, bx2, by2 = fb
    if mode == "replace":
        return fb
    if mode == "union":
        return (min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2))
    return ((x1 + bx1) / 2, (y1 + by1) / 2, (x2 + bx2) / 2, (y2 + by2) / 2)


def valid_fit(d, f, s_seed):
    return f is not None and d[4] >= s_seed and f["box"][2] - f["box"][0] >= 2 and f["box"][3] - f["box"][1] >= 2


_REFIT = {}


def refit(key, attn, t_low, box, owned):
    """想法 5（解释掉已占用的曲线再拟合）：attention 按 t_low 二值化，把已配出去的曲线（中心线 ± 厚度/2 + 2 px）从 band 里抹掉，
    再从这个检测的原框出发拟合一次。key 用于缓存。"""
    k = (key, t_low, tuple((round(f["xv"], 1), round(f["yv"], 1), round(f["slope"], 3)) for f in owned))
    if k not in _REFIT:
        band = ib.attn_to_band(attn, t_low)
        H, W = band.shape
        ys = np.arange(H, dtype=np.float64)[:, None]
        for f in owned:
            yc = curve_y(f["xv"], f["yv"], f["slope"], np.arange(W))
            band &= ~(np.abs(ys - yc[None, :]) <= f["thick"] / 2 + 2)
        nf = fit_seed(band, box)
        _REFIT[k] = None if nf is None else {"box": fit_box(nf, H, W), "xv": nf["xv"], "yv": nf["yv"], "slope": nf["slope"], "thick": nf["thick"]}
    return _REFIT[k]


def assign_image(d, fits_img, sc, base, cfg, attn=None, key=None):
    _, s_seed, mode, nms_thr = base
    if cfg is None:
        return ib.nms(ib.apply_fits(d, fits_img, sc, s_seed, mode), nms_thr)
    delta, assign, gate, dup, thr, alpha = cfg[:6]
    rcfg = cfg[6] if len(cfg) > 6 else None          # (t_low, rgate) 或 None
    sx, sy = sc
    n = len(d)
    raw = [(p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy) for p in d]
    rf = [fits_img.get(str(i), fits_img.get(i)) for i in range(n)]
    rf = [f if valid_fit(d[i], f, s_seed) else None for i, f in enumerate(rf)]
    order = sorted(range(n), key=lambda i: -d[i][4])
    # 1. 曲线
    reps, cl = [], [None] * n
    for i in order:
        f = rf[i]
        if f is None:
            continue
        for c, r in enumerate(reps):
            if abs(r["xv"] - f["xv"]) <= delta and abs(r["yv"] - f["yv"]) <= delta:
                cl[i] = c
                break
        else:
            cl[i] = len(reps)
            reps.append(f)
    # 2. 配对
    match = {}
    if assign == "greedy":
        for i in order:
            if cl[i] is not None and cl[i] not in match.values():
                match[i] = cl[i]
    elif reps:
        cand = [i for i in order if d[i][4] >= s_seed]
        if cand:
            M = np.array([[topc_dist(raw[i], r) for r in reps] for i in cand])
            C = np.where(M <= gate, M, 1e6)
            for a, b in zip(*linear_sum_assignment(C)):
                if C[a, b] < 1e6:
                    match[cand[a]] = int(b)
    owner = {c: i for i, c in match.items()}
    # 3. 输出：(原图坐标框, 分数, 检测序号, 占用者或 None；占用者非 None = 保留原框的撞曲线检测)
    items = []
    for i in range(n):
        if i in match:
            b = combine(raw[i], reps[match[i]]["box"], mode)
            items.append(([b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy], d[i][4], i, None))
            continue
        o = owner.get(cl[i]) if cl[i] is not None else None
        if o is None:
            items.append((list(d[i][:4]), d[i][4], i, None))
            continue
        if rcfg is not None:
            owned = [reps[c] for c in owner]
            nf = refit((key, i), attn, rcfg[0], raw[i], owned)
            if (nf is not None and nf["box"][2] - nf["box"][0] >= 2 and nf["box"][3] - nf["box"][1] >= 2
                    and all(abs(f["xv"] - nf["xv"]) > delta or abs(f["yv"] - nf["yv"]) > delta for f in owned)
                    and topc_dist(raw[i], nf) <= rcfg[1]):
                b = combine(raw[i], nf["box"], mode)
                items.append(([b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy], d[i][4], i, o))
                continue
        if alpha <= 0:
            continue
        if dup == "all":
            distinct = True
        elif dup == "iou":
            distinct = box_iou(d[i], d[o]) < thr
        else:
            distinct = topc_dist(raw[i], reps[cl[i]]) > thr
        if distinct:
            items.append((list(d[i][:4]), d[i][4] * alpha, i, o))
    # 4. NMS
    keep = []
    for it in sorted(items, key=lambda t: -t[1]):
        sup = False
        for k in keep:
            if it[3] is not None or k[3] is not None:
                if it[3] == k[2] or k[3] == it[2]:
                    continue
                sup = nms_thr is not None and box_iou(d[it[2]], d[k[2]]) > nms_thr
            else:
                sup = nms_thr is not None and box_iou(it[0], k[0]) > nms_thr
            if sup:
                break
        if not sup:
            keep.append(it)
    return [[*k[0], float(k[1])] for k in keep]


_GT = {}


def gts(phase):
    if phase not in _GT:
        _GT[phase] = {f: json.load(open(ib.data_dir(f) / "gt_boxes.json"))[phase] for f in ib.FOLDS}
    return _GT[phase]


def system(src, seed, phase, cfg):
    base = ib.base_fit_cfg(src)
    out = {}
    for f in ib.FOLDS:
        dets, metas, attn = ib.load_cache(src, f, seed, phase)
        fits = ib.fits_for(src, f, seed, phase, base[0])
        out[f] = {stem: assign_image(d, fits[stem], metas[stem]["scale"], base, cfg, attn[stem] if attn is not None else None,
                                     (src, f, seed, phase, stem)) for stem, d in dets.items()}
    return out


def eval_seed(job):
    src, seed, cfgs = job
    res = []
    for cfg in cfgs:
        sv, tv, nv = ib.pooled_f1(system(src, seed, "val", cfg), gts("val"))
        conf, f1v = select_conf(sv, tv, nv)
        st, tt, nt = ib.pooled_f1(system(src, seed, "test", cfg), gts("test"))
        res.append((f1v, prf_at(st, tt, nt, conf)["F1"], conf))
    return src, seed, res


def strata(job):
    """测试集上 grouped / alone 目标的召回率（阈值 = 该 seed 验证集选的）。"""
    src, seed, cfg, conf = job
    pt = system(src, seed, "test", cfg)
    gt_all = {}
    for f in ib.FOLDS:
        gt_all.update(gts("test")[f])
    labels = ds.label_gt_boxes(gt_all)[0]
    c = Counter()
    for f in ib.FOLDS:
        for stem, item in gts("test")[f].items():
            W, H = item["width"], item["height"]
            preds = [list(clip_box(p, W, H)) for p in pt[f].get(stem, [])]
            m = ds.match_detail(preds, item["boxes"], conf)[0]
            c["np"] += sum(p[4] >= conf for p in preds)
            for j in range(len(item["boxes"])):
                g = "g" if labels[(stem, j)]["grouped"].startswith("grouped") else "a"
                c[g + "n"] += 1
                c[g + "h"] += m[j] is not None
    return src, seed, str(cfg), {"rg": c["gh"] / c["gn"], "ra": c["ah"] / c["an"], "n_pred": c["np"]}


REFIT_T = (0.2, 0.35, 0.5, 0.7)
REFIT_GATE = (0.3, 0.6)


def family(cfg):
    if cfg is None:
        return "base"
    if len(cfg) > 6 and cfg[6] is not None:
        return "5_refit"
    _, assign, _, dup, _, a = cfg[:6]
    if assign == "hung":
        return "3_hung"
    if dup == "geo":
        return "2_geo"
    if dup == "iou":
        return "r20_iou"
    return "1_decay"


FAMILIES = ("base", "1_decay", "2_geo", "r20_iou", "3_hung", "5_refit", "all")


def run_grid(cf, seeds, jobs):
    """cf: {src: [cfg]}。返回 {src: {seed: [(val F1, test F1, conf)]}}。"""
    R = {s: {} for s in cf}
    with Pool(jobs) as pool:
        for src, seed, res in pool.imap_unordered(eval_seed, [(s, k, cf[s]) for s in cf for k in seeds]):
            R[src][seed] = res
            print(f"  done {src} seed{seed}", flush=True)
    return R


def to_arrays(R, src, cfgs, seeds):
    return [np.array([[R[src][k][j][x] for k in seeds] for j in range(len(cfgs))]) for x in range(3)]


def load_grid(path):
    g = json.load(open(path))
    cfgs = [None if c is None else tuple(tuple(x) if isinstance(x, list) else x for x in c) for c in g["cfgs"]]
    return cfgs, np.array(g["val"]), np.array(g["test"]), np.array(g["conf"]), g["seeds"]


def save_grid(path, cfgs, V, T, C, seeds):
    json.dump({"cfgs": [None if c is None else list(c) for c in cfgs], "val": V.tolist(), "test": T.tolist(), "conf": C.tolist(), "seeds": seeds},
              open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", default=list(SRCS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stage", type=int, default=1, help="1 = 想法 1–3 的网格；2 = 在第 1 阶段每类选中的参数上加想法 5（重拟合），然后汇总")
    ap.add_argument("--smoke", action="store_true", help="只跑 base + 几组参数、1 个 seed，检查 base 是否等于 *_fitbox")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds
    if args.smoke:
        seeds = seeds[:1]
        cf = {s: [None, (12, "greedy", None, "all", None, 0.5), (12, "hung", 0.6, "geo", 0.3, 0.5),
                  (12, "greedy", None, "all", None, 0.5, (0.35, 0.6)), (12, "hung", 0.6, "geo", 0.3, 0.5, (0.5, 0.3))] for s in args.srcs}
        R = run_grid(cf, seeds, args.jobs)
        for src in args.srcs:
            ref = per_seed(f"{src}_fitbox").get(seeds[0], {}).get("F1")
            print(src, "fitbox F1 (run dir)", ref, "| base here", R[src][seeds[0]][0][1])
            for cfg, r in zip(cf[src], R[src][seeds[0]]):
                print("   ", cfg, "val %.3f test %.3f conf %.2f" % r)
        return
    if args.stage == 1:
        cf = {s: grid() for s in args.srcs}
        R = run_grid(cf, seeds, args.jobs)
        for src in args.srcs:
            save_grid(OUT / f"grid_{src}.json", cf[src], *to_arrays(R, src, cf[src], seeds), seeds)
    else:
        cf = {}
        for src in args.srcs:
            cfgs, V, T, C, sd = load_grid(OUT / f"grid_{src}.json")
            assert sd == seeds
            bases = []
            for fam in ("1_decay", "2_geo", "3_hung"):
                idx = [j for j, c in enumerate(cfgs) if family(c) == fam]
                bases.append(cfgs[max(idx, key=lambda j: V[j].mean())])
            cf[src] = [b + ((tl, rg),) for b in dict.fromkeys(bases) for tl in REFIT_T for rg in REFIT_GATE]
        R = run_grid(cf, seeds, args.jobs)
        for src in args.srcs:
            save_grid(OUT / f"grid2_{src}.json", cf[src], *to_arrays(R, src, cf[src], seeds), seeds)
    summarize(args.srcs, seeds, args.jobs)


def summarize(srcs, seeds, jobs):
    chosen = {}
    for src in srcs:
        cfgs, V, T, C, _ = load_grid(OUT / f"grid_{src}.json")
        if (OUT / f"grid2_{src}.json").exists():
            c2, V2, T2, C2, _ = load_grid(OUT / f"grid2_{src}.json")
            cfgs, V, T, C = cfgs + c2, np.vstack([V, V2]), np.vstack([T, T2]), np.vstack([C, C2])
        for fam in FAMILIES:
            idx = [j for j, c in enumerate(cfgs) if fam == "all" or family(c) == fam]
            if not idx:
                continue
            j = max(idx, key=lambda j: V[j].mean())
            chosen[(src, fam)] = (cfgs[j], V[j], T[j], C[j])
    fams = [f for f in FAMILIES if all((s, f) in chosen for s in srcs)]

    # grouped / alone 召回率：每个来源的每个选中参数
    jobs_ = {}
    for (src, fam), (cfg, V, T, C) in chosen.items():
        for n, k in enumerate(seeds):
            jobs_[(src, k, str(cfg))] = (src, k, cfg, float(C[n]))
    S = {}
    with Pool(jobs) as pool:
        for src, seed, cfg_s, r in pool.imap_unordered(strata, list(jobs_.values())):
            S[(src, seed, cfg_s)] = r

    none_f1 = per_seed("hbatbox_flipfix_none")
    lines = ["# 第 22 轮：fit 之后检测和曲线的配对（想法 1–3、5）", "",
             "每个来源、每类做法各自在验证集上选一组参数（10 个 seed 的验证 F1 均值最大）；测试 F1 按 seed 配对。",
             "Δ base = 相对同一来源原来的 fit 流程；Δ none = 相对无 attention 的 baseline（hbatbox_flipfix_none）。",
             "grouped / alone = 测试集上成簇 / 单独目标的召回率，n_pred = 阈值以上的预测数（10 个 seed 平均）。",
             "参数 = (delta, assign, gate, dup, thr, alpha[, (refit t_low, refit gate)])。", "",
             "| 来源 | 做法 | 选中参数 | val F1 | test F1 | Δ base [95% CI] | p | 胜/平/负 | Δ none [95% CI] | grouped | alone | n_pred |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for src in srcs:
        Tb = chosen[(src, "base")][2]
        for fam in fams:
            cfg, V, T, C = chosen[(src, fam)]
            pb = paired(list(Tb), list(T))
            pn = paired([none_f1[k]["F1"] for k in seeds], list(T))
            st = [S[(src, k, str(cfg))] for k in seeds]
            rg, ra, npd = (float(np.mean([s[x] for s in st])) for x in ("rg", "ra", "n_pred"))
            summ[f"{src}|{fam}"] = {"cfg": cfg, "val": float(V.mean()), "test": float(T.mean()), "d_base": pb, "d_none": pn, "rg": rg, "ra": ra, "n_pred": npd}
            lines.append(f"| {src.replace('hbatbox_flipfix_', '')} | {fam} | {cfg} | {V.mean():.3f} | {T.mean():.3f} | "
                         f"{pb['delta_mean']:+.3f} [{pb['ci95'][0]:+.3f}, {pb['ci95'][1]:+.3f}] | {pb['wilcoxon_p']:.3f} | {pb['wins']}/{pb['ties']}/{pb['losses']} | "
                         f"{pn['delta_mean']:+.3f} [{pn['ci95'][0]:+.3f}, {pn['ci95'][1]:+.3f}] | {rg:.3f} | {ra:.3f} | {npd:.0f} |")
    lines += ["", "## band − box-mask（同一类做法，各自在验证集选参数），按 seed 配对", "",
              "| 头 | 做法 | band | box-mask | Δ [95% CI] | p | 胜/平/负 |", "|---|---|---|---|---|---|---|"]
    for head, (a, b) in {"conv": ("hbatbox_flipfix_abs", "hbatbox_flipfix_boxmask"),
                         "unet": ("hbatbox_flipfix_abs_unet", "hbatbox_flipfix_boxmask_unet")}.items():
        if a not in srcs or b not in srcs:
            continue
        for fam in fams:
            Ta, Tb_ = chosen[(a, fam)][2], chosen[(b, fam)][2]
            p = paired(list(Tb_), list(Ta))
            summ[f"{head}|{fam}|band-boxmask"] = p
            lines.append(f"| {head} | {fam} | {Ta.mean():.3f} | {Tb_.mean():.3f} | {p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | "
                         f"{p['wilcoxon_p']:.3f} | {p['wins']}/{p['ties']}/{p['losses']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(summ, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
