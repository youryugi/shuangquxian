import json, sys, numpy as np
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import infer_bandbox as ib
import diag_strata as ds
from evaluate import box_iou, clip_box
from r20_curve_nms import refined_flags

def own_nms(d, ref, rf, nms_thr, delta, tau):
    """Score order. A refined det whose curve (apex within delta px) is already owned by a kept det:
    if its raw box overlaps every owner's raw box by < tau it is a distinct detection whose own band is missing ->
    keep it with its raw (unrefined) box; otherwise it is a duplicate -> drop. Then usual box-IoU NMS, except that
    reverted dets are compared raw-to-raw with tau."""
    order = sorted(range(len(ref)), key=lambda i: -ref[i][4])
    keep = []   # (i, out_box, fit_or_None, reverted)
    for i in order:
        out, f, rev = ref[i], rf[i], False
        if f is not None:
            owners = [k for k in keep if k[2] is not None and abs(k[2]["xv"] - f["xv"]) <= delta and abs(k[2]["yv"] - f["yv"]) <= delta]
            if owners:
                if all(box_iou(d[i], d[k[0]]) < tau for k in owners):
                    out, f, rev = d[i], None, True
                else:
                    continue
        sup = False
        for k in keep:
            if rev or k[3]:
                sup = box_iou(d[i], d[k[0]]) >= tau
            else:
                sup = nms_thr is not None and box_iou(out, k[1]) > nms_thr
            if sup:
                break
        if not sup:
            keep.append((i, out, f, rev))
    return [k[1] for k in keep]

def system(src, seed, fold, phase, cfg):
    t, s_seed, mode, nms_thr = ib.base_fit_cfg(src)
    dets, metas, _ = ib.load_cache(src, fold, seed, phase)
    fits = ib.fits_for(src, fold, seed, phase, t)
    out = {}
    for stem, d in dets.items():
        ref = ib.apply_fits(d, fits[stem], metas[stem]["scale"], s_seed, mode)
        out[stem] = ib.nms(ref, nms_thr) if cfg is None else own_nms(d, ref, refined_flags(d, fits[stem], s_seed), nms_thr, *cfg)
    return out

if __name__ == "__main__":
    src = sys.argv[1]
    gt_val = {f: json.load(open(ib.data_dir(f) / "gt_boxes.json"))["val"] for f in ib.FOLDS}
    gt_test = {f: json.load(open(ib.data_dir(f) / "gt_boxes.json"))["test"] for f in ib.FOLDS}
    gt_all = {}
    for f in ib.FOLDS: gt_all.update(gt_test[f])
    labels = ds.label_gt_boxes(gt_all)[0]
    cfgs = [None] + [(dl, tau) for dl in (4, 8, 12) for tau in (0.2, 0.3, 0.5)]
    print(src, "base", ib.base_fit_cfg(src))
    print("cfg (delta, tau) | val F1 | test F1 | test recall grouped / alone")
    for cfg in cfgs:
        vs, ts, rg, ra = [], [], [], []
        for seed in range(10):
            pv = {f: system(src, seed, f, "val", cfg) for f in ib.FOLDS}
            pt = {f: system(src, seed, f, "test", cfg) for f in ib.FOLDS}
            sv, tv, nv = ib.pooled_f1(pv, gt_val); conf, f1v = ib.select_conf(sv, tv, nv)
            st, tt, nt = ib.pooled_f1(pt, gt_test)
            vs.append(f1v); ts.append(ib.prf_at(st, tt, nt, conf)["F1"])
            c = Counter()
            for f in ib.FOLDS:
                for stem, item in gt_test[f].items():
                    W, H = item["width"], item["height"]
                    m = ds.match_detail([list(clip_box(p, W, H)) for p in pt[f].get(stem, [])], item["boxes"], conf)[0]
                    for j in range(len(item["boxes"])):
                        g = "g" if labels[(stem, j)]["grouped"].startswith("grouped") else "a"
                        c[g + "n"] += 1; c[g + "h"] += m[j] is not None
            rg.append(c["gh"] / c["gn"]); ra.append(c["ah"] / c["an"])
        print(f"{str(cfg):12s} | {np.mean(vs):.3f} | {np.mean(ts):.3f} | {np.mean(rg):.3f} / {np.mean(ra):.3f}", flush=True)
