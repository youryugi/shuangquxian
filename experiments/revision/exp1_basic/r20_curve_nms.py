import json, sys, itertools, numpy as np
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import infer_bandbox as ib
import diag_strata as ds
from evaluate import box_iou, clip_box

def refined_flags(dets, fits_img, s_seed):
    out = []
    for i, d in enumerate(dets):
        f = fits_img.get(str(i), fits_img.get(i))
        ok = f is not None and d[4] >= s_seed and f["box"][2] - f["box"][0] >= 2 and f["box"][3] - f["box"][1] >= 2
        out.append(f if ok else None)
    return out

def curve_nms(ref, rf, thr_box, delta, cap):
    """ref: refined dets; rf[i]: fit or None. Same-curve pairs (apex within delta px) are suppressed;
    different-curve pairs are kept unless IoU > cap; pairs involving an unrefined det use box-IoU NMS at thr_box."""
    order = sorted(range(len(ref)), key=lambda i: -ref[i][4])
    keep = []
    for i in order:
        sup = False
        for k in keep:
            iou = box_iou(ref[i], ref[k])
            if rf[i] is not None and rf[k] is not None:
                same = abs(rf[i]["xv"] - rf[k]["xv"]) <= delta and abs(rf[i]["yv"] - rf[k]["yv"]) <= delta
                sup = same or (cap is not None and iou > cap)
            else:
                sup = thr_box is not None and iou > thr_box
            if sup:
                break
        if not sup:
            keep.append(i)
    return [ref[i] for i in keep]

def system(src, seed, fold, phase, cfg):
    t, s_seed, mode, nms_thr = ib.base_fit_cfg(src)
    dets, metas, _ = ib.load_cache(src, fold, seed, phase)
    fits = ib.fits_for(src, fold, seed, phase, t)
    out = {}
    for stem, d in dets.items():
        ref = ib.apply_fits(d, fits[stem], metas[stem]["scale"], s_seed, mode)
        if cfg is None:
            out[stem] = ib.nms(ref, nms_thr)
        else:
            out[stem] = curve_nms(ref, refined_flags(d, fits[stem], s_seed), nms_thr, *cfg)
    return out

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "hbatbox_flipfix_abs"
    gt_val = {f: json.load(open(ib.data_dir(f) / "gt_boxes.json"))["val"] for f in ib.FOLDS}
    gt_test = {f: json.load(open(ib.data_dir(f) / "gt_boxes.json"))["test"] for f in ib.FOLDS}
    gt_all = {}
    for f in ib.FOLDS: gt_all.update(gt_test[f])
    labels = ds.label_gt_boxes(gt_all)[0]
    cfgs = [None] + [(dl, cap) for dl in (4, 8, 12, 16) for cap in (None, 0.7, 0.85)]
    print(src, "base", ib.base_fit_cfg(src))
    print("cfg (delta, cap) | val F1 | test F1 | test recall grouped / alone")
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
