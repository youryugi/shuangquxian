import json, sys
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import infer_bandbox as ib
import diag_strata as ds
from evaluate import box_iou, clip_box
RUNS = ib.HERE / "runs_cv"
src = sys.argv[1] if len(sys.argv) > 1 else "hbatbox_flipfix_abs"
t, s_seed, mode, nms_thr = ib.base_fit_cfg(src)
gt_all = {}
for f in ib.FOLDS: gt_all.update(json.load(open(ib.data_dir(f) / "gt_boxes.json"))["test"])
labels = ds.label_gt_boxes(gt_all)[0]
C = Counter()
for seed in range(10):
    conf_raw = ds.pooled_val_conf(ds.load_seed(src, seed, RUNS)[0])
    conf_fit = ds.pooled_val_conf(ds.load_seed(src + "_fitbox", seed, RUNS)[0])
    for f in ib.FOLDS:
        dets, metas, _ = ib.load_cache(src, f, seed, "test")
        fits = ib.fits_for(src, f, seed, "test", t)
        gtf = json.load(open(ib.data_dir(f) / "gt_boxes.json"))["test"]
        for stem, item in gtf.items():
            W, H = item["width"], item["height"]
            d = [list(clip_box(p, W, H)) for p in dets[stem]]
            ref = [list(clip_box(p, W, H)) for p in ib.apply_fits(dets[stem], fits[stem], metas[stem]["scale"], s_seed, mode)]
            post = ib.nms(ref, nms_thr)
            m_raw = ds.match_detail(d, item["boxes"], conf_raw)[0]
            m_fit = ds.match_detail(post, item["boxes"], conf_fit)[0]
            fi = fits[stem]
            for j, g in enumerate(item["boxes"]):
                grp = "grouped" if labels[(stem, j)]["grouped"].startswith("grouped") else labels[(stem, j)]["grouped"]
                r_ok, f_ok = m_raw[j] is not None, m_fit[j] is not None
                C[(grp, "raw", r_ok, "fit", f_ok)] += 1
                if r_ok and not f_ok:
                    p = m_raw[j][0]; i = next(k for k, q in enumerate(d) if q == list(p))
                    fit_i = fi.get(str(i)); rb = ref[i]
                    if fit_i is None or d[i][4] < s_seed:
                        cause = "not refined"
                    elif rb not in post:
                        cause = "suppressed by NMS"
                    elif rb[4] < conf_fit:
                        cause = "below fit-system conf"
                    else:
                        same = [k for k, v in fi.items() if v is not None and int(k) != i
                                and abs(v["xv"] - fit_i["xv"]) < 6 and abs(v["yv"] - fit_i["yv"]) < 6]
                        cause = "refined box off, shares curve with another det" if same else "refined box off, own curve"
                    C[(grp, "lost", cause)] += 1
for grp in ("grouped", "alone"):
    tot = sum(v for k, v in C.items() if k[0] == grp and k[1] == "raw")
    print(f"== {grp}: {tot/10:.1f} GT per seed")
    for r in (True, False):
        for fo in (True, False):
            print(f"   raw hit={r!s:5} fit hit={fo!s:5}: {C[(grp,'raw',r,'fit',fo)]/10:.1f}")
    for k, v in sorted(C.items(), key=lambda kv: -kv[1]):
        if k[0] == grp and k[1] == "lost":
            print(f"   lost because {k[2]}: {v/10:.1f}")
