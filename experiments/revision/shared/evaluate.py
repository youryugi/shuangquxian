"""Unified box-level evaluation for every detector (and for the paper's own method).

Prediction file format (written by every train_*.py):
    preds_<phase>.json = {stem: [[x1, y1, x2, y2, score], ...]}   pixel coords of the ORIGINAL image
GT: <dataset_dir>/gt_boxes.json written by prepare_data.py.

Protocol (same for all models, see README):
  * every prediction is clipped to the image ([0, width] x [0, height] from gt_boxes.json) before matching,
    so only the in-image part of a box counts, the same as the GT boxes (hbat.hyperbola_to_bbox clips them);
    a prediction entirely outside the image keeps zero area and can only be a FP (2026-11-21)
  * greedy one-to-one matching, IoU >= 0.5, predictions sorted by score (same rule as the paper's evaluate())
  * primary number: F1 on TEST at the confidence threshold that maximises F1 on VAL ("val-selected")
  * also reported: AP50 (all-point interpolation), P/R/F1 at fixed thresholds, oracle F1 (best threshold on test)
  * per-category breakdown when gt_boxes.json carries a 'category' per image (ds1: pipe / rebar)

CLI
    python evaluate.py run <run_dir> --data <dataset_dir>            -> writes <run_dir>/metrics.json
    python evaluate.py summary <runs_root> [--out summary.csv]        -> one row per run + table.md
    python evaluate.py compare <runs_root> <exp_a> <exp_b> [--dataset ds2] -> paired stats
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IOU_THR = 0.5
FIXED_CONFS = (0.25, 0.5)
CONF_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ----------------------------------------------------------------------------- core
def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip_box(p, w, h):
    """[x1, y1, x2, y2, *rest] clipped to [0, w] x [0, h]; extra fields (score) are kept."""
    return [min(max(p[0], 0.0), w), min(max(p[1], 0.0), h), min(max(p[2], 0.0), w), min(max(p[3], 0.0), h), *p[4:]]


def clip_preds(pred_dict, gt_dict):
    """{stem: dets} with every det clipped to its image; stems without width/height in gt_dict are left as they are."""
    out = {}
    for stem, dets in pred_dict.items():
        item = gt_dict.get(stem) or {}
        w, h = item.get("width"), item.get("height")
        out[stem] = [clip_box(p, float(w), float(h)) for p in dets] if (w and h) else dets
    return out


def match_image(preds, gts, iou_thr=IOU_THR):
    """Greedy one-to-one matching. preds: [[x1,y1,x2,y2,score]], sorted desc by score inside.
    Returns list of (score, is_tp) for every prediction and the number of GT."""
    order = sorted(preds, key=lambda p: -p[4])
    matched = [False] * len(gts)
    out = []
    for p in order:
        bi, bj = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            iou = box_iou(p, g)
            if iou > bi:
                bi, bj = iou, j
        if bj >= 0 and bi >= iou_thr:
            matched[bj] = True
            out.append((p[4], True))
        else:
            out.append((p[4], False))
    return out, len(gts)


def collect(pred_dict, gt_dict, stems=None, iou_thr=IOU_THR):
    """Flatten matches over images -> (scores, tps, n_gt). Predictions are clipped to the image first."""
    scores, tps, n_gt = [], [], 0
    for stem in (stems if stems is not None else gt_dict.keys()):
        item = gt_dict[stem]
        gts = item["boxes"]
        preds = pred_dict.get(stem, [])
        w, h = item.get("width"), item.get("height")
        if w and h:
            preds = [clip_box(p, float(w), float(h)) for p in preds]
        m, n = match_image(preds, gts, iou_thr)
        n_gt += n
        for s, tp in m:
            scores.append(s)
            tps.append(tp)
    return np.asarray(scores, dtype=float), np.asarray(tps, dtype=bool), n_gt


def prf_at(scores, tps, n_gt, conf):
    keep = scores >= conf
    tp = int(tps[keep].sum())
    fp = int(keep.sum()) - tp
    fn = n_gt - tp
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


def f1_curve(scores, tps, n_gt, grid=CONF_GRID):
    return np.array([prf_at(scores, tps, n_gt, c)["F1"] for c in grid])


def select_conf(scores, tps, n_gt, grid=CONF_GRID):
    """Threshold maximising F1; ties resolved by taking the middle of the best plateau."""
    f1 = f1_curve(scores, tps, n_gt, grid)
    best = f1.max()
    idx = np.where(np.isclose(f1, best, atol=1e-9))[0]
    return float(grid[idx[len(idx) // 2]]), float(best)


def ap50(scores, tps, n_gt):
    """VOC-style all-point-interpolated AP at the matching IoU."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tps[order].astype(float)
    fp = 1.0 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_split(pred_dict, gt_dict, conf, stems=None):
    s, t, n = collect(pred_dict, gt_dict, stems)
    out = {"n_images": len(stems) if stems is not None else len(gt_dict), "n_gt": int(n), "n_pred_raw": int(len(s))}
    out["AP50"] = ap50(s, t, n)
    out["at_selected"] = {"conf": conf, **prf_at(s, t, n, conf)}
    out["at_fixed"] = {str(c): prf_at(s, t, n, c) for c in FIXED_CONFS}
    oc, of = select_conf(s, t, n)
    out["oracle"] = {"conf": oc, "F1": of}
    return out


def evaluate_run(run_dir, data_dir, write=True, gt_name="gt_boxes.json", out_name="metrics.json"):
    """gt_name / out_name let the same predictions be scored against an alternative GT file
    (e.g. gt_boxes_official.json -> metrics_official.json) without retraining."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    gt = json.load(open(data_dir / gt_name, encoding="utf-8"))
    res = {"run_dir": str(run_dir), "data_dir": str(data_dir), "iou_thr": IOU_THR, "gt_file": gt_name}
    pv = run_dir / "preds_val.json"
    pt = run_dir / "preds_test.json"
    if not pt.exists():
        raise FileNotFoundError(pt)
    preds_test = json.load(open(pt, encoding="utf-8"))
    if pv.exists():
        preds_val = json.load(open(pv, encoding="utf-8"))
        s, t, n = collect(preds_val, gt["val"])
        conf, f1v = select_conf(s, t, n)
        res["val"] = {"selected_conf": conf, "F1_at_selected": f1v, "AP50": ap50(s, t, n), "n_gt": int(n)}
    else:
        conf = 0.25
        res["val"] = {"selected_conf": conf, "note": "no preds_val.json; fixed 0.25 used"}
    res["test"] = evaluate_split(preds_test, gt["test"], conf)
    # per-category / per-stratum breakdown (image-level attributes)
    for attr in ("category", "stratum"):
        vals = sorted({v.get(attr) for v in gt["test"].values() if v.get(attr) is not None})
        if len(vals) > 1:
            res[f"test_by_{attr}"] = {}
            for val in vals:
                stems = [k for k, v in gt["test"].items() if v.get(attr) == val]
                res[f"test_by_{attr}"][val] = evaluate_split(preds_test, gt["test"], conf, stems)
    info = run_dir / "run_info.json"
    if info.exists():
        res["run_info"] = json.load(open(info, encoding="utf-8"))
    if write:
        with open(run_dir / out_name, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return res


# ----------------------------------------------------------------------------- aggregation
def _flat_row(m):
    ri = m.get("run_info", {})
    t = m["test"]
    row = {
        "dataset": ri.get("dataset"), "exp": ri.get("exp"), "seed": ri.get("seed"), "model": ri.get("model"),
        "pretrained": ri.get("pretrained"), "params_M": ri.get("params_M_inference", ri.get("params_M")),
        "GFLOPs": ri.get("GFLOPs"),
        "epochs": ri.get("epochs"), "train_minutes": ri.get("train_minutes"),
        "val_conf": m["val"].get("selected_conf"), "val_F1": m["val"].get("F1_at_selected"),
        "test_F1": t["at_selected"]["F1"], "test_P": t["at_selected"]["P"], "test_R": t["at_selected"]["R"],
        "test_AP50": t["AP50"], "test_F1_oracle": t["oracle"]["F1"],
        "test_F1_c0.25": t["at_fixed"]["0.25"]["F1"], "test_F1_c0.5": t["at_fixed"]["0.5"]["F1"],
        "n_gt": t["n_gt"], "run_dir": m["run_dir"],
    }
    for attr in ("category", "stratum"):
        for k, v in m.get(f"test_by_{attr}", {}).items():
            row[f"F1_{attr}_{k}"] = v["at_selected"]["F1"]
    return row


def find_metrics(runs_root):
    return sorted(Path(runs_root).rglob("metrics.json"))


def summary(runs_root, out_csv=None):
    rows = [_flat_row(json.load(open(p, encoding="utf-8"))) for p in find_metrics(runs_root)]
    if not rows:
        print("no metrics.json under", runs_root)
        return rows
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    out_csv = Path(out_csv) if out_csv else Path(runs_root) / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # grouped table
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["exp"])].append(r)
    lines = ["| dataset | exp | n | F1 (val-sel) mean±std | median | AP50 mean±std | F1 oracle | params M | GFLOPs |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (ds, exp), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        f1 = np.array([r["test_F1"] for r in rs]); ap = np.array([r["test_AP50"] for r in rs])
        fo = np.array([r["test_F1_oracle"] for r in rs])
        pm = rs[0].get("params_M"); gf = rs[0].get("GFLOPs")
        lines.append(f"| {ds} | {exp} | {len(rs)} | {f1.mean():.3f} ± {f1.std(ddof=1) if len(rs) > 1 else 0:.3f} | "
                     f"{np.median(f1):.3f} | {ap.mean():.3f} ± {ap.std(ddof=1) if len(rs) > 1 else 0:.3f} | "
                     f"{fo.mean():.3f} | {pm if pm is None else f'{pm:.2f}'} | {gf if gf is None else f'{gf:.1f}'} |")
    table = "\n".join(lines)
    (out_csv.with_name("table.md")).write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n-> {out_csv}\n-> {out_csv.with_name('table.md')}")
    return rows


def compare(runs_root, exp_a, exp_b, dataset=None, metric="test_F1", n_boot=10000, seed=0):
    """Paired comparison by seed: Δ = B − A. Mean, bootstrap 95% CI, Wilcoxon signed-rank, wins/ties/losses."""
    rows = [_flat_row(json.load(open(p, encoding="utf-8"))) for p in find_metrics(runs_root)]
    by = defaultdict(dict)
    for r in rows:
        if dataset and r["dataset"] != dataset:
            continue
        by[r["exp"]][r["seed"]] = r[metric]
    seeds = sorted(set(by.get(exp_a, {})) & set(by.get(exp_b, {})))
    if not seeds:
        print(f"no common seeds between {exp_a} and {exp_b}")
        return None
    a = np.array([by[exp_a][s] for s in seeds]); b = np.array([by[exp_b][s] for s in seeds])
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(b, a, zero_method="zsplit", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    except Exception as e:  # scipy missing or n too small
        p = float("nan"); print("wilcoxon unavailable:", e)
    res = {"dataset": dataset, "A": exp_a, "B": exp_b, "metric": metric, "n": len(seeds), "seeds": seeds,
           "mean_A": float(a.mean()), "mean_B": float(b.mean()), "delta_mean": float(d.mean()),
           "delta_median": float(np.median(d)), "ci95": ci, "wilcoxon_p": p,
           "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
           "per_seed": {str(s): {"A": float(x), "B": float(y)} for s, x, y in zip(seeds, a, b)}}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


# ----------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("run_dir"); r.add_argument("--data", required=True)
    s = sub.add_parser("summary"); s.add_argument("runs_root"); s.add_argument("--out", default=None)
    c = sub.add_parser("compare"); c.add_argument("runs_root"); c.add_argument("exp_a"); c.add_argument("exp_b")
    c.add_argument("--dataset", default=None); c.add_argument("--metric", default="test_F1")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        m = evaluate_run(args.run_dir, args.data)
        t = m["test"]["at_selected"]
        print(f"val conf {m['val'].get('selected_conf')}: test F1 {t['F1']:.4f} P {t['P']:.4f} R {t['R']:.4f} AP50 {m['test']['AP50']:.4f}")
    elif args.cmd == "summary":
        summary(args.runs_root, args.out)
    elif args.cmd == "compare":
        compare(args.runs_root, args.exp_a, args.exp_b, args.dataset, args.metric)


if __name__ == "__main__":
    main()
