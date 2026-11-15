"""C 类漏检（附近有框但 IoU 0.1-0.5）诊断：中心偏移 vs 宽高误差、真值宽高/真值中心替换后的可救回比例、相邻目标干扰。
用法：python diag_box_size.py（读 runs_cv/ds2/hbatbox_aug_{none,abs}）"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector_comparison"))
from evaluate import collect, select_conf, box_iou
RUNS = ROOT / "basic_ds2" / "runs_cv" / "ds2"; DS = ROOT / "detector_comparison" / "datasets"
def greedy(preds, gts, thr=0.5):
    preds = sorted(preds, key=lambda p: -p[4]); used = [False]*len(gts)
    for p in preds:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if not used[j]:
                i = box_iou(p[:4], g)
                if i >= best: best, bj = i, j
        if bj >= 0: used[bj] = True
    return used
for exp in ["hbatbox_aug_none", "hbatbox_aug_abs"]:
    R = []
    for seed in range(10):
        sv, tv, nv, pf = [], [], 0, {}
        for f in range(5):
            rd = RUNS/exp/f"f{f}_seed{seed}"; gt = json.load(open(DS/f"ds2_f{f}_s100_hbatbox"/"gt_boxes.json", encoding="utf-8"))
            s,t,n = collect(json.load(open(rd/"preds_val.json")), gt["val"]); sv.append(s); tv.append(t); nv += n
            pf[f] = (json.load(open(rd/"preds_test.json")), gt["test"])
        conf,_ = select_conf(np.concatenate(sv), np.concatenate(tv), nv)
        for f,(pt,gtt) in pf.items():
            for stem,g in gtt.items():
                gts = g["boxes"]; kept = [p for p in pt.get(stem,[]) if p[4] >= conf]
                m = greedy(kept, gts)
                for j,gb in enumerate(gts):
                    if m[j] or not kept: continue
                    ious = [box_iou(p[:4], gb) for p in kept]; k = int(np.argmax(ious))
                    if not (0.1 <= ious[k] < 0.5): continue
                    p = kept[k]
                    gw, gh = gb[2]-gb[0], gb[3]-gb[1]; pw, ph = p[2]-p[0], p[3]-p[1]
                    gcx, gcy = (gb[0]+gb[2])/2, (gb[1]+gb[3])/2; pcx, pcy = (p[0]+p[2])/2, (p[1]+p[3])/2
                    # oracle: 预测中心 + GT 宽高 ; GT 中心 + 预测宽高
                    o_wh = box_iou([pcx-gw/2, pcy-gh/2, pcx+gw/2, pcy+gh/2], gb)
                    o_c = box_iou([gcx-pw/2, gcy-ph/2, gcx+pw/2, gcy+ph/2], gb)
                    # 这个预测是不是其实落在另一个 GT 上（相邻双曲线）
                    other = max([box_iou(p[:4], o) for jj,o in enumerate(gts) if jj!=j], default=0)
                    # 预测框是否包含 GT 或被 GT 包含 / 多条合并
                    R.append(((pcx-gcx)/gw, (pcy-gcy)/gh, pw/gw, ph/gh, o_wh, o_c, other, len(gts), ious[k]))
    R = np.array(R)
    print(f"== {exp}  C 类 {len(R)}")
    print(f"  中心偏移/框宽 |dx| 中位 {np.median(abs(R[:,0])):.2f}, |dy|/框高 中位 {np.median(abs(R[:,1])):.2f}; dy 符号: 预测偏下 {np.mean(R[:,1]>0):.0%}")
    print(f"  宽比 pred/gt 中位 {np.median(R[:,2]):.2f} (p10 {np.percentile(R[:,2],10):.2f}, p90 {np.percentile(R[:,2],90):.2f}); 偏小(<0.8) {np.mean(R[:,2]<0.8):.0%} 偏大(>1.25) {np.mean(R[:,2]>1.25):.0%}")
    print(f"  高比 pred/gt 中位 {np.median(R[:,3]):.2f} (p10 {np.percentile(R[:,3],10):.2f}, p90 {np.percentile(R[:,3],90):.2f}); 偏小 {np.mean(R[:,3]<0.8):.0%} 偏大 {np.mean(R[:,3]>1.25):.0%}")
    print(f"  换成 GT 宽高(保留预测中心)后 IoU>=0.5: {np.mean(R[:,4]>=0.5):.0%};  换成 GT 中心(保留预测宽高)后: {np.mean(R[:,5]>=0.5):.0%}")
    print(f"  该预测其实与另一个 GT IoU>=0.3（框到了相邻/合并的双曲线）: {np.mean(R[:,6]>=0.3):.0%}; 所在图框数>=2: {np.mean(R[:,7]>=2):.0%}")
