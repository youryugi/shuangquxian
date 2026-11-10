"""
多 seed 确认注意力监督的有效性：每个 seed 跑 with_attn / no_attn 消融，汇总 mean±std。
验证 "显式注意力监督提升检测(尤其 precision)" 不是单次运气。
"""
import os
import csv
from datetime import datetime

import numpy as np

import attn_cnn as ac

# 注意：make_split 的默认参数 train_frac/val_frac 在定义时已绑定 0.5/0.25，
# 改 TRAIN_FRAC 无效，必须 patch 显式传入新比例。
_orig_make_split = ac.exp.make_split
ac.exp.make_split = lambda n, s: _orig_make_split(n, s, train_frac=0.70, val_frac=0.15)

SEEDS = [0, 1, 2, 3, 4]


def main():
    now = datetime.now()
    work = os.path.join(os.getcwd(), f"attn_multiseed_{now.strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = ac.AttnDataset(input_size=ac.input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    n_total = len(full)
    print(f"n_total={n_total}  seeds={SEEDS}")

    rows = []
    for seed in SEEDS:
        train_idx, val_idx, test_idx = ac.exp.make_split(n_total, seed)
        print(f"\n=== seed {seed}  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ===")
        for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
            ac.SEED = seed  # 让 train_model 内 set_seed 用当前 seed
            model = ac.train_model(use_attn, full, train_idx, val_idx, ac.num_epochs, work, f"seed{seed}_{tag}")
            m = ac.evaluate(model, full, test_idx)
            print(f"  seed{seed} {tag}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} F1={m['bbox_F1']:.4f} attn_iou={m['attn_band_iou']:.4f}")
            row = {"seed": seed, "config": tag}; row.update(m); rows.append(row)

    # 汇总
    keys = ["bbox_P", "bbox_R", "bbox_F1", "attn_band_iou"]
    print("\n" + "=" * 70)
    print(f"{'config':>12}" + "".join(f"{k:>16}" for k in keys))
    for tag in ["with_attn", "no_attn"]:
        sub = [r for r in rows if r["config"] == tag]
        line = f"{tag:>12}"
        for k in keys:
            vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
            if vals:
                line += f"{np.mean(vals):>8.4f}±{np.std(vals):<7.4f}"
            else:
                line += f"{'nan':>16}"
        print(line)
    print("=" * 70)

    with open(os.path.join(work, "multiseed_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
