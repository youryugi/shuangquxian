"""
超参数搜索：LR × LAM_ATT，num_epochs 通过 val 早停自动挑选。

数据划分（核心问题的答案）：
  train  —— 只用来更新权重（普通训练）
  val    —— 不参与训练；每隔 EVAL_EVERY 轮在其上跑一次 bbox_F1，
            既用来在同一次训练里挑"最优 epoch"（相当于免费顺带搜了 num_epochs），
            也用来跨 (lr, lam_att) 组合比较、挑最优超参组合
  test   —— 全程不碰，只在最后用选出的最优组合 + 对应 checkpoint 跑一次，
            得到无偏的最终指标（决不能用 test 来挑超参，否则等于用答案调参）

复用 attn_cnn_merged_final.py 里的数据集 / 模型 / loss / evaluate 等实现，
避免整份复制一遍容易出现两边逻辑不一致的问题。
"""
import os
import csv
import time
import shutil
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import attn_cnn_merged_final as base

# ══════════════════════════════════════════════════════════════════════════════
# 搜索配置（可调）
# ══════════════════════════════════════════════════════════════════════════════
SEEDS       = [10, 11, 12, 13, 14,15,16,17,18,19]        # 划分种子：每个种子独立 70/15/15 划分，结果跨 seed 取平均以降低小数据集的方差
                                   # 想要更稳的结论就加 seed，但训练次数 = len(SEEDS) * (len(LR_GRID)*len(LAM_ATT_GRID)+1)，随之线性增加
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15                 # 其余 15% 为 test（全程不参与训练/选参）

LR_GRID      = [2e-4, 5e-4, 1e-3]  # 以脚本里原来的 5e-4 为中心扫
LAM_ATT_GRID = [0.1,0.3,0.5,0.7, 1.0,3,5,7,10]          # 沿用 attn_cnn_merged_final.py 里的 LAM_ATT
ATTN_MODE    = "abs"                # 固定用 abs（当前启用的注意力监督方式），只搜 lr / lam_att

MAX_EPOCHS  = 150                   # 训练上限（沿用原脚本的 num_epochs）
EVAL_EVERY  = 5                    # 每隔几轮在 val 上评估一次、检查是否刷新最优 checkpoint
SELECT_KEY  = "bbox_F1"            # 挑 best epoch / best 超参组合用的指标

INCLUDE_NONE_BASELINE = True        # 顺带跑一个不用注意力监督的 baseline（lr 固定 5e-4）方便对比


# ══════════════════════════════════════════════════════════════════════════════
def train_search_run(attn_mode, lr, lam, full, tr_idx, va_idx, seed,
                      max_epochs, eval_every, fuse, augment):
    """训练一次 (attn_mode, lr, lam) 组合，每 eval_every 轮在 val 上评估一次，
    返回 val 上 SELECT_KEY 最高的那个 epoch 的 checkpoint + 指标（早停式选 epoch）。"""
    base.set_seed(seed)
    use_attn = (attn_mode != "none")
    if lam is not None:
        base.LAM_ATT_CUR = lam

    tr_set = Subset(full, tr_idx)
    if augment:
        tr_set = base.AugWrapper(tr_set)
    _pin = (base.device.type == "cuda")
    tl = DataLoader(tr_set, batch_size=base.batch_size, shuffle=True,
                     num_workers=base.NUM_WORKERS, pin_memory=_pin,
                     persistent_workers=(base.NUM_WORKERS > 0),
                     worker_init_fn=base._worker_init, collate_fn=base.collate)

    model = base.AttnBBoxNet(in_ch=1, base_ch=base.BASE_CH, use_attn=use_attn, fuse=fuse).to(base.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"score": -1.0, "epoch": 0, "metrics": None, "state": None}
    for ep in range(1, max_epochs + 1):
        model.train()
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(base.device, non_blocking=_pin) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=base.device.type, dtype=torch.bfloat16, enabled=(base.device.type == "cuda")):
                loss = base.compute_loss(model, img, hm, wh, off, pk, band, attn_mode)
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % eval_every == 0 or ep == max_epochs:
            model.eval()
            m = base.evaluate(model, full, va_idx)
            if m[SELECT_KEY] >= best["score"]:
                best["score"] = m[SELECT_KEY]; best["epoch"] = ep; best["metrics"] = m
                best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--lr_grid", nargs="+", type=float, default=LR_GRID)
    parser.add_argument("--lam_grid", nargs="+", type=float, default=LAM_ATT_GRID)
    parser.add_argument("--attn_mode", default=ATTN_MODE, choices=["abs", "soft"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fuse", default=base.FUSE, choices=["gate", "concat"])
    parser.add_argument("--no_none_baseline", action="store_true", default=not INCLUDE_NONE_BASELINE)
    args = parser.parse_args()

    work = os.path.join(os.getcwd(), f"attn_cnn_hparam_search_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(work, os.path.basename(__file__)))

    print("Using device:", base.device)
    full = base.AttnDataset(input_size=base.input_size, hm_stride=base.HM_STRIDE, sigma=base.HM_SIGMA)
    n = len(full)
    test_frac = round(1.0 - args.train_frac - args.val_frac, 4)
    print(f"[hparam_search] n_total={n} seeds={args.seeds} lr_grid={args.lr_grid} lam_grid={args.lam_grid} "
          f"attn_mode={args.attn_mode} epochs={args.epochs} eval_every={args.eval_every} "
          f"split={args.train_frac}/{args.val_frac}/{test_frac}", flush=True)

    splits = {}
    search_rows = []
    none_rows = []

    for seed in args.seeds:
        tr, va, te = base.make_split(n, seed, train_frac=args.train_frac, val_frac=args.val_frac)
        splits[seed] = (tr, va, te)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)

        if not args.no_none_baseline:
            t0 = time.perf_counter()
            best = train_search_run("none", 5e-4, None, full, tr, va, seed,
                                     args.epochs, args.eval_every, args.fuse, base.AUGMENT)
            m = best["metrics"]
            print(f"  [seed{seed}] none baseline: best_epoch={best['epoch']}/{args.epochs} "
                  f"val_F1={m['bbox_F1']:.4f}  [{time.perf_counter() - t0:.1f}s]", flush=True)
            torch.save(best["state"], os.path.join(work, f"seed{seed}_none_bestval.pth"))
            none_rows.append({"seed": seed, "best_epoch": best["epoch"], "val_F1": m["bbox_F1"]})

        for lr in args.lr_grid:
            for lam in args.lam_grid:
                t0 = time.perf_counter()
                tag = f"seed{seed}_lr{lr}_lam{lam}"
                best = train_search_run(args.attn_mode, lr, lam, full, tr, va, seed,
                                         args.epochs, args.eval_every, args.fuse, base.AUGMENT)
                m = best["metrics"]
                print(f"  [seed{seed}] lr={lr} lam={lam}: best_epoch={best['epoch']}/{args.epochs} "
                      f"val_P={m['bbox_P']:.4f} val_R={m['bbox_R']:.4f} val_F1={m['bbox_F1']:.4f} "
                      f"attn_iou={m['attn_band_iou']:.4f}  [{time.perf_counter() - t0:.1f}s]", flush=True)
                torch.save(best["state"], os.path.join(work, f"{tag}_bestval.pth"))
                search_rows.append({"seed": seed, "lr": lr, "lam_att": lam, "best_epoch": best["epoch"],
                                     "val_P": m["bbox_P"], "val_R": m["bbox_R"], "val_F1": m["bbox_F1"],
                                     "val_attn_iou": m["attn_band_iou"]})

    # ── 汇总：按 (lr, lam) 求跨 seed 的平均 val_F1，挑最优组合 ──
    combos = sorted({(r["lr"], r["lam_att"]) for r in search_rows})
    agg = []
    for lr, lam in combos:
        sub = [r for r in search_rows if r["lr"] == lr and r["lam_att"] == lam]
        agg.append({"lr": lr, "lam_att": lam,
                     "val_F1_mean": float(np.mean([r["val_F1"] for r in sub])),
                     "val_F1_std": float(np.std([r["val_F1"] for r in sub])),
                     "best_epoch_mean": float(np.mean([r["best_epoch"] for r in sub]))})
    agg.sort(key=lambda a: -a["val_F1_mean"])

    print("\n" + "=" * 70)
    if none_rows:
        f1s = [r["val_F1"] for r in none_rows]
        print(f"[baseline] none（不加注意力监督）: val_F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"{'lr':>10}{'lam_att':>10}{'val_F1_mean':>14}{'val_F1_std':>12}{'best_epoch~':>13}")
    for a in agg:
        print(f"{a['lr']:>10}{a['lam_att']:>10}{a['val_F1_mean']:>14.4f}{a['val_F1_std']:>12.4f}{a['best_epoch_mean']:>13.1f}")
    print("=" * 70)

    best_cfg = agg[0]
    print(f"\n>>> 最优组合（按 val_F1 均值选出）: lr={best_cfg['lr']}  lam_att={best_cfg['lam_att']}  "
          f"平均 best_epoch≈{best_cfg['best_epoch_mean']:.1f}", flush=True)

    # ── 用选出的最优组合，在从未参与选参的 test 集上跑一次，给出无偏最终指标 ──
    test_rows = []
    for seed in args.seeds:
        tr, va, te = splits[seed]
        tag = f"seed{seed}_lr{best_cfg['lr']}_lam{best_cfg['lam_att']}"
        ckpt_path = os.path.join(work, f"{tag}_bestval.pth")
        model = base.AttnBBoxNet(in_ch=1, base_ch=base.BASE_CH, use_attn=True, fuse=args.fuse).to(base.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=base.device))
        model.eval()
        m = base.evaluate(model, full, te)
        gap = base.attn_gap(model, full, te)
        print(f"  [FINAL TEST] seed{seed}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} F1={m['bbox_F1']:.4f} "
              f"attn_iou={m['attn_band_iou']:.4f} gap={gap:.4f}", flush=True)
        test_rows.append({"seed": seed, **m, "attn_gap": gap})

    keys = ["bbox_P", "bbox_R", "bbox_F1", "attn_band_iou", "attn_gap"]
    print("\n" + "=" * 70)
    print(f"最终无偏 TEST 指标（lr={best_cfg['lr']}, lam_att={best_cfg['lam_att']}，"
          f"跨 {len(args.seeds)} 个 seed 划分取平均）：")
    for k in keys:
        vals = [r[k] for r in test_rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            print(f"  {k:>14} = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 70)

    with open(os.path.join(work, "search_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(search_rows[0].keys())); w.writeheader(); w.writerows(search_rows)
    with open(os.path.join(work, "final_test_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys())); w.writeheader(); w.writerows(test_rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
