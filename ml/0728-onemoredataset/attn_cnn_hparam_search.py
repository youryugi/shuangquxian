"""
超参数搜索：对 none 和 abs 两种模式分别独立搜索，而不是只调 abs、none 用固定 lr。
  none —— 没有 lam_att，只搜 LR_GRID
  abs  —— 搜 LR_GRID × LAM_ATT_GRID
num_epochs 通过 val 早停自动挑选，不用单独网格搜。

解码峰值阈值 HM_THRESH（decode 时热力图峰值的阈值）单独在训练之后搜：
它只影响推理/解码，不影响训练本身，所以不用重新训练——直接复用每个 (mode, seed)
已经存好的 best-val checkpoint，在 val 上换不同阈值跑 evaluate 就行，很便宜。
注意这是"训练完再搜阈值"，best_epoch 的选择本身仍然是在默认 HM_THRESH 下挑的，
不是阈值和 epoch 联合搜索——如果想要更严谨的联合最优，要在训练循环里也扫阈值，
但那样每次 val 评估的开销会成倍增加，目前先用这个更便宜的两阶段做法。

数据划分（核心问题的答案）：
  train  —— 只用来更新权重（普通训练）
  val    —— 不参与训练；每隔 EVAL_EVERY 轮在其上跑一次 bbox_F1，
            既用来在同一次训练里挑"最优 epoch"（相当于免费顺带搜了 num_epochs），
            也用来在各自模式内部跨超参组合比较、挑最优组合
  test   —— 全程不碰，只在最后用每个模式各自选出的最优组合 + 对应 checkpoint 跑一次，
            得到无偏的最终指标（决不能用 test 来挑超参，否则等于用答案调参）

之所以两种模式分开搜：如果 abs 认真调了 lr、none 却用脚本里原来写死的 5e-4，
最后 none vs abs 的对比就不公平——可能只是 abs 调得更细，而不是注意力监督本身更好。
分开搜完，各自用各自的最优超参，才是"两种方法都调到最好"之后的公平对比。

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
MODES       = ["none", "abs"]      # 分别独立搜索的模式；none 只搜 lr，abs 搜 lr × lam_att
SEEDS       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 划分种子：每个种子独立 70/15/15 划分，结果跨 seed 取平均以降低小数据集的方差
                                    # 想要更稳的结论就加 seed，但训练次数随之线性增加
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15                  # 其余 15% 为 test（全程不参与训练/选参）

LR_GRID      = [2e-4, 5e-4, 1e-3]   # none 和 abs 各自都在这个网格上搜 lr
LAM_ATT_GRID = [0.1, 0.3, 0.5, 0.7, 1.0, 3, 5, 7, 10]   # 只用于 abs（none 没有注意力 loss，用不上）

MAX_EPOCHS  = 150                   # 训练上限（沿用原脚本的 num_epochs）
EVAL_EVERY  = 5                     # 每隔几轮在 val 上评估一次、检查是否刷新最优 checkpoint
SELECT_KEY  = "bbox_F1"             # 挑 best epoch / best 超参组合用的指标

HM_THRESH_GRID = [round(0.05 + 0.025 * i, 4) for i in range(31)]   # 0.05~0.80，步长 0.025：更全更细的阈值网格（训练完后在 val 上搜，不用重训）


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


def grid_for_mode(mode, lr_grid, lam_grid):
    """none 没有 lam_att：网格只是 lr；abs/soft：网格是 lr × lam_att。"""
    if mode == "none":
        return [(lr, None) for lr in lr_grid]
    return [(lr, lam) for lr in lr_grid for lam in lam_grid]


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=MODES, choices=["none", "abs", "soft"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--lr_grid", nargs="+", type=float, default=LR_GRID)
    parser.add_argument("--lam_grid", nargs="+", type=float, default=LAM_ATT_GRID)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fuse", default=base.FUSE, choices=["gate", "concat"])
    parser.add_argument("--hm_thresh_grid", nargs="+", type=float, default=HM_THRESH_GRID)
    args = parser.parse_args()

    work = os.path.join(os.getcwd(), f"attn_cnn_hparam_search_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(work, os.path.basename(__file__)))

    print("Using device:", base.device)
    full = base.AttnDataset(input_size=base.input_size, hm_stride=base.HM_STRIDE, sigma=base.HM_SIGMA)
    n = len(full)
    test_frac = round(1.0 - args.train_frac - args.val_frac, 4)
    print(f"[hparam_search] n_total={n} modes={args.modes} seeds={args.seeds} lr_grid={args.lr_grid} "
          f"lam_grid={args.lam_grid} epochs={args.epochs} eval_every={args.eval_every} "
          f"split={args.train_frac}/{args.val_frac}/{test_frac}", flush=True)

    splits = {}
    search_rows = []

    for seed in args.seeds:
        tr, va, te = base.make_split(n, seed, train_frac=args.train_frac, val_frac=args.val_frac)
        splits[seed] = (tr, va, te)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)

        for mode in args.modes:
            for lr, lam in grid_for_mode(mode, args.lr_grid, args.lam_grid):
                t0 = time.perf_counter()
                lam_tag = "-" if lam is None else lam
                tag = f"seed{seed}_{mode}_lr{lr}_lam{lam_tag}"
                best = train_search_run(mode, lr, lam, full, tr, va, seed,
                                         args.epochs, args.eval_every, args.fuse, base.AUGMENT)
                m = best["metrics"]
                print(f"  [seed{seed}] {mode:>4} lr={lr} lam={lam_tag}: best_epoch={best['epoch']}/{args.epochs} "
                      f"val_P={m['bbox_P']:.4f} val_R={m['bbox_R']:.4f} val_F1={m['bbox_F1']:.4f} "
                      f"attn_iou={m['attn_band_iou']:.4f}  [{time.perf_counter() - t0:.1f}s]", flush=True)
                torch.save(best["state"], os.path.join(work, f"{tag}_bestval.pth"))
                search_rows.append({"seed": seed, "mode": mode, "lr": lr, "lam_att": lam_tag,
                                     "best_epoch": best["epoch"], "val_P": m["bbox_P"], "val_R": m["bbox_R"],
                                     "val_F1": m["bbox_F1"], "val_attn_iou": m["attn_band_iou"]})

    # ── 汇总：每个 mode 内部按 (lr, lam) 求跨 seed 的平均 val_F1，各自挑最优组合 ──
    best_cfg_by_mode = {}
    for mode in args.modes:
        mode_rows = [r for r in search_rows if r["mode"] == mode]
        combos = sorted({(r["lr"], r["lam_att"]) for r in mode_rows})
        agg = []
        for lr, lam in combos:
            sub = [r for r in mode_rows if r["lr"] == lr and r["lam_att"] == lam]
            agg.append({"lr": lr, "lam_att": lam,
                        "val_F1_mean": float(np.mean([r["val_F1"] for r in sub])),
                        "val_F1_std": float(np.std([r["val_F1"] for r in sub])),
                        "best_epoch_mean": float(np.mean([r["best_epoch"] for r in sub]))})
        agg.sort(key=lambda a: -a["val_F1_mean"])

        print("\n" + "=" * 70)
        print(f"模式 {mode} 的搜索结果（按 val_F1 均值降序）：")
        print(f"{'lr':>10}{'lam_att':>10}{'val_F1_mean':>14}{'val_F1_std':>12}{'best_epoch~':>13}")
        for a in agg:
            print(f"{a['lr']:>10}{str(a['lam_att']):>10}{a['val_F1_mean']:>14.4f}{a['val_F1_std']:>12.4f}{a['best_epoch_mean']:>13.1f}")
        print("=" * 70)

        best_cfg_by_mode[mode] = agg[0]
        print(f">>> [{mode}] 最优组合（按 val_F1 均值选出）: lr={agg[0]['lr']}  lam_att={agg[0]['lam_att']}  "
              f"平均 best_epoch≈{agg[0]['best_epoch_mean']:.1f}", flush=True)

    # ── 用每个 mode 已选出的最优 (lr, lam) checkpoint，在 val 上搜 HM_THRESH（不用重训，只是换阈值重新解码）──
    thresh_rows = []
    best_thresh_by_mode = {}
    for mode in args.modes:
        cfg = best_cfg_by_mode[mode]
        for seed in args.seeds:
            tr, va, te = splits[seed]
            tag = f"seed{seed}_{mode}_lr{cfg['lr']}_lam{cfg['lam_att']}"
            ckpt_path = os.path.join(work, f"{tag}_bestval.pth")
            model = base.AttnBBoxNet(in_ch=1, base_ch=base.BASE_CH,
                                      use_attn=(mode != "none"), fuse=args.fuse).to(base.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=base.device))
            model.eval()
            for th in args.hm_thresh_grid:
                m = base.evaluate(model, full, va, hm_thresh=th)
                thresh_rows.append({"mode": mode, "seed": seed, "hm_thresh": th,
                                     "val_P": m["bbox_P"], "val_R": m["bbox_R"], "val_F1": m["bbox_F1"]})

        mode_thresh_rows = [r for r in thresh_rows if r["mode"] == mode]
        th_agg = []
        for th in args.hm_thresh_grid:
            sub = [r for r in mode_thresh_rows if r["hm_thresh"] == th]
            th_agg.append({"hm_thresh": th,
                           "val_F1_mean": float(np.mean([r["val_F1"] for r in sub])),
                           "val_F1_std": float(np.std([r["val_F1"] for r in sub]))})
        th_agg.sort(key=lambda a: -a["val_F1_mean"])

        print("\n" + "-" * 50)
        print(f"模式 {mode} 的 HM_THRESH 搜索结果（按 val_F1 均值降序）：")
        print(f"{'hm_thresh':>12}{'val_F1_mean':>14}{'val_F1_std':>12}")
        for a in th_agg:
            print(f"{a['hm_thresh']:>12}{a['val_F1_mean']:>14.4f}{a['val_F1_std']:>12.4f}")
        print("-" * 50)

        best_thresh_by_mode[mode] = th_agg[0]["hm_thresh"]
        print(f">>> [{mode}] 最优 HM_THRESH（按 val_F1 均值选出）: {best_thresh_by_mode[mode]}", flush=True)

    # ── 每个 mode 用各自选出的最优组合 + 最优阈值，在从未参与选参的 test 集上跑一次，给出无偏最终指标 ──
    test_rows = []
    for mode in args.modes:
        cfg = best_cfg_by_mode[mode]
        th = best_thresh_by_mode[mode]
        for seed in args.seeds:
            tr, va, te = splits[seed]
            tag = f"seed{seed}_{mode}_lr{cfg['lr']}_lam{cfg['lam_att']}"
            ckpt_path = os.path.join(work, f"{tag}_bestval.pth")
            model = base.AttnBBoxNet(in_ch=1, base_ch=base.BASE_CH,
                                      use_attn=(mode != "none"), fuse=args.fuse).to(base.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=base.device))
            model.eval()
            m = base.evaluate(model, full, te, hm_thresh=th)
            gap = base.attn_gap(model, full, te)
            print(f"  [FINAL TEST] {mode} seed{seed} (hm_thresh={th}): P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
                  f"F1={m['bbox_F1']:.4f} attn_iou={m['attn_band_iou']:.4f} gap={gap:.4f}", flush=True)
            test_rows.append({"mode": mode, "seed": seed, "hm_thresh": th, **m, "attn_gap": gap})

    keys = ["bbox_P", "bbox_R", "bbox_F1", "attn_band_iou", "attn_gap"]
    print("\n" + "=" * 90)
    print("最终无偏 TEST 指标（每个模式各自用调好的最优超参，跨 seed 划分取平均）：")
    for mode in args.modes:
        cfg = best_cfg_by_mode[mode]
        sub = [r for r in test_rows if r["mode"] == mode]
        print(f"\n[{mode}]  lr={cfg['lr']} lam_att={cfg['lam_att']}")
        for k in keys:
            vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
            if vals:
                print(f"  {k:>14} = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 90)

    with open(os.path.join(work, "search_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(search_rows[0].keys())); w.writeheader(); w.writerows(search_rows)
    with open(os.path.join(work, "final_test_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys())); w.writeheader(); w.writerows(test_rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
