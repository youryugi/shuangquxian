"""
双向跨数据集验证：none vs abs 两种注意力模式，谁在没见过的数据集上掉得更少。

两个数据集：
  A = dataset3/augmented_utilities（现有训练用的手标数据增强版）
  B = dataset3/images-selected（Roboflow 导出，来源不同，标注 schema 兼容）

两个方向都跑（互相当对方的域外测试集）：
  方向1： train/val 在 A 内部划分 → 挑 best epoch → 在 B 全量上测（A → B）
  方向2： train/val 在 B 内部划分 → 挑 best epoch → 在 A 全量上测（B → A）

每个方向内部：
  train —— 更新权重
  val   —— 不参与训练，每隔 EVAL_EVERY 轮评一次 bbox_F1，挑本次训练里最好的 epoch
            （同域早停，不涉及跨数据集）
  跨域测试的那个数据集 —— 全程不参与训练/选参，只在最后跑一次 evaluate

复用 attn_cnn_merged_final.py（本目录下的副本，AttnDataset 已支持传入
img_dir/hyp_json/rect_json 覆盖默认路径，用来同时指向 A、B 两个数据集）。
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
# 配置（可调）
# ══════════════════════════════════════════════════════════════════════════════
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ml/0729-crossdataset -> ml -> repo

# 数据集 A：沿用 base 里的默认路径（augmented_utilities），不用覆盖
DOMAIN_A_NAME = "augmented_utilities"
DOMAIN_A_KW   = {}

# 数据集 B：images-selected（Roboflow 导出，跟 A 来源不同）
DOMAIN_B_NAME = "images-selected-shuangquxian"
_B_DIR = os.path.join(_REPO_ROOT, "dataset3", "images-selected-shuangquxian")
DOMAIN_B_KW = {"img_dir": _B_DIR,
               "hyp_json": os.path.join(_B_DIR, "annotations.json"),
               "rect_json": os.path.join(_B_DIR, "annotations_rect.json")}

# 两个方向都跑：(train_domain_name, test_domain_name)
DIRECTIONS = [(DOMAIN_A_NAME, DOMAIN_B_NAME), (DOMAIN_B_NAME, DOMAIN_A_NAME)]

MODES  = ["none", "abs"]           # 只对比这两种；跟 attn_cnn_merged_final.py 里当前启用的模式一致
SEEDS  = [10, 11, 12, 13, 14,15,16,17,18,19]              # train/val 划分种子（在当前方向的训练域内部），结果跨 seed 取平均
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.3                   # 不留同域 test：泛化能力直接看另一个数据集上的表现

LR       = 5e-4                     # 如果超参搜索已经跑出更优的 lr/lam_att，改这两个常量即可
LAM_ATT  = 0.7
FUSE     = base.FUSE

MAX_EPOCHS = 150
EVAL_EVERY = 5
SELECT_KEY = "bbox_F1"              # 挑 best epoch 用的指标（同域 val 上）


# ══════════════════════════════════════════════════════════════════════════════
def train_and_select(attn_mode, lr, lam, full_train, tr_idx, va_idx, seed,
                      max_epochs, eval_every, fuse, augment):
    """在训练域的 train 上训练，每 eval_every 轮在同域 val 上评一次，
    返回 val bbox_F1 最高那个 epoch 的 checkpoint + 指标。"""
    base.set_seed(seed)
    use_attn = (attn_mode != "none")
    if lam is not None:
        base.LAM_ATT_CUR = lam

    tr_set = Subset(full_train, tr_idx)
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
            m = base.evaluate(model, full_train, va_idx)
            if m[SELECT_KEY] >= best["score"]:
                best["score"] = m[SELECT_KEY]; best["epoch"] = ep; best["metrics"] = m
                best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=MODES, choices=["none", "abs", "soft"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lam_att", type=float, default=LAM_ATT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    parser.add_argument("--fuse", default=FUSE, choices=["gate", "concat"])
    parser.add_argument("--augment", action="store_true", default=base.AUGMENT)
    parser.add_argument("--one_way", choices=["a2b", "b2a"], default=None,
                        help="只跑单个方向（默认双向都跑）：a2b = A→B，b2a = B→A")
    args = parser.parse_args()

    directions = DIRECTIONS
    if args.one_way == "a2b":
        directions = [(DOMAIN_A_NAME, DOMAIN_B_NAME)]
    elif args.one_way == "b2a":
        directions = [(DOMAIN_B_NAME, DOMAIN_A_NAME)]

    work = os.path.join(os.getcwd(), f"attn_cnn_cross_dataset_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(work, os.path.basename(__file__)))

    print("Using device:", base.device)
    domain_kw = {DOMAIN_A_NAME: DOMAIN_A_KW, DOMAIN_B_NAME: DOMAIN_B_KW}
    domains = {name: base.AttnDataset(input_size=base.input_size, hm_stride=base.HM_STRIDE,
                                       sigma=base.HM_SIGMA, **kw)
               for name, kw in domain_kw.items()}
    for name, ds in domains.items():
        print(f"[cross_dataset] domain '{name}': n={len(ds)}", flush=True)
    print(f"[cross_dataset] directions={directions} modes={args.modes} seeds={args.seeds} "
          f"lr={args.lr} lam_att={args.lam_att} epochs={args.epochs}", flush=True)

    rows = []
    for train_name, test_name in directions:
        train_full, test_full = domains[train_name], domains[test_name]
        n_train = len(train_full)
        test_idx_all = list(range(len(test_full)))
        direction_tag = f"{train_name}→{test_name}"
        print(f"\n########## 方向: {direction_tag} ##########", flush=True)

        for seed in args.seeds:
            tr, va, _ = base.make_split(n_train, seed, train_frac=args.train_frac, val_frac=args.val_frac)
            print(f"\n=== [{direction_tag}] seed {seed}  train={len(tr)} val={len(va)}（同域，仅用于挑 epoch） "
                  f"cross_test={len(test_full)}（异域，全量） ===", flush=True)

            for mode in args.modes:
                t0 = time.perf_counter()
                lam = args.lam_att if mode != "none" else None
                tag = f"{train_name}_seed{seed}_{mode}"
                best = train_and_select(mode, args.lr, lam, train_full, tr, va, seed,
                                         args.epochs, args.eval_every, args.fuse, args.augment)
                va_m = best["metrics"]
                torch.save(best["state"], os.path.join(work, f"{tag}_bestval.pth"))

                model = base.AttnBBoxNet(in_ch=1, base_ch=base.BASE_CH,
                                          use_attn=(mode != "none"), fuse=args.fuse).to(base.device)
                model.load_state_dict(best["state"])
                model.eval()
                cross_m = base.evaluate(model, test_full, test_idx_all)
                cross_gap = base.attn_gap(model, test_full, test_idx_all)

                print(f"  [{direction_tag} seed{seed}] {mode:>4}: best_epoch={best['epoch']}/{args.epochs}  "
                      f"val(同域) F1={va_m['bbox_F1']:.4f}  |  "
                      f"cross(异域) P={cross_m['bbox_P']:.4f} R={cross_m['bbox_R']:.4f} "
                      f"F1={cross_m['bbox_F1']:.4f} attn_iou={cross_m['attn_band_iou']:.4f} "
                      f"gap={cross_gap:.4f}  [{time.perf_counter() - t0:.1f}s]", flush=True)

                rows.append({"direction": direction_tag, "train_domain": train_name, "test_domain": test_name,
                             "seed": seed, "mode": mode, "best_epoch": best["epoch"],
                             "val_F1": va_m["bbox_F1"],
                             "cross_P": cross_m["bbox_P"], "cross_R": cross_m["bbox_R"],
                             "cross_F1": cross_m["bbox_F1"], "cross_attn_iou": cross_m["attn_band_iou"],
                             "cross_attn_gap": cross_gap})

    # ── 汇总 ──
    keys = ["val_F1", "cross_P", "cross_R", "cross_F1", "cross_attn_iou", "cross_attn_gap"]

    def _print_table(title, sub_rows, group_key):
        print(f"\n{title}")
        print(f"{group_key:>20}" + "".join(f"{k:>17}" for k in keys))
        groups = sorted({r[group_key] for r in sub_rows}, key=lambda g: str(g))
        for g in groups:
            sub = [r for r in sub_rows if r[group_key] == g]
            line = f"{str(g):>20}"
            for k in keys:
                vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
                line += f"{np.mean(vals):>9.4f}±{np.std(vals):<6.4f}" if vals else f"{'nan':>17}"
            print(line)

    print("\n" + "=" * 120)
    for train_name, test_name in directions:
        direction_tag = f"{train_name}→{test_name}"
        sub = [r for r in rows if r["direction"] == direction_tag]
        _print_table(f"方向 {direction_tag}（按 mode 汇总，跨 seed 取平均）：", sub, "mode")
    print("\n" + "-" * 120)
    _print_table("两个方向合并（按 mode 汇总，跨方向+seed 取平均，看整体域外泛化）：", rows, "mode")
    print("=" * 120)
    print("解读：val_F1 是同域（训练所用数据集内部 val）上的表现，cross_* 是异域（另一个数据集）上的表现；\n"
          "      同域高、异域低 = 过拟合到了训练数据集本身；abs 相对 none 的 cross_F1 差值就是注意力监督\n"
          "      带来的域外泛化增益（或损失，如果是负的）。两个方向都好才算稳健，只有单方向好可能是\n"
          "      两个数据集本身难度/规模不对称造成的。", flush=True)

    with open(os.path.join(work, "cross_dataset_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
