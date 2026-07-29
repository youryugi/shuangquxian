"""
消融实验：彻底关掉 shape_loss(W_SHAPE=0)，只用裸参数 L1(W_NOOBJ/W_VERTEX/
W_SLOPE)训练，同样用 Optuna 调到各自最优，和 tune_loss_weights.py(带
shape_loss)的最优结果做公平对比——用来验证 shape_loss 本身是否真的有用。

为什么必须单独调参，不能直接把 W_SHAPE 设成 0 就完事
--------------------------------------------------------------------------
现在 hyperbola_param_regress.py 里 W_VERTEX/W_SLOPE 默认只有 0.5，是在假设
"shape_loss 才是主力"的前提下定的。如果只是把 W_SHAPE 设成 0、W_VERTEX/
W_SLOPE 还留在 0.5，这个"裸 L1"对照组是被人为削弱的——赢了说明不了 shape_loss
真的有用，只能说明 0.5 的权重不够。所以裸 L1 方案也要经过同样力度的 Optuna
搜索，两边都拿到公平的调参机会，再比谁的上限高，这个结论才站得住脚。

和 tune_loss_weights.py 的区别
--------------------------------------------------------------------------
- W_SHAPE 永远固定为 0（不搜索，直接关闭 shape_loss）。
- 只搜 W_NOOBJ/W_VERTEX/W_SLOPE 三个自由度。
- W_VERTEX/W_SLOPE 的搜索范围比 tune_loss_weights.py 里宽很多(上限从 5.0
  提到 20.0)——没了 shape_loss 兜底，这两项要独自扛起全部顶点/斜率监督，
  给它们更大的可搜索空间，是对裸 L1 方案的公平对待，不能让它输在"权重范围
  太窄"这种无关紧要的地方。
其余设计（猴子补丁、每个 trial 重新 set_seed、评估复用 sweep_thresholds
等）和 tune_loss_weights.py 完全一致，细节见那边的注释。

用法：
    python tune_loss_weights_no_shape.py --n_trials 30 --search_epochs 30
    python tune_loss_weights_no_shape.py --n_trials 30 --final_epochs 80

跑完之后把这里的最佳 F1 和 tune_loss_weights.py（带 shape_loss）的最佳 F1
放在一起比——两边都是各自调到最优后的结果，谁高谁低才有意义。
"""
import os
import random
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

import hyperbola_param_regress as H

_HERE = os.path.dirname(os.path.abspath(__file__))

W_OBJ_FIXED = 1.0
W_SHAPE_FIXED = 0.0   # 消融核心：shape_loss 彻底关闭
SEARCH_SPACE = {
    "W_NOOBJ": (0.1, 2.0),
    "W_VERTEX": (0.05, 20.0),
    "W_SLOPE": (0.05, 20.0),
}


def _import_optuna():
    try:
        import optuna
        return optuna
    except ImportError as e:
        raise SystemExit(
            "loss 权重搜索需要 optuna（当前环境未安装）。\n"
            "  pip install optuna"
        ) from e


def _set_weights(w_noobj, w_vertex, w_slope):
    """猴子补丁：直接改 hyperbola_param_regress 模块里的全局权重常量，
    W_SHAPE 永远设为 0（消融：关闭 shape_loss）。
    """
    H.W_OBJ = W_OBJ_FIXED
    H.W_NOOBJ = w_noobj
    H.W_VERTEX = w_vertex
    H.W_SLOPE = w_slope
    H.W_SHAPE = W_SHAPE_FIXED


def train_one_trial(weights, epochs, tl, vl, device, backbone, pretrained):
    """用给定的一组 loss 权重（W_SHAPE 固定=0）训练一个模型，
    返回 (val集最佳F1, 训好的model)。"""
    _set_weights(*weights)
    H.set_seed(H.SEED)  # 每个 trial 都从同样的初始化/数据顺序开始，只有权重不同

    model = H.ParamRegressNet(backbone=backbone, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=H.LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ep in range(epochs):
        model.train()
        for imgs, targets, _, _ in tl:
            imgs, targets = imgs.to(device), targets.to(device)
            loss, _parts = H.compute_loss(model(imgs), targets)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

    _best_th, _best_nd, best_m = H.sweep_thresholds(model, vl, device)
    return best_m["F1"], model


def make_objective(tl, vl, device, args):
    def objective(trial):
        w_noobj = trial.suggest_float("W_NOOBJ", *SEARCH_SPACE["W_NOOBJ"], log=True)
        w_vertex = trial.suggest_float("W_VERTEX", *SEARCH_SPACE["W_VERTEX"], log=True)
        w_slope = trial.suggest_float("W_SLOPE", *SEARCH_SPACE["W_SLOPE"], log=True)
        f1, _model = train_one_trial((w_noobj, w_vertex, w_slope),
                                      args.search_epochs, tl, vl, device,
                                      args.backbone, args.pretrained)
        print(f"    trial {trial.number:3d}: W_NOOBJ={w_noobj:.3f} W_VERTEX={w_vertex:.3f} "
              f"W_SLOPE={w_slope:.3f} (W_SHAPE=0)  ->  F1={f1:.4f}")
        return f1
    return objective


def main():
    optuna = _import_optuna()

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--search_epochs", type=int, default=30,
                     help="每个 trial 训练的 epoch 数（比正式训练少，先看权重相对好坏）")
    ap.add_argument("--final_epochs", type=int, default=0,
                     help="搜完是否用最优权重再跑一次完整训练；0=不跑，只报告最优权重")
    ap.add_argument("--backbone", default=H.BACKBONE,
                     choices=["scratch", "resnet18", "resnet34", "resnet50", "resnet101"])
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=H.PRETRAINED)
    ap.add_argument("--img_dir", default=H.IMG_DIR)
    ap.add_argument("--json", default=None)
    ap.add_argument("--train_frac", type=float, default=H.TRAIN_FRAC)
    args = ap.parse_args()

    json_path = args.json or os.path.join(args.img_dir, "annotations.json")
    H.set_seed(H.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = H.HyperbolaParamDataset(args.img_dir, json_path)
    print(f"数据集：{len(full)} 张有标注图像  |  设备={device}  |  backbone={args.backbone}")
    print("[消融] W_SHAPE 固定=0（shape_loss 关闭），只搜 W_NOOBJ/W_VERTEX/W_SLOPE")
    if len(full) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    # 和 hyperbola_param_regress.main() 完全相同的切分流程，保证 val 集一致，
    # 也和 tune_loss_weights.py 用同一个 seed，两边的 train/val 切分逐张对应。
    idx = list(range(len(full)))
    random.shuffle(idx)
    n_tr = int(len(idx) * args.train_frac)
    tr = Subset(full, idx[:n_tr])
    va = Subset(full, idx[n_tr:])
    tl = DataLoader(tr, batch_size=H.BATCH_SIZE, shuffle=True, collate_fn=H.collate)
    vl = DataLoader(va, batch_size=H.BATCH_SIZE, shuffle=False, collate_fn=H.collate)

    print(f"[Optuna] {args.n_trials} 组 trial，每组训练 {args.search_epochs} epoch"
          f"（正式训练默认 {H.EPOCHS}）……")
    print(f"[Optuna] 搜索空间（对数均匀，W_OBJ 固定={W_OBJ_FIXED}，W_SHAPE 固定={W_SHAPE_FIXED}）："
          f"{SEARCH_SPACE}")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=H.SEED))
    study.optimize(make_objective(tl, vl, device, args), n_trials=args.n_trials)

    print("\n[Optuna] 最优权重（消融：无 shape_loss）：")
    for k, v in study.best_params.items():
        print(f"    {k} = {v:.4f}")
    print(f"    W_OBJ = {W_OBJ_FIXED}（固定，未搜索）")
    print(f"    W_SHAPE = {W_SHAPE_FIXED}（消融：固定关闭）")
    print(f"    最佳验证 F1（{args.search_epochs} epoch）= {study.best_value:.4f}")
    print("    -> 把这个 F1 和 tune_loss_weights.py（带 shape_loss）调出来的最佳 F1 对比，"
          "谁高谁低才说明 shape_loss 有没有用。")

    if args.final_epochs > 0:
        print(f"\n用最优权重跑一次完整训练（{args.final_epochs} epoch）……")
        best = study.best_params
        weights = (best["W_NOOBJ"], best["W_VERTEX"], best["W_SLOPE"])
        f1, model = train_one_trial(weights, args.final_epochs, tl, vl, device,
                                     args.backbone, args.pretrained)
        print(f"最终 F1（完整 thres×nms 扫描后最优）= {f1:.4f}")
        tag = "tuned_noshape_" + datetime.now().strftime("%m%d_%H%M")
        out = os.path.join(_HERE, tag + ".pth")
        torch.save(model.state_dict(), out)
        print("已保存模型：", out)
    else:
        print("\n把上面的最优权重手动填进 hyperbola_param_regress.py 的 "
              "W_NOOBJ/W_VERTEX/W_SLOPE（W_SHAPE 设 0），再跑一次完整训练即可；"
              "或者直接加 --final_epochs 80 让本脚本自动跑。")


if __name__ == "__main__":
    main()
