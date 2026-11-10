"""
用 Optuna 对 baseline_keypoint_regress.py 的训练阶段 loss 权重
(W_NOOBJ, W_POINT) 做系统性搜索，和 tune_loss_weights.py 是同一套方法论，
只是换成关键点方法自己的 loss（见 baseline_keypoint_regress.compute_loss）。

为什么只搜 2 个权重，比主脚本的 4 个少
--------------------------------------------------------------------------
baseline_keypoint_regress.py 的 loss 只有 3 项：
    loss_obj(BCE, 用 W_OBJ/W_NOOBJ 加权) + W_POINT * loss_point(顶点+两端点 L1)
没有主脚本里 shape_loss 那一套（点坐标误差本身就是深度无关的，不需要曲线
采样项，见 baseline_keypoint_regress.py 文件头注释），所以自由度天然更少：
W_OBJ 固定=1.0 当参照，只搜 W_NOOBJ 和 W_POINT 这两个自由度。

设计（和 tune_loss_weights.py 完全一致，细节见那边的注释）
--------------------------------------------------------------------------
1. 猴子补丁直接改 baseline_keypoint_regress 模块的全局权重常量，
   compute_loss() 调用时才查找这些名字，改完立刻生效。
2. 每个 trial 用较少的 epoch(--search_epochs)训练，评估复用
   baseline_keypoint_regress.sweep_thresholds()（thres×nms 粗调+精调，
   围绕粗网格最优点自适应展开邻域，不会走 hyperbola_param_regress.py
   之前那个"精调区间写死、找错地方"的老路）。
3. 每个 trial 训练前都 set_seed(SEED)，隔离"权重选择"以外的随机性。

用法：
    python tune_keypoint_loss_weights.py --n_trials 30 --search_epochs 30
    python tune_keypoint_loss_weights.py --n_trials 30 --final_epochs 80
需要 optuna（pip install optuna）。
"""
import os
import random
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

import baseline_keypoint_regress as K
from hyperbola_param_regress import IMG_DIR, HYP_JSON

_HERE = os.path.dirname(os.path.abspath(__file__))

W_OBJ_FIXED = 1.0
SEARCH_SPACE = {
    "W_NOOBJ": (0.1, 2.0),
    "W_POINT": (0.5, 20.0),
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


def _set_weights(w_noobj, w_point):
    """猴子补丁：直接改 baseline_keypoint_regress 模块里的全局权重常量，
    之后调用 K.compute_loss(...) 会立刻用上这组新权重。
    """
    K.W_OBJ = W_OBJ_FIXED
    K.W_NOOBJ = w_noobj
    K.W_POINT = w_point


def train_one_trial(weights, epochs, tl, vl, device, backbone, pretrained):
    """用给定的一组 loss 权重训练一个关键点模型，返回 (val集最佳F1, 训好的model)。"""
    _set_weights(*weights)
    K.set_seed(K.SEED)  # 每个 trial 都从同样的初始化/数据顺序开始，只有权重不同

    model = K.KeypointNet(backbone=backbone, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=K.LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ep in range(epochs):
        model.train()
        for imgs, targets, _, _ in tl:
            imgs, targets = imgs.to(device), targets.to(device)
            loss, _parts = K.compute_loss(model(imgs), targets)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

    _best_th, _best_nd, best_m = K.sweep_thresholds(model, vl, device)
    return best_m["F1"], model


def make_objective(tl, vl, device, args):
    def objective(trial):
        w_noobj = trial.suggest_float("W_NOOBJ", *SEARCH_SPACE["W_NOOBJ"], log=True)
        w_point = trial.suggest_float("W_POINT", *SEARCH_SPACE["W_POINT"], log=True)
        f1, _model = train_one_trial((w_noobj, w_point), args.search_epochs,
                                      tl, vl, device, args.backbone, args.pretrained)
        print(f"    trial {trial.number:3d}: W_NOOBJ={w_noobj:.3f} W_POINT={w_point:.3f}  ->  F1={f1:.4f}")
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
    ap.add_argument("--backbone", default=K.BACKBONE,
                     choices=["scratch", "resnet18", "resnet34", "resnet50", "resnet101"])
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=K.PRETRAINED)
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--json", default=None)
    ap.add_argument("--train_frac", type=float, default=K.TRAIN_FRAC)
    args = ap.parse_args()

    json_path = args.json or os.path.join(args.img_dir, "annotations.json")
    K.set_seed(K.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = K.KeypointDataset(args.img_dir, json_path)
    print(f"数据集：{len(full)} 张有标注图像  |  设备={device}  |  backbone={args.backbone}")
    if len(full) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    # 和 baseline_keypoint_regress.main() 完全相同的切分流程，保证 val 集一致。
    idx = list(range(len(full)))
    random.shuffle(idx)
    n_tr = int(len(idx) * args.train_frac)
    tr = Subset(full, idx[:n_tr])
    va = Subset(full, idx[n_tr:])
    tl = DataLoader(tr, batch_size=K.BATCH_SIZE, shuffle=True, collate_fn=K.collate)
    vl = DataLoader(va, batch_size=K.BATCH_SIZE, shuffle=False, collate_fn=K.collate)

    print(f"[Optuna] {args.n_trials} 组 trial，每组训练 {args.search_epochs} epoch"
          f"（正式训练默认 {K.EPOCHS}）……")
    print(f"[Optuna] 搜索空间（对数均匀，W_OBJ 固定={W_OBJ_FIXED}）：{SEARCH_SPACE}")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=K.SEED))
    study.optimize(make_objective(tl, vl, device, args), n_trials=args.n_trials)

    print("\n[Optuna] 最优权重：")
    for k, v in study.best_params.items():
        print(f"    {k} = {v:.4f}")
    print(f"    W_OBJ = {W_OBJ_FIXED}（固定，未搜索）")
    print(f"    最佳验证 F1（{args.search_epochs} epoch）= {study.best_value:.4f}")

    if args.final_epochs > 0:
        print(f"\n用最优权重跑一次完整训练（{args.final_epochs} epoch）……")
        best = study.best_params
        weights = (best["W_NOOBJ"], best["W_POINT"])
        f1, model = train_one_trial(weights, args.final_epochs, tl, vl, device,
                                     args.backbone, args.pretrained)
        print(f"最终 F1（完整 thres×nms 扫描后最优）= {f1:.4f}")
        tag = "kp_tuned_" + datetime.now().strftime("%m%d_%H%M")
        out = os.path.join(_HERE, tag + ".pth")
        torch.save(model.state_dict(), out)
        print("已保存模型：", out)
    else:
        print("\n把上面的最优权重手动填进 baseline_keypoint_regress.py 的 "
              "W_NOOBJ/W_POINT，再跑一次完整训练即可；"
              "或者直接加 --final_epochs 80 让本脚本自动跑。")


if __name__ == "__main__":
    main()
