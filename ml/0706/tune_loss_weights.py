"""
用 Optuna 对训练阶段的 loss 权重 (W_NOOBJ, W_VERTEX, W_SLOPE, W_SHAPE) 做
系统性搜索，替代 hyperbola_param_regress.py 里手拍的默认值。

为什么要单独写一个脚本，而不是塞进主脚本
--------------------------------------------------------------------------
decode 阶段的 thres/nms_dist 已经在主脚本里做了"免费"的粗调+精调——不需要
重新训练，缓存一次前向结果就能扫完。loss 权重不一样：换一组权重意味着要
重新训练一遍模型，每个候选都要付训练成本，所以用 Optuna 这种贝叶斯搜索
（比网格搜索样本效率高，参考文献综述里也提到网格搜索维度一多组合会爆炸）
比较合适，而不是再手写一个粗网格。

设计
--------------------------------------------------------------------------
1. W_OBJ 固定为 1.0 作为参照，只搜其余 4 个权重相对它的比例——loss 是几项
   加权求和，只有权重之间的相对大小影响优化动态，同时搜 5 个自由度是冗余的。
2. 用"猴子补丁"直接改 hyperbola_param_regress 模块里的全局权重常量再调用
   它的 compute_loss()——因为 compute_loss 里这些权重名字是调用时才从模块
   全局命名空间查找的，不是定义时绑定的，改完模块属性立刻生效，不需要改
   主脚本一个字、不需要维护第二份 loss 实现。
3. 每个 trial 用比正式训练少的 epoch(--search_epochs，默认 30 vs 正式的 80)
   训练，兼顾搜索速度和结果参考价值；搜完可以用 --final_epochs 让脚本自动
   拿最优权重再跑一次完整训练。
4. 每个 trial 的评估直接复用主脚本的 sweep_thresholds()（同一套 thres×nms
   粗调+精调方法论）——如果只用固定阈值 0.5 评估，前面已经验证过这套 loss
   训出来的 objectness 在阈值 0.5 上几乎全是噪声（F1≈0），会让所有 trial
   看起来一样烂，搜索完全学不到东西。所以评估阶段的"调参充分程度"不能打折。
5. 每个 trial 训练前都重新 set_seed(SEED)——保证每组权重都从同样的初始化、
   同样的 batch 顺序开始，把"权重选择"以外的随机性隔离掉，trial 之间的
   差异只来自权重本身。

注意（和之前讨论的 train/val/test 三路切分一致）：本脚本在 val 集上搜权重、
也在同一个 val 集上报告 F1——用于搜索是合理的（这正是验证集的用途），但正式
写进论文的最终数字，等你把 val/test 拆开之后，应该只在没被这个搜索过程碰过
的 test 集上评一次，不要直接引用这里搜索阶段报的 F1。

用法：
    python tune_loss_weights.py --n_trials 30 --search_epochs 30
    python tune_loss_weights.py --n_trials 30 --final_epochs 80   # 搜完自动用最优权重跑一次完整训练
需要 optuna（pip install optuna），懒加载，不装的话只有跑 main() 时才报错。
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
SEARCH_SPACE = {
    # (低, 高)，log=True 做对数均匀采样——loss 权重合适的量级差异可以很大，
    # 线性均匀采样会把大部分采样点浪费在不敏感的区间。
    "W_NOOBJ": (0.1, 2.0),
    "W_VERTEX": (0.05, 5.0),
    "W_SLOPE": (0.05, 5.0),
    "W_SHAPE": (0.3, 30.0),
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


def _set_weights(w_noobj, w_vertex, w_slope, w_shape):
    """猴子补丁：直接改 hyperbola_param_regress 模块里的全局权重常量，
    之后调用 H.compute_loss(...) 会立刻用上这组新权重（原理见文件头注释）。
    """
    H.W_OBJ = W_OBJ_FIXED
    H.W_NOOBJ = w_noobj
    H.W_VERTEX = w_vertex
    H.W_SLOPE = w_slope
    H.W_SHAPE = w_shape


def train_one_trial(weights, epochs, tl, vl, device, backbone, pretrained):
    """用给定的一组 loss 权重训练一个模型，返回 (val集最佳F1, 训好的model)。"""
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
        w_shape = trial.suggest_float("W_SHAPE", *SEARCH_SPACE["W_SHAPE"], log=True)
        f1, _model = train_one_trial((w_noobj, w_vertex, w_slope, w_shape),
                                      args.search_epochs, tl, vl, device,
                                      args.backbone, args.pretrained)
        print(f"    trial {trial.number:3d}: W_NOOBJ={w_noobj:.3f} W_VERTEX={w_vertex:.3f} "
              f"W_SLOPE={w_slope:.3f} W_SHAPE={w_shape:.3f}  ->  F1={f1:.4f}")
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
    if len(full) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    # 和 hyperbola_param_regress.main() 完全相同的切分流程，保证 val 集一致。
    idx = list(range(len(full)))
    random.shuffle(idx)
    n_tr = int(len(idx) * args.train_frac)
    tr = Subset(full, idx[:n_tr])
    va = Subset(full, idx[n_tr:])
    tl = DataLoader(tr, batch_size=H.BATCH_SIZE, shuffle=True, collate_fn=H.collate)
    vl = DataLoader(va, batch_size=H.BATCH_SIZE, shuffle=False, collate_fn=H.collate)

    print(f"[Optuna] {args.n_trials} 组 trial，每组训练 {args.search_epochs} epoch"
          f"（正式训练默认 {H.EPOCHS}）……")
    print(f"[Optuna] 搜索空间（对数均匀，W_OBJ 固定={W_OBJ_FIXED}）：{SEARCH_SPACE}")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=H.SEED))
    study.optimize(make_objective(tl, vl, device, args), n_trials=args.n_trials)

    print("\n[Optuna] 最优权重：")
    for k, v in study.best_params.items():
        print(f"    {k} = {v:.4f}")
    print(f"    W_OBJ = {W_OBJ_FIXED}（固定，未搜索）")
    print(f"    最佳验证 F1（{args.search_epochs} epoch）= {study.best_value:.4f}")

    if args.final_epochs > 0:
        print(f"\n用最优权重跑一次完整训练（{args.final_epochs} epoch）……")
        best = study.best_params
        weights = (best["W_NOOBJ"], best["W_VERTEX"], best["W_SLOPE"], best["W_SHAPE"])
        f1, model = train_one_trial(weights, args.final_epochs, tl, vl, device,
                                     args.backbone, args.pretrained)
        print(f"最终 F1（完整 thres×nms 扫描后最优）= {f1:.4f}")
        tag = "tuned_" + datetime.now().strftime("%m%d_%H%M")
        out = os.path.join(_HERE, tag + ".pth")
        torch.save(model.state_dict(), out)
        print("已保存模型：", out)
    else:
        print("\n把上面的最优权重手动填进 hyperbola_param_regress.py 的 "
              "W_NOOBJ/W_VERTEX/W_SLOPE/W_SHAPE，再跑一次完整训练即可；"
              "或者直接加 --final_epochs 80 让本脚本自动跑。")


if __name__ == "__main__":
    main()
