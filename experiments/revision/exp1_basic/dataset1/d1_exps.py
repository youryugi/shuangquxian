"""数据集 1 的实验定义（run_seeds.py / infer_bandbox.py / aggregate_cv.py 共用，改这里一处即可）。

实验名 d1_<config>_<arm>：
  config  输入尺寸 × 热图 sigma（dataset1/README.md 的第 1 轮：只在 baseline 上按合并验证集选，选定后三个 arm 共用）
            s224    224 输入，sigma 原样（6 个格子 = 48 px）—— 等于数据集 2 的设置
            s224gb  224 输入，sigma 按框大小（CenterNet gaussian_radius）
            s448    448 输入，sigma 原样（6 个格子）
            s448gb  448 输入，sigma 按框大小
            s320 / s384  中间尺寸，sigma 原样（第 1 轮补充）
            s224d4 / s224d4gb  224 输入、检测头步长 4（sigma 原样 = 12 格 = 48 px / 按框大小）
  arm     none = 无 attention（baseline）；abs = band 监督（本文方法）；boxmask = 填实的 HBAT 框监督（对照）
所有 d1 实验共用：HBAT 派生框（*_hbatbox 数据目录）、几何翻转增强、训练集 3 份副本每轮随机取一份（--copy-sampling one）。
与数据集 2 一样，配对的 arm 必须用同一个 --workers（run_seeds.py 默认 2；d1 统一用 --workers 1）。
"""
from __future__ import annotations

import itertools

D1_CONFIGS = {
    "s224":   ["--input-size", "224"],
    "s224gb": ["--input-size", "224", "--hm-sigma", "box"],
    "s448":   ["--input-size", "448"],
    "s448gb": ["--input-size", "448", "--hm-sigma", "box"],
    # 第 1 轮补充：224 → 448 时 pipe 掉、rebar 涨，加两个中间尺寸（sigma 原样，box sigma 在两个尺寸上都不占优）
    "s320":   ["--input-size", "320"],
    "s384":   ["--input-size", "384"],
    # 第 1 轮补充 2：输入一变大 pipe 就掉（感受野约 70–85 px，pipe 在 224 上宽约 70 px）。
    # 224 输入 + 检测头步长 4（数据集 2 第 8 轮的结构）：rebar 中心撞格从 19 降到 1，感受野不变
    "s224d4":   ["--input-size", "224", "--det-stride", "4"],
    "s224d4gb": ["--input-size", "224", "--det-stride", "4", "--hm-sigma", "box"],
}
D1_COMMON = ["--augment", "--copy-sampling", "one"]
D1_ARMS = {
    "none":    ["--mode", "none"],
    "abs":     ["--mode", "abs"],
    "boxmask": ["--mode", "abs", "--mask-dir", "masks_box"],
}

# run_seeds.py 的 EXPS 格式：实验名 -> (数据目录后缀, 额外参数)
D1_EXPS = {f"d1_{c}_{a}": ("_hbatbox", D1_ARMS[a] + D1_COMMON + cargs)
           for c, cargs in D1_CONFIGS.items() for a in D1_ARMS}
D1_SRC_EXPS = tuple(D1_EXPS)


def d1_variants(T_GRID, SEED_GRID, MODES, NMS_GRID, K_GRID):
    """infer_bandbox.py 的 VARIANTS 格式：变体名 -> (来源实验, 参数网格)。网格与数据集 2 的 hbatbox_flipfix_* 完全相同。"""
    v = {}
    for c in D1_CONFIGS:
        none, abs_, box = f"d1_{c}_none", f"d1_{c}_abs", f"d1_{c}_boxmask"
        v[f"{none}_scale"] = (none, [(k,) for k in K_GRID])
        for src in (abs_, box):
            v[f"{src}_fitbox"] = (src, list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID)))
            v[f"{src}_bandbox_nms"] = (src, list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID)))
        v[f"{abs_}_gtfitbox"] = (abs_, list(itertools.product(SEED_GRID, MODES, NMS_GRID)))
    return v


def d1_pairs():
    """aggregate_cv.py --dataset ds1 的默认配对（A:B，报告 B − A）。"""
    p = []
    # 第 1 轮：baseline 的输入尺寸 × sigma（选参看合并验证 F1，见 d1_select_config.py）
    base = [f"d1_{c}_none" for c in D1_CONFIGS]
    p += [f"{a}:{b}" for a, b in itertools.combinations(base, 2)]
    # 第 2 轮：每个 config 下的主比较（和数据集 2 Round 18 同一组配对）
    for c in D1_CONFIGS:
        e = lambda s: f"d1_{c}_{s}"   # noqa: E731
        p += [f"{e(a)}:{e(b)}" for a, b in (
            ("none", "abs"), ("none", "boxmask"), ("boxmask", "abs"),
            ("none", "none_scale"), ("none", "abs_fitbox"), ("none_scale", "abs_fitbox"), ("abs", "abs_fitbox"),
            ("boxmask_fitbox", "abs_fitbox"), ("boxmask_bandbox_nms", "abs_fitbox"), ("abs_bandbox_nms", "abs_fitbox"),
            ("none", "boxmask_bandbox_nms"), ("abs_fitbox", "abs_gtfitbox"))]
    return p
