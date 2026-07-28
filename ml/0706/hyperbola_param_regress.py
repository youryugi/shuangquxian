"""
双曲线「参数回归」单文件版（anchor-free 网格头）。

结论先行 —— 到底该回归哪几个参数？
--------------------------------------------------------------------------
标注文件 annotations.json（新格式，见 dataset3/biaozhu-hyperbola-annotator.py）
里每条双曲线有 5 个字段：
    x_vertex, y_vertex, slope, span, thickness
但 GPR 点/柱状目标的双曲线，物理自由度只有 3 个：(x0, t0, v)
    x_vertex  <->  x0   顶点横坐标
    y_vertex  <->  t0   顶点深度（图像顶部=地表=time-zero，所以 a = y_vertex）
    slope     <->  v    渐近线斜率 a/b，编码波速（b = y_vertex / slope）
中心线完全由这 3 个量决定：
    y_c(x) = y_v + a * ( sqrt(1 + ((x - x_v)/b)^2) - 1 ),  a = y_v, b = y_v/slope

而 span、thickness 不是双曲线的几何自由度：
    - span      只是标注时「画多长的两臂 / 截断范围」，曲线本身延伸到无穷；
    - thickness 只是把中心线加粗成带的「带宽」，本数据里基本是常数(≈19~27)。
所以你的直觉是对的：**只回归顶点 (x_vertex, y_vertex) + 斜率 slope 这 3 个数
就足以唯一确定双曲线形状**。span/thickness 若需要可另外回归或直接取常数，
它们对「形状/曲线」没有影响，只影响画图/带区域。

本脚本做什么
--------------------------------------------------------------------------
一张图里可能有多条双曲线（数量不定），所以不是「一张图 -> 3 个数」的纯回归，
而是沿用你 ml/0624 里的 YOLO anchor-free 思路：把图切成网格，每个格子回归
    [objectness, dx, dy, log_slope]
即「这个格子里有没有顶点 + 顶点精确位置 + 斜率」。把 YOLO 的 (x,y,w,h)
换成 (x,y,slope) —— 正是「只回归顶点 + 斜率」。

用法：
    python hyperbola_param_regress.py                # 训练 + 评估
    python hyperbola_param_regress.py --epochs 60
数据默认取 dataset3/images-selected-shuangquxian（新 slope 格式）；
若 JSON 是旧格式(width/height 或 a/b) 也会被 normalize_obj 自动换算成 slope。

Loss 设计（标注是"带"而非离散关键点，见 compute_loss / y_center）：
    标注是双曲线带，参数空间里的 L1（对 x_vertex/y_vertex/slope 分别求误差）
    和"带贴不贴合"是脱节的：slope 和 y_vertex 耦合，同样的 slope 误差在
    深顶点处会被 b=y_v/slope 放大成很大的臂端偏移，浅顶点处却几乎不可见。
    所以主损失改成沿 GT span 采样中心线、比较预测/GT 曲线的竖直差
    (shape_loss)，直接度量带的贴合程度，天然覆盖了三个参数的耦合关系；
    独立的顶点/斜率 L1 只保留很小权重，作为早期训练的辅助梯度。
"""
import os
import json
import math
import random
import argparse
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

# 严格确定性：CUBLAS_WORKSPACE_CONFIG 必须在 import torch（初始化 cuBLAS）之前设置
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ══════════════════════════════════════════════════════════════════════════
# 超参数
# ══════════════════════════════════════════════════════════════════════════
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))         # ml/0706 -> ml -> repo

IMG_DIR    = os.path.join(_REPO_ROOT, "dataset3", "images-selected-shuangquxian")
HYP_JSON   = os.path.join(IMG_DIR, "annotations.json")

INPUT_SIZE = 640          # 正方形输入
STRIDE     = 16           # 网格步长 -> 40x40 网格
GRID       = INPUT_SIZE // STRIDE

BATCH_SIZE = 8
EPOCHS     = 80
LR         = 5e-4
SEED       = 42
TRAIN_FRAC = 0.8

BACKBONE   = "resnet50"    # 主干网络：scratch(零依赖) / resnet18 / resnet34 / resnet50 / resnet101（命令行 --backbone 可覆盖）
PRETRAINED = True         # backbone 为 resnet* 时是否加载 ImageNet 预训练权重（离线环境改 False / 用 --no-pretrained）
# [Optuna] 最优权重：
#     W_NOOBJ = 1.5636
#     W_VERTEX = 0.3399
#     W_SLOPE = 1.9055
#     W_SHAPE = 0.3021
#     W_OBJ = 1.0（固定，未搜索）
# loss 权重
W_OBJ      = 1.0          # objectness BCE
W_NOOBJ    = 0.5          # 负样本 objectness 权重（负样本远多于正样本）
W_VERTEX   = 0.5          # 顶点 offset L1（小权重，只是稳早期收敛的辅助项）
W_SLOPE    = 0.5          # log(slope) L1（同上，辅助项）
W_SHAPE    = 3.0          # 中心线采样损失——真正衡量"带"贴不贴合的主项
#最优权重
W_NOOBJ = 1.5636
W_VERTEX = 0.3399
W_SLOPE = 1.9055
W_SHAPE = 0.3021
SHAPE_SAMPLES = 16        # 沿 GT span 采样点数
#绿色 = GT(标注的真实双曲线),来自 annotations.json 里的原始标注对象。
#红色 = 模型预测,在你选定的最优 (thres, nms_dist) 下解码+NMS 后的检测结果。
# 评估：预测顶点与 GT 顶点距离 <= 该像素阈值(原图尺度)算命中
MATCH_DIST = 20.0
OBJ_THRES  = 0.5
NMS_DIST   = 24.0         # NMS：顶点距离 <= 该阈值(原图尺度)视为重复检测，只留 objectness 最高的一个
                          # 略大于 STRIDE(=16) 的对角线距离(≈22.6)，保证同一顶点周围
                          # 网格邻居(含对角)总能被合并，而不只是上下左右四邻

# 阈值 × NMS 距离扫描：两者都是 decode 期参数，训练完直接对缓存的预测重新
# 打分即可，不需要重新训练，扫描本身零训练成本。
DO_SWEEP   = True         # 训练结束后是否做这个扫描；关掉则直接用 OBJ_THRES/NMS_DIST 默认值评估
FINE_SWEEP = True         # True: 额外在下面的细网格里搜全局最优（只报最优点，不整表打印）
                          # False: 只打印粗网格，快但可能不是全局最优（比如卡在网格边界）
THRESH_SWEEP = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]   # 粗网格：打印成表，人眼看趋势
NMS_SWEEP    = [12.0, 16.0, 20.0, 24.0, 32.0, 40.0]     # 粗网格：打印成表，人眼看趋势
# 精细网格用"围绕粗网格最优点的偏移量"表示，而不是写死一个绝对区间——
# 换了 loss 权重/backbone 之后真正的最优点会漂移，固定绝对区间(比如早期
# 认定的 0.95~0.995)会跟丢，精细搜索白跑还搜不到比粗网格更好的结果（曾经
# 真实发生过：粗网格最优在 thres=0.7，精细搜索却锁死在 [0.95,0.995] 里找，
# 两边报的"最优"对不上）。偏移量列表包含 0.0，保证精细搜索至少能复现粗网格
# 的最优点，不会更差。
THRESH_FINE_DELTAS = [-0.20, -0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
NMS_FINE_DELTAS    = [-16.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 16.0]

# 训练结束后，在最优 (thres, nms_dist) 下把预测和 GT 一起画出来存图，肉眼检查
# 而不是只看 P/R/F1 这几个数字。
DO_VISUALIZE  = True       # 是否在训练结束后导出对比图
VIS_MAX_IMAGES = 20        # 最多导出多少张验证集图（图多了没必要全存）
VIS_PRED_SPAN = 300.0      # 预测曲线画多长两臂用的固定常数——不借用 GT 的 span，
                          # 保持预测这一侧从头到尾不碰任何 GT 信息（GT 曲线画自己的真实 span，不受影响）


# ══════════════════════════════════════════════════════════════════════════
# 复现性：只 torch.manual_seed 不够——cuDNN 默认会为卷积挑"跑得最快"的算法，
# GPU 并行归约顺序不固定，同一个 seed 每次训练的浮点结果仍可能有细微差异，
# 80 个 epoch 累积下来可能看出明显偏差。写法与 ml/0624/attn_cnn_yolo_final.py
# 的 set_seed() 一致。
# ══════════════════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ══════════════════════════════════════════════════════════════════════════
# 标注归一化（兼容旧格式 width/height、a/b -> slope），移植自标注工具
# ══════════════════════════════════════════════════════════════════════════
def normalize_obj(obj):
    obj = dict(obj)
    if "span" not in obj and "width" in obj:
        obj["span"] = obj["width"]
    obj.setdefault("span", 100.0)
    if "slope" not in obj:
        if "a" in obj and obj.get("b"):
            obj["slope"] = round(obj["a"] / obj["b"], 2)
        elif "height" in obj:
            # 旧抛物线/矩形派生：height 是从顶点到臂端的竖直高度
            half_w = max(1.0, obj["span"] / 2.0)
            # y_edge = y_v + a*(sqrt(1+(half_w/b)^2)-1), a=y_v, 反解 slope 较繁；
            # 这里用一阶近似 slope ≈ height / half_w（够作训练初值）。
            obj["slope"] = round(max(0.1, obj["height"] / half_w), 2)
        else:
            obj["slope"] = 1.0
    obj["slope"] = float(max(0.05, obj["slope"]))
    return obj


# ══════════════════════════════════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════════════════════════════════
class HyperbolaParamDataset(Dataset):
    """返回 (image_tensor, target_grid, gt_list)。

    target_grid: [5, GRID, GRID]  通道 = [obj, dx, dy, log_slope, span]
        obj      : 该格是否为某顶点所在格 (0/1)
        dx,dy    : 顶点在格内的相对偏移 (0~1)
        log_slope: ln(slope)
        span     : GT 的水平画带范围（输入尺度像素），仅用于 shape_loss
                   采样区间，不参与网络预测/回归。
    gt_list: [(x_px, y_px, slope), ...]  原图像素尺度，供评估用
    """

    def __init__(self, img_dir, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.img_dir = img_dir
        self.items = []
        for name, objs in raw.items():
            path = os.path.join(img_dir, name)
            if not os.path.exists(path):
                continue
            objs = [normalize_obj(o) for o in objs]
            if objs:
                self.items.append((name, objs))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, objs = self.items[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w0, h0 = img.size
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1)      # [3,H,W]

        sx = INPUT_SIZE / w0
        sy = INPUT_SIZE / h0

        target = torch.zeros(5, GRID, GRID, dtype=torch.float32)
        gt_list = []
        for o in objs:
            # 原图 -> 输入尺度
            xin = o["x_vertex"] * sx
            yin = o["y_vertex"] * sy
            span_in = o["span"] * sx
            # slope 是 a/b = 竖直半轴/水平半轴 的比值，是无量纲比例，
            # 但图像各向异性缩放会改变它：a 按 sy 缩、b 按 sx 缩，
            # 故输入尺度下 slope' = slope * (sy/sx)。评估时再换算回去。
            slope_in = o["slope"] * (sy / sx)
            if not (0 <= xin < INPUT_SIZE and 0 <= yin < INPUT_SIZE):
                continue
            gx = int(xin // STRIDE)
            gy = int(yin // STRIDE)
            gx = min(gx, GRID - 1)
            gy = min(gy, GRID - 1)
            dx = (xin - gx * STRIDE) / STRIDE
            dy = (yin - gy * STRIDE) / STRIDE
            target[0, gy, gx] = 1.0
            target[1, gy, gx] = dx
            target[2, gy, gx] = dy
            target[3, gy, gx] = math.log(max(0.05, slope_in))
            target[4, gy, gx] = span_in
            gt_list.append((o["x_vertex"], o["y_vertex"], o["slope"]))

        return img_t, target, gt_list, (w0, h0)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    gts = [b[2] for b in batch]
    sizes = [b[3] for b in batch]
    return imgs, targets, gts, sizes


# ══════════════════════════════════════════════════════════════════════════
# 模型：主干（from-scratch 或 预训练 ResNet）+ 参数回归头
#
# backbone 和 loss 是完全解耦的两件事：预训练权重只初始化"图像->特征图"这一段，
# compute_loss/y_center/shape_loss 管的是"特征图->顶点+斜率"这一段，换 backbone
# 不需要碰 loss 一个字。做法和 ml/0624/attn_cnn_yolo_resnet.py 完全一致——那边
# 已经在同一批数据上验证过 ImageNet 预训练 backbone 能跑通，这里直接复用同样的
# 接入方式（torchvision.models 里的 ResNet，取某一层输出喂给检测头）。
#
# 数据集只有 318 张图、从零训练一个 CNN 偏小，预训练的通用边缘/纹理特征这时候
# 价值最大——这也是本任务更该加预训练的原因，而不是"loss 不同就不能用"。
# ══════════════════════════════════════════════════════════════════════════
def conv_bn(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ScratchBackbone(nn.Module):
    """原来的轻量卷积主干（无任何依赖），stride=16，输出 256 通道。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32))
        self.d1 = nn.Sequential(conv_bn(32, 64, stride=2), conv_bn(64, 64))     # /2
        self.d2 = nn.Sequential(conv_bn(64, 128, stride=2), conv_bn(128, 128))  # /4
        self.d3 = nn.Sequential(conv_bn(128, 256, stride=2), conv_bn(256, 256)) # /8
        self.d4 = nn.Sequential(conv_bn(256, 256, stride=2), conv_bn(256, 256)) # /16
        self.out_ch = 256

    def forward(self, x):
        x = self.stem(x)
        x = self.d1(x); x = self.d2(x); x = self.d3(x); x = self.d4(x)
        return x


def _import_torchvision():
    try:
        import torchvision
        return torchvision
    except ImportError as e:
        raise SystemExit(
            "使用 --backbone resnet* 需要先安装 torchvision（当前环境未安装）。\n"
            "  pip install torchvision\n"
            "或改用 --backbone scratch（默认，无额外依赖）。"
        ) from e


class ResNetBackbone(nn.Module):
    """torchvision ResNet 主干，取 stride=16 的 layer3 输出（和检测头分辨率对齐）。
    通道数随深度自动推断：resnet18/34 -> 256；resnet50/101 -> 1024。
    做法与 ml/0624/attn_cnn_yolo_resnet.py 的 ResNetBackbone 一致。
    """
    _WEIGHTS = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
    }

    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()
        if name not in self._WEIGHTS:
            raise ValueError(f"unsupported backbone: {name} (choose from {list(self._WEIGHTS)})")
        torchvision = _import_torchvision()
        weights = getattr(torchvision.models, self._WEIGHTS[name]).IMAGENET1K_V1 if pretrained else None
        m = getattr(torchvision.models, name)(weights=weights)
        # conv1+bn+relu+maxpool(s4) -> layer1(s4) -> layer2(s8) -> layer3(s16)
        self.stage = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        with torch.no_grad():
            out = self.stage(torch.zeros(1, 3, 64, 64))
        self.out_ch = out.shape[1]

    def forward(self, x):
        return self.stage(x)


class ParamRegressNet(nn.Module):
    """输入 [B,3,640,640] -> 输出 [B,4,40,40]（stride=16）。

    backbone="scratch"(默认，零依赖) 或 "resnet18/34/50/101"(torchvision 预训练)。
    """

    def __init__(self, backbone="scratch", pretrained=True):
        super().__init__()
        if backbone == "scratch":
            self.backbone = ScratchBackbone()
        else:
            self.backbone = ResNetBackbone(name=backbone, pretrained=pretrained)
        feat_ch = self.backbone.out_ch
        self.head = nn.Sequential(
            conv_bn(feat_ch, 256),
            nn.Conv2d(256, 4, 1),   # [obj, dx, dy, log_slope]
        )
        # objectness 偏置初始化为负，训练初期抑制大量负样本的梯度爆炸
        self.head[-1].bias.data[0] = -4.0

    def forward(self, x):
        return self.head(self.backbone(x))     # [B,4,GRID,GRID]


# ══════════════════════════════════════════════════════════════════════════
# 损失
# ══════════════════════════════════════════════════════════════════════════
def y_center(x, x_vertex, y_vertex, slope):
    """双曲线中心线 y_c(x)，全程可微。y_vertex 下限裁剪防止 a/b 除零爆炸。"""
    a = y_vertex.clamp(min=1.0)
    b = a / slope.clamp(min=1e-3)
    return y_vertex + a * (torch.sqrt(1.0 + ((x - x_vertex) / b) ** 2) - 1.0)


def compute_loss(pred, target):
    B, _, G, _ = pred.shape
    obj_t = target[:, 0]                      # [B,G,G]
    pos = obj_t > 0.5
    neg = ~pos

    obj_logit = pred[:, 0]
    bce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    # 正/负样本分开求均值再加权：否则 ~2 个正样本会被 ~1600 个简单负样本
    # 稀释掉，objectness 永远学不起来（P/R 恒为 0）。
    loss_pos = bce[pos].mean() if pos.any() else torch.zeros((), device=bce.device)
    loss_neg = bce[neg].mean() if neg.any() else torch.zeros((), device=bce.device)
    loss_obj = W_OBJ * loss_pos + W_NOOBJ * loss_neg

    if pos.any():
        device = pred.device
        gy_idx, gx_idx = torch.meshgrid(
            torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
        gy_idx = gy_idx.unsqueeze(0).expand(B, -1, -1)[pos].float()
        gx_idx = gx_idx.unsqueeze(0).expand(B, -1, -1)[pos].float()

        dx_p = torch.sigmoid(pred[:, 1])[pos]
        dy_p = torch.sigmoid(pred[:, 2])[pos]
        ls_p = pred[:, 3][pos]

        dx_g = target[:, 1][pos]
        dy_g = target[:, 2][pos]
        ls_g = target[:, 3][pos]
        span_g = target[:, 4][pos]

        # 顶点 offset / log(slope) 小权重 L1：只是给早期训练一个直接梯度，
        # 真正衡量"带"贴合程度的是下面的 shape_loss（中心线采样）。
        loss_vertex = F.l1_loss(dx_p, dx_g) + F.l1_loss(dy_p, dy_g)
        loss_slope = F.l1_loss(ls_p, ls_g)

        xv_p = (gx_idx + dx_p) * STRIDE
        yv_p = (gy_idx + dy_p) * STRIDE
        s_p = torch.exp(ls_p)
        xv_g = (gx_idx + dx_g) * STRIDE
        yv_g = (gy_idx + dy_g) * STRIDE
        s_g = torch.exp(ls_g)

        # 沿 GT span 采样 K 个 x（以 GT 顶点为中心），比较预测/GT 中心线的
        # 竖直差。这一项同时约束 x_vertex、y_vertex、slope 三者的耦合关系，
        # 直接对应"带"标注的贴合程度，而不是三个独立参数的误差之和。
        t = torch.linspace(-0.5, 0.5, SHAPE_SAMPLES, device=device)      # [K]
        xs = xv_g.unsqueeze(1) + t.unsqueeze(0) * span_g.unsqueeze(1)    # [N,K]
        yp = y_center(xs, xv_p.unsqueeze(1), yv_p.unsqueeze(1), s_p.unsqueeze(1))
        yg = y_center(xs, xv_g.unsqueeze(1), yv_g.unsqueeze(1), s_g.unsqueeze(1))
        # 除以 STRIDE，使量纲与 dx/dy(取值~0-1) 大致同一数量级
        loss_shape = F.smooth_l1_loss(yp / STRIDE, yg / STRIDE)
    else:
        loss_vertex = torch.zeros((), device=pred.device)
        loss_slope = torch.zeros((), device=pred.device)
        loss_shape = torch.zeros((), device=pred.device)

    total = loss_obj + W_VERTEX * loss_vertex + W_SLOPE * loss_slope + W_SHAPE * loss_shape
    return total, {"obj": loss_obj.item(),
                   "vertex": loss_vertex.detach().item(),
                   "slope": loss_slope.detach().item(),
                   "shape": loss_shape.detach().item()}


# ══════════════════════════════════════════════════════════════════════════
# 解码与评估
# ══════════════════════════════════════════════════════════════════════════
def decode(pred, size, thres=OBJ_THRES):
    """单张预测 [4,G,G] -> [(x_px, y_px, slope), ...]（原图尺度）。"""
    w0, h0 = size
    sx = INPUT_SIZE / w0
    sy = INPUT_SIZE / h0
    obj = torch.sigmoid(pred[0])
    dx = torch.sigmoid(pred[1])
    dy = torch.sigmoid(pred[2])
    slope_in = torch.exp(pred[3])
    out = []
    ys, xs = torch.where(obj >= thres)
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        xin = (gx + dx[gy, gx].item()) * STRIDE
        yin = (gy + dy[gy, gx].item()) * STRIDE
        s_in = slope_in[gy, gx].item()
        # 换回原图尺度
        x_px = xin / sx
        y_px = yin / sy
        slope = s_in * (sx / sy)
        out.append((x_px, y_px, slope, obj[gy, gx].item()))
    return out


def nms(dets, dist_thres=NMS_DIST):
    """贪心距离 NMS：按 objectness 从高到低保留，顶点距离 < dist_thres 的
    后续检测视为对同一个顶点的重复响应，予以抑制。

    没有这一步，网格头在真实顶点周围一圈格子上会同时给出偏高的
    objectness（CNN 感受野让邻近格子响应相近），导致同一条双曲线产生
    一大片"检测"：只有离 GT 最近的一个被算命中，其余全部计为误检，
    precision 被严重拉低而与模型实际质量不符。
    """
    dets = sorted(dets, key=lambda d: -d[3])
    kept = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > dist_thres for k in kept):
            kept.append(d)
    return kept


def _score_at_threshold(preds, gts, sizes, thres, nms_dist=NMS_DIST):
    """给定一批已缓存的预测，在某个 (objectness 阈值, NMS 距离) 下解码+NMS+匹配，算 P/R/F1。"""
    tp = fp = fn = 0
    vx_err, vy_err, slope_err = [], [], []
    for p, gt, size in zip(preds, gts, sizes):
        dets = decode(p, size, thres=thres)
        dets = nms(dets, dist_thres=nms_dist)  # 去掉同一顶点的重复检测，再匹配 GT
        # 贪心匹配（按 objectness 从高到低，nms() 已排好序）
        used = [False] * len(gt)
        for dx_, dy_, ds, _ in dets:
            best, bj = MATCH_DIST, -1
            for j, (gx, gy, gs) in enumerate(gt):
                if used[j]:
                    continue
                d = math.hypot(dx_ - gx, dy_ - gy)
                if d < best:
                    best, bj = d, j
            if bj >= 0:
                used[bj] = True
                tp += 1
                gx, gy, gs = gt[bj]
                vx_err.append(abs(dx_ - gx))
                vy_err.append(abs(dy_ - gy))
                slope_err.append(abs(ds - gs))
            else:
                fp += 1
        fn += used.count(False)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "P": prec, "R": rec, "F1": f1,
        "vertex_mae": float(np.mean(vx_err + vy_err)) if vx_err else float("nan"),
        "slope_mae": float(np.mean(slope_err)) if slope_err else float("nan"),
    }


@torch.no_grad()
def _collect_predictions(model, loader, device):
    model.eval()
    preds, gts, sizes = [], [], []
    for imgs, _, g, sz in loader:
        p = model(imgs.to(device)).cpu()
        preds.extend(p)
        gts.extend(g)
        sizes.extend(sz)
    return preds, gts, sizes


def evaluate(model, loader, device, thres=OBJ_THRES, nms_dist=NMS_DIST):
    preds, gts, sizes = _collect_predictions(model, loader, device)
    return _score_at_threshold(preds, gts, sizes, thres, nms_dist=nms_dist)


# obj/noobj 平衡 BCE 训出来的 objectness 天然是"大量格子中等偏高、少数格子
# 很高"的分布，固定阈值 0.5 会放进一堆中等置信度的背景格子（表现为 R 很高但
# P 很低）。你在 ml/0624/attn_cnn_yolo_final.py 里已经用同样的 obj/noobj
# 损失验证过：真正好用的阈值在 0.9~0.97 附近，需要扫描找，不能拍脑袋定。
#
# thres 和 nms_dist 都只是 decode 期的后处理参数，不影响训练好的权重，扫描
# 不需要重新训练——用缓存的一次前向结果对整个网格逐点打分，成本几乎为零。
# DO_SWEEP/FINE_SWEEP/*_SWEEP/*_FINE_RANGE 见文件顶部"超参数"区。
def _frange(start, stop, step):
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(n)]


def sweep_thresholds(model, loader, device, thresholds=THRESH_SWEEP, nms_dists=NMS_SWEEP):
    """只跑一次前向，缓存预测后：
      1) 在手选的粗网格(thresholds x nms_dists)上打印一张可读的 P/R/F1 表；
      2) 若 FINE_SWEEP=True，围绕粗网格找到的最优点，用 THRESH_FINE_DELTAS x
         NMS_FINE_DELTAS 展开邻域网格精调（不整表打印，只报最优点）。精调
         范围自适应地跟着粗网格的最优点走，不会出现"粗网格最优在别处、
         精调却锁死在旧区间找不到"的情况。
    返回 F1 最高的 (thres, nms_dist, metrics)（精调更优时以精调结果为准，
    偏移量含 0.0 保证精调至少不会比粗网格差）。
    """
    preds, gts, sizes = _collect_predictions(model, loader, device)

    grid = {}
    for th in thresholds:
        for nd in nms_dists:
            grid[(th, nd)] = _score_at_threshold(preds, gts, sizes, th, nms_dist=nd)

    cell_w = 19   # "0.xxx/0.xxx/0.xxx" 是 17 字符，留 2 格间距避免和相邻列粘在一起
    print("    (每格 = P/R/F1)")
    print("    thres\\nms" + "".join(f"{nd:>{cell_w}.0f}" for nd in nms_dists))
    for th in thresholds:
        row = "".join(
            f"{grid[(th, nd)]['P']:.3f}/{grid[(th, nd)]['R']:.3f}/{grid[(th, nd)]['F1']:.3f}".rjust(cell_w)
            for nd in nms_dists
        )
        print(f"    {th:>7.2f}  {row}")

    best_th, best_nd = max(grid, key=lambda k: grid[k]["F1"])
    best_m = grid[(best_th, best_nd)]

    if FINE_SWEEP:
        fine_thres = sorted({min(0.999, max(0.001, best_th + d)) for d in THRESH_FINE_DELTAS})
        fine_nms = sorted({max(4.0, best_nd + d) for d in NMS_FINE_DELTAS})
        fbest_th = fbest_nd = fbest_m = None
        for th in fine_thres:
            for nd in fine_nms:
                m = _score_at_threshold(preds, gts, sizes, th, nms_dist=nd)
                if fbest_m is None or m["F1"] > fbest_m["F1"]:
                    fbest_th, fbest_nd, fbest_m = th, nd, m
        print(f"    [精细搜索：围绕粗网格最优 thres={best_th:.2f} nms_dist={best_nd:.0f} 展开 "
              f"{len(fine_thres)}x{len(fine_nms)}={len(fine_thres) * len(fine_nms)} 组合] "
              f"最优 thres={fbest_th:.3f} nms_dist={fbest_nd:.1f}  "
              f"P={fbest_m['P']:.3f} R={fbest_m['R']:.3f} F1={fbest_m['F1']:.3f} "
              f"vtxMAE={fbest_m['vertex_mae']:.1f}px slpMAE={fbest_m['slope_mae']:.3f}")
        if fbest_m["F1"] > best_m["F1"]:
            best_th, best_nd, best_m = fbest_th, fbest_nd, fbest_m

    return best_th, best_nd, best_m


# ══════════════════════════════════════════════════════════════════════════
# 可视化：最优 (thres, nms_dist) 下的预测 vs GT
#
# gt_list/decode() 只携带 (x,y,slope)，没有 span/thickness（这两个不是回归
# 目标，见文件顶部说明）。画图时中心线只需要 (x,y,slope) 就能唯一确定，
# span 只决定画多长的两臂——GT 曲线用它自己标注的真实 span；预测曲线用固定常数
# VIS_PRED_SPAN，不借用同图 GT 的 span，避免"预测这一侧偷看了 GT"的嫌疑
# （只影响画多长，不影响曲线形状/预测数值本身，但干净地不碰 GT 更经得起推敲）。
# ══════════════════════════════════════════════════════════════════════════
def _centerline_points(x_vertex, y_vertex, slope, span, n_points=100):
    """双曲线中心线的画图采样点（numpy 版，公式与 y_center 一致，仅用于画图）。"""
    a = max(1.0, y_vertex)
    b = a / max(1e-3, slope)
    half = max(2.0, span) / 2.0
    xs = np.linspace(x_vertex - half, x_vertex + half, n_points)
    ys = y_vertex + a * (np.sqrt(1.0 + ((xs - x_vertex) / b) ** 2) - 1.0)
    return list(zip(xs.tolist(), ys.tolist()))


def _draw_vertex(draw, x, y, color, r=6):
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, 255), fill=color, width=2)


@torch.no_grad()
def visualize_predictions(model, subset, device, thres, nms_dist, out_dir, max_images=VIS_MAX_IMAGES):
    """在给定 (thres, nms_dist) 下跑推理，把 GT(绿)和预测(红)的双曲线中心线画在
    原图上对比，保存到 out_dir。subset 可以是 HyperbolaParamDataset 或它的 Subset
    （比如训练时的验证集 va），会自动定位到底层数据集读取原图和完整标注(含 span)。
    """
    base = subset.dataset if isinstance(subset, Subset) else subset
    order = subset.indices if isinstance(subset, Subset) else list(range(len(subset)))
    order = list(order)[:max_images]
    if not order:
        print("可视化：验证集为空，跳过。")
        return

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i in order:
        name, objs = base.items[i]
        orig = Image.open(os.path.join(base.img_dir, name)).convert("RGB")
        w0, h0 = orig.size

        img_in = orig.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img_in, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = model(img_t)[0].cpu()
        dets = nms(decode(pred, (w0, h0), thres=thres), dist_thres=nms_dist)

        vis = orig.copy()
        draw = ImageDraw.Draw(vis, "RGBA")

        for x, y, s, _score in dets:
            draw.line(_centerline_points(x, y, s, VIS_PRED_SPAN), fill=(255, 60, 60, 230), width=3)
            _draw_vertex(draw, x, y, (255, 60, 60, 255))

        for o in objs:
            draw.line(_centerline_points(o["x_vertex"], o["y_vertex"], o["slope"], o["span"]),
                      fill=(0, 220, 0, 230), width=3)
            _draw_vertex(draw, o["x_vertex"], o["y_vertex"], (0, 220, 0, 255))

        vis.save(os.path.join(out_dir, name))

    print(f"可视化对比图(绿=GT，红=预测，thres={thres:.3f} nms_dist={nms_dist:.1f})已保存到：{out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--json", default=None)
    ap.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--backbone", default=BACKBONE,
                     choices=["scratch", "resnet18", "resnet34", "resnet50", "resnet101"],
                     help="scratch=原来的零依赖轻量主干；resnet*=torchvision 预训练主干")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=PRETRAINED,
                     help="配合 --backbone resnet*：不加载 ImageNet 预训练权重（离线环境用）")
    args = ap.parse_args()

    json_path = args.json or os.path.join(args.img_dir, "annotations.json")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = HyperbolaParamDataset(args.img_dir, json_path)
    print(f"数据集：{len(full)} 张有标注图像  |  设备={device}  |  "
          f"backbone={args.backbone}(pretrained={args.pretrained if args.backbone != 'scratch' else 'N/A'})")
    if len(full) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    idx = list(range(len(full)))
    random.shuffle(idx)
    n_tr = int(len(idx) * args.train_frac)
    tr = Subset(full, idx[:n_tr])
    va = Subset(full, idx[n_tr:])

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = ParamRegressNet(backbone=args.backbone, pretrained=args.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"obj": 0, "vertex": 0, "slope": 0, "shape": 0}
        for imgs, targets, _, _ in tl:
            imgs, targets = imgs.to(device), targets.to(device)
            loss, parts = compute_loss(model(imgs), targets)
            opt.zero_grad(); loss.backward(); opt.step()
            for k in agg:
                agg[k] += parts[k]
        sch.step()
        n = max(1, len(tl))
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            m = evaluate(model, vl, device)
            print(f"[ep {ep:3d}] obj={agg['obj']/n:.3f} vtx={agg['vertex']/n:.3f} "
                  f"slp={agg['slope']/n:.3f} shp={agg['shape']/n:.3f} | val P={m['P']:.3f} R={m['R']:.3f} "
                  f"F1={m['F1']:.3f} vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    tag = datetime.now().strftime("hyp_param_%m%d_%H%M")

    if DO_SWEEP:
        print("阈值 × NMS 距离联合扫描（decode 期参数，缓存一次前向即可，无需重新训练）：")
        vis_th, vis_nms, best_m = sweep_thresholds(model, vl, device)
        print(f"  -> 最佳组合 thres={vis_th:.3f} nms_dist={vis_nms:.1f}  "
              f"P={best_m['P']:.3f} R={best_m['R']:.3f} "
              f"F1={best_m['F1']:.3f} vtxMAE={best_m['vertex_mae']:.1f}px slpMAE={best_m['slope_mae']:.3f}")
    else:
        vis_th, vis_nms = OBJ_THRES, NMS_DIST
        m = evaluate(model, vl, device, thres=vis_th, nms_dist=vis_nms)
        print(f"DO_SWEEP=False，直接用默认 OBJ_THRES={OBJ_THRES} / NMS_DIST={NMS_DIST} 评估：")
        print(f"  P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
              f"vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    if DO_VISUALIZE:
        vis_dir = os.path.join(_HERE, f"vis_{tag}")
        visualize_predictions(model, va, device, vis_th, vis_nms, vis_dir)

    out = os.path.join(_HERE, tag + ".pth")
    torch.save(model.state_dict(), out)
    print("已保存模型：", out)


if __name__ == "__main__":
    main()
