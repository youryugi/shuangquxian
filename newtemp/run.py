"""
run.py — 双曲线带检测：消融实验框架
========================================================
修改 CONFIG（固定参数）和 ABLATION_CONFIGS（实验组），然后运行。
每组实验结果保存在 BASE_OUTPUT_DIR/<name>/ 下，
最终汇总表格输出到控制台并保存为 ablation_results.json。
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — 所有实验共享的固定参数
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR        = r"C:\Users\79152\Desktop\github\shuangquxian\biaozhumore\Utilities"
BASE_OUTPUT_DIR = "./output"

IMG_SIZE    = 224
BATCH_SIZE  = 4
EPOCHS      = 50
LR          = 1e-4
TEST_RATIO  = 0.2
SEED        = 42
NUM_WORKERS = 0

# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION_CONFIGS — 每一行是一组实验，修改想要对比的参数即可
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    # ── 基线 ──────────────────────────────────────────────────────────────────
    {"name": "baseline",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    # ── 损失权重消融 ───────────────────────────────────────────────────────────
    {"name": "no_sym",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 0.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    {"name": "no_iou",
     "W_PARAM": 1.0, "W_IOU": 0.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    {"name": "iou_weight_4",
     "W_PARAM": 1.0, "W_IOU": 4.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    # ── 迭代次数消融 ───────────────────────────────────────────────────────────
    {"name": "iter_0",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 0,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    {"name": "iter_1",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 1,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    {"name": "iter_5",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 5,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 20.},

    # ── 预训练消融 ─────────────────────────────────────────────────────────────
    {"name": "no_pretrain",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": False, "scale": 0.1, "sharpness": 20.},

    # ── 精化步长消融 ───────────────────────────────────────────────────────────
    {"name": "scale_0p05",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.05, "sharpness": 20.},

    {"name": "scale_0p2",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.2, "sharpness": 20.},

    # ── 软掩码锐度消融 ─────────────────────────────────────────────────────────
    {"name": "sharpness_10",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 10.},

    {"name": "sharpness_40",
     "W_PARAM": 1.0, "W_IOU": 2.0, "W_SYM": 1.0,
     "N_ITERS": 3,   "N_ARC": 32,  "N_NORMAL": 8,
     "PRETRAINED": True, "scale": 0.1, "sharpness": 40.},
]

# ══════════════════════════════════════════════════════════════════════════════
#  以下无需修改
# ══════════════════════════════════════════════════════════════════════════════

import os, json, math, random, time, warnings
import numpy as np
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as tvm
import torchvision.transforms.functional as TF

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. 数据集
# ══════════════════════════════════════════════════════════════════════════════

def hyperbola_mask(x_v, y_v, w, h, t, img_w, img_h):
    mask   = np.zeros((img_h, img_w), dtype=np.float32)
    half_w = max(w / 2.0, 1.0)
    half_t = t / 2.0
    x_left  = max(0, int(x_v - half_w))
    x_right = min(img_w - 1, int(x_v + half_w))
    for x in range(x_left, x_right + 1):
        dx      = (x - x_v) / half_w
        y_c     = y_v + h * (dx ** 2)
        y_top   = max(0,        int(math.floor(y_c - half_t)))
        y_bot   = min(img_h-1,  int(math.ceil (y_c + half_t)))
        if y_top <= y_bot:
            mask[y_top:y_bot+1, x] = 1.0
    return mask


def normalize_params(x_v, y_v, w, h, t, img_w, img_h):
    return np.array([x_v/img_w, y_v/img_h, w/img_w, h/img_h, t/img_h], dtype=np.float32)


def denormalize_params(p, img_size):
    return (p[0]*img_size, p[1]*img_size, p[2]*img_size, p[3]*img_size, p[4]*img_size)


class HyperbolaDataset(Dataset):
    def __init__(self, data_dir, img_size, augment=False):
        self.img_size = img_size
        self.augment  = augment
        ann_path = os.path.join(data_dir, "annotations.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"annotations.json not found in {data_dir}")
        with open(ann_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.samples = []
        for fname, objs in raw.items():
            if not objs:
                continue
            img_path = os.path.join(data_dir, fname)
            if not os.path.exists(img_path):
                continue
            for obj in objs:
                self.samples.append((img_path, obj))
        if not self.samples:
            raise RuntimeError("Dataset is empty — check annotations.json and image paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, obj = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        ow, oh = image.size

        x_v = float(obj["x_vertex"])
        y_v = float(obj["y_vertex"])
        w   = float(obj["width"])
        h   = float(obj["height"])
        t   = float(obj["thickness"])

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        sx, sy = self.img_size / ow, self.img_size / oh
        x_v *= sx;  y_v *= sy
        w   *= sx;  h   *= sy;  t *= sy

        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            x_v   = self.img_size - x_v

        mask   = torch.from_numpy(hyperbola_mask(x_v, y_v, w, h, t, self.img_size, self.img_size))
        params = torch.from_numpy(normalize_params(x_v, y_v, w, h, t, self.img_size, self.img_size))
        image  = TF.to_tensor(image)

        return {"image": image, "params": params, "mask": mask, "path": img_path}


# ══════════════════════════════════════════════════════════════════════════════
#  2. 模型
# ══════════════════════════════════════════════════════════════════════════════

class CurveAlignedSampler(nn.Module):
    def __init__(self, n_arc=32, n_normal=8):
        super().__init__()
        self.n_arc    = n_arc
        self.n_normal = n_normal

    def forward(self, feat, params):
        B, C, H, W = feat.shape
        N, M = self.n_arc, self.n_normal
        x_v, y_v = params[:,0], params[:,1]
        w,   h   = params[:,2], params[:,3]
        t        = params[:,4]

        s = torch.linspace(-1., 1., N, device=feat.device)
        n = torch.linspace(-1., 1., M, device=feat.device)

        x_arc = x_v[:,None] + s[None,:] * (w[:,None] / 2.)
        y_arc = y_v[:,None] + h[:,None] * (s[None,:] ** 2)

        y_off = (t[:,None,None] / 2.) * n[None,None,:]

        x_grid = x_arc[:,:,None].expand(B, N, M)
        y_grid = y_arc[:,:,None] + y_off.expand(B, N, M)

        grid = torch.stack([x_grid*2-1, y_grid*2-1], dim=-1)

        return F.grid_sample(feat, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


class ArcAttention(nn.Module):
    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        xt = x.permute(0, 2, 1)
        out, _ = self.attn(xt, xt, xt)
        return self.norm(xt + out).permute(0, 2, 1)


class InitHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(in_ch * 64, 256), nn.ReLU(True),
            nn.Linear(256, 64),          nn.ReLU(True),
            nn.Linear(64, 5),            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


class RefineHead(nn.Module):
    def __init__(self, in_ch, scale=0.1):
        super().__init__()
        mid = 128
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid, 3, padding=1), nn.BatchNorm1d(mid), nn.ReLU(True),
            nn.Conv1d(mid,   mid, 3, padding=1), nn.BatchNorm1d(mid), nn.ReLU(True),
        )
        self.attn  = ArcAttention(mid, n_heads=4)
        self.head  = nn.Sequential(
            nn.Linear(mid, 64), nn.ReLU(True),
            nn.Linear(64, 5),   nn.Tanh(),
        )
        self.scale = scale

    def forward(self, strip):
        x = strip.max(dim=-1).values
        x = self.conv(x)
        x = self.attn(x)
        x = x.mean(dim=-1)
        return self.head(x) * self.scale


class HyperbolaNet(nn.Module):
    def __init__(self, n_iters=3, n_arc=32, n_normal=8, pretrained=True, scale=0.1):
        super().__init__()
        self.n_iters = n_iters
        bb = tvm.resnet18(
            weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3,
        )
        self.init_head   = InitHead(256)
        self.sampler     = CurveAlignedSampler(n_arc, n_normal)
        self.refine_head = RefineHead(256, scale=scale)

    def forward(self, image):
        feat   = self.backbone(image)
        params = self.init_head(feat)
        init_p = params
        iters  = []
        for _ in range(self.n_iters):
            strip  = self.sampler(feat, params)
            params = (params + self.refine_head(strip)).clamp(0., 1.)
            iters.append(params)
        return {"init": init_p, "iters": iters, "final": params}


# ══════════════════════════════════════════════════════════════════════════════
#  3. 损失
# ══════════════════════════════════════════════════════════════════════════════

def render_soft_mask(params, img_size, sharpness=20.):
    B  = params.shape[0]
    H  = W = img_size
    dv = params.device
    x_v = params[:,0]*W;  y_v = params[:,1]*H
    w   = params[:,2]*W;  h   = params[:,3]*H;  t = params[:,4]*H

    px = torch.arange(W, device=dv, dtype=torch.float32).view(1,1,W)
    py = torch.arange(H, device=dv, dtype=torch.float32).view(1,H,1)

    half_w  = (w/2.).clamp(min=1.).view(B,1,1)
    dx      = (px - x_v.view(B,1,1)) / half_w
    y_c     = y_v.view(B,1,1) + h.view(B,1,1) * (dx**2)
    dist    = (py - y_c).abs() - (t/2.).view(B,1,1)
    soft    = torch.sigmoid(-dist * sharpness)

    in_rng  = ((px >= (x_v - w/2.).view(B,1,1)) &
                (px <= (x_v + w/2.).view(B,1,1))).float()
    return soft * in_rng


def soft_iou_loss(pred_params, gt_mask, img_size, sharpness=20.):
    pred = render_soft_mask(pred_params, img_size, sharpness).view(pred_params.shape[0], -1)
    gt   = gt_mask.to(pred.device).view(gt_mask.shape[0], -1)
    inter = (pred * gt).sum(1)
    union = (pred + gt - pred * gt).sum(1)
    return (1. - inter / (union + 1e-6)).mean()


def param_l2(pred, gt):
    w = torch.tensor([2.,2.,1.,1.,1.], device=pred.device)
    return ((pred - gt)**2 * w).mean()


def sym_loss(strip):
    B, C, N, M = strip.shape
    mid = N // 2
    l   = strip[:,:,:mid,:].reshape(B,-1)
    r   = strip[:,:,N-mid:,:].flip(2).reshape(B,-1)
    return (1. - F.cosine_similarity(l, r, dim=1)).mean()


def compute_loss(outputs, gt_params, gt_mask, img_size, strip=None,
                 w_param=1.0, w_iou=2.0, w_sym=1.0, sharpness=20.):
    n    = len(outputs["iters"])
    loss = param_l2(outputs["init"], gt_params)
    for i, p in enumerate(outputs["iters"]):
        loss = loss + ((i+1)/n) * param_l2(p, gt_params)
    loss = loss * w_param
    loss = loss + w_iou  * soft_iou_loss(outputs["final"], gt_mask, img_size, sharpness)
    loss = loss + w_param * param_l2(outputs["final"], gt_params)
    if strip is not None and w_sym > 0:
        loss = loss + w_sym * sym_loss(strip)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
#  4. 训练 / 评估
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, epoch, cfg):
    model.train()
    total, n = 0., 0
    for step, batch in enumerate(loader):
        images    = batch["image"].to(DEVICE)
        gt_params = batch["params"].to(DEVICE)
        gt_mask   = batch["mask"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        with torch.no_grad():
            feat = model.backbone(images)
        strip = model.sampler(feat, outputs["final"].detach())

        loss = compute_loss(
            outputs, gt_params, gt_mask, IMG_SIZE, strip,
            w_param=cfg["W_PARAM"], w_iou=cfg["W_IOU"],
            w_sym=cfg["W_SYM"],     sharpness=cfg["sharpness"],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        bs     = images.shape[0]
        total += loss.item() * bs
        n     += bs
        if (step+1) % max(1, len(loader)//3) == 0:
            print(f"  [{cfg['name']}] Epoch {epoch} [{step+1}/{len(loader)}] loss={loss.item():.4f}")

    return total / n


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total_loss, total_iou, n = 0., 0., 0
    all_errs = {k: [] for k in ["x_vertex","y_vertex","width","height","thickness"]}

    for batch in loader:
        images    = batch["image"].to(DEVICE)
        gt_params = batch["params"].to(DEVICE)
        gt_mask   = batch["mask"].to(DEVICE)

        outputs = model(images)
        loss    = compute_loss(
            outputs, gt_params, gt_mask, IMG_SIZE,
            w_param=cfg["W_PARAM"], w_iou=cfg["W_IOU"],
            w_sym=cfg["W_SYM"],     sharpness=cfg["sharpness"],
        )
        bs = images.shape[0]
        total_loss += loss.item() * bs
        n          += bs

        for b in range(bs):
            fp = outputs["final"][b].cpu().numpy()
            gp = gt_params[b].cpu().numpy()
            fx_v,fy_v,fw,fh,ft  = denormalize_params(fp, IMG_SIZE)
            gx_v,gy_v,gw,gh,gt_ = denormalize_params(gp, IMG_SIZE)
            pm = hyperbola_mask(fx_v,fy_v,fw,fh,ft,  IMG_SIZE,IMG_SIZE)
            gm = hyperbola_mask(gx_v,gy_v,gw,gh,gt_, IMG_SIZE,IMG_SIZE)
            inter = (pm*gm).sum();  union = (pm+gm-pm*gm).sum()
            total_iou += float(inter/(union+1e-6))
            for ki,(pv,gv) in enumerate(zip(
                [fx_v,fy_v,fw,fh,ft],[gx_v,gy_v,gw,gh,gt_]
            )):
                all_errs[list(all_errs.keys())[ki]].append(abs(pv-gv))

    return {
        "loss": total_loss/n,
        "iou":  total_iou/n,
        "param_mae": {k: float(np.mean(v)) for k,v in all_errs.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. 可视化
# ══════════════════════════════════════════════════════════════════════════════

def draw_band(draw, x_v, y_v, w, h, t, color, n_pts=120):
    half_w = max(w/2., 1.)
    upper, lower = [], []
    for i in range(n_pts+1):
        x  = (x_v - half_w) + (w/n_pts)*i
        dx = (x - x_v) / half_w
        yc = y_v + h*(dx**2)
        upper.append((x, yc - t/2.))
        lower.append((x, yc + t/2.))
    pts = upper + list(reversed(lower))
    if len(pts) >= 3:
        draw.polygon(pts, fill=(*color, 25), outline=(*color, 120))
    center = [(x, y_v+h*(((x-x_v)/half_w)**2)) for x,_ in upper]
    if len(center) >= 2:
        draw.line(center, fill=(*color, 150), width=2)
    r=5; draw.ellipse((x_v-r,y_v-r,x_v+r,y_v+r), fill=(255,255,0,255))


@torch.no_grad()
def save_vis(model, loader, epoch, output_dir, max_imgs=8):
    model.eval()
    saved = 0
    colors_iter = [(255,100,100),(255,180,50),(100,220,100),(80,180,255)]
    labels_iter = ["Init","Iter1","Iter2","Final"]

    for batch in loader:
        images    = batch["image"].to(DEVICE)
        gt_params = batch["params"]
        paths     = batch["path"]
        outputs   = model(images)

        for b in range(images.shape[0]):
            if saved >= max_imgs:
                return
            orig = Image.open(paths[b]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            orig_rgba = orig.convert("RGBA")

            # 右图：迭代过程 — 在透明图层上画，再合并到原图
            overlay_r = Image.new("RGBA", orig_rgba.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(overlay_r, "RGBA")
            all_p = ([outputs["init"][b].cpu().numpy()]
                     + [p[b].cpu().numpy() for p in outputs["iters"]])
            for i, p in enumerate(all_p):
                xv,yv,w,h,t = denormalize_params(p, IMG_SIZE)
                c = colors_iter[min(i, len(colors_iter)-1)]
                dr.text((8, 8+i*18), f"{labels_iter[min(i,3)]}: xv={xv:.0f} yv={yv:.0f}", fill=(*c,255))
                draw_band(dr, xv, yv, w, h, t, c)
            right = Image.alpha_composite(orig_rgba, overlay_r)

            # 左图：最终预测 vs GT — 同样用独立图层
            overlay_l = Image.new("RGBA", orig_rgba.size, (0, 0, 0, 0))
            dl = ImageDraw.Draw(overlay_l, "RGBA")
            fp   = outputs["final"][b].cpu().numpy()
            gp   = gt_params[b].numpy()
            xv,yv,w,h,t        = denormalize_params(fp, IMG_SIZE)
            gxv,gyv,gw,gh,gt_  = denormalize_params(gp, IMG_SIZE)
            draw_band(dl, xv, yv, w, h, t,   (0,230,80))
            draw_band(dl, gxv,gyv,gw,gh,gt_, (80,140,255))
            dl.text((8, IMG_SIZE-22), "Green=Pred  Blue=GT", fill=(255,255,255,220))
            left = Image.alpha_composite(orig_rgba, overlay_l)

            canvas = Image.new("RGB", (IMG_SIZE*2, IMG_SIZE), (25,25,25))
            canvas.paste(left.convert("RGB"),  (0,0))
            canvas.paste(right.convert("RGB"), (IMG_SIZE,0))
            canvas.save(os.path.join(output_dir, "vis", f"epoch{epoch:03d}_{saved:02d}.png"))
            saved += 1


# ══════════════════════════════════════════════════════════════════════════════
#  6. 单次实验
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg, train_loader, test_loader):
    name    = cfg["name"]
    out_dir = os.path.join(BASE_OUTPUT_DIR, name)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    model = HyperbolaNet(
        n_iters=cfg["N_ITERS"], n_arc=cfg["N_ARC"], n_normal=cfg["N_NORMAL"],
        pretrained=cfg["PRETRAINED"], scale=cfg["scale"],
    ).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total/1e6:.2f}M params")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_iou = 0.
    history  = []

    for epoch in range(1, EPOCHS+1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, epoch, cfg)
        scheduler.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            metrics = evaluate(model, test_loader, cfg)
            elapsed = time.time() - t0
            print(f"  [{name}] Epoch {epoch}/{EPOCHS} ({elapsed:.1f}s)")
            print(f"    Train loss: {train_loss:.4f}  Test loss: {metrics['loss']:.4f}  IoU: {metrics['iou']:.4f}")
            print(f"    Param MAE : " + "  ".join(f"{k}={v:.1f}px" for k,v in metrics["param_mae"].items()))

            save_vis(model, test_loader, epoch, out_dir)

            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "iou": best_iou, "cfg": cfg,
                }, os.path.join(out_dir, "best.pth"))
                print(f"    ✓ Best model saved (IoU={best_iou:.4f})")

            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
        else:
            elapsed = time.time() - t0
            print(f"  [{name}] Epoch {epoch}/{EPOCHS} ({elapsed:.1f}s)  train_loss={train_loss:.4f}")

    # 加载最优权重做最终评估
    best_ckpt = torch.load(os.path.join(out_dir, "best.pth"), map_location=DEVICE)
    model.load_state_dict(best_ckpt["model"])
    final = evaluate(model, test_loader, cfg)

    return {
        "name":       name,
        "best_iou":   best_iou,
        "final_iou":  final["iou"],
        "final_loss": final["loss"],
        "param_mae":  final["param_mae"],
        "cfg":        cfg,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  7. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 数据（所有实验共享同一份划分）──
    full_ds = HyperbolaDataset(DATA_DIR, IMG_SIZE, augment=False)
    n       = len(full_ds)
    idx     = list(range(n))
    random.shuffle(idx)
    n_test  = max(1, int(n * TEST_RATIO))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_ds = HyperbolaDataset(DATA_DIR, IMG_SIZE, augment=True)
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type=="cuda",
        drop_last=len(train_idx) >= BATCH_SIZE,
    )
    test_loader = DataLoader(
        Subset(full_ds, test_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type=="cuda",
    )
    print(f"Train: {len(train_idx)} samples  |  Test: {len(test_idx)} samples")
    print(f"Running {len(ABLATION_CONFIGS)} experiments\n")

    # ── 逐组实验 ──
    all_results = []
    for cfg in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Experiment: {cfg['name']}")
        print(f"  W_PARAM={cfg['W_PARAM']}  W_IOU={cfg['W_IOU']}  W_SYM={cfg['W_SYM']}")
        print(f"  N_ITERS={cfg['N_ITERS']}  N_ARC={cfg['N_ARC']}  N_NORMAL={cfg['N_NORMAL']}")
        print(f"  PRETRAINED={cfg['PRETRAINED']}  scale={cfg['scale']}  sharpness={cfg['sharpness']}")
        print(f"{'='*60}")
        result = run_experiment(cfg, train_loader, test_loader)
        all_results.append(result)

    # ── 汇总对比表 ──
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    hdr = f"{'Experiment':<18} {'IoU':>7} {'Loss':>8} {'xv_MAE':>8} {'yv_MAE':>8} {'w_MAE':>8} {'h_MAE':>8} {'t_MAE':>8}"
    print(hdr)
    print("-"*90)
    for r in all_results:
        m = r["param_mae"]
        print(f"{r['name']:<18} {r['final_iou']:>7.4f} {r['final_loss']:>8.4f}"
              f" {m['x_vertex']:>8.2f} {m['y_vertex']:>8.2f}"
              f" {m['width']:>8.2f} {m['height']:>8.2f} {m['thickness']:>8.2f}")

    # ── 保存汇总 ──
    summary_path = os.path.join(BASE_OUTPUT_DIR, "ablation_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
    print(f"Per-experiment files in: {BASE_OUTPUT_DIR}/<name>/")


if __name__ == "__main__":
    main()
