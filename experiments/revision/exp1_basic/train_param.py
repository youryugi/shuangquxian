"""第 17 轮：参数化双曲线检测头——网络直接输出 HBAT 的双曲线参数，框由参数闭式算出。

    python train_param.py --data <ds2_f*_s100_hbatbox> --head param  --seed 0 --runs runs_cv --fold-in-path --exp hbatbox_augfix_param
    python train_param.py --data ... --head pseudo ...
    python train_param.py --data ... --head box ...        # 对照：原 baseline，只修了翻转

动机（README 第 17 轮）：HBAT 标注本身就是每个目标 5 个参数（x_vertex, y_vertex, slope, span, thickness；tail 在数据集 2 恒为 0），
band 和 HBAT 框都是从参数画出来的：
    y_c(x) = y_v + a·(sqrt(1 + ((x − x_v)/b)²) − 1),  a = max(1, |y_v|),  b = a / slope
    框 = [x_v − span/2, y_v − thickness/2, x_v + span/2, y_c(x_v ± span/2) + thickness/2]，裁到图内（span 先按图底截断）
所以让网络直接回归参数，而不是回归"框中心 + 宽高"。

三个 head（backbone、热图头、offset 头完全相同，配对初始化与 train_basic.py 相同）：
  box     原 baseline：热图峰 = 框中心，回归 (w, h)。和 hbatbox_aug_none 唯一的区别是翻转增强修好了（见下）
  param   热图峰 = 双曲线顶点，回归 (log slope, span, thickness)；框 = 上式。目标来自 HBAT 标注的真实参数
  pseudo  结构、损失、初始化与 param 逐项相同；目标参数从 HBAT 框反推（顶点 = 框上边中点 + T0/2，span = 框宽，
          thickness 固定 T0 = 16 px，slope 由框高解出）——能完全复原同一个框，但不含 band 标注里的几何信息
          = "参数化输出本身"的对照，只用框标注
param / pseudo 的损失：focal(热图) + L1(offset) + L1(参数) + GIoU(由预测参数算出的框, GT 框)。
无参数的目标（param_status unparam，数据集 2 有 6 个，它们的框来自官方框）在 param 里也用反推参数。

翻转增强（train_basic.py 的 augment_sample 有 bug）：它把热图 / offset 当数组翻转、x offset 取负，等于把目标中心放在 27 − fx 格，
而图上真实位置是 28 − fx 格——每个翻转样本的目标都向左偏 8 px（fliptest：翻转样本中心误差均值 −8.00 px，未翻转 0.00）。
第 4–16 轮所有带 --augment 的 run 都受影响（预测框中心平均偏左 2–4 px）。这里先在几何上翻转（x → W − x），再画目标。
翻转的随机数用法与 train_basic.py 相同（worker 内 random.random() < 0.5）。

输出与 train_basic.py 相同（run_info.json / preds_{val,test}.json / metrics*.json），aggregate_cv.py 直接可用；
另存 preds_{val,test}_params.json：每个检测的 [x_v, y_v, slope, span, thickness, score]（box head 没有）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import hbat  # noqa: E402
import train_basic as tb  # noqa: E402
from train_basic import evaluate_run  # noqa: E402

INPUT = 224
STRIDE = 8
G = INPUT // STRIDE          # 28
SIGMA = 6                    # 热图高斯 sigma（格），与 train_basic.py 相同
T0 = 16.0                    # pseudo 的固定 thickness（px）；数据集 2 标注厚度中位数 16.3
LOGS_SCALE = 4.0             # 参数向量里 slope 存成 log(slope) / 4，和 span/224、thickness/224 同一量级
LAM_GIOU = 1.0
OFF_CLIP_PARAM = 2.0         # 顶点可在图外（数据集 2 有 7 个），解码时 offset 放宽到 ±2 格；box head 与原样一样 ±0.5
device = tb.device


# ───────────────────────── 几何 ─────────────────────────
def pseudo_params(box):
    """HBAT 框 [x1,y1,x2,y2] -> (x_v, y_v, slope, span, thickness)，代回 derive_box 得到同一个框（slope 落在 [0.2, 20] 内时）。"""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    th = min(T0, 0.5 * h)
    yv = y1 + th / 2.0
    drop = h - th
    a = max(1.0, abs(yv))
    half = w / 2.0
    slope = (a / half) * math.sqrt(max((1.0 + drop / a) ** 2 - 1.0, 1e-8))
    return (x1 + x2) / 2.0, yv, float(np.clip(slope, 0.2, 20.0)), w, th


def derive_box_np(xv, yv, slope, span, th):
    a = max(1.0, abs(yv)); half = span / 2.0
    ye = yv + a * (math.sqrt(1.0 + (half * slope / a) ** 2) - 1.0)
    return [min(max(xv - half, 0.0), INPUT), min(max(yv - th / 2.0, 0.0), INPUT),
            min(max(xv + half, 0.0), INPUT), min(max(ye + th / 2.0, 0.0), INPUT)]


def derive_box_t(xv, yv, logs4, span_n, th_n):
    """torch 版（可导）：输入同形张量，返回 (..., 4) 的像素框，裁到图内。"""
    slope = torch.exp(logs4 * LOGS_SCALE)
    span, th = span_n * INPUT, th_n * INPUT
    a = yv.abs().clamp(min=1.0)
    half = span / 2.0
    ye = yv + a * (torch.sqrt(1.0 + (half * slope / a) ** 2) - 1.0)
    box = torch.stack([xv - half, yv - th / 2.0, xv + half, ye + th / 2.0], dim=-1)
    return box.clamp(0.0, float(INPUT))


def giou_loss(p, g):
    """p, g: (N, 4) 像素框。"""
    ix1 = torch.maximum(p[:, 0], g[:, 0]); iy1 = torch.maximum(p[:, 1], g[:, 1])
    ix2 = torch.minimum(p[:, 2], g[:, 2]); iy2 = torch.minimum(p[:, 3], g[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ap = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = ap + ag - inter + 1e-6
    cx1 = torch.minimum(p[:, 0], g[:, 0]); cy1 = torch.minimum(p[:, 1], g[:, 1])
    cx2 = torch.maximum(p[:, 2], g[:, 2]); cy2 = torch.maximum(p[:, 3], g[:, 3])
    c = (cx2 - cx1) * (cy2 - cy1) + 1e-6
    return (1.0 - (inter / union - (c - union) / c)).mean()


# ───────────────────────── 数据 ─────────────────────────
class ParamDataset(Dataset):
    """每张图的目标 = gt_boxes.json 的 HBAT 框；每个框配上 HBAT 参数（派生框与之逐坐标相同的那个对象），没有则反推。"""

    def __init__(self, data_dir, phase, head, augment=False):
        self.data_dir, self.phase, self.head, self.augment = Path(data_dir), phase, head, augment
        man = json.load(open(self.data_dir / "manifest.json", encoding="utf-8"))
        assert man.get("box_source") == "hbat", "第 17 轮只在 HBAT 框数据上跑"
        ann = json.load(open(man["hbat_json"], encoding="utf-8"))
        self.gt = json.load(open(self.data_dir / "gt_boxes.json", encoding="utf-8"))[phase]
        self.stems = sorted(self.gt)
        img_dir = self.data_dir / "images" / phase
        self.files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
        self.inst = {}
        self.n_real = self.n_fallback = 0
        for s in self.stems:
            r = self.gt[s]
            assert (r["width"], r["height"]) == (INPUT, INPUT), "数据集 2 全是 224×224；别的尺寸要先加缩放"
            objs = [o for o in ann[r["band_key"]] if o.get("x_vertex") is not None]
            cands = []
            for o in objs:
                n = hbat.normalize_obj(o)
                if not hbat.has_band(n):
                    continue
                c = hbat.cap_span_to_bottom(n, (INPUT, INPUT))
                cands.append(((c["x_vertex"], c["y_vertex"], c["slope"], c["span"], c["thickness"]),
                              derive_box_np(c["x_vertex"], c["y_vertex"], c["slope"], c["span"], c["thickness"])))
            lst = []
            for box in r["boxes"]:
                real = next((p for p, b in cands if max(abs(b[i] - box[i]) for i in range(4)) < 0.05), None)
                if real is not None:
                    self.n_real += 1
                else:
                    self.n_fallback += 1
                ps = pseudo_params(box)
                prm = real if (head == "param" and real is not None) else ps
                lst.append({"box": list(box), "params": prm, "real": real})
            self.inst[s] = lst

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = np.array(Image.open(self.files[stem]).convert("L"), dtype=np.float32) / 255.0
        flip = self.augment and random.random() < 0.5
        if flip:
            img = img[:, ::-1].copy()
        hm = np.zeros((G, G), np.float32)
        peak = np.zeros((G, G), np.float32)
        off = np.zeros((2, G, G), np.float32)
        size = np.zeros((3 if self.head != "box" else 2, G, G), np.float32)
        gbox = np.zeros((4, G, G), np.float32)
        for it in self.inst[stem]:
            x1, y1, x2, y2 = it["box"]
            xv, yv, slope, span, th = it["params"]
            if flip:
                x1, x2 = INPUT - x2, INPUT - x1
                xv = INPUT - xv
            if self.head == "box":
                kx, ky = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            else:
                kx, ky = xv, yv
            fx, fy = kx / STRIDE, ky / STRIDE
            xi = int(np.clip(round(fx), 0, G - 1)); yi = int(np.clip(round(fy), 0, G - 1))
            tb.render_gaussian(hm, min(max(fx, 0.0), G - 1), min(max(fy, 0.0), G - 1), SIGMA)
            peak[yi, xi] = 1.0
            off[0, yi, xi] = fx - xi; off[1, yi, xi] = fy - yi
            if self.head == "box":
                size[0, yi, xi] = (x2 - x1) / INPUT; size[1, yi, xi] = (y2 - y1) / INPUT
            else:
                size[0, yi, xi] = math.log(slope) / LOGS_SCALE
                size[1, yi, xi] = span / INPUT
                size[2, yi, xi] = th / INPUT
            gbox[:, yi, xi] = (x1, y1, x2, y2)
        meta = {"stem": stem}
        return (torch.from_numpy(img).unsqueeze(0), torch.from_numpy(hm).unsqueeze(0), torch.from_numpy(peak).unsqueeze(0),
                torch.from_numpy(off), torch.from_numpy(size), torch.from_numpy(gbox), meta)


def collate(batch):
    *ts, metas = zip(*batch)
    return [torch.stack(t) for t in ts] + [list(metas)]


# ───────────────────────── 模型 ─────────────────────────
def build_model(head, seed):
    """backbone + 热图头 + offset 头 + (box: wh 头) 用 seed 初始化——与 train_basic.build_model(use_attn=False) 逐参数相同；
    param / pseudo 的参数头（3 通道）用 seed + 10000 初始化，两者逐参数相同。"""
    torch.manual_seed(seed)
    net = tb.AttnBBoxNet(in_ch=1, base_ch=tb.BASE_CH, use_attn=False)
    if head != "box":
        torch.manual_seed(seed + 10000)
        mid = tb.BASE_CH * 8
        net.wh_head = nn.Sequential(nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False), nn.BatchNorm2d(mid // 2),
                                    nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 3, 1))
    return net.to(device)


def _features(net, x):
    f1, x = net.down1(x); f2, x = net.down2(x); f3, x = net.down3(x)
    return net.bottleneck(x)


def run_heads(net, img, head):
    """返回 hm_logit, size（box: sigmoid(w, h)；param/pseudo: [log(slope)/4 原样, sigmoid span, sigmoid th]）, offset。"""
    feat = _features(net, img)
    hm_logit = net.heatmap_head(feat)
    off = net.offset_head(feat)
    raw = net.wh_head(feat)
    if head == "box":
        size = torch.sigmoid(raw)
    else:
        size = torch.cat([raw[:, :1], torch.sigmoid(raw[:, 1:])], dim=1)
    return hm_logit, size, off


def grid_xy(B, dev):
    ys, xs = torch.meshgrid(torch.arange(G, device=dev, dtype=torch.float32),
                            torch.arange(G, device=dev, dtype=torch.float32), indexing="ij")
    return xs.expand(B, G, G), ys.expand(B, G, G)


def compute_loss(net, batch, head):
    img, hm, peak, off, size, gbox = batch
    hm_logit, size_p, off_p = run_heads(net, img, head)
    hm_logit, size_p, off_p = hm_logit.float(), size_p.float(), off_p.float()
    l_hm = tb.focal_loss_heatmap(hm_logit, hm, peak)
    l_off = tb.masked_l1(off_p, off, peak)
    l_size = tb.masked_l1(size_p, size, peak)
    if head == "box":
        return l_hm + l_size + l_off, {"hm": l_hm.item(), "size": l_size.item(), "off": l_off.item(), "giou": 0.0}
    m = peak[:, 0] > 0.5
    if m.any():
        gx, gy = grid_xy(img.shape[0], img.device)
        xv = (gx + off_p[:, 0]) * STRIDE; yv = (gy + off_p[:, 1]) * STRIDE
        pb = derive_box_t(xv[m], yv[m], size_p[:, 0][m], size_p[:, 1][m], size_p[:, 2][m])
        gb = gbox.permute(0, 2, 3, 1)[m]
        l_giou = giou_loss(pb, gb)
    else:
        l_giou = hm_logit.sum() * 0.0
    loss = l_hm + l_off + l_size + LAM_GIOU * l_giou
    return loss, {"hm": l_hm.item(), "size": l_size.item(), "off": l_off.item(), "giou": l_giou.item()}


# ───────────────────────── 推理 ─────────────────────────
@torch.no_grad()
def predict_all(net, ds, head, thresh=0.001, max_det=50):
    net.eval()
    preds, params = {}, {}
    for i in range(len(ds)):
        img = ds[i][0].unsqueeze(0).to(device)
        stem = ds.stems[i]
        hm_logit, size_p, off_p = run_heads(net, img, head)
        hm = torch.sigmoid(hm_logit[0, 0]).float().cpu().numpy()
        hm_nms = tb._heatmap_nms(hm, tb.NMS_KERNEL)
        ys, xs = np.where(hm_nms >= thresh)
        sc = hm_nms[ys, xs]; order = np.argsort(-sc, kind="stable")[:max_det]
        ys, xs, sc = ys[order], xs[order], sc[order]
        sz, of = size_p[0].float().cpu().numpy(), off_p[0].float().cpu().numpy()
        dets, prm = [], []
        for yi, xi, s in zip(ys, xs, sc):
            if head == "box":
                bw, bh = float(sz[0, yi, xi]) * INPUT, float(sz[1, yi, xi]) * INPUT
                cx = (xi + float(np.clip(of[0, yi, xi], -0.5, 0.5))) * STRIDE
                cy = (yi + float(np.clip(of[1, yi, xi], -0.5, 0.5))) * STRIDE
                dets.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(s)])
            else:
                xv = (xi + float(np.clip(of[0, yi, xi], -OFF_CLIP_PARAM, OFF_CLIP_PARAM))) * STRIDE
                yv = (yi + float(np.clip(of[1, yi, xi], -OFF_CLIP_PARAM, OFF_CLIP_PARAM))) * STRIDE
                slope = math.exp(float(sz[0, yi, xi]) * LOGS_SCALE)
                span, th = float(sz[1, yi, xi]) * INPUT, float(sz[2, yi, xi]) * INPUT
                b = derive_box_np(xv, yv, slope, span, th)
                if b[2] - b[0] < 1.0 or b[3] - b[1] < 1.0:
                    continue
                dets.append(b + [float(s)])
                prm.append([xv, yv, slope, span, th, float(s)])
        preds[stem] = dets
        params[stem] = prm
    return preds, params


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", required=True, choices=["box", "param", "pseudo"])
    ap.add_argument("--exp", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=tb.EPOCHS)
    ap.add_argument("--no-augment", action="store_true", help="关掉（修好的）水平翻转")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--fold-in-path", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="DataLoader worker 数。翻转随机数按 worker 播种：要和旧 hbatbox_aug_* run（全部 workers=1）逐样本配对就必须是 1")
    args = ap.parse_args()

    data_dir = Path(args.data)
    manifest = json.load(open(data_dir / "manifest.json", encoding="utf-8"))
    dataset_name, fold = manifest.get("dataset", data_dir.name), manifest.get("fold", 0)
    exp = args.exp or f"param_{args.head}"
    run_dir = Path(args.runs) / dataset_name / exp / (f"f{fold}_seed{args.seed}" if args.fold_in_path else f"seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run_dir={run_dir}", flush=True)

    tb.set_seed(args.seed)
    augment = not args.no_augment
    tr = ParamDataset(data_dir, "train", args.head, augment=augment)
    va = ParamDataset(data_dir, "val", args.head)
    te = ParamDataset(data_dir, "test", args.head)
    print(f"train {len(tr)} (real params {tr.n_real}, box-derived {tr.n_fallback})  val {len(va)}  test {len(te)}  "
          f"head={args.head} augment={augment} seed={args.seed}", flush=True)
    pin = device.type == "cuda"
    order_gen = torch.Generator(); order_gen.manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=tb.BATCH, shuffle=True, num_workers=args.workers, pin_memory=pin,
                    persistent_workers=args.workers > 0, worker_init_fn=tb._worker_init, collate_fn=collate, generator=order_gen)

    net = build_model(args.head, args.seed)
    fingerprint = tb.init_fingerprint(net)
    print(f"paired init fingerprint (shared layers): {fingerprint}", flush=True)
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=tb.LR)
    n_params = sum(p.numel() for p in net.parameters())

    t0 = time.perf_counter()
    rows = []
    for ep in range(1, args.epochs + 1):
        net.train()
        acc = {"loss": 0.0, "hm": 0.0, "size": 0.0, "off": 0.0, "giou": 0.0}
        for *batch, _ in tl:
            batch = [t.to(device, non_blocking=pin) for t in batch]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, parts = compute_loss(net, batch, args.head)
            opt.zero_grad(); loss.backward(); opt.step()
            acc["loss"] += loss.item()
            for k, v in parts.items():
                acc[k] += v
        nb = max(len(tl), 1)
        rows.append({"epoch": ep, **{k: v / nb for k, v in acc.items()}})
        if ep % 10 == 0 or ep == args.epochs:
            r = rows[-1]
            print(f"  epoch {ep}/{args.epochs}  loss {r['loss']:.4f}  hm {r['hm']:.4f}  size {r['size']:.4f}  "
                  f"off {r['off']:.4f}  giou {r['giou']:.4f}", flush=True)
    train_minutes = (time.perf_counter() - t0) / 60
    torch.save(net.state_dict(), run_dir / "final.pth")
    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for name, ds in (("val", va), ("test", te)):
        preds, params = predict_all(net, ds, args.head)
        json.dump(preds, open(run_dir / f"preds_{name}.json", "w", encoding="utf-8"))
        if args.head != "box":
            json.dump(params, open(run_dir / f"preds_{name}_params.json", "w", encoding="utf-8"))

    info = dict(dataset=dataset_name, exp=exp, seed=args.seed, fold=fold, head=args.head,
                model=f"AttnBBoxNet(base_ch={tb.BASE_CH}) backbone + CenterNet heatmap/offset + {args.head} size head",
                box_source=manifest.get("box_source"), augment=augment, flip="geometric (fixed)",
                paired_init=True, init_fingerprint=fingerprint, n_train=len(tr),
                n_inst_real_params={"train": tr.n_real, "val": va.n_real, "test": te.n_real},
                n_inst_box_derived={"train": tr.n_fallback, "val": va.n_fallback, "test": te.n_fallback},
                T0=T0, lam_giou=LAM_GIOU if args.head != "box" else None,
                imgsz=INPUT, epochs=args.epochs, batch=tb.BATCH, lr=tb.LR, optimizer="Adam", amp="bfloat16",
                model_selection="last epoch", params_M=round(n_params / 1e6, 4), train_minutes=round(train_minutes, 2),
                torch=torch.__version__, data_dir=str(data_dir), args=vars(args))
    json.dump(info, open(run_dir / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    m = evaluate_run(run_dir, data_dir)
    for alt in ("official", "hbat"):
        if (data_dir / f"gt_boxes_{alt}.json").exists():
            evaluate_run(run_dir, data_dir, gt_name=f"gt_boxes_{alt}.json", out_name=f"metrics_{alt}.json")
    t = m["test"]["at_selected"]
    print(f"[{dataset_name}/{exp}/f{fold}_seed{args.seed}] val conf {m['val']['selected_conf']:.2f} -> test F1 {t['F1']:.4f} "
          f"(P {t['P']:.3f} R {t['R']:.3f})  AP50 {m['test']['AP50']:.4f}  | {n_params / 1e6:.3f} M, {train_minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
