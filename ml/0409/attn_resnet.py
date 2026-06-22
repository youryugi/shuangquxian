"""
预训练 ResNet18 backbone + 双曲线带注意力监督 + CenterNet 检测头。
- backbone: ImageNet 预训练 resnet18（灰度 repeat 成 3 通道喂入），微调用小 lr。
- layer1(stride4)接注意力头（双曲线带监督），layer2(stride8)接检测头，注意力门控 feat*(1+gate)。
- 复用 attn_cnn 的 dataset / loss / predict / evaluate。
- 内置 with_attn vs no_attn 消融 + 注意力可视化。

跑：C:/Users/79152/.conda/envs/gpr/python.exe attn_resnet.py
"""
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights

import attn_cnn as ac

device     = ac.device
input_size = ac.input_size
N_VIS      = 100000  # 全部 test


class AttnResNet(nn.Module):
    def __init__(self, use_attn=True, pretrained=True):
        super().__init__()
        self.use_attn = use_attn
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1   # stride4, 64ch
        self.layer2 = m.layer2   # stride8, 128ch
        c1, c2 = 64, 128
        if use_attn:
            self.attn_head = nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True), nn.Conv2d(c1, 1, 1))
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True), nn.Conv2d(c2, 1, 1))
        self.wh_head = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True), nn.Conv2d(c2, 2, 1))
        self.offset_head = nn.Sequential(
            nn.Conv2d(c2, c2 // 2, 3, padding=1, bias=False), nn.BatchNorm2d(c2 // 2),
            nn.ReLU(inplace=True), nn.Conv2d(c2 // 2, 2, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)                 # 灰度 -> 3 通道喂预训练网络
        x = self.stem(x)
        f1 = self.layer1(x)                      # stride4
        f2 = self.layer2(f1)                     # stride8
        a_logit = None
        if self.use_attn:
            a_logit = self.attn_head(f1)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            f2 = f2 * (1.0 + gate)
        return self.heatmap_head(f2), torch.sigmoid(self.wh_head(f2)), self.offset_head(f2), a_logit


def train_model(use_attn, full, tr, va, n_ep, work, tag):
    ac.exp.set_seed(ac.SEED)
    tl = DataLoader(Subset(full, tr), batch_size=ac.batch_size, shuffle=True, num_workers=0, collate_fn=ac.collate)
    vl = DataLoader(Subset(full, va), batch_size=ac.batch_size, shuffle=False, num_workers=0, collate_fn=ac.collate)
    model = AttnResNet(use_attn=use_attn, pretrained=True).to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith(("stem", "layer1", "layer2"))]
    hd = [p for n, p in model.named_parameters() if not n.startswith(("stem", "layer1", "layer2"))]
    opt = torch.optim.Adam([{"params": bb, "lr": 1e-4}, {"params": hd, "lr": 5e-4}])
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = ac.compute_loss(model, img, hm, wh, off, pk, band)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, hm, wh, off, pk, band, _ in vl:
                img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += ac.compute_loss(model, img, hm, wh, off, pk, band).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.55, hc, 0.45, 0)


def visualize(model, full, te, work):
    H, W = input_size
    for k, i in enumerate(te[:N_VIS]):
        img_t, _, _, _, _, _, meta = full[i]
        boxes, scores, A = ac.predict(model, img_t.unsqueeze(0))
        gray = (img_t[0].numpy() * 255).astype(np.uint8)
        panel_orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panel_attn = overlay(gray, A) if A is not None else panel_orig.copy()
        gt_band = np.zeros((H, W), np.float32)
        for o in meta["objects"]:
            gt_band = np.maximum(gt_band, ac.exp.rasterize_hyperbola_band_mask(
                H, W, o["x_vertex"], o["y_vertex"], o["width"], o["height"], o["thickness"]))
        panel_gt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); panel_gt[gt_band > 0.5] = (0, 200, 0)
        panel_pred = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for b in boxes:
            cv2.rectangle(panel_pred, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 220), 2)
        for p, t in [(panel_orig, "Original"), (panel_attn, "Attention"), (panel_gt, "GT band"), (panel_pred, "Pred")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(work, f"{k:02d}_{os.path.splitext(meta['image_name'])[0]}.png"), np.hstack([panel_orig, panel_attn, panel_gt, panel_pred]))


def main():
    work = os.path.join(os.getcwd(), f"attn_resnet_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = ac.AttnDataset(input_size=input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = ac.exp.make_split(len(full), ac.SEED)
    print(f"ResNet18(pretrained) backbone  train={len(tr)} val={len(va)} test={len(te)}")

    results = {}
    vis_model = None
    for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
        print(f"\n=== {tag} ===")
        model = train_model(use_attn, full, tr, va, ac.num_epochs, work, tag)
        m = ac.evaluate(model, full, te)
        print(f"  {tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        results[tag] = m
        if use_attn:
            vis_model = model

    print("\n" + "=" * 56)
    print(f"{'config':>12}{'bbox_P':>10}{'bbox_R':>10}{'bbox_F1':>10}{'attn_iou':>10}")
    for tag in ["with_attn", "no_attn"]:
        r = results[tag]
        print(f"{tag:>12}{r['bbox_P']:>10.4f}{r['bbox_R']:>10.4f}{r['bbox_F1']:>10.4f}{r['attn_band_iou']:>10.4f}")
    print("（对比 from-scratch CNN：with_attn F1≈0.49）")

    visualize(vis_model, full, te, work)
    print(f"\nSaved models + {min(N_VIS, len(te))} visuals -> {work}")


if __name__ == "__main__":
    main()
