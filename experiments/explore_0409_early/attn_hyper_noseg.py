"""
0616 HyperbolaNet 去掉 seg_head + 注意力监督。
动机：0616 的 seg_head 已用双曲线带 mask 监督，和注意力监督冗余 → 之前增益 mixed。
去掉 seg_head 后，注意力成为【唯一】的双曲线带监督，消除冗余；其余保留 0616 强检测
（顶点检测 + 参数精修 + band IoU loss）。

单随机种子，with_attn vs no_attn 消融 + 全部 test 可视化。
"""
import os
import importlib.util
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

device     = exp.device
input_size = exp.input_size
SEED       = 0
LAM_ATT    = 1.0
N_VIS      = 100000


class AttnHyperNoSegNet(exp.HyperbolaNet):
    """继承 HyperbolaNet（含顶点/参数/offset/精修），加注意力分支；forward 不再算 seg。"""
    def __init__(self, in_ch=1, base_ch=32, use_attn=True):
        super().__init__(in_ch=in_ch, base_ch=base_ch)
        self.use_attn = use_attn
        self.last_a_logit = None
        if use_attn:
            self.attn_head = nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
                nn.Conv2d(base_ch * 2, 1, 1))

    def forward(self, x):
        f1, x = self.down1(x)
        f2, x = self.down2(x)
        f3, x = self.down3(x)
        x = self.bottleneck(x)
        if self.use_attn:
            a_logit = self.attn_head(f3)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            x = x * (1.0 + gate)
            self.last_a_logit = a_logit
        else:
            self.last_a_logit = None
        feat = x
        hm_logit = self.heatmap_head(x)
        param_out = torch.sigmoid(self.param_head(x))
        offset_out = self.offset_head(x)
        return hm_logit, param_out, offset_out, None, feat   # seg_logit=None（已去掉）


def attn_loss(a_logit, band):
    bce = F.binary_cross_entropy_with_logits(a_logit, band)
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (p * band).sum() / (p.sum() + band.sum() + 1e-6)
    return bce + dice


def _band_target(gt_seg):
    band = F.avg_pool2d(gt_seg, 4)            # 640 -> 160 (stride4)
    return (band > 0.5).float()


def compute_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg):
    """0616 的检测 loss（去掉 seg loss）+ 注意力 loss。"""
    hm_h = input_size[0] // exp.HM_STRIDE
    hm_w = input_size[1] // exp.HM_STRIDE
    pred_hm, pred_param, pred_offset, _, feat = model(images)

    l_focal  = exp.focal_loss_heatmap(pred_hm, gt_hm, peak_mask)
    l_coarse = exp.masked_param_loss_spatial(pred_param, gt_param, peak_mask)
    l_offset = exp.masked_offset_loss(pred_offset, gt_offset, peak_mask)
    l_refine, l_band = exp.compute_refinement_losses(
        model, feat, pred_param, pred_offset, gt_param, gt_offset, peak_mask,
        input_h=input_size[0], input_w=input_size[1], hm_h=hm_h, hm_w=hm_w,
        att_temp=exp.ATT_TEMP, band_temp=exp.BAND_TEMP, loss_fn=exp.BAND_LOSS_FN,
        band_h=exp.BAND_RES, band_w=exp.BAND_RES)

    loss = (l_focal + exp.LAM * l_coarse + l_offset
            + exp.LAM_REFINE * l_refine + exp.LAM_BAND * l_band)   # 无 seg loss
    if model.use_attn:
        loss = loss + LAM_ATT * attn_loss(model.last_a_logit, _band_target(gt_seg))
    return loss


def train_model(use_attn, full, tr, va, n_ep, work, tag):
    exp.set_seed(SEED)
    tl = DataLoader(exp.AugWrapper(Subset(full, tr)), batch_size=exp.batch_size,
                    shuffle=True, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    vl = DataLoader(Subset(full, va), batch_size=exp.batch_size,
                    shuffle=False, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    model = AttnHyperNoSegNet(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=exp.LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg, _ in tl:
            images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg = [
                t.to(device) for t in (images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg, _ in vl:
                images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg = [
                    t.to(device) for t in (images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += compute_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg).item()
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
        meta = full[i][-1]
        img_u8, pred_mask, _ = exp.predict_single_image(
            model, meta["image_path"], input_size, device,
            obj_thresh=exp.HM_THRESH, nms_k=exp.nms_kernel, max_det=exp.max_det)
        A = torch.sigmoid(model.last_a_logit[0, 0]).float().cpu().numpy() if model.last_a_logit is not None else None
        gt_band = exp.build_gt_mask_from_meta(meta, input_size)
        panel_orig = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        panel_attn = overlay(img_u8, A) if A is not None else panel_orig.copy()
        panel_pred = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR); panel_pred[pred_mask > 0.5] = (0, 0, 220)
        panel_gt = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR); panel_gt[gt_band > 0.5] = (0, 200, 0)
        for p, t in [(panel_orig, "Original"), (panel_attn, "Attention"), (panel_pred, "Pred band"), (panel_gt, "GT band")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(work, f"{k:02d}_{os.path.splitext(meta['image_name'])[0]}.png"),
                    np.hstack([panel_orig, panel_attn, panel_pred, panel_gt]))


def main():
    work = os.path.join(os.getcwd(), f"attn_hypernoseg_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = exp.HyperbolaDataset(
        image_dir=exp.data_sources[0]["image_dir"],
        annotation_json=exp.data_sources[0]["annotation_json"],
        input_size=input_size, hm_stride=exp.HM_STRIDE, sigma=exp.HM_SIGMA)
    tr, va, te = exp.make_split(len(full), SEED)
    print(f"HyperbolaNet(no seg_head) + attention  train={len(tr)} val={len(va)} test={len(te)}")

    keys = ["global_iou", "pixel_f1", "mAP50", "instance_recall", "mean_vertex_dist"]
    results = {}
    vis_model = None
    for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
        print(f"\n=== {tag} ===")
        model = train_model(use_attn, full, tr, va, exp.num_epochs, work, tag)
        m = exp.evaluate(model, Subset(full, te), device, input_size, obj_thresh=exp.HM_THRESH)
        print(f"  {tag}: " + "  ".join(f"{k}={m[k]:.4f}" for k in keys))
        results[tag] = m
        if use_attn:
            vis_model = model

    print("\n" + "=" * 78)
    print(f"{'config':>12}" + "".join(f"{k:>15}" for k in keys))
    for tag in ["with_attn", "no_attn"]:
        print(f"{tag:>12}" + "".join(f"{results[tag][k]:>15.4f}" for k in keys))
    print("=" * 78)
    visualize(vis_model, full, te, work)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
