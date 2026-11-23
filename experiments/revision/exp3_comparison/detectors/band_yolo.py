"""Hyperbola-band supervision as a plug-in for ultralytics YOLO (approach A: auxiliary loss).

What it does (see README, "插件实验"):
  * The YOLO architecture is untouched. A small auxiliary head (3x3 Conv-BN-SiLU + 1x1 Conv) is attached to one
    intermediate feature map (default: neck P3 output, layer 15 of yolov8*, stride 8) and predicts a single-channel
    "where is the hyperbola band" logit.
  * The logit is trained against the HBAT band mask with BCE + Dice, exactly like the paper's attention branch.
    The head's output is NOT fed back into the detector (approach A), so inference cost is unchanged and the head
    can simply be ignored (or deleted) at test time.
  * The mask rides along as a 4th image channel through ultralytics' own augmentation pipeline (mosaic, affine,
    flips, letterbox ...), which guarantees pixel-exact alignment between image and mask. HSV jitter and the BGR->RGB
    step are applied to the first three channels only, so the detector sees exactly what the baseline sees.
  * Loss resolution: logits are bilinearly upsampled and the full-res mask is average-pooled to imgsz/band_stride
    (default 4 -> 160x160 for 640 input, the paper's f3 resolution). Soft targets = band coverage per cell, which
    avoids the "band is < 2 cells wide at stride 8" problem.

Usage from Python:
    from band_yolo import train_band
    trainer = train_band(overrides=dict(model="yolov8s.pt", data=".../data.yaml", epochs=100, ...),
                         band=dict(gain=1.0, layer=15, stride=4, warmup_epochs=3, mask_dir="masks_band"))
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Format, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import colorstr
from ultralytics.utils.patches import imread

BAND_DEFAULTS = dict(
    gain=1.0,            # lambda: weight of (BCE + Dice) relative to ultralytics' box/cls/dfl gains (7.5/0.5/1.5)
    layer=15,            # layer index whose output feeds the aux head (yolov8: 15 = neck P3/8, 4 = backbone P3/8)
    hidden=64,           # hidden channels of the aux head
    stride=4,            # resolution at which the loss is computed: imgsz / stride
    target="soft",       # soft: fraction of band pixels per cell; hard: (fraction > 0.5)
    warmup_epochs=3,     # linearly ramp the band loss over the first epochs (protects pretrained features)
    mask_dir="masks_band",   # sibling of images/ inside the dataset directory
    mask_missing="error",    # error | zero  (what to do when a train image has no mask png)
)


# ----------------------------------------------------------------------------- data
class BandRandomHSV(RandomHSV):
    """Apply HSV jitter to the 3 colour channels only; leave the mask channel alone."""

    def __call__(self, labels):
        img = labels["img"]
        if img.ndim == 3 and img.shape[-1] == 4:
            sub = {"img": np.ascontiguousarray(img[..., :3])}
            super().__call__(sub)
            labels["img"] = np.concatenate([sub["img"], img[..., 3:]], axis=-1)
            return labels
        return super().__call__(labels)


class BandFormat(Format):
    """Keep ultralytics' BGR->RGB handling for the colour channels; append the mask channel unchanged."""

    def _format_img(self, img):
        if img.ndim == 3 and img.shape[-1] == 4:
            rgb = super()._format_img(np.ascontiguousarray(img[..., :3]))
            m = torch.from_numpy(np.ascontiguousarray(img[..., 3:].transpose(2, 0, 1)))
            return torch.cat([rgb, m], 0)
        return super()._format_img(img)


class BandYOLODataset(YOLODataset):
    """YOLODataset whose images carry the band mask as a 4th channel (loaded from <mask_dir>/<stem>.png)."""

    def __init__(self, *args, mask_dir, mask_missing="error", **kwargs):
        self.mask_dir = Path(mask_dir)
        self.mask_missing = mask_missing
        super().__init__(*args, **kwargs)

    def _load_mask(self, im_file, shape_hw):
        p = self.mask_dir / (Path(im_file).stem + ".png")
        m = imread(str(p), flags=cv2.IMREAD_GRAYSCALE) if p.exists() else None
        if m is None:
            if self.mask_missing == "error":
                raise FileNotFoundError(f"band mask not found for {im_file}: {p}")
            m = np.zeros(shape_hw, dtype=np.uint8)
        if m.ndim == 3:  # ultralytics' imread keeps a trailing channel axis for grayscale
            m = m[..., 0]
        if m.shape[:2] != tuple(shape_hw):
            m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        return m

    def load_image(self, i, rect_mode=True, resize_short=False):
        im = self.ims[i]
        if im is None:
            f = self.im_files[i]
            im = imread(f, flags=cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            m = self._load_mask(f, im.shape[:2])
            im = np.concatenate([im, m[..., None]], axis=-1)  # BGR + mask
            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(int(np.ceil(w0 * r)), self.imgsz), min(int(np.ceil(h0 * r)), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        for tr in t.transforms:
            if type(tr) is RandomHSV:
                tr.__class__ = BandRandomHSV
            elif type(tr) is Format:
                tr.__class__ = BandFormat
        return t


# ----------------------------------------------------------------------------- model
class BandAuxDetectionModel(DetectionModel):
    """DetectionModel + auxiliary band head. Detection path is byte-for-byte the stock model."""

    def __init__(self, cfg="yolov8s.yaml", ch=3, nc=None, verbose=True, band=None):
        b = {**BAND_DEFAULTS, **(band or {})}
        self.aux_layer = int(b["layer"])
        self._aux_feat = None
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        with torch.no_grad():
            self.model.eval()
            self.forward(torch.zeros(1, ch, 256, 256))
            self.model.train()
        c = self._aux_feat.shape[1]
        self.aux_head = nn.Sequential(Conv(c, int(b["hidden"]), 3, 1), nn.Conv2d(int(b["hidden"]), 1, 1))
        self.band_gain = float(b["gain"])
        self.band_stride = int(b["stride"])
        self.band_target = b["target"]
        self.band_scale = 1.0  # warm-up multiplier, set by the trainer each epoch
        self._aux_feat = None

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, feat = [], None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i == self.aux_layer:
                feat = x
        self._aux_feat = feat
        return x

    def band_loss(self, band):
        """BCE + Dice between the aux logit and the (binarised, full-res) band mask at imgsz/stride."""
        logit = self.aux_head(self._aux_feat)
        H, W = band.shape[-2:]
        res = (max(1, H // self.band_stride), max(1, W // self.band_stride))
        tgt = F.adaptive_avg_pool2d(band.float(), res)
        if self.band_target == "hard":
            tgt = (tgt > 0.5).float()
        logit = F.interpolate(logit.float(), size=res, mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logit, tgt)
        p = torch.sigmoid(logit)
        inter = (p * tgt).sum(dim=(1, 2, 3))
        den = p.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * inter + 1e-6) / (den + 1e-6)
        return bce + dice.mean()

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        det_loss, det_items = self.criterion(preds, batch)  # (3,) * batch_size, (3,) detached
        bs = batch["img"].shape[0]
        band = batch.get("band")
        if band is not None and self._aux_feat is not None and self.band_gain > 0:
            item = self.band_gain * self.band_scale * self.band_loss(band)
        else:
            item = det_loss.new_zeros(())
        total = torch.cat([det_loss.view(-1), (item * bs).view(1)])
        items = torch.cat([det_items.view(-1), item.detach().view(1)])
        return total, items


# ----------------------------------------------------------------------------- trainer
def _band_warmup(trainer):
    w = int(trainer.band.get("warmup_epochs", 0))
    scale = 1.0 if w <= 0 else min(1.0, (trainer.epoch + 1) / w)
    m = trainer.model
    if hasattr(m, "band_scale"):
        m.band_scale = scale


class BandTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, band=None):
        self.band = {**BAND_DEFAULTS, **(band or {})}
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", _band_warmup)

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        img_path = Path(img_path)
        mask_dir = img_path.parents[1] / self.band["mask_dir"] / img_path.name
        return BandYOLODataset(
            img_path=str(img_path), imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
            rect=False, cache=self.args.cache or None, single_cls=self.args.single_cls or False, stride=gs, pad=0.0,
            prefix=colorstr("train: "), task=self.args.task, classes=self.args.classes, data=self.data,
            fraction=self.args.fraction, mask_dir=mask_dir, mask_missing=self.band["mask_missing"],
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        img = batch["img"]
        if img.shape[1] == 4:
            batch["band"] = (img[:, 3:4] > 0.5).float()  # mosaic/letterbox padding (114/255) -> background
            batch["img"] = img[:, :3].contiguous()
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = BandAuxDetectionModel(cfg, nc=self.data["nc"], ch=3, verbose=verbose, band=self.band)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "band_loss"
        from copy import copy
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def plot_training_samples(self, batch, ni):
        try:  # before super(): ultralytics' plot_images converts the batch tensors to numpy in place
            self._save_band_overlay(batch, ni)
        except Exception as e:  # never let a debug plot kill training
            print(f"[band_yolo] overlay plot failed: {e}")
        super().plot_training_samples(batch, ni)

    @staticmethod
    def _np(x):
        return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)

    def _save_band_overlay(self, batch, ni, max_imgs=8):
        """Alignment check: training image (after all augmentation) with the band mask in green and GT boxes in red."""
        if "band" not in batch:
            return
        imgs = self._np(batch["img"][:max_imgs])
        if imgs.max() <= 1.0:
            imgs = imgs * 255
        imgs = imgs.transpose(0, 2, 3, 1).astype(np.uint8)  # RGB
        band = self._np(batch["band"][:max_imgs, 0]) > 0.5
        bidx = self._np(batch["batch_idx"])
        boxes = self._np(batch["bboxes"])  # normalized xywh
        tiles = []
        for i in range(imgs.shape[0]):
            im = imgs[i].copy()
            h, w = im.shape[:2]
            green = np.zeros_like(im); green[..., 1] = 255
            im = np.where(band[i][..., None], (0.5 * im + 0.5 * green).astype(np.uint8), im)
            for cx, cy, bw, bh in boxes[bidx == i]:
                x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
            tiles.append(cv2.resize(im, (320, 320)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[r:r + 4], axis=1) for r in range(0, len(tiles), 4)]
        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(str(self.save_dir / f"train_batch{ni}_band.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def train_band(overrides: dict, band: dict | None = None, trainer_cls=None):
    trainer = (trainer_cls or BandTrainer)(overrides=overrides, band=band)
    trainer.train()
    return trainer


# ----------------------------------------------------------------------------- band output at test time
def _letterbox(im, imgsz, pad_value=114):
    """Same geometry as ultralytics LetterBox(auto=False, center=True): resize long side to imgsz, pad to a square."""
    h0, w0 = im.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    if (nw, nh) != (w0, h0):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = int(round((imgsz - nh) / 2 - 0.1)), int(round((imgsz - nw) / 2 - 0.1))
    out = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = im
    return out, (top, left, nh, nw)


@torch.no_grad()
def predict_band(net, img_paths, imgsz, device):
    """Run the aux head on each image. Returns {stem: uint8 probability map (0..255) at the ORIGINAL image size}.

    The detection outputs are ignored here (they are exported separately by model.predict); this only reads the
    feature map the aux head is attached to. Used to measure band quality (band IoU, as in exp1_basic Round 16) and,
    later, to turn the predicted band into boxes (exp1_basic Round 12 / 15 style)."""
    net = net.to(device).eval()
    dtype = next(net.parameters()).dtype
    out = {}
    for p in img_paths:
        p = Path(p)
        im = imread(str(p), flags=cv2.IMREAD_COLOR)
        h0, w0 = im.shape[:2]
        lb, (top, left, nh, nw) = _letterbox(im, imgsz)
        x = torch.from_numpy(np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1))).to(device, dtype)[None] / 255.0
        net(x)
        logit = net.aux_head(net._aux_feat)
        prob = torch.sigmoid(F.interpolate(logit.float(), size=(imgsz, imgsz), mode="bilinear", align_corners=False))
        prob = prob[0, 0, top:top + nh, left:left + nw].cpu().numpy()
        if prob.shape != (h0, w0):
            prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out[p.stem] = np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)
    net._aux_feat = None
    return out


def band_counts(pred_u8, gt_u8, t=0.5):
    """Pixel counts for pooled band IoU / precision / recall (pool by summing over images, as diag_band_quality.py)."""
    p = pred_u8 >= int(round(t * 255))
    g = gt_u8 > 127
    return {"inter": int((p & g).sum()), "union": int((p | g).sum()), "pred": int(p.sum()), "gt": int(g.sum())}
