from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def hyperbola_to_bbox_from_params(
    x_vertex: float,
    y_vertex: float,
    width: float,
    height: float,
    thickness: float,
    image_w: float,
    image_h: float,
) -> Tuple[float, float, float, float]:
    half_w = max(2.0, width) / 2.0
    x_min = max(0.0, x_vertex - half_w)
    x_max = min(image_w, x_vertex + half_w)
    y_min = max(0.0, y_vertex - max(1.0, thickness) / 2.0)
    y_max = min(image_h, y_vertex + max(1.0, height) + max(1.0, thickness) / 2.0)
    return x_min, y_min, x_max, y_max


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HyperbolaE2EDetector(nn.Module):
    """
    One-stage end-to-end detector.
    Output channels per cell:
      0: objectness logit
      1-5: params (xv_n, yv_n, w_n, h_n, t_n)
    """

    def __init__(self, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, width, stride=2),      # /2
            ConvBlock(width, width),
            ConvBlock(width, width * 2, stride=2),  # /4
            ConvBlock(width * 2, width * 2),
            ConvBlock(width * 2, width * 4, stride=2),  # /8
            ConvBlock(width * 4, width * 4),
            ConvBlock(width * 4, width * 4, stride=2),  # /16
            ConvBlock(width * 4, width * 4),
        )
        self.head = nn.Conv2d(width * 4, 6, kernel_size=1)
        self.stride = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        return self.head(feat)


def compute_loss(
    pred: torch.Tensor,
    target_obj: torch.Tensor,
    target_param: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # pred: [B, 6, H, W]
    obj_logit = pred[:, 0:1]
    param_raw = pred[:, 1:6]

    obj_loss = F.binary_cross_entropy_with_logits(obj_logit, target_obj)

    pos_mask = target_obj > 0.5
    n_pos = pos_mask.sum().clamp(min=1.0)

    param_pred = torch.sigmoid(param_raw)
    param_diff = torch.abs(param_pred - target_param)
    param_loss = (param_diff * pos_mask).sum() / n_pos

    total = obj_loss + 2.5 * param_loss
    return {
        "loss": total,
        "obj_loss": obj_loss,
        "param_loss": param_loss,
    }


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N, 4], b: [M, 4]
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)

    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_thr]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def decode_predictions(
    pred_map: torch.Tensor,
    img_size: int,
    stride: int,
    conf_thr: float = 0.35,
) -> List[Dict[str, float]]:
    """
    Decode one image prediction map [6, H, W] to list of detections.
    """
    assert pred_map.dim() == 3 and pred_map.shape[0] == 6

    obj = torch.sigmoid(pred_map[0])
    param = torch.sigmoid(pred_map[1:6])

    H, W = obj.shape
    ys, xs = torch.where(obj >= conf_thr)
    if ys.numel() == 0:
        return []

    scores = obj[ys, xs]
    xv_n = param[0, ys, xs]
    yv_n = param[1, ys, xs]
    pw_n = param[2, ys, xs]
    ph_n = param[3, ys, xs]
    pt_n = param[4, ys, xs]

    out = []
    for idx in range(ys.numel()):
        out.append(
            {
                "score": float(scores[idx].item()),
                "x_vertex_n": float(xv_n[idx].item()),
                "y_vertex_n": float(yv_n[idx].item()),
                "width_n": float(pw_n[idx].item()),
                "height_n": float(ph_n[idx].item()),
                "thickness_n": float(pt_n[idx].item()),
            }
        )
    return out
