"""核对准备好的数据集目录（prepare_data.py 的输出）是否符合 revision 要求，并出一张叠加图供目检。

    python check_data.py [--data ../detector_comparison/datasets/ds2_f0_s100] [--n 16]

输出（写到本目录）：
    check_masks.png   前 n 张训练图：绿色 = HBAT 导出的双曲线带掩码，红框 = 官方框
    check_data.txt    各划分的图像数 / 官方框数 / 带像素占比 / 带与官方框的重叠率
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "detector_comparison" / "datasets" / "ds2_f0_s100"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--mask-dir", default="masks_band")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    data = Path(args.data)
    gt = json.load(open(data / "gt_boxes.json", encoding="utf-8"))
    manifest = json.load(open(data / "manifest.json", encoding="utf-8"))

    lines = [f"data_dir: {data}", f"hbat_json: {manifest.get('hbat_json')}", f"box_source: {manifest.get('box_source')}",
             f"missing_mask policy: {manifest.get('missing_mask')}  train_items_skipped: {manifest.get('train_items_skipped')}", ""]
    tiles = []
    for phase in ("train", "val", "test"):
        n_img = n_box = 0
        band_frac, inside = [], []
        for stem in sorted(gt[phase]):
            rec = gt[phase][stem]
            n_img += 1
            n_box += len(rec["boxes"])
            mpath = data / args.mask_dir / phase / f"{stem}.png"
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None:
                lines.append(f"  [{phase}] {stem}: 缺掩码文件 {mpath}")
                continue
            m = m > 127
            band_frac.append(m.mean())
            if m.any() and rec["boxes"]:
                box_m = np.zeros_like(m)
                for x1, y1, x2, y2 in rec["boxes"]:
                    box_m[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = True
                inside.append((m & box_m).sum() / m.sum())
            if phase == "train" and len(tiles) < args.n:
                img = cv2.imread(str(next((data / "images" / phase).glob(stem + ".*"))))
                over = img.copy()
                over[m] = (0.4 * over[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
                for x1, y1, x2, y2 in rec["boxes"]:
                    cv2.rectangle(over, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(over, stem, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                tiles.append(over)
        lines.append(f"[{phase}] images {n_img}  official boxes {n_box}  "
                     f"band pixel fraction mean {np.mean(band_frac):.3f}  "
                     f"images with empty band {sum(1 for b in band_frac if b == 0)}  "
                     f"band pixels inside official boxes: median {np.median(inside):.2f}, min {np.min(inside):.2f} (n={len(inside)})")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "check_data.txt").write_text(txt + "\n", encoding="utf-8")
    if tiles:
        cols = 4
        rows = int(np.ceil(len(tiles) / cols))
        h, w = tiles[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(t, (w, h))
        cv2.imwrite(str(HERE / "check_masks.png"), grid)
        print(f"-> {HERE / 'check_masks.png'}")


if __name__ == "__main__":
    main()
