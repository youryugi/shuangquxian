"""第 18 轮：band 对 box-mask 的优势是不是"选参选出来的"（只读 runs_cv/bandbox/sweep.md，不训练）。

    python diag_config_robust.py

选参只看验证集，所以两臂各自选中的配置可能不同；如果其中一臂恰好选中一个测试上很差的配置，配对差值会被放大。
这里换一个角度：对 sweep 里**同一组参数**，比较 band 臂和 box-mask 臂的测试 F1（10 seed 均值），
数 band 赢了多少组、差值中位数多少。每对 (band 变体, box-mask 变体) 的参数网格相同。
另给两种子集：按合并方式（union / replace / blend）分开；两臂验证 F1 均值排前 20% 的参数组（对两臂中立的"合理区域"）。
输出 runs_cv/bandbox/config_robust.md。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "runs_cv" / "bandbox" / "sweep.md"
OUT = HERE / "runs_cv" / "bandbox" / "config_robust.md"
PAIRS = [  # (box-mask 臂, band 臂)
    ("hbatbox_flipfix_boxmask_fitbox", "hbatbox_flipfix_abs_fitbox"),
    ("hbatbox_flipfix_boxmask_unet_fitbox", "hbatbox_flipfix_abs_unet_fitbox"),
    ("hbatbox_flipfix_boxmask_bandbox", "hbatbox_flipfix_abs_bandbox"),
    ("hbatbox_flipfix_boxmask_unet_bandbox", "hbatbox_flipfix_abs_unet_bandbox"),
    ("hbatbox_flipfix_boxmask_bandbox_nms", "hbatbox_flipfix_abs_bandbox_nms"),
] + [(f"hbatbox_flipfix_boxmask_frac{p}_{v}", f"hbatbox_flipfix_abs_frac{p}_{v}")   # 第 19 轮：各训练比例
     for p in (25, 50, 75) for v in ("fitbox", "bandbox", "bandbox_nms")]


def parse():
    """{variant: {config_str: (val, test)}}"""
    out, cur = {}, None
    for line in SWEEP.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (\S+)（来源", line)
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        m = re.match(r"^\| (\(.*\)) \| ([\d.]+) \| ([\d.]+) ± [\d.]+(?: \*\*←\*\*)? \|$", line)
        if m and cur is not None:
            cur[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def main():
    sw = parse()
    L = ["# 同一组参数下 band 对 box-mask（测试 F1，10 seed 均值；来自 sweep.md）", "",
         "| box-mask 臂 → band 臂 | 参数组数 | band 赢 | 差值中位数 | 差值 [min, max] | union / replace / blend 下 band 赢 | 验证前 20% 参数组 band 赢（中位数） | 两臂各自选中的配置 → 差值 |",
         "|---|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        if a not in sw or b not in sw:
            L.append(f"| {a} → {b} | 缺 | | | | |")
            continue
        keys = sorted(set(sw[a]) & set(sw[b]))
        d = np.array([sw[b][k][1] - sw[a][k][1] for k in keys])
        ca = max(sw[a], key=lambda k: sw[a][k][0]); cb = max(sw[b], key=lambda k: sw[b][k][0])
        by_mode = []
        for m in ("union", "replace", "blend"):
            dm = np.array([sw[b][k][1] - sw[a][k][1] for k in keys if f"'{m}'" in k])
            by_mode.append(f"{(dm > 0).sum()}/{len(dm)}" if len(dm) else "—")
        top = sorted(keys, key=lambda k: -(sw[a][k][0] + sw[b][k][0]) / 2)[:max(1, len(keys) // 5)]
        dt = np.array([sw[b][k][1] - sw[a][k][1] for k in top])
        L.append(f"| {a} → {b} | {len(keys)} | {(d > 0).sum()}/{len(keys)} | {np.median(d):+.3f} | [{d.min():+.3f}, {d.max():+.3f}] | "
                 f"{' / '.join(by_mode)} | {(dt > 0).sum()}/{len(dt)}（{np.median(dt):+.3f}） | "
                 f"{sw[a][ca][1]:.3f} → {sw[b][cb][1]:.3f}（{sw[b][cb][1] - sw[a][ca][1]:+.3f}） |")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
