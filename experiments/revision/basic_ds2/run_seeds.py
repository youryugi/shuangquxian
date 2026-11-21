"""批量跑 5 折 × 多 seed 的实验矩阵（并行若干个进程；已有 metrics.json 的 run 跳过）。

    python run_seeds.py                                   # 默认 4 个实验 × 5 折 × seed 0-9，4 个并行
    python run_seeds.py --exps hbatbox_aug_none hbatbox_aug_abs --folds 0 --seeds 0 1 2 --jobs 2

每个 run 的控制台输出写到 <runs>/log_<exp>_f<fold>_seed<k>.txt，总日志 <runs>/run_seeds.log。
run 目录：<runs>/ds2/<exp>/f<fold>_seed<k>/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent / "detector_comparison" / "datasets"
PY = sys.executable

# 实验名 -> (数据目录后缀, 额外参数)
EXPS = {
    "hbatbox_aug_none": ("_hbatbox", ["--mode", "none", "--augment"]),   # 主设置：HBAT 派生框 + 翻转，无注意力
    "hbatbox_aug_abs":  ("_hbatbox", ["--mode", "abs", "--augment"]),    # 主设置：本文方法
    "aug_none":         ("",         ["--mode", "none", "--augment"]),   # 对照：官方框 + 翻转
    "aug_abs":          ("",         ["--mode", "abs", "--augment"]),
    "hbatbox_none":     ("_hbatbox", ["--mode", "none"]),
    "hbatbox_abs":      ("_hbatbox", ["--mode", "abs"]),
    "basic_none":       ("",         ["--mode", "none"]),
    "basic_abs":        ("",         ["--mode", "abs"]),
    # ── 第 7 轮：上限诊断，真实带直接当 gate 融合（训练 + 推理都给真实带）
    "hbatbox_aug_oracle":      ("_hbatbox", ["--mode", "oracle", "--augment"]),
    # ── 第 8 轮：把带的范围信息做进网络
    "hbatbox_aug_abs_extent":  ("_hbatbox", ["--mode", "abs", "--augment", "--fuse", "extent"]),        # a: 4 通道范围特征
    "hbatbox_aug_oracle_extent": ("_hbatbox", ["--mode", "oracle", "--augment", "--fuse", "extent"]),
    "hbatbox_aug_none_s4":     ("_hbatbox", ["--mode", "none", "--augment", "--det-stride", "4"]),     # b: 检测头在 56×56
    "hbatbox_aug_abs_s4":      ("_hbatbox", ["--mode", "abs", "--augment", "--det-stride", "4"]),
    # ── 第 9 轮：数据量曲线（训练子集只由 fold 决定，两臂相同）
    "hbatbox_aug_none_frac50": ("_hbatbox", ["--mode", "none", "--augment", "--train-frac", "0.5"]),
    "hbatbox_aug_abs_frac50":  ("_hbatbox", ["--mode", "abs", "--augment", "--train-frac", "0.5"]),
    "hbatbox_aug_none_frac25": ("_hbatbox", ["--mode", "none", "--augment", "--train-frac", "0.25"]),
    "hbatbox_aug_abs_frac25":  ("_hbatbox", ["--mode", "abs", "--augment", "--train-frac", "0.25"]),
    # ── 第 10 轮：λ 扫描 + gate 融合（尽职调查）
    "hbatbox_aug_abs_lam0.1":  ("_hbatbox", ["--mode", "abs", "--augment", "--lam-att", "0.1"]),
    "hbatbox_aug_abs_lam0.3":  ("_hbatbox", ["--mode", "abs", "--augment", "--lam-att", "0.3"]),
    "hbatbox_aug_abs_lam1.5":  ("_hbatbox", ["--mode", "abs", "--augment", "--lam-att", "1.5"]),
    "hbatbox_aug_abs_gate":    ("_hbatbox", ["--mode", "abs", "--augment", "--fuse", "gate"]),
    # ── 第 13 轮：对照——attention 头用填实的 HBAT 框当监督（masks_box，没有任何 band 形状），之后做同样的 mask → 框步骤
    "hbatbox_aug_boxmask":     ("_hbatbox", ["--mode", "abs", "--augment", "--mask-dir", "masks_box"]),
    # ── 第 16 轮：U 形解码的 attention 头，在 224×224 上输出和监督（--attn-head unet）；框掩码对照用同一个头
    "hbatbox_aug_abs_unet":     ("_hbatbox", ["--mode", "abs", "--augment", "--attn-head", "unet"]),
    "hbatbox_aug_boxmask_unet": ("_hbatbox", ["--mode", "abs", "--augment", "--attn-head", "unet", "--mask-dir", "masks_box"]),
    # ── 第 17 轮：train_param.py（第三项 = 脚本）。翻转增强修好（几何翻转）；box = 原 baseline 只修翻转，
    #    param = 顶点 + 真实 HBAT 参数，pseudo = 同结构、参数从框反推（只用框标注的对照）
    "hbatbox_augfix_none":   ("_hbatbox", ["--head", "box"], "train_param.py"),
    "hbatbox_augfix_param":  ("_hbatbox", ["--head", "param"], "train_param.py"),
    "hbatbox_augfix_pseudo": ("_hbatbox", ["--head", "pseudo"], "train_param.py"),
    # ── 2026-11-21：train_basic.py 的翻转 bug（目标偏 8 px）修好后重跑第 4–16 轮的核心 arm（参数与对应的 hbatbox_aug_* 相同）
    "hbatbox_flipfix_none":         ("_hbatbox", ["--mode", "none", "--augment"]),
    "hbatbox_flipfix_abs":          ("_hbatbox", ["--mode", "abs", "--augment"]),
    "hbatbox_flipfix_boxmask":      ("_hbatbox", ["--mode", "abs", "--augment", "--mask-dir", "masks_box"]),
    "hbatbox_flipfix_oracle":       ("_hbatbox", ["--mode", "oracle", "--augment"]),
    "hbatbox_flipfix_abs_unet":     ("_hbatbox", ["--mode", "abs", "--augment", "--attn-head", "unet"]),
    "hbatbox_flipfix_boxmask_unet": ("_hbatbox", ["--mode", "abs", "--augment", "--attn-head", "unet", "--mask-dir", "masks_box"]),
    # 回归检查：--flip-legacy 应逐位复现旧的 hbatbox_aug_none
    "hbatbox_fliplegacy_none":      ("_hbatbox", ["--mode", "none", "--augment", "--flip-legacy"]),
    # 第 19 轮：固定测试集 + 嵌套训练子集（回应 R2-6）。每折测试集、验证集不变，训练集取 25 / 50 / 75%（子集只由 fold 决定，
    # 25 ⊂ 50 ⊂ 75 ⊂ 100，各 arm、各 seed 相同）；100% = 上面的 hbatbox_flipfix_*。必须 --workers 1。
    **{f"hbatbox_flipfix_{arm}_frac{p}": ("_hbatbox", args + ["--train-frac", f"0.{p}"])
       for p in (25, 50, 75)
       for arm, args in (("none", ["--mode", "none", "--augment"]),
                         ("abs", ["--mode", "abs", "--augment"]),
                         ("boxmask", ["--mode", "abs", "--augment", "--mask-dir", "masks_box"]))},
}
DEFAULT_EXPS = ["hbatbox_aug_none", "hbatbox_aug_abs", "aug_none", "aug_abs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=DEFAULT_EXPS, choices=list(EXPS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--jobs", type=int, default=4, help="并行进程数")
    ap.add_argument("--workers", type=int, default=2, help="每个进程的 DataLoader worker 数")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    log = open(runs / "run_seeds.log", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    # 顺序：同一 (fold, seed) 的各实验相邻，配对的两臂尽量一起完成
    jobs = []
    for fold in args.folds:
        for seed in args.seeds:
            for exp in args.exps:
                suffix, extra, *script = EXPS[exp]
                script = script[0] if script else "train_basic.py"
                data = DATASETS / f"ds2_f{fold}_s100{suffix}"
                run_dir = runs / "ds2" / exp / f"f{fold}_seed{seed}"
                if (run_dir / "metrics.json").exists():
                    continue
                if not data.exists():
                    raise SystemExit(f"dataset dir missing: {data} (run prepare_data.py --fold {fold} first)")
                cmd = [PY, "-X", "utf8", str(HERE / script), "--data", str(data), "--exp", exp, "--seed", str(seed),
                       "--runs", str(runs), "--fold-in-path", "--workers", str(args.workers), *extra, *args.extra]
                jobs.append((exp, fold, seed, cmd))
    say(f"{len(jobs)} runs to do  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    if args.dry_run:
        for exp, fold, seed, cmd in jobs:
            print(exp, fold, seed, " ".join(cmd[3:]))
        return

    running = []   # (proc, exp, fold, seed, logfile, t0)
    t_start = time.time()
    done = 0
    while jobs or running:
        while jobs and len(running) < args.jobs:
            exp, fold, seed, cmd = jobs.pop(0)
            lf = open(runs / f"log_{exp}_f{fold}_seed{seed}.txt", "w", encoding="utf-8")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((p, exp, fold, seed, lf, time.time()))
        time.sleep(2)
        still = []
        for p, exp, fold, seed, lf, t0 in running:
            if p.poll() is None:
                still.append((p, exp, fold, seed, lf, t0))
                continue
            lf.close()
            done += 1
            try:
                last = (runs / f"log_{exp}_f{fold}_seed{seed}.txt").read_text(encoding="utf-8").strip().splitlines()[-1]
            except Exception:
                last = ""
            say(f"[{done}] {exp} f{fold} seed{seed} rc={p.returncode} {time.time() - t0:.0f}s | {last[:160]}")
        running = still
    say(f"ALL_DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
