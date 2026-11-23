"""YOLOv8 + band 辅助监督（插件）：5 折 × 多 seed 的实验矩阵。和 exp1_basic 同一个严格协议、同一个目录格式。

    python run_cv.py --exps y8s_pre_none y8s_pre_band y8s_pre_boxmask --seeds 0 1 2          # Round 1 pilot
    python run_cv.py --exps y8s_pre_none y8s_pre_band --folds 0 --seeds 0 --epochs 2 --runs runs_smoke   # 冒烟
    python run_cv.py --check                                                                   # 配对检查（fingerprint）

训练调用 ../detectors/train_ultralytics.py（插件本体是 ../detectors/band_yolo.py）。
run 目录：<runs>/ds2/<exp>/f<fold>_seed<k>/ —— 和 exp1_basic/runs_cv 一样，所以下面两个脚本可以直接用：
    python ../../exp1_basic/aggregate_cv.py --runs runs_cv --pairs y8s_pre_none:y8s_pre_band ...   # 5 折合并 + 配对统计
    python ../../exp1_basic/diag_strata.py  --runs runs_cv --exps ... --pairs ...                 # grouped / near miss / F1@0.7
每个 run 的控制台输出：<runs>/log_<exp>_f<fold>_seed<k>.txt；总日志 <runs>/run_cv.log。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETCMP = HERE.parent / "detectors"            # 检测器训练脚本（train_ultralytics.py、train_frcnn.py、band_yolo.py），训练时 cwd
SHARED = HERE.parents[1] / "shared"         # 数据目录与评估口径
DATASETS = SHARED / "datasets"
PY = sys.executable

# 所有 arm 共用：HBAT box 数据目录、最后一个 epoch 的权重（不在 13 张图的验证集上按 mAP50-95 挑 epoch）
COMMON = ["--weights", "last", "--epochs", "100", "--imgsz", "640", "--batch", "16"]
PRE, SCR = ["--model", "yolov8s.pt"], ["--model", "yolov8s.yaml"]
BAND = ["--band", "--save-band"]
# 只开左右翻转（和 exp1_basic 的轻量模型一样）：关掉 mosaic、缩放、平移、HSV 等；fliplr 保持 0.5。
# 检测器对比主表用这个（../detectors/README.md 第 0 节，2026-11-21）。环境里没有 albumentations，所以没有额外的模糊/CLAHE。
FLIP_ONLY = ["--extra", json.dumps({"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "hsv_h": 0.0,
                                    "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                                    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "bgr": 0.0},
                                   separators=(",", ":"))]
SCR_EPOCHS = 100   # 从零训练的轮数：y8s_scr_flip_none_e100/e200/e300 在 5 折合并验证集上比，100 最高（runs_cv/scr_epochs.md）

def EP(n):
    # 关掉早停（patience 0）：ultralytics 默认 patience 100 会按验证集 mAP 提前停，学习率没退火完，
    # 也违反"取最后一个 epoch、不按验证集挑"的协议。100 epoch 时 patience 100 本来就停不下来，所以只影响 > 100 的。
    return ["--epochs", str(n), "--patience", "0"]

# 实验名 -> 额外参数
EXPS = {
    # ── 主比较（COCO 预训练）
    "y8s_pre_none":     PRE,                                            # baseline
    "y8s_pre_band":     PRE + BAND,                                     # + band 辅助监督（接法 A）
    "y8s_pre_boxmask":  PRE + BAND + ["--mask-dir", "masks_box"],       # 对照：同一个头，监督换成填实的 HBAT box
    # ── 噪声底（A/A）：头接上但损失权重为 0；和 none 的差 = 管线 + GPU 非确定性，不含任何监督
    "y8s_pre_g0":       PRE + ["--band", "--band-gain", "0"],
    "y8s_pre_none_rep": PRE,                                            # 同一 seed 再跑一次 baseline
    # ── 从零训练（没有预训练特征时 band 是否更有用）
    "y8s_scr_none":     SCR,
    "y8s_scr_band":     SCR + BAND,
    "y8s_scr_boxmask":  SCR + BAND + ["--mask-dir", "masks_box"],
    # ── 检测器对比（只开左右翻转，和轻量模型对齐）
    "y8s_pre_flip_none":   PRE + FLIP_ONLY,
    "y8s_scr_flip_none":   SCR + FLIP_ONLY + EP(SCR_EPOCHS),
    "y8s_scr_flip_band":   SCR + FLIP_ONLY + BAND + EP(SCR_EPOCHS),
    # 从零训练的轮数候选（只看验证集）
    "y8s_scr_flip_none_e100": SCR + FLIP_ONLY + EP(100),
    "y8s_scr_flip_none_e200": SCR + FLIP_ONLY + EP(200),
    "y8s_scr_flip_none_e300": SCR + FLIP_ONLY + EP(300),
    # Faster R-CNN R50-FPN v2，COCO 预训练；train_frcnn.py 只做左右翻转
    "frcnn_pre":           ["--weights", "last"],
}
FRCNN = {"frcnn_pre"}   # 这些 exp 用 train_frcnn.py（参数不同）
DEFAULT_EXPS = ["y8s_pre_none", "y8s_pre_band", "y8s_pre_boxmask"]


def ensure_weights(names):
    """多进程同时下载同一个 .pt 会互相覆盖，先在这里下载好（存到 ../detectors/，*.pt 不入库）。"""
    from ultralytics.utils.downloads import attempt_download_asset
    for n in names:
        p = DETCMP / n
        if not p.exists():
            print("downloading", n, flush=True)
            attempt_download_asset(str(p))


def check_pairs(runs):
    """同一 (fold, seed) 下，所有 arm 的初始检测权重和第一个 batch 必须一致（预训练和从零分两组比）。"""
    tree = defaultdict(dict)
    for ri in (runs / "ds2").glob("*/f*_seed*/run_info.json"):
        info = json.load(open(ri, encoding="utf-8"))
        # 同一个模型、同一个增广配置才要求初始权重和第一个 batch 相同（全套增广 vs 只翻转的第一个 batch 本来就不同）
        extra = (info.get("args") or {}).get("extra", "{}")
        key = (info.get("fold"), info.get("seed"), "pre" if info.get("pretrained") else "scr",
               str(info.get("model")), "flip" if '"mosaic":0' in extra.replace(" ", "") else "full")
        tree[key][info["exp"]] = info.get("fingerprint") or {}
    bad = 0
    for key in sorted(tree, key=str):
        fps = tree[key]
        if len(fps) < 2:   # 只有一个 arm（如 Faster R-CNN，没有 fingerprint）就没有可配对检查的
            continue
        w = {e: f.get("init_det_weights") for e, f in fps.items()}
        b = {e: f.get("first_batch") for e, f in fps.items()}
        ok = len(set(w.values())) == 1 and len(set(b.values())) == 1 and None not in w.values()
        bad += not ok
        if not ok or len(fps) > 1:
            print(("OK  " if ok else "BAD ") + f"fold {key[0]} seed {key[1]} {key[2]}/{key[4]}: " +
                  ", ".join(f"{e}={w[e]}/{b[e]}" for e in sorted(fps)))
    print(f"{len(tree)} (fold, seed) groups, {bad} with mismatching fingerprints")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", default=DEFAULT_EXPS, choices=list(EXPS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--jobs", type=int, default=1, help="并行进程数（5060 Ti 8 GB：1；5090：3–4）")
    ap.add_argument("--workers", type=int, default=1, help="每个进程的 DataLoader worker 数（Windows 上每个 worker 都加载一遍 torch + CUDA DLL，多了会页面文件不足）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 100 epoch（只用于冒烟）")
    ap.add_argument("--plots", action="store_true", help="保留 ultralytics 的图（含 train_batch*_band.jpg 对齐检查图）")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="只做配对检查")
    args = ap.parse_args()
    runs = Path(args.runs)
    if not runs.is_absolute():
        runs = (Path.cwd() / runs).resolve()
    if args.check:
        sys.exit(1 if check_pairs(runs) else 0)
    runs.mkdir(parents=True, exist_ok=True)
    log = open(runs / "run_cv.log", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    # 顺序：同一 (fold, seed) 的各 arm 相邻，配对的几个 arm 尽量一起完成
    jobs = []
    for fold in args.folds:
        for seed in args.seeds:
            for exp in args.exps:
                data = DATASETS / f"ds2_f{fold}_s100_hbatbox"
                run_dir = runs / "ds2" / exp / f"f{fold}_seed{seed}"
                if (run_dir / "metrics.json").exists():
                    continue
                if not data.exists():
                    raise SystemExit(f"dataset dir missing: {data}\n  -> cd ../../shared && python prepare_data.py "
                                     f"--dataset ds2 --fold {fold} --box-source hbat")
                base = [PY, "-X", "utf8", None, "--data", str(data), "--exp", exp, "--seed", str(seed),
                        "--run-name", f"f{fold}_seed{seed}", "--runs", str(runs), "--workers", str(args.workers)]
                if exp in FRCNN:
                    base[3] = str(DETCMP / "train_frcnn.py")
                    cmd = base + ["--device", f"cuda:{args.device}", *EXPS[exp]]
                else:
                    base[3] = str(DETCMP / "train_ultralytics.py")
                    cmd = base + ["--device", args.device, *COMMON, *EXPS[exp]]
                    if not args.plots:
                        cmd += ["--no-plots"]
                if args.epochs is not None:
                    cmd += ["--epochs", str(args.epochs)]
                cmd += args.extra
                jobs.append((exp, fold, seed, cmd))
    say(f"{len(jobs)} runs to do  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    if args.dry_run:
        for exp, fold, seed, cmd in jobs:
            print(exp, fold, seed, " ".join(cmd[3:]))
        return
    ensure_weights(sorted({c[c.index("--model") + 1] for *_, c in jobs
                           if "--model" in c and c[c.index("--model") + 1].endswith(".pt")}))

    running = []   # (proc, exp, fold, seed, logfile, t0)
    t_start = time.time()
    done = 0
    while jobs or running:
        while jobs and len(running) < args.jobs:
            exp, fold, seed, cmd = jobs.pop(0)
            lf = open(runs / f"log_{exp}_f{fold}_seed{seed}.txt", "w", encoding="utf-8")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(DETCMP))
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
    check_pairs(runs)


if __name__ == "__main__":
    main()
