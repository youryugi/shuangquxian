"""并行跑 train_curve.py 的 fold × seed × arm（已完成的跳过）。

    python run_curve.py --exps curve_hough curve_local curve_boxgrid curve_hough_boxpseudo --jobs 8
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent / "shared" / "datasets"

ARMS = {
    "curve_hough":           ["--layout", "hough"],
    "curve_local":           ["--layout", "local"],
    "curve_boxgrid":         ["--layout", "boxgrid"],
    "curve_hough_boxpseudo": ["--layout", "hough", "--targets", "boxpseudo"],
}
# 第 1 轮 config 选择：顶点热图 sigma（在 curve_local 的合并验证集上选，所有 arm 共用）
for _s in ("1", "2", "4"):
    for _a in ("hough", "local"):
        ARMS[f"pilot_{_a}_sig{_s}"] = ["--layout", _a, "--hm-sigma", _s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True, choices=list(ARMS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    runs = Path(args.runs); runs.mkdir(parents=True, exist_ok=True)
    log = open(runs / "run_curve.log", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    jobs = []
    for seed in args.seeds:
        for fold in args.folds:
            for exp in args.exps:
                if (runs / "ds2" / exp / f"f{fold}_seed{seed}" / "metrics.json").exists():
                    continue
                data = DATASETS / f"ds2_f{fold}_s100_hbatbox"
                cmd = [sys.executable, "-X", "utf8", str(HERE / "train_curve.py"), "--data", str(data), "--exp", exp,
                       "--seed", str(seed), "--runs", str(runs), "--workers", "1", *ARMS[exp], *args.extra]
                jobs.append((exp, fold, seed, cmd))
    say(f"{len(jobs)} runs to do ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    running, done, t_start = [], 0, time.time()
    while jobs or running:
        while jobs and len(running) < args.jobs:
            exp, fold, seed, cmd = jobs.pop(0)
            lf = open(runs / f"log_{exp}_f{fold}_seed{seed}.txt", "w", encoding="utf-8")
            running.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE)), exp, fold, seed, lf, time.time()))
        time.sleep(2)
        still = []
        for p, exp, fold, seed, lf, t0 in running:
            if p.poll() is None:
                still.append((p, exp, fold, seed, lf, t0)); continue
            lf.close(); done += 1
            last = (runs / f"log_{exp}_f{fold}_seed{seed}.txt").read_text(encoding="utf-8").strip().splitlines()[-1:]
            say(f"[{done}] {exp} f{fold} seed{seed} rc={p.returncode} {time.time() - t0:.0f}s | {(last[0] if last else '')[:150]}")
        running = still
    say(f"ALL_DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
