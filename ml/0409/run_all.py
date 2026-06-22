"""
顺序跑多个实验脚本：跑完一个自动跑下一个，记录每个的耗时和成败。
某个失败也继续跑后面的（想"失败即停"见文件末尾说明）。

运行（用 gpr 环境的 python）：
  C:/Users/79152/.conda/envs/gpr/python.exe run_all.py
"""
import os
import sys
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
PY    = sys.executable   # 用当前解释器（你用 gpr 的 python 跑本脚本，子进程也用 gpr）

# 想跑哪些、什么顺序，改这里即可
SCRIPTS = [
    "yolo_scratch.py",
    "yolo_pretrained.py",
    # "bbox_cnn_baseline.py",
    # "0616-1.py",
]


def main():
    results = []
    for s in SCRIPTS:
        print(f"\n{'='*70}\n>>> RUN  {s}\n{'='*70}", flush=True)
        t0 = time.time()
        ret = subprocess.run([PY, s], cwd=_HERE)
        dt  = (time.time() - t0) / 60.0
        status = "OK" if ret.returncode == 0 else f"FAILED(code={ret.returncode})"
        print(f"<<< {s}  {status}  用时 {dt:.1f} min", flush=True)
        results.append((s, status, dt))

    print(f"\n{'='*70}\n全部完成：")
    for s, status, dt in results:
        print(f"  {s:<28} {status:<18} {dt:.1f} min")


if __name__ == "__main__":
    main()
