"""Parameters, FLOPs and inference latency for every finished run (for the cost columns of the comparison table).

    python model_stats.py runs/ --imgsz 640 [--device cuda:0] [--iters 100]

Writes <run_dir>/model_stats.json and runs/model_stats.csv. Latency = batch 1, fp32, model forward only
(no NMS, no image decoding), median over --iters iterations after 20 warm-up iterations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "shared"   # 共享的数据导出与评估代码（common.py、evaluate.py、prepare_data.py、datasets/）
for _p in (HERE, SHARED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def count_flops(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() / 1e9


def latency_ms(model, x, iters=100, warmup=20):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if x[0].is_cuda if isinstance(x, list) else x.is_cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            model(x)
            if x[0].is_cuda if isinstance(x, list) else x.is_cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def load_run_model(run_dir: Path, info: dict, device):
    """Return (callable model, input) for an ultralytics or torchvision run."""
    imgsz = int(info.get("imgsz", 640))
    if info.get("framework", "").startswith("ultralytics"):
        import band_yolo  # noqa: F401  (unpickling BandAuxDetectionModel)
        from ultralytics.nn.tasks import load_checkpoint
        model, _ = load_checkpoint(info["best_weights"])
        model = model.float().to(device).eval()
        if hasattr(model, "aux_head"):  # approach A: the aux head is not needed at inference
            model.aux_head = torch.nn.Identity()
        model.fuse() if hasattr(model, "fuse") else None
        return model, torch.zeros(1, 3, imgsz, imgsz, device=device)
    if "fasterrcnn" in info.get("model", ""):
        from train_frcnn import build_model
        m = build_model(bool(info.get("pretrained", True)), imgsz)
        m.load_state_dict(torch.load(run_dir / "train" / "best.pt", map_location="cpu"))
        m = m.to(device).eval()
        return m, [torch.zeros(3, imgsz, imgsz, device=device)]
    raise ValueError(f"unknown run type: {info.get('framework')} / {info.get('model')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_root")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for info_p in sorted(Path(args.runs_root).rglob("run_info.json")):
        run_dir = info_p.parent
        out_p = run_dir / "model_stats.json"
        info = json.load(open(info_p, encoding="utf-8"))
        if out_p.exists() and not args.force:
            rows.append(json.load(open(out_p, encoding="utf-8")))
            continue
        try:
            model, x = load_run_model(run_dir, info, device)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            try:
                gflops = count_flops(model, x)
            except Exception as e:
                print(f"  FLOPs failed for {run_dir}: {e}")
                gflops = None
            lat = latency_ms(model, x, args.iters)
            st = dict(dataset=info.get("dataset"), exp=info.get("exp"), seed=info.get("seed"), model=info.get("model"),
                      imgsz=info.get("imgsz"), params_M=round(n_params, 3),
                      GFLOPs=None if gflops is None else round(gflops, 2), latency_ms_b1=round(lat, 2),
                      device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", run_dir=str(run_dir))
            json.dump(st, open(out_p, "w", encoding="utf-8"), indent=1)
            rows.append(st)
            print(f"{info.get('dataset')}/{info.get('exp')}/seed{info.get('seed')}: {n_params:.2f} M, "
                  f"{st['GFLOPs']} GFLOPs, {lat:.2f} ms")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  skip {run_dir}: {e}")
    if rows:
        keys = list(rows[0].keys())
        with open(Path(args.runs_root) / "model_stats.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("->", Path(args.runs_root) / "model_stats.csv")


if __name__ == "__main__":
    main()
