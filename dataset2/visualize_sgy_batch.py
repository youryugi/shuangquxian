from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy
import segyio


def read_radargram(path: Path) -> tuple[np.ndarray, float | None]:
    """Returns (data, time_range_ns). time_range_ns is None if header lacks valid dt."""
    try:
        with segyio.open(str(path), ignore_geometry=True) as segy_file:
            traces = segyio.tools.collect(segy_file.trace[:])
            data = np.asarray(traces, dtype=np.float32).T
            dt_us = segyio.tools.dt(segy_file) / 1000.0  # segyio returns dt in microseconds×1000 (i.e. value/1000 = µs)
            ns = data.shape[0]
            time_range_ns = (ns - 1) * dt_us * 1000.0 if dt_us > 0 else None
        return data, time_range_ns
    except RuntimeError:
        segy_stream = _read_segy(str(path))
        trace_arrays = [np.asarray(trace.data, dtype=np.float32) for trace in segy_stream.traces]
        max_samples = max(len(trace_array) for trace_array in trace_arrays)
        padded = np.zeros((len(trace_arrays), max_samples), dtype=np.float32)
        for index, trace_array in enumerate(trace_arrays):
            padded[index, : len(trace_array)] = trace_array
        # obspy stores dt in seconds in binary_file_header.sample_interval
        try:
            dt_s = segy_stream.binary_file_header.sample_interval * 1e-6  # stored as µs
            ns = padded.shape[1]
            time_range_ns = (ns - 1) * dt_s * 1e9 if dt_s > 0 else None
        except Exception:
            time_range_ns = None
        return padded.T, time_range_ns


def remove_background(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=1, keepdims=True)


def dewow(data: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return data
    pad = window_samples // 2
    padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window_samples, dtype=np.float32) / float(window_samples)
    smoothed = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, padded)
    return data - smoothed


def top_mute(data: np.ndarray, mute_samples: int) -> np.ndarray:
    if mute_samples <= 0:
        return data
    muted = data.copy()
    muted[:mute_samples, :] = 0
    return muted


def preprocess_data(
    data: np.ndarray,
    apply_background_removal: bool,
    dewow_window: int,
    mute_samples: int,
) -> np.ndarray:
    processed = data.astype(np.float32, copy=True)
    if dewow_window > 1:
        processed = dewow(processed, dewow_window)
    if apply_background_removal:
        processed = remove_background(processed)
    if mute_samples > 0:
        processed = top_mute(processed, mute_samples)
    return processed


def clip_range(data: np.ndarray, percentile: float) -> tuple[float, float]:
    amplitude = np.abs(data)
    vmax = float(np.percentile(amplitude, percentile))
    if vmax == 0:
        vmax = float(np.max(amplitude) or 1.0)
    return -vmax, vmax


def render_image(
    data: np.ndarray,
    output_path: Path,
    percentile: float,
) -> None:
    vmin, vmax = clip_range(data, percentile)
    n_samples, n_traces = data.shape
    dpi = 100
    fig = plt.figure(figsize=(n_traces / dpi, n_samples / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(
        data,
        cmap="gray",
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


# ── 参数配置 ──────────────────────────────────────────────
INPUT_DIR    = Path(__file__).parent      # 输入文件夹（递归搜索 .sgy）
OUTPUT_DIR   = r"./vis"      # 输出文件夹（平铺，不建子目录）
TIME_RANGE_NS_FALLBACK = 50.0  # 文件头缺少有效 dt 时的兜底时间范围（纳秒）
BACKGROUND_REMOVAL  = False  # 是否做背景去除
DEWOW_WINDOW        = 0      # dewow 滑动平均窗口（样本数，0=不做）
MUTE_NS             = 5.0    # 压制直达波：前 N 纳秒置零（0=不做）
PERCENTILE          = 99.0   # 振幅截断百分位数
# ─────────────────────────────────────────────────────────


def flat_output_name(sgy_path: Path, input_root: Path) -> str:
    parts = sgy_path.relative_to(input_root).with_suffix("").parts
    return "_".join(parts) + ".png"


def main() -> None:
    input_root = Path(INPUT_DIR).expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sgy_files = sorted(input_root.rglob("*.sgy"))
    if not sgy_files:
        raise FileNotFoundError(f"No .sgy files found under {input_root}")

    for sgy_path in sgy_files:
        data, time_range_ns = read_radargram(sgy_path)
        if time_range_ns is None or time_range_ns <= 0:
            time_range_ns = TIME_RANGE_NS_FALLBACK
            print(f"[warn] {sgy_path.name}: no valid dt in header, using TIME_RANGE_NS_FALLBACK={TIME_RANGE_NS_FALLBACK} ns")
        else:
            print(f"[info] {sgy_path.name}: time_range={time_range_ns:.2f} ns (from header)")
        sample_interval_ns = time_range_ns / max(data.shape[0] - 1, 1)
        mute_samples = int(round(MUTE_NS / sample_interval_ns)) if MUTE_NS > 0 else 0
        data = preprocess_data(data, BACKGROUND_REMOVAL, DEWOW_WINDOW, mute_samples)
        output_path = output_dir / flat_output_name(sgy_path, input_root)
        render_image(data, output_path, PERCENTILE)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
