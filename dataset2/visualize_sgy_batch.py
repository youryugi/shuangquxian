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
            dt_ps = segyio.tools.dt(segy_file)  # RadSys/RadarMap SEG-Y variant: field is picoseconds, not microseconds
            ns = data.shape[0]
            time_range_ns = (ns - 1) * dt_ps / 1000.0 if dt_ps > 0 else None
        return data, time_range_ns
    except RuntimeError:
        segy_stream = _read_segy(str(path))
        trace_arrays = [np.asarray(trace.data, dtype=np.float32) for trace in segy_stream.traces]
        max_samples = max(len(trace_array) for trace_array in trace_arrays)
        padded = np.zeros((len(trace_arrays), max_samples), dtype=np.float32)
        for index, trace_array in enumerate(trace_arrays):
            padded[index, : len(trace_array)] = trace_array
        # RadSys/RadarMap SEG-Y variant: sample_interval_in_microseconds field is actually picoseconds
        try:
            dt_ps = segy_stream.binary_file_header.sample_interval_in_microseconds
            ns = padded.shape[1]
            time_range_ns = (ns - 1) * dt_ps / 1000.0 if dt_ps > 0 else None
        except Exception:
            time_range_ns = None
        return padded.T, time_range_ns


def remove_background(data: np.ndarray, window_traces: int) -> np.ndarray:
    """Subtract a local mean trace (moving average across traces) to suppress the
    direct wave and horizontal system ringing. window_traces<=1 or >= n_traces
    falls back to a single whole-profile average trace."""
    n_traces = data.shape[1]
    if window_traces <= 1 or window_traces >= n_traces:
        return data - np.mean(data, axis=1, keepdims=True)
    pad_left = (window_traces - 1) // 2
    pad_right = window_traces - 1 - pad_left
    padded = np.pad(data, ((0, 0), (pad_left, pad_right)), mode="edge")
    kernel = np.ones(window_traces, dtype=np.float32) / float(window_traces)
    background = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return data - background


def dewow(data: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return data
    pad_top = (window_samples - 1) // 2
    pad_bottom = window_samples - 1 - pad_top
    padded = np.pad(data, ((pad_top, pad_bottom), (0, 0)), mode="edge")
    kernel = np.ones(window_samples, dtype=np.float32) / float(window_samples)
    smoothed = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, padded)
    return data - smoothed


def top_mute(data: np.ndarray, mute_samples: int) -> np.ndarray:
    """Crop off the top mute_samples rows entirely rather than zeroing them, so the
    direct-wave/ringing region isn't rendered at all."""
    if mute_samples <= 0:
        return data
    return data[mute_samples:, :]


def agc(data: np.ndarray, window_samples: int, max_gain: float) -> np.ndarray:
    """Automatic gain control: normalize amplitude with a sliding time window so
    deep, attenuated reflections become as visible as shallow ones. Gain is capped
    at max_gain so near-silent regions (e.g. a muted top) aren't blown up to noise."""
    if window_samples <= 1:
        return data
    abs_data = np.abs(data)
    pad_top = (window_samples - 1) // 2
    pad_bottom = window_samples - 1 - pad_top
    padded = np.pad(abs_data, ((pad_top, pad_bottom), (0, 0)), mode="edge")
    kernel = np.ones(window_samples, dtype=np.float32) / float(window_samples)
    local_level = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, padded)
    reference_level = float(np.mean(abs_data)) or 1.0
    gain = np.clip(reference_level / np.maximum(local_level, reference_level / max_gain), 1.0 / max_gain, max_gain)
    return data * gain


def preprocess_data(
    data: np.ndarray,
    apply_background_removal: bool,
    background_window: int,
    dewow_window: int,
    apply_gain: bool,
    agc_window: int,
    agc_max_gain: float,
    mute_samples: int,
) -> np.ndarray:
    processed = data.astype(np.float32, copy=True)
    if dewow_window > 1:
        processed = dewow(processed, dewow_window)
    if apply_background_removal:
        processed = remove_background(processed, background_window)
    if apply_gain:
        processed = agc(processed, agc_window, agc_max_gain)
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
INPUT_DIR    = Path(__file__).parent   # 输入文件夹（递归搜索 .sgy，目前只处理 01 文件夹）
OUTPUT_DIR   = r"./vis"      # 输出文件夹（平铺，不建子目录）
TIME_RANGE_NS_FALLBACK = 50.0  # 文件头缺少有效 dt 时的兜底时间范围（纳秒）
MIN_VALID_SAMPLES   = 100    # 每道采样点数低于此值视为文件头损坏，直接跳过（正常应为 512）
BACKGROUND_REMOVAL  = True   # 是否做背景去除（滑动窗口平均道相减，压制直达波/水平振铃，业内标准做法）
BACKGROUND_WINDOW   = 201     # 背景去除滑动窗口宽度（道数，建议奇数；<=1 或 >= 总道数时退化为整条测线平均）
DEWOW_WINDOW        = 0      # dewow 滑动平均窗口（样本数，0=不做）
GAIN_ENABLED        = False   # 是否做增益补偿（AGC，抬升深层弱反射，让深浅振幅更均衡）
AGC_WINDOW_NS       = 10.0   # AGC 滑动时间窗口（纳秒）
AGC_MAX_GAIN        = 20.0   # 增益上限（避免把静区/弱噪声区放大过头）
MUTE_NS             = 4  # 压制直达波：裁掉前 N 纳秒（0=不做；实测 01 文件夹残留振铃约需 20ns 才能回落到基线附近）
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
        if data.shape[0] < MIN_VALID_SAMPLES:
            print(f"[skip] {sgy_path.name}: only {data.shape[0]} samples/trace (corrupt SEG-Y header), skipping")
            continue
        if time_range_ns is None or time_range_ns <= 0:
            time_range_ns = TIME_RANGE_NS_FALLBACK
            print(f"[warn] {sgy_path.name}: no valid dt in header, using TIME_RANGE_NS_FALLBACK={TIME_RANGE_NS_FALLBACK} ns")
        else:
            print(f"[info] {sgy_path.name}: time_range={time_range_ns:.2f} ns (from header)")
        sample_interval_ns = time_range_ns / max(data.shape[0] - 1, 1)
        mute_samples = int(round(MUTE_NS / sample_interval_ns)) if MUTE_NS > 0 else 0
        agc_window_samples = int(round(AGC_WINDOW_NS / sample_interval_ns)) if GAIN_ENABLED else 0
        data = preprocess_data(
            data,
            BACKGROUND_REMOVAL,
            BACKGROUND_WINDOW,
            DEWOW_WINDOW,
            GAIN_ENABLED,
            agc_window_samples,
            AGC_MAX_GAIN,
            mute_samples,
        )
        output_path = output_dir / flat_output_name(sgy_path, input_root)
        render_image(data, output_path, PERCENTILE)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
