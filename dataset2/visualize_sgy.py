from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy
import segyio


def read_radargram(path: Path) -> np.ndarray:
    try:
        with segyio.open(str(path), ignore_geometry=True) as segy_file:
            traces = segyio.tools.collect(segy_file.trace[:])
        return np.asarray(traces, dtype=np.float32).T
    except RuntimeError:
        segy_stream = _read_segy(str(path))
        trace_arrays = [np.asarray(trace.data, dtype=np.float32) for trace in segy_stream.traces]
        max_samples = max(len(trace_array) for trace_array in trace_arrays)
        padded = np.zeros((len(trace_arrays), max_samples), dtype=np.float32)
        for index, trace_array in enumerate(trace_arrays):
            padded[index, : len(trace_array)] = trace_array
        return padded.T


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
    title: str,
    percentile: float,
    time_range_ns: float | None,
) -> None:
    vmin, vmax = clip_range(data, percentile)
    fig, axis = plt.subplots(figsize=(12, 6), dpi=180)
    extent = None
    ylabel = "Sample"
    if time_range_ns is not None:
        extent = [0, data.shape[1] - 1, time_range_ns, 0]
        ylabel = "Time (ns)"
    image = axis.imshow(
        data,
        cmap="gray",
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("Trace")
    axis.set_ylabel(ylabel)
    fig.colorbar(image, ax=axis, label="Amplitude")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def iter_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.sgy"))
    return [input_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SEG-Y radargrams as PNG images.")
    parser.add_argument("input", nargs="?", default=".", help="SEG-Y file or folder containing .sgy files")
    parser.add_argument("--output-dir", default="visualizations", help="Directory for output PNG files")
    parser.add_argument("--time-range-ns", type=float, default=50.0, help="Total two-way travel time in nanoseconds")
    parser.add_argument(
        "--background-removal",
        action="store_true",
        help="Subtract the trace-wise horizontal background to suppress direct wave and ringing",
    )
    parser.add_argument(
        "--dewow-window",
        type=int,
        default=0,
        help="Moving-average window in samples for dewow filtering before background removal",
    )
    parser.add_argument(
        "--mute-ns",
        type=float,
        default=5,
        help="Zero the first N nanoseconds to mute the direct wave arrival",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Symmetric amplitude clip percentile for image contrast",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sgy_files = iter_inputs(input_path)
    if not sgy_files:
        raise FileNotFoundError(f"No .sgy files found in {input_path}")

    for sgy_path in sgy_files:
        data = read_radargram(sgy_path)
        sample_interval_ns = args.time_range_ns / max(data.shape[0] - 1, 1)
        mute_samples = int(round(args.mute_ns / sample_interval_ns)) if args.mute_ns > 0 else 0
        data = preprocess_data(data, args.background_removal, args.dewow_window, mute_samples)
        output_path = output_dir / f"{sgy_path.stem}.png"
        render_image(data, output_path, sgy_path.name, args.percentile, args.time_range_ns)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()