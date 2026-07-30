from __future__ import annotations

from pathlib import Path

from visualize_sgy_batch import (
    AGC_MAX_GAIN,
    AGC_WINDOW_NS,
    BACKGROUND_REMOVAL,
    BACKGROUND_WINDOW,
    DEWOW_WINDOW,
    GAIN_ENABLED,
    MIN_VALID_SAMPLES,
    MUTE_NS,
    PERCENTILE,
    TIME_RANGE_NS_FALLBACK,
    flat_output_name,
    preprocess_data,
    read_radargram,
    render_image,
)


def invert_polarity(data):
    """Flip the sign of the whole trace set. Same reflection events, opposite
    polarity — used as a cheap augmentation for hyperbola recognition, since a
    GPR system's polarity convention depends on antenna wiring and isn't something
    a detector should be sensitive to."""
    return -data


# ── 参数配置 ──────────────────────────────────────────────
INPUT_DIR    = Path(__file__).parent / "01"          # 输入文件夹，和 visualize_sgy_batch.py 保持一致
OUTPUT_DIR   = Path(__file__).parent / "vis_inverted"  # 输出文件夹（极性反转后的图，平铺）
# ─────────────────────────────────────────────────────────


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
        sample_interval_ns = time_range_ns / max(data.shape[0] - 1, 1)
        mute_samples = int(round(MUTE_NS / sample_interval_ns)) if MUTE_NS > 0 else 0
        agc_window_samples = int(round(AGC_WINDOW_NS / sample_interval_ns)) if GAIN_ENABLED else 0

        processed = preprocess_data(
            data,
            BACKGROUND_REMOVAL,
            BACKGROUND_WINDOW,
            DEWOW_WINDOW,
            GAIN_ENABLED,
            agc_window_samples,
            AGC_MAX_GAIN,
            mute_samples,
        )
        inverted = invert_polarity(processed)

        output_path = output_dir / flat_output_name(sgy_path, input_root)
        render_image(inverted, output_path, PERCENTILE)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
