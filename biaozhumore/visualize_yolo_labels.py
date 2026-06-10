import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def choose_directory(title: str, fallback_prompt: str) -> Path:
    """Open a folder picker; fallback to console input if GUI is unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass

    user_input = input(fallback_prompt).strip().strip('"').strip("'")
    if not user_input:
        raise RuntimeError("No directory selected.")
    return Path(user_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO txt labels on images from another folder."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory that contains image files.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory that contains YOLO txt files (same stem as image).",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Show images even if label txt is missing.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save rendered images to this directory.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Rectangle line width.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=10,
        help="Label text size.",
    )
    return parser.parse_args()


def read_yolo_txt(label_path: Path):
    rows = []
    if not label_path.exists():
        return rows

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] Skip malformed line {line_no} in {label_path}")
                continue

            try:
                cls_id = int(float(parts[0]))
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(f"[WARN] Skip non-numeric line {line_no} in {label_path}")
                continue

            rows.append((cls_id, x_c, y_c, w, h))
    return rows


def to_xyxy_norm(x_c: float, y_c: float, w: float, h: float):
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return x1, y1, x2, y2


def draw_one(ax, image_w: int, image_h: int, label_rows, line_width: float, font_size: int):
    for cls_id, x_c, y_c, w, h in label_rows:
        x1n, y1n, x2n, y2n = to_xyxy_norm(x_c, y_c, w, h)

        x1 = max(0.0, min(image_w - 1.0, x1n * image_w))
        y1 = max(0.0, min(image_h - 1.0, y1n * image_h))
        x2 = max(0.0, min(image_w - 1.0, x2n * image_w))
        y2 = max(0.0, min(image_h - 1.0, y2n * image_h))

        rect = patches.Rectangle(
            (x1, y1),
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
            linewidth=line_width,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(0.0, y1 - 3),
            f"cls {cls_id}",
            color="yellow",
            fontsize=font_size,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
        )


def collect_images(images_dir: Path, exts_csv: str):
    exts = {e.strip().lower() for e in exts_csv.split(",") if e.strip()}
    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    images.sort()
    return images


def save_figure(fig, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)


def wait_for_nav_key(fig) -> str:
    """Wait for a key press in figure window and return navigation action."""
    state = {"action": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key in {"right", " ", "enter", "n", "d"}:
            state["action"] = "next"
            plt.close(fig)
        elif key in {"left", "p", "a"}:
            state["action"] = "prev"
            plt.close(fig)
        elif key in {"escape", "q"}:
            state["action"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    # Block until the window is closed by a key press or manual close.
    plt.show()
    if state["action"] is None:
        return "quit"
    return state["action"]


def main():
    args = parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir

    if images_dir is None:
        images_dir = choose_directory(
            title="Select images folder",
            fallback_prompt="请输入图片文件夹路径: ",
        )
    if labels_dir is None:
        labels_dir = choose_directory(
            title="Select YOLO labels folder",
            fallback_prompt="请输入标签文件夹路径: ",
        )

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"labels-dir not found: {labels_dir}")

    images = collect_images(images_dir, args.exts)
    if not images:
        raise RuntimeError(f"No images found in: {images_dir}")

    kept = []
    for img_path in images:
        txt_path = labels_dir / f"{img_path.stem}.txt"
        if args.show_missing or txt_path.exists():
            kept.append((img_path, txt_path))

    if not kept:
        raise RuntimeError(
            "No image/label pairs to show. Use --show-missing to include images without txt labels."
        )

    total = len(kept)
    print(f"[INFO] Total images to visualize: {total}")
    print("[INFO] Controls in image window:")
    print("       next: Right / Space / Enter / N")
    print("       prev: Left / P")
    print("       quit: Esc / Q")

    idx = 0
    while 0 <= idx < total:
        img_path, txt_path = kept[idx]
        label_rows = read_yolo_txt(txt_path)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.axis("off")

        if label_rows:
            draw_one(ax, w, h, label_rows, args.line_width, args.font_size)
            title = f"[{idx + 1}/{total}] {img_path.name} | labels: {len(label_rows)}"
        else:
            title = f"[{idx + 1}/{total}] {img_path.name} | labels: missing/empty"
        ax.set_title(title)

        if args.save_dir is not None:
            out_path = args.save_dir / img_path.name
            save_figure(fig, out_path)
            print(f"[SAVE] {out_path}")

        plt.tight_layout()
        action = wait_for_nav_key(fig)

        if action == "quit":
            break
        if action == "prev":
            idx = max(0, idx - 1)
        else:
            idx += 1


if __name__ == "__main__":
    main()
