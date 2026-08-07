"""
One-off: merge the two polarity-normal labeled sources into a single flat
YOLO dataset with one unified "hyperbola" class.

Sources:
  dataset2/vis      - flat, folder "01" only, class id 0 named "object".
  dataset2/vis-all  - nested (covers 01..013), labels scattered under leaf
                       Radargrams/Radarmaps folders that each carry their own
                       classes.txt. Class ids/names vary by an annotation-tool
                       mistake (some folders have ["utilities","utility"] and
                       only ever use id 1; others have just ["utility"], id 0)
                       -- but every labeled box is the same object: a
                       hyperbola, so all class ids are remapped to 0 here.

Flat names for the vis-all side are built relative to each image's own
top-level folder (e.g. "02.1_Radargrams_Path1.png"), matching the same
convention dataset2/vis already uses and what
dataset2/polarity_features.py's flat_output_name() produces one segment
deeper -- so train_yolo_bbox.py's --invert-aug lookup keeps matching these
names against dataset2/vis_inverted_all without any extra logic.

Run once (re-run to rebuild from scratch after re-annotating):
    python build_merged_dataset.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

_HERE = Path(__file__).parent
VIS_DIR = (_HERE / ".." / ".." / "dataset2" / "vis").resolve()
VIS_ALL_DIR = (_HERE / ".." / ".." / "dataset2" / "vis-all").resolve()
OUTPUT_DIR = _HERE / "jixing-merged"
CLASS_NAME = "hyperbola"


def remap_label(src_label_path: Path, dst_label_path: Path) -> None:
    """Copy a YOLO label file, forcing every box's class id to 0."""
    out_lines = []
    for line in src_label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        parts[0] = "0"
        out_lines.append(" ".join(parts))
    dst_label_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def collect_vis() -> list[tuple[Path, Path, str]]:
    """(image_path, label_path, flat_name) for dataset2/vis, already flat."""
    pairs = []
    for img_path in sorted(VIS_DIR.glob("*.png")):
        label_path = img_path.with_suffix(".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            pairs.append((img_path, label_path, img_path.name))
    return pairs


def collect_vis_all() -> list[tuple[Path, Path, str]]:
    """Same, for every leaf folder under dataset2/vis-all that has a
    classes.txt (i.e. was actually annotated)."""
    pairs = []
    for classes_path in sorted(VIS_ALL_DIR.rglob("classes.txt")):
        leaf_dir = classes_path.parent
        top_dir = leaf_dir.relative_to(VIS_ALL_DIR).parts[0]
        top_path = VIS_ALL_DIR / top_dir
        for img_path in sorted(leaf_dir.glob("*.png")):
            label_path = img_path.with_suffix(".txt")
            if not (label_path.exists() and label_path.stat().st_size > 0):
                continue
            flat_name = "_".join(img_path.relative_to(top_path).parts)
            pairs.append((img_path, label_path, flat_name))
    return pairs


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    vis_pairs = collect_vis()
    vis_all_pairs = collect_vis_all()

    seen: dict[str, Path] = {}
    for img_path, label_path, flat_name in vis_pairs + vis_all_pairs:
        if flat_name in seen:
            raise RuntimeError(f"name collision: {flat_name} <- {img_path} and {seen[flat_name]}")
        seen[flat_name] = img_path
        shutil.copy2(img_path, OUTPUT_DIR / flat_name)
        remap_label(label_path, OUTPUT_DIR / (Path(flat_name).stem + ".txt"))

    (OUTPUT_DIR / "classes.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")

    print(f"[merge] vis: {len(vis_pairs)} labeled images")
    print(f"[merge] vis-all: {len(vis_all_pairs)} labeled images")
    print(f"[merge] total merged: {len(vis_pairs) + len(vis_all_pairs)} -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
