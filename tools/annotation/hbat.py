"""HBAT: hyperbola-band annotation tool (paper Fig. 1).

Usage
-----
    python hbat.py                                   # pick folder in the GUI
    python hbat.py --folder DIR                      # open DIR directly
    python hbat.py --folder DIR --annotator B        # write annotations_B.json (+ timelog_B.jsonl)
    python hbat.py --folder DIR --gt PATH            # official GT boxes: a dir of .txt (YOLO or DOTA/OBB)
                                                     # or a JSON {image: [{x1,y1,width,height}|[x1,y1,x2,y2]]}
    python hbat.py --folder DIR --hide-gt            # load GT but start hidden (annotation study)
    python hbat.py --folder DIR --gt none            # never load GT boxes
    python hbat.py --folder DIR --extend             # start with the extended canvas on (toggle in the GUI):
                                                     # grey margin left/right 20 % and top 50 %, none at the
                                                     # bottom (--pad-side / --pad-top to change)
    python hbat.py --folder DIR --out OUTDIR         # write annotations / timelog / rect json to OUTDIR
                                                     # (default: next to the images; use --out to keep
                                                     # official data folders untouched)

Without --gt the tool looks for  <folder>/../labelTxt  (Roboflow YOLOv5-OBB export, when the
folder is named "images"), then <folder>/gt_boxes.json, <folder>/../gt_boxes.json, and any
directory whose name starts with "gt_boxes" next to the folder or next to its parent.

Files written to --out (default: next to the images)
----------------------------------------------------
    annotations[_<annotator>].json   hyperbola-band annotations (one list per image)
    annotations_rect[_<annotator>].json  derived boxes (clipped to the image), for comparison with GT
    timelog[_<annotator>].jsonl      one JSON line per event (session_start, image_enter/leave,
                                     add/update/delete) with durations in seconds

Annotation record (per object)
------------------------------
    label, x_vertex, y_vertex, slope, span, thickness      band parameters (slope/span/thickness are
                                                           null when param_status == "unparam")
    tail             pixels the band is extended *downward* beyond the first-arc ribbon, to cover the
                     multiples / ringing column hanging under a strong reflector (0 = none; legacy
                     records without the field read as 0). The apex, slope and span still describe
                     the first (shallowest) arc only, so geometry is comparable across objects;
                     tail only enlarges the mask / derived box. ringing := tail > 0
    reaches_bottom   True when the arms run off the image bottom; span was then capped automatically
                     to the smallest value giving the same in-image band (bottom = end of the time
                     window, not truncation; left/right/top exits keep the annotated span)
    gt_box_id        index of the official box this band belongs to (null if no GT)
    visibility       clear | medium | weak
    completeness     both | single | apex
    distortion       fit | deviates
    vertex_hidden    True when the apex lies inside the image but cannot be seen and was inferred
                     from the arms (apex outside the image is detected from the coordinates instead)
    extended_canvas  True when the object was saved with the extended canvas on (geometry outside the
                     image kept). With it off, the vertex is clamped into the image and the span capped
                     at the left/right edges on save, so nothing outside the image is stored
    param_status     ok | rough | unparam
"""
import os
import sys
import json
import math
import time
import argparse
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

RECT_FILL     = (0, 180, 255, 60)
RECT_OUTLINE  = (0, 180, 255, 200)
RECT_SEL      = (255, 200, 0, 220)
BAND_STYLE_TEXT = {"fill": "带显示: 填充", "outline": "带显示: 轮廓", "hidden": "带显示: 隐藏"}
MARGIN_BG     = (55, 55, 55)          # grey margin around the image (vertex may sit there)
MARGIN_EDGE   = (230, 230, 230)       # outline of the true image border
GT_OUTLINE    = (255, 0, 255, 220)
GT_TEXT       = (255, 0, 255, 255)

# Per-object labels: key -> list of (code, display text). First entry is the default.
LABEL_SCHEMA = {
    "visibility":   [("clear", "清晰 clear"), ("medium", "一般 medium"), ("weak", "弱 weak")],
    "completeness": [("both", "双翼 both"), ("single", "单翼 single"), ("apex", "仅顶点 apex")],
    "distortion":   [("fit", "符合 fit"), ("deviates", "明显偏离 deviates")],
    "param_status": [("ok", "正常 ok"), ("rough", "勉强 rough"), ("unparam", "无法参数化 unparam")],
}
LABEL_TITLES = {"visibility": "可见度", "completeness": "完整度",
                "distortion": "畸变", "param_status": "参数化"}
LABEL_DEFAULTS = {k: v[0][0] for k, v in LABEL_SCHEMA.items()}
BAND_KEYS = ("slope", "span", "thickness")


# ---------------------------------------------------------------------------
# GPR true-hyperbola model (single shape parameter: slope)
#
#   A GPR reflection from a point / cylindrical target is a hyperbola whose
#   physical DOF are only (x0, t0, v): apex position, apex two-way time and
#   wave velocity.  With the image top taken as the surface (time-zero), the
#   apex depth t0 equals the vertical semi-axis, so:
#
#       a = y_vertex                 (vertical semi-axis = apex depth)
#       b = y_vertex / slope         (slope = a/b = 2/v encodes velocity)
#
#   Centerline (downward-opening upper branch, apex at the top):
#       y_c(x) = y_v + a * ( sqrt(1 + ((x - x_v) / b)^2) - 1 )
#
#   So once the apex is clicked, a single "slope" fully fixes the shape; the
#   apex curvature follows automatically (kappa = slope^2 / (2 * y_v)).
#
#   Parameters:
#       x_vertex, y_vertex : apex pixel coordinates (may lie outside the image
#                            when only one arm is visible)
#       slope              : asymptote slope a/b (velocity)
#       span               : horizontal extent used to draw / truncate the arms
#       thickness          : vertical band thickness
#       tail               : extra downward extent (ringing / multiples), default 0
#
#   The band is a vertical ribbon around the first arc, optionally prolonged
#   downward by the tail:
#       { (x, y) : -thickness/2 <= y - y_c(x) <= thickness/2 + tail }.
#   With tail = 0 this is the symmetric ribbon |y - y_c(x)| <= thickness/2.
#
#   Everything that draws a band (GUI preview, mask export, thickness sweeps)
#   must go through hyperbola_band_polygon() / band_mask() below so that the
#   supervision masks are pixel-identical to what the annotator saw.
# ---------------------------------------------------------------------------

def hyperbola_ab(y_vertex, slope):
    """Physical semi-axes from apex depth and slope (surface = image top)."""
    a = max(1.0, abs(y_vertex))
    s = max(1e-3, slope)
    return a, a / s


def normalize_obj(obj):
    """Normalize legacy fields (width->span, a/b/height -> slope) and fill label defaults."""
    obj = dict(obj)
    obj.setdefault("label", "hyperbola")
    # Legacy field rename: width -> span (horizontal draw extent).
    if "span" not in obj and "width" in obj:
        obj["span"] = obj["width"]
    if "slope" not in obj:
        if "a" in obj and obj.get("b"):
            # Previous two-semi-axis schema.
            obj["slope"] = round(obj["a"] / obj["b"], 2)
        elif "height" in obj and obj.get("span"):
            # Old parabola schema y_v + height * (dx)^2.
            half_w = max(1.0, obj["span"] / 2.0)
            obj["slope"] = round(2.0 * obj["height"] / half_w, 2)
        else:
            obj["slope"] = None
    obj.setdefault("span", None)
    obj.setdefault("thickness", None)
    obj.setdefault("gt_box_id", None)
    try:
        obj["tail"] = max(0.0, float(obj.get("tail") or 0.0))
    except (TypeError, ValueError):
        obj["tail"] = 0.0
    obj.setdefault("vertex_hidden", False)
    obj.setdefault("extended_canvas", False)
    for k, d in LABEL_DEFAULTS.items():
        obj.setdefault(k, d)
    if any(obj.get(k) is None for k in BAND_KEYS):
        obj["param_status"] = "unparam"
        for k in BAND_KEYS:
            obj[k] = None
    return obj


def has_band(obj):
    return all(obj.get(k) is not None for k in BAND_KEYS)


def hyperbola_band_polygon(obj, n_points=None, thickness_scale=1.0, min_thickness=1.0,
                           tail_scale=1.0):
    """Band polygon and centerline in image pixel coordinates.

    Returns (polygon, centerline) or (None, None) when the object has no band.
    The polygon is upper edge (left->right) followed by the lower edge (right->left); the lower
    edge sits thickness/2 + tail below the centerline (tail_scale = 0 drops the tail, e.g. to
    rasterize the first-arc ribbon alone or to turn the tail into an ignore region).
    By default the curve is sampled at every integer x inside the span plus the two end points
    and the apex, i.e. on a grid fixed to the image, so the in-image polygon does not depend on
    the span (e.g. before/after bottom capping). Pass n_points for uniform sampling instead.
    thickness_scale / min_thickness exist for the sensitivity sweeps; the GUI
    always uses 1.0 / 1.0.
    """
    obj = normalize_obj(obj)
    if not has_band(obj):
        return None, None
    x_vertex  = obj["x_vertex"]
    y_vertex  = obj["y_vertex"]
    a, b      = hyperbola_ab(y_vertex, obj["slope"])
    span      = max(2.0, obj["span"])
    thickness = max(min_thickness, obj["thickness"] * thickness_scale)
    tail      = max(0.0, obj["tail"] * tail_scale)
    half_w    = span / 2.0
    x_left    = x_vertex - half_w
    x_right   = x_vertex + half_w
    if n_points is None:
        xs = {x_left, x_right, x_vertex}
        xs.update(range(math.floor(x_left) + 1, math.ceil(x_right)))
        xs = sorted(xs)
    else:
        xs = [x_left + (x_right - x_left) * i / n_points for i in range(n_points + 1)]

    upper, lower, centerline = [], [], []
    for x in xs:
        yc = y_vertex + a * (math.sqrt(1.0 + ((x - x_vertex) / b) ** 2) - 1.0)
        upper.append((x, yc - thickness / 2.0))
        lower.append((x, yc + thickness / 2.0 + tail))
        centerline.append((x, yc))

    return upper + list(reversed(lower)), centerline


def band_mask(obj, img_size, **kw):
    """Binary band mask (PIL 'L', 0/255) rasterized exactly like the GUI preview."""
    w, h = img_size
    mask = Image.new("L", (w, h), 0)
    polygon, _ = hyperbola_band_polygon(obj, **kw)
    if polygon is not None:
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return mask


def bottom_span(obj, img_h):
    """Smallest span whose band still covers everything visible above the image bottom.

    Beyond |x - x_v| = dx_cap the band's *upper* edge (y_c - thickness/2) is below img_h, so the
    whole vertical slice is outside the image. Any span larger than 2*dx_cap draws exactly the same
    in-image band. Returns None when the arms never reach the bottom (or the object has no band).
    The tail only lowers the *lower* edge, so it does not enter here.
    """
    obj = normalize_obj(obj)
    if not has_band(obj):
        return None
    a, b = hyperbola_ab(obj["y_vertex"], obj["slope"])
    depth = img_h + obj["thickness"] / 2.0 - obj["y_vertex"]   # how far below the apex the upper edge must go
    if depth <= 0:
        return None                     # the band starts below the image: nothing to cap
    q = 1.0 + depth / a
    return 2.0 * b * math.sqrt(q * q - 1.0)


def cap_span_to_bottom(obj, img_size):
    """Bottom exit is the end of the recording time window, not a truncated target.

    If the arms run off the image bottom, shrink span to bottom_span() so that span is
    uniquely defined (it no longer depends on how far the annotator dragged the slider) and
    the derived box is not wider than the visible band. The in-image band is unchanged.
    Sets obj["reaches_bottom"] (True/False). Left/right/top exits are NOT touched: those are
    real truncation and keep the annotated span.
    """
    obj = normalize_obj(obj)
    if not has_band(obj) or img_size is None:
        obj["reaches_bottom"] = False
        return obj
    cap = bottom_span(obj, img_size[1])
    if cap is not None and obj["span"] > cap:
        obj["span"] = round(cap, 2)
        obj["reaches_bottom"] = True
    else:
        obj["reaches_bottom"] = False
    return obj


def clamp_to_image(obj, img_size):
    """Used when the extended canvas is OFF: nothing may be annotated outside the image.

    The vertex is clamped into [0, W] x [0, H], and the span is capped at 2 * max(x_v, W - x_v):
    beyond that both arms are outside the image on the left/right, so a longer span adds nothing.
    (Arms leaving the bottom are handled by cap_span_to_bottom.) Returns (obj, changed).
    """
    obj = normalize_obj(obj)
    w, h = img_size
    changed = False
    xv = min(max(obj["x_vertex"], 0.0), float(w))
    yv = min(max(obj["y_vertex"], 0.0), float(h))
    if (xv, yv) != (obj["x_vertex"], obj["y_vertex"]):
        obj["x_vertex"], obj["y_vertex"] = round(xv, 2), round(yv, 2)
        changed = True
    if has_band(obj):
        side = 2.0 * max(xv, w - xv)
        if obj["span"] > side:
            obj["span"] = round(side, 2)
            changed = True
    return obj, changed


def hyperbola_to_bbox(obj, img_size=None):
    """Axis-aligned box derived from a band; clipped to the image when img_size is given.

    With img_size the span is first capped at the image bottom (see cap_span_to_bottom), so the
    box is never wider than the band that is actually visible.
    Returns None for objects without a band, or whose box falls entirely outside the image.
    """
    obj = normalize_obj(obj)
    if not has_band(obj):
        return None
    if img_size is not None:
        obj = cap_span_to_bottom(obj, img_size)
    half_w = obj["span"] / 2.0
    a, b = hyperbola_ab(obj["y_vertex"], obj["slope"])
    y_edge = obj["y_vertex"] + a * (math.sqrt(1.0 + (half_w / b) ** 2) - 1.0)
    x1 = obj["x_vertex"] - half_w
    x2 = obj["x_vertex"] + half_w
    y1 = obj["y_vertex"] - obj["thickness"] / 2.0
    y2 = y_edge + obj["thickness"] / 2.0 + obj["tail"]
    if img_size is not None:
        w, h = img_size
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 <= x1 or y2 <= y1:
            return None
    return {
        "label":     obj["label"],
        "gt_box_id": obj.get("gt_box_id"),
        "x1":        round(x1, 2),
        "y1":        round(y1, 2),
        "width":     round(x2 - x1, 2),
        "height":    round(y2 - y1, 2),
    }


# ---------------------------------------------------------------------------
# Official GT boxes (reference only; never modified by this tool)
# ---------------------------------------------------------------------------

def _parse_box_txt(path, img_size):
    """One box per line. YOLO: `cls cx cy w h` (normalized). DOTA/OBB: 8 coords, class, difficulty."""
    boxes = []
    W, H = img_size
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            nums = []
            for s in tok:
                try:
                    nums.append(float(s))
                except ValueError:
                    break
            if len(nums) >= 8:
                xs, ys = nums[0:8:2], nums[1:8:2]
                label = tok[8] if len(tok) > 8 else ""
                boxes.append((min(xs), min(ys), max(xs), max(ys), label))
            elif len(nums) == 5:
                cls, cx, cy, w, h = nums
                boxes.append(((cx - w / 2) * W, (cy - h / 2) * H,
                              (cx + w / 2) * W, (cy + h / 2) * H, str(int(cls))))
    return boxes


def _parse_box_json_entry(entries):
    boxes = []
    for e in entries:
        if isinstance(e, dict):
            if "x2" in e:
                boxes.append((e["x1"], e["y1"], e["x2"], e["y2"], str(e.get("label", ""))))
            else:
                boxes.append((e["x1"], e["y1"], e["x1"] + e["width"], e["y1"] + e["height"],
                              str(e.get("label", ""))))
        else:
            x1, y1, x2, y2 = e[:4]
            boxes.append((x1, y1, x2, y2, ""))
    return boxes


class GTSource:
    """Reads official boxes for an image name from a .txt directory or a JSON file."""

    def __init__(self, path):
        self.path = path
        self.json = None
        if os.path.isfile(path) and path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                self.json = json.load(f)

    @staticmethod
    def auto_detect(folder):
        parent = os.path.dirname(os.path.normpath(folder))
        cands = []
        if os.path.basename(os.path.normpath(folder)).lower() == "images":
            cands.append(os.path.join(parent, "labelTxt"))
        cands += [os.path.join(folder, "gt_boxes.json"), os.path.join(parent, "gt_boxes.json")]
        # a sibling (or parent's sibling) directory named gt_boxes*  e.g. additional_utilities/gt_boxes_official_aug6
        for base in (parent, os.path.dirname(parent)):
            try:
                cands += [os.path.join(base, d) for d in sorted(os.listdir(base))
                          if d.lower().startswith("gt_boxes") and os.path.isdir(os.path.join(base, d))]
            except OSError:
                pass
        for c in cands:
            if os.path.exists(c):
                return GTSource(c)
        return None

    def boxes_for(self, image_name, img_size):
        """List of dicts {id, x1, y1, x2, y2, label}; id = line order in the GT file."""
        stem = os.path.splitext(image_name)[0]
        raw = []
        if self.json is not None:
            entries = self.json.get(image_name) or self.json.get(stem) or []
            raw = _parse_box_json_entry(entries)
        else:
            p = os.path.join(self.path, stem + ".txt")
            if os.path.exists(p):
                raw = _parse_box_txt(p, img_size)
        return [dict(id=i, x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=b[4]) for i, b in enumerate(raw)]


def assign_gt_box(obj, gt_boxes, img_size):
    """Pick the GT box a band belongs to.

    Score = area of (clipped band box ∩ GT box) / area of clipped band box. For objects
    without a band the vertex point is used instead. Returns (best_id or None, ambiguous).
    ambiguous is True when a second box scores at least half of the best one.
    """
    if not gt_boxes:
        return None, False
    bb = hyperbola_to_bbox(obj, img_size)
    scores = []
    if bb is not None:
        ax1, ay1 = bb["x1"], bb["y1"]
        ax2, ay2 = ax1 + bb["width"], ay1 + bb["height"]
        area = max(1e-6, bb["width"] * bb["height"])
        for g in gt_boxes:
            iw = max(0.0, min(ax2, g["x2"]) - max(ax1, g["x1"]))
            ih = max(0.0, min(ay2, g["y2"]) - max(ay1, g["y1"]))
            scores.append((iw * ih / area, g["id"]))
    else:
        xv, yv = obj.get("x_vertex"), obj.get("y_vertex")
        if xv is None or yv is None:
            return None, False
        for g in gt_boxes:
            inside = g["x1"] <= xv <= g["x2"] and g["y1"] <= yv <= g["y2"]
            scores.append((1.0 if inside else 0.0, g["id"]))
    scores.sort(reverse=True)
    if not scores or scores[0][0] <= 0:
        return None, False
    ambiguous = len(scores) > 1 and scores[1][0] >= 0.5 * scores[0][0]
    return scores[0][1], ambiguous


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class HyperbolaAnnotator:
    def __init__(self, root, annotator=None, gt_source=None, show_gt=True, folder=None, out_dir=None,
                 extend=False, pad_side=0.2, pad_top=0.5):
        self.root = root
        # extended canvas: grey margin left/right (pad_side) and on top (pad_top), none at the bottom
        self.extend = tk.BooleanVar(value=bool(extend))
        self.pad_side = max(0.0, float(pad_side))
        self.pad_top = max(0.0, float(pad_top))
        self.root.title("HBAT: hyperbola-band annotation tool")
        self.root.geometry("1900x960")

        self.annotator = (annotator or "").strip() or None
        self.out_dir_arg = out_dir               # None = next to the images
        self.out_dir = ""
        self.gt_source_arg = gt_source          # None = auto-detect, "none" = never load
        self.gt_source = None
        self.show_gt = tk.BooleanVar(value=show_gt)
        self.gt_boxes = []

        self.image_dir = ""
        self.image_paths = []
        self.current_index = 0

        self.annotations_path = ""
        self.timelog_path = ""
        self.annotations = {}

        self.original_image = None
        self.current_image_name = None

        self.display_w = 700
        self.display_h = 700
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.left_tk = None
        self.right_tk = None

        self.current_objects = []
        self.selected_object_index = None

        # timing
        self._t_image_enter = None
        self._t_obj_start = None
        self._loading = False
        self._last_obj_s = None          # duration of the last add/update
        self._session_n = 0              # add/update count this session
        self._session_s = 0.0            # summed durations this session

        # "hyperbola" or "rectangle"
        self.view_mode = tk.StringVar(value="hyperbola")

        self.var_x         = tk.DoubleVar(value=200)
        self.var_y         = tk.DoubleVar(value=200)
        self.var_slope     = tk.DoubleVar(value=1.0)  # asymptote slope a/b
        self.var_span      = tk.DoubleVar(value=200)  # horizontal draw extent
        self.var_thickness = tk.DoubleVar(value=12)
        self.var_tail      = tk.DoubleVar(value=0)   # downward extension for ringing / multiples
        self.var_name      = tk.StringVar(value="hyperbola")
        self.var_gt_box    = tk.StringVar(value="auto")
        self.var_labels    = {k: tk.StringVar(value=d) for k, d in LABEL_DEFAULTS.items()}
        self.var_vertex_hidden = tk.BooleanVar(value=False)
        self.band_style  = tk.StringVar(value="outline")   # fill | outline | hidden  (display only)
        self.var_opacity = tk.DoubleVar(value=60)          # % applied to band / vertex alpha (display only)
        self.sliders = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        if folder:
            self.open_folder(folder)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        top_bar = ttk.Frame(self.root)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(top_bar, text="Select Image Folder",      command=self.choose_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Previous",                 command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Next",                     command=self.next_image).pack(side=tk.LEFT, padx=4)
        # jump: group number (1..N) or image name / stem
        self.var_goto = tk.StringVar()
        goto_entry = ttk.Entry(top_bar, textvariable=self.var_goto, width=8)
        goto_entry.pack(side=tk.LEFT, padx=(8, 2))
        goto_entry.bind("<Return>", lambda *_: self.goto_group())
        ttk.Button(top_bar, text="跳转 Go", command=self.goto_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="Save Current Annotations", command=self.save_current_annotations).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Save All",                 command=self.save_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Export Rectangles JSON",   command=self.export_rectangles).pack(side=tk.LEFT, padx=4)

        self.toggle_btn = ttk.Button(top_bar, text="切换视图: 矩形", command=self.toggle_view_mode)
        self.toggle_btn.pack(side=tk.LEFT, padx=12)
        self.gt_btn = ttk.Button(top_bar, text="GT框: 隐藏", command=self.toggle_gt)
        self.gt_btn.pack(side=tk.LEFT, padx=4)
        self.extend_btn = ttk.Button(top_bar, text="拓展画布: 关", command=self.toggle_extend)
        self.extend_btn.pack(side=tk.LEFT, padx=4)
        self._update_extend_btn()

        # band display (view only, never changes the data)
        self.style_btn = ttk.Button(top_bar, text=BAND_STYLE_TEXT[self.band_style.get()], command=self.cycle_band_style)
        self.style_btn.pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(top_bar, text="透明度").pack(side=tk.LEFT)
        ttk.Scale(top_bar, from_=10, to=100, variable=self.var_opacity, orient="horizontal", length=110,
                  command=lambda *_: self.refresh_preview()).pack(side=tk.LEFT, padx=4)
        self._update_gt_btn()

        self.info_label = ttk.Label(top_bar, text="Please select an image folder first")
        self.info_label.pack(side=tk.LEFT, padx=12)

        # live timer, refreshed twice a second
        self.timer_label = ttk.Label(top_bar, text="", foreground="#b00060")
        self.timer_label.pack(side=tk.RIGHT, padx=12)
        self._tick_timer()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_panel = ttk.LabelFrame(image_frame, text="Original Image (+ official GT boxes)")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        self.left_canvas = tk.Canvas(left_panel, width=self.display_w, height=self.display_h, bg="black")
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.left_canvas.bind("<Button-1>", self.on_left_click)

        right_panel = ttk.LabelFrame(image_frame, text="Annotation Preview")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        self.right_canvas = tk.Canvas(right_panel, width=self.display_w, height=self.display_h, bg="black")
        self.right_canvas.pack(fill=tk.BOTH, expand=True)

        # ---- control panel ----
        self.control_frame = ttk.LabelFrame(main_frame, text="Parameter Controls")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        row = 0
        ttk.Label(self.control_frame, text="Class Name").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(self.control_frame, textvariable=self.var_name, width=18).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        row += 1

        self._make_slider(self.control_frame, "x_vertex",  self.var_x,         row, 0, 1000); row += 1
        self._make_slider(self.control_frame, "y_vertex",  self.var_y,         row, 0, 1000); row += 1
        self._make_slider(self.control_frame, "slope",     self.var_slope,     row, 0.1, 20);  row += 1
        self._make_slider(self.control_frame, "span",      self.var_span,      row, 10, 900); row += 1
        self._make_slider(self.control_frame, "thickness", self.var_thickness, row, 1, 100);  row += 1
        self._make_slider(self.control_frame, "tail",      self.var_tail,      row, 0, 400);  row += 1
        self.span_note = ttk.Label(self.control_frame, text="", foreground="#0060b0")
        self.span_note.grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        # GT box assignment
        ttk.Label(self.control_frame, text="GT box").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.gt_combo = ttk.Combobox(self.control_frame, textvariable=self.var_gt_box, width=16,
                                     state="readonly", values=["auto", "none"])
        self.gt_combo.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        self.gt_combo.bind("<<ComboboxSelected>>", self._on_gt_combo_selected)
        row += 1

        # Per-object labels
        label_frame = ttk.LabelFrame(self.control_frame, text="Labels (defaults = first option)")
        label_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
        row += 1
        for r, (key, options) in enumerate(LABEL_SCHEMA.items()):
            ttk.Label(label_frame, text=LABEL_TITLES[key]).grid(row=r, column=0, sticky="w", padx=4, pady=2)
            fr = ttk.Frame(label_frame)
            fr.grid(row=r, column=1, sticky="w")
            for code, text in options:
                ttk.Radiobutton(fr, text=text, value=code, variable=self.var_labels[key],
                                command=self._on_control_change).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(label_frame, text="顶点不可见（外推） vertex hidden — 顶点在图内但看不见，由腿推出",
                        variable=self.var_vertex_hidden, command=self._on_control_change).grid(
            row=len(LABEL_SCHEMA), column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))

        self.btn_add = ttk.Button(self.control_frame, text="Add as New Annotation",    command=self.add_object)
        self.btn_add.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1
        self.btn_upd = ttk.Button(self.control_frame, text="Update Selected Annotation", command=self.update_selected_object)
        self.btn_upd.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1
        self.btn_del = ttk.Button(self.control_frame, text="Delete Selected Annotation", command=self.delete_selected_object)
        self.btn_del.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1

        ttk.Label(self.control_frame, text="Annotations in Current Image").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4)); row += 1

        self.object_listbox = tk.Listbox(self.control_frame, width=46, height=10)
        self.object_listbox.grid(row=row, column=0, columnspan=2, padx=6, pady=4, sticky="nsew")
        self.object_listbox.bind("<<ListboxSelect>>", self.on_select_object)
        row += 1

        hint = (
            "Instructions:\n"
            "1. Click left image to set vertex x/y.\n"
            "   拓展画布 on: grey margin left/right 20 %, top 50 %\n"
            "   (none at the bottom) to click a vertex outside the\n"
            "   image; such objects are marked /ext.\n"
            "   Off: anything outside is clamped to the border\n"
            "2. slope = the only shape knob; a/b are auto-derived\n"
            "   from vertex depth (a=y_v, b=y_v/slope)\n"
            "3. span = horizontal draw extent; one band per\n"
            "   visible hyperbola, even inside one GT box.\n"
            "   tail = 振铃/多次反射: fit vertex/slope/span on the\n"
            "   FIRST arc only, then drag tail down to the last\n"
            "   multiple still clearly above background. One\n"
            "   ringing column = one object, never one per arc.\n"
            "   Arms off the image bottom: just drag span past\n"
            "   it; span is capped automatically on save (↓)\n"
            "4. GT box: 'auto' picks the official box with the\n"
            "   largest overlap and asks when ambiguous\n"
            "5. Labels: change only when different from default;\n"
            "   'unparam' saves a record without a band\n"
            "6. Edit: click an item in the list (its params load\n"
            "   into the controls), adjust, then Update.\n"
            "   Add / Update / Delete write the file at once;\n"
            "   Next/Prev skips same-group images\n"
            "7. Timing of every add/update is logged to timelog\n"
            "8. 跳转: group number (57) or image name (030,\n"
            "   128.jpg, 3172); Enter to jump, saves first"
        )
        ttk.Label(self.control_frame, text=hint, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=10)

        for v in [self.var_x, self.var_y, self.var_span, self.var_thickness, self.var_tail]:
            v.trace_add("write", lambda *_: self._on_control_change())
        # Quantize slope to 2 decimals as the slider moves, then refresh.
        self.var_slope.trace_add("write", lambda *_: self._quantize_var(self.var_slope))

    def _on_control_change(self):
        if not self._loading:
            self._obj_timer_start()
        self.refresh_preview()

    def _quantize_var(self, var):
        if getattr(self, "_quantizing", False):
            return
        self._quantizing = True
        try:
            var.set(round(float(var.get()), 2))
        except (ValueError, tk.TclError):
            pass  # ignore mid-typing intermediate states
        finally:
            self._quantizing = False
        self._on_control_change()

    def _make_slider(self, parent, text, variable, row, from_, to_):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        scale = ttk.Scale(frame, from_=from_, to=to_, variable=variable, orient="horizontal")
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sliders[text] = scale
        ttk.Entry(frame, textvariable=variable, width=8).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------ timing ----
    def _obj_timer_start(self):
        if self._t_obj_start is None and self.original_image is not None:
            self._t_obj_start = time.perf_counter()

    def _obj_timer_stop(self):
        d = None if self._t_obj_start is None else round(time.perf_counter() - self._t_obj_start, 3)
        self._t_obj_start = None
        if d is not None:
            self._last_obj_s = d
            self._session_n += 1
            self._session_s += d
        return d

    def _tick_timer(self):
        """Refresh the timer label: current object, current image, last object, session average."""
        now = time.perf_counter()
        parts = []
        if self._t_obj_start is not None:
            parts.append(f"当前目标 {now - self._t_obj_start:5.1f} s")
        else:
            parts.append("当前目标  --  ")
        if self._t_image_enter is not None:
            parts.append(f"本图 {now - self._t_image_enter:5.0f} s")
        if self._last_obj_s is not None:
            parts.append(f"上一个 {self._last_obj_s:.1f} s")
        if self._session_n:
            parts.append(f"本次 {self._session_n} 个, 平均 {self._session_s / self._session_n:.1f} s")
        self.timer_label.config(text="⏱ " + "  |  ".join(parts))
        self._tick_id = self.root.after(500, self._tick_timer)

    def log_event(self, action, **kw):
        if not self.timelog_path:
            return
        rec = dict(ts=datetime.datetime.now().isoformat(timespec="seconds"),
                   annotator=self.annotator, image=self.current_image_name, action=action)
        rec.update(kw)
        try:
            with open(self.timelog_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[timelog] write failed: {e}", file=sys.stderr)

    # ---------------------------------------------------- view mode toggle --
    def toggle_view_mode(self):
        if self.view_mode.get() == "hyperbola":
            self.view_mode.set("rectangle")
            self.toggle_btn.config(text="切换视图: 双曲线")
            # Disable editing controls — rectangles can't be pushed back
            for btn in (self.btn_add, self.btn_upd, self.btn_del):
                btn.state(["disabled"])
            self.left_canvas.unbind("<Button-1>")
        else:
            self.view_mode.set("hyperbola")
            self.toggle_btn.config(text="切换视图: 矩形")
            for btn in (self.btn_add, self.btn_upd, self.btn_del):
                btn.state(["!disabled"])
            self.left_canvas.bind("<Button-1>", self.on_left_click)
        self.refresh_preview()

    def toggle_gt(self):
        self.show_gt.set(not self.show_gt.get())
        self._update_gt_btn()
        self.log_event("toggle_gt", show_gt=self.show_gt.get())
        self.refresh_both_views()

    def _update_gt_btn(self):
        if self.gt_source is None and self.image_dir:
            self.gt_btn.config(text="GT框: 无", state="disabled")
        else:
            self.gt_btn.config(text="GT框: 隐藏" if self.show_gt.get() else "GT框: 显示", state="normal")

    # -------------------------------------------------- group helpers ------
    @staticmethod
    def get_group_name(filename):
        return os.path.splitext(filename)[0].rsplit(".", 1)[0]

    def get_group_members(self, filename):
        g = self.get_group_name(filename)
        return [os.path.basename(p) for p in self.image_paths
                if self.get_group_name(os.path.basename(p)) == g]

    def propagate_to_group(self, source_name):
        objects = self.annotations.get(source_name, [])
        for member in self.get_group_members(source_name):
            if member != source_name:
                self.annotations[member] = [dict(obj) for obj in objects]

    def get_group_start_indices(self):
        starts, seen = [], set()
        for i, path in enumerate(self.image_paths):
            g = self.get_group_name(os.path.basename(path))
            if g not in seen:
                seen.add(g)
                starts.append(i)
        return starts

    # ------------------------------------------------------- file I/O ------
    def _suffix(self):
        return f"_{self.annotator}" if self.annotator else ""

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder):
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(SUPPORTED_EXTS)]
        if not paths:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return
        self.image_dir = folder
        self.image_paths = paths
        self.current_index = 0
        self.out_dir = self.out_dir_arg or folder
        os.makedirs(self.out_dir, exist_ok=True)
        self.annotations_path = os.path.join(self.out_dir, f"annotations{self._suffix()}.json")
        self.timelog_path = os.path.join(self.out_dir, f"timelog{self._suffix()}.jsonl")
        if self.gt_source_arg is None:
            self.gt_source = GTSource.auto_detect(folder)
        elif str(self.gt_source_arg).lower() == "none":
            self.gt_source = None
        else:
            self.gt_source = GTSource(self.gt_source_arg)
        self._update_gt_btn()
        self.load_annotations_file()
        self.current_image_name = None
        self.log_event("session_start", folder=folder, out_dir=self.out_dir, n_images=len(paths),
                       annotations=os.path.basename(self.annotations_path),
                       gt=None if self.gt_source is None else self.gt_source.path,
                       show_gt=self.show_gt.get(), extend=self.extend.get(),
                       pad_side=self.pad_side, pad_top=self.pad_top)
        self.load_image()

    def load_annotations_file(self):
        if os.path.exists(self.annotations_path):
            try:
                with open(self.annotations_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Normalize any old-schema objects on load.
                self.annotations = {
                    name: [normalize_obj(o) for o in objs]
                    for name, objs in raw.items()
                }
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to read {os.path.basename(self.annotations_path)}.\n{e}")
                self.annotations = {}
        else:
            self.annotations = {}

    def save_annotations_file(self):
        try:
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save {os.path.basename(self.annotations_path)}:\n{e}")

    def _image_size(self, img_name):
        cache = self.__dict__.setdefault("_size_cache", {})
        if img_name not in cache:
            try:
                with Image.open(os.path.join(self.image_dir, img_name)) as im:
                    cache[img_name] = im.size
            except Exception:
                cache[img_name] = None
        return cache[img_name]

    def write_rectangles(self):
        """Derived boxes (clipped to each image) -> annotations_rect[_X].json. Returns the path or None."""
        if not self.annotations_path:
            return None
        self._commit_current()
        rect_data = {}
        for img_name, objects in self.annotations.items():
            size = self._image_size(img_name)
            rect_data[img_name] = [b for b in (hyperbola_to_bbox(o, size) for o in objects) if b is not None]
        out_path = os.path.join(self.out_dir, f"annotations_rect{self._suffix()}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rect_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[rect export] failed: {e}", file=sys.stderr)
            return None
        return out_path

    def export_rectangles(self):
        """Button: same as the automatic export on page turn / close, with a confirmation dialog."""
        if not self.annotations_path:
            messagebox.showwarning("Warning", "No folder selected yet.")
            return
        out_path = self.write_rectangles()
        if out_path:
            messagebox.showinfo("Exported", f"Rectangle annotations saved to:\n{out_path}")
        else:
            messagebox.showerror("Error", "Export failed (see console).")

    # ---------------------------------------------------- image loading -----
    def load_image(self):
        if not self.image_paths:
            return
        if self.current_image_name is not None and self._t_image_enter is not None:
            self.log_event("image_leave", duration_s=round(time.perf_counter() - self._t_image_enter, 3),
                           n_objects=len(self.current_objects))
        img_path = self.image_paths[self.current_index]
        self.current_image_name = os.path.basename(img_path)
        try:
            self.original_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image:\n{img_path}\n{e}")
            return

        w, h = self.original_image.size
        self._loading = True
        try:
            self.update_slider_ranges(w, h)

            self.gt_boxes = self.gt_source.boxes_for(self.current_image_name, (w, h)) if self.gt_source else []
            self.gt_combo.configure(values=["auto", "none"] + [str(g["id"]) for g in self.gt_boxes])
            self.var_gt_box.set("auto")

            self.current_objects = self.annotations.get(self.current_image_name, [])
            if not self.current_objects:
                for member in self.get_group_members(self.current_image_name):
                    if member != self.current_image_name and self.annotations.get(member):
                        self.current_objects = [dict(obj) for obj in self.annotations[member]]
                        break

            self.selected_object_index = None
            self.refresh_object_list()

            if self.current_objects:
                self.set_controls_from_object(self.current_objects[0])
            else:
                self.var_x.set(w / 2)
                self.var_y.set(h / 3)
                self.var_slope.set(1.0)
                self.var_span.set(max(80, w / 3))
                self.var_thickness.set(12)
                self.var_tail.set(0)
                self.reset_labels()
        finally:
            self._loading = False
        self._t_image_enter = time.perf_counter()
        self._t_obj_start = None
        self.log_event("image_enter", n_objects=len(self.current_objects), n_gt=len(self.gt_boxes))

        self.refresh_both_views()

        starts = self.get_group_start_indices()
        current_group = self.get_group_name(self.current_image_name)
        gpos = next(i for i, idx in enumerate(starts)
                    if self.get_group_name(os.path.basename(self.image_paths[idx])) == current_group)
        members = self.get_group_members(self.current_image_name)
        who = f"[{self.annotator}] " if self.annotator else ""
        self.info_label.config(
            text=f"{who}Group {gpos + 1}/{len(starts)}   {self.current_image_name}"
                 f"   ({len(members)} in group)   {w}x{h}   GT boxes: {len(self.gt_boxes)}")

    def update_slider_ranges(self, w, h):
        """Vertex sliders cover exactly the visible canvas: the image, plus the margins when the
        extended canvas is on (left/right PAD_SIDE, top PAD_TOP, no bottom margin)."""
        left, top, right, _ = self._pad_box()
        xr, yr = (-left, w + right), (-top, h)
        if "x_vertex" in self.sliders:
            self.sliders["x_vertex"].configure(from_=xr[0], to=xr[1])
        if "y_vertex" in self.sliders:
            self.sliders["y_vertex"].configure(from_=yr[0], to=yr[1])
        if "span" in self.sliders:
            self.sliders["span"].configure(from_=10, to=2 * w)
        # clamp (not reset) the current controls into the new range
        try:
            self.var_x.set(min(max(self.var_x.get(), xr[0]), xr[1]))
            self.var_y.set(min(max(self.var_y.get(), yr[0]), yr[1]))
        except tk.TclError:
            pass
        if self.var_span.get() > 2 * w: self.var_span.set(w / 3)

    def toggle_extend(self):
        self.extend.set(not self.extend.get())
        self._update_extend_btn()
        self.log_event("toggle_extend", extend=self.extend.get())
        if self.original_image is not None:
            was = self._loading
            self._loading = True
            try:
                self.update_slider_ranges(*self.original_image.size)
            finally:
                self._loading = was
        self.refresh_both_views()

    def _update_extend_btn(self):
        self.extend_btn.config(text="拓展画布: 开" if self.extend.get() else "拓展画布: 关")

    # --------------------------------------------------------- display ------
    def fit_image_to_display(self, image):
        w, h = image.size
        scale = min(self.display_w / w, self.display_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS), scale

    def _pad_box(self):
        """Margins (left, top, right, bottom) in image pixels; all zero when the extended canvas is off.

        No bottom margin: arms leaving the bottom are the end of the time window, not a missing
        part of the target (see cap_span_to_bottom).
        """
        if self.original_image is None or not self.extend.get():
            return 0, 0, 0, 0
        w, h = self.original_image.size
        side = int(round(self.pad_side * w))
        return side, int(round(self.pad_top * h)), side, 0

    def _pad_px(self):
        """Offset (left, top) of the image on the canvas."""
        left, top, _, _ = self._pad_box()
        return left, top

    def _canvas_size(self):
        w, h = self.original_image.size
        left, top, right, bottom = self._pad_box()
        return w + left + right, h + top + bottom

    def _padded_canvas(self):
        """Original image on a grey margin (extended canvas on), with a thin outline at the true
        image border. Annotation coordinates stay in image pixels; everything drawn on this canvas
        is shifted by _pad_px().
        """
        img = self.original_image
        left, top, right, bottom = self._pad_box()
        if not (left or top or right or bottom):
            return img.copy()
        w, h = img.size
        canvas = Image.new("RGB", self._canvas_size(), MARGIN_BG)
        canvas.paste(img, (left, top))
        ImageDraw.Draw(canvas).rectangle([left - 1, top - 1, left + w, top + h], outline=MARGIN_EDGE, width=1)
        return canvas

    def refresh_both_views(self):
        self.refresh_left_view()
        self.refresh_preview()

    def refresh_left_view(self):
        if self.original_image is None:
            return
        img = self._padded_canvas()
        if self.show_gt.get() and self.gt_boxes:
            self.draw_gt_boxes(ImageDraw.Draw(img, "RGBA"))
        disp_img, scale = self.fit_image_to_display(img)
        self.scale_x = self.scale_y = scale
        self.left_tk = ImageTk.PhotoImage(disp_img)
        self.left_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.left_canvas.create_image(x0, y0, anchor="nw", image=self.left_tk)

    def refresh_preview(self, *_):
        if self.original_image is None:
            return
        preview_img = self._padded_canvas()
        draw = ImageDraw.Draw(preview_img, "RGBA")

        mode = self.view_mode.get()

        if self.show_gt.get() and self.gt_boxes:
            self.draw_gt_boxes(draw)

        # Draw saved annotations
        for idx, obj in enumerate(self.current_objects):
            selected = (idx == self.selected_object_index)
            if mode == "hyperbola":
                self.draw_hyperbola_band(draw, obj, selected=selected)
            else:
                self.draw_rect_from_hyperbola(draw, obj, selected=selected)

        # Draw live preview from controls (only in hyperbola mode)
        live = self.get_current_object_from_controls()
        if mode == "hyperbola":
            self.draw_hyperbola_band(draw, live, selected=False)
        notes, base = [], live
        if not self.extend.get():
            base, changed = clamp_to_image(live, self.original_image.size)
            if changed:
                notes.append("拓展画布关闭：保存时超出图像的部分收回到图像边界")
        capped = cap_span_to_bottom(base, self.original_image.size)
        if capped["reaches_bottom"]:
            notes.append(f"↓ 两翼触底：保存时 span 自动截为 {capped['span']:.1f}（图内的带不变）")
        self.span_note.config(text="\n".join(notes))

        disp_img, _ = self.fit_image_to_display(preview_img)
        self.right_tk = ImageTk.PhotoImage(disp_img)
        self.right_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.right_canvas.create_image(x0, y0, anchor="nw", image=self.right_tk)
        self.refresh_left_view()

    # ------------------------------------------------------ drawing ---------
    def draw_gt_boxes(self, draw):
        px, py = self._pad_px()
        for g in self.gt_boxes:
            draw.rectangle([g["x1"] + px, g["y1"] + py, g["x2"] + px, g["y2"] + py], outline=GT_OUTLINE, width=2)
            draw.text((g["x1"] + px + 3, g["y1"] + py + 2), f"GT{g['id']}", fill=GT_TEXT)

    def _alpha(self, base):
        """Scale an alpha value (0-255) by the display-opacity slider."""
        return int(round(base * self.var_opacity.get() / 100.0))

    def draw_hyperbola_band(self, draw, obj, color=None, outline=None, selected=False):
        """Display only (masks are made by band_mask). Style: fill / outline / hidden, alpha from the slider."""
        style = self.band_style.get()
        if style == "hidden":
            return
        px, py = self._pad_px()
        xv, yv = obj["x_vertex"] + px, obj["y_vertex"] + py
        r = 4 if selected else 3
        a_line = self._alpha(255)
        yellow = (255, 255, 0, a_line)
        polygon, centerline = hyperbola_band_polygon(obj)
        if polygon is None:
            # unparametrizable target: hollow marker only, no band
            draw.ellipse((xv - r - 3, yv - r - 3, xv + r + 3, yv + r + 3), outline=yellow, width=2)
            draw.text((xv + r + 5, yv - r - 5), "?", fill=yellow)
            return
        polygon = [(x + px, y + py) for x, y in polygon]
        centerline = [(x + px, y + py) for x, y in centerline]
        n = len(polygon) // 2
        upper, lower = polygon[:n], polygon[n:]
        edge = (0, 255, 255, a_line) if selected else (0, 255, 0, a_line)
        if style == "fill":
            draw.polygon(polygon, fill=(0, 255, 0, self._alpha(110)))
            draw.line(centerline, fill=edge, width=2 if selected else 1)
        else:  # "outline": only the two band edges, the echo between them stays untouched
            draw.line(upper, fill=edge, width=2 if selected else 1)
            draw.line(lower, fill=edge, width=2 if selected else 1)
        if normalize_obj(obj)["tail"] > 0:
            # first-arc ribbon boundary, so the annotator sees where the tail starts
            inner, _ = hyperbola_band_polygon(obj, tail_scale=0.0)
            m = len(inner) // 2
            draw.line([(x + px, y + py) for x, y in inner[m:]], fill=edge, width=1)
        if obj.get("vertex_hidden"):
            draw.ellipse((xv - r, yv - r, xv + r, yv + r), outline=yellow, width=1)   # inferred apex: hollow
        else:
            draw.ellipse((xv - r, yv - r, xv + r, yv + r), fill=yellow)

    def cycle_band_style(self):
        order = ["fill", "outline", "hidden"]
        self.band_style.set(order[(order.index(self.band_style.get()) + 1) % len(order)])
        self.style_btn.config(text=BAND_STYLE_TEXT[self.band_style.get()])
        self.refresh_preview()

    def draw_rect_from_hyperbola(self, draw, obj, selected=False):
        bbox = hyperbola_to_bbox(obj, self.original_image.size if self.original_image else None)
        px, py = self._pad_px()
        xv, yv = obj["x_vertex"] + px, obj["y_vertex"] + py
        r = 6 if selected else 4
        if bbox is None:
            draw.ellipse((xv - r - 3, yv - r - 3, xv + r + 3, yv + r + 3), outline=(255, 255, 0, 255), width=2)
            return
        x1, y1 = bbox["x1"] + px, bbox["y1"] + py
        x2, y2 = x1 + bbox["width"], y1 + bbox["height"]
        draw.rectangle([x1, y1, x2, y2], fill=RECT_FILL)
        oc = RECT_SEL if selected else RECT_OUTLINE
        draw.rectangle([x1, y1, x2, y2], outline=oc, width=2)
        draw.ellipse((xv - r, yv - r, xv + r, yv + r), fill=(255, 255, 0, 255))

    # ---------------------------------------------------------- events ------
    def on_left_click(self, event):
        if self.original_image is None:
            return
        # clicks anywhere on the (possibly extended) canvas are accepted
        cw, ch = self._canvas_size()
        px, py = self._pad_px()
        scale = min(self.display_w / cw, self.display_h / ch)
        disp_w, disp_h = max(1, int(cw * scale)), max(1, int(ch * scale))
        x0 = (self.display_w - disp_w) // 2
        y0 = (self.display_h - disp_h) // 2
        if not (x0 <= event.x <= x0 + disp_w and y0 <= event.y <= y0 + disp_h):
            return
        self._obj_timer_start()
        self.var_x.set(round((event.x - x0) / scale - px, 2))
        self.var_y.set(round((event.y - y0) / scale - py, 2))

    def get_current_object_from_controls(self):
        labels = {k: v.get() for k, v in self.var_labels.items()}
        unparam = labels["param_status"] == "unparam"
        obj = {
            "label":     self.var_name.get().strip() or "hyperbola",
            "x_vertex":  round(float(self.var_x.get()), 2),
            "y_vertex":  round(float(self.var_y.get()), 2),
            "slope":     None if unparam else round(float(self.var_slope.get()), 2),
            "span":      None if unparam else round(float(self.var_span.get()), 2),
            "thickness": None if unparam else round(float(self.var_thickness.get()), 2),
            "tail":      0.0 if unparam else round(max(0.0, float(self.var_tail.get())), 2),
            "gt_box_id": None,
        }
        obj.update(labels)
        obj["vertex_hidden"] = bool(self.var_vertex_hidden.get())
        return obj

    def _finalize(self, obj):
        """Apply the canvas rules before saving and show the stored values in the controls.

        Extended canvas OFF: nothing outside the image is kept (vertex clamped to the border,
        span capped at the left/right edges). Extended canvas ON: geometry outside the image is
        kept and the object is marked extended_canvas = True. In both modes arms leaving the
        bottom are capped (no bottom margin; end of the time window).
        """
        size = self.original_image.size
        if not self.extend.get():
            obj, _ = clamp_to_image(obj, size)
        obj = cap_span_to_bottom(obj, size)
        obj["extended_canvas"] = bool(self.extend.get())
        was = self._loading
        self._loading = True              # not a user edit: do not start the next object's timer
        try:
            self.var_x.set(obj["x_vertex"])
            self.var_y.set(obj["y_vertex"])
            if has_band(obj):
                self.var_span.set(obj["span"])
        finally:
            self._loading = was
        return obj

    def _on_gt_combo_selected(self, *_):
        self._gt_box_user_set = True
        self._obj_timer_start()

    def _resolve_gt_box(self, obj, new_object=False):
        """Fill obj['gt_box_id'] from the combobox, asking the annotator when 'auto' is ambiguous.

        For a NEW object, a box id that was only copied into the combobox by selecting an
        existing object (not picked by the annotator) is ignored and 'auto' is used instead;
        otherwise "select an object -> adjust -> Add" silently inherits the wrong box.
        """
        choice = self.var_gt_box.get()
        if new_object and choice not in ("auto", "none") and not getattr(self, "_gt_box_user_set", False):
            choice = "auto"
        if choice == "none" or not self.gt_boxes:
            obj["gt_box_id"] = None
            return True
        if choice != "auto":
            obj["gt_box_id"] = int(choice)
            return True
        best, ambiguous = assign_gt_box(obj, self.gt_boxes, self.original_image.size)
        if best is None:
            ok = messagebox.askyesno("GT box", "This band does not overlap any official GT box.\n"
                                               "Save it with gt_box_id = none (official miss)?")
            if not ok:
                return False
            obj["gt_box_id"] = None
            return True
        if ambiguous:
            ans = simpledialog.askinteger(
                "GT box", f"Overlaps several GT boxes. Enter the GT id (suggested {best}):",
                initialvalue=best, minvalue=0, maxvalue=len(self.gt_boxes) - 1)
            if ans is None:
                return False
            best = ans
        obj["gt_box_id"] = best
        return True

    # ------------------------------------------------- annotation CRUD ------
    def add_object(self):
        if self.original_image is None:
            return
        obj = self._finalize(self.get_current_object_from_controls())
        if not self._resolve_gt_box(obj, new_object=True):
            return
        self.current_objects.append(obj)
        self.selected_object_index = len(self.current_objects) - 1
        self.log_event("add", duration_s=self._obj_timer_stop(), obj_index=self.selected_object_index, obj=obj)
        self.reset_labels()
        self._autosave()
        self.refresh_object_list()
        self.refresh_both_views()

    def update_selected_object(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        obj = self._finalize(self.get_current_object_from_controls())
        if not self._resolve_gt_box(obj):
            return
        self.current_objects[self.selected_object_index] = obj
        self.log_event("update", duration_s=self._obj_timer_stop(), obj_index=self.selected_object_index, obj=obj)
        self._autosave()
        self.refresh_object_list()
        self.refresh_both_views()

    def delete_selected_object(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        self.log_event("delete", obj_index=self.selected_object_index,
                       obj=self.current_objects[self.selected_object_index])
        del self.current_objects[self.selected_object_index]
        self.selected_object_index = None
        self._autosave()
        self.refresh_object_list()
        self.refresh_both_views()

    def refresh_object_list(self):
        self.object_listbox.delete(0, tk.END)
        for i, obj in enumerate(self.current_objects):
            obj = normalize_obj(obj)
            gt = "-" if obj.get("gt_box_id") is None else str(obj["gt_box_id"])
            tags = "/".join(obj[k][:3] for k in LABEL_SCHEMA) + ("/vh" if obj.get("vertex_hidden") else "") \
                + ("/ext" if obj.get("extended_canvas") else "")
            if has_band(obj):
                geo = (f"x={obj['x_vertex']:.0f} y={obj['y_vertex']:.0f} "
                       f"s={obj['slope']:.2f} sp={obj['span']:.0f}{'↓' if obj.get('reaches_bottom') else ''} "
                       f"t={obj['thickness']:.0f}"
                       + (f" tail={obj['tail']:.0f}" if obj["tail"] > 0 else ""))
            else:
                geo = f"x={obj['x_vertex']:.0f} y={obj['y_vertex']:.0f} [no band]"
            self.object_listbox.insert(tk.END, f"[{i}] {obj.get('label','hyperbola')} GT{gt} | {geo} | {tags}")
        if (self.selected_object_index is not None and
                0 <= self.selected_object_index < len(self.current_objects)):
            self.object_listbox.selection_set(self.selected_object_index)

    def on_select_object(self, _):
        sel = self.object_listbox.curselection()
        self.selected_object_index = sel[0] if sel else None
        if self.selected_object_index is not None:
            # load its parameters and labels into the controls right away, so that
            # "select -> adjust -> Update" is the whole edit cycle
            self.set_controls_from_object(self.current_objects[self.selected_object_index])
        self.refresh_both_views()

    def load_selected_to_controls(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        self.set_controls_from_object(self.current_objects[self.selected_object_index])

    def reset_labels(self):
        for k, d in LABEL_DEFAULTS.items():
            self.var_labels[k].set(d)
        self.var_vertex_hidden.set(False)
        self.var_gt_box.set("auto")
        self._gt_box_user_set = False

    def set_controls_from_object(self, obj):
        obj = normalize_obj(obj)
        was_loading = self._loading
        self._loading = True
        try:
            self.var_name.set(obj.get("label", "hyperbola"))
            self.var_x.set(obj["x_vertex"])
            self.var_y.set(obj["y_vertex"])
            if has_band(obj):
                self.var_slope.set(obj["slope"])
                self.var_span.set(obj["span"])
                self.var_thickness.set(obj["thickness"])
            self.var_tail.set(obj["tail"])
            for k in LABEL_SCHEMA:
                self.var_labels[k].set(obj.get(k, LABEL_DEFAULTS[k]))
            self.var_vertex_hidden.set(bool(obj.get("vertex_hidden", False)))
            self.var_gt_box.set("none" if obj.get("gt_box_id") is None else str(obj["gt_box_id"]))
            self._gt_box_user_set = False     # loaded from the object, not chosen by the annotator
        finally:
            self._loading = was_loading
        self.refresh_preview()

    # --------------------------------------------------- save / navigate ----
    def _commit_current(self):
        if self.current_image_name is None:
            return
        self.annotations[self.current_image_name] = self.current_objects
        self.propagate_to_group(self.current_image_name)

    def _autosave(self):
        """Write the annotation file after every add / update / delete (no manual save needed)."""
        if self.current_image_name is None:
            return
        self._commit_current()
        self.save_annotations_file()

    def on_close(self):
        """Window close: save, log the time spent on the last image, then exit."""
        if self.current_image_name is not None:
            self._autosave()
            self.write_rectangles()
            if self._t_image_enter is not None:
                self.log_event("image_leave", duration_s=round(time.perf_counter() - self._t_image_enter, 3),
                               n_objects=len(self.current_objects))
            self.log_event("session_end")
        if getattr(self, "_tick_id", None):
            try:
                self.root.after_cancel(self._tick_id)
            except tk.TclError:
                pass
        self.root.destroy()

    def save_current_annotations(self):
        if self.current_image_name is None:
            return
        self._commit_current()
        self.save_annotations_file()
        members = self.get_group_members(self.current_image_name)
        msg = f"Saved: {self.current_image_name}"
        if len(members) > 1:
            msg += f"\nSynced to {len(members) - 1} other image(s) in the same group."
        messagebox.showinfo("Saved", msg)

    def save_all(self):
        self._commit_current()
        self.save_annotations_file()
        messagebox.showinfo("Saved", f"All annotations saved to:\n{self.annotations_path}")

    def _step_group(self, delta):
        if not self.image_paths:
            return
        self._commit_current()
        self.save_annotations_file()
        self.write_rectangles()
        starts = self.get_group_start_indices()
        g = self.get_group_name(self.current_image_name)
        pos = next(i for i, idx in enumerate(starts)
                   if self.get_group_name(os.path.basename(self.image_paths[idx])) == g)
        self.current_index = starts[(pos + delta) % len(starts)]
        self.load_image()

    def _resolve_goto(self, text):
        """Group number or image name. Returns the image index of the group's first image, or None.

        A plain number without a leading zero that is within 1..N is a group number (as shown in the
        info bar). Anything else -- a leading zero ("030"), an extension ("128.jpg"), a number larger
        than N ("3172"), letters -- is looked up as an image name / stem / group name / prefix.
        """
        text = text.strip()
        if not text or not self.image_paths:
            return None
        starts = self.get_group_start_indices()
        names = [os.path.basename(p) for p in self.image_paths]
        if text.isdigit() and not text.startswith("0") and 1 <= int(text) <= len(starts):
            return starts[int(text) - 1]
        low = text.lower()
        for cand in (lambda n: n.lower() == low,                                    # exact file name
                     lambda n: os.path.splitext(n)[0].lower() == low,               # stem: 030
                     lambda n: self.get_group_name(n).lower() == low,               # group: 3172_jpg
                     lambda n: os.path.splitext(n)[0].lower().startswith(low)):     # prefix: 3172
            hits = [i for i, n in enumerate(names) if cand(n)]
            if hits:
                g = self.get_group_name(names[hits[0]])
                return next(i for i in starts if self.get_group_name(names[i]) == g)
        return None

    def goto_group(self):
        """Jump to a group typed in the entry; the current image is saved first, like Next/Prev."""
        target = self._resolve_goto(self.var_goto.get())
        if target is None:
            n = len(self.get_group_start_indices()) if self.image_paths else 0
            messagebox.showinfo("跳转", f"找不到“{self.var_goto.get()}”。\n请输入组号 1–{n}，或图名（如 030、030.jpg）。")
            return
        self._commit_current()
        self.save_annotations_file()
        self.write_rectangles()
        self.current_index = target
        self.load_image()
        self.var_goto.set("")

    def prev_image(self):
        self._step_group(-1)

    def next_image(self):
        self._step_group(+1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HBAT hyperbola-band annotation tool")
    ap.add_argument("--folder", help="image folder to open at start")
    ap.add_argument("--annotator", help="annotator name; writes annotations_<name>.json and timelog_<name>.jsonl")
    ap.add_argument("--gt", help="official GT boxes: .txt directory (YOLO / DOTA-OBB) or JSON file; 'none' disables")
    ap.add_argument("--hide-gt", action="store_true", help="start with GT boxes hidden (annotation study)")
    ap.add_argument("--out", help="output directory for annotations/timelog (default: the image folder)")
    ap.add_argument("--extend", action="store_true",
                    help="start with the extended canvas on (can be toggled in the GUI)")
    ap.add_argument("--pad-side", type=float, default=0.2,
                    help="extended canvas: left/right margin, fraction of width (default 0.2)")
    ap.add_argument("--pad-top", type=float, default=0.5,
                    help="extended canvas: top margin, fraction of height (default 0.5)")
    args = ap.parse_args(argv)

    root = tk.Tk()
    app = HyperbolaAnnotator(root, annotator=args.annotator, gt_source=args.gt,
                             show_gt=not args.hide_gt, folder=args.folder, out_dir=args.out,
                             extend=args.extend, pad_side=args.pad_side, pad_top=args.pad_top)
    root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()
