import os
import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

HYPER_FILL    = (0, 255, 0, 70)
HYPER_OUTLINE = (0, 255, 0, 160)


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
#       x_vertex, y_vertex : apex pixel coordinates
#       slope              : asymptote slope a/b (velocity)
#       span               : horizontal extent used to draw / truncate the arms
#       thickness          : vertical band thickness
#
#   The band is a vertical ribbon: { (x, y) : |y - y_c(x)| <= thickness / 2 }.
# ---------------------------------------------------------------------------

def hyperbola_ab(y_vertex, slope):
    """Physical semi-axes from apex depth and slope (surface = image top)."""
    a = max(1.0, abs(y_vertex))
    s = max(1e-3, slope)
    return a, a / s


def normalize_obj(obj):
    """Normalize legacy fields (width->span, a/b/height -> slope)."""
    obj = dict(obj)
    # Legacy field rename: width -> span (horizontal draw extent).
    if "span" not in obj and "width" in obj:
        obj["span"] = obj["width"]
    obj.setdefault("span", 100.0)
    if "slope" not in obj:
        if "a" in obj and obj.get("b"):
            # Previous two-semi-axis schema.
            obj["slope"] = round(obj["a"] / obj["b"], 2)
        elif "height" in obj:
            # Old parabola schema y_v + height * (dx)^2.
            half_w = max(1.0, obj["span"] / 2.0)
            obj["slope"] = round(2.0 * obj["height"] / half_w, 2)
        else:
            obj["slope"] = 1.0
    return obj


class HyperbolaAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("GPR Hyperbola Annotation Tool")
        self.root.geometry("1900x900")

        self.image_dir = ""
        self.image_paths = []
        self.current_index = 0

        self.annotations_path = ""
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

        self.var_x         = tk.DoubleVar(value=200)
        self.var_y         = tk.DoubleVar(value=200)
        self.var_slope     = tk.DoubleVar(value=1.0)  # asymptote slope a/b
        self.var_span      = tk.DoubleVar(value=200)  # horizontal draw extent
        self.var_thickness = tk.DoubleVar(value=12)
        self.var_name      = tk.StringVar(value="hyperbola")
        self.sliders = {}

        self._build_ui()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        top_bar = ttk.Frame(self.root)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(top_bar, text="Select Image Folder",      command=self.choose_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Previous",                 command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Next",                     command=self.next_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Save Current Annotations", command=self.save_current_annotations).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Save All",                 command=self.save_all).pack(side=tk.LEFT, padx=4)

        self.info_label = ttk.Label(top_bar, text="Please select an image folder first")
        self.info_label.pack(side=tk.LEFT, padx=12)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_panel = ttk.LabelFrame(image_frame, text="Original Image")
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
        self._make_slider(self.control_frame, "slope",     self.var_slope,     row, 0.1, 5);  row += 1
        self._make_slider(self.control_frame, "span",      self.var_span,      row, 10, 600); row += 1
        self._make_slider(self.control_frame, "thickness", self.var_thickness, row, 1, 150);  row += 1

        self.btn_add = ttk.Button(self.control_frame, text="Add as New Annotation",    command=self.add_object)
        self.btn_add.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1
        self.btn_upd = ttk.Button(self.control_frame, text="Update Selected Annotation", command=self.update_selected_object)
        self.btn_upd.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1
        self.btn_del = ttk.Button(self.control_frame, text="Delete Selected Annotation", command=self.delete_selected_object)
        self.btn_del.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1

        ttk.Label(self.control_frame, text="Annotations in Current Image").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4)); row += 1

        self.object_listbox = tk.Listbox(self.control_frame, width=32, height=10)
        self.object_listbox.grid(row=row, column=0, columnspan=2, padx=6, pady=4, sticky="nsew")
        self.object_listbox.bind("<<ListboxSelect>>", self.on_select_object)
        row += 1

        ttk.Button(self.control_frame, text="Load Parameters from Selected",
                   command=self.load_selected_to_controls).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6); row += 1

        hint = (
            "Instructions:\n"
            "1. Click left image to set vertex x/y\n"
            "2. slope = the only shape knob; a/b are\n"
            "   auto-derived from vertex depth (a=y_v,\n"
            "   b=y_v/slope), matching the GPR physics\n"
            "3. span = horizontal draw extent\n"
            "4. Add / Update / Delete annotations\n"
            "5. Next/Prev moves through images one by one"
        )
        ttk.Label(self.control_frame, text=hint, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=10)

        for v in [self.var_x, self.var_y, self.var_span, self.var_thickness]:
            v.trace_add("write", lambda *_: self.refresh_preview())
        # Quantize slope to 2 decimals as the slider moves, then refresh.
        self.var_slope.trace_add("write", lambda *_: self._quantize_var(self.var_slope))

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
        self.refresh_preview()

    def _make_slider(self, parent, text, variable, row, from_, to_):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        scale = ttk.Scale(frame, from_=from_, to=to_, variable=variable, orient="horizontal")
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sliders[text] = scale
        ttk.Entry(frame, textvariable=variable, width=8).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------- file I/O ------
    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(SUPPORTED_EXTS)]
        if not paths:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return
        self.image_dir = folder
        self.image_paths = paths
        self.current_index = 0
        self.annotations_path = os.path.join(folder, "annotations.json")
        self.load_annotations_file()
        self.load_image()

    def load_annotations_file(self):
        if os.path.exists(self.annotations_path):
            try:
                with open(self.annotations_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.annotations = {
                    name: [normalize_obj(o) for o in objs]
                    for name, objs in raw.items()
                }
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to read annotations.json.\n{e}")
                self.annotations = {}
        else:
            self.annotations = {}

    def save_annotations_file(self):
        try:
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations.json:\n{e}")

    # ---------------------------------------------------- image loading -----
    def load_image(self):
        if not self.image_paths:
            return
        img_path = self.image_paths[self.current_index]
        self.current_image_name = os.path.basename(img_path)
        try:
            self.original_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image:\n{img_path}\n{e}")
            return

        w, h = self.original_image.size
        self.update_slider_ranges(w, h)

        self.current_objects = self.annotations.get(self.current_image_name, [])

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

        self.refresh_both_views()

        self.info_label.config(text=self.current_image_name)

    def update_slider_ranges(self, w, h):
        if "x_vertex" in self.sliders:
            self.sliders["x_vertex"].configure(from_=0, to=max(1, w - 1))
        if "y_vertex" in self.sliders:
            self.sliders["y_vertex"].configure(from_=0, to=max(1, h - 1))
        if self.var_x.get() > w:    self.var_x.set(w / 2)
        if self.var_y.get() > h:    self.var_y.set(h / 2)
        if self.var_span.get() > w: self.var_span.set(w / 3)

    # --------------------------------------------------------- display ------
    def fit_image_to_display(self, image):
        w, h = image.size
        scale = min(self.display_w / w, self.display_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS), scale

    def refresh_both_views(self):
        self.refresh_left_view()
        self.refresh_preview()

    def refresh_left_view(self):
        if self.original_image is None:
            return
        disp_img, scale = self.fit_image_to_display(self.original_image.copy())
        self.scale_x = self.scale_y = scale
        self.left_tk = ImageTk.PhotoImage(disp_img)
        self.left_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.left_canvas.create_image(x0, y0, anchor="nw", image=self.left_tk)

    def refresh_preview(self, *_):
        if self.original_image is None:
            return
        preview_img = self.original_image.copy()
        draw = ImageDraw.Draw(preview_img, "RGBA")

        # Draw saved annotations
        for idx, obj in enumerate(self.current_objects):
            selected = (idx == self.selected_object_index)
            self.draw_hyperbola_band(draw, obj, selected=selected)

        # Draw live preview from controls
        self.draw_hyperbola_band(draw, self.get_current_object_from_controls(),
                                 color=HYPER_FILL, outline=HYPER_OUTLINE, selected=False)

        disp_img, _ = self.fit_image_to_display(preview_img)
        self.right_tk = ImageTk.PhotoImage(disp_img)
        self.right_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.right_canvas.create_image(x0, y0, anchor="nw", image=self.right_tk)
        self.refresh_left_view()

    # ------------------------------------------------------ drawing ---------
    def hyperbola_band_polygon(self, obj, n_points=160):
        obj       = normalize_obj(obj)
        x_vertex  = obj["x_vertex"]
        y_vertex  = obj["y_vertex"]
        a, b      = hyperbola_ab(y_vertex, obj["slope"])
        span      = max(2.0, obj["span"])
        thickness = max(1.0, obj["thickness"])
        half_w    = span / 2.0
        x_left    = x_vertex - half_w
        x_right   = x_vertex + half_w

        upper, lower, centerline = [], [], []
        for i in range(n_points + 1):
            t  = i / n_points
            x  = x_left + (x_right - x_left) * t
            yc = y_vertex + a * (math.sqrt(1.0 + ((x - x_vertex) / b) ** 2) - 1.0)
            upper.append((x, yc - thickness / 2.0))
            lower.append((x, yc + thickness / 2.0))
            centerline.append((x, yc))

        return upper + list(reversed(lower)), centerline

    def draw_hyperbola_band(self, draw, obj,
                            color=HYPER_FILL, outline=HYPER_OUTLINE, selected=False):
        polygon, centerline = self.hyperbola_band_polygon(obj)
        draw.polygon(polygon, fill=color)
        draw.line(centerline, fill=outline, width=2)
        xv, yv = obj["x_vertex"], obj["y_vertex"]
        r = 6 if selected else 4
        draw.ellipse((xv - r, yv - r, xv + r, yv + r), fill=(255, 255, 0, 255))
        if selected:
            draw.line(centerline, fill=(0, 255, 255, 255), width=3)

    # ---------------------------------------------------------- events ------
    def on_left_click(self, event):
        if self.original_image is None:
            return
        disp_img, scale = self.fit_image_to_display(self.original_image)
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        if not (x0 <= event.x <= x0 + disp_img.width and
                y0 <= event.y <= y0 + disp_img.height):
            return
        self.var_x.set((event.x - x0) / scale)
        self.var_y.set((event.y - y0) / scale)

    def get_current_object_from_controls(self):
        return {
            "label":     self.var_name.get().strip() or "hyperbola",
            "x_vertex":  round(float(self.var_x.get()), 2),
            "y_vertex":  round(float(self.var_y.get()), 2),
            "slope":     round(float(self.var_slope.get()), 2),
            "span":      round(float(self.var_span.get()), 2),
            "thickness": round(float(self.var_thickness.get()), 2),
        }

    # ------------------------------------------------- annotation CRUD ------
    def add_object(self):
        if self.original_image is None:
            return
        self.current_objects.append(self.get_current_object_from_controls())
        self.selected_object_index = len(self.current_objects) - 1
        self.refresh_object_list()
        self.refresh_both_views()

    def update_selected_object(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        self.current_objects[self.selected_object_index] = self.get_current_object_from_controls()
        self.refresh_object_list()
        self.refresh_both_views()

    def delete_selected_object(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        del self.current_objects[self.selected_object_index]
        self.selected_object_index = None
        self.refresh_object_list()
        self.refresh_both_views()

    def refresh_object_list(self):
        self.object_listbox.delete(0, tk.END)
        for i, obj in enumerate(self.current_objects):
            obj = normalize_obj(obj)
            self.object_listbox.insert(
                tk.END,
                f"[{i}] {obj.get('label','hyperbola')} | "
                f"x={obj['x_vertex']:.1f}, y={obj['y_vertex']:.1f}, "
                f"slope={obj['slope']:.2f}, "
                f"span={obj['span']:.1f}, t={obj['thickness']:.1f}"
            )
        if (self.selected_object_index is not None and
                0 <= self.selected_object_index < len(self.current_objects)):
            self.object_listbox.selection_set(self.selected_object_index)

    def on_select_object(self, _):
        sel = self.object_listbox.curselection()
        self.selected_object_index = sel[0] if sel else None
        self.refresh_both_views()

    def load_selected_to_controls(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        self.set_controls_from_object(self.current_objects[self.selected_object_index])

    def set_controls_from_object(self, obj):
        obj = normalize_obj(obj)
        self.var_name.set(obj.get("label", "hyperbola"))
        self.var_x.set(obj["x_vertex"])
        self.var_y.set(obj["y_vertex"])
        self.var_slope.set(obj["slope"])
        self.var_span.set(obj["span"])
        self.var_thickness.set(obj["thickness"])

    # --------------------------------------------------- save / navigate ----
    def _commit_current(self):
        if self.current_image_name is None:
            return
        self.annotations[self.current_image_name] = self.current_objects

    def save_current_annotations(self):
        if self.current_image_name is None:
            return
        self._commit_current()
        self.save_annotations_file()
        messagebox.showinfo("Saved", f"Saved: {self.current_image_name}")

    def save_all(self):
        self._commit_current()
        self.save_annotations_file()
        messagebox.showinfo("Saved", f"All annotations saved to:\n{self.annotations_path}")

    def prev_image(self):
        if not self.image_paths:
            return
        self._commit_current()
        self.save_annotations_file()
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_image()

    def next_image(self):
        if not self.image_paths:
            return
        self._commit_current()
        self.save_annotations_file()
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = HyperbolaAnnotator(root)
    root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    root.mainloop()
