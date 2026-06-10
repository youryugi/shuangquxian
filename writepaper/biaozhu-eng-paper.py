import os
import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # or 2
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class HyperbolaAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Hyperbola Annotation Tool")
        self.root.geometry("1900x900")

        self.image_dir = ""
        self.image_paths = []
        self.current_index = 0

        self.annotations_path = ""
        self.annotations = {}  # {filename: [obj1, obj2, ...]}

        self.original_image = None
        self.current_image_name = None

        self.display_w = 700
        self.display_h = 700
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.left_tk = None
        self.right_tk = None

        self.current_objects = []   # Existing annotations for current image
        self.selected_object_index = None

        self.var_x = tk.DoubleVar(value=200)
        self.var_y = tk.DoubleVar(value=200)
        self.var_width = tk.DoubleVar(value=150)
        self.var_height = tk.DoubleVar(value=80)
        self.var_thickness = tk.DoubleVar(value=12)
        self.var_name = tk.StringVar(value="hyperbola")
        self.sliders = {}

        self._build_ui()

    def _build_ui(self):
        top_bar = ttk.Frame(self.root)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(top_bar, text="Select Image Folder", command=self.choose_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Save Current Annotations", command=self.save_current_annotations).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_bar, text="Save All", command=self.save_all).pack(side=tk.LEFT, padx=4)

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

        control_frame = ttk.LabelFrame(main_frame, text="Parameter Controls")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        row = 0
        ttk.Label(control_frame, text="Class Name").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(control_frame, textvariable=self.var_name, width=18).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        row += 1

        self._make_slider(control_frame, "x_vertex", self.var_x, row, 0, 1000)
        row += 1
        self._make_slider(control_frame, "y_vertex", self.var_y, row, 0, 1000)
        row += 1
        self._make_slider(control_frame, "width", self.var_width, row, 10, 500)
        row += 1
        self._make_slider(control_frame, "height", self.var_height, row, 1, 300)
        row += 1
        self._make_slider(control_frame, "thickness", self.var_thickness, row, 1, 100)
        row += 1

        ttk.Button(control_frame, text="Add as New Annotation", command=self.add_object).grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        row += 1
        ttk.Button(control_frame, text="Update Selected Annotation", command=self.update_selected_object).grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        row += 1
        ttk.Button(control_frame, text="Delete Selected Annotation", command=self.delete_selected_object).grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        row += 1

        ttk.Label(control_frame, text="Annotations in Current Image").grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))
        row += 1

        self.object_listbox = tk.Listbox(control_frame, width=28, height=10)
        self.object_listbox.grid(row=row, column=0, columnspan=2, padx=6, pady=4, sticky="nsew")
        self.object_listbox.bind("<<ListboxSelect>>", self.on_select_object)
        row += 1

        ttk.Button(control_frame, text="Load Parameters from Selected", command=self.load_selected_to_controls).grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        row += 1

        hint = (
            "Instructions:\n"
            "1. Click on the left image to set vertex x/y\n"
            "2. Adjust width / height / thickness\n"
            "3. The right image shows real-time annotation preview\n"
            "4. Click 'Add as New Annotation' to save to current image\n"
            "5. Save before switching images"
        )
        ttk.Label(control_frame, text=hint, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=10)

        for v in [self.var_x, self.var_y, self.var_width, self.var_height, self.var_thickness]:
            v.trace_add("write", lambda *args: self.refresh_preview())

    def _make_slider(self, parent, text, variable, row, from_, to_):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=4)

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", padx=6, pady=4)

        scale = ttk.Scale(frame, from_=from_, to=to_, variable=variable, orient="horizontal")
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sliders[text] = scale

        entry = ttk.Entry(frame, textvariable=variable, width=8)
        entry.pack(side=tk.LEFT, padx=4)

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return

        image_paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(SUPPORTED_EXTS)
        ]

        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return

        self.image_dir = folder
        self.image_paths = image_paths
        self.current_index = 0

        self.annotations_path = os.path.join(folder, "annotations.json")
        self.load_annotations_file()
        self.load_image()

    def load_annotations_file(self):
        if os.path.exists(self.annotations_path):
            try:
                with open(self.annotations_path, "r", encoding="utf-8") as f:
                    self.annotations = json.load(f)
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to read annotations.json. Empty annotations will be used.\n{e}")
                self.annotations = {}
        else:
            self.annotations = {}

    def save_annotations_file(self):
        try:
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations.json:\n{e}")

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
            first = self.current_objects[0]
            self.set_controls_from_object(first)
        else:
            self.var_x.set(w / 2)
            self.var_y.set(h / 3)
            self.var_width.set(max(60, w / 4))
            self.var_height.set(max(30, h / 8))
            self.var_thickness.set(12)

        self.refresh_both_views()

        self.info_label.config(
            text=f"{self.current_index + 1}/{len(self.image_paths)}   "
                 f"{self.current_image_name}   "
                 f"Original size: {w}x{h}"
        )

    def update_slider_ranges(self, w, h):
        # Adapt x/y slider ranges to current image size.
        if "x_vertex" in self.sliders:
            self.sliders["x_vertex"].configure(from_=0, to=max(1, w - 1))
        if "y_vertex" in self.sliders:
            self.sliders["y_vertex"].configure(from_=0, to=max(1, h - 1))

        # Keep current values in range after updating slider bounds.
        if self.var_x.get() > w:
            self.var_x.set(w / 2)
        if self.var_y.get() > h:
            self.var_y.set(h / 2)
        if self.var_width.get() > w:
            self.var_width.set(w / 4)
        if self.var_height.get() > h:
            self.var_height.set(h / 8)

    def fit_image_to_display(self, image):
        w, h = image.size
        scale = min(self.display_w / w, self.display_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized, scale

    def refresh_both_views(self):
        self.refresh_left_view()
        self.refresh_preview()

    def refresh_left_view(self):
        if self.original_image is None:
            return

        disp_img, scale = self.fit_image_to_display(self.original_image.copy())
        self.scale_x = scale
        self.scale_y = scale

        self.left_tk = ImageTk.PhotoImage(disp_img)
        self.left_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.left_canvas.create_image(x0, y0, anchor="nw", image=self.left_tk)

    def refresh_preview(self, *args):
        if self.original_image is None:
            return

        preview_img = self.original_image.copy()
        draw = ImageDraw.Draw(preview_img, "RGBA")

        # Draw saved annotations first.
        for idx, obj in enumerate(self.current_objects):
            self.draw_hyperbola_band(draw, obj, selected=(idx == self.selected_object_index))

        # Then draw preview from current control values.
        preview_obj = self.get_current_object_from_controls()
        self.draw_hyperbola_band(draw, preview_obj, color=(0, 255, 0, 110), outline=(0, 255, 0, 220), selected=False)

        disp_img, scale = self.fit_image_to_display(preview_img)
        self.right_tk = ImageTk.PhotoImage(disp_img)

        self.right_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.right_canvas.create_image(x0, y0, anchor="nw", image=self.right_tk)

        self.refresh_left_view()

    def on_left_click(self, event):
        if self.original_image is None:
            return

        disp_img, scale = self.fit_image_to_display(self.original_image)
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2

        if not (x0 <= event.x <= x0 + disp_img.width and y0 <= event.y <= y0 + disp_img.height):
            return

        img_x = (event.x - x0) / scale
        img_y = (event.y - y0) / scale

        self.var_x.set(img_x)
        self.var_y.set(img_y)

    def get_current_object_from_controls(self):
        return {
            "label": self.var_name.get().strip() or "hyperbola",
            "x_vertex": round(float(self.var_x.get()), 2),
            "y_vertex": round(float(self.var_y.get()), 2),
            "width": round(float(self.var_width.get()), 2),
            "height": round(float(self.var_height.get()), 2),
            "thickness": round(float(self.var_thickness.get()), 2),
        }

    def hyperbola_band_polygon(self, obj, n_points=120):
        x_vertex = obj["x_vertex"]
        y_vertex = obj["y_vertex"]
        width = max(2.0, obj["width"])
        height = max(1.0, obj["height"])
        thickness = max(1.0, obj["thickness"])

        half_w = width / 2.0
        x_left = x_vertex - half_w
        x_right = x_vertex + half_w

        upper_pts = []
        lower_pts = []

        for i in range(n_points + 1):
            t = i / n_points
            x = x_left + (x_right - x_left) * t
            dx = (x - x_vertex) / half_w
            y_center = y_vertex + height * (dx ** 2)
            upper_pts.append((x, y_center - thickness / 2.0))
            lower_pts.append((x, y_center + thickness / 2.0))

        polygon = upper_pts + list(reversed(lower_pts))
        centerline = []
        for i in range(n_points + 1):
            t = i / n_points
            x = x_left + (x_right - x_left) * t
            dx = (x - x_vertex) / half_w
            y_center = y_vertex + height * (dx ** 2)
            centerline.append((x, y_center))

        return polygon, centerline

    def draw_hyperbola_band(self, draw, obj, color=(255, 0, 0, 90), outline=(255, 0, 0, 220), selected=False):
        polygon, centerline = self.hyperbola_band_polygon(obj)
        draw.polygon(polygon, fill=color)
        draw.line(centerline, fill=outline, width=2)

        # Draw vertex point.
        xv = obj["x_vertex"]
        yv = obj["y_vertex"]
        r = 4 if not selected else 6
        draw.ellipse((xv - r, yv - r, xv + r, yv + r), fill=(255, 255, 0, 255))

        if selected:
            draw.line(centerline, fill=(0, 255, 255, 255), width=3)

    def draw_hyperbola_band_scaled(self, draw, obj, scale, color=(255, 0, 0, 90), outline=(255, 0, 0, 220), selected=False):
        polygon, centerline = self.hyperbola_band_polygon(obj)
        polygon = [(x * scale, y * scale) for x, y in polygon]
        centerline = [(x * scale, y * scale) for x, y in centerline]

        draw.polygon(polygon, fill=color)
        draw.line(centerline, fill=outline, width=2)

        xv = obj["x_vertex"] * scale
        yv = obj["y_vertex"] * scale
        r = 4 if not selected else 6
        draw.ellipse((xv - r, yv - r, xv + r, yv + r), fill=(255, 255, 0, 255))

        if selected:
            draw.line(centerline, fill=(0, 255, 255, 255), width=3)

    def add_object(self):
        if self.original_image is None:
            return
        obj = self.get_current_object_from_controls()
        self.current_objects.append(obj)
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
            text = (
                f"[{i}] {obj.get('label', 'hyperbola')} | "
                f"x={obj['x_vertex']:.1f}, y={obj['y_vertex']:.1f}, "
                f"w={obj['width']:.1f}, h={obj['height']:.1f}, t={obj['thickness']:.1f}"
            )
            self.object_listbox.insert(tk.END, text)

        if self.selected_object_index is not None and 0 <= self.selected_object_index < len(self.current_objects):
            self.object_listbox.selection_set(self.selected_object_index)

    def on_select_object(self, event):
        selection = self.object_listbox.curselection()
        if not selection:
            self.selected_object_index = None
        else:
            self.selected_object_index = selection[0]
        self.refresh_both_views()

    def load_selected_to_controls(self):
        if self.selected_object_index is None:
            messagebox.showinfo("Info", "Please select an annotation in the list first.")
            return
        obj = self.current_objects[self.selected_object_index]
        self.set_controls_from_object(obj)

    def set_controls_from_object(self, obj):
        self.var_name.set(obj.get("label", "hyperbola"))
        self.var_x.set(obj["x_vertex"])
        self.var_y.set(obj["y_vertex"])
        self.var_width.set(obj["width"])
        self.var_height.set(obj["height"])
        self.var_thickness.set(obj["thickness"])

    def save_current_annotations(self):
        if self.current_image_name is None:
            return
        self.annotations[self.current_image_name] = self.current_objects
        self.save_annotations_file()
        messagebox.showinfo("Saved", f"Current image annotations saved: {self.current_image_name}")

    def save_all(self):
        if self.current_image_name is not None:
            self.annotations[self.current_image_name] = self.current_objects
        self.save_annotations_file()
        messagebox.showinfo("Saved", f"All annotations have been saved to:\n{self.annotations_path}")

    def prev_image(self):
        if not self.image_paths:
            return
        self.annotations[self.current_image_name] = self.current_objects
        self.save_annotations_file()

        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.annotations[self.current_image_name] = self.current_objects
        self.save_annotations_file()

        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = HyperbolaAnnotator(root)
    root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    root.mainloop()