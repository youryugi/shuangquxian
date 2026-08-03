import os
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

BOX_FILL       = (0, 255, 0, 70)
BOX_OUTLINE    = (0, 255, 0, 200)
LIVE_OUTLINE   = (255, 255, 0, 200)
SELECT_OUTLINE = (0, 255, 255, 255)


# ---------------------------------------------------------------------------
# Standard YOLO detection label format:
#   <folder>/<image>.jpg
#   <folder>/<image>.txt   -> one line per box: "class_id xc yc w h" (0..1)
#   <folder>/classes.txt   -> one class name per line, index = class_id
# ---------------------------------------------------------------------------

def read_yolo_txt(label_path, classes, img_w, img_h):
    objs = []
    if not os.path.exists(label_path):
        return objs
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                xc, yc, ww, hh = (float(v) for v in parts[1:5])
                xc *= img_w; yc *= img_h; ww *= img_w; hh *= img_h
                label = classes[cid] if 0 <= cid < len(classes) else str(cid)
                objs.append({
                    "label": label,
                    "x1": xc - ww / 2.0, "y1": yc - hh / 2.0,
                    "x2": xc + ww / 2.0, "y2": yc + hh / 2.0,
                })
    except Exception:
        pass
    return objs


def write_yolo_txt(label_path, objects, classes, img_w, img_h):
    lines = []
    for obj in objects:
        label = obj.get("label", "object")
        if label not in classes:
            classes.append(label)
        cid = classes.index(label)

        x1, x2 = sorted((obj["x1"], obj["x2"]))
        y1, y2 = sorted((obj["y1"], obj["y2"]))
        x1 = max(0.0, min(img_w, x1)); x2 = max(0.0, min(img_w, x2))
        y1 = max(0.0, min(img_h, y1)); y2 = max(0.0, min(img_h, y2))

        xc = (x1 + x2) / 2.0 / img_w
        yc = (y1 + y2) / 2.0 / img_h
        ww = (x2 - x1) / img_w
        hh = (y2 - y1) / img_h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_classes_file(classes_path, classes):
    with open(classes_path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")


class BBoxAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO BBox Annotation Tool")
        self.root.geometry("1900x900")

        self.image_dir = ""
        self.image_paths = []
        self.current_index = 0

        self.classes_path = ""
        self.classes = []

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
        self._dragging = False

        self.var_x1   = tk.DoubleVar(value=100)
        self.var_y1   = tk.DoubleVar(value=100)
        self.var_x2   = tk.DoubleVar(value=300)
        self.var_y2   = tk.DoubleVar(value=300)
        self.var_name = tk.StringVar(value="object")
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

        self.info_label = ttk.Label(top_bar, text="Please select an image folder first")
        self.info_label.pack(side=tk.LEFT, padx=12)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_panel = ttk.LabelFrame(image_frame, text="Original Image (drag to draw a box)")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        self.left_canvas = tk.Canvas(left_panel, width=self.display_w, height=self.display_h, bg="black")
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.left_canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.left_canvas.bind("<B1-Motion>", self.on_left_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self.on_left_release)

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

        self._make_slider(self.control_frame, "x1", self.var_x1, row, 0, 1000); row += 1
        self._make_slider(self.control_frame, "y1", self.var_y1, row, 0, 1000); row += 1
        self._make_slider(self.control_frame, "x2", self.var_x2, row, 0, 1000); row += 1
        self._make_slider(self.control_frame, "y2", self.var_y2, row, 0, 1000); row += 1

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
            "1. Drag on the left image to draw a box;\n"
            "   releasing the mouse adds it automatically\n"
            "2. x1,y1 = top-left corner, x2,y2 = bottom-\n"
            "   right corner (pixel coords), editable too\n"
            "3. Select an item in the list, then\n"
            "   Update / Delete / Load Parameters\n"
            "4. Next/Prev moves through images one by one\n"
            "5. Saved as YOLO txt (class xc yc w h, 0..1)\n"
            "   + classes.txt in the image folder"
        )
        ttk.Label(self.control_frame, text=hint, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=10)

        for v in [self.var_x1, self.var_y1, self.var_x2, self.var_y2]:
            v.trace_add("write", lambda *_: self.refresh_preview())

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
        self.classes_path = os.path.join(folder, "classes.txt")
        self.load_classes_file()
        self.load_image()

    def load_classes_file(self):
        self.classes = []
        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, "r", encoding="utf-8") as f:
                    self.classes = [line.strip() for line in f if line.strip()]
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to read classes.txt.\n{e}")
                self.classes = []

    def _label_path_for(self, img_path):
        base, _ = os.path.splitext(img_path)
        return base + ".txt"

    def _save_current_to_disk(self):
        if not self.image_paths or self.original_image is None:
            return
        img_path = self.image_paths[self.current_index]
        w, h = self.original_image.size
        write_yolo_txt(self._label_path_for(img_path), self.current_objects, self.classes, w, h)
        write_classes_file(self.classes_path, self.classes)

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

        self.current_objects = read_yolo_txt(self._label_path_for(img_path), self.classes, w, h)

        self.selected_object_index = None
        self.refresh_object_list()

        if self.current_objects:
            self.set_controls_from_object(self.current_objects[0])
        else:
            self.var_x1.set(w * 0.25)
            self.var_y1.set(h * 0.25)
            self.var_x2.set(w * 0.5)
            self.var_y2.set(h * 0.5)

        self.refresh_both_views()

        self.info_label.config(text=self.current_image_name)

    def update_slider_ranges(self, w, h):
        if "x1" in self.sliders:
            self.sliders["x1"].configure(from_=0, to=max(1, w - 1))
        if "x2" in self.sliders:
            self.sliders["x2"].configure(from_=0, to=max(1, w - 1))
        if "y1" in self.sliders:
            self.sliders["y1"].configure(from_=0, to=max(1, h - 1))
        if "y2" in self.sliders:
            self.sliders["y2"].configure(from_=0, to=max(1, h - 1))
        if self.var_x1.get() > w: self.var_x1.set(w / 4)
        if self.var_x2.get() > w: self.var_x2.set(w / 2)
        if self.var_y1.get() > h: self.var_y1.set(h / 4)
        if self.var_y2.get() > h: self.var_y2.set(h / 2)

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
            self.draw_bbox(draw, obj, selected=selected)

        # Draw live preview from controls
        self.draw_bbox(draw, self.get_current_object_from_controls(),
                        color=(0, 0, 0, 0), outline=LIVE_OUTLINE, selected=False)

        disp_img, _ = self.fit_image_to_display(preview_img)
        self.right_tk = ImageTk.PhotoImage(disp_img)
        self.right_canvas.delete("all")
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        self.right_canvas.create_image(x0, y0, anchor="nw", image=self.right_tk)
        self.refresh_left_view()

    # ------------------------------------------------------ drawing ---------
    def draw_bbox(self, draw, obj, color=BOX_FILL, outline=BOX_OUTLINE, selected=False):
        x1, x2 = sorted((obj["x1"], obj["x2"]))
        y1, y2 = sorted((obj["y1"], obj["y2"]))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=outline, width=3 if selected else 2)
        label = obj.get("label", "object")
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(255, 255, 0, 255))
        if selected:
            draw.rectangle([x1, y1, x2, y2], outline=SELECT_OUTLINE, width=3)

    # ---------------------------------------------------------- events ------
    def _canvas_to_image_coords(self, event):
        if self.original_image is None:
            return None
        disp_img, scale = self.fit_image_to_display(self.original_image)
        x0 = (self.display_w - disp_img.width) // 2
        y0 = (self.display_h - disp_img.height) // 2
        w, h = self.original_image.size
        ix = max(0.0, min(w, (event.x - x0) / scale))
        iy = max(0.0, min(h, (event.y - y0) / scale))
        return ix, iy

    def on_left_press(self, event):
        pt = self._canvas_to_image_coords(event)
        if pt is None:
            return
        self._dragging = True
        self.var_x1.set(pt[0]); self.var_y1.set(pt[1])
        self.var_x2.set(pt[0]); self.var_y2.set(pt[1])

    def on_left_drag(self, event):
        if not self._dragging:
            return
        pt = self._canvas_to_image_coords(event)
        if pt is None:
            return
        self.var_x2.set(pt[0]); self.var_y2.set(pt[1])

    def on_left_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        if abs(self.var_x2.get() - self.var_x1.get()) < 3 or abs(self.var_y2.get() - self.var_y1.get()) < 3:
            return  # too small: treat as an accidental click, not a box
        self.add_object()

    def get_current_object_from_controls(self):
        x1, x2 = sorted((round(float(self.var_x1.get()), 2), round(float(self.var_x2.get()), 2)))
        y1, y2 = sorted((round(float(self.var_y1.get()), 2), round(float(self.var_y2.get()), 2)))
        return {
            "label": self.var_name.get().strip() or "object",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
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
            self.object_listbox.insert(
                tk.END,
                f"[{i}] {obj.get('label', 'object')} | "
                f"x1={obj['x1']:.1f}, y1={obj['y1']:.1f}, "
                f"x2={obj['x2']:.1f}, y2={obj['y2']:.1f}"
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
        self.var_name.set(obj.get("label", "object"))
        self.var_x1.set(obj["x1"])
        self.var_y1.set(obj["y1"])
        self.var_x2.set(obj["x2"])
        self.var_y2.set(obj["y2"])

    # --------------------------------------------------- save / navigate ----
    def save_current_annotations(self):
        if self.current_image_name is None:
            return
        self._save_current_to_disk()
        messagebox.showinfo("Saved", f"Saved: {self._label_path_for(self.image_paths[self.current_index])}")

    def prev_image(self):
        if not self.image_paths:
            return
        self._save_current_to_disk()
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_image()

    def next_image(self):
        if not self.image_paths:
            return
        self._save_current_to_disk()
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = BBoxAnnotator(root)
    root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    root.mainloop()
