"""数据集 1 裁剪框人工核对工具（去掉截图的坐标轴白边）。

    python tools/annotation/crop_tool.py                  # 从第一张没确认的开始
    python tools/annotation/crop_tool.py --start 3245     # 从指定原图名开始

读写 data/primary_roboflow/hbat/crop.json（建议框由 d1_crop.py --suggest 生成，不存在时自动生成）。
图像始终读官方导出包里的原图（只读）。确认过的框由 d1_todo.py 应用到标注文件夹：
裁剪后统一缩放到 640×640，官方框与已有标注一起换算。

界面
  左：整图。红色半透明 = 会被裁掉的部分；绿色 = 官方框（裁剪框切到官方框时下方会提示）。
      鼠标拖动裁剪框的边或角来调整。
  右：四个角的放大图（每格一个原图像素）。在放大图里单击 = 把这个角移到点击的像素：
      左上 / 左下 / 右上 / 右下 格里点的是“裁剪后保留的第一个 / 最后一个像素”。
快捷键
  Enter / 空格   确认并到下一张          ← →（无选中边时）上一张 / 下一张；PageUp / PageDown 同
  1 2 3 4        选中 左 / 上 / 右 / 下 边；选中后 ← → ↑ ↓ 移动 1 px（Shift = 5 px），Esc 取消选中
  Ctrl+C / Ctrl+V  复制 / 粘贴裁剪框（同一版式的图连续粘贴）
  F              整图（不裁）            R  恢复建议框
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_crop  # noqa: E402

ZOOM = 6            # 放大图里一个原图像素的边长
ZWIN = 36           # 放大图的原图窗口（像素）
TINT = (255, 40, 40)
STATUS_TEXT = {'suggest_layout': '建议（版式）', 'suggest_auto': '建议（自动检测）',
               'edited': '已修改，未确认', 'checked': '已确认'}
EDGE_NAMES = ('左', '上', '右', '下')


def read_obb(path):
    boxes = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 8:
                    xs, ys = [float(v) for v in p[0:8:2]], [float(v) for v in p[1:8:2]]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


class CropTool:
    def __init__(self, root, R, start=None):
        self.root, self.R = root, R
        self.items = d1_crop.load_split(R)['items']
        self.doc = d1_crop.load_crops(R)
        if self.doc is None or len(self.doc.get('crops', {})) < len(self.items):
            print('生成建议框 …')
            self.doc = d1_crop.suggest(R)
        self.crops = self.doc['crops']
        self.names = sorted(self.items)
        self.idx = 0
        if start and start in self.names:
            self.idx = self.names.index(start)
        else:
            todo = [i for i, n in enumerate(self.names) if self.crops[n]['status'] != 'checked']
            self.idx = todo[0] if todo else 0
        self.clip = None
        self.sel_edge = None
        self.drag = None
        self.dirty = False
        self.touched = set()
        scr_h = root.winfo_screenheight()
        self.scale = max(0.8, min(1.4, (scr_h - 260) / 640))
        self.var_unchecked = tk.BooleanVar(value=False)
        self._build_ui()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.load()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        r.title('数据集 1 裁剪框核对')
        bar = ttk.Frame(r)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Button(bar, text='◀ 上一张', command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='下一张 ▶', command=lambda: self.step(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='确认并下一张 (Enter)', command=self.confirm).pack(side=tk.LEFT, padx=8)
        ttk.Button(bar, text='复制框', command=self.copy_box).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='粘贴框', command=self.paste_box).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='整图 (F)', command=self.full_box).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='恢复建议 (R)', command=self.reset_box).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(bar, text='只看未确认', variable=self.var_unchecked).pack(side=tk.LEFT, padx=10)
        self.var_goto = tk.StringVar()
        e = ttk.Entry(bar, textvariable=self.var_goto, width=14)
        e.pack(side=tk.LEFT, padx=(10, 2))
        e.bind('<Return>', lambda *_: self.goto())
        ttk.Button(bar, text='跳转', command=self.goto).pack(side=tk.LEFT)
        self.lbl_progress = ttk.Label(bar, text='')
        self.lbl_progress.pack(side=tk.RIGHT, padx=6)

        body = ttk.Frame(r)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8)
        n = int(640 * self.scale)
        self.cv = tk.Canvas(body, width=n, height=n, bg='#444', highlightthickness=0, cursor='crosshair')
        self.cv.pack(side=tk.LEFT)
        self.cv.bind('<Motion>', self.on_motion)
        self.cv.bind('<ButtonPress-1>', self.on_press)
        self.cv.bind('<B1-Motion>', self.on_drag)
        self.cv.bind('<ButtonRelease-1>', self.on_release)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, padx=10, anchor='n')
        zs = ZWIN * ZOOM
        self.zc = {}
        for k, (row, col, title) in {'tl': (0, 0, '左上'), 'tr': (0, 1, '右上'),
                                     'bl': (2, 0, '左下'), 'br': (2, 1, '右下')}.items():
            ttk.Label(right, text=title).grid(row=row, column=col)
            c = tk.Canvas(right, width=zs, height=zs, bg='#222', highlightthickness=0, cursor='tcross')
            c.grid(row=row + 1, column=col, padx=4, pady=2)
            c.bind('<ButtonPress-1>', lambda ev, k=k: self.on_zoom_click(k, ev))
            self.zc[k] = c

        form = ttk.Frame(right)
        form.grid(row=4, column=0, columnspan=2, pady=8, sticky='w')
        self.vars = []
        for i, t in enumerate(('x0 左', 'y0 上', 'x1 右', 'y1 下')):
            ttk.Label(form, text=t).grid(row=0, column=2 * i, padx=(6, 2))
            v = tk.StringVar()
            sp = ttk.Spinbox(form, from_=0, to=640, textvariable=v, width=5, command=self.on_form)
            sp.grid(row=0, column=2 * i + 1)
            sp.bind('<Return>', lambda *_: self.on_form())
            sp.bind('<FocusOut>', lambda *_: self.on_form())
            self.vars.append(v)
        self.lbl_info = ttk.Label(right, text='', justify=tk.LEFT, wraplength=zs * 2)
        self.lbl_info.grid(row=5, column=0, columnspan=2, sticky='w')
        self.lbl_warn = tk.Label(right, text='', fg='#c00', justify=tk.LEFT, wraplength=zs * 2)
        self.lbl_warn.grid(row=6, column=0, columnspan=2, sticky='w')
        ttk.Label(right, foreground='#666', justify=tk.LEFT, text=(
            'Enter/空格 确认并下一张；← → 上一张/下一张\n'
            '1–4 选中 左/上/右/下 边，方向键移动 1 px（Shift 5 px），Esc 取消\n'
            '放大图里单击 = 把该角移到点击的像素（保留的第一个/最后一个像素）\n'
            'Ctrl+C / Ctrl+V 复制粘贴框；F 整图；R 恢复建议')).grid(row=7, column=0, columnspan=2, sticky='w', pady=6)

        for key in ('<Return>', '<space>'):
            r.bind(key, self._key(self.confirm))
        r.bind('<Prior>', self._key(lambda: self.step(-1)))
        r.bind('<Next>', self._key(lambda: self.step(1)))
        r.bind('<Control-c>', self._key(self.copy_box))
        r.bind('<Control-v>', self._key(self.paste_box))
        r.bind('<KeyPress-f>', self._key(self.full_box))
        r.bind('<KeyPress-r>', self._key(self.reset_box))
        r.bind('<Escape>', self._key(lambda: self.select_edge(None)))
        for i in range(4):
            r.bind(f'<KeyPress-{i + 1}>', self._key(lambda i=i: self.select_edge(i)))
        for key, d in (('Left', -1), ('Right', 1), ('Up', -1), ('Down', 1)):
            r.bind(f'<{key}>', self._key(lambda key=key, d=d: self.arrow(key, d)))
            r.bind(f'<Shift-{key}>', self._key(lambda key=key, d=d: self.arrow(key, 5 * d)))

    def _key(self, fn):
        def h(ev):
            if isinstance(ev.widget, (ttk.Entry, ttk.Spinbox, tk.Entry)):
                return None
            fn()
            return 'break'
        return h

    # ------------------------------------------------------------------ data
    @property
    def name(self):
        return self.names[self.idx]

    @property
    def c(self):
        return self.crops[self.name]

    def load(self):
        it = self.items[self.name]
        self.img = Image.open(os.path.join(self.R, it['eval_file'])).convert('RGB')
        self.W, self.H = self.img.size
        i = it['files'].index(it['eval_file'])
        self.gt = read_obb(os.path.join(self.R, it['labels'][i]))
        self.base = self.img.resize((int(self.W * self.scale), int(self.H * self.scale)), Image.LANCZOS)
        self.sel_edge = None
        self.refresh()

    def save(self):
        """合并保存：重新读磁盘上的 crop.json，只写回本窗口改过条目的 box / status。
        applied 等字段由 d1_todo.py 维护，工具开着时 d1_todo 可能已经改过，不能用内存里的旧副本覆盖。"""
        disk = d1_crop.load_crops(self.R) or self.doc
        for n in self.touched:
            dc = disk['crops'].setdefault(n, dict(self.crops[n]))
            dc['box'] = list(self.crops[n]['box'])
            dc['status'] = self.crops[n]['status']
        d1_crop.save_crops(self.R, disk)
        self.doc, self.crops = disk, disk['crops']
        self.touched = set()
        self.dirty = False

    def set_box(self, box, edited=True):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0 = min(max(x0, 0), self.W - 8)
        y0 = min(max(y0, 0), self.H - 8)
        x1 = min(max(x1, x0 + 8), self.W)
        y1 = min(max(y1, y0 + 8), self.H)
        new = [x0, y0, x1, y1]
        if new != self.c['box']:
            self.c['box'] = new
            if edited and self.c['status'] != 'edited':
                self.c['status'] = 'edited'
            self.touched.add(self.name)
            self.dirty = True
        self.refresh()

    # ------------------------------------------------------------------ actions
    def step(self, d):
        if self.dirty:
            self.save()
        n = len(self.names)
        j = self.idx
        for _ in range(n):
            j = (j + d) % n
            if not self.var_unchecked.get() or self.crops[self.names[j]]['status'] != 'checked':
                break
        else:
            messagebox.showinfo('完成', '全部已确认')
            return
        self.idx = j
        self.load()

    def confirm(self):
        self.c['status'] = 'checked'
        self.touched.add(self.name)
        self.save()
        if all(c['status'] == 'checked' for c in self.crops.values()):
            self.refresh()
            messagebox.showinfo('完成', f'{len(self.crops)} 张全部已确认。关闭后运行 d1_todo.py 应用裁剪。')
            return
        self.step(1)

    def copy_box(self):
        self.clip = list(self.c['box'])
        self.refresh()

    def paste_box(self):
        if self.clip:
            self.set_box(self.clip)

    def full_box(self):
        self.set_box([0, 0, self.W, self.H])

    def reset_box(self):
        self.set_box(self.c['suggest'])

    def goto(self):
        q = self.var_goto.get().strip()
        hit = [i for i, n in enumerate(self.names) if n == q] or \
              [i for i, n in enumerate(self.names) if n.startswith(q)]
        if hit:
            if self.dirty:
                self.save()
            self.idx = hit[0]
            self.load()

    def select_edge(self, i):
        self.sel_edge = i
        self.refresh()

    def arrow(self, key, d):
        if self.sel_edge is None:
            if key in ('Left', 'Right'):
                self.step(-1 if key == 'Left' else 1)
            return
        horizontal = self.sel_edge in (0, 2)
        if horizontal != (key in ('Left', 'Right')):
            return
        b = list(self.c['box'])
        b[self.sel_edge] += d
        self.set_box(b)

    def on_form(self):
        try:
            b = [int(float(v.get())) for v in self.vars]
        except ValueError:
            return
        self.set_box(b)

    # ------------------------------------------------------------------ mouse
    def _hit(self, ex, ey):
        """离鼠标最近的边（显示坐标 8 px 以内），返回边的下标列表。"""
        s = self.scale
        x0, y0, x1, y1 = (v * s for v in self.c['box'])
        hits = []
        inside_y = y0 - 8 <= ey <= y1 + 8
        inside_x = x0 - 8 <= ex <= x1 + 8
        if inside_y and abs(ex - x0) <= 8:
            hits.append(0)
        if inside_y and abs(ex - x1) <= 8:
            hits.append(2)
        if inside_x and abs(ey - y0) <= 8:
            hits.append(1)
        if inside_x and abs(ey - y1) <= 8:
            hits.append(3)
        return hits

    def on_motion(self, ev):
        h = self._hit(ev.x, ev.y)
        cur = 'crosshair'
        if h == [0] or h == [2]:
            cur = 'sb_h_double_arrow'
        elif h == [1] or h == [3]:
            cur = 'sb_v_double_arrow'
        elif len(h) == 2:
            cur = 'fleur'
        self.cv.configure(cursor=cur)

    def on_press(self, ev):
        self.drag = self._hit(ev.x, ev.y) or None
        if self.drag and len(self.drag) == 1:
            self.sel_edge = self.drag[0]

    def on_drag(self, ev):
        if not self.drag:
            return
        b = list(self.c['box'])
        x, y = ev.x / self.scale, ev.y / self.scale
        for e in self.drag:
            b[e] = x if e in (0, 2) else y
        self.set_box(b)

    def on_release(self, _):
        self.drag = None

    def _zoom_origin(self, k):
        x0, y0, x1, y1 = self.c['box']
        cx = x0 if k[1] == 'l' else x1
        cy = y0 if k[0] == 't' else y1
        ox = min(max(cx - ZWIN // 2, 0), self.W - ZWIN)
        oy = min(max(cy - ZWIN // 2, 0), self.H - ZWIN)
        return ox, oy

    def on_zoom_click(self, k, ev):
        ox, oy = self._zoom_origin(k)
        px, py = ox + ev.x // ZOOM, oy + ev.y // ZOOM
        b = list(self.c['box'])
        if k[1] == 'l':
            b[0] = px
        else:
            b[2] = px + 1
        if k[0] == 't':
            b[1] = py
        else:
            b[3] = py + 1
        self.set_box(b)

    # ------------------------------------------------------------------ drawing
    def refresh(self):
        c = self.c
        x0, y0, x1, y1 = c['box']
        s = self.scale
        # 主图：裁掉的部分染红
        im = self.base.copy()
        mask = Image.new('L', im.size, 110)
        ImageDraw.Draw(mask).rectangle([x0 * s, y0 * s, x1 * s - 1, y1 * s - 1], fill=0)
        im = Image.composite(Image.new('RGB', im.size, TINT), im, mask)
        dr = ImageDraw.Draw(im)
        for g in self.gt:
            dr.rectangle([g[0] * s, g[1] * s, g[2] * s, g[3] * s], outline=(0, 230, 0))
        for i, (a, b) in enumerate((((x0, y0), (x0, y1)), ((x0, y0), (x1, y0)),
                                    ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)))):
            col = (255, 220, 0) if i == self.sel_edge else (255, 0, 0)
            dr.line([a[0] * s, a[1] * s, b[0] * s, b[1] * s], fill=col, width=3 if i == self.sel_edge else 1)
        self.tk_main = ImageTk.PhotoImage(im)
        self.cv.delete('all')
        self.cv.create_image(0, 0, anchor='nw', image=self.tk_main)
        # 四角放大
        self.tk_zoom = {}
        arr = np.asarray(self.img)
        for k, cvz in self.zc.items():
            ox, oy = self._zoom_origin(k)
            sub = arr[oy:oy + ZWIN, ox:ox + ZWIN].copy()
            keep = np.zeros(sub.shape[:2], bool)
            keep[max(0, y0 - oy):max(0, y1 - oy), max(0, x0 - ox):max(0, x1 - ox)] = True
            sub[~keep] = (sub[~keep] * 0.55 + np.array(TINT) * 0.45).astype(np.uint8)
            z = Image.fromarray(sub).resize((ZWIN * ZOOM, ZWIN * ZOOM), Image.NEAREST)
            dz = ImageDraw.Draw(z)
            for xx in (x0, x1):
                if 0 <= xx - ox <= ZWIN:
                    dz.line([(xx - ox) * ZOOM, 0, (xx - ox) * ZOOM, ZWIN * ZOOM], fill=(255, 0, 0))
            for yy in (y0, y1):
                if 0 <= yy - oy <= ZWIN:
                    dz.line([0, (yy - oy) * ZOOM, ZWIN * ZOOM, (yy - oy) * ZOOM], fill=(255, 0, 0))
            self.tk_zoom[k] = ImageTk.PhotoImage(z)
            cvz.delete('all')
            cvz.create_image(0, 0, anchor='nw', image=self.tk_zoom[k])
        # 表单与信息
        for v, val in zip(self.vars, c['box']):
            v.set(str(val))
        OW, OH = d1_crop.OUTPUT_SIZE
        w, h = x1 - x0, y1 - y0
        sx, sy = OW / w, OH / h
        info = (f'{self.name}    {c["file"]}\n'
                f'状态：{STATUS_TEXT.get(c["status"], c["status"])}    建议来源：{c.get("suggest_source", "")}\n'
                f'裁掉：左 {x0}  上 {y0}  右 {self.W - x1}  下 {self.H - y1} px；保留 {w}×{h}\n'
                f'输出 {OW}×{OH}：横向 ×{sx:.3f}，纵向 ×{sy:.3f}（纵横比变化 {abs(sx / sy - 1) * 100:.1f}%）')
        if self.sel_edge is not None:
            info += f'\n选中：{EDGE_NAMES[self.sel_edge]}边'
        if self.clip:
            info += f'\n剪贴板：{self.clip}'
        self.lbl_info.configure(text=info)
        warn = []
        for i, g in enumerate(self.gt):
            area = max(1e-6, (g[2] - g[0]) * (g[3] - g[1]))
            iw = max(0.0, min(g[2], x1) - max(g[0], x0))
            ih = max(0.0, min(g[3], y1) - max(g[1], y0))
            out = 1 - iw * ih / area
            if out > 0.02:
                warn.append(f'官方框 {i}：{out * 100:.0f}% 在裁剪框外')
        self.lbl_warn.configure(text='\n'.join(warn))
        n_ok = sum(1 for v in self.crops.values() if v['status'] == 'checked')
        self.lbl_progress.configure(text=f'{self.idx + 1}/{len(self.names)}    已确认 {n_ok}/{len(self.crops)}')

    def on_close(self):
        if self.dirty:
            self.save()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='仓库根目录（默认按脚本位置推出，可在任意目录运行）')
    ap.add_argument('--start', default=None, help='从这个原图名开始')
    args = ap.parse_args()
    root = tk.Tk()
    CropTool(root, args.repo, args.start)
    root.mainloop()


if __name__ == '__main__':
    main()
