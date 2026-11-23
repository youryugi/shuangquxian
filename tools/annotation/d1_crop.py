"""数据集 1：去掉截图留下的坐标轴白边，只保留雷达图绘图区，并统一输出尺寸。

背景（2026-11-23）：数据集 1 的大部分图是 GPR 软件的截图，上边和左边（不少还有右边、下边）带着白底的
坐标轴区域：刻度、数字、轴标题、状态栏。这些区域不是数据，而且 HBAT 的双曲线模型以图像顶边为地表
（a = y_vertex），顶部白边会让模型的地表位置错位。所以标注用的工作副本只保留绘图区。

流程
  1. 建议框（本模块 --suggest）：截图版式相同的图白边尺寸相同，已逐张核对过的 7 种版式（LAYOUTS）
     直接给统一的框；其余的图（line* 系列没有白边；hyperbola1–25、56、59、80–99 是从长剖面里截的，
     位置和缩放各不相同）用自动检测给初值。
  2. 人工核对：tools/annotation/crop_tool.py 逐张显示建议框和四个角的放大图，微调后按 Enter 确认。
  3. 应用（d1_todo.py 每次运行时调用 apply_checked）：只有确认过的框才生效。图像裁剪后统一缩放到
     OUTPUT_SIZE（640×640，与 Roboflow 导出时的拉伸处理一致），写进 hbat/images；官方框写进
     hbat/labelTxt；已有标注换算到新坐标。

data/primary_roboflow/hbat/crop.json
    output_size  [640, 640]
    crops.<原图名>:
        file       eval_file 的文件名（= 标注文件的键）
        orig_size  [W, H]
        box        [x0, y0, x1, y1]  绘图区，原图像素，右下开（裁剪 = img[y0:y1, x0:x1]）
        status     suggest_layout / suggest_auto / edited（改过但没确认）/ checked
        suggest    建议框（"恢复建议" 用）
        applied    hbat/images 与 annotations.json 当前所在的坐标系对应的框；没有 = 整图
同一原图名的 3 份光度副本几何相同，共用同一个框。

坐标系：frame(box) 把原图坐标 (x, y) 映射到 ((x − x0)·OW/(x1 − x0), (y − y0)·OH/(y1 − y0))。
  * 官方框：4 个角点逐点映射后截到图内；行数和顺序不变（gt_box_id 依赖行号）。
  * HBAT band：顶点映射；span × 横向缩放，thickness、tail × 纵向缩放；HBAT 模型以图像顶边为地表，
    顶边移动后 slope 重新拟合以保持同一条中心线（d1_duplicates.transfer_object）。

用法（在仓库根目录，需 cv2）：
    python tools/annotation/d1_crop.py --suggest       # 生成 / 更新建议框（不动已确认的）
    python tools/annotation/crop_tool.py               # 人工核对
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPLIT = 'data/splits/dataset1_roboflow.json'
HBAT_DIR = 'data/primary_roboflow/hbat'
CROP_FILE = f'{HBAT_DIR}/crop.json'
OUTPUT_SIZE = (640, 640)
# 换算后中心线偏差超过这个值（新坐标像素 RMS）的目标打上 REVIEW_KEY，d1_todo 把这些图列为 review
REVIEW_RMS = 1.0
REVIEW_KEY = 'crop_refit_rms'

# 已逐张核对（四角放大图）的截图版式：同一版式白边尺寸相同，用统一的框。
LAYOUTS = {
    'A': ([39, 32, 639, 625], ['3170', '3172', '3174', '3176', '3178', '3180', '3182', '3184', '3186', '3188',
                               '3198', '3200', '3202', '3204', '3206', '3208', '3210', '3212', '3214', '3216',
                               '3232', '3234']),
    'B': ([48, 33, 638, 625], ['3218', '3220', '3222', '3224', '3226', '3228', '3230']),
    'C': ([23, 39, 640, 625], ['3241', '3243']),
    'D': ([29, 32, 639, 625], ['3247', '3249', '3254', '3262', '3263', '3264', '3269', '3279']),
    'E': ([27, 32, 640, 625], ['3289', '3290', '3291', '3292', '3296']),
    'F': ([26, 32, 640, 625], ['hyperbola28', 'hyperbola29', 'hyperbola30', 'hyperbola31', 'hyperbola32',
                               'hyperbola39', 'hyperbola43', 'hyperbola44', 'hyperbola45', 'hyperbola60',
                               'hyperbola61', 'hyperbola64', 'hyperbola65', 'hyperbola66', 'hyperbola68',
                               'hyperbola70', 'hyperbola78', 'hyperbola79']),
    'G': ([35, 32, 639, 625], ['hyperbola46', 'hyperbola47', 'hyperbola48', 'hyperbola49', 'hyperbola50',
                               'hyperbola51']),
}

# 自动检测参数（只用来给初值，最终以人工确认为准）
WHITE_MIN = 200      # 白边像素的亮度阈值上限（背景本身更暗时取 背景 − BG_TOL）
BG_TOL = 30
BG_MIN = 150         # 最外几行（列）比这个暗 = 这条边没有白边
SAT_MAX = 25         # 白边是无彩色的
PLOT_MAX = 0.5       # 绘图区的行（列）白像素比例 < 这个值
RUN = 6              # 连续这么多行（列）都满足才算进入绘图区
JUMP = 0.45          # 白边 → 绘图区的突变幅度
SEARCH = 0.25        # 每条边只在 25% 范围内找


def imread(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite_jpg(path, img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok, path
    buf.tofile(path)


# ------------------------------------------------------------------ 自动检测（初值）
def _edge(p):
    """p: 从图像边缘往里的白像素比例。返回白边宽度（0 = 没有白边）。"""
    n = len(p)
    for y in range(0, int(SEARCH * n)):
        if p[y:y + RUN].max() < PLOT_MAX:
            if y == 0:
                return 0
            before = p[max(0, y - 3):y].max()
            after = p[min(y + 1, n - 1)]
            return y if before - after >= JUMP else 0
    return 0


def detect_plot_box(img):
    """BGR 图像 → 绘图区 [x0, y0, x1, y1]（原图坐标，右下开）。最外侧 ≤3 px 的黑色边框线算作白边。"""
    h, w = img.shape[:2]
    im = img.astype(np.int16)
    gray = im.mean(axis=2)
    sat = im.max(axis=2) - im.min(axis=2)
    x0, y0, x1, y1 = 0, 0, w, h
    for _ in range(2):
        g, s = gray[y0:y1, x0:x1], sat[y0:y1, x0:x1]
        d = {}
        for side, axis, flip in (('t', 1, False), ('b', 1, True), ('l', 0, False), ('r', 0, True)):
            gg, ss = (g, s) if axis == 1 else (g.T, s.T)        # 行方向
            if flip:
                gg, ss = gg[::-1], ss[::-1]
            med = np.median(gg[:6], axis=1)
            lead = 0
            while lead < 3 and med[lead] < 100:
                lead += 1
            B = float(np.median(med[lead:lead + 2]))
            if B < BG_MIN:
                d[side] = 0
                continue
            m = (gg >= min(B - BG_TOL, WHITE_MIN)) & (ss <= SAT_MAX)
            p = m.mean(axis=1)
            p[:lead] = 1.0
            d[side] = _edge(p)
        x0, y0, x1, y1 = x0 + d['l'], y0 + d['t'], x1 - d['r'], y1 - d['b']
    return [int(x0), int(y0), int(x1), int(y1)]


# ------------------------------------------------------------------ crop.json
def load_split(R):
    with open(os.path.join(R, SPLIT), encoding='utf-8') as f:
        return json.load(f)


def load_crops(R):
    p = os.path.join(R, CROP_FILE)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_crops(R, doc):
    p = os.path.join(R, CROP_FILE)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def suggest(R, verbose=True):
    """生成 / 更新建议框。已确认（checked）或改过（edited）的条目不动。"""
    items = load_split(R)['items']
    doc = load_crops(R) or {}
    doc['_doc'] = ('d1_crop.py / crop_tool.py：数据集 1 每个原图名的绘图区裁剪框 box [x0, y0, x1, y1]（原图像素，右下开），'
                   '裁剪后统一缩放到 output_size。applied = hbat/images 与 annotations.json 当前对应的框。说明见 d1_crop.py。')
    doc['output_size'] = list(OUTPUT_SIZE)
    crops = doc.setdefault('crops', {})
    layout_of = {n: (k, box) for k, (box, ms) in LAYOUTS.items() for n in ms}
    n_new = 0
    for name, it in sorted(items.items()):
        c = crops.get(name)
        if c and c.get('status') in ('checked', 'edited'):
            continue
        img = imread(os.path.join(R, it['eval_file']))
        h, w = img.shape[:2]
        if name in layout_of:
            k, box = layout_of[name]
            st, src = 'suggest_layout', f'layout {k}'
        else:
            box, st, src = detect_plot_box(img), 'suggest_auto', 'auto'
        new = {'file': os.path.basename(it['eval_file']), 'orig_size': [w, h], 'box': list(box),
               'status': st, 'suggest': list(box), 'suggest_source': src}
        if c and 'applied' in c:
            new['applied'] = c['applied']
        if c != new:
            crops[name] = new
            n_new += 1
    save_crops(R, doc)
    if verbose:
        st = [c['status'] for c in crops.values()]
        print(f'{CROP_FILE}：{len(crops)} 张，更新 {n_new} 条；' +
              '，'.join(f'{s} {st.count(s)}' for s in ('suggest_layout', 'suggest_auto', 'edited', 'checked')))
    return doc


# ------------------------------------------------------------------ 坐标换算
def frame_matrix(box, out_size=OUTPUT_SIZE):
    """原图坐标 → 裁剪并缩放后坐标，3×3。box 为 None 表示整图不变（调用方负责）。"""
    x0, y0, x1, y1 = box
    sx, sy = out_size[0] / (x1 - x0), out_size[1] / (y1 - y0)
    return np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]], float)


def applied_matrix(c, out_size=OUTPUT_SIZE):
    """crop.json 条目当前生效的 原图 → 工作坐标 矩阵（未应用 = 单位阵，图像保持原尺寸）。"""
    if not c or not c.get('applied'):
        return np.eye(3)
    return frame_matrix(c['applied'], out_size)


def is_full(box, orig_size):
    return list(box) == [0, 0, orig_size[0], orig_size[1]]


def render(img, box, out_size=OUTPUT_SIZE):
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1]
    if (x1 - x0, y1 - y0) == tuple(out_size):
        return sub.copy()
    interp = cv2.INTER_AREA if (x1 - x0) > out_size[0] else cv2.INTER_LINEAR
    return cv2.resize(sub, tuple(out_size), interpolation=interp)


def transform_obb_text(text, M, out_size):
    """Roboflow labelTxt（每行 4 个点 + 类别 + difficult）→ 新坐标。行数、顺序不变（截到图内）。"""
    W, H = out_size
    out = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 8:
            if ln.strip():
                out.append(ln)
            continue
        pts = []
        for x, y in zip(parts[0:8:2], parts[1:8:2]):
            u, v, _ = M @ np.array([float(x), float(y), 1.0])
            pts.append(f'{min(max(u, 0.0), W):.2f} {min(max(v, 0.0), H):.2f}')
        out.append(' '.join(pts + parts[8:]))
    return '\n'.join(out) + ('\n' if out else '')


def centerline_rms(src, dst, M):
    """换算前后中心线的偏差（新坐标像素 RMS），在新图范围内采样。"""
    import d1_duplicates as dd
    import hbat
    o, n = hbat.normalize_obj(src), hbat.normalize_obj(dst)
    if not (hbat.has_band(o) and hbat.has_band(n)):
        return 0.0
    xs = np.linspace(o['x_vertex'] - o['span'] / 2, o['x_vertex'] + o['span'] / 2, 101)
    ys = dd._centerline(o['x_vertex'], o['y_vertex'], o['slope'], xs)
    P = np.stack([xs, ys, np.ones_like(xs)], 1) @ M.T
    ok = (P[:, 0] >= 0) & (P[:, 0] <= OUTPUT_SIZE[0]) & (P[:, 1] >= 0) & (P[:, 1] <= OUTPUT_SIZE[1])
    if ok.sum() < 3:
        return 0.0
    yc = dd._centerline(n['x_vertex'], n['y_vertex'], n['slope'], P[ok, 0])
    return float(np.sqrt(np.mean((yc - P[ok, 1]) ** 2)))


def transform_objects(objs, M, out_size, gt_boxes=None):
    """一张图的 HBAT 目标换到新坐标；给了 gt_boxes 就按新坐标重新分配 gt_box_id。返回 (新列表, 最大中心线 RMS)。"""
    import d1_duplicates as dd
    import hbat
    out, worst = [], 0.0
    for o in objs:
        t = dd.transfer_object(o, M[:2], out_size)
        for k in ('source', 'copied_from'):
            if k in o:
                t[k] = o[k]
        if gt_boxes is not None:
            gid, _ = hbat.assign_gt_box(t, gt_boxes, out_size)
            t['gt_box_id'] = gid
        rms = max(centerline_rms(o, t, M), float(o.get(REVIEW_KEY) or 0.0))
        t.pop(REVIEW_KEY, None)
        if rms > REVIEW_RMS:
            t[REVIEW_KEY] = round(rms, 2)    # HBAT 里更新这个目标时会自动去掉（HBAT 按控件重建对象）
        worst = max(worst, rms)
        out.append(t)
    return out, worst


# ------------------------------------------------------------------ 应用到标注文件夹
def check_applied(R, items, crops):
    """核对 hbat/images 里的图是否就是 applied 框生成的（applied 为空 = 官方原图原样）。
    返回不一致的原图名列表。crop.json 的 applied 被意外改掉时，继续应用会把标注重复换算，必须先停下。"""
    img_dir = os.path.join(R, HBAT_DIR, 'images')
    bad = []
    for name, it in sorted(items.items()):
        dst = os.path.join(img_dir, os.path.basename(it['eval_file']))
        if not os.path.exists(dst):
            continue
        have = np.fromfile(dst, np.uint8).tobytes()
        src = os.path.join(R, it['eval_file'])
        box = (crops.get(name) or {}).get('applied')
        if not box:
            want = np.fromfile(src, np.uint8).tobytes()
        else:
            ok, buf = cv2.imencode('.jpg', render(imread(src), box), [cv2.IMWRITE_JPEG_QUALITY, 100])
            want = buf.tobytes()
        if have != want:
            bad.append(name)
    return bad


def sync_image(R, it, c):
    """按 applied 生成 hbat/images 与 hbat/labelTxt 中的文件（不存在时）。返回新写的文件数。"""
    img_dir = os.path.join(R, HBAT_DIR, 'images')
    lab_dir = os.path.join(R, HBAT_DIR, 'labelTxt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    i = it['files'].index(it['eval_file'])
    src_img, src_lab = os.path.join(R, it['eval_file']), os.path.join(R, it['labels'][i])
    dst_img = os.path.join(img_dir, os.path.basename(src_img))
    dst_lab = os.path.join(lab_dir, os.path.basename(src_lab))
    n = 0
    box = (c or {}).get('applied')
    if not box:
        for s, d in ((src_img, dst_img), (src_lab, dst_lab)):
            if not os.path.exists(d):
                shutil.copy2(s, d)
                n += 1
        return n
    if not os.path.exists(dst_img):
        imwrite_jpg(dst_img, render(imread(src_img), box))
        n += 1
    if not os.path.exists(dst_lab):
        with open(src_lab, encoding='utf-8') as f:
            txt = transform_obb_text(f.read(), frame_matrix(box), OUTPUT_SIZE)
        with open(dst_lab, 'w', encoding='utf-8') as f:
            f.write(txt)
        n += 1
    return n


def apply_checked(R, items, doc, ann, verbose=True):
    """把已确认、且与当前 applied 不同的框应用到 hbat/images、hbat/labelTxt 和 ann（原地修改）。

    ann 里属于这个原图名的所有键（3 份副本）都换算，坐标系始终与 hbat/images 一致。
    返回 (应用的张数, [(原图名, 最大中心线 RMS)])。调用方负责保存 ann 和 doc。"""
    import hbat
    gt = hbat.GTSource(os.path.join(R, HBAT_DIR, 'labelTxt'))
    n, report = 0, []
    for name, c in sorted(doc['crops'].items()):
        if c.get('status') != 'checked':
            continue
        it = items[name]
        new = None if is_full(c['box'], c['orig_size']) else list(c['box'])
        if new == c.get('applied'):
            continue
        M = (frame_matrix(new) if new else np.eye(3)) @ np.linalg.inv(applied_matrix(c))
        out_size = tuple(OUTPUT_SIZE) if new else tuple(c['orig_size'])
        # 先换图像和官方框（删掉旧文件再按新框生成）
        i = it['files'].index(it['eval_file'])
        for p in (os.path.join(R, HBAT_DIR, 'images', os.path.basename(it['eval_file'])),
                  os.path.join(R, HBAT_DIR, 'labelTxt', os.path.basename(it['labels'][i]))):
            if os.path.exists(p):
                os.remove(p)
        c['applied'] = new
        if new is None:
            c.pop('applied')
        sync_image(R, it, c)
        boxes = gt.boxes_for(c['file'], out_size)
        worst = 0.0
        for f in it['files']:
            k = os.path.basename(f)
            if ann.get(k):
                ann[k], w = transform_objects(ann[k], M, out_size, boxes)
                worst = max(worst, w)
        n += 1
        report.append((name, worst))
    if verbose and n:
        bad = [(k, w) for k, w in report if w > 1.0]
        print(f'已应用 {n} 张的裁剪框（图像、官方框、标注一起换算）' +
              (f'；中心线换算偏差 > 1 px：' + '，'.join(f'{k} {w:.1f}' for k, w in bad) if bad else ''))
    return n, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='仓库根目录（默认按脚本位置推出，可在任意目录运行）')
    ap.add_argument('--suggest', action='store_true', help='生成 / 更新建议框（不动已确认的）')
    args = ap.parse_args()
    if args.suggest:
        suggest(args.repo)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
