"""数据集 1 中“同一张图”的检测与标注复制。

背景（2026-11-22 核查）：数据集 1 的 170 个原图名里，有些是同一张雷达图被截了好几次屏——显示设置
（对比度、配色）或裁剪位置略有不同，图里的双曲线位置完全相同，但官方框每次都是分别画的、互不一致。
这些图的 band 只需要标一次：每组选一张“代表图”在 HBAT 里标，其余成员由本模块自动复制。

判定“同一张图”（只在同一个来源组内比较；同一张图的相似度必然高于分组阈值，所以一定同组）：
  1. ORB 特征 + RANSAC 估计相似变换（平移 + 等比缩放 + 旋转）；
  2. 对齐后去噪（3×3 中值滤波，去掉 Roboflow 加的椒盐噪点），取带通（高斯 σ1.5 − σ8），
     去掉四周 10% 边框，切成 64×64 小块逐块算相关系数；
  3. 小块相关系数的中位数 >= TILE_MEDIAN_MIN（0.85）即为同一张图。
     核查时同一张图的组合都在 0.90 以上，钢筋排相邻测线（版面相同、内容不同）最高 0.76，
     中间只有 line17359_0 / line2888_0-46 一对 0.81，按保守原则不算重复。

复制：顶点按变换映射；span、thickness、tail 乘缩放比；slope 在新顶点下按中心线最小二乘重新拟合
（HBAT 模型以图像顶边为地表，竖直平移后形状参数要跟着变）；随后按目标图像做与 HBAT 保存时相同的
处理（扩展画布关时收回图内、底边截 span），并按目标图自己的官方框重新分配 gt_box_id。
复制出的目标带 "source": "duplicate_copy" 和 "copied_from": <代表图文件名>。

用法（在仓库根目录，需 cv2）：
    python tools/annotation/d1_duplicates.py --find       # 重新检测，写 data/primary_roboflow/hbat/duplicates.json
    python tools/annotation/d1_duplicates.py --check      # 用两边都有旧标注的组检验复制精度
复制本身由 tools/annotation/d1_todo.py 每次运行时调用 propagate() 完成。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbat  # noqa: E402

SPLIT = 'data/splits/dataset1_roboflow.json'
HBAT_DIR = 'data/primary_roboflow/hbat'
DUP_FILE = f'{HBAT_DIR}/duplicates.json'
TILE_MEDIAN_MIN = 0.85
TILE, BORDER = 64, 0.10


# ------------------------------------------------------------------ 检测
def _load(R, it):
    return cv2.imdecode(np.fromfile(os.path.join(R, it['eval_file']), np.uint8), cv2.IMREAD_GRAYSCALE)


def _bandpass(x):
    x = cv2.medianBlur(x, 3).astype(np.float32)
    return cv2.GaussianBlur(x, (0, 0), 1.5) - cv2.GaussianBlur(x, (0, 0), 8)


def estimate_transform(src, dst):
    """src → dst 的相似变换 2×3（像素坐标），以及 RANSAC 内点数。失败返回 (None, 0)。"""
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(src, None)
    kb, db = orb.detectAndCompute(dst, None)
    if da is None or db is None:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 10:
        return None, 0
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    cv2.setRNGSeed(0)
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return None, 0
    return M, int(inl.sum())


def tile_median(src, dst, M):
    h, w = dst.shape
    A = _bandpass(cv2.warpAffine(src, M, (w, h)))
    B = _bandpass(dst)
    valid = cv2.warpAffine(np.ones(src.shape, np.float32), M, (w, h)) > 0.99
    mg = int(BORDER * w)
    ncc = []
    for y in range(mg, h - mg - TILE + 1, TILE):
        for x in range(mg, w - mg - TILE + 1, TILE):
            if not valid[y:y + TILE, x:x + TILE].all():
                continue
            p, q = A[y:y + TILE, x:x + TILE].ravel(), B[y:y + TILE, x:x + TILE].ravel()
            if p.std() < 3 or q.std() < 3:
                continue
            ncc.append(abs(float(np.corrcoef(p, q)[0, 1])))
    return (float(np.median(ncc)), len(ncc)) if len(ncc) >= 5 else (0.0, len(ncc))


def snap(M):
    """几乎是恒等变换时取精确恒等，避免复制时引入亚像素抖动。"""
    s = math.hypot(M[0, 0], M[1, 0])
    if abs(s - 1) < 0.003 and abs(M[0, 2]) < 1.5 and abs(M[1, 2]) < 1.5 and abs(M[1, 0]) < 0.003:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return M


def ann_status(objs):
    if not objs:
        return 'missing'
    if any(o.get('source') == 'legacy_init' for o in objs):
        return 'legacy_init'
    if any(o.get('source') == 'duplicate_copy' for o in objs):
        return 'copy'
    return 'manual'


def find_duplicates(R):
    with open(os.path.join(R, SPLIT), encoding='utf-8') as f:
        items = json.load(f)['items']
    with open(os.path.join(R, HBAT_DIR, 'annotations.json'), encoding='utf-8') as f:
        ann = json.load(f)
    groups = {}
    for k, it in items.items():
        groups.setdefault(it['group'], []).append(k)
    img = {}
    pairs = []
    for g, ks in sorted(groups.items()):
        ks = sorted(ks)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                for k in (a, b):
                    if k not in img:
                        img[k] = _load(R, items[k])
                M, n_inl = estimate_transform(img[a], img[b])
                if M is None:
                    continue
                med, n_tiles = tile_median(img[a], img[b], M)
                pairs.append((a, b, med, n_tiles, n_inl, M))
    # 并查集合并成簇
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b, med, *_ in pairs:
        if med >= TILE_MEDIAN_MIN:
            par[find(a)] = find(b)
    clusters = {}
    for k in par:
        clusters.setdefault(find(k), []).append(k)
    clusters = sorted((sorted(v) for v in clusters.values() if len(v) > 1), key=lambda v: v[0])

    rank = {'manual': 0, 'copy': 3, 'legacy_init': 1, 'missing': 2}
    out = []
    for ms in clusters:
        st = {k: ann_status(ann.get(os.path.basename(items[k]['eval_file']))) for k in ms}
        rep = min(ms, key=lambda k: (rank[st[k]], k))
        members = []
        for k in ms:
            if k == rep:
                continue
            M, n_inl = estimate_transform(img[rep], img[k])
            med, _ = tile_median(img[rep], img[k], M)
            M = snap(M)
            members.append({'name': k, 'file': os.path.basename(items[k]['eval_file']),
                            'transform_from_rep': [[round(float(v), 6) for v in row] for row in M],
                            'scale': round(math.hypot(M[0, 0], M[1, 0]), 4),
                            'tile_median': round(med, 3), 'orb_inliers': n_inl})
        out.append({'representative': rep, 'rep_file': os.path.basename(items[rep]['eval_file']),
                    'group': items[rep]['group'], 'members': members})
    near = sorted(((a, b, med) for a, b, med, *_ in pairs if 0.70 <= med < TILE_MEDIAN_MIN), key=lambda x: -x[2])
    doc = {'generator': 'tools/annotation/d1_duplicates.py --find',
           'criterion': f'同组内 ORB 相似变换对齐后，去噪带通图 {TILE}×{TILE} 小块相关系数中位数 >= {TILE_MEDIAN_MIN}',
           'n_pairs_compared': len(pairs),
           'near_misses': [{'a': a, 'b': b, 'tile_median': round(m, 3)} for a, b, m in near],
           'clusters': out}
    path = os.path.join(R, DUP_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    n_mem = sum(len(c['members']) for c in out)
    print(f'比较了 {len(pairs)} 对（同组内）；重复簇 {len(out)} 个，共 {n_mem + len(out)} 张，可少标 {n_mem} 张 → {DUP_FILE}')
    for c in out:
        print(f'  代表 {c["representative"]:15s} ← ' + '，'.join(
            f'{m["name"]}（{m["tile_median"]:.2f}，缩放 {m["scale"]:.3f}）' for m in c['members']))
    print('接近但未达阈值：' + ('，'.join(f'{a}/{b} {m:.2f}' for a, b, m in near) or '无'))
    return doc


# ------------------------------------------------------------------ 复制
def _centerline(xv, yv, slope, xs):
    a, b = hbat.hyperbola_ab(yv, slope)
    return yv + a * (np.sqrt(1.0 + ((xs - xv) / b) ** 2) - 1.0)


def transfer_object(obj, M, dst_size):
    """把一个 HBAT 目标按 2×3 变换 M 搬到新坐标（代表图 → 成员图；或原图 → 裁剪缩放后，见 d1_crop）。

    span（横向）乘横向缩放比，thickness、tail（竖向）乘竖向缩放比；等比变换时两者相同。"""
    M = np.asarray(M, float)[:2]
    sx = math.hypot(M[0, 0], M[1, 0])
    sy = math.hypot(M[0, 1], M[1, 1])
    o = hbat.normalize_obj(obj)
    new = dict(o)
    xv, yv = o['x_vertex'], o['y_vertex']
    new['x_vertex'] = round(float(M[0, 0] * xv + M[0, 1] * yv + M[0, 2]), 2)
    new['y_vertex'] = round(float(M[1, 0] * xv + M[1, 1] * yv + M[1, 2]), 2)
    if hbat.has_band(o):
        span = o['span']
        if o.get('reaches_bottom'):
            span = max(span, 4.0 * dst_size[0])     # 底边截过的 span 先放开，换算后再按新图重新截
        new['span'] = round(span * sx, 2)
        new['thickness'] = round(o['thickness'] * sy, 2)
        new['tail'] = round(o.get('tail', 0.0) * sy, 2)
        # 中心线采样点 → 变换 → 在新顶点下拟合 slope
        half = min(o['span'] / 2.0, 2.0 * dst_size[0])
        xs = np.linspace(xv - half, xv + half, 201)
        ys = _centerline(xv, yv, o['slope'], xs)
        P = np.stack([xs, ys, np.ones_like(xs)], 1) @ M.T
        ok = (P[:, 0] >= 0) & (P[:, 0] <= dst_size[0]) & (P[:, 1] >= 0) & (P[:, 1] <= dst_size[1])
        if ok.sum() < 10:
            ok[:] = True
        best = None
        for k in np.arange(max(0.05, o['slope'] * 0.5), o['slope'] * 2.0, 0.01):
            err = float(np.mean((_centerline(new['x_vertex'], new['y_vertex'], k, P[ok, 0]) - P[ok, 1]) ** 2))
            if best is None or err < best[0]:
                best = (err, k)
        new['slope'] = round(float(best[1]), 2)
    if not new.get('extended_canvas'):
        if 0 <= new['x_vertex'] <= dst_size[0] and 0 <= new['y_vertex'] <= dst_size[1]:
            new, _ = hbat.clamp_to_image(new, dst_size)
        else:
            # 顶点被裁到图外（例如原来压在坐标轴区域下面）：保留几何，按 HBAT 的拓展画布语义标记
            new['extended_canvas'] = True
    new = hbat.cap_span_to_bottom(new, dst_size)
    return new


def propagate(R, ann, items, gt_dir, dup_doc=None, verbose=True, crops=None):
    """代表图已人工标完（无 legacy_init / duplicate_copy）时，把它的标注复制给各成员。

    成员若已有人工标注（不是旧初值、也不是复制来的）则不覆盖，只报告冲突。返回 (复制数, 冲突列表)。
    transform_from_rep 是原图坐标之间的变换；crops（crop.json 的 crops）给出各图当前的裁剪坐标系，
    复制时先从代表图的工作坐标退回原图，再进入成员图的工作坐标。
    """
    import d1_crop
    if dup_doc is None:
        path = os.path.join(R, DUP_FILE)
        if not os.path.exists(path):
            return 0, []
        with open(path, encoding='utf-8') as f:
            dup_doc = json.load(f)
    gt = hbat.GTSource(os.path.join(R, gt_dir))
    n, conflicts = 0, []
    for c in dup_doc['clusters']:
        src = ann.get(c['rep_file']) or []
        if ann_status(src) != 'manual':
            continue
        for m in c['members']:
            cur = ann.get(m['file']) or []
            if ann_status(cur) == 'manual':
                conflicts.append((c['representative'], m['name']))
                continue
            cr, cm = (crops or {}).get(c['representative']), (crops or {}).get(m['name'])
            if cm and cm.get('applied'):
                size = tuple(d1_crop.OUTPUT_SIZE)
            else:
                with hbat.Image.open(os.path.join(R, items[m['name']]['eval_file'])) as im:
                    size = im.size
            M0 = np.vstack([np.asarray(m['transform_from_rep'], float), [0, 0, 1]])
            M = d1_crop.applied_matrix(cm) @ M0 @ np.linalg.inv(d1_crop.applied_matrix(cr))
            boxes = gt.boxes_for(m['file'], size)
            objs = []
            for o in src:
                t = transfer_object(o, M, size)
                gid, amb = hbat.assign_gt_box(t, boxes, size)
                t['gt_box_id'] = gid
                t.pop(d1_crop.REVIEW_KEY, None)
                t['source'] = 'duplicate_copy'
                t['copied_from'] = c['rep_file']
                objs.append(t)
            if json.dumps(objs, sort_keys=True) != json.dumps(cur, sort_keys=True):
                ann[m['file']] = objs
                n += 1
    if verbose and conflicts:
        print('注意：以下重复图两边都有人工标注，没有覆盖，请人工决定保留哪个：'
              + '，'.join(f'{a}/{b}' for a, b in conflicts))
    return n, conflicts


# ------------------------------------------------------------------ 检验
def check(R):
    """两边都有旧标注（同一批人画的）的成员：复制出的带 vs 成员自己的带，逐目标最佳 band IoU。"""
    with open(os.path.join(R, SPLIT), encoding='utf-8') as f:
        items = json.load(f)['items']
    with open(os.path.join(R, HBAT_DIR, 'annotations.json'), encoding='utf-8') as f:
        ann = json.load(f)
    with open(os.path.join(R, DUP_FILE), encoding='utf-8') as f:
        doc = json.load(f)
    import d1_crop
    crops = (d1_crop.load_crops(R) or {}).get('crops', {})
    for c in doc['clusters']:
        src = ann.get(c['rep_file']) or []
        for m in c['members']:
            dst = ann.get(m['file']) or []
            if not src or not dst:
                continue
            cr, cm = crops.get(c['representative']), crops.get(m['name'])
            size = tuple(d1_crop.OUTPUT_SIZE) if cm and cm.get('applied') else (640, 640)
            M0 = np.vstack([np.asarray(m['transform_from_rep'], float), [0, 0, 1]])
            M = d1_crop.applied_matrix(cm) @ M0 @ np.linalg.inv(d1_crop.applied_matrix(cr))
            moved = [transfer_object(o, M, size) for o in src]
            mk = lambda o: np.array(hbat.band_mask(o, size)) > 0
            ious = []
            for o in dst:
                if not hbat.has_band(hbat.normalize_obj(o)):
                    continue
                a = mk(o)
                ious.append(max((float((a & mk(t)).sum() / max(1, (a | mk(t)).sum()))
                                 for t in moved if hbat.has_band(t)), default=0.0))
            print(f'{c["representative"]:>13s} → {m["name"]:<14s} 带数 {len(src)} → {len(dst)}；'
                  f'逐目标最佳 band IoU：{", ".join(f"{v:.2f}" for v in ious)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='仓库根目录（默认按脚本位置推出，可在任意目录运行）')
    ap.add_argument('--find', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.find:
        find_duplicates(args.repo)
    if args.check:
        check(args.repo)
    if not (args.find or args.check):
        ap.print_help()


if __name__ == '__main__':
    main()
