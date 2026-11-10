"""按审稿意见逐项检查：现有数据能否支撑每个修改步骤。只读，不改仓库。

用法（需 cv2，conda env gpr）：
    python tools/data_audit/revision_data_check.py <仓库根目录> <Mendeley 官方 GPR_data 解压目录>
官方数据：https://data.mendeley.com/datasets/ww7fd9t325/1 （GPR_data.rar）
输出：来源与重叠、被排除样本、可见度/截断/重叠分布、分组划分模拟、标注一致性、HBAT 参数分布。
"""
import os, sys, json, re, random, collections, itertools
import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')
REPO, OFF = sys.argv[1], sys.argv[2]
P = lambda *a: os.path.join(REPO, *a)
J = lambda p: json.load(open(p, encoding='utf-8'))


def load(p, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def stem_p(f):  # 主数据集原图名
    return f.split('_jpg.rf.')[0]


def stem_a(f):  # 第二数据集原图编号
    return f.split('_aug_')[0].replace('.jpg', '')


def band_mask(h, w, o):
    span = o.get('span', o.get('width', 100.0)); hw = max(1.0, span / 2)
    height = o.get('height', o.get('slope', 1.0) * hw / 2)
    th = max(float(o['thickness']), 1.0)
    xs = np.linspace(o['x_vertex'] - hw, o['x_vertex'] + hw, max(40, int(span)) + 1)
    yc = o['y_vertex'] + height * ((xs - o['x_vertex']) / (hw + 1e-6)) ** 2
    poly = np.array(list(zip(xs, yc - th / 2)) + list(zip(xs[::-1], (yc + th / 2)[::-1])), np.float32)
    raw_out = float(np.mean((poly[:, 0] < 0) | (poly[:, 0] > w - 1) | (poly[:, 1] < 0) | (poly[:, 1] > h - 1)))
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1); poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m, raw_out


def section(t):
    print('\n' + '=' * 8, t, '=' * 8)


# ---------------------------------------------------------------- 1. 来源 / 重叠
section('1 来源与重叠 (R2-5, R2-6)')
D405 = P('data/primary_roboflow/all_405')
SELH = P('data/primary_roboflow/selected_318_hbat')
g405 = collections.defaultdict(list)
for f in sorted(os.listdir(D405)):
    if f.endswith('.jpg'):
        g405[stem_p(f)].append(f)
sel = J(os.path.join(SELH, 'annotations.json'))
gsel = collections.defaultdict(list)
for f in sel:
    gsel[stem_p(f)].append(f)
fam = lambda s: 'hyperbola*' if s.startswith('hyperbola') else ('line*' if s.startswith('line') else 'numeric')
print('主数据集组数 all/selected:', len(g405), len(gsel))
for k in ['numeric', 'hyperbola*', 'line*']:
    print(f'  {k}: all {sum(fam(s)==k for s in g405)}, selected {sum(fam(s)==k for s in gsel)}')
nums = sorted(int(s) for s in g405 if fam(s) == 'numeric')
print('  numeric 编号:', nums)
lines = collections.defaultdict(list)
for s in g405:
    m = re.match(r'line(\d+)_(\d+)-(\d+)', s)
    if m:
        lines[m.group(1)].append((int(m.group(2)), int(m.group(3)), s))
print('  line* 测线数:', len(lines), ' 每条测线的裁剪段数:', collections.Counter(len(v) for v in lines.values()))
for ln, segs in sorted(lines.items(), key=lambda kv: -len(kv[1]))[:6]:
    print('   ', ln, sorted(segs)[:8])
print('  line* 其他命名:', [s for s in g405 if fam(s) == 'line*' and not re.match(r'line(\d+)_(\d+)-(\d+)', s)][:10])


def dup_pairs(items, size):
    """items: [(id, img)]. 用左半幅模板在另一张图里滑窗匹配，检测平移重叠/近重复。"""
    small = [(i, cv2.resize(im, (size, size)).astype(np.float32)) for i, im in items]
    out = []
    for (a, A), (b, B) in itertools.combinations(small, 2):
        best = 0
        for T, S in ((A, B), (B, A)):
            for tpl in (T[:, :size // 2], T[:, size // 2:]):
                r = cv2.matchTemplate(S, tpl, cv2.TM_CCOEFF_NORMED).max()
                best = max(best, float(r))
        if best > 0.85:
            out.append((round(best, 3), a, b))
    return sorted(out, reverse=True)


reps = [(s, load(os.path.join(D405, fs[0]))) for s, fs in g405.items()]
dp = dup_pairs(reps, 128)
print('主数据集 跨组近重复/平移重叠对 (相关>0.85):', len(dp))
for x in dp[:15]:
    print('   ', x)
comp = {s: s for s in g405}
def find(x):
    while comp[x] != x:
        x = comp[x]
    return x
for _, a, b in dp:
    comp[find(a)] = find(b)
cl = collections.Counter(find(s) for s in g405)
print('  合并重叠后独立来源数:', len(cl), ' 最大簇:', cl.most_common(3))

RAW = P('data/additional_utilities/raw_131')
AUG = P('data/additional_utilities/augmented_247_hbat')
raw_items = [(f[:-4], load(os.path.join(RAW, f))) for f in sorted(os.listdir(RAW)) if f.endswith('.jpg')]
print('第二数据集 原图尺寸:', collections.Counter(im.shape for _, im in raw_items))
dp2 = dup_pairs(raw_items, 112)
print('第二数据集 原图之间近重复/平移重叠对 (相关>0.85):', len(dp2))
for x in dp2[:15]:
    print('   ', x)

# ---------------------------------------------------------------- 2. 被排除 / 弱样本
section('2 被排除的样本与弱/不完整样本来源 (R2-1, R2-10, R1-2)')
excl = sorted(set(g405) - set(gsel))
LAB = P('data/primary_roboflow/labels_dota_405')
def dota_n(f):
    p = os.path.join(LAB, f[:-4] + '.txt')
    return sum(1 for l in open(p) if len(l.split()) >= 9) if os.path.exists(p) else None
print('主数据集 被排除组:', len(excl), ' 图片', sum(len(g405[s]) for s in excl),
      ' DOTA 目标数(每组取一份):', sum(dota_n(g405[s][0]) or 0 for s in excl),
      ' 其中 DOTA 目标为 0 的组:', sum((dota_n(g405[s][0]) or 0) == 0 for s in excl))
print('  被排除组按来源:', collections.Counter(fam(s) for s in excl))
print('  selected 组的 DOTA 目标数 vs HBAT 目标数(每组取一份):',
      sum(dota_n(gsel[s][0]) or 0 for s in gsel), sum(len(sel[gsel[s][0]]) for s in gsel))
a247 = J(os.path.join(AUG, 'annotations.json'))
present = set(os.listdir(AUG))
src_used = {stem_a(f) for f in present if f.endswith('.jpg')}
raw_ann = J(os.path.join(RAW, 'annotations.json'))
print('第二数据集 已用原图编号:', len(src_used), ' 未使用原图:', 131 - len(src_used),
      ' 未使用原图的 raw_131 目标数:', sum(len(v) for k, v in raw_ann.items() if k[:-4] not in src_used))
print('  每个已用原图的增强份数:', collections.Counter(collections.Counter(stem_a(f) for f in present if f.endswith('.jpg')).values()))
OAUG = os.path.join(OFF, 'augmented_utilities')
ylab = os.path.join(OAUG, 'annotations', 'YOLO_format')
ofiles = [f for f in os.listdir(OAUG) if f.endswith('.jpg')]
yn = {f: sum(1 for l in open(os.path.join(ylab, f[:-4] + '.txt')) if l.strip()) for f in ofiles if os.path.exists(os.path.join(ylab, f[:-4] + '.txt'))}
print('官方 augmented_utilities:', len(ofiles), ' 有 YOLO 标签', len(yn), ' 目标总数', sum(yn.values()),
      ' 未进入本地的图片', len(set(ofiles) - present))
for d in ['cavities', 'augmented_cavities', 'intact']:
    print(f'  官方 {d}:', len([f for f in os.listdir(os.path.join(OFF, d)) if f.endswith(('.jpg', '.png'))]))


# ---------------------------------------------------------------- 3. 可见度 / 杂波
section('3 可见度、截断、重叠分布 (R2-2, R1-2)')
def vis_stats(img_dir, ann, names):
    rows = []
    for n in names:
        im = load(os.path.join(img_dir, n)).astype(np.float32)
        h, w = im.shape
        hp = np.abs(im - cv2.GaussianBlur(im, (0, 0), 3))
        objs = [o for o in ann[n] if o.get('label') == 'hyperbola']
        masks = [band_mask(h, w, o) for o in objs]
        for i, (o, (m, out)) in enumerate(zip(objs, masks)):
            k = max(3, int(o['thickness']) * 2 + 1)
            ring = cv2.dilate(m, np.ones((k, k), np.uint8)) - m
            if m.sum() < 10 or ring.sum() < 10:
                continue
            cnr = (hp[m > 0].mean() - hp[ring > 0].mean()) / (hp[ring > 0].std() + 1e-6)
            ov = max([float((m & mm).sum()) / m.sum() for j, (mm, _) in enumerate(masks) if j != i] or [0])
            rows.append(dict(img=n, cnr=float(cnr), trunc=out, overlap=ov, thick=o['thickness'], span=o.get('span')))
    return rows

for name, d, ann in [('主数据集 selected_318_hbat', SELH, sel), ('第二数据集 augmented_247_hbat', AUG, a247)]:
    names = [n for n in ann if os.path.exists(os.path.join(d, n))]
    rs = vis_stats(d, ann, names)
    c = np.array([r['cnr'] for r in rs])
    q = np.percentile(c, [10, 33, 50, 67, 90])
    print(f'{name}: 目标 {len(rs)}, CNR 分位 10/33/50/67/90 = {np.round(q, 2)}')
    print(f'  CNR<0.3(弱) {np.sum(c<0.3)}, 0.3-1 {np.sum((c>=0.3)&(c<1))}, >=1 {np.sum(c>=1)}')
    print(f'  带被图像边界截断>10%: {sum(r["trunc"]>0.1 for r in rs)}; 与其他带重叠>10%: {sum(r["overlap"]>0.1 for r in rs)}')
    print(f'  thickness 分位 10/50/90 = {np.round(np.percentile([r["thick"] for r in rs], [10,50,90]),1)}, '
          f'span 分位 = {np.round(np.percentile([r["span"] for r in rs], [10,50,90]),1)}')
    per_img = collections.Counter(len(ann[n]) for n in names)
    print('  每图目标数:', sorted(per_img.items()))


# ---------------------------------------------------------------- 4. 分组固定测试集
section('4 按原图分组 + 固定测试集 + 递增训练子集 (R2-5, R2-6, R2-8)')
def sim(groups, ann, seed=0, test_frac=0.2, val_frac=0.1):
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    nt, nv = round(len(keys) * test_frac), round(len(keys) * val_frac)
    te, va, tr = keys[:nt], keys[nt:nt + nv], keys[nt + nv:]
    cnt = lambda ks: (sum(len(groups[k]) for k in ks), sum(len(ann.get(f, [])) for k in ks for f in groups[k]))
    print(f'  组: test {len(te)} val {len(va)} train-pool {len(tr)}; test 图/目标 {cnt(te)}; 只算每组一份时 test 目标 '
          f'{sum(len(ann.get(groups[k][0], [])) for k in te)}')
    for frac in (0.25, 0.5, 0.75, 1.0):
        sub = tr[:max(1, round(len(tr) * frac))]
        print(f'    train {int(frac*100)}%: 组 {len(sub)}, 图/目标 {cnt(sub)}')

print('主数据集 selected_318_hbat（按原图名分组）:'); sim(gsel, sel)
gA = collections.defaultdict(list)
for f in present:
    if f.endswith('.jpg') and f in a247:
        gA[stem_a(f)].append(f)
print('第二数据集 augmented_247_hbat（按原图编号分组）:'); sim(gA, a247)
print('对照：现有随机划分下，测试集图片的原图同时出现在训练集里的比例（主数据集, seed 0-9, 7:3）:')
names = sorted(sel)
leak = []
for sd in range(10):
    idx = list(range(len(names))); random.Random(sd).shuffle(idx)
    tr = {stem_p(names[i]) for i in idx[:round(0.7 * len(idx))]}
    te = [names[i] for i in idx[round(0.7 * len(idx)):]]
    leak.append(np.mean([stem_p(n) in tr for n in te]))
print('  ', np.round(leak, 2), '(示意；脚本实际用 np.random，比例相近)')


# ---------------------------------------------------------------- 5. 标注一致性
section('5 标注一致性 (R1-4, R2-4)')
def match(A, B, scale=1.0):
    """贪心按顶点距离匹配；返回匹配数与距离、厚度差。"""
    used, ds, dt = set(), [], []
    for a in A:
        best = None
        for j, b in enumerate(B):
            if j in used:
                continue
            d = np.hypot(a['x_vertex'] - b['x_vertex'], a['y_vertex'] - b['y_vertex'])
            if best is None or d < best[0]:
                best = (d, j)
        if best and best[0] < 30 * scale:
            used.add(best[1]); ds.append(best[0]); dt.append(abs(a['thickness'] - B[best[1]]['thickness']))
    return len(ds), ds, dt

# 5a 同一原图的 3 个 Roboflow 副本（只有光度差异）是否独立标注、是否一致
same = diff = 0; dists = []
for s, fs in gsel.items():
    anns = [sel[f] for f in fs]
    for a, b in itertools.combinations(anns, 2):
        if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
            same += 1
        else:
            diff += 1
            if len(a) == len(b):
                n, ds, _ = match(a, b); dists += ds
print(f'5a 主数据集 同原图副本两两标注: 完全相同 {same} 对, 不同 {diff} 对; 不同时顶点距离中位数 '
      f'{np.median(dists) if dists else float("nan"):.1f}px (640 图)')
# 5b 旧格式 bbox 版 vs HBAT 版（同一批图，两种方案）
selb = J(P('data/primary_roboflow/selected_318_bbox/annotations.json'))
tot = (0, 0, 0); ds_all = []; dt_all = []
for n in sel:
    n_m, ds, dt = match(selb.get(n, []), sel[n])
    tot = (tot[0] + len(selb.get(n, [])), tot[1] + len(sel[n]), tot[2] + n_m); ds_all += ds; dt_all += dt
print(f'5b 主数据集 bbox 版 {tot[0]} 目标 vs HBAT 版 {tot[1]} 目标, 顶点匹配 {tot[2]}; 顶点距离中位数 {np.median(ds_all):.1f}px, '
      f'90% {np.percentile(ds_all,90):.1f}px; 厚度差中位数 {np.median(dt_all):.1f}px')
# 5c 第二数据集：raw_131 旧格式 vs aug_6（与原图几乎相同的增强）的 HBAT 标注
n_raw = n_aug = n_m = 0; ds_all = []; dt_all = []
for f in present:
    if f.endswith('_aug_6.jpg') and f in a247:
        r = raw_ann.get(stem_a(f) + '.jpg', [])
        k, ds, dt = match(r, a247[f], 224 / 640)
        n_raw += len(r); n_aug += len(a247[f]); n_m += k; ds_all += ds; dt_all += dt
print(f'5c 第二数据集 aug_6 图: raw_131 旧标注 {n_raw} 目标 vs HBAT {n_aug} 目标, 匹配 {n_m}; '
      f'顶点距离中位数 {np.median(ds_all) if ds_all else float("nan"):.1f}px (224 图)')
# 5d raw_131 三个版本
vers = {'raw_131(218)': raw_ann,
        '164': J(P('_待确认/utilities_annotation_versions/biaozhu_Utilities_164targets.json')),
        '219': J(P('_待确认/utilities_annotation_versions/biaozhumore_Utilities_219targets.json'))}
for a, b in [('raw_131(218)', '164'), ('raw_131(218)', '219')]:
    k = sum(match(vers[a].get(n, []), vers[b].get(n, []), 224 / 640)[0] for n in vers[a])
    exact = sum(vers[a].get(n) == vers[b].get(n) for n in vers[a])
    print(f'5d {a} vs {b}: 匹配目标 {k}; 整图标注完全相同 {exact}/131')
# 5e 同一原图不同增强副本之间（第二数据集，几何增强会变化，只看 aug_6 vs aug_1 噪声版：位置应不变）
ds_all = []
for s, fs in gA.items():
    f1, f6 = f'{s}_aug_1.jpg', f'{s}_aug_6.jpg'
    if f1 in a247 and f6 in a247 and f1 in present and f6 in present:
        ds_all += match(a247[f1], a247[f6], 224 / 640)[1]
print(f'5e 第二数据集 aug_1(噪声) vs aug_6(≈原图) 的 HBAT 标注顶点距离: n={len(ds_all)}, 中位数 '
      f'{np.median(ds_all) if ds_all else float("nan"):.1f}px, 90% {np.percentile(ds_all,90) if ds_all else float("nan"):.1f}px')

# ---------------------------------------------------------------- 6. 复杂双曲线参数
section('6 HBAT 参数在复杂情形下 (R1-4)')
for name, ann in [('主数据集', sel), ('第二数据集', a247)]:
    sl = np.array([o['slope'] for v in ann.values() for o in v])
    print(f'{name}: slope 取值种类 {len(set(np.round(sl,2)))}, 分位 10/50/90 = {np.round(np.percentile(sl,[10,50,90]),2)}; '
          f'字段集合 {sorted({k for v in ann.values() for o in v for k in o})}')
