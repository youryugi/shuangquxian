"""为两个数据集生成固定的分组划分文件（回应 R2-5、R2-6）。

用法（需 cv2，conda env gpr）：
    python tools/splits/make_splits.py <仓库根目录> --report          # 只打印不同阈值下的分组情况
    python tools/splits/make_splits.py <仓库根目录>                     # 用默认阈值 0.75 生成 data/splits/*.json

原则（2026-11-11 决定，见 paper/revision/revision_map.md 第 0 步）：
- 使用全部数据，不做筛选。数据集 1 = Roboflow gpr-hhalx v1 导出包中的全部 170 个原图名；
  数据集 2 = Mendeley Utilities 的全部 131 张原图。
- 划分单位是“来源组”：按图像相似度把同源图像（同一原图的副本、重叠裁剪、同名剖面系列）合成一组，
  一组只能整体进入训练、验证或测试。
- 在全部数据上一次定好划分；还没有 HBAT 标注的图也先分好组，补完标注直接使用。
- 数据集 1 不使用 Roboflow 官方 train / valid / test 划分（官方 test 仅 17 张、被个别图主导，且与 train
  有同源图像和同一测线），三份合并后重新分组；核查数字由 official_split_audit 写进 README。

划分结构：5 个分组折（按层分配、尽量均衡）。fold 0 的测试集就是“固定测试集”。
对每个折 k：test = 第 k 折；val = 其余组中约 10%（占全部）；train_pool = 剩余；
train_subsets = train_pool 的嵌套子集（25/50/75/100%，小子集一定包含在大子集里）。
"""
import os, re, sys, json, argparse, itertools, collections, random, datetime
import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8')

D1_EXPORT = 'data/primary_roboflow/original_GPR.v1i.yolov5-obb'
D1_LEGACY = 'data/primary_roboflow/selected_318_hbat'
D2_IMG = 'data/additional_utilities/original/original_Utilities'  # 官方原图，原样保存（2026-11-14 移入 original/）
D2_RAW = 'data/additional_utilities/legacy/raw_131'  # 只读其中自行标注的 annotations.json（旧格式，用于分层）
D2_AUG = 'data/additional_utilities/legacy/augmented_247_hbat'

# 数据集 1 中的钢筋排（混凝土结构扫描）图像。2026-11-11 目检得出，待用户逐张确认。
# 只用于分层和按类别报告，不用于排除。
D1_REBAR = {
    '3170', '3172', '3174', '3176', '3178', '3180', '3182', '3184', '3186', '3188',
    '3218', '3220', '3222', '3224', '3226', '3228', '3230',
    '3241', '3245', '3254', '3262', '3263', '3264', '3269', '3279',
}

SPLIT_SEED = 20261111
N_FOLDS = 5
VAL_FRAC = 0.10            # 占全部组的比例
SUBSET_FRACS = (0.25, 0.50, 0.75, 1.00)


def load_gray(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)


def pair_similarity(small):
    """两图互相用左/右半幅作模板滑窗匹配，取最大归一化相关系数（与 tools/data_audit/source_clusters.py 相同）。"""
    keys = sorted(small)
    size = next(iter(small.values())).shape[0]
    S = {}
    for a, b in itertools.combinations(keys, 2):
        A, B = small[a], small[b]
        best = 0.0
        for T, X in ((A, B), (B, A)):
            for tp in (T[:, :size // 2], T[:, size // 2:]):
                best = max(best, float(cv2.matchTemplate(X, tp, cv2.TM_CCOEFF_NORMED).max()))
        S[(a, b)] = best
    return S


def components(keys, edges):
    par = {k: k for k in keys}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    comp = collections.defaultdict(list)
    for k in keys:
        comp[find(k)].append(k)
    return sorted((sorted(v) for v in comp.values()), key=lambda v: v[0])


# ------------------------------------------------------------------ 数据集读取
def d1_items(R):
    items = {}
    for split in ('train', 'valid', 'test'):
        d = os.path.join(R, D1_EXPORT, split, 'images')
        for f in sorted(os.listdir(d)):
            if not f.endswith('.jpg'):
                continue
            stem = f.split('_jpg.rf.')[0]
            it = items.setdefault(stem, {'files': [], 'labels': [], 'roboflow_split': split})
            it['files'].append(f'{D1_EXPORT}/{split}/images/{f}')
            it['labels'].append(f'{D1_EXPORT}/{split}/labelTxt/{f[:-4]}.txt')
    legacy = collections.defaultdict(list)
    for f in sorted(os.listdir(os.path.join(R, D1_LEGACY))):
        if f.endswith('.jpg'):
            legacy[f.split('_jpg.rf.')[0]].append(f'{D1_LEGACY}/{f}')
    for stem, it in items.items():
        it['eval_file'] = it['files'][0]
        with open(os.path.join(R, it['labels'][0]), encoding='utf-8') as fh:
            it['n_targets_roboflow'] = sum(1 for l in fh if l.strip())
        it['family'] = 'numeric' if stem[0].isdigit() else ('hyperbolaN' if stem.startswith('hyperbola') else 'line')
        it['category'] = 'rebar' if stem in D1_REBAR else 'pipe'
        it['in_original_paper'] = stem in legacy
        it['legacy_hbat_files'] = legacy.get(stem, [])
        it['stratum'] = 'rebar' if it['category'] == 'rebar' else it['family']
    return items


def d1_name_series(stem):
    """同一剖面的系列裁剪按文件名强制合组，例如 line2888_0-21 与 line2888_0-95。"""
    m = re.match(r'(line\d+)_\d+-\d+$', stem)
    return m.group(1) if m else None


def d2_items(R):
    with open(os.path.join(R, D2_RAW, 'annotations.json'), encoding='utf-8') as fh:
        ann = json.load(fh)
    aug = collections.defaultdict(list)
    for f in sorted(os.listdir(os.path.join(R, D2_AUG))):
        if f.endswith('.jpg'):
            aug[f.split('_aug_')[0]].append(f'{D2_AUG}/{f}')
    items = {}
    for f in sorted(ann):
        k = f[:-4]
        n = len(ann[f])
        items[k] = {
            'files': [f'{D2_IMG}/{f}'],
            'eval_file': f'{D2_IMG}/{f}',
            'n_targets_legacy': n,
            'in_original_paper': k in aug,
            'legacy_hbat_files': aug.get(k, []),
            # 分层：原论文用过的 / 没用过的（后者多为截断、弱、重叠目标）；无目标图单独一层
            'stratum': 'no_target' if n == 0 else ('used' if k in aug else 'unused'),
        }
    return items


def small_images(R, items, size):
    return {k: cv2.resize(load_gray(os.path.join(R, it['eval_file'])), (size, size)).astype(np.float32)
            for k, it in items.items()}


# ------------------------------------------------------------------ 分组与划分
def build_groups(items, S, th, name_series=None):
    edges = [(a, b) for (a, b), v in S.items() if v > th]
    if name_series:
        series = collections.defaultdict(list)
        for k in items:
            s = name_series(k)
            if s:
                series[s].append(k)
        for ks in series.values():
            edges += [(ks[0], k) for k in ks[1:]]
    return components(sorted(items), edges)


def group_stratum(members, items):
    c = collections.Counter(items[m]['stratum'] for m in members)
    return sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]


def make_folds(groups, items, n_folds, seed):
    """按层分配：每层内组按大小降序，依次放进当前该层最少（其次总量最少）的折。"""
    rng = random.Random(seed)
    by_stratum = collections.defaultdict(list)
    for gi, g in enumerate(groups):
        by_stratum[group_stratum(g, items)].append(gi)
    fold_of = {}
    tot = [0] * n_folds
    for st in sorted(by_stratum):
        gis = by_stratum[st]
        rng.shuffle(gis)
        gis.sort(key=lambda gi: -len(groups[gi]))
        cnt = [0] * n_folds
        for gi in gis:
            f = min(range(n_folds), key=lambda j: (cnt[j], tot[j], j))
            fold_of[gi] = f
            cnt[f] += len(groups[gi])
            tot[f] += len(groups[gi])
    return fold_of


def interleave_by_stratum(gis, groups, items, rng):
    """把一批组按层轮流排开，使任意前缀的层比例都接近整体比例（用于 val 抽取和嵌套子集）。"""
    by = collections.defaultdict(list)
    for gi in gis:
        by[group_stratum(groups[gi], items)].append(gi)
    for st in by:
        rng.shuffle(by[st])
    n = len(gis)
    pos = []
    for st in sorted(by):
        m = len(by[st])
        pos += [((j + 0.5) / m, st, gi) for j, gi in enumerate(by[st])]
    pos.sort(key=lambda x: (x[0], x[1]))
    assert len(pos) == n
    return [gi for _, _, gi in pos]


def build_dataset_split(name, items, groups, seed):
    fold_of = make_folds(groups, items, N_FOLDS, seed)
    gid = lambda gi: f'g{gi:03d}'
    folds = []
    # 验证集按图像项数取约 VAL_FRAC，放不下的大组跳过（避免整个大组落进验证集、挤占训练池）
    val_budget = round(VAL_FRAC * len(items))
    for k in range(N_FOLDS):
        rng = random.Random(seed * 100 + k)
        test = [gi for gi in range(len(groups)) if fold_of[gi] == k]
        rest = interleave_by_stratum([gi for gi in range(len(groups)) if fold_of[gi] != k], groups, items, rng)
        val, used = [], 0
        for gi in rest:
            if used + len(groups[gi]) <= val_budget:
                val.append(gi)
                used += len(groups[gi])
            if used >= val_budget:
                break
        pool = interleave_by_stratum([gi for gi in rest if gi not in val], groups, items, rng)
        subsets = {f'{int(p * 100)}': [gid(gi) for gi in pool[:max(1, round(p * len(pool)))]] for p in SUBSET_FRACS}
        folds.append({'test': [gid(gi) for gi in test], 'val': [gid(gi) for gi in val],
                      'train_pool': [gid(gi) for gi in pool], 'train_subsets': subsets})
    groups_out = {gid(gi): g for gi, g in enumerate(groups)}
    item_group = {m: gid(gi) for gi, g in enumerate(groups) for m in g}
    for k, it in items.items():
        it['group'] = item_group[k]
    return {'folds': folds, 'groups': groups_out}


def check_no_leak(split, S, th):
    """校验：任一折中，测试组与训练/验证组之间没有相似度超过阈值的图像对。"""
    g_of = {m: g for g, ms in split['groups'].items() for m in ms}
    worst = 0.0
    for f in split['folds']:
        te = set(f['test'])
        other = set(f['val']) | set(f['train_pool'])
        assert not te & other
        for (a, b), v in S.items():
            ga, gb = g_of[a], g_of[b]
            if (ga in te and gb in other) or (gb in te and ga in other):
                worst = max(worst, v)
    assert worst <= th, worst
    return worst


def summarize(name, items, split, target_key):
    lines = [f'### {name}', '',
             f'- 项目数 {len(items)}，来源组 {len(split["groups"])}，'
             f'最大组 {max(len(v) for v in split["groups"].values())} 个',
             '', '| 折 | 测试组 | 测试项 | 测试目标 | 验证组 / 项 | 训练池组 | 训练池项 | 测试集各层（项） |', '|---|---|---|---|---|---|---|---|']
    for k, f in enumerate(split['folds']):
        def members(gs):
            return [m for g in gs for m in split['groups'][g]]
        te, va, tp = members(f['test']), members(f['val']), members(f['train_pool'])
        strata = collections.Counter(items[m]['stratum'] for m in te)
        lines.append(f'| {k}{"（固定测试集）" if k == 0 else ""} | {len(f["test"])} | {len(te)} | '
                     f'{sum(items[m][target_key] for m in te)} | {len(f["val"])} / {len(va)} | {len(f["train_pool"])} | {len(tp)} | '
                     + '，'.join(f'{s} {n}' for s, n in sorted(strata.items())) + ' |')
    f0 = split['folds'][0]
    lines += ['', 'fold 0 嵌套训练子集（组 / 项）：' + '，'.join(
        f'{p}% {len(gs)} / {sum(len(split["groups"][g]) for g in gs)}' for p, gs in f0['train_subsets'].items()), '']
    return '\n'.join(lines)


def official_split_audit(items, S, th, name_series):
    """数据集 1：Roboflow 官方 train / valid / test 划分的核查，说明为什么不用它（写进 README）。"""
    sims = collections.defaultdict(float)
    for (a, b), v in S.items():
        sims[(a, b)] = sims[(b, a)] = v
    by = collections.defaultdict(list)
    for k, it in items.items():
        by[it['roboflow_split']].append(k)
    lines = ['#### 为什么不用 Roboflow 官方划分', '',
             '官方导出包自带 train / valid / test 三份。本项目把三份合并成 170 个原图名，按上面的来源组重新做 5 折，'
             '不使用官方划分。理由（由本脚本从数据直接算出）：', '',
             '| 官方划分 | 原图名 | 图像文件 | 官方目标 | 目标最多的一张图 | 与 train 同源的原图名（相似度 > 阈值 / > 0.85） |',
             '|---|---|---|---|---|---|']
    for sp in ('train', 'valid', 'test'):
        ks = sorted(by[sp])
        n_t = {k: items[k]['n_targets_roboflow'] for k in ks}
        tot = sum(n_t.values())
        top = max(ks, key=lambda k: n_t[k])
        if sp == 'train':
            leak = '—'
        else:
            best = {k: max(sims[(k, t)] for t in by['train']) for k in ks}
            hi = sorted(k for k in ks if best[k] > th)
            leak = (f'{len(hi)} / {sum(best[k] > 0.85 for k in ks)}：' + '、'.join(hi)) if hi else '0'
        lines.append(f'| {sp} | {len(ks)} | {sum(len(items[k]["files"]) for k in ks)} | {tot} | '
                     f'{top}：{n_t[top]} 个（{n_t[top] / tot:.0%}） | {leak} |')
    cross = sorted((a, b, sims[(a, b)]) for a in by['valid'] for b in by['test'] if sims[(a, b)] > th)
    series = collections.defaultdict(collections.Counter)
    for k, it in items.items():
        s = name_series(k)
        if s:
            series[s][it['roboflow_split']] += 1
    spread = {s: c for s, c in series.items() if len(c) > 1}
    rebar = {sp: sum(items[k]['n_targets_roboflow'] for k in by[sp] if items[k]['category'] == 'rebar')
             for sp in ('valid', 'test')}
    lines += ['',
              f'- valid 与 test 之间同源（相似度 > {th}）：'
              + ('，'.join(f'{a} ↔ {b}（{v:.2f}）' for a, b, v in cross) if cross else '无') + '。',
              '- 同一测线的系列裁剪被分进了不同的官方划分：'
              + ('；'.join(f'{s}（' + '，'.join(f'{sp} {c[sp]}' for sp in ('train', 'valid', 'test') if c[sp]) + '）'
                          for s, c in sorted(spread.items())) if spread else '无') + '。',
              f'- valid / test 的官方目标中，钢筋排分别占 {rebar["valid"]} / {rebar["test"]} 个。',
              '- 官方 train 的每个原图名有 3 份光度副本，它们都留在 train 内，没有跨进 valid / test（这一点官方划分是对的）。',
              '- 结论：官方 test 太小，结果被个别图主导，而且与训练集有同源图像和同一测线，不能作为正式评估。'
              '图像与标签一张不删，只是重新划分；分组规则在看任何结果之前定好。', '']
    return '\n'.join(lines)


def test_fold_of(split):
    return {g: k for k, f in enumerate(split['folds']) for g in f['test']}


def write_group_table(path, name, items, split, target_key):
    """可读的分组表：每组一行，按测试折、组号排序。"""
    tf = test_fold_of(split)
    lines = [f'# {name}：来源组一览', '',
             '由 `tools/splits/make_splits.py` 生成，不要手工修改。“测试折”= 该组在 5 折交叉验证中作为测试集的那一折；'
             'fold 0 即固定测试集。', '',
             '| 组 | 测试折 | 层 | 项数 | 目标数 | 成员 |', '|---|---|---|---|---|---|']
    for g in sorted(split['groups'], key=lambda g: (tf[g], g)):
        ms = split['groups'][g]
        lines.append(f'| {g} | {tf[g]} | {group_stratum(ms, items)} | {len(ms)} | '
                     f'{sum(items[m][target_key] for m in ms)} | {", ".join(ms)} |')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


def write_gallery(path, datasets):
    """缩略图网页：按数据集、测试折分节，每组一行缩略图。图片用相对路径引用仓库内文件，需在本机浏览器打开。"""
    css = ('body{font:14px system-ui,"Microsoft YaHei",sans-serif;margin:16px;background:#f6f7f8;color:#1c2329}'
           'h1{font-size:22px}h2{font-size:18px;margin-top:28px}h3{font-size:15px;margin:18px 0 6px}'
           '.g{display:flex;gap:10px;align-items:flex-start;background:#fff;border:1px solid #d9dee2;border-radius:6px;'
           'padding:6px 8px;margin:6px 0}.g.multi{border-color:#c9822b;background:#fff8ef}'
           '.meta{width:150px;flex:none;font-size:12px;color:#55606a}.meta b{color:#1c2329;font-size:14px}'
           '.ims{display:flex;flex-wrap:wrap;gap:6px}figure{margin:0;width:120px}figure img{width:120px;height:120px;'
           'object-fit:cover;display:block;border:1px solid #ccc}figcaption{font-size:11px;word-break:break-all}'
           'nav a{margin-right:12px}')
    body = ['<h1>来源组一览</h1>',
            '<p>由 <code>tools/splits/make_splits.py</code> 生成。每行一个来源组；橙色底 = 含 2 张以上图像的组。'
            '按“测试折”分节：fold 0 即固定测试集。</p>', '<nav>']
    body += [f'<a href="#{tag}">{name}</a>' for tag, name, _, _, _ in datasets] + ['</nav>']
    for tag, name, items, split, tkey in datasets:
        tf = test_fold_of(split)
        body.append(f'<h2 id="{tag}">{name}（{len(items)} 项，{len(split["groups"])} 组）</h2>')
        for k in range(len(split['folds'])):
            gs = sorted((g for g in split['groups'] if tf[g] == k), key=lambda g: (-len(split['groups'][g]), g))
            body.append(f'<h3>fold {k}{"（固定测试集）" if k == 0 else ""}：{len(gs)} 组，'
                        f'{sum(len(split["groups"][g]) for g in gs)} 项</h3>')
            for g in gs:
                ms = split['groups'][g]
                figs = ''.join(f'<figure><img loading="lazy" src="../../{items[m]["eval_file"]}" alt="{m}">'
                               f'<figcaption>{m}</figcaption></figure>' for m in ms)
                body.append(f'<div class="g{" multi" if len(ms) > 1 else ""}"><div class="meta"><b>{g}</b><br>'
                            f'{group_stratum(ms, items)} · {len(ms)} 项 · {sum(items[m][tkey] for m in ms)} 目标</div>'
                            f'<div class="ims">{figs}</div></div>')
    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>来源组一览</title>'
            f'<style>{css}</style></head><body>' + '\n'.join(body) + '</body></html>')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('--report', action='store_true', help='只打印不同阈值下的分组情况，不写文件')
    # 0.75：高于两个数据集全部图像对相似度的 99% 分位（数据集 1 为 0.726，数据集 2 为 0.65），
    # 能并上同一剖面相邻裁剪（0.75–0.80 的那些对），规则统一且偏保守。
    ap.add_argument('--th1', type=float, default=0.75, help='数据集 1 的相似度阈值')
    ap.add_argument('--th2', type=float, default=0.75, help='数据集 2 的相似度阈值')
    args = ap.parse_args()
    R = args.repo

    it1, it2 = d1_items(R), d2_items(R)
    print(f'数据集 1：{len(it1)} 个原图名；数据集 2：{len(it2)} 张原图。计算相似度……', flush=True)
    S1 = pair_similarity(small_images(R, it1, 128))
    S2 = pair_similarity(small_images(R, it2, 112))

    if args.report:
        for name, items, S, ns in (('数据集 1', it1, S1, d1_name_series), ('数据集 2', it2, S2, None)):
            v = np.array(list(S.values()))
            print(f'\n{name} 图像对相似度分位 50/90/99/99.9 = {np.round(np.percentile(v, [50, 90, 99, 99.9]), 3)}')
            for th in (0.65, 0.70, 0.75, 0.80, 0.85):
                g = build_groups(items, S, th, ns)
                sizes = sorted((len(x) for x in g), reverse=True)
                print(f'  阈值 {th:.2f}: 组数 {len(g):3d}，最大的 5 组 {sizes[:5]}，'
                      f'最大组前几项 {max(g, key=len)[:6]}')
        return

    out_dir = os.path.join(R, 'data', 'splits')
    os.makedirs(out_dir, exist_ok=True)
    md = ['# 分组划分文件', '',
          f'由 `tools/splits/make_splits.py` 生成（{datetime.date.today().isoformat()}），不要手工修改。', '',
          f'- 划分随机种子 {SPLIT_SEED}；{N_FOLDS} 个分组折，fold 0 的测试集为固定测试集；'
          f'验证集约占 {int(VAL_FRAC * 100)}%；嵌套训练子集 {", ".join(f"{int(p * 100)}%" for p in SUBSET_FRACS)}。',
          f'- 相似度阈值：数据集 1 {args.th1}，数据集 2 {args.th2}。数据集 1 另把 `lineNNNN_0-K` 同名剖面系列强制合组。',
          '- 数据集 1 的 `category`（pipe / rebar）来自 2026-11-11 目检，待逐张确认；只用于分层和按类别报告，不用于排除。',
          '- 数据集 1 不使用 Roboflow 官方 train / valid / test 划分（`items.*.roboflow_split` 只作记录），'
          '理由见下文“为什么不用 Roboflow 官方划分”。',
          '- 数据集 1 每个原图名的 3 份光度副本同属一组；训练用全部副本，验证 / 测试只评 `eval_file` 那一份；'
          'HBAT 也只标 `eval_file` 那一份（进度与待标清单：`tools/annotation/d1_todo.py`）。',
          '- 查看分组：`dataset1_groups.md`、`dataset2_groups.md`（每组一行）；`groups_gallery.html`（缩略图，在本机浏览器打开）。', '']
    gallery = []
    for fname, name, items, S, th, ns, tkey in (
            ('dataset1_roboflow.json', '数据集 1（Roboflow gpr-hhalx v1）', it1, S1, args.th1, d1_name_series, 'n_targets_roboflow'),
            ('dataset2_mendeley.json', '数据集 2（Mendeley Utilities 原图）', it2, S2, args.th2, None, 'n_targets_legacy')):
        groups = build_groups(items, S, th, ns)
        split = build_dataset_split(name, items, groups, SPLIT_SEED)
        worst = check_no_leak(split, S, th)
        doc = {
            'dataset': name,
            'generator': 'tools/splits/make_splits.py',   # 不写生成日期，保证同样输入每次生成的文件逐字节相同
            'params': {'similarity_threshold': th, 'split_seed': SPLIT_SEED, 'n_folds': N_FOLDS,
                       'val_frac': VAL_FRAC, 'subset_fracs': list(SUBSET_FRACS), 'fixed_test_fold': 0},
            'max_cross_split_similarity': round(worst, 4),
            'items': items,
            **split,
        }
        with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
        s = summarize(name, items, split, tkey)
        md += [s, f'测试组与训练/验证组之间的最大图像相似度：{worst:.3f}（阈值 {th}）', '']
        if ns is d1_name_series:
            md.append(official_split_audit(items, S, th, ns))
        print(s)
        write_group_table(os.path.join(out_dir, fname.split('_')[0] + '_groups.md'), name, items, split, tkey)
        gallery.append((fname.split('_')[0], name, items, split, tkey))
    write_gallery(os.path.join(out_dir, 'groups_gallery.html'), gallery)
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md))
    print('已写入', out_dir)


if __name__ == '__main__':
    main()
