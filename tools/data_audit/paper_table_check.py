"""在仓库所有结果 CSV 中查找与 paper/manuscript/2main.tex 表格数字一致的分组（mean±std 四舍五入到两位）。

用法：python tools/data_audit/paper_table_check.py <仓库根目录>
TARGETS 里的数字抄自论文；改表后同步更新这里。
"""
import os, sys, glob, collections
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
REPO = sys.argv[1]

# 目标：(P_mean, P_std, R_mean, R_std, F1_mean, F1_std)，None 表示不比
TARGETS = {
    'T2 0.5 base': (0.81, .19, .43, .09, .53, .07), 'T2 0.5 ours': (.86, .08, .49, .07, .62, .06),
    'T2 0.6 base': (.71, .20, .56, .08, .60, .10), 'T2 0.6 ours': (.75, .24, .65, .06, .66, .12),
    'T2 0.7 base': (.75, .29, .68, .10, .68, .20), 'T2 0.7 ours': (.91, .08, .78, .04, .84, .03),
    'T2 0.8 base': (.84, .21, .82, .07, .81, .15), 'T2 0.8 ours': (.91, .12, .88, .04, .89, .07),
    'T2 0.9 base': (.87, .23, .90, .06, .87, .16), 'T2 0.9 ours': (.85, .21, .93, .04, .87, .16),
    'T3 SE': (.86, .20, .82, .04, .82, .11), 'T3 CBAM': (.75, .17, .82, .03, .77, .10),
    'T3 NL': (.84, .28, .74, .23, .78, .25), 'T3 CA': (.80, .23, .84, .09, .80, .17),
    'T4 0.5 base': (.62, None, .75, None, .67, None), 'T4 0.5 ours': (.70, None, .77, None, .73, None),
    'T4 0.7 base': (.68, None, .83, None, .75, None), 'T4 0.7 ours': (.76, None, .83, None, .79, None),
    'T5 P2A base': (.27, .09, .09, .03, .13, .04), 'T5 P2A ours': (.26, .07, .08, .04, .11, .04),
    'T5 A2P base': (.12, .03, .17, .05, .13, .03), 'T5 A2P ours': (.12, .05, .14, .05, .12, .04),
}
found = collections.defaultdict(list)
for f in glob.glob(os.path.join(REPO, '**', '*.csv'), recursive=True):
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    cols = {c: c for c in df.columns}
    trip = [('bbox_P', 'bbox_R', 'bbox_F1'), ('cross_P', 'cross_R', 'cross_F1')]
    tr = next((t for t in trip if all(c in df.columns for c in t)), None)
    if tr is None or 'seed' not in df.columns:
        continue
    keys = [k for k in ('direction', 'train_frac', 'config', 'mode', 'lam_att') if k in df.columns]
    for g, sub in (df.groupby(keys) if keys else [((), df)]):
        if len(sub) < 3:
            continue
        stats = []
        for c in tr:
            x = pd.to_numeric(sub[c], errors='coerce').dropna()
            stats += [round(x.mean() + 1e-9, 2), round(x.std(ddof=0) + 1e-9, 2), round(x.std(ddof=1) + 1e-9, 2)]
        for name, t in TARGETS.items():
            ok = True
            for i in range(3):
                m, s0, s1 = stats[3 * i: 3 * i + 3]
                if abs(m - t[2 * i]) > 0.0051 or (t[2 * i + 1] is not None and min(abs(s0 - t[2 * i + 1]), abs(s1 - t[2 * i + 1])) > 0.0051):
                    ok = False
            if ok:
                found[name].append((os.path.relpath(f, REPO), g, len(sub)))
for name in TARGETS:
    print(name, '->', found.get(name) or '未找到')
