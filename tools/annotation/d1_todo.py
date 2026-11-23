"""数据集 1 的 HBAT 标注文件夹、进度与待标清单。

数据集 1 = Roboflow gpr-hhalx v1 官方导出包（data/primary_roboflow/original_GPR.v1i.yolov5-obb）的全部
170 个原图名。官方 train / valid / test 三份合并，划分见 data/splits/dataset1_roboflow.json，不使用官方划分。
官方 train 里每个原图名有 3 份光度副本，几何与标签完全相同：只标划分文件里指定的那一份（eval_file），
下游脚本对 3 份副本都用这份标注（experiments/revision/shared/common.py 的 _candidate_keys）。

本脚本做这些事：
0. 收拢误存的标注：HBAT 启动时没加 --out，会把 annotations.json / timelog.jsonl 写进 hbat/images/。
   这里把其中非空的标注合并进 hbat/annotations.json（以它为准），时间日志追加到 hbat/timelog.jsonl，
   原文件移到 hbat/backup/。
1. 标注文件夹：每个原图名的 eval_file 图像及其官方框放在
       data/primary_roboflow/hbat/images/      （每个原图名一张，文件名与官方导出包相同）
       data/primary_roboflow/hbat/labelTxt/    （官方框，HBAT 自动读取并显示）
   官方导出包只读、保持原样。文件名不变，所以已有标注（annotations.json 按文件名记录）直接可用。
   裁剪（d1_crop.py / crop_tool.py）：crop.json 里已确认的裁剪框在这里应用——图像裁掉坐标轴白边后
   统一缩放到 640×640，官方框和已有标注一起换算。HBAT 开着时（时间日志里最后一次会话没有结束）不应用。
2. 打印进度（按 eval_file 的标注判断）：
       missing      没有任何标注
       legacy_init  由旧标注自动转换的初值（seed_from_legacy.py），还没人工核对
       partial      已人工标，但带的数量少于官方框数（一个官方框可以套多条，所以 >= 才算标完）
       review       已人工标，但裁剪后换算的中心线偏差 > 1 px（目标带 crop_refit_rms），请在 HBAT 里看一眼，
                    调整后点更新（Update）即消除
       done         已人工标，带数 >= 官方框数
3. 待标清单 data/primary_roboflow/hbat/todo.txt，配合 hbat.py --only 只显示没标完的图。
4. 同一张图的重复截图（data/primary_roboflow/hbat/duplicates.json，由 d1_duplicates.py --find 生成）：
   每组只标代表图；成员不进清单，代表图人工标完后由本脚本自动复制到成员（"source": "duplicate_copy"）。
   请在关闭 HBAT 之后运行本脚本（HBAT 开着时会用内存中的标注覆盖文件，复制会在下次运行时补回）。

用法（在仓库根目录）：
    python tools/annotation/d1_todo.py                       # 同步文件夹、打印进度、写清单
    python tools/annotation/d1_todo.py --status missing      # 清单只放没标的
然后运行它打印出的 hbat.py 命令。每标完一批重新运行本脚本，清单会自动更新。
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_crop  # noqa: E402
import d1_duplicates  # noqa: E402

SPLIT = 'data/splits/dataset1_roboflow.json'
HBAT_DIR = 'data/primary_roboflow/hbat'
STATUSES = ('missing', 'legacy_init', 'partial', 'review', 'done', 'duplicate')


def n_official(path):
    with open(path, encoding='utf-8') as f:
        return sum(1 for ln in f if len(ln.split()) >= 8)


def item_status(objs, n_gt):
    if not objs:
        return 'missing'
    if any(o.get('source') == 'legacy_init' for o in objs):
        return 'legacy_init'
    if any(o.get(d1_crop.REVIEW_KEY) for o in objs):
        return 'review'
    return 'done' if len(objs) >= n_gt else 'partial'


def sync_workdir(R, items, crops):
    """按 crop.json 的 applied 生成 hbat/images、hbat/labelTxt 里缺的文件（未裁剪的图 = 官方文件原样复制）。"""
    copied = sum(d1_crop.sync_image(R, it, crops.get(name)) for name, it in items.items())
    img_dir = os.path.join(R, HBAT_DIR, 'images')
    want = {os.path.basename(it['eval_file']) for it in items.values()}
    extra = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png')) and f not in want)
    if extra:
        print(f'注意：{HBAT_DIR}/images 里有 {len(extra)} 个不属于数据集 1 的文件：{extra[:5]}')
    return copied


def merge_stray(R, ann):
    """HBAT 没加 --out 时写进 hbat/images/ 的文件：标注合并、日志追加、原件移到 hbat/backup/。返回合并的图数。"""
    img_dir = os.path.join(R, HBAT_DIR, 'images')
    stray = os.path.join(img_dir, 'annotations.json')
    log = os.path.join(img_dir, 'timelog.jsonl')
    if not os.path.exists(stray) and not os.path.exists(log):
        return 0
    bak = os.path.join(R, HBAT_DIR, 'backup')
    os.makedirs(bak, exist_ok=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    n = 0
    if os.path.exists(stray):
        with open(stray, encoding='utf-8') as f:
            new = json.load(f)
        for k, objs in new.items():
            if objs and objs != ann.get(k):
                if d1_duplicates.ann_status(ann.get(k)) == 'manual':
                    print(f'  {k}：主文件里已有人工标注，用 images/ 里较新的覆盖（旧的在 backup/ 的主文件备份里）')
                ann[k] = objs
                n += 1
        main_ann = os.path.join(R, HBAT_DIR, 'annotations.json')
        if os.path.exists(main_ann):
            shutil.copy2(main_ann, os.path.join(bak, f'{ts}_annotations_before_merge.json'))
        shutil.move(stray, os.path.join(bak, f'{ts}_images_annotations.json'))
    if os.path.exists(log):
        with open(log, encoding='utf-8') as f:
            lines = f.read()
        with open(os.path.join(R, HBAT_DIR, 'timelog.jsonl'), 'a', encoding='utf-8') as f:
            f.write(lines if lines.endswith('\n') or not lines else lines + '\n')
        shutil.move(log, os.path.join(bak, f'{ts}_images_timelog.jsonl'))
    rect = os.path.join(img_dir, 'annotations_rect.json')
    if os.path.exists(rect):
        shutil.move(rect, os.path.join(bak, f'{ts}_images_annotations_rect.json'))
    print(f'已把 {HBAT_DIR}/images/ 里误存的 HBAT 标注合并进主文件：{n} 张图（原件在 {HBAT_DIR}/backup/）。'
          f'以后启动 HBAT 请带 --out {HBAT_DIR}\n')
    return n


def hbat_running(R):
    """时间日志里最后一次 session_start 之后没有 session_end = HBAT 可能还开着。"""
    last = None
    for p in (os.path.join(R, HBAT_DIR, 'timelog.jsonl'), os.path.join(R, HBAT_DIR, 'images', 'timelog.jsonl')):
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8') as f:
            for ln in f:
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                if e.get('action') in ('session_start', 'session_end') and (last is None or e['ts'] >= last['ts']):
                    last = e
    return last is not None and last['action'] == 'session_start'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='仓库根目录（默认按脚本位置推出，可在任意目录运行）')
    ap.add_argument('--status', default='missing,partial,legacy_init,review',
                    help='写进清单的状态，逗号分隔（默认：除 done 以外全部）')
    args = ap.parse_args()
    R = args.repo

    def A(rel):
        """打印用的绝对路径（命令可在任意目录直接复制运行）。"""
        return os.path.abspath(os.path.join(R, rel)).replace(os.sep, '/')
    want = set(args.status.split(','))
    assert want <= set(STATUSES) - {'duplicate'}, want

    with open(os.path.join(R, SPLIT), encoding='utf-8') as f:
        sp = json.load(f)
    items = sp['items']
    keys = [os.path.basename(it['eval_file']) for it in items.values()]
    assert len(set(keys)) == len(keys) == 170, len(keys)

    ann_path = os.path.join(R, HBAT_DIR, 'annotations.json')
    ann = {}
    if os.path.exists(ann_path):
        with open(ann_path, encoding='utf-8') as f:
            ann = json.load(f)

    def save_ann():
        tmp = ann_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ann_path)

    # 0. 误存在 images/ 里的标注
    running = hbat_running(R)
    if running and not args.force:
        print('注意：时间日志显示 HBAT 可能还开着（最后一次会话没有结束）。本次不合并标注、不应用裁剪；'
              '关闭 HBAT 后再运行（确认已关闭可加 --force）。\n')
    elif merge_stray(R, ann):
        save_ann()

    # 1. 裁剪 + 标注文件夹
    doc = d1_crop.load_crops(R) or d1_crop.suggest(R, verbose=False)
    crops = doc['crops']
    bad = d1_crop.check_applied(R, items, crops)
    if bad:
        sys.exit(f'错误：{len(bad)} 张标注文件夹里的图与 crop.json 的 applied 记录对不上（例如 {bad[:5]}）。\n'
                 f'applied 可能被别的程序覆盖了（如裁剪工具开着时运行了 d1_todo）。为免标注被重复换算，本次不做任何改动；请先修复 crop.json。')
    if not running or args.force:
        n_app, _ = d1_crop.apply_checked(R, items, doc, ann)
        if n_app:
            save_ann()
            d1_crop.save_crops(R, doc)
    copied = sync_workdir(R, items, crops)
    n_chk = sum(1 for c in crops.values() if c.get('status') == 'checked')
    n_app = sum(1 for c in crops.values() if c.get('applied'))
    print(f'标注文件夹 {HBAT_DIR}/images：{len(items)} 张（本次新写 {copied} 个文件）；'
          f'裁剪框已确认 {n_chk}/{len(crops)}，其中 {n_app} 张已裁剪并缩放到 '
          f'{d1_crop.OUTPUT_SIZE[0]}×{d1_crop.OUTPUT_SIZE[1]}')
    if n_chk < len(crops):
        print(f'    核对裁剪框：python {A("tools/annotation/crop_tool.py")}（建议先核对再标注，标注后再改框也会自动换算）')
    print()

    test_fold = {g: k for k, fo in enumerate(sp['folds']) for g in fo['test']}

    # 重复截图：代表图人工标完 → 复制给成员
    dup_path = os.path.join(R, d1_duplicates.DUP_FILE)
    member_of, rep_size = {}, {}
    if os.path.exists(dup_path):
        with open(dup_path, encoding='utf-8') as f:
            dup_doc = json.load(f)
        for c in dup_doc['clusters']:
            rep_size[c['representative']] = len(c['members'])
            for m in c['members']:
                member_of[m['name']] = c['representative']
        if not running or args.force:
            n_copied, _ = d1_duplicates.propagate(R, ann, items, f'{HBAT_DIR}/labelTxt', dup_doc, crops=crops)
            if n_copied:
                save_ann()
                print(f'已把代表图的标注复制到 {n_copied} 张重复图\n')
    else:
        print(f'注意：没有 {d1_duplicates.DUP_FILE}，重复截图会被重复标注；先运行 d1_duplicates.py --find\n')

    rows = []
    for name, it in sorted(items.items()):
        key = os.path.basename(it['eval_file'])
        objs = ann.get(key) or []
        n_gt = n_official(os.path.join(R, it['labels'][it['files'].index(it['eval_file'])]))
        st = 'duplicate' if name in member_of else item_status(objs, n_gt)
        rows.append(dict(name=name, key=key, category=it['category'], fold=test_fold[it['group']],
                         n_band=len(objs), n_gt=n_gt, status=st))

    print(f'进度（标注文件 {HBAT_DIR}/annotations.json）')
    print(f'{"":14s}' + ''.join(f'{s:>13s}' for s in STATUSES) + f'{"合计":>8s}')
    for cat in ('pipe', 'rebar', None):
        sel = [r for r in rows if cat is None or r['category'] == cat]
        c = collections.Counter(r['status'] for r in sel)
        lab = {'pipe': '管线类 pipe', 'rebar': '钢筋排 rebar', None: '全部'}[cat]
        print(f'{lab:14s}' + ''.join(f'{c[s]:13d}' for s in STATUSES) + f'{len(sel):8d}')
    tgt = collections.Counter()
    for r in rows:
        tgt[r['status']] += r['n_gt']
    print('官方目标数     ' + ''.join(f'{tgt[s]:13d}' for s in STATUSES) + f'{sum(tgt.values()):8d}')
    for st in ('partial', 'review', 'done'):
        sel = [r for r in rows if r['status'] == st]
        if sel:
            print(f'{st}：' + '，'.join(f'{r["name"]}（{r["n_band"]}/{r["n_gt"]}）' for r in sel))
    if member_of:
        copied = sum(1 for r in rows if r['status'] == 'duplicate'
                     and d1_duplicates.ann_status(ann.get(r['key'])) == 'copy')
        print(f'duplicate：{len(member_of)} 张是其他图的重复截图，不用标（已复制 {copied} 张，'
              f'其余等代表图标完后自动复制）')

    sel = sorted((r for r in rows if r['status'] in want), key=lambda r: r['key'])
    todo = os.path.join(R, HBAT_DIR, 'todo.txt')
    with open(todo, 'w', encoding='utf-8') as f:
        f.write('# 由 tools/annotation/d1_todo.py 生成；数据集 1 待标的图（状态：'
                + ','.join(s for s in STATUSES if s in want) + '）\n')
        for r in sel:
            dup = f' 代表图，标完自动复制到 {rep_size[r["name"]]} 张重复图' if r['name'] in rep_size else ''
            f.write(f'{r["key"]}  # {r["name"]} {r["status"]} band {r["n_band"]}/官方框 {r["n_gt"]} '
                    f'{r["category"]} fold{r["fold"]}{dup}\n')
    print(f'\n待标清单 {HBAT_DIR}/todo.txt：{len(sel)} 张。标注命令：')
    print(f'    python "{A("tools/annotation/hbat.py")}" --folder "{A(HBAT_DIR + "/images")}" '
          f'--only "{A(HBAT_DIR + "/todo.txt")}" --out "{A(HBAT_DIR)}"')
    print(f'看全部 170 张（含已标完的）去掉 --only 即可：')
    print(f'    python "{A("tools/annotation/hbat.py")}" --folder "{A(HBAT_DIR + "/images")}" --out "{A(HBAT_DIR)}"')


if __name__ == '__main__':
    main()
