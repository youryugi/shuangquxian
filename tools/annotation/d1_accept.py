"""把"在 HBAT 里看过、没改"的初值记为人工接受。

用户的核对方式：在 HBAT 里逐张看，看着不对的目标拖改后点 Update（HBAT 重建目标，source 字段消失），
看着没问题的不动。所以"--since 之后打开过（timelog 有 image_enter）且仍未改的目标"= 人工接受。

  source == 'legacy_init'   →  source = 'legacy_accepted'，accepted_at = 该图最后一次打开的时间
  crop_refit_rms（裁剪换算偏差 > 1 px 的复查标记）→ 改名 crop_refit_rms_accepted（保留数值）

--since 必须不早于最后一次应用裁剪的时间：之前看到的是裁剪前的坐标，不能算核对过换算后的结果。
重复截图的成员不处理（它们由代表图自动复制覆盖）。

用法：python tools/annotation/d1_accept.py --since 2026-11-23T15:10:40 [--dry-run]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_crop  # noqa: E402
import d1_duplicates  # noqa: E402
import d1_todo  # noqa: E402

HBAT_DIR = d1_crop.HBAT_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='仓库根目录（默认按脚本位置推出，可在任意目录运行）')
    ap.add_argument('--since', required=True, help='只认这个时间之后的查看记录，ISO 格式，如 2026-11-23T15:10:40')
    ap.add_argument('--dry-run', action='store_true', help='只打印，不写文件')
    args = ap.parse_args()
    R = args.repo

    if d1_todo.hbat_running(R):
        sys.exit('时间日志显示 HBAT 可能还开着；关闭后再运行。')

    items = d1_crop.load_split(R)['items']
    dup_path = os.path.join(R, d1_duplicates.DUP_FILE)
    members = set()
    if os.path.exists(dup_path):
        with open(dup_path, encoding='utf-8') as f:
            for cl in json.load(f)['clusters']:
                members |= {m['name'] for m in cl['members']}

    last_view = {}
    with open(os.path.join(R, HBAT_DIR, 'timelog.jsonl'), encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            if e.get('action') == 'image_enter' and e['ts'] >= args.since:
                last_view[e['image']] = max(e['ts'], last_view.get(e['image'], ''))

    ann_path = os.path.join(R, HBAT_DIR, 'annotations.json')
    with open(ann_path, encoding='utf-8') as f:
        ann = json.load(f)

    n_leg = n_rev = 0
    touched, not_viewed = [], []
    for name, it in sorted(items.items()):
        if name in members:
            continue
        key = os.path.basename(it['eval_file'])
        objs = ann.get(key) or []
        pending = [o for o in objs if o.get('source') == 'legacy_init' or d1_crop.REVIEW_KEY in o]
        if not pending:
            continue
        if key not in last_view:
            not_viewed.append(name)
            continue
        for o in pending:
            if o.get('source') == 'legacy_init':
                o['source'] = 'legacy_accepted'
                o['accepted_at'] = last_view[key]
                n_leg += 1
            if d1_crop.REVIEW_KEY in o:
                o[d1_crop.REVIEW_KEY + '_accepted'] = o.pop(d1_crop.REVIEW_KEY)
                o['accepted_at'] = last_view[key]
                n_rev += 1
        touched.append(name)

    print(f'接受：{len(touched)} 张图，初值目标 {n_leg} 个，裁剪复查目标 {n_rev} 个')
    if not_viewed:
        print(f'未处理（{args.since} 之后没打开过）：{not_viewed}')
    if args.dry_run:
        print('（--dry-run，未写文件）')
        return
    tmp = ann_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ann_path)


if __name__ == '__main__':
    main()
