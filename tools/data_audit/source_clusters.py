"""按图像相似度把原图聚成“独立来源”簇（同一场景改名、同一剖面重复裁剪会归为一簇）。

用法：python tools/data_audit/source_clusters.py <仓库根目录>
方法：两图缩放后，用一半幅面作模板在另一图中滑窗匹配，相关系数 > 阈值即视为同源，再做并查集合并。
用途：分组划分（同一簇只能整体进入训练或测试），回应 R2-5。
"""
import os, sys, itertools, collections, numpy as np, cv2
sys.stdout.reconfigure(encoding='utf-8')
R=sys.argv[1]
def ld(p): return cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_GRAYSCALE)
def clusters(items,size,th):
    sm={i:cv2.resize(im,(size,size)).astype(np.float32) for i,im in items}
    par={i:i for i in sm}
    def f(x):
        while par[x]!=x: x=par[x]
        return x
    for a,b in itertools.combinations(sm,2):
        A,B=sm[a],sm[b]; best=0
        for T,Sx in ((A,B),(B,A)):
            for tp in (T[:,:size//2],T[:,size//2:]):
                best=max(best,float(cv2.matchTemplate(Sx,tp,cv2.TM_CCOEFF_NORMED).max()))
        if best>th: par[f(a)]=f(b)
    c=collections.defaultdict(list)
    for i in sm: c[f(i)].append(i)
    return list(c.values())
selp=os.path.join(R,'data/primary_roboflow/selected_318_hbat')
g={}
for fn in sorted(os.listdir(selp)):
    if fn.endswith('.jpg'): g.setdefault(fn.split('_jpg.rf.')[0],fn)
aug=os.path.join(R,'data/additional_utilities/augmented_247_hbat')
used=sorted({fn.split('_aug_')[0] for fn in os.listdir(aug) if fn.endswith('.jpg')})
for th in (0.85,0.80):
    c1=clusters([(s,ld(os.path.join(selp,f))) for s,f in g.items()],128,th)
    c2=clusters([(s,ld(os.path.join(R,'data/additional_utilities/raw_131',s+'.jpg'))) for s in used],112,th)
    for nm,c,n in (('主数据集 selected 106 组',c1,len(g)),('第二数据集 已用 71 原图',c2,len(used))):
        multi=sorted([x for x in c if len(x)>1],key=len,reverse=True)
        print(f'阈值{th} {nm}: 独立簇 {len(c)}, 多成员簇 {len(multi)} 覆盖 {sum(map(len,multi))} 组;', [sorted(x) for x in multi][:12])
