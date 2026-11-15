"""分辨率诊断：带掩码从 224 降到注意力分辨率后的厚度、保留率、混合格比例。
用法（在 experiments/revision/ 下）：python basic_ds2/diag_resolution.py detector_comparison/datasets/ds2_f0_s100 [56|112]"""
import sys, glob, os, numpy as np, cv2
from scipy import ndimage

d = sys.argv[1]
ATT = int(sys.argv[2]) if len(sys.argv) > 2 else 56


def thick(b):
    cs, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    P = sum(cv2.arcLength(c, True) for c in cs)
    return 2 * b.sum() / max(P, 1)


rows = []
shapes = set()
for p in sorted(glob.glob(os.path.join(d, 'masks_band', '**', '*.png'), recursive=True)):
    m = cv2.imread(p, 0) > 127
    if m.sum() == 0:
        continue
    shapes.add(m.shape)
    m224 = cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    a = cv2.resize(m224.astype(np.float32), (ATT, ATT), interpolation=cv2.INTER_AREA)
    b = a > 0.5
    up = cv2.resize(b.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    kept = (m224 & up).sum() / max(m224.sum(), 1)          # recall of band pixels after 56-binarize
    extra = (up & ~m224).sum() / max(up.sum(), 1)          # fraction of 56-mask area that is off-band
    n224 = ndimage.label(m224)[1]
    nA = ndimage.label(b)[1]
    partial = ((a > 0) & (a < 1)).sum() / max((a > 0).sum(), 1)  # mixed cells (band+background)
    rows.append((thick(m224), m224.mean(), b.mean(), kept, extra, n224, nA, partial, a[a > 0].mean()))

r = np.array(rows, float)
print('n masks', len(r), 'mask shapes', shapes, 'attn res', ATT)
names = ['band_thick_224px', 'area_frac_224', f'area_frac_{ATT}', 'band_px_kept', 'offband_frac_in_lowres',
         'ncomp_224', f'ncomp_{ATT}', 'mixed_cell_frac', 'mean_soft_value']
for i, nm in enumerate(names):
    print(f'{nm:24s} median {np.median(r[:, i]):.3f}  p10 {np.percentile(r[:, i], 10):.3f}  p90 {np.percentile(r[:, i], 90):.3f}')
print(f'thickness in {ATT}-grid cells (median)', round(np.median(r[:, 0]) * ATT / 224, 2))
print('masks losing >50% band px', int((r[:, 3] < 0.5).sum()), '/', len(r))
print('masks fragmenting (more comps at low res)', int((r[:, 6] > r[:, 5]).sum()))
print('masks fully vanished at low res', int((r[:, 2] == 0).sum()))
