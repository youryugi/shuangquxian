# 第 20 轮：形状外推 / 曲线归属 / 加权损失（fit 定框）

按 seed 配对（5 折合并，验证集选阈值），B − A。

| A → B | n | F1 A | F1 B | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 [95% CI] | 胜 |
|---|---|---|---|---|---|---|---|---|
| hbatbox_flipfix_abs_fitbox → hbatbox_flipfix_abs_fitext | 10 | 0.585 | 0.585 | +0.000 [+0.000, +0.000] | 1.000 | 0/10/0 | +0.000 [+0.000, +0.000] | 0/10 |
| hbatbox_flipfix_abs_fitbox → hbatbox_flipfix_abs_fitmirror | 10 | 0.585 | 0.583 | -0.001 [-0.010, +0.007] | 0.846 | 5/0/5 | +0.002 [-0.007, +0.011] | 7/10 |
| hbatbox_flipfix_abs_fitbox → hbatbox_flipfix_abs_fittrack | 10 | 0.585 | 0.588 | +0.003 [-0.006, +0.013] | 0.846 | 5/0/5 | -0.002 [-0.013, +0.008] | 4/10 |
| hbatbox_flipfix_boxmask_fitbox → hbatbox_flipfix_boxmask_fitmirror | 10 | 0.499 | 0.478 | -0.021 [-0.034, -0.008] | 0.020 | 1/0/9 | -0.021 [-0.037, -0.006] | 3/10 |
| hbatbox_flipfix_abs_fitbox → hbatbox_flipfix_abs_wloss_fitbox | 10 | 0.585 | 0.585 | +0.001 [-0.022, +0.024] | 0.922 | 4/0/6 | -0.006 [-0.037, +0.026] | 4/10 |
| hbatbox_flipfix_boxmask_fitbox → hbatbox_flipfix_boxmask_wloss_fitbox | 10 | 0.499 | 0.492 | -0.007 [-0.023, +0.010] | 0.432 | 3/0/7 | -0.032 [-0.051, -0.009] | 1/10 |
| hbatbox_flipfix_none → hbatbox_flipfix_abs_wloss_fitbox | 10 | 0.472 | 0.585 | +0.113 [+0.092, +0.136] | 0.002 | 10/0/0 | +0.164 [+0.144, +0.188] | 10/10 |
| hbatbox_flipfix_boxmask_wloss_fitbox → hbatbox_flipfix_abs_wloss_fitbox | 10 | 0.492 | 0.585 | +0.094 [+0.083, +0.105] | 0.002 | 10/0/0 | +0.156 [+0.136, +0.173] | 10/10 |
| hbatbox_flipfix_boxmask_wloss_bandbox_nms → hbatbox_flipfix_abs_wloss_fitbox | 10 | 0.524 | 0.585 | +0.062 [+0.044, +0.079] | 0.002 | 10/0/0 | +0.069 [+0.052, +0.086] | 10/10 |
| hbatbox_flipfix_abs_bandbox_nms → hbatbox_flipfix_abs_wloss_bandbox_nms | 10 | 0.570 | 0.567 | -0.003 [-0.022, +0.013] | 0.846 | 7/0/3 | +0.004 [-0.023, +0.030] | 5/10 |
