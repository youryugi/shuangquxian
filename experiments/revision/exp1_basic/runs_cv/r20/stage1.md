# 第 20 轮：加权 attention 损失（第一阶段）

按 seed 配对（5 折合并，验证集选阈值），B − A。

| A → B | n | F1 A | F1 B | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 [95% CI] | 胜 |
|---|---|---|---|---|---|---|---|---|
| hbatbox_flipfix_abs → hbatbox_flipfix_abs_wloss | 10 | 0.496 | 0.504 | +0.008 [-0.004, +0.020] | 0.275 | 6/0/4 | +0.018 [-0.003, +0.041] | 7/10 |
| hbatbox_flipfix_boxmask → hbatbox_flipfix_boxmask_wloss | 10 | 0.504 | 0.485 | -0.019 [-0.037, +0.001] | 0.160 | 3/0/7 | -0.005 [-0.031, +0.021] | 5/10 |
| hbatbox_flipfix_none → hbatbox_flipfix_abs_wloss | 10 | 0.472 | 0.504 | +0.032 [+0.009, +0.053] | 0.027 | 8/0/2 | +0.035 [+0.008, +0.065] | 8/10 |
| hbatbox_flipfix_boxmask_wloss → hbatbox_flipfix_abs_wloss | 10 | 0.485 | 0.504 | +0.019 [+0.002, +0.035] | 0.105 | 7/0/3 | +0.016 [-0.011, +0.042] | 6/10 |
