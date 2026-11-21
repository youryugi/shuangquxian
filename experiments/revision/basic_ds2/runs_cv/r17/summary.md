# 第 17 轮：参数化头（新口径：预测裁到图内）

| exp | seeds | F1 | AP50 | F1@IoU0.7 | 框中心 dx (px) | 顶点误差中位数 (px) | slope |log 比| 中位数 | span 相对误差中位数 |
|---|---|---|---|---|---|---|---|---|
| hbatbox_aug_none | 10 | 0.478 ± 0.020 | 0.365 | 0.176 | -3.38 | 6.9 | 0.108 | 0.150 |
| hbatbox_augfix_none | 10 | 0.471 ± 0.028 | 0.383 | 0.223 | -0.01 | 5.1 | 0.104 | 0.147 |
| hbatbox_flipfix_none | 10 | 0.472 ± 0.030 | 0.371 | 0.220 | 0.19 | 5.0 | 0.111 | 0.148 |
| hbatbox_augfix_pseudo | 10 | 0.545 ± 0.015 | 0.473 | 0.246 | 0.02 | 4.1 | 0.138 | 0.149 |
| hbatbox_augfix_param | 10 | 0.550 ± 0.021 | 0.468 | 0.269 | 0.11 | 3.3 | 0.126 | 0.130 |
| hbatbox_aug_abs | 10 | 0.499 ± 0.036 | 0.402 | 0.182 | -3.19 | 6.8 | 0.116 | 0.163 |
| hbatbox_aug_abs_bandbox | 10 | 0.551 ± 0.026 | 0.495 | 0.258 | -2.52 | 7.2 | 0.115 | 0.145 |
| hbatbox_aug_none_s2plain | 10 | 0.632 ± 0.021 | 0.616 | 0.412 | -0.06 | 5.2 | 0.108 | 0.104 |
| hbatbox_aug_abs_fitbox | 10 | 0.584 ± 0.029 | 0.539 | 0.299 | -2.51 | 7.2 | 0.126 | 0.151 |

| A → B | n | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 | ΔF1@0.7 |
|---|---|---|---|---|---|---|
| hbatbox_aug_none → hbatbox_augfix_none | 10 | -0.007 [-0.027, +0.013] | 0.557 | 4/0/6 | +0.018 | +0.047 |
| hbatbox_flipfix_none → hbatbox_augfix_none | 10 | -0.001 [-0.024, +0.021] | 0.922 | 6/0/4 | +0.012 | +0.003 |
| hbatbox_augfix_none → hbatbox_augfix_pseudo | 10 | +0.073 [+0.056, +0.092] | 0.002 | 10/0/0 | +0.091 | +0.023 |
| hbatbox_augfix_none → hbatbox_augfix_param | 10 | +0.079 [+0.058, +0.098] | 0.002 | 10/0/0 | +0.085 | +0.046 |
| hbatbox_augfix_pseudo → hbatbox_augfix_param | 10 | +0.006 [-0.013, +0.024] | 0.625 | 6/0/4 | -0.005 | +0.023 |
| hbatbox_aug_none_s2plain → hbatbox_augfix_none | 10 | -0.161 [-0.172, -0.147] | 0.002 | 0/0/10 | -0.233 | -0.189 |
| hbatbox_aug_none_s2plain → hbatbox_augfix_param | 10 | -0.082 [-0.100, -0.068] | 0.002 | 0/0/10 | -0.147 | -0.143 |
| hbatbox_aug_abs_bandbox → hbatbox_augfix_param | 10 | -0.001 [-0.023, +0.023] | 1.000 | 5/0/5 | -0.027 | +0.011 |
