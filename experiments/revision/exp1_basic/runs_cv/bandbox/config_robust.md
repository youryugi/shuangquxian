# 同一组参数下 band 对 box-mask（测试 F1，10 seed 均值；来自 sweep.md）

| box-mask 臂 → band 臂 | 参数组数 | band 赢 | 差值中位数 | 差值 [min, max] | union / replace / blend 下 band 赢 | 验证前 20% 参数组 band 赢（中位数） | 两臂各自选中的配置 → 差值 |
|---|---|---|---|---|---|---|---|
| hbatbox_flipfix_boxmask_fitbox → hbatbox_flipfix_abs_fitbox | 180 | 145/180 | +0.048 | [-0.038, +0.180] | 60/60 / 40/60 / 45/60 | 33/36（+0.052） | 0.500 → 0.585（+0.085） |
| hbatbox_flipfix_boxmask_unet_fitbox → hbatbox_flipfix_abs_unet_fitbox | 180 | 180/180 | +0.105 | [+0.035, +0.169] | 60/60 / 60/60 / 60/60 | 36/36（+0.051） | 0.505 → 0.585（+0.080） |
| hbatbox_flipfix_boxmask_bandbox → hbatbox_flipfix_abs_bandbox | 60 | 24/60 | -0.015 | [-0.095, +0.128] | 18/20 / 3/20 / 3/20 | 6/12（+0.000） | 0.504 → 0.565（+0.061） |
| hbatbox_flipfix_boxmask_unet_bandbox → hbatbox_flipfix_abs_unet_bandbox | 60 | 26/60 | -0.007 | [-0.046, +0.071] | 20/20 / 6/20 / 0/20 | 4/12（-0.012） | 0.521 → 0.555（+0.034） |
| hbatbox_flipfix_boxmask_bandbox_nms → hbatbox_flipfix_abs_bandbox_nms | 180 | 72/180 | -0.015 | [-0.095, +0.131] | 54/60 / 9/60 / 9/60 | 18/36（-0.001） | 0.541 → 0.570（+0.029） |
| hbatbox_flipfix_boxmask_frac25_fitbox → hbatbox_flipfix_abs_frac25_fitbox | 180 | 104/180 | +0.019 | [-0.089, +0.115] | 60/60 / 16/60 / 28/60 | 28/36（+0.043） | 0.397 → 0.443（+0.046） |
| hbatbox_flipfix_boxmask_frac25_bandbox → hbatbox_flipfix_abs_frac25_bandbox | 60 | 25/60 | -0.008 | [-0.101, +0.092] | 15/20 / 4/20 / 6/20 | 4/12（-0.005） | 0.335 → 0.344（+0.009） |
| hbatbox_flipfix_boxmask_frac25_bandbox_nms → hbatbox_flipfix_abs_frac25_bandbox_nms | 180 | 72/180 | -0.009 | [-0.101, +0.101] | 42/60 / 12/60 / 18/60 | 17/36（-0.004） | 0.380 → 0.382（+0.002） |
| hbatbox_flipfix_boxmask_frac50_fitbox → hbatbox_flipfix_abs_frac50_fitbox | 180 | 107/180 | +0.019 | [-0.062, +0.156] | 60/60 / 28/60 / 19/60 | 36/36（+0.088） | 0.449 → 0.528（+0.079） |
| hbatbox_flipfix_boxmask_frac50_bandbox → hbatbox_flipfix_abs_frac50_bandbox | 60 | 19/60 | -0.037 | [-0.138, +0.141] | 15/20 / 3/20 / 1/20 | 6/12（+0.000） | 0.461 → 0.487（+0.026） |
| hbatbox_flipfix_boxmask_frac50_bandbox_nms → hbatbox_flipfix_abs_frac50_bandbox_nms | 180 | 59/180 | -0.037 | [-0.138, +0.144] | 47/60 / 9/60 / 3/60 | 21/36（+0.006） | 0.473 → 0.495（+0.022） |
| hbatbox_flipfix_boxmask_frac75_fitbox → hbatbox_flipfix_abs_frac75_fitbox | 180 | 124/180 | +0.044 | [-0.072, +0.144] | 60/60 / 28/60 / 36/60 | 36/36（+0.082） | 0.484 → 0.564（+0.080） |
| hbatbox_flipfix_boxmask_frac75_bandbox → hbatbox_flipfix_abs_frac75_bandbox | 60 | 27/60 | -0.015 | [-0.096, +0.128] | 20/20 / 3/20 / 4/20 | 11/12（+0.030） | 0.508 → 0.553（+0.045） |
| hbatbox_flipfix_boxmask_frac75_bandbox_nms → hbatbox_flipfix_abs_frac75_bandbox_nms | 180 | 81/180 | -0.015 | [-0.097, +0.133] | 60/60 / 9/60 / 12/60 | 35/36（+0.035） | 0.504 → 0.556（+0.052） |
