# 预测 band 的质量（测试集，224×224，阈值 0.5；10 个 seed 的均值 ± std）

| exp | band IoU | precision | recall |
|---|---|---|---|
| hbatbox_aug_abs | 0.381 ± 0.018 | 0.645 | 0.488 |
| hbatbox_aug_abs_unet | 0.421 ± 0.008 | 0.676 | 0.528 |
| hbatbox_aug_boxmask | 0.202 ± 0.012 | 0.221 | 0.711 |
| hbatbox_aug_boxmask_unet | 0.208 ± 0.011 | 0.225 | 0.742 |

## 每个目标的 band 覆盖率（该目标的 band 像素里被预测到的比例）

| 分层 | 取值 | n | hbatbox_aug_abs | hbatbox_aug_abs_unet | hbatbox_aug_boxmask | hbatbox_aug_boxmask_unet |
|---|---|---|---|---|---|---|
| all | all | 216 | 0.447 | 0.489 | 0.669 | 0.701 |
| grouped | alone | 123 | 0.456 | 0.508 | 0.652 | 0.686 |
| grouped | grouped(official box has >=2 bands) | 93 | 0.436 | 0.464 | 0.692 | 0.721 |
| n_boxes | 1 target in image | 71 | 0.486 | 0.544 | 0.673 | 0.715 |
| n_boxes | >=2 targets in image | 145 | 0.429 | 0.462 | 0.668 | 0.694 |
| width | medium (53-80px) | 70 | 0.470 | 0.548 | 0.686 | 0.756 |
| width | narrow (<53px) | 70 | 0.365 | 0.391 | 0.586 | 0.616 |
| width | wide (>=80px) | 76 | 0.503 | 0.526 | 0.731 | 0.729 |
