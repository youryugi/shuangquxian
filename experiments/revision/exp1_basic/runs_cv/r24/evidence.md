# band 在 weak / grouped 候选上的信号（测试集，10 seeds；AUC 每个 seed 算后取均值）

## W：阈值以下的检测，能不能变成 TP

| 来源 | 每 seed 正例 / 负例 | AUC score | AUC band_ev | AUC inbox | weak 正例数 | weak AUC score | weak AUC band_ev | weak AUC inbox |
|---|---|---|---|---|---|---|---|---|
| abs | 45.0 / 509.9 | 0.658 | 0.592 | 0.567 | 12.8 | 0.637 | 0.380 | 0.354 |
| boxmask | 46.7 / 531.9 | 0.709 | 0.447 | 0.557 | 12.0 | 0.649 | 0.314 | 0.396 |

## G：撞曲线的检测，真邻居 vs 重复

| 来源 | 每 seed 正例 / 负例 | AUC score | AUC neg_iou_occ | AUC refit_ev |
|---|---|---|---|---|
| abs | 6.0 / 33.1 | 0.535 | 0.573 | 0.571 |
| boxmask | 4.4 / 22.6 | 0.626 | 0.742 | 0.621 |

