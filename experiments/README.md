# experiments/ 导航

两个阶段，各有自己的 README：

| 阶段 | 目录 | 状态 | 看什么 |
|---|---|---|---|
| **投稿版**（JSTARS-2026-06074） | [submission/](submission/README.md) | 代码冻结，只读 | 论文每张表对应哪个脚本、哪次运行；已知问题 |
| **大修阶段**（2026-11 起） | [revision/](revision/README.md) | 进行中 | 审稿意见 → 实验编号 → 状态 → 当前结论 |

## 每个实验目录的统一约定

```
<experiment>/
  README.md     文档：开头一段给答案 + 一页总览表，然后按实验轮次（时间顺序）写"做了什么 → 为什么 → 结果 → 学到什么"，最后是下一步和附录（设置、文件位置）
  *.py          代码：训练 / 评估 / 诊断脚本
  runs*/        结果：每个 run 一个目录，含 run_info.json（全部参数）、metrics.json、预测 JSON、日志
```

- 正式数字只认 README 里标为"正式结果"的那一节；历史记录里的数字标了"已取代"的不要引用。
- 结果目录里的 `*.pth / *.pt / *.png / *.jpg / *.npz` 不入库（.gitignore）；重跑脚本可以再生成。
- 运行环境：`C:/Users/79152/.conda/envs/gpr/python.exe`（torch 2.10 + cu128，ultralytics 8.4.70）。

## 当前最重要的结论（2026-11-15）

详见 [revision/basic_ds2/README.md](revision/basic_ds2/README.md)（开头答案 + 一页总览）。

- 数据集 2 新标注、干净协议（配对初始化 + 5 折 + 10 seed）下，**注意力监督对检测 F1 没有可测出的增益**：ΔF1 = +0.004 [−0.021, +0.028]。论文 Table IV 的 0.67 → 0.73 不能复现。
- 注意力分支本身正常（学到了带、检测头在用），瓶颈在召回：约 4 成漏检是"框画小了"。用预测的带只扩不缩地修正框，同一批模型 F1 0.479 → 0.541（10 seed 全胜）——说明带的信息有用但没进入框回归。这只是输出机制，不进论文主张。
- **最高优先级待办**：数据集 1 用同一套干净协议复跑基础实验，决定论文主张怎么写。

## 相关文档

- 论文修改计划：[paper/revision/revision_map.md](../paper/revision/revision_map.md)（审稿意见逐条分析、实验计划、自查发现 S1–S5）
- 数据说明：[data/README.md](../data/README.md)
- 标注工具：[tools/annotation/hbat.py](../tools/annotation/hbat.py)（开 HBAT 请加 `--out`，见 basic_ds2 README 顶部警告）
