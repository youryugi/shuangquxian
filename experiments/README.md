# experiments/ 导航

两个阶段，各有自己的 README：

| 阶段 | 目录 | 状态 | 看什么 |
|---|---|---|---|
| **投稿版**（JSTARS-2026-06074） | [submission/](submission/README.md) | 代码冻结，只读 | 论文每张表对应哪个脚本、哪次运行；已知问题 |
| **大修阶段**（2026-11 起） | [revision/](revision/README.md) | 进行中 | 审稿意见 → 实验编号 → 状态 → 当前结论 |

## 每个实验目录的统一约定

```
<experiment>/
  README.md     文档：开头几行给答案 → 术语表 → "结论怎么一步步变的" → 一页总览表 → 按 Round 时间线
                （每轮：为什么做 / 做了什么 / 结果 / 结论；次要大表折叠在 <details> 里）→ 下一步 → 附录
  *.py          代码：训练 / 评估 / 诊断脚本
  runs*/        结果：每个 run 一个目录，含 run_info.json（全部参数）、metrics.json、预测 JSON、日志
```

- 正式数字只认 README 里标为"正式结果"的那一节；历史记录里的数字标了"已取代"的不要引用。
- 结果目录里的 `*.pth / *.pt / *.png / *.jpg / *.npz` 不入库（.gitignore）；重跑脚本可以再生成。
- 术语用英文（baseline、arm、fold、recall、box refinement 等），每个 README 开头有术语表；不用自造的中文简称。
- 运行环境：本机 `C:/Users/79152/.conda/envs/gpr/python.exe`（torch 2.10 + cu128，ultralytics 8.4.70）；IWAC1 / IWAC2（RTX 5090，conda `gpr`，torch 2.11 + cu128），见 basic_ds2 README 附录 D。

## 当前最重要的结论（2026-11-16，basic_ds2 Round 16）

详见 [revision/basic_ds2/README.md](revision/basic_ds2/README.md)。

- 数据集 2 新标注、严格协议下，**attention 走 fusion 对检测 F1 的提升是 +0.004 / +0.019（两块显卡），和换显卡同量级**。论文 Table IV 的 0.67 → 0.73 不能复现。
- band 用在 box refinement 上 +0.072（10/10 seeds），但其中只有 +0.025 是 band 形状特有的，其余只用框标注的 dense 监督也能拿到。
- **不用 band、只加一个 stage 2 框回归就是 +0.159**，超过所有用 band 的方案。band 形状唯一替代不了的地方是 grouped targets（band 准确时 recall 比框回归高 0.10），但预测的 band 不够准，受数据量限制。
- **需要决定**：论文怎么面对 stage 2 这个更强的对照。**最高优先级的实验**：数据集 1 严格协议重跑（等 band 标注补齐）。

## 相关文档

- 论文修改计划：[paper/revision/revision_map.md](../paper/revision/revision_map.md)（审稿意见逐条分析、实验计划、自查发现 S1–S5）
- 数据说明：[data/README.md](../data/README.md)
- 标注工具：[tools/annotation/hbat.py](../tools/annotation/hbat.py)（开 HBAT 请加 `--out`，见 basic_ds2 README 顶部警告）
