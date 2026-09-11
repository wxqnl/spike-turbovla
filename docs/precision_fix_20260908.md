# 训练与评估的矩阵乘法精度修复

首次记录：2026-09-08；完整结果更新：2026-09-11。

## 问题和处理

旧训练入口使用 `float32_matmul_precision=high`，校准和评估却默认 `highest`。两边都使用 FP32 张量，不代表矩阵乘法内部精度相同。SmoothSpike 的冻结文本表示对这个设置敏感。

新训练及校准显式使用 `highest`、关闭 CUDA matmul / cuDNN TF32，并在 checkpoint 的 `numerics` 中记录实际设置。评估 `--float32_matmul_precision auto` 在模型构建及首次前向之前读取它。

旧 checkpoint 缺少该元数据时，auto 保留历史评估默认 highest 并警告。需要区分：

- 旧 Cross 联合训练 EMA：显式使用 `--precision fp32 --float32_matmul_precision high`。
- 旧后处理校准 SmoothSpike：沿用校准时的 highest，使用 `--checkpoint_weights raw`。
- A / AA / B：使用已刷新 BN 的 EMA，`--precision fp32 --float32_matmul_precision auto`。

SmoothSpike 首次前向会缓存冻结 H1 的旋转变换。比较不同精度时使用独立进程或重新创建 encoder；策略构建后更改全局精度会触发推理检查，不能复用不匹配的缓存。这次精度干预没有更改旧策略权重、SNN 阈值、mask 或 Cross 结构。

## 同一 Cross 50k EMA 的完整 2,000 回合

相同 40 个任务，每任务初始状态 0–49；seed 7；FP32；relative；chunk/open-loop 12；env 256；wait 10。表中每套 500 回合。

| 评估精度 | Spatial | Object | Goal | Long | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 highest，与训练不匹配 | 41.6% | 74.0% | 58.6% | 66.6% | 60.20%（1,204/2,000） |
| high，与旧训练一致 | 91.6% | 97.6% | 95.0% | 79.4% | 90.90%（1,818/2,000） |

high 完整评估已于 2026-09-08 21:56 完成，40 个任务及整体退出码为 0。逐回合配对：双方成功 1,123，仅 high 成功 695，仅旧 highest 成功 81，双方失败 101，净增 614 次成功（30.70 个百分点）。全部回合均纳入，没有挑选任务或重跑择优。

这是对整个策略数值设置的干预，不能把全部差值只归因于文本编码器某一层。它也不证明其他模型可以随意切换 high / highest。

## 较早的 200 回合诊断

| Suite | 旧 highest | 匹配 high |
| --- | ---: | ---: |
| Spatial | 20/50 | 47/50 |
| Object | 35/50 | 49/50 |
| Goal | 28/50 | 48/50 |
| Long | 33/50 | 41/50 |
| 总计 | 116/200（58.0%） | 185/200（92.5%） |

该表固定使用每任务前五个初始状态。92.5% 是 200 回合诊断分数，不能替代完整结果 90.90%。

## 数值检查和后续小试

相同 40 条训练指令上，公开冻结 SmoothSpike 的 high 与 highest 有效 token 相对 MSE 为 0.171887845。修复后 highest 配置下，B=128 训练接口与 B=1 推理接口的有效 token 相对 MSE 为 0.004138468、余弦约 0.997821；这不是逐位相同的声明。

随后从公共组件重新初始化的 highest 5k 联合训练已完成，200 回合为 52/200（26.0%），分套为 17、19、15、1 次成功。它不是继续训练旧 Cross 50k，也不能用短训分数否定旧权重精度对齐的结果。

当前五版完整结果与训练差异见[结果记录](results.md)；计数和来源见[结果数据](results/libero_20260911.json)。这些都是单训练 seed 的项目实验，不宣称多 seed 稳定改善、统计等效或实测节能。
