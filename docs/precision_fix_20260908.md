# 训练与评估的矩阵乘法精度修复

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-08
- Verification Status: ANALYZED
- Version Label: matmul_precision_fix_v1

## 实现问题与改动

旧训练入口调用 `torch.set_float32_matmul_precision("high")`；校准和评估使用 `highest`。
FP32张量类型不能代替内部矩阵乘法精度设置。

新训练和预设流水线固定 `highest`，校准也显式设置 `highest`。
策略checkpoint的 `numerics` 保存矩阵乘法策略以及CUDA matmul / cuDNN的TF32标志。
后处理校准输出同样记录实际数值设置，避免沿用源策略的旧信息。
评估的 `--float32_matmul_precision auto` 读取checkpoint元数据，并在模型构建与首次前向之前应用。

旧checkpoint没有这些元数据，`auto` 保留历史评估的 `highest` 并提示用户确认。
旧版Cross联合训练使用了 `high`，需要显式传 `--float32_matmul_precision high`；
旧版后处理校准SmoothSpike则使用 `highest`。不要把这两类权重混用同一覆盖选项。

SmoothSpike会缓存首次前向计算的冻结H1旋转。不同精度的比较采用独立进程或新建encoder；
策略构建后若其他代码改变进程全局精度，推理会报错，不继续使用可能不匹配的缓存。
本修复没有更改SNN阈值、mask、T4、Cross结构或旧策略权重。

## 同权重、同200局对照

Checkpoint：独立联合训练Cross50k的EMA；SDT-V3 + 冻结SmoothSpikeT4 + spike-SDSA CrossT4。
训练每卡128、三卡全局384、累积1、seed42、warmup3334；原优化器与蒸馏权重保持不变。

四套各10任务，每任务官方初始状态0–4共5次，seed7逐任务重置、FP32、
relative、chunk/open-loop12、env256、wait10、二值夹爪不再归一化。
参照列取旧完整评估中完全相同的200个episode ID，不拿旧2,000局均值作配对参照。

| Suite | 旧评估highest | 训练一致high |
| --- | ---: | ---: |
| Spatial | 20/50（40.0%） | 47/50（94.0%） |
| Object | 35/50（70.0%） | 49/50（98.0%） |
| Goal | 28/50（56.0%） | 48/50（96.0%） |
| Long | 33/50（66.0%） | 41/50（82.0%） |
| 总计 | 116/200（58.0%） | 185/200（92.5%） |

净增69次成功、34.5个百分点。逐局：两边成功110，仅high成功75，仅highest成功6，两边失败9。
40个任务中23个改善、2个退化、15个持平。
两边权重不变；干预的是整个策略的矩阵乘法精度，不能把全部收益只分配给文本encoder某一层。

已核对新评估40次严格EMA加载（1,172张量）、40个task与overall退出0、
200个唯一episode、实际high设置，以及events/task/summary成功数一致。
完整2,000局high复测正在进行；本表不能作为其最终成绩。

## 数值检查与后续训练

相同40条训练指令，公开冻结SmoothSpike的high与highest输出有效token相对MSE为0.171887845。
这项测量固定了batch与权重。

修复后的highest数值检查使用实际B=128训练接口和B=1推理接口：
有效token相对MSE为0.004138468、余弦0.997820735；PAD相对MSE约4.06e-13。
它通过预先设定的0.005差异门槛，但不是逐位完全相同。
对公共ANN接口的有效token相对MSE为0.003643746、余弦约0.998297。

新5k联合训练从公共组件初始化，不继承旧LIBERO学生策略，文本骨干仍冻结。
本轮只调整数值策略；没有加入新融合蒸馏损失。5k结果尚未完成，不作性能承诺。

## 统计解释与边界

Overall Confidence: CAUTION。该固定样本支持精度错配明显影响闭环结果；尚无完整benchmark、
多训练seed、未见语言泛化或节能证据。未计算依赖独立episode假设的显著性检验；
没有声称各版本统计等效。状态为ANALYZED，本次是干预对照，不是同条件复现。

Fallacy scan：11/11已检查。

| 项目 | 检查结论 |
| --- | --- |
| Simpson | 四套方向均改善；同时保留任务级退化，不只看总分。 |
| Ecological | 结论限定这些任务/episode，不外推到任意指令。 |
| Berkson | 固定前5个初始状态，不按成功与否筛选；不代表全部状态。 |
| Collider | 全200局结束后汇总，不以完成速度或成功作纳入条件。 |
| Base rate | 每套分母50、整体200；明确给出成功数。 |
| Regression to mean | 同权重配对提供对照，但单seed不能排除所有重跑波动。 |
| Survivorship | 40任务全部退出0，无丢弃episode。 |
| Look-elsewhere | 报告两种指定精度及全部四套，无阈值/模型扫描择优。 |
| Garden of forking paths | 探索性诊断；固定协议、保留旧结果，不包装为预注册研究。 |
| Correlation/causation | 精度是明确干预；不据此归因某个内部模块或所有历史退化。 |
| Reverse causality | 精度设置先于rollout，成功/失败不改变该设置。 |

原融合中的文本Self-Attention/FFN和动作头仍为ANN；该结果不证明全策略已纯SNN。
