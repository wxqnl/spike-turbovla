# LIBERO 实验结果

更新：2026-09-14。以下分数来自本项目的实际评估，不是 TurboVLA 论文分数。当前采用新 A 80k 作为上游基线；逐维 WTA、多阈值 WTA、Shiftmax＋逐维统一归入 Action Head 修改类。旧 A 50k 与既有动作头成绩保留为历史记录。

## 当前基线与 Action Head 修改类

新 A 和 Spike FFN 均完成 40 个任务、每任务初态 0–49，共 2,000 回合；四套各 500 回合。逐任务、分套与总计一致，所有任务及整体退出码为 0。本次只整理既有结果，没有重新训练或评测。

| 版本 | Spatial | Object | Goal | Long（libero_10） | Overall | 成功 / 总回合 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 新 A 80k：固定上游 + ANN 动作头 | 95.00%（475） | 95.40%（477） | 96.60%（483） | 82.40%（412） | **92.35%** | **1847/2000** |
| Action Head：Spike FFN 对照 | 79.40%（397） | 91.40%（457） | 82.20%（411） | 52.60%（263） | **76.40%** | **1528/2000** |
| Action Head：逐维 WTA | — | — | — | — | 未公布完整评测 | — |
| Action Head：多阈值 WTA | — | — | — | — | 未公布完整评测 | — |
| Action Head：Shiftmax＋逐维 | — | — | — | — | 未公布完整评测 | — |

括号中是每套 500 回合的成功数。三个空白行是仍在训练的独立实验，不能用当前训练 loss 或早期预测误差替代成功率。

- 新 A：4 卡 × 32 × 累积 1，global batch 128，80,000 次更新、10.24M 样本，warmup 10,000 步；与旧 A 50k / global384 是独立训练。相比旧 A 91.05% 增加 26 次成功、1.30 个百分点，但训练量、batch 和更新历史不同。
- Spike FFN：前 1,000 步按 global128 训练，此后 global1024，最终 10,875 步；样本数为 `1000 × 128 + 9875 × 1024 = 10,240,000`，不能直接乘 `10875 × 1024`。仅 FFN 支路脉冲化，状态投影和注意力保留 ANN。
- 逐维 WTA / 多阈值 WTA：各 2 卡 × 192；Shiftmax＋逐维：4 卡 × 96。三组均为 global384、26,667 步、10,240,128 次样本曝光，按 1,280,000 个样本 warmup。每组从公共组件初始化，probe 权重、旧 A、旧 Deep、旧策略及优化器不作为初始化。
- 结构和公共初始化路径固定不等于上游参数冻结：除 SmoothSpike 文本骨干外，其余七组参数联合训练。保留专家动作 L1 + 0.1 feature KD + 0.1 action KD，teacher 仅提供训练期监督。
- 评测统一使用训练数据 128 × 16 刷新后的 EMA；刷新执行 0 次优化器更新。FP32 / highest、CUDA matmul 与 cuDNN TF32 关闭，seed7、legacy sentence / corrected head、relative、chunk/open-loop12、wait10、env256、夹爪不额外归一化，不录视频。
- Shiftmax 的四 rank / 每卡96与 WTA 的双 rank / 每卡192会改变数据分片、顺序及本地 BN；同时更换了路由与 V 处理。这些比较没有多训练 seed，不作严格单变量或统计显著性结论。

新 A 完成于 **2026-09-13 21:39:27（UTC+8）**；Spike FFN 完成于 **2026-09-13 15:24:08（UTC+8）**。[逐任务计数与原始来源](results/libero_20260913.json) · [配置与运行命令](action_head_variants.md)

## 历史五版完整结果

每版四套共 40 个任务，每任务 50 个初始状态，共 2,000 回合。所有表内完整评估的 40 个任务和整体退出码均为 0，任务、分套与整体计数一致。本次整理没有重新训练或重跑 rollout。

| 项目 | 仅 Vision spike | + SmoothSpike（校准版） | + Spike Cross（旧版） | + Spike Cross（新版） | + Spike Action Head |
| --- | ---: | ---: | ---: | ---: | ---: |
| 视觉编码器 | SDT-V3 19M | SDT-V3 19M | SDT-V3 19M | SDT-V3 19M | SDT-V3 19M |
| 冻结文本骨干 | ANN BERT | SmoothSpike T4 | SmoothSpike T4 | SmoothSpike T4 | SmoothSpike T4 |
| 视觉 / 文本投影 | ANN | ANN（文本校准） | ANN | Signed PLIF | 同新版 Spike Cross |
| Cross | ANN | ANN | 旧二值 SDSA | 旧二值 SDSA | 同新版 Spike Cross |
| 文本 Self-Attention / FFN | ANN | ANN | ANN | WTA / Spike FFN | 同新版 Spike Cross |
| 状态 / 动作头 | ANN | ANN | ANN | ANN | Spike SDSA T4 |
| Spatial | 90.6% | 90.2% | 91.6% | 95.4% | 80.0% |
| Object | 93.8% | 93.0% | 97.6% | 99.4% | 84.8% |
| Goal | 92.4% | 95.4% | 95.0% | 94.2% | 66.6% |
| Long | 78.6% | 76.4% | 79.4% | 75.2% | 45.4% |
| Overall | 88.85% | 88.75% | 90.90% | 91.05% | 69.20% |
| 成功 / 总回合 | 1777/2000 | 1775/2000 | 1818/2000 | 1821/2000 | 1384/2000 |
| 策略训练步数 | 70,000 | 70k 基座 + 文本投影校准 | 50,000 | 50,000 | 18,750 |
| GPU 数 | 3 | 沿用基座 | 3 | 2 | 8 |
| 每卡 microbatch | 128 | 基座 128 | 128 | 96 → 192 | 128 |
| 梯度累积 | 1 | 基座 1 | 1 | 2 → 1 | 1 |
| 全局 batch | 384 | 基座 384 | 384 | 384 | 1024 |
| warmup 步数 | 3334 | 沿用基座 | 3334 | 3334 | 1251 |
| 学习率日程总步数 | 70,000 | 沿用基座 | 70,000 | 70,000 | 18,750 |
| 策略样本曝光 | 26.88M | 继承 26.88M；无新增策略训练 | 19.20M | 19.20M | 19.20M |
| 评估权重 | EMA | 校准后的 raw | EMA | 训练数据刷新 BN 的 EMA | 训练数据刷新 BN 的 EMA |

### 训练记录应怎样读

- **SmoothSpike 校准版**继承 Vision 70k 的 EMA 策略，仅拟合文本投影；文件中 `global_step=0`、`source_global_step=70000`。它不是另一次独立 70k 联合训练，评估文件使用 raw。
- **旧 Cross**从公共组件独立联合训练，原目标 70k，表中是停止后保存的 50k EMA。主表使用与训练匹配的 `high` 完整评估：1,818/2,000，90.90%。此前错误 `highest` 设置的 60.20% 留作诊断，不再作为这个模型的主成绩。
- **新版 Spike Cross**从公共组件初始化。前 5,000 步为 2 卡 × 每卡 96 × 累积 2；从保存的第 5,000 步恢复模型、优化器和 EMA，后续改为 2 卡 × 每卡 192 × 累积 1，始终保持 global batch 384。microbatch 和重计算设置发生过变化，不能写成全程每卡 192，也不承诺两种 microbatch 的 BN 数值等价。
- **Spike Action Head 基础版**保留新版 Spike Cross 的投影和融合结构，改用脉冲状态投影及动作解码器。它从公共初始化重新训练，不是拿已经训练好的新版 Spike Cross 继续改头。8 × 128 × 18,750 = 19.20M 样本曝光，与新版 Spike Cross 相同，但 batch 和优化器更新次数不同。
- 表中的学习率日程是记录的 `lr_schedule_steps`，不等于实际完成步数。各正式联合训练使用专家动作 L1 + 0.1 视觉特征 MSE 蒸馏 + 0.1 动作 MSE 蒸馏；文本骨干冻结。视觉学习率 1e-5，其余可训练部分 5e-5，训练 seed 42。

新版 Spike Cross 比旧 Cross 高 0.15 个百分点（3 次成功），Spike Action Head 基础版比新版 Spike Cross 低 21.85 个百分点。这里仅描述这组固定回合的结果。初始化、训练历史、batch、更新次数、mask 和数值策略并不完全相同；这不是严格单变量消融，也没有多训练 seed 或统计显著性结论。

## 评估协议与数值差异

共同协议：Spatial / Object / Goal / Long（`libero_10`）各 10 个任务，每任务初始状态 0–49；任务 seed 7；FP32；相对动作；action chunk 和 open-loop 均为 12；环境图像 256；开始等待 10 步；二值夹爪不再归一化。

| 版本 | 文本句子 mask | head mask | 矩阵乘法 / TF32 | 权重处理 |
| --- | --- | --- | --- | --- |
| Vision 70k | legacy | legacy | 旧 task JSON 未记录 numerics，不补写 TF32 标志 | 原 EMA |
| SmoothSpike 校准版 | legacy | corrected | 沿用校准时的 highest；旧 task JSON 未记录完整 numerics | 从 Vision EMA 派生的 raw |
| 旧 Cross 50k | legacy | corrected | high；CUDA matmul、cuDNN TF32 均为 true | 原 EMA |
| 新版 Spike Cross、Spike Action Head 基础版及状态修复版、完整 WTA 融合 | legacy | corrected | highest；两类 TF32 均为 false | 训练数据 128 × 16 帧刷新 BN 后的 EMA |

新版 Spike Cross、Spike Action Head 版与完整 WTA 融合版的 EMA 统计刷新只读取记录的 LIBERO **训练数据**，执行 0 个优化步骤；不读取评估轨迹。刷新文件只供推理，不作为续训 checkpoint。`refresh_wta_norms.py` 保留历史名称，但支持这三种模型。旧 Cross 不走这项刷新流程。

## 补充：完整 WTA 融合版与 Spike Action Head 状态修复

| 版本 | Spatial | Object | Goal | Long | Overall | 成功 / 总回合 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整 WTA 融合 | 96.0% | 91.4% | 90.4% | 62.8% | 85.15% | 1703/2000 |
| Spike Action Head 状态修复 + 1,875 步续训 | 84.0% | 87.0% | 73.2% | 50.6% | 73.70% | 1474/2000 |

- 完整 WTA 融合版使用 8 卡 × 128、18,750 步、warmup 1,251，样本曝光 19.20M；不是主表里的旧 Spike Cross。
- Spike Action Head 状态修复版以 Spike Action Head 基础版为基座，先选取 512 步头部修复权重，再进行全策略续训。表中评估的是保存的 **1,875 步** checkpoint，新增全策略曝光 1.92M；不是停止时未保存的第 2,349 步，也不是完成了计划中的 18,750 步续训。
- 修复采用状态群体编码 32、范围 4、状态注意力增益 4；默认 Spike Action Head 基础版仍为 0、4、1。修复版的 73.70% 比 Spike Action Head 基础版高 4.50 个百分点，但增加了训练和选择预算，不能把差值全部归因于编码修复。

## Spike Action Head v2：只有离线筛选，没有新全量成绩

9 月 11 日的旧头对照与四个新候选各完成 4,096 步 head-only 适配。四个新候选均未通过既定离线门槛，没有启动配对闭环或正式八卡训练。

最佳新候选“共享投影 + 完整三层脉冲读出”的导出 EMA 动作 L1 为 0.099340，旧头对照为 0.091872；新候选仍高约 8.13%。这不是 LIBERO 成功率，不混入主表，也没有据此替换 Spike Action Head 基础版的默认实现。完整候选源码、权重和日志保留在实验目录。

## 历史诊断

| 记录 | 评估量 | 结果 | 说明 |
| --- | ---: | ---: | --- |
| 旧 Cross 50k，错误 highest | 2,000 | 60.20%（1,204/2,000） | 与旧 high 训练不匹配；保留错误设置记录 |
| 同一旧 Cross 50k，匹配 high | 2,000 | 90.90%（1,818/2,000） | 主表采用此完整结果 |
| 旧 Cross 50k，匹配 high 的小样本检查 | 200 | 92.50%（185/200） | 不能替代 2,000 回合成绩 |
| 新 highest 配方，从公共初始化训练 5k | 200 | 26.00%（52/200） | 已于 9 月 8 日完成，不是旧 Cross 的重评 |
| 同一 Vision 训练轨迹的 55k EMA | 2,000 | 88.65%（1,773/2,000） | 主表采用与 SmoothSpike 基座对应的 70k |

更早的 200 回合记录依次为：Vision 174/200；未校准 SmoothSpike 替换 147/200；耦合 mask 修改 110/200；已训练 Vision 策略的文本校准版 179/200。它们属于调试阶段，不覆盖主表完整结果。[精度修复记录](precision_fix_20260908.md)

## 数据来源

[当前基线与 FFN 汇总](results/libero_20260913.json) 保存新 A 80k 和 Spike FFN 的原始 summary、逐任务计数与训练分段说明。[历史机器可读汇总](results/libero_20260911.json) 保存原七个完整结果的模型定义、训练预算、实际评估协议、每任务计数、原始 summary 路径和完成时间。路径对应各实验服务器的 `/data/Minko`，新 A 在 node42，当前动作头实验在 node43。

原始权重、数据、视频及完整日志不随 GitHub 代码提交，也没有因本次整理被删除。原始 summary 中的历史验证标记保持原样；这里的计数核对不等于额外复跑或独立复现。

| 版本 | 完成时间（UTC+8） |
| --- | --- |
| Vision 70k | 2026-09-08 00:19 |
| SmoothSpike 校准版 | 2026-09-07 19:46 |
| 旧 Cross，high 完整重评 | 2026-09-08 21:56 |
| 新版 Spike Cross | 2026-09-11 21:35 |
| Spike Action Head 基础版 | 2026-09-11 14:01 |
| 完整 WTA 融合版 | 2026-09-10 12:03 |
| Spike Action Head 状态修复 + 1,875 步 | 2026-09-11 19:58 |
