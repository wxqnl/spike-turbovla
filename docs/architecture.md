# 模型结构与实现边界

更新：2026-09-11。本页对应已合入的模型代码；成功率与训练预算见[结果表](results.md)。

## 五版主线

| 模块 | 仅 Vision | + SmoothSpike | + 旧 Spike Cross | A | 原 AA |
| --- | --- | --- | --- | --- | --- |
| 视觉骨干 | SDT-V3 19M | 同左 | 同左 | 同左 | 同左 |
| 冻结文本骨干 | ANN BERT | SmoothSpike T4 | SmoothSpike T4 | SmoothSpike T4 | SmoothSpike T4 |
| 视觉 / 文本投影 | ANN | ANN | ANN | Signed PLIF + 时序 BN | 同 A |
| 双向 Cross | ANN | ANN | 二值 SDSA / PLIF | 保留旧 SDSA | 同 A |
| 六层文本 Self-Attention / FFN | ANN | ANN | ANN | Signed WTA / Spike FFN | 同 A |
| 状态投影 / ACT 动作解码 | ANN | ANN | ANN | ANN | Spike SDSA T4 |
| 连续动作输出 | 12 × 7 | 12 × 7 | 12 × 7 | 12 × 7 | 12 × 7 |

`ann` 预设仍可使用 DINOv3 + ANN BERT + ANN Cross；它是另一个训练入口，主表没有为它补造本项目全量成绩。SmoothSpike 校准版是已训练 Vision 策略的后处理版本，不等同于新运行 `--preset vision_text`。

## 旧 Spike Cross

`turbovla/models/components/spike_fusion.py` 保留双向 SDSA 式 Cross：Q/K/V 为二值事件，使用无 Softmax 的重合计数；PLIF 的膜电位只存在于本次 forward 的 T=4 循环。每个时间步先投影，再均值读出。LayerNorm、残差、DropPath 和连续膜电位仍存在。

旧 Cross 版本的文本 Self-Attention、FFN、投影及动作头都是 ANN。该 Cross 是对 [Spike-driven Transformer V2](https://arxiv.org/abs/2404.03663) 思路的本项目适配，没有官方跨模态预训练权重。

## A：补齐投影和文本交互

配置为 `backbone_type=spike_text_only`、`cross_attention_type=spike_sdsa`、`text_only_spike_projections=true`。

- 视觉 / 文本投影使用 {-1,0,+1} 的双极性 PLIF 事件、Linear 和按时间步统计的 masked BN。它们先做 T=4 均值读出，再进入旧 Cross。
- 每个文本块使用 Signed WTA Self-Attention（文本 top-k=3）和 Spike FFN；每层在自己的 T=4 循环后读出。
- 旧 Cross 的二值 Q/K/V、LayerNorm、DropPath=0.1、残差风格与内部 T=4 不变。A 没有将 Cross 换成 WTA。
- 新增时序 BN 为 53 层：投影 5 层、六个文本块共 48 层。动作头仍是原 ANN ACT。

主要实现：`spike_text_only.py`、`wta_fusion.py`、`wta_operators.py` 与 `turbovla.py`。保留 `text_only_spike_projections=false` 只为兼容早期 ANN 投影的历史 A；它不是本表的 A，也不能与新版 A 直接互续训练。

## AA：在 A 上更换完整动作头

配置在 A 基础上增加 `action.head_type=spike_sdsa`，实现位于 `turbovla/models/spike_action_head.py`。

输入是 A 已读出的视觉 / 文本 memory 和 8 维状态。状态投影生成两个状态 token；三层 ACT 解码器在 T=4 内执行 Self-Attention、Cross-Attention 和 FFN，再通过三层脉冲驱动读出预测 12 × 7 动作。

注意力 Q/K/V 是 0/1 事件；状态、FFN 和读出投影使用有符号单位事件。最终输出为 `tanh(mean_T(linear(events)))`，保持连续动作。masked 时序 BN、连续残差和膜电位不被计作纯脉冲突触计算。状态在每个观测的 forward 内重置，不跨控制回合累计。

原 AA 默认 `spike_state_population=0`、`spike_state_population_range=4`、`spike_state_attention_gain=1`。可选修复版使用 32、4、4：增加固定错相阈值的群体编码残差，并在二值重合计算之后加权状态贡献。它有独立结果和额外训练预算，不覆盖原 AA。

## B：保留的补充实验

B 的配置为 `backbone_type=spike_wta`、`cross_attention_type=spike_wta`。投影至六层融合一直保留 T=4，最后统一读出；Signed WTA Cross 两个方向分别选最多 3 个文本 source、8 个视觉 source，动作头仍是 ANN。

因此 A/B 不只改变一个 Cross 算子：Cross 的归一化、残差、DropPath 及时间组织也不同。B 的实现和 `train_wta.py` 保留，但不拿它替换主表中的旧 Cross。

早期 `spike_temporal` 代码仅作为历史兼容路径保留，没有在本次主表中给它附加成绩。未通过筛选的 AA-v2 候选仍在独立实验快照，没有合入这些默认模型。

## 训练、EMA 与部署

所有正式联合训练都冻结预训练文本骨干，更新视觉、投影、融合及动作模块。这是“冻结语言骨干条件下的联合训练”，不是全部参数都训练。

A / AA / B 的 EMA 必须在 checkpoint 记录的训练数据上刷新视觉 BN 与新时序 BN。当前正式结果使用 128 批 × 每批 16 帧、0 个优化步骤。评估加载器拒绝未刷新的 EMA；训练续跑使用原训练 checkpoint，不能使用 evaluation-only 导出。梯度重计算保护 BN，避免同一个时间步因重算重复更新统计。

SDT-V3 保留其多级 SFA 表达；SmoothSpike 保留全局 H1 及逆变换，H2/H3 已融合进权重。不能把整个策略描述成处处二值。GPU 实现仍使用稠密张量；当前结果没有给出实测节能、稀疏硬件加速或未见语言泛化证据。
