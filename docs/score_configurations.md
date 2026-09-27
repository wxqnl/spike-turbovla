# 得分对应配置

主推 Spike-Driven VLA 的 LIBERO 轻量增强版和公共 MT50 40k EMA 成绩见[当前 benchmark 与代码](spike_driven.md)。本页保留早期 ANN 基线与逐模块消融的历史对照，不以旧实验编号命名主推架构。

README中的版本名和每个成功率均直达本页的对应实验。每节区分历史实际运行与可复现入口；未完成实验没有最终分数。

<a id="current-common"></a>

## 近期实验共同设置

以下新A、A-no-WTA、旧FFN-only及①—⑤共用评测：seed7，每任务50固定初态0—49；FP32/highest、CUDA/cuDNN TF32关闭、legacy sentence / corrected head mask、relative、chunk/open-loop12、wait10、env256、二值夹爪不归一化、OSMesa、无视频。每组自身EMA仅用训练数据128×16刷新BN，optimizer更新为0。完整四套为2000回合，Long-only为500。

新A、A-no-WTA及①—⑤训练：seed42，AdamW(.9,.95)/eps1e-8/wd1e-10/clip1/EMA.999，vision LR1e-5、其他5e-5；warmup后恒定（min_lr_ratio=1），不是WSD。四套no-noops LeRobot数据；L1+.1 featureKD+.1 actionKD，textKD0；仅文本骨干冻结，其余七组联合训练。旧FFN-only另见自身分段记录，不能套用未核实的统一fresh配方。

[训练入口](../experiments/libero/train_action_head.py) · [训练与评测步骤](../experiments/libero/README.md#best-a-reproduce) · [模型组装](../turbovla/models/turbovla.py) · [训练器](../turbovla/training/trainer.py)

<a id="a_base"></a>

## 新 A（80k，ANN动作头）

libero_spatial: 475/500=95.00%；libero_object: 477/500=95.40%；libero_goal: 483/500=96.60%；libero_10: 412/500=82.40%。

四套Overall：**92.35%（1847/2000）**。

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | ANN MLP，2-token |
| 动作头 / 消融边界 | self/cross/FFN/readout全部ANN |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头为ANN，无独立spike积分T |
| batch / 更新 / 样本 | 4×32，global128，acc1；80000步；10240000样本 |
| warmup / LR | 10000步；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 关闭；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 a_base](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant a_base` |
| 实现代码 | [action_head.py](../turbovla/models/action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L84)；相对来源 `fusionA_turbovla_base_20260912/evaluation/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="a_no_wta"></a>

## A-no-WTA（文本SDSA式点积）

libero_10: 393/500=78.60%；libero_spatial: 451/500=90.20%；libero_object: 478/500=95.60%；libero_goal: 481/500=96.20%。

四套Overall：**90.15%（1803/2000）**。

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | ANN MLP，2-token |
| 动作头 / 消融边界 | action head同A；仅六层文本WTA改signed-spike点积 |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头为ANN，无独立spike积分T |
| batch / 更新 / 样本 | 4×32，global128，acc1；80000步；10240000样本 |
| warmup / LR | 10000步；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 关闭；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 a_no_wta](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant a_no_wta` |
| 实现代码 | [action_head.py](../turbovla/models/action_head.py) · [wta_fusion.py](../turbovla/models/components/wta_fusion.py) · [wta_operators.py](../turbovla/models/components/wta_operators.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L123)；相对来源 `fusionA_no_wta_turbovla_base_20260916/evaluation_full2000/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="ffn_only"></a>

## 旧 FFN-only

libero_spatial: 397/500=79.40%；libero_object: 457/500=91.40%；libero_goal: 411/500=82.20%；libero_10: 263/500=52.60%。

四套Overall：**76.40%（1528/2000）**。

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | A的ANN状态，2-token |
| 动作头 / 消融边界 | self/cross/readout为ANN，仅decoder FFN脉冲化 |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头spike分支T4 |
| batch / 更新 / 样本 | 前1000步global128；之后9875步global1024；共10875步、10240000样本 |
| 复现边界 | 历史分段训练，无统一fresh variant；不得将10875×1024当作本次样本预算 |
| 记录 | [原分段训练记录](results/libero_20260913.json)；warmup等未在该分段摘要完整列出的设置不补写为统一fresh值 |
| 实现代码 | [hybrid_action_head.py](../turbovla/models/hybrid_action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L162)；相对来源 `fusionA_heads_4plus4_20260912/ffn/evaluation/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="shiftmax_perdim"></a>

## ① 原始Shiftmax＋逐维

libero_spatial: 456/500=91.20%；libero_object: 486/500=97.20%；libero_goal: 424/500=84.80%；libero_10: 307/500=61.40%。

四套Overall：**83.65%（1673/2000）**。

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | perdim_plif，8-token |
| 动作头 / 消融边界 | 入口PLIF；Q/K脉冲、V连续；Shiftmax；spike FFN/readout |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头spike分支T4 |
| batch / 更新 / 样本 | 4×96，global384，acc1；26667步；10240128样本 |
| warmup / LR | 1280000样本（CLI 3334步）；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 开启；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 shiftmax_perdim](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant shiftmax_perdim` |
| 实现代码 | [full_state_action_head.py](../turbovla/models/full_state_action_head.py) · [shiftmax_action_head.py](../turbovla/models/shiftmax_action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L201)；相对来源 `fusionA_shiftmax_state8_20260913/perdim/evaluation/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="shiftmax_continuous_inputs"></a>

## ② 连续attention入口＋脉冲FFN

libero_spatial: 433/500=86.60%；libero_object: 449/500=89.80%；libero_goal: 403/500=80.60%；libero_10: 264/500=52.80%。

四套Overall：**77.45%（1549/2000）**。

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | perdim_plif，8-token |
| 动作头 / 消融边界 | 连续attention入口；Q/K脉冲、V连续；Shiftmax；spike FFN/readout |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头spike分支T4 |
| batch / 更新 / 样本 | 4×96，global384，acc1；26667步；10240128样本 |
| warmup / LR | 1280000样本（CLI 3334步）；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 开启；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 shiftmax_continuous_inputs](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant shiftmax_continuous_inputs` |
| 实现代码 | [full_state_action_head.py](../turbovla/models/full_state_action_head.py) · [shiftmax_action_head.py](../turbovla/models/shiftmax_action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L240)；相对来源 `fusionA_shiftmax_continuous_inputs_state8_20260914/perdim/evaluation/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="shiftmax_continuous_inputs_ann_ffn"></a>

## ③ 连续attention入口＋ANN FFN

libero_spatial: 450/500=90.00%；libero_object: 487/500=97.40%；libero_goal: 420/500=84.00%；libero_10: 302/500=60.40%。

四套Overall：**82.95%（1659/2000）**。

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | perdim_plif，8-token |
| 动作头 / 消融边界 | 连续attention入口；Shiftmax；ANN FFN（ReLU/dropout0.1）；spike readout |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头spike分支T4 |
| batch / 更新 / 样本 | 4×96，global384，acc1；26667步；10240128样本 |
| warmup / LR | 1280000样本（CLI 3334步）；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 开启；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 shiftmax_continuous_inputs_ann_ffn](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant shiftmax_continuous_inputs_ann_ffn` |
| 实现代码 | [full_state_action_head.py](../turbovla/models/full_state_action_head.py) · [shiftmax_action_head.py](../turbovla/models/shiftmax_action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L279)；相对来源 `fusionA_shiftmax_continuous_inputs_ann_ffn_state8_20260914/perdim/evaluation/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="shiftmax_prelif_ann_ffn"></a>

## ④ 入口PLIF＋ANN FFN

libero_10: 333/500=66.60%。

**仅Long500，不存在四套Overall。**

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | perdim_plif，8-token |
| 动作头 / 消融边界 | 入口PLIF；Shiftmax；ANN FFN（ReLU/dropout0.1）；spike readout |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头spike分支T4 |
| batch / 更新 / 样本 | 4×96，global384，acc1；26667步；10240128样本 |
| warmup / LR | 1280000样本（CLI 3334步）；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 开启；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 shiftmax_prelif_ann_ffn](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant shiftmax_prelif_ann_ffn` |
| 实现代码 | [full_state_action_head.py](../turbovla/models/full_state_action_head.py) · [shiftmax_action_head.py](../turbovla/models/shiftmax_action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L318)；相对来源 `fusionA_shiftmax_prelif_ann_ffn_state8_20260915/perdim/evaluation_long/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="shiftmax_projectfirst"></a>

## ⑤ 状态先投影后PLIF＋脉冲FFN

libero_10: 283/500=56.60%。

**仅Long500，不存在四套Overall。**

| 配置 | 本次得分对应值 |
| --- | --- |
| 状态 | project_then_plif，8-token；先坐标Linear1→256，再hidden PLIF |
| 动作头 / 消融边界 | 入口PLIF；Shiftmax；spike FFN/readout |
| 上游 | SDT-V3＋SmoothSpike T4＋spike_text_only；A-no-WTA之外均保留文本Top3 WTA＋原SDSA Cross |
| T | 上游T4；动作头spike分支T4 |
| batch / 更新 / 样本 | 4×96，global384，acc1；26667步；10240128样本 |
| warmup / LR | 1280000样本（CLI 3334步）；vision1e-5、其他5e-5，warmup后恒定 |
| 初始化 | 公共组件fresh；无已训练A或probe权重。维护恢复等历史见原记录，不承诺RNG bitwise复现 |
| 视觉重计算 | 开启；上游Cross开启、projection关闭 |
| 可执行配方 | [扩展配方中的 shiftmax_projectfirst](../experiments/libero/configs/extensions/action_head_variants.json)；入口参数 `--variant shiftmax_projectfirst` |
| 实现代码 | [full_state_action_head.py](../turbovla/models/full_state_action_head.py) · [shiftmax_action_head.py](../turbovla/models/shiftmax_action_head.py) |
| 结果记录 | [结果JSON](results/libero_20260918.json#L342)；相对来源 `fusionA_shiftmax_projectfirst_state8_spike_ffn_20260916/perdim/evaluation_long/summary.json` |

[返回README结果表](../README.md#current-results) · [共同评测协议](#current-common)

<a id="historical"></a>

## 历史结果配置

以下使用当时的训练、mask、精度与权重处理，不套用新A的80k配方。记录缺失的字段保持缺失。

<a id="historical-vision70k"></a>

### 仅 Vision spike：88.85%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "vision": "SDT-V3 19M",
    "text": "frozen ANN BERT",
    "projection": "ANN",
    "cross": "ANN",
    "text_interaction": "ANN",
    "action_head": "ANN"
  },
  "training": {
    "optimizer_steps": 70000,
    "base_optimizer_steps": 0,
    "gpus": 3,
    "batch_per_gpu": 128,
    "gradient_accumulation": 1,
    "global_batch": 384,
    "warmup_steps": 3334,
    "lr_schedule_steps": 70000,
    "sample_exposures": 26880000,
    "seed": 42
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "ema",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "legacy",
    "numerics": null,
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[turbovla.py](../turbovla/models/turbovla.py) · [action_head.py](../turbovla/models/action_head.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)

<a id="historical-smoothspike_calibrated"></a>

### Vision + SmoothSpike（校准版）：88.75%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "vision": "SDT-V3 19M",
    "text": "frozen SmoothSpike T4",
    "projection": "ANN; calibrated text interface",
    "cross": "ANN",
    "text_interaction": "ANN",
    "action_head": "ANN"
  },
  "training": {
    "optimizer_steps": 0,
    "base_model": "vision70k",
    "base_optimizer_steps": 70000,
    "base_global_batch": 384,
    "base_gpus": 3,
    "base_batch_per_gpu": 128,
    "base_gradient_accumulation": 1,
    "base_warmup_steps": 3334,
    "additional_policy_sample_exposures": 0,
    "seed": 42,
    "method": "Post-hoc text projection calibration; inherited vision70k EMA policy, exported as raw weights."
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "raw",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "corrected",
    "numerics": null,
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[smoothspike.py](../turbovla/text/smoothspike.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)

<a id="historical-cross50k_high"></a>

### 再加 Spike Cross（旧版，high 对齐）：90.90%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "vision": "SDT-V3 19M",
    "text": "frozen SmoothSpike T4",
    "projection": "ANN",
    "cross": "binary SDSA / PLIF T4",
    "text_interaction": "ANN",
    "action_head": "ANN"
  },
  "training": {
    "optimizer_steps": 50000,
    "original_target_steps": 70000,
    "gpus": 3,
    "batch_per_gpu": 128,
    "gradient_accumulation": 1,
    "global_batch": 384,
    "warmup_steps": 3334,
    "lr_schedule_steps": 70000,
    "sample_exposures": 19200000,
    "seed": 42
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "ema",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "corrected",
    "numerics": {
      "checkpoint_recorded": null,
      "cuda_matmul_allow_tf32": true,
      "cudnn_allow_tf32": true,
      "float32_matmul_precision": "high",
      "requested": "high"
    },
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[spike_fusion.py](../turbovla/models/components/spike_fusion.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)

<a id="historical-A"></a>

### 新版 Spike Cross：91.05%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "vision": "SDT-V3 19M",
    "text": "frozen SmoothSpike T4",
    "projection": "signed PLIF T4",
    "cross": "unchanged old binary SDSA / PLIF T4",
    "text_interaction": "signed WTA Self-Attention + spike FFN",
    "action_head": "ANN"
  },
  "training": {
    "optimizer_steps": 50000,
    "gpus": 2,
    "global_batch": 384,
    "warmup_steps": 3334,
    "lr_schedule_steps": 70000,
    "sample_exposures": 19200000,
    "seed": 42,
    "batch_phases": [
      {
        "from_step": 1,
        "through_step": 5000,
        "batch_per_gpu": 96,
        "gradient_accumulation": 2
      },
      {
        "from_step": 5001,
        "through_step": 50000,
        "batch_per_gpu": 192,
        "gradient_accumulation": 1
      }
    ],
    "resume": "Resumed model, optimizer, and EMA at saved step 5000; enabled full vision gradient checkpointing."
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "ema",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "corrected",
    "numerics": {
      "checkpoint_recorded": "highest",
      "cuda_matmul_allow_tf32": false,
      "cudnn_allow_tf32": false,
      "float32_matmul_precision": "highest",
      "requested": "auto"
    },
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[spike_text_only.py](../turbovla/models/components/spike_text_only.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)

<a id="historical-AA"></a>

### 新版 Spike Cross + Spike Action Head：69.20%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "vision": "SDT-V3 19M",
    "text": "frozen SmoothSpike T4",
    "projection": "same as new Spike Cross",
    "cross": "same as new Spike Cross",
    "text_interaction": "same as new Spike Cross",
    "action_head": "T4 spike state projection + SDSA decoder + signed spike readout; continuous final actions"
  },
  "training": {
    "optimizer_steps": 18750,
    "gpus": 8,
    "batch_per_gpu": 128,
    "gradient_accumulation": 1,
    "global_batch": 1024,
    "warmup_steps": 1251,
    "lr_schedule_steps": 18750,
    "sample_exposures": 19200000,
    "seed": 42,
    "initialization": "Fresh common components; not continued from trained new Spike Cross or full WTA fusion policies."
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "ema",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "corrected",
    "numerics": {
      "checkpoint_recorded": "highest",
      "cuda_matmul_allow_tf32": false,
      "cudnn_allow_tf32": false,
      "float32_matmul_precision": "highest",
      "requested": "auto"
    },
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[spike_action_head.py](../turbovla/models/spike_action_head.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)

<a id="historical-B"></a>

### 完整 WTA 融合：85.15%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "vision": "SDT-V3 19M",
    "text": "frozen SmoothSpike T4",
    "projection": "signed PLIF T4",
    "cross": "signed multi-winner WTA, text top-k=3 / vision top-k=8",
    "text_interaction": "signed WTA Self-Attention + spike FFN",
    "action_head": "ANN"
  },
  "training": {
    "optimizer_steps": 18750,
    "gpus": 8,
    "batch_per_gpu": 128,
    "gradient_accumulation": 1,
    "global_batch": 1024,
    "warmup_steps": 1251,
    "lr_schedule_steps": 18750,
    "sample_exposures": 19200000,
    "seed": 42,
    "vision_gradient_checkpoint_scope": "early"
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "ema",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "corrected",
    "numerics": {
      "checkpoint_recorded": "highest",
      "cuda_matmul_allow_tf32": false,
      "cudnn_allow_tf32": false,
      "float32_matmul_precision": "highest",
      "requested": "auto"
    },
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[wta_fusion.py](../turbovla/models/components/wta_fusion.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)

<a id="historical-AA_statefix_1875"></a>

### Spike Action Head 状态修复 + 1,875 步续训：73.70%

以下JSON摘自已发布历史记录，仅列该得分的架构、训练与评测配置：

```json
{
  "architecture": {
    "base": "AA",
    "spike_state_population": 32,
    "spike_state_population_range": 4,
    "spike_state_attention_gain": 4
  },
  "training": {
    "base_model": "AA",
    "base_optimizer_steps": 18750,
    "selected_head_repair_steps": 512,
    "continuation_optimizer_steps": 1875,
    "gpus": 8,
    "batch_per_gpu": 128,
    "gradient_accumulation": 1,
    "global_batch": 1024,
    "continuation_sample_exposures": 1920000,
    "seed": 42,
    "stop_note": "The evaluated checkpoint is step 1875, not the unsaved step-2349 tail."
  },
  "protocol": {
    "precision": "fp32",
    "checkpoint_weights": "ema",
    "effective_sentence_mask": "legacy",
    "effective_head_mask": "corrected",
    "numerics": {
      "checkpoint_recorded": "highest",
      "cuda_matmul_allow_tf32": false,
      "cudnn_allow_tf32": false,
      "float32_matmul_precision": "highest",
      "requested": "auto"
    },
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "control_mode": "relative",
    "env_img_res": 256
  }
}
```

实现：[spike_action_head.py](../turbovla/models/spike_action_head.py)。历史配置不等于默认启动器当前值；批量变更和停止/续训历史必须保留。

[历史结果与协议解释](results.md#historical-results) · [历史训练入口](training_variants.md) · [返回README](../README.md#historical-results)
