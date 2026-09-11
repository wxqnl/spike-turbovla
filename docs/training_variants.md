# 脉冲融合与动作头：训练及评估入口

完成[根目录安装和资产准备](../README.md)，在仓库根目录加载 `.env` 后使用本页。三个入口都只负责训练，训练结束后需要单独导出 EMA、评估；不会自动排队或启动后台监控。原来的四个 `run_experiment --preset` 仍使用 hybrid 路径，不会隐式变成新版 Spike Cross 或 Spike Action Head 版。

## 入口对应关系

| 版本 | Python 入口 | 融合配置 | 动作头 | checkpoint 前缀 |
| --- | --- | --- | --- | --- |
| 新版 Spike Cross | `experiments.libero.train_spike_text_only` | spike_text_only + spike projections | ANN | spike_text_projected |
| Spike Action Head 基础版 | `experiments.libero.train_spike_aa` | 同新版 Spike Cross | spike_sdsa | spike_aa |
| 完整 WTA 融合版 | `experiments.libero.train_wta` | spike_wta | ANN | spike_wta |

Spike Action Head 基础版入口固定为本次八卡、19.20M 样本曝光配方，会核对实际 torchrun world size。要研究其他预算，应显式创建新实验配方，不能把不同设置混入 Spike Action Head 基础版的记录。

## 公共文本投影初始化

新建一个输出目录；这一步只对齐公共 BERT + GroundingDINO 的文本接口，不读取已经训好的新版 Spike Cross、完整 WTA 融合版、Spike Action Head 版策略，也不读取评估轨迹。

```bash
SPIKE_TEXT_INIT_DIR="$SPIKE_RUN_ROOT/common_text_init_new"
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.prepare_cross_joint \
  --mode calibrate --output_dir "$SPIKE_TEXT_INIT_DIR"
```

结果为 `$SPIKE_TEXT_INIT_DIR/text_projection.pth`。训练过程中投影仍参与更新，SmoothSpike 骨干冻结。此初始化与“对已训练 Vision 70k 策略进行后处理校准”是不同流程。

以下 `--vision_checkpoint` 可以替换为下载到其他位置的 SDT-V3 19M 权重。训练数据、BERT、DINOv3、SmoothSpike、GroundingDINO、教师和 stats 均使用根目录配置的路径。

## 先检查展开的训练参数

以 Spike Action Head 基础版为例，预览不导入训练器、不加载权重、不创建输出，也不使用 GPU：

```bash
python -m experiments.libero.train_spike_aa \
  --run_dir "$SPIKE_RUN_ROOT/aa_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 18750 --batch_size 128 --grad_accum_steps 1 \
  --dry-run --world-size 8
```

`--world-size` 只用于预览。正式训练由 torchrun 的实际进程数决定，不允许用该参数虚报卡数。新版 Spike Cross / 完整 WTA 融合版入口也支持相同预览；新版 Spike Cross 的示例使用 `--world-size 2`。

## 新版 Spike Cross：两卡，global batch 384

下面启动一场新版 Spike Cross 训练，采用已完成运行后半段的 microbatch 配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module experiments.libero.train_spike_text_only \
  --run_dir "$SPIKE_RUN_ROOT/a_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 50000 --batch_size 192 --grad_accum_steps 1 \
  --warmup_steps 3334 --lr_schedule_steps 70000 \
  --save_steps 5000 --resume_mode none \
  --vision_gradient_checkpointing --vision_checkpoint_scope all \
  --approve_training
```

报告中的新版 Spike Cross 在前 5,000 步用每卡 96、累积 2，然后从第 5,000 步 checkpoint 完整恢复，改为每卡 192、累积 1；两段 global batch 都是 384。上面的统一 microbatch 命令不能被称为原历史轨迹的逐位复现。原始两阶段命令保留在结果来源目录的 `train_launch2.log` 和 `train_perf.log`。

## Spike Action Head 基础版：八卡，global batch 1024

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 --module experiments.libero.train_spike_aa \
  --run_dir "$SPIKE_RUN_ROOT/aa_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 18750 --batch_size 128 --grad_accum_steps 1 \
  --save_steps 1875 --resume_mode none \
  --vision_gradient_checkpointing --vision_checkpoint_scope all \
  --approve_training
```

入口采用 warmup 1,251、日程 18,750、19.20M 样本曝光，动作头采用基础版 Spike SDSA；不会启用状态群体修复或 Spike Action Head v2。它从公共初始化开始，不加载已训练新版 Spike Cross / 完整 WTA 融合版学生权重。

## 完整 WTA 融合：八卡补充实验

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 --module experiments.libero.train_wta \
  --run_dir "$SPIKE_RUN_ROOT/b_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 18750 --batch_size 128 --grad_accum_steps 1 \
  --warmup_steps 1251 --lr_schedule_steps 18750 \
  --save_steps 1875 --extra_save_steps 20 375 --resume_mode none \
  --vision_gradient_checkpointing --vision_checkpoint_scope early \
  --approve_training
```

所有正式训练命令都要求明确的 `--approve_training`。请先确认选定卡空闲；直接 torchrun 入口不会替你抢占或调度资源。已有 checkpoint 的目录不能当作新实验覆盖。续训显式使用 `--resume_mode all`，只加载同结构、同语义的原训练 checkpoint。

## EMA 导出：训练数据刷新 BN

以 Spike Action Head 基础版最终 checkpoint 为例；新版 Spike Cross / 完整 WTA 融合版使用对应的文件名前缀。只在训练结束并保存完整 EMA 后运行，输出必须是新路径：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.refresh_wta_norms \
  --checkpoint "$SPIKE_RUN_ROOT/aa_new/checkpoints/spike_aa_18750.pth" \
  --output "$SPIKE_RUN_ROOT/aa_new/evaluation/spike_aa_18750_ema_trainbn128x16.pth" \
  --batches 128 --batch_size 16
```

刷新只用 checkpoint 记录的训练数据，更新视觉 BN 与时序 BN 统计，不更新参数；冻结文本骨干和 Dropout 保持 eval。导出是 `ema_evaluation_only`，使用 `--checkpoint_weights ema` 加载，不用于续训。未刷新 EMA 会被评估加载器拒绝。

迁移 checkpoint 到另一机器时，需要同时确保其中记录的模型资产、源码和训练数据路径可用；尤其统计刷新会读取原训练路径。仓库提供公共组件下载步骤，未附带本项目训练权重。

## 完整评估

将[LIBERO 评估命令](../experiments/libero/README.md#评估)中的 checkpoint 改为上述导出文件，保留 `--checkpoint_weights ema --precision fp32 --float32_matmul_precision auto`。

四套分别运行 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`；每任务 50 回合，共 2,000 回合。每套使用独立输出目录，不合并重复任务。只运行一个 suite 的 500 回合不能报告成全量结果。

旧 Cross 50k 必须显式 high，校准 SmoothSpike 必须 raw/highest；不要套用新版 Spike Cross / Spike Action Head 版的刷新协议去改旧权重。完整协议差异见[结果记录](results.md#评估协议与数值差异)。

## 可选状态修复与发布范围

`repair_state_head.py` 保留已执行的真实特征采集、头部适配和导出工具。它不是默认 Spike Action Head 版训练的一部分；从其导出继续训练属于独立实验，并具有额外预算。入口参数可通过 `python -m experiments.libero.repair_state_head --help` 查看。

本次源码主要来自已评估 Spike Action Head 基础版快照；可选状态修复取自 `fusionAA_statefix_20260911/source_snapshot`，完整 WTA 融合版的日程参数来自其八卡正式训练快照。发布时整合共享入口的预算预览和实际 world-size 检查，不改变已评估模型的前向计算。

临时 GPU 调度器、节点专用启动脚本、未通过筛选的 Spike Action Head v2 候选、缓存与训练产物没有复制进正式代码目录；原文件全部保留在各自实验目录。第三方模型主体仍由原下载脚本和集成补丁准备，未在本次发布中重新实现或简化。
