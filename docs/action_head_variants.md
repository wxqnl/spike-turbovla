# Action Head 修改类：配置与运行命令

更新：2026-09-14。新 A 80k 是 ANN 动作头基线；逐维 WTA、多阈值 WTA、Shiftmax＋逐维统一放在本类。新 A 已完成 2,000 回合评测，成功率为 **92.35%（1847/2000）**；三个新动作头尚未公布完整评测。

当前服务器上的三组训练和评测等待器已经启动。下面的训练、刷新、评测命令用于**另外新建的实验**，不能在现有运行目录重复执行。本次发布不改正在运行的源码快照，不重启、扩卡或重跑评测。

## 1. 固定的上游与训练范围

- 双视角 SDT-V3 19M，输入224、输出14×14；在线 SmoothSpike BERT，T=4。
- signed PLIF 视觉/文本投影，`spike_text_only` 融合；保留旧二值 Spike SDSA Cross，文本 Self-Attention / FFN 使用脉冲实现。
- hidden256、8 heads、6层融合、FFN2048。状态输入8维，动作输出7维，chunk/open-loop12。

“固定”指结构与公共初始化方案。视觉、投影、融合、状态分支、动作解码器仍联合训练，只有文本骨干冻结。每组从 SDT-V3、SmoothSpike、GroundingDINO 等公共组件初始化；公共文本投影初始化只使用训练指令校正接口，不含已训练策略。公开 TurboVLA 是冻结教师，保持自己的2个 state token，不从学生继承8-token配置。三组不加载新 A、旧 Deep、旧 AA、容量预检的策略或优化器。

[机器可读配置](../experiments/libero/configs/action_head_variants.json)由[统一入口](../experiments/libero/train_action_head.py)直接读取。入口核对实际 world size、每卡 batch、累积次数和总步数；没有匹配的配方就停止，不会静默改预算。

## 2. 三种动作头

| 配置项 | 逐维 WTA | 多阈值 WTA | Shiftmax＋逐维 |
| --- | --- | --- | --- |
| `--variant` | `perdim` | `multithreshold` | `shiftmax_perdim` |
| `action_head_type` | `full_spike_state8_wta` | `full_spike_state8_wta` | `full_spike_state8_shiftmax` |
| `spike_state_encoding` | `perdim_plif` | `multithreshold` | `perdim_plif` |
| state token | 8，每个状态坐标一个 | 8，每个状态坐标一个 | 8，每个状态坐标一个 |
| 状态事件 | 每维 signed PLIF，1→256投影 | 每维32阈值，32→256投影 | 同逐维 WTA |
| 神经元时间 / ACT层数 | T4 / 3层 | T4 / 3层 | T4 / 3层 |
| Self / Cross 注意力 | binary multi-winner，top-k 3 / 8 | 同左 | Shiftmax，无 top-k |
| Q / K / V | signed单位事件 | signed单位事件 | Q/K为signed事件，V为归一化投影电流 |
| 头部归一化 | token-local LayerNorm | 同左 | 同左 |
| 状态位置 / 隐藏宽度 | 逐坐标位置偏置 / 256 | 同左 | 同左 |
| 最终动作 | 脉冲驱动投影，T维电流均值后tanh | 同左 | 同左 |

状态先使用固定的逐维训练数据统计归一化。每个神经元时间步都接收全部8个坐标，**不把8维切成4段**。多阈值版每维每步32个阈值，T4交错共128个边界覆盖标准化区间[-4,4]，减去零状态的事件码后输出-1/0/+1；它不是旧群体残差修复的续跑。逐维 PLIF 配置中保留 `state_population=32` 以统一元数据，但该分支实际输入事件维度为1。

WTA 的 softmax 只用于 surrogate backward，前向是二值多赢家路由。Shiftmax 自注意力与交叉注意力使用：

```text
w_i = 2 ** (x_i - ceil(log2(sum_j(2 ** x_j))))
x = Q @ K.T
```

有效行的权重和在(0.5,1]，全padding行输出0；有效但没有发放脉冲的key不视为padding。Q/K为单位signed事件，因此QK分数为整数；不增加sqrt(d)、score std归一化或分数取整。ceil采用明确实现的STE归一化反向，保留跨key竞争。V不再额外二值化，聚合后经归一化及signed PLIF再投影。

该实现参考 [Bipolar Self-attention for Spiking Transformers，NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/9316cfcb0a81e53a1f35b4353f115571-Abstract-Conference.html)，适配了 ACT、PLIF、LayerNorm 与 padding。它不是论文整网复现。归一化、电流、膜电位、残差、V聚合及最终机器人动作保留连续量；GPU上使用稠密FP32张量，不宣称整数硬件、稀疏硬件、实测节能或加速。

实现：[逐坐标状态与WTA头](../turbovla/models/full_state_action_head.py)、[共享WTA解码器](../turbovla/models/deep_spike_action_head.py)、[Shiftmax头](../turbovla/models/shiftmax_action_head.py)。共享文件保留历史名称，引用它不表示恢复已停止的旧 Deep 实验。

## 3. 训练预算

| 配置 | 新 A 基线 | 逐维 WTA | 多阈值 WTA | Shiftmax＋逐维 |
| --- | ---: | ---: | ---: | ---: |
| `--variant` | `a_base` | `perdim` | `multithreshold` | `shiftmax_perdim` |
| GPU / rank数 | 4 | 2 | 2 | 4 |
| 每卡batch / 累积 | 32 / 1 | 192 / 1 | 192 / 1 | 96 / 1 |
| global batch | 128 | 384 | 384 | 384 |
| 优化器更新步数 | 80,000 | 26,667 | 26,667 | 26,667 |
| 样本曝光次数 | 10,240,000 | 10,240,128 | 10,240,128 | 10,240,128 |
| warmup | 10,000步 | 1,280,000样本 | 同左 | 同左 |
| vision重计算 | 关闭 | all | all | all |
| Cross / head重计算 | 开启 / ANN头不适用 | 开启 / 开启 | 开启 / 开启 | 开启 / 开启 |
| projection重计算 | 关闭 | 关闭 | 关闭 | 关闭 |
| checkpoint前缀 | `spike_a_turbovla_base` | `spike_state8_perdim` | `spike_state8_multithreshold` | `spike_shiftmax_perdim` |

三种动作头以 `ceil(10,240,000 / 384) = 26,667` 对齐预算，多出的128次曝光小于一个global batch。样本warmup保留A配方的128样本前瞻项；`warmup_steps=3334`，实际调度以 `lr_warmup_samples=1280000` 为准。新A使用原10,000步warmup。

公共设置为 seed42、FP32/highest、两类TF32关闭；AdamW(0.9,0.95)、eps1e-8、weight decay1e-10、clip1、EMA0.999；视觉LR1e-5，其余5e-5，warmup后保持不变。损失是动作L1 + 0.1 featureKD + 0.1 actionKD，textKD0。每1,000步、step20和final保存，日志每20步报告一次，4个DataLoader worker/rank。

四rank×96与双rank×192虽然global batch相同，但数据分片顺序和每rank BN不同。Shiftmax还同时改变路由与V路径。与A、FFN比较时batch和更新历史也不同，不能写成严格单变量消融。训练曲线只能说明拟合进度，最终能力看2,000回合评测。

## 4. 准备环境与预览

先完成[根README的依赖、权重和四套训练数据准备](../README.md)，加载 `.env`，并按[公共文本投影初始化](training_variants.md#公共文本投影初始化)准备 `SPIKE_TEXT_INIT_DIR/text_projection.pth`。三个新动作头使用同一份初始化文件。

服务器操作先 `ssh New-H100-3`，再 `cd /data/Minko`。把 `SPIKE_CODE_ROOT` 指向**已更新的发布仓库**，从工作目录使用它的模块：

```bash
# 发布副本；若另行检出仓库，请改为对应路径
export SPIKE_CODE_ROOT=/data/Minko/spike-turbovla-release-20260911
export PYTHONPATH="$SPIKE_CODE_ROOT:$SPIKE_CODE_ROOT/third_party/vla_adapter:$SPIKE_CODE_ROOT/third_party/starvla_runtime"
# 使用已准备好的项目Python环境，并加载该检出的.env
set -a
source "$SPIKE_CODE_ROOT/.env"
set +a
unset NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

# 三个新动作头复用已准备好的公共初始化；不重新校准现有运行
export SPIKE_TEXT_INIT_DIR="$SPIKE_RUN_ROOT/common_text_init_new"
```

下列示例要求已经选用安装了项目依赖的Python。公共模型、数据、运行输出不在Git仓库内；GitHub发布不包含训练checkpoint。只安装代码而没有准备资产，不能直接启动模型。

逐维 WTA 预览，不创建输出、不导入训练器、不使用GPU：

```bash
python -m experiments.libero.train_action_head \
  --variant perdim --world-size 2 --dry-run \
  --run_dir "$SPIKE_RUN_ROOT/action_head_perdim_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 26667 --batch_size 192 --grad_accum_steps 1
```

多阈值预览改为 `--variant multithreshold`；Shiftmax预览使用 `--variant shiftmax_perdim --world-size 4 --batch_size 96`。`--world-size`只用于预览，正式运行以torchrun的实际world size为准。`--help`列出公共参数；本入口固定配方中的warmup、保存周期、重计算等值，展开后的JSON是最终配置。

## 5. 新建训练的命令

必须先确认指定卡空闲，并使用未存在的新运行目录。以下三条不是要依次执行的脚本；根据本次要复现的版本选择一条。当前服务器上的三组实验不再执行这些命令。

### 逐维 WTA：2卡×192

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module experiments.libero.train_action_head \
  --variant perdim --run_dir "$SPIKE_RUN_ROOT/action_head_perdim_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 26667 --batch_size 192 --grad_accum_steps 1 \
  --resume_mode none --approve_training
```

### 多阈值 WTA：2卡×192

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module experiments.libero.train_action_head \
  --variant multithreshold --run_dir "$SPIKE_RUN_ROOT/action_head_multithreshold_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 26667 --batch_size 192 --grad_accum_steps 1 \
  --resume_mode none --approve_training
```

### Shiftmax＋逐维：4卡×96

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 --module experiments.libero.train_action_head \
  --variant shiftmax_perdim --run_dir "$SPIKE_RUN_ROOT/action_head_shiftmax_perdim_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 26667 --batch_size 96 --grad_accum_steps 1 \
  --resume_mode none --approve_training
```

### 新 A 基线配方：4卡×32、80k

以下是另建基线实验的命令；已经完成的新 A 不重跑。GPU编号仅为新运行的示例，不授权自动使用node42的空卡或占用当前node43任务。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 --module experiments.libero.train_action_head \
  --variant a_base --run_dir "$SPIKE_RUN_ROOT/a_base_80k_new" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 80000 --batch_size 32 --grad_accum_steps 1 \
  --resume_mode none --approve_training
```

## 6. EMA刷新与完整评测

统一入口只训练，不自动创建后台队列。对于新建实验，训练进程成功退出、final checkpoint保存完整后，再刷新及评测。部署中的三组已有各自控制器自动衔接，不额外运行本节。

以另建的Shiftmax实验为例，只使用本组的一张空闲卡，训练数据128×16刷新BN，0次optimizer更新：

```bash
export SPIKE_HEAD_RUN="$SPIKE_RUN_ROOT/action_head_shiftmax_perdim_new"
export SPIKE_HEAD_EMA="$SPIKE_HEAD_RUN/evaluation/spike_shiftmax_perdim_26667_ema_trainbn128x16.pth"
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.refresh_wta_norms \
  --checkpoint "$SPIKE_HEAD_RUN/checkpoints/spike_shiftmax_perdim_26667.pth" \
  --output "$SPIKE_HEAD_EMA" --batches 128 --batch_size 16
```

逐维、多阈值分别替换运行目录和前缀 `spike_state8_perdim`、`spike_state8_multithreshold`；新A使用 `spike_a_turbovla_base_80000.pth`。刷新输出仅供推理，不作为训练恢复文件；迁移机器时仍要提供checkpoint记录的公共资产和训练数据路径。

下面顺序评估四套，各10任务×50回合，总计2,000回合。目录必须是新的，不重复汇总任务：

```bash
mkdir -p "$SPIKE_HEAD_RUN/evaluation/final_2000"
for SPIKE_SUITE in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.evaluate \
    --ckpt_path "$SPIKE_HEAD_EMA" --checkpoint_weights ema \
    --libero_root "${SPIKE_LIBERO_ROOT:-$SPIKE_WORK_ROOT/benchmarks/LIBERO}" \
    --dinov3_path "${SPIKE_DINO_PATH:-$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/dinov3-vitb16}" \
    --bert_path "${SPIKE_BERT_PATH:-$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/bert-base-uncased}" \
    --stats_path "$SPIKE_CODE_ROOT/experiments/libero/configs/libero_all4_stats.json" \
    --stats_key libero_all4_no_noops --normalize_binary_gripper false \
    --task_suite_name "$SPIKE_SUITE" --num_trials_per_task 50 \
    --seed 7 --precision fp32 --float32_matmul_precision highest \
    --control_mode relative --chunk_size 12 --num_open_loop_steps 12 --num_steps_wait 10 \
    --env_img_res 256 --mujoco_gl osmesa --pyopengl_platform osmesa \
    --save_video false \
    --result_json_path "$SPIKE_HEAD_RUN/evaluation/final_2000/$SPIKE_SUITE.json" \
    --log_path "$SPIKE_HEAD_RUN/evaluation/final_2000/$SPIKE_SUITE.log" || break
done
```

本例依赖已配置的LIBERO/OSMesa环境，使用初态0–49，不用评估轨迹调BN或权重。每套均成功完成后，按四份JSON的成功数/总回合数相加计算overall；只有全部四套完成才能标记最终2,000回合结果。

## 7. 已部署实验的位置与允许操作

路径均相对于对应服务器的 `/data/Minko/experiments/spike-turbovla/libero`：

| 组别 | 节点 / 原GPU | 运行目录 | 当前阶段 |
| --- | --- | --- | --- |
| 新A | node42 / 2,3,5,6 | `fusionA_turbovla_base_20260912` | 训练与评测已完成 |
| Spike FFN | node43 / 原0–3 | `fusionA_heads_4plus4_20260912/ffn` | 训练与评测已完成 |
| 逐维 WTA | node43 / 4,5 | `fusionA_fullspike_state8_pair_20260912/perdim` | 训练中，已有评测等待器 |
| 多阈值 WTA | node43 / 6,7 | `fusionA_fullspike_state8_pair_20260912/multithreshold` | 训练中，已有评测等待器 |
| Shiftmax＋逐维 | node43 / 0,1,2,3 | `fusionA_shiftmax_state8_20260913/perdim` | 训练中，已有评测等待器 |

当前允许只读查看各根的 `status.json`、`training/train.log`、`training/checkpoints`、`evaluation/status.json` 和结果。原WTA两组维持每组2卡，不恢复已取消的扩卡提议；不得重复启动、抢卡、重置GPU、自动重试或恢复旧Deep。node42释放的卡没有自动分配权限。通过node43访问node42只用于读取新A结果，不使用node41。

```bash
# 当前三组只读状态；不会启动或终止任何任务
for SPIKE_EXISTING_RUN in \
  /data/Minko/experiments/spike-turbovla/libero/fusionA_fullspike_state8_pair_20260912/perdim \
  /data/Minko/experiments/spike-turbovla/libero/fusionA_fullspike_state8_pair_20260912/multithreshold \
  /data/Minko/experiments/spike-turbovla/libero/fusionA_shiftmax_state8_20260913/perdim; do
  sed -n '1,180p' "$SPIKE_EXISTING_RUN/status.json"
  sed -n '1,100p' "$SPIKE_EXISTING_RUN/evaluation/status.json"
done
```

GitHub中的统一入口读取可移植配方；服务器已有的 `run_arm.py` / `run_eval_after_training.py` 继续读取各自固定的 `source_snapshot`。发布不移动这些目录、不覆盖运行参数，也不把控制器、依赖缓存、checkpoint或视频加入Git。新动作头核心来自实际训练快照；旧SDSA状态群体修复接口和默认值继续保留，避免破坏历史模型。
