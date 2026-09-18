# LIBERO 复现说明

更新：2026-09-15。想复现当前最好结果，请先使用本页的 **新 A（Vision＋Text＋Cross 脉冲化、完整 ANN 动作头），92.35%** 流程。入口固定为 `train_action_head --variant a_base`；后面的旧 hybrid 预设、`train_wta` 和 Spike Action Head 是其他实验。

<a id="best-a-reproduce"></a>

## 当前最佳版本：新 A，92.35%

已完成四套各 500 回合、总计 2,000 回合：Spatial 95.00%、Object 95.40%、Goal 96.60%、Long 82.40%，总计 **1847/2000 = 92.35%**。[根 README](../../README.md)保留各版本的完整结果。

### 2026-09-18 实验更新

完整四套结果：新A **92.35%**，A−WTA **90.15%**，① **83.65%**，② **77.45%**，③ **82.95%**；④仅Long **66.6%（333/500）**，⑤仅Long **56.6%（283/500）**。⑥、minimal-cross BS96续训及渐进全头组尚无最终评测结果（运行快照见根README）。

②—⑥、A−WTA、minimal-cross与渐进头的实现现已收录。[变体配方与入口](../../docs/action_head_variants.md)区分fresh复现和minimal-cross分段续训；[机器可读结果](../../docs/results/libero_20260918.json)区分四套2000与Long500，不对未完成实验补分。

新A的warmup为10,000步，从峰值约0.1倍升到峰值；之后视觉1e-5、其余5e-5恒定到80,000步，`min_lr_ratio=1`。下述80k入口不使用WSD，也没有实际cosine衰减。渐进组的半余弦alpha是动作输出混合日程，与LR不同。

### 1. 先找到对应代码

| 内容 | 代码 / 配置 |
| --- | --- |
| 推荐训练入口 | [train_action_head.py](train_action_head.py)，选择 `--variant a_base` |
| 固定训练配方 | [configs/action_head_variants.json](configs/action_head_variants.json) 的 `a_base` 与 `common_training` |
| 新 A 的融合参数 | [train_spike_text_only.py](train_spike_text_only.py) 的 `text_only_arguments` |
| 共享启动器 | [train_wta.py](train_wta.py) 的 `main`；由推荐入口覆盖参数后调用 |
| 完整模型 / ANN 动作头 | [turbovla.py](../../turbovla/models/turbovla.py)、[action_head.py](../../turbovla/models/action_head.py) |
| 脉冲文本块 / SDSA Cross | [spike_text_only.py](../../turbovla/models/components/spike_text_only.py)、[spike_fusion.py](../../turbovla/models/components/spike_fusion.py) |
| 训练循环 | [train_mixed.py](../../turbovla/training/train_mixed.py)、[trainer.py](../../turbovla/training/trainer.py) |

新 A 的实际配置为 `spike_text_only` 融合、`spike_sdsa` Cross、脉冲视觉 / 文本投影和 `action_head_type=ann`。文本与 Cross 使用 T=4；8 维状态经 ANN MLP 生成 **2 个 state token**，3 层 ACT 的注意力、FFN 和动作读出均为 ANN，输出 12×7 动作块。仅文本骨干冻结，其余模块联合训练。

`train_wta` 的文件名容易混淆：本流程只复用它的启动函数。直接运行它会训练完整 WTA 融合版（85.15%）。`run_experiment --preset vision_text_cross` 是旧 hybrid 配方；`--variant shiftmax_perdim` 是近期①动作头实验（83.65%）。这些入口都不能替代新 A 的 `--variant a_base`。

### 2. 准备依赖、模型、数据和工作目录

先完成[根 README 的安装、模型下载与四套数据准备](../../README.md)。必须准备 SDT-V3 19M、已融合的 SmoothSpike BERT、公共 BERT / DINOv3、TurboVLA LIBERO 教师和 GroundingDINO 权重，以及本页下方列出的四套 LeRobot v2.1 数据。GitHub 不包含这些资产，也不包含产生 92.35% 的已训练策略；只 clone 代码不能直接运行模型。

在仓库根目录按 `.env.example` 创建 `.env`，填好 `SPIKE_WORK_ROOT`；数据或模型不在默认位置时，设置相应的 `SPIKE_*_PATH`、`SPIKE_DATA_ROOT` 等变量。运行输出必须放在代码目录外。

激活已安装项目依赖的 Python 环境，先进入自己 clone 的仓库根目录，再执行以下环境配置。`SPIKE_CODE_ROOT` 自动取当前仓库路径；`.env` 中的工作目录、模型和数据路径由读者自行配置。

```bash
export SPIKE_CODE_ROOT="$PWD"
set -a
source "$SPIKE_CODE_ROOT/.env"
set +a
export PYTHONPATH="$SPIKE_CODE_ROOT:$SPIKE_CODE_ROOT/third_party/vla_adapter:$SPIKE_CODE_ROOT/third_party/starvla_runtime"
export SPIKE_RUN_ROOT="${SPIKE_RUN_ROOT:-$SPIKE_WORK_ROOT/experiments/spike-turbovla/libero}"
export SPIKE_TEXT_INIT_DIR="$SPIKE_RUN_ROOT/common_text_init_a80k_new"
export SPIKE_A_RUN="$SPIKE_RUN_ROOT/a_base_80k_new"
unset NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
```

`.env` 中的 `SPIKE_CUDA_DEVICES` 属于旧三卡流水线，不要把它设成四个编号。本流程用正式命令的 `CUDA_VISIBLE_DEVICES` 选择四张卡，实际进程数由 `--nproc_per_node=4` 确定。以下 GPU 编号均是示例，执行前必须确认空闲；这些命令用于新实验，不要覆盖已完成或正在运行的目录。

### 3. 准备公共文本投影初始化

首次复现时，选一张空闲卡，运行一次：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.prepare_cross_joint \
  --mode calibrate --output_dir "$SPIKE_TEXT_INIT_DIR"
```

产物为 `$SPIKE_TEXT_INIT_DIR/text_projection.pth`。这一步只用训练指令校正公共 BERT / GroundingDINO 与 SmoothSpike 的投影接口，不加载已训练 A 策略，也不读取 eval 轨迹。已有同方案的公共初始化文件时，直接把 `SPIKE_TEXT_INIT_DIR` 指向它，不重复生成。不要使用本页末尾“已训练策略的文本接口校正”来代替这一步。

### 4. 预览并启动 80k 训练

不占 GPU 的参数预览：

```bash
python -m experiments.libero.train_action_head \
  --variant a_base --world-size 4 --dry-run \
  --run_dir "$SPIKE_A_RUN" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 80000 --batch_size 32 --grad_accum_steps 1
```

预览应显示 `global_batch=128`、`optimizer_steps=80000`、`sample_exposures=10240000`；重复出现的底层参数以最后一次覆盖为准，新 A 最终为 `spike_text_only / spike_sdsa / ann`。预览只检查命令展开，不检查资产是否齐全，也不是训练完成证明。

确认四张卡空闲、公共初始化已准备好，并且 `$SPIKE_A_RUN` 尚未存在后，启动正式训练：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 --module experiments.libero.train_action_head \
  --variant a_base --run_dir "$SPIKE_A_RUN" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 80000 --batch_size 32 --grad_accum_steps 1 \
  --resume_mode none --approve_training \
  2>&1 | tee "${SPIKE_A_RUN}.train.log"
```

| 配置 | 新 A 固定值 |
| --- | --- |
| GPU / batch / 累积 / 总步数 | 4 / 每卡32 / 1 / 80,000 |
| 样本曝光 / warmup | 10,240,000 / 10,000步 |
| 初始化 / seed | 公共组件、新动作头与优化器；不恢复旧策略 / 42 |
| 学习率 / 损失 | 视觉1e-5、其余5e-5；action L1 + 0.1 feature KD + 0.1 action KD |
| 精度 / EMA | FP32、highest、CUDA与cuDNN TF32关闭 / 0.999 |
| 保存 | 每1,000步、step20和final；前缀 `spike_a_turbovla_base` |

配方会拒绝不匹配的卡数、batch、累积次数或步数。训练日志在 `${SPIKE_A_RUN}.train.log`，最终 checkpoint 在 `$SPIKE_A_RUN/checkpoints/spike_a_turbovla_base_80000.pth`。这是前台训练命令；长任务可在自己的持久终端中执行。**本入口只训练，不自动进行 EMA 刷新或评测。**

<a id="best-a-eval"></a>

### 5. 训练结束后导出 EMA，再评测四套

仅在训练正常退出、80,000步 checkpoint 保存完整后执行。使用一张空闲卡，在原训练数据上刷新 EMA 的 BN：

```bash
export SPIKE_A_EMA="$SPIKE_A_RUN/evaluation/spike_a_turbovla_base_80000_ema_trainbn128x16.pth"
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.refresh_wta_norms \
  --checkpoint "$SPIKE_A_RUN/checkpoints/spike_a_turbovla_base_80000.pth" \
  --output "$SPIKE_A_EMA" --batches 128 --batch_size 16
```

必须完成128/128个batch，optimizer更新为0。导出文件只用于推理 / eval，不用于续训；`refresh_wta_norms` 的历史命名不意味着把新 A 改成 WTA。输出路径必须是新的；不要反复刷新同一已完成实验。

安装并配置 LIBERO 与 OSMesa 后，按以下命令顺序运行四套，各10任务×50个初态（0–49），总计2,000回合：

```bash
mkdir -p "$SPIKE_A_RUN/evaluation/final_2000"
for SPIKE_SUITE in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES=0 python -m experiments.libero.evaluate \
    --ckpt_path "$SPIKE_A_EMA" --checkpoint_weights ema \
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
    --result_json_path "$SPIKE_A_RUN/evaluation/final_2000/$SPIKE_SUITE.json" \
    --log_path "$SPIKE_A_RUN/evaluation/final_2000/$SPIKE_SUITE.log" || break
done
```

沿用 checkpoint 的 legacy sentence / corrected head mask；不要用评估数据调整权重、BN或初始化。四个 JSON 必须分别完成500回合，合计2,000；overall按成功数相加后除以总回合数计算。某套失败或未结束时，只报告部分结果，不重复计分、不自动重试。

92.35%是一次已完成实验的记录，不保证新运行得到相同小数；旧50k A的91.05%是另一场实验。上面的输出路径均相对于读者自己的 `SPIKE_RUN_ROOT`，不依赖作者服务器。

## 数据布局

`SPIKE_DATA_ROOT` 下需要四个 LeRobot 数据集，每个包含 `meta/tasks.jsonl`、episode 元数据、Parquet 与两路视频：

```text
libero_10_no_noops_1.0.0_lerobot/
libero_goal_no_noops_1.0.0_lerobot/
libero_object_no_noops_1.0.0_lerobot/
libero_spatial_no_noops_1.0.0_lerobot/
```

使用 `binary_open` 夹爪语义，`normalize_binary_gripper=false`。统计文件为 `configs/libero_all4_stats.json`，键为 `libero_all4_no_noops`。不要把其他夹爪编码的数据直接套进这套配方。

`configs/online_text_layout.json` 仅记录40条训练指令的 padding 布局，没有 embedding 缓存。保留 legacy 句子/位置协议，学生和教师采用 corrected batch/head mask。历史 ANN 动作头会消费 padding 行，因此公共文本接口校正同时对齐有效行和 padding 行；Spike Action Head 版的脉冲动作头使用显式 memory padding mask。

## 历史 hybrid 预设：独立联合训练

以下保留旧 hybrid 三卡70k流程，不是上方92.35%的新 A。只有要研究这些旧预设时，才在完成根目录安装、权重与 `.env` 配置后运行：

```bash
python -m experiments.libero.run_experiment --preset vision_text_cross --run-name cross_s42
```

| 项目 | 默认配方 |
| --- | --- |
| 优化步 / batch | 70,000 / 每卡128 × 3卡，累积1 |
| 初始化 | 按所选预设加载公开视觉 / 文本权重及 GroundingDINO；新动作头与新优化器 |
| 冻结范围 | BERT / SmoothSpike 文本骨干；DINOv3 未使用的 mask token |
| 学习率 | 视觉骨干 1e-5，其余 5e-5 |
| warmup / weight decay | 3,334步 / 1e-10 |
| 优化器 / EMA | AdamW，betas=(0.9,0.95) / 0.999 |
| 精度 / seed | FP32、float32_matmul_precision=highest / 42 |
| 损失 | 专家动作 L1 + 0.1视觉特征 MSE 蒸馏 + 0.1动作 MSE 蒸馏 |

各模块组合与下载命令见[根目录 README](../../README.md)。所有组合独立初始化并联合训练；ANN BERT 跳过文本校正。

SmoothSpike 文本初始化只用训练指令对齐公开 BERT + GroundingDINO 投影，不使用先前训练完成的策略或评估轨迹。随后该投影参与联合优化。启动前检查使用8个真实训练帧和纯专家动作 L1，不执行优化器更新。

流水线保存源码快照、初始化信息、第一步参数更新记录、训练日志和退出码。最终目标步数的 EMA 权重通过检查且训练退出0后，自动开始四套评估。运行目录已存在时拒绝重开，失败不自动重试。

底层入口也可独立使用：

```bash
python -m experiments.libero.run_experiment --help
python -m experiments.libero.train --help
python -m experiments.libero.evaluate --help
python -m experiments.libero.prepare_cross_joint --help
```

## 新版 Spike Cross 与脉冲动作头专用入口

当前最佳新 A 使用本页开头的 `train_action_head --variant a_base`。历史50k Spike Cross、Spike Action Head基础版与完整WTA融合版使用[历史专用训练配方](../../docs/training_variants.md)；近期动作头变体见[Action Head配方](../../docs/action_head_variants.md)。这些版本与旧 `vision_text_cross` 预设分开使用，脉冲融合版本的EMA仍需先用原训练数据刷新BN。

## 评估

当前最佳新 A 请使用上方 [EMA导出与四套评测](#best-a-eval)。下面保留旧hybrid checkpoint的单套示例，共10任务 × 每任务50次。四套分别运行共2,000局；仅旧 `run_experiment` 流水线支持用 `--eval-trials 50` 在训练后自动评估，默认每任务5次、共200局。新 A 的独立训练入口不自动eval，两种评测量不能混报。

```bash
mkdir -p outputs/eval
CUDA_VISIBLE_DEVICES=4 python -m experiments.libero.evaluate \
  --ckpt_path /path/to/spike_turbovla_70000.pth \
  --checkpoint_weights ema \
  --libero_root "${SPIKE_LIBERO_ROOT:-$SPIKE_WORK_ROOT/benchmarks/LIBERO}" \
  --dinov3_path "${SPIKE_DINO_PATH:-$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/dinov3-vitb16}" --bert_path "${SPIKE_BERT_PATH:-$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/bert-base-uncased}" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops --normalize_binary_gripper false \
  --task_suite_name libero_spatial --num_trials_per_task 50 \
  --seed 7 --precision fp32 --control_mode relative \
  --float32_matmul_precision auto \
  --chunk_size 12 --num_open_loop_steps 12 --num_steps_wait 10 \
  --env_img_res 256 --mujoco_gl osmesa --pyopengl_platform osmesa \
  --save_video false --result_json_path outputs/eval/spatial.json \
  --log_path outputs/eval/spatial.log
```

其余三套为 `libero_object`、`libero_goal`、`libero_10`，每套使用独立输出路径。可用 `--task_ids` 做任务级调度；同一个任务不要重复记分。

评估旧的文本校正 checkpoint 时使用 `--checkpoint_weights raw`。该文件由 vision-only EMA 策略派生，但文件自身不含 EMA 字段，不能选 `ema`。
同时显式选择 `--float32_matmul_precision highest`。如果评估的是旧high训练器产生的Cross联合训练EMA，
则把示例的auto改为high；不要对所有旧SmoothSpike权重一律选high。

## 已训练策略的文本接口校正

这是单独的阶段实验，不是上面的独立联合训练：

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.libero.calibrate_text_projection \
  --baseline_checkpoint /path/to/vision_only_70000.pth \
  --smoothspike_path "${SPIKE_TEXT_PATH:-$SPIKE_WORK_ROOT/pretrained/SmoothSpike/smoothspike-bert-base-fused}" \
  --output_dir /path/to/new_text_calibration
```

脚本保留原策略的 EMA 视觉、融合和动作参数，仅拟合新的768→256文本投影。它使用40条训练指令及40个真实训练帧进行接口/动作对照；这些诊断数值不是 rollout 成功率，也不验证未见语言泛化。

Ridge相对正则系数为0.01。部署仍在线运行 SmoothSpike，不使用教师文本查表。输出 `spike_text_projection_calibrated.pth` 需另行评估。

评估实现沿用 [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) 的 rollout 协议；保留第三方来源与许可证。
