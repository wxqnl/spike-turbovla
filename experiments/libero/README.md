# LIBERO 复现说明

2026-09-08修复了训练/评估矩阵乘法精度错配。新训练用highest，评估auto读checkpoint元数据。
旧Cross联合训练权重需显式high；旧后处理校准SmoothSpike使用highest和raw。
仅指定FP32不够。[迁移说明](../../README.md#重要更新2026-09-08-精度修复)
与[同权重对照结果](../../docs/precision_fix_20260908.md)。

## 数据布局

`SPIKE_DATA_ROOT` 下需要四个 LeRobot 数据集，每个包含 `meta/tasks.jsonl`、episode 元数据、Parquet 与两路视频：

```text
libero_10_no_noops_1.0.0_lerobot/
libero_goal_no_noops_1.0.0_lerobot/
libero_object_no_noops_1.0.0_lerobot/
libero_spatial_no_noops_1.0.0_lerobot/
```

使用 `binary_open` 夹爪语义，`normalize_binary_gripper=false`。统计文件为 `configs/libero_all4_stats.json`，键为 `libero_all4_no_noops`。不要把其他夹爪编码的数据直接套进这套配方。

`configs/online_text_layout.json` 仅记录40条训练指令的 padding 布局，没有 embedding 缓存。保留 legacy 句子/位置协议，学生和教师采用 corrected batch/head mask。padding token 当前也会进入动作头，因此接口校正同时对齐有效行和 padding 行。

## 独立联合训练

完成根目录安装、权重与 `.env` 配置后运行：

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

## 评估

下面是单套完整评估，共10任务 × 每任务50次。四套分别运行共2,000局；流水线默认每任务5次、共200局；启动时传入 `--eval-trials 50` 可在训练后自动跑2,000局，两者不能混报。

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
