# v2：Spatially Grounded Spike Attention（SGSA）

v2 改的是 spike-driven-vla 脉冲动作头里 cross-attention 的读取方式，让动作跟着目标的位置走。动作头仍然是脉冲驱动的。

## 问题

base 的典型失败是"差一点"：抓对了物体，却放在目标旁边几厘米处。原因在动作头读取视觉特征的那一步：目标位置在视觉特征里还在，读完就丢了。

- **归一化跳变**：Shiftmax 把分母取到大于行和的下一个 2 的整数次幂。行和一跨过 2 的幂，整行权重就翻倍或减半，跟目标怎么动无关。
- **求和抵消**：加权求和把各个 token 的内容加在一起。目标从一个 token 移到相邻 token 时，一边减少的正好被另一边补上，位置信息被抵消。

## 方法

SGSA 有两个部件，都不增加可学习参数：

| 部件 | 做法 | 配方开关 |
| --- | --- | --- |
| 细化归一化 | 分母的指数取到 1/16 而不是整数，行和跨过台阶时整行权重最多变 4.4%；实现为 16 项常数表加移位 | `ifr_norm_bits: 4` |
| 符号场 | 每个 head 一半的视觉 value 通道乘上所在 patch 的固定 ±1 空间图案（沿行、列方向的方波，4 个尺度 × 2 个相位），读取结果因此带上内容所在的位置；只翻符号 | `ifr_sign_field: true` |

两个部件加在 I5 的动作头上。I5 的 cross-attention 在时间步上累积脉冲匹配再路由（`ifr_integrate`），分数右移 2 位（`ifr_shift: 2`），视觉 value 加上 patch 位置值（`ifr_locate`），视觉 token 带可学习的 patch 位置编码（`position_embedding: learned_patch`）。

实现：[`IFRCrossAttention`、`ShiftmaxRouting`](../turbovla/models/shiftmax_action_head.py)。所有开关关闭时就是原来的 Shiftmax 头。

## 配方

配方在 [ifr.json](../experiments/libero/configs/extensions/ifr.json) 和 [vision_attribution.json](../experiments/libero/configs/extensions/vision_attribution.json)。

| 配方 | 内容 | 训练量 |
| --- | --- | --- |
| `f7_sdtv3_kd_ifr_signfield` | v2 最终模型：I5 头 + 细化归一化 + 符号场 | v1 标准配方：4 卡 × 64（全局 256）× 40k，KD |
| `i5_sdtv3_nokd_ifr_locate` | 消融起点，两个部件都不加 | 快速配方：4 卡 × 80（全局 320）× 32k，无 KD |
| `i8_sdtv3_nokd_ifr_finenorm` | 只加细化归一化 | 快速配方 |
| `a1_sdtv3_nokd_ifr_signonly` | 只加符号场 | 快速配方 |
| `i7_sdtv3_nokd_ifr_signfield` | 两个部件都加 | 快速配方 |
| `d0_sdtv3_nokd` | 快速配方下的 base 动作头 | 快速配方 |
| `c0_sdtv3_nokd_softmax_cross` | 对照：cross-attention 换成连续 softmax（不是脉冲驱动） | 快速配方 |

快速配方与 v1 标准配方的样本量相同（10.24M），学习率按 sqrt(320/256) 放大。其余消融（I0 到 I4、I6、I9、F9）见 `ifr.json`，视觉编码器归因（D1 到 D3）见 [VISION_ATTRIBUTION.md](../experiments/libero/VISION_ATTRIBUTION.md)。

## 训练

环境、预训练组件和文本投影初始化与 base 相同（[README](../README.md) 第 1 到 4 节）。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module experiments.libero.train_action_head --variant f7_sdtv3_kd_ifr_signfield \
  --run_dir "$SPIKE_RUN_ROOT/f7_sdtv3_kd_ifr_signfield" \
  --vision_checkpoint "$SPIKE_VISION_CHECKPOINT" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 40000 --batch_size 64 --grad_accum_steps 1 --resume_mode none --approve_training
```

快速配方改成 `--max_steps 32000 --batch_size 80`。入口会核对进程数、每卡 batch、梯度累积和步数是否与配方一致。加 `--world-size 4 --dry-run` 可以先查看展开后的训练参数。

## 评测

1. 用训练数据刷新 EMA 的 BN 统计（128 × 16 帧，不更新参数）：

   ```bash
   python -m experiments.libero.refresh_wta_norms --checkpoint <run>/checkpoints/f7_sdtv3_kd_ifr_signfield_40000.pth \
     --output <ema>.pth --batches 128 --batch_size 16
   ```

2. 四个套件按官方协议评测（seed 7，每套 10 个任务 × 50 个初始状态），同时记录每一步的仿真器状态：

   ```bash
   bash experiments/libero/analysis/launch/failure_runs.sh <out> 16 \
     libero_spatial,libero_object,libero_goal,libero_10 f7=<ema>.pth:ema
   ```

3. 跟随增益、冲突测试和 LIBERO-plus 见 [分析脚本](../experiments/libero/analysis/README.md)。

## 结果

两个模型在同一台机器、同一评测协议下评测，各用自己刷新过的 EMA：

| 模型 | Spatial | Object | Goal | Long | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 base | 472 | 482 | 483 | 366 | 1803/2000（90.2%） |
| v2（F7） | 461 | 490 | 472 | 389 | 1812/2000（90.6%） |

v1 base 的 1841/2000 来自它最初的评测环境；上表把同一个 checkpoint 和 F7 放在同一环境里重测。

目标跟随（LIBERO-Spatial，10 个任务 × 10 个初始状态）：

| 指标 | v1 base | v2（F7） |
| --- | ---: | ---: |
| 跟随增益：搬运开始时平移盘子（落点位移 / 盘子位移，理想值 1） | 0.24 | 0.34 |
| 跟随增益：第一帧平移盘子 | 0.22 | 0.35 |
| 冲突测试：状态与视觉给出矛盾目标时，跟随状态的比例 | 0.62 | 0.31 |
