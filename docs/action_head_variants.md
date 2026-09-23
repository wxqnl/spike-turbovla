# LIBERO 变体：代码、配方与复现

更新：2026-09-23。结果见[根README](../README.md)与[结果JSON](results/libero_20260918.json)。以下命令仅用于新建实验，不操作已有训练或评测。

[按得分查对应配置与代码](score_configurations.md)

主推架构请先看 [Spike-Driven VLA](spike_driven.md)。

## 统一入口

入口为 `python -m experiments.libero.train_action_head --variant NAME`，读取[配方JSON](../experiments/libero/configs/action_head_variants.json)。先用 `--dry-run` 查看展开参数；正式训练要求 `--approve_training`、匹配的torchrun进程数，以及代码目录外尚不存在的输出目录。它不自动运行EMA刷新或评测。

| NAME | 对应实验 | rank × batch | steps / warmup | 评测记录 |
| --- | --- | --- | --- | --- |
| `a_base` | 新A，ANN头 | 4×32 | 80000 / 10000步 | 四套92.35% |
| `a_no_wta` | A-no-WTA，六层文本改SDSA式点积 | 4×32 | 80000 / 10000步 | 四套90.15% |
| `perdim` | 逐维WTA，8-token | 2×192 | 26667 / 1280000样本 | 用户取消评测 |
| `multithreshold` | 多阈值WTA，8-token | 2×192 | 同上 | 用户取消评测 |
| `shiftmax_perdim` | ① 逐维PLIF＋入口PLIF＋spike FFN | 4×96 | 同上 | 四套83.65% |
| `shiftmax_continuous_inputs` | ② 连续attention入口＋spike FFN | 4×96 | 同上 | 四套77.45% |
| `shiftmax_continuous_inputs_ann_ffn` | ③ 连续attention入口＋ANN FFN | 4×96 | 同上 | 四套82.95% |
| `shiftmax_prelif_ann_ffn` | ④ 入口PLIF＋ANN FFN | 4×96 | 同上 | 仅Long66.6% |
| `shiftmax_projectfirst` | ⑤ 状态先投影后PLIF＋入口PLIF＋spike FFN | 4×96 | 同上 | 仅Long56.6% |
| `spike_driven` | Spike-Driven VLA，原始未增强配方 | 4×96 | 26667 / 3334步 | 四套89.65% |
| `spike_driven_lightaug` | Spike-Driven VLA，主推轻量RGB增强配方 | 8×32 | 40000 / 5000步 | 四套92.05% |
| `minimal_cross_wta` | A形状，只改action cross内部；原小batch前段 | 4×32 | 80000 / 10000步 | 被BSmax续训替代；不是当前BS96配方 |
| `progressive_fullwta_t8` | 共享参数渐进全头WTA，2-token | 4×80 | 32000 / 1280000样本 | 未完成 |

①—⑤与Spike-Driven VLA均为8个逐坐标state token、head T4、3层decoder/hidden256/heads8/FFN2048/readout256→512→512→7、horizon12。A、A-no-WTA和minimal-cross为A原ANN 2-token状态MLP。最初两种WTA也是8-token/T4，但前向为二值Top3/Top8，Shiftmax系列保留连续V。

上游共用SDT-V3、SmoothSpike T4、脉冲投影和spike_text_only融合；除A-no-WTA文本路由外，六层文本Top3 WTA和SDSA Cross保持不变。文本骨干冻结，其余七组联合训练；固定结构不等于冻结上游权重。

## 初始化与优化

Fresh配方使用公共组件和同方案公共文本投影初始化，不加载训练好的A、其他变体或容量探测权重。数据为原四套no-noops LeRobot训练集；action L1 + 0.1 raw featureKD + 0.1 raw actionKD，textKD=0。

固定seed42、FP32/highest、CUDA与cuDNN TF32关闭；AdamW(0.9,0.95)、eps1e-8、weight decay1e-10、clip1、EMA0.999。视觉LR1e-5、其余5e-5，不随batch放大。warmup从峰值约0.1倍线性升到峰值，之后恒定，min_lr_ratio=1。样本warmup沿用128样本前瞻项。WSD与真正cosine LR衰减均未实施。

A、A-no-WTA、minimal-cross、渐进组及`spike_driven_lightaug`关闭vision activation checkpointing；①—⑤、`spike_driven`及两种逐维WTA开启。上游Cross开启、projection关闭；spike head保留重计算支持。原ANN头与minimal-cross继承ANN执行路径，没有额外head重计算包装。

## 新建实验示例

先按[LIBERO安装与公共初始化说明](../experiments/libero/README.md#best-a-reproduce)准备依赖、数据、公共权重、`.env` 与 `SPIKE_TEXT_INIT_DIR`。从自己的代码检出配置 `PYTHONPATH`，输出放在代码目录外。下例选择⑤；其他版本按表替换NAME、world-size、batch和steps。

```bash
export SPIKE_CODE_ROOT="$PWD"
set -a
source "$SPIKE_CODE_ROOT/.env"
set +a
export PYTHONPATH="$SPIKE_CODE_ROOT:$SPIKE_CODE_ROOT/third_party/vla_adapter:$SPIKE_CODE_ROOT/third_party/starvla_runtime"
unset NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
export SPIKE_HEAD_RUN="$SPIKE_RUN_ROOT/shiftmax_projectfirst_new"

python -m experiments.libero.train_action_head \
  --variant shiftmax_projectfirst --world-size 4 --dry-run \
  --run_dir "$SPIKE_HEAD_RUN" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 26667 --batch_size 96 --grad_accum_steps 1

# 仅在这四张卡已确认空闲、输出目录尚不存在时执行
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 --module experiments.libero.train_action_head \
  --variant shiftmax_projectfirst --run_dir "$SPIKE_HEAD_RUN" \
  --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 26667 --batch_size 96 --grad_accum_steps 1 \
  --resume_mode none --approve_training
```

统一入口仅发布fresh配方，明确拒绝resume；恢复已有实验必须使用其原控制器或经过核对的底层训练命令。以上示例不是服务器队列命令，不会替代现有控制器。

<a id="minimal-cross-bsmax"></a>

## minimal-cross BSmax的历史分段配方

当前在训组并非上表fresh 4×32运行到80000：用户在自身step20保存后改为4×96，从这个唯一前段checkpoint恢复model、optimizer、scheduler和EMA。未落盘的小batch步数未计入续训预算，数据迭代器/RNG不是bitwise延续。

- 前段：20更新×global128=2560样本。
- 后段：26660更新×global384；最终global_step=26680，总计10240000样本。
- scheduler：warmup_steps=3347、lr_warmup_samples=1280000、lr_sample_offset=2560、lr_step_offset=20、lr_reference_batch=128、lr_schedule_steps=26680、min_lr_ratio=1。
- 架构与学习率未改；vision重计算仍关闭。96为当时真实完整模型容量探测选出的安全batch，不承诺其他机器可用同batch。
- checkpoint前缀为 `spike_a_minimal_crosswta_bsmax`，保存step40、每1000与最终；step20链接仅是自身前段恢复点。

本次收录该头的真实实现与上述恢复协议，不把BS96运行伪装为fresh 4×96，也不提供会自动复制私人断点或启动旧控制器的脚本。代码和公共初始化足够新建独立实验，但复现这次历史轨迹还需自己的前段完整断点与明确的全状态恢复流程。

<a id="progressive-fullwta"></a>

## 渐进全头：日程与T

发布的是目前实际运行的 `progressive_fullwta_t8`：上游T4、action head T8、2个state token。上游先聚合为连续memory，head内部独立积分，不把T4张量直接解释成T8。T4替换尚未实施。

同一套原Linear、packed QKV、embedding和位置参数同时服务ANN与spike计算，没有阶段切换时拷权重。ANN端点保留A原forward、激活与dropout；spike端点覆盖state、self/cross输入及QKV/context、FFN和readout，使用signed BipolarPLIF及selfTop3/crossTop8 WTA。每次观测重置神经元状态，最终为tanh(mean_T currents)。

p=completed_updates/32000，p≤0.8时alpha=0，0.8<p<0.85时半余弦升到1，之后alpha=1。输出为(1−alpha)×ANN_actions+alpha×Spike_actions；两个端点分别跳过不用的分支。前25600步ANN、过渡1600步、最后4800步纯spike，额外保存25600/27200。alpha和更新计数作为buffer进入raw与EMA checkpoint；恢复按global_step重算。alpha=0时spike专用参数没有梯度是设计行为。

这不是LR衰减。没有按loss提前转换、早停或自动选择T；后续改T或LR应另记实验版本。

## EMA与评测

每组成功完成训练并保存最终checkpoint后，用自己的训练数据128×16刷新EMA BN，optimizer更新为0。用[现有EMA与评测入口](../experiments/libero/README.md#best-a-eval)，替换本组checkpoint前缀与最终步数；不借用其他组权重、不重做已完成刷新。渐进组最终必须global_step32000、alpha1、head T8。

新A、A-no-WTA、①②③及Spike-Driven VLA均已有四套各500的完整结果。④⑤的已发布历史结果仅Long500；该口径的评测命令只选 `--task_suite_name libero_10 --num_trials_per_task 50`，10任务固定初态0—49。其余协议为seed7、FP32/highest、TF32off、legacy/corrected、relative、chunk/open-loop12、wait10、env256、binary gripper不归一化、OSMesa、不录视频。

Long-only不能产生四套overall。旧逐维WTA、多阈值WTA评测已取消，旧Deep已停止，错误的minimal-cross ANN回退组被替代；本次发布不重启它们。

## 代码与验证范围

[来源记录](code_provenance_20260918.md)列出实际训练快照与发布文件。核心head实现直接来自这些快照，共享配置、构造器、parser和训练日程进行了合并，保留旧SDSA population-repair字段及默认值。

CPU回归测试覆盖真实完整action head、公开配方的实际parser、渐进端点和checkpoint buffer；它不代替GPU容量测试、全模型训练或rollout，也不承诺发布合并后的新运行bitwise复现历史成绩。训练数据、私人权重、服务器编排与凭据不发布。
