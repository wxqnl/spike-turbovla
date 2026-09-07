# spike-turbovla

基于 [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) 的逐模块脉冲化实验，当前面向 LIBERO。

当前发布代码使用 SDT-V3 视觉编码器、冻结的 SmoothSpike BERT，以及双向 SDSA 式 Spike Cross-Attention。融合层中的文本 Self-Attention、FFN、投影接口和动作头仍有 ANN 运算，因此当前模型属于 SNN–ANN 混合架构，不能称为全脉冲 VLA。

## 当前结构与训练范围

| 部分 | 实现 | 联合训练 |
| --- | --- | --- |
| 视觉编码器 | SDT-V3 19M，官方 SFA 实现 | 更新 |
| 文本编码器 | SmoothSpike BERT，T=4，在线推理 | 冻结 |
| 视觉 / 文本投影 | 连续投影；文本接口经过初始化校正 | 更新 |
| 双向 Cross 核心 | 二值 Q/K/V，PLIF，无 Softmax，T=4 | 更新 |
| 六层中的文本 Self-Attention / FFN | 原 ANN 模块 | 更新 |
| 状态投影与动作头 | ANN ACT decoder，12×7 动作块 | 更新 |

[结构边界](docs/architecture.md) · [训练与评估](experiments/libero/README.md) · [阶段结果](docs/results.md) · [第三方来源](third_party/README.md)

## 安装

使用 Python 3.10。先安装与机器 CUDA 兼容的 PyTorch；以下是当前实验使用的版本：

```bash
git clone https://github.com/wxqnl/spike-turbovla.git
cd spike-turbovla
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python scripts/setup_third_party.py
pip install -e ".[libero]"
```

下载脚本需要 Git 和 curl，只获取 SDT-V3 / SmoothSpike 的模型代码并应用本项目集成补丁，不下载权重。源码来自上游 main 分支；若上游变化导致补丁不兼容，脚本会停止，不会静默换模型。

另行安装 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 并完成其仿真器配置。当前无头评估使用 OSMesa。LeRobot 数据加载不需要 TensorFlow；只有使用 RLDS 数据时才安装 `.[rlds]`。

## 数据和预训练权重

数据、权重、日志不进入 Git。配置位置见 [.env.example](.env.example)。

- 四套 LIBERO no-noops 数据：spatial、object、goal、libero_10；当前训练使用 LeRobot 格式。
- [SDT-V3](https://github.com/BICLab/Spike-Driven-Transformer-V3)：19M 的 `V3_19.0M_1x4.pth`，必须匹配 `Efficient_Spiking_Transformer_l`。加载器检查实际键名和形状。
- [SmoothSpike](https://github.com/CayleyZ/SmoothSpike)：BERT-base T=4 预训练权重及 tokenizer/config。使用融合 H2/H3 后的权重，保留 H1。
- [TurboVLA](https://huggingface.co/H-EmbodVis/TurboVLA)：公开教师、DINOv3 图像处理配置和 BERT。
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)：`groundingdino_swint_ogc.pth`。

若下载的是未融合的 SmoothSpike 权重：

```bash
python -m third_party.SmoothSpike.convertor \
  --input /path/to/smoothspike-bert-base \
  --output /path/to/smoothspike-bert-base-fused
```

## 启动当前联合训练

```bash
cp .env.example .env
# 编辑 .env 后：
set -a
source .env
set +a
python -m experiments.libero.run_cross_joint_stage
```

固定配方为三卡、每卡 batch 128、全局 batch 384、70,000 优化步、seed 42。流水线依次完成训练指令上的文本接口校正、真实数据梯度检查、联合训练、最终 EMA checkpoint 的四套 LIBERO 评估。它不会抢占已用 GPU，也不会覆盖既有运行目录或自动重试失败训练。更换实验使用新的 `SPIKE_RUN_NAME`。

学生从公开预训练组件与新动作头初始化，优化器从零开始；不继承本项目之前训好的 LIBERO 策略。冻结 SmoothSpike，其余上述模块使用同一 LIBERO 动作损失联合更新。教师仅参与训练。

## 目录

```text
turbovla/                 模型、数据加载、训练、推理
experiments/libero/       联合训练、接口校正、评估入口及配置
scripts/                  第三方模型源码准备
docs/                     结构边界、实验协议与阶段结果
third_party/              运行依赖、来源说明、集成补丁与许可证
```

上游 RoboTwin 相关代码保留供参考，本项目尚未验证其脉冲版本。当前 GPU 实现使用稠密张量模拟脉冲，不代表已获得实测加速或节能收益。根目录许可证沿用 TurboVLA；第三方组件适用各自上游条款，见来源说明。
