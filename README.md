# spike-turbovla

基于 [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) 的 LIBERO 逐模块脉冲化实验。vision encoder、text encoder 和 Cross-Attention 核心可以独立选择。

## 重要更新：2026-09-08 精度修复

旧训练器使用 `float32_matmul_precision=high`，文本校准和评估却默认 `highest`。
两边都叫 FP32，并不代表矩阵乘法的内部精度相同；这会明显改变 SmoothSpike 输出。

先保存本地改动，再更新代码：`git pull --ff-only origin main`。

- **新训练**：所有预设显式使用 `highest`；checkpoint 保存 `numerics`，训练后的评估自动读取。
- **旧 Cross 联合训练权重**：如果来自旧版 `high` 训练器，评估必须加
  `--precision fp32 --float32_matmul_precision high`。仅更新代码、保留 `auto` 不会自动纠正旧权重。
- **后处理校准得到的旧 SmoothSpike 权重**：沿用校准时的 `highest`，使用
  `--checkpoint_weights raw --precision fp32 --float32_matmul_precision highest`，不要一律改成 `high`。
- **含新元数据的 checkpoint**：使用 `--precision fp32 --float32_matmul_precision auto`。
  缺少元数据时 `auto` 保留旧评估的 `highest` 并发出警告；其他旧实验应查实际训练/校准配置后明确选择。

同一份 Cross50k EMA、相同200局，训练/评估精度对齐后由116/200（58.0%）恢复到185/200（92.5%）。
这是200局诊断结果，完整2,000局正在重评，不能把92.5%当作完整benchmark分数。
[数值检查与逐套对照](docs/precision_fix_20260908.md)

| 预设 `--preset` | Vision | Text（冻结） | Cross 核心 |
| --- | --- | --- | --- |
| `ann` | DINOv3 ViT-B/16 | BERT-base-uncased | ANN |
| `vision` | SDT-V3 19M | BERT-base-uncased | ANN |
| `vision_text` | SDT-V3 19M | SmoothSpike BERT，T=4 | ANN |
| `vision_text_cross`（默认） | SDT-V3 19M | SmoothSpike BERT，T=4 | Spike SDSA，T=4 |

这些预设用于独立训练消融，不会加载上一阶段训好的策略。即使三个开关全部开启，融合层中的文本 Self-Attention、FFN、投影接口和动作头仍有 ANN 运算，当前仍是 SNN–ANN 混合架构。

[结构边界](docs/architecture.md) · [评估与数据细节](experiments/libero/README.md) · [历史结果与协议](docs/results.md) · [第三方来源](third_party/README.md)

## 1. 安装

使用 Python 3.10。下面是当前实验的 PyTorch 版本；按机器驱动选择兼容的 CUDA wheel：

```bash
git clone https://github.com/wxqnl/spike-turbovla.git
cd spike-turbovla
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python scripts/setup_third_party.py
pip install -e ".[libero]"
```

`setup_third_party.py` 需要 Git、curl，获取 SDT-V3 / SmoothSpike 源码并应用集成补丁，不下载模型权重。源码来自上游 main；上游变更导致补丁不兼容时会停止。训练会保存实际使用的源码快照。

另外安装 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，完成其 assets、BDDL、init states 和仿真依赖配置。当前无头评估使用 OSMesa；系统需有 OSMesa 与 FFmpeg。LeRobot 数据入口不需要 TensorFlow。

## 2. 配置工作目录

以下命令都在仓库根目录执行，使用 bash。代码与数据 / 权重 / 运行输出分开放：

```bash
cp .env.example .env
# 编辑 .env，至少把 SPIKE_WORK_ROOT 改成实际工作目录
set -a
source .env
set +a
```

`.env` 不会自动加载。默认路径如下，各路径均可通过 [.env.example](.env.example) 中的变量覆盖：

```text
$SPIKE_WORK_ROOT/
├── pretrained/
│   ├── TurboVLA/
│   │   ├── backbones/bert-base-uncased/
│   │   ├── backbones/dinov3-vitb16/
│   │   └── checkpoints/libero/turbovla_libero.pth
│   ├── SDT-V3/V3_19.0M_1x4.pth
│   ├── SmoothSpike/smoothspike-bert-base-fused/
│   └── GroundingDINO/groundingdino_swint_ogc.pth
├── datasets/libero/                 四套 LeRobot v2.1 no-noops 数据
├── benchmarks/LIBERO/
└── experiments/spike-turbovla/libero/
```

`SPIKE_RUN_ROOT` 必须在代码目录外。权重、数据、运行输出和 `.env` 均不上传 Git。

## 3. 模型在哪里下载

| 模型 / 文件 | 下载来源 | 哪些组合需要 |
| --- | --- | --- |
| BERT-base-uncased | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | 全部；ANN 学生、教师及文本校正参考 |
| DINOv3 ViT-B/16 LVD-1689M | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 全部；ANN 学生或训练教师及图像处理配置 |
| TurboVLA LIBERO 教师 | [H-EmbodVis/TurboVLA 的 LIBERO 权重](https://huggingface.co/H-EmbodVis/TurboVLA/tree/main/checkpoints/libero) | 全部预设，训练时蒸馏；推理不需要教师 |
| GroundingDINO Swin-T OGC | [官方 release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) | 全部；初始化交互层及文本投影 |
| SDT-V3 19M，`V3_19.0M_1x4.pth` | [AdaS 作者提供的预训练权重镜像](https://github.com/CayleyZ/AdaS/releases/tag/segmentation-sdtv3-19m-pretrained) | `--vision sdtv3_19m` |
| SmoothSpike BERT-base | [作者 ModelScope 仓库](https://modelscope.cn/models/kailai1104/SmoothSpike)，[tokenizer/config](https://github.com/CayleyZ/SmoothSpike/tree/main/tokenizer_files) | `--text smoothspike_bert`，下载后需融合 H2/H3 |

### BERT、DINOv3、教师和交互层初始化

TurboVLA 权重仓库不附带 BERT / DINOv3 模型目录，三者要分别下载。DINOv3 是 gated 模型：先在其模型页申请访问并接受条款，再用有权限的 Hugging Face 账号登录：

```bash
hf auth login
hf download google-bert/bert-base-uncased \
  --include config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors \
  --local-dir "$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/bert-base-uncased"

hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --include config.json preprocessor_config.json model.safetensors LICENSE.md \
  --local-dir "$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/dinov3-vitb16"

hf download H-EmbodVis/TurboVLA checkpoints/libero/turbovla_libero.pth \
  --local-dir "$SPIKE_WORK_ROOT/pretrained/TurboVLA"

mkdir -p "$SPIKE_WORK_ROOT/pretrained/GroundingDINO"
curl -fL --retry 3 \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "$SPIKE_WORK_ROOT/pretrained/GroundingDINO/groundingdino_swint_ogc.pth"
```

只需要 GroundingDINO 的权重文件，无需安装它的完整检测工程。以上命令使用默认目录；如果覆盖了 `SPIKE_*_PATH`，下载位置也要对应修改。

### SDT-V3 19M

本项目加载 `Efficient_Spiking_Transformer_l`，去掉分类头后有18,626,504个骨干参数，输出360通道、14×14特征图。不要用10M权重替换。

```bash
mkdir -p "$SPIKE_WORK_ROOT/pretrained/SDT-V3"
curl -fL --retry 3 \
  https://github.com/CayleyZ/AdaS/releases/download/segmentation-sdtv3-19m-pretrained/V3_19.0M_1x4.pth \
  -o "$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth"
```

已下载核对：该镜像文件与本项目当前使用的同门提供权重逐字节一致，并通过19M模型的严格加载。它由 AdaS 作者托管，不是 BICLab 官方托管。原始模型来源仍是 [SDT-V3](https://github.com/BICLab/Spike-Driven-Transformer-V3)。截至2026-09-07，[官方下载说明](https://github.com/BICLab/Spike-Driven-Transformer-V3/blob/main/SDT_V3/Classification/Model_Base/Train_Base.md) 的19M链接页面显示10M文件名，因此这里提供了明确的19M镜像；加载器仍会检查实际架构、键名和形状。

### SmoothSpike BERT：下载 → 补齐 tokenizer → 融合

作者发布的是未融合权重。本项目使用 `spikingbert_rot_inf.py`，必须先折叠逐层 H2/H3，并保留全局 H1。不要直接加载未融合权重，也不要替换成普通 Transformers 的 `BertModel`。[作者说明](https://github.com/CayleyZ/SmoothSpike#pretrained-weights)

```bash
pip install modelscope
modelscope download --model kailai1104/SmoothSpike \
  checkpoints/smoothspike-bert-base/model.safetensors \
  --local_dir "$SPIKE_WORK_ROOT/pretrained/SmoothSpike"

SPIKE_SMOOTH_RAW="$SPIKE_WORK_ROOT/pretrained/SmoothSpike/checkpoints/smoothspike-bert-base"
for SPIKE_TOKENIZER_FILE in config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
  curl -fL --retry 3 \
    "https://raw.githubusercontent.com/CayleyZ/SmoothSpike/main/tokenizer_files/$SPIKE_TOKENIZER_FILE" \
    -o "$SPIKE_SMOOTH_RAW/$SPIKE_TOKENIZER_FILE"
done

python -m third_party.SmoothSpike.convertor \
  --input "$SPIKE_SMOOTH_RAW" \
  --config "$SPIKE_SMOOTH_RAW" \
  --output "$SPIKE_WORK_ROOT/pretrained/SmoothSpike/smoothspike-bert-base-fused"
```

转换在 CPU 上完成。`--config` 不能省略，否则上游转换器会去找当前目录下的 `tokenizer_files`。转换后目录应包含 `model.safetensors`、上述5个 tokenizer/config 文件及 `fusion_report.json`；报告应显示移除24个逐层旋转参数、保留 `bert.H1`，无剩余 H2/H3。

## 4. 数据准备

在 `$SPIKE_DATA_ROOT`（默认 `$SPIKE_WORK_ROOT/datasets/libero`）下准备：

```text
libero_spatial_no_noops_1.0.0_lerobot/
libero_object_no_noops_1.0.0_lerobot/
libero_goal_no_noops_1.0.0_lerobot/
libero_10_no_noops_1.0.0_lerobot/
```

每套需要 LeRobot v2.1 的 `meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、Parquet 和双视角视频。夹爪语义为 `binary_open`，不再归一化二值夹爪。直接把原始 HDF5 / RLDS 改目录名不能变成这个格式；仓库不附带训练数据或自动转换脚本。数据字段与独立评估见 [LIBERO 说明](experiments/libero/README.md)。已有 RLDS 数据可安装 `.[rlds]` 后使用底层训练入口，不能直接交给本节的 LeRobot 预设。

## 5. 各组合如何启动训练

先完成前面的安装、下载、数据准备，并加载 `.env`。每条命令是一场独立实验；在同三张卡上选一条启动，不要同时执行。

```bash
# 只替换视觉：SDT-V3 + 冻结 ANN BERT + ANN Cross
python -m experiments.libero.run_experiment \
  --preset vision --run-name vision_s42

# 视觉 + 文本：SDT-V3 + 冻结 SmoothSpike + ANN Cross
python -m experiments.libero.run_experiment \
  --preset vision_text --run-name vision_text_s42

# 再替换 Cross 核心：SDT-V3 + 冻结 SmoothSpike + Spike SDSA Cross
python -m experiments.libero.run_experiment \
  --preset vision_text_cross --run-name vision_text_cross_s42

# 新训 ANN 对照，不是直接评估公开 TurboVLA checkpoint
python -m experiments.libero.run_experiment \
  --preset ann --run-name ann_s42
```

也可单独覆盖模块，例如保留 ANN BERT、只替换 vision 和 Cross：

```bash
python -m experiments.libero.run_experiment \
  --preset vision --cross spike_sdsa --run-name vision_cross_s42

# 只替换 text，保留 DINOv3 和 ANN Cross
python -m experiments.libero.run_experiment \
  --preset ann --text smoothspike_bert --run-name text_s42
```

独立开关为 `--vision {dinov3,sdtv3_19m}`、`--text {bert,smoothspike_bert}`、`--cross {ann,spike_sdsa}`。显式 `--preset` 会清掉环境中的旧模块覆盖，再应用本次命令的模块开关，防止上一场实验的设置混入。

先看实际展开的配置与训练命令，不创建输出、不加载模型、不占 GPU：

```bash
python -m experiments.libero.run_experiment --preset vision_text_cross --dry-run
python -m experiments.libero.run_experiment --help
```

安装后也可用 `spike-turbovla-experiment` 替代 `python -m experiments.libero.run_experiment`。旧入口 `python -m experiments.libero.run_cross_joint_stage` 保留，读取同一套 `SPIKE_*` 环境变量。

### 训练量、显存与评估量

默认三张卡 `4,5,6`、每卡128、梯度累积1、全局 batch 384、70,000优化步、seed 42。所有预设使用相同损失：

```text
LIBERO 专家动作 L1 + 0.1 × 教师视觉特征 MSE + 0.1 × 教师动作 MSE
```

视觉、投影、六层交互、动作头共同更新；BERT / SmoothSpike 文本骨干冻结，文本投影可训练。DINOv3 中未参与前向的 mask token 也保持冻结。学生从公开预训练组件初始化，动作头和优化器从零开始。教师只提供训练期目标，不进入推理主干。

SmoothSpike 组合会先用40条训练指令对齐公开 BERT + GroundingDINO 的文本投影接口，不读取旧策略或评估轨迹；校正后的投影随后联合训练。ANN BERT 组合跳过此步骤。流水线接着做8个真实训练帧的 BC 梯度检查、保存源码快照、开始训练，成功结束后自动评估最终 EMA 权重。

不同组合显存占用不同。如果128放不下，可以减小 microbatch 并增加梯度累积，保持全局 batch 384：

```bash
python -m experiments.libero.run_experiment \
  --preset vision_text_cross --run-name cross_bs64_acc2 \
  --gpus 4,5,6 --batch-size 64 --grad-accum-steps 2 \
  --max-steps 70000 --seed 42 --eval-trials 50
```

全局 batch = 每卡 batch × 3 × 梯度累积。SDT-V3 含 BatchNorm，改变 microbatch 即使保持全局 batch，也不是严格数值等价。默认学生 FP32、视觉学习率1e-5、其余5e-5、warmup 3,334步、EMA 0.999；修改 `--max-steps` 不会自动缩短 warmup。DINOv3 学生使用256像素，SDT-V3 按预训练规格使用224像素。

`--eval-trials 5` 是默认的200局快速评估，`--eval-trials 50` 是四套共2,000局完整评估。固定评估 seed 7；`--seed` 仅改变训练种子。流水线目前要求三张卡，不会抢占已有进程。已有运行目录或失败记录时拒绝覆盖 / 自动重试，重新实验请换 `--run-name`。

输出位于 `$SPIKE_RUN_ROOT`：

- `<run-name>/initialization.json`：实际模块组合、初始化来源与训练量。
- `<run-name>/train.log`、`joint_training_audit.json`：损失与首步联合更新记录。
- `<run-name>/checkpoints/spike_turbovla_70000.pth`：默认最终权重；改步数时文件名同步变化。
- `<run-name>_control/status.json`：流水线阶段或报错。
- `<run-name>_eval_200/summary.json` 或 `_eval_2000/summary.json`：完整结束后的汇总。

八种模块组合已通过启动参数解析检查；新增消融组合尚未逐一完成训练和 rollout，不能套用历史得分。尤其历史 `vision + SmoothSpike` 的校正结果来自已训练视觉策略的后处理，不等于本次 `vision_text` 独立联合训练结果。实验来源与分数分开记录在 [阶段结果](docs/results.md)。

## 目录与实现边界

```text
turbovla/                 模型、数据加载、训练、推理
experiments/libero/       可选组合、训练流水线、文本校正、评估及配置
scripts/                  第三方模型源码准备
docs/                     架构与阶段结果
third_party/              运行依赖、集成补丁与许可证
```

当前 Spike Cross 核心使用二值 Q/K/V、PLIF 和无 Softmax 的 SDSA 式交互。文本 Self-Attention / FFN 与 ANN ACT 动作解码器尚未脉冲化。GPU 上使用稠密张量模拟脉冲，不能据此声称实测节能或加速。上游 RoboTwin 代码保留供参考，尚未验证脉冲配方。根目录沿用 TurboVLA 代码许可证，模型权重与第三方代码需遵守各自条款。
