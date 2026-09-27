# 代码来源与发布范围（更新于 2026-09-23）

这次提交只修改独立发布仓库；不改已部署source_snapshot、训练参数、控制器或评测协议。没有重新训练、刷新EMA或重新评测已完成实验。

| 发布文件 / 功能 | 实际源码快照标识 |
| --- | --- |
| `full_state_action_head.py`、`shiftmax_action_head.py`；早期Shiftmax消融与Spike-Driven VLA的入口、FFN和state选择 | `fusionA_shiftmax_ann_state8_continuous_inputs_spike_ffn_20260916/source_snapshot` |
| `minimal_cross_wta_head.py` | `fusionA_base_minimal_crosswta_bsmax_20260918/source_snapshot`，不是已停止的ANN回退组 |
| `progressive_full_wta_head.py` | `fusionA_progressive_fullwta_t8_20260918/source_snapshot` |
| 文本SDSA路由 `wta_fusion.py`、`wta_operators.py` | `fusionA_no_wta_turbovla_base_20260916/source_snapshot` |
| `configuration.py`、`turbovla.py`、`trainer.py` | 将以上分支接入发布版，同时保留历史头类型、population-repair字段及默认值 |
| `train_action_head.py`、`configs/spike_driven_vla.json`、`configs/extensions/action_head_variants.json` | 主推 base 与历史扩展的可移植 fresh 配方；不包含服务器控制器 |
| `docs/results/libero_20260918.json` | 各组既有最终summary，去掉私人权重位置；在训进度为22:24快照 |
| `light_augmentation.py`、`libero_lerobot.py` 与 `train_mixed.py` 的训练专用增强接线 | `fusionA_six_lightaug_bs256_40k_20260922/source_snapshot` |

核心head和文本路由文件与来源逐字节比较；不用文件摘要替代源码检查。共享接口合并保持旧字段，避免用较窄快照覆盖历史兼容代码。当前渐进T8照实发布；讨论中的T4与WSD没有混入该版本。

所有原始运行日志、权重和source_snapshot保留在各自实验目录。这里只发布实现、配方、聚合结果和文档；没有发布训练数据、权重、GPU/PID控制脚本、依赖缓存、视频或凭据。

结果是单训练seed记录，CPU回归与参数检查只验证本次代码合并，不构成新的训练或评测成绩。minimal-cross的BSmax结果依赖自身step20全状态恢复，统一fresh入口不是该历史续训的替代品。

本次发布检查：12个配方均通过实际mixed-suite训练parser；完整3层hidden256/FFN2048动作头的CPU前向、反向、有限梯度检查通过；渐进alpha端点、共享模块及buffer严格恢复通过；SDSA静默行/掩码、结果分母、Python语法、JSON与文档相对链接检查通过。六个核心实现文件与相应快照逐字节一致。没有使用GPU或加载历史训练权重。

在安装了项目依赖的环境运行回归检查：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -p 'test_release_variants.py' -v
```
