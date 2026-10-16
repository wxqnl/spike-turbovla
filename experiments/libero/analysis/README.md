# v2 分析脚本

v2 的评测记录与诊断脚本。方法和配方见 [docs/v2_sgsa.md](../../../docs/v2_sgsa.md)。

## 运行方式

`launch/` 下的启动脚本按 GPU 分发进程。先把 `launch/env.example.sh` 复制成 `launch/env.sh`，并设置 `SPIKE_WORK_ROOT`；也可以用 `SPIKE_ENV` 指向别处的副本。工作目录布局写在 `env.example.sh` 开头。所有脚本使用与 base 相同的评测设置：seed 7、FP32/highest、relative 控制、chunk 12、open-loop 12、wait 10、256 像素渲染、OSMesa。

单独运行的分析脚本，参数和输出格式写在各自文件开头的说明里。

## 评测与失败分析

| 脚本 | 作用 |
| --- | --- |
| `launch/failure_runs.sh` + `failure_rollout.py` | 官方 LIBERO 评测（每个任务一个进程，50 个初始状态按顺序跑），同时逐步记录仿真器状态 |
| `failure_taxonomy.py` | 按记录给失败分类，并在相同初始状态上比较模型 |
| `placement_gain.py` | LIBERO-Spatial 上，物体是否放到了目标的实际位置 |

## 目标跟随

| 脚本 | 作用 |
| --- | --- |
| `launch/counterfactual_runs.sh` + `counterfactual_rollout.py` | LIBERO-Spatial 盘子平移反事实：在搬运开始和第一帧平移盘子，记录落点和开环探针 |
| `counterfactual_analysis.py` | 由反事实记录算跟随增益（落点位移 / 盘子位移）和各阶段的盘子位置可解码性 |
| `launch/reliance_runs.sh` + `reliance_policy.py` | 状态 / 视觉冲突测试：两者给出矛盾的目标时，动作跟随哪一个 |

## 动作头内部的读取

| 脚本 | 作用 |
| --- | --- |
| `head_probe.py`、`head_analysis.py`、`key_r2.py` | 盘子位置在动作头的哪一层丢失 |
| `cross_probe.py` | cross-attention 内部逐级（视觉 key 脉冲、value、Shiftmax 权重、context、context 脉冲）的盘子位置 |
| `attend_probe.py`、`attend_analysis.py` | 动作头 cross-attention 落在盘子、目标碗和夹爪上的比例 |
| `read_split_probe.py`、`split_ratio.py` | 读取变化拆成路由变化和内容变化 |
| `route_test_probe.py` | 不重训、推理时干预路由（冻结路由、邻域平均、精确或细化归一化、符号场等），用 `read_split_probe.py` 的探针测量 |
| `token_position_probe.py` | 动作头读到的视觉 token 内容里是否带有自身的图像位置 |

## v1 base 的诊断

| 脚本 | 作用 |
| --- | --- |
| `ckpt_routing_probe.py`、`head_routing_share.py`、`wta_ties.py` | 训练好的 base 的路由统计、按记忆分组的路由份额、top-k 平局 |
| `reliance_probe.py`、`state_precision_probe.py`、`proprio_quantization.py` | 状态依赖和状态编码精度 |
| `vision_pathway_probe.py`、`object_probe.py`、`render_scenes.py`、`depth_probe.py`、`sdt_quant_probe.py` | 脉冲视觉通路在哪里丢失物体位置（仿真器真值） |
| `demo_grasp_order.py` | LIBERO-Long 演示中先抓哪个物体 |

## LIBERO-plus

| 脚本 | 作用 |
| --- | --- |
| `launch/libero_plus_runs.sh` + `libero_plus_rollout.py` | LIBERO-plus（arXiv 2510.13626）按类别评测，每个扰动任务跑一次，逐步记录仿真器状态；可断点续跑 |
| `libero_plus_summary.py` | 按模型、套件、类别、难度汇总成功率，并给出配对计数 |
| `libero_plus/lplus_paired.py` | 布局扰动上两个模型的配对比较（移动物体 / 新增物体分开，精确 McNemar 检验） |
| `libero_plus_distractor.py`、`libero_plus/distractor_dose.py`、`libero_plus/distractor_debug.py` | 新增物体任务：失败类型、按新增物体数量的成功率、抓错了什么 |
| `libero_plus/layout_shift.py` | 布局扰动中原有物体相对 LIBERO 位置的偏移 |
