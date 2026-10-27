# Motion Tracking 分阶段验证记录

日期：2026-10-27。对应 [算法审查](supermimic_algorithm_audit.md) 后的修复，运行说明见 [算法文档](motion_tracking_algorithm.md)。代码保留独立 `supermimic_rl`，未恢复 RSL-RL；command 保留 BeyondMimic 的推进顺序。

## 环境与数据

- Python 3.12、uv 锁定依赖、本地 Isaac Lab 3.0、Newton/MJWarp。
- NVIDIA GeForce RTX 3060 Laptop，6 GiB 显存。
- 本次实际数据：`../z_lab/src/z_lab/tasks/mimic/config/g1/motions/dance.npz`，50 Hz、xyzw、1457 帧。
- 集成测试在临时目录裁出 6/54 帧片段来检查边界，没有修改原 NPZ。
- 当前仓库根目录没有 `assets/motions/`，因此未声称验证了用户全部 LAFAN1/BONES。后续将现有 NPZ 放入该目录，或用 `--motion_dir` 指向实际目录即可。

## 阶段结果

| 阶段 | 检查内容 | 结果 |
|---|---|---|
| 环境 | reset/step、partial reset、终止状态和观测历史、接触摩擦、CoM 派生动力学 | 3 项真实集成测试通过 |
| Replay | 完整状态/元数据对应、终止 next state、短失败轨迹、覆盖不读未来、恢复不串 episode | 聚焦单元测试通过 |
| 可微任务模型 | USD FK 的 link 位姿和 CoM 速度、解析 reward/proprioception/goal、硬终止、状态梯度 | 真实仿真一致性测试通过 |
| 学习与恢复 | 固定真实窗口拟合下降、模型 action Jacobian、actor 跨模型梯度、冻结模块不变、真实参数更新、checkpoint 恢复 | 真实集成测试通过 |
| 独立 train | 完整网络、默认 smoke 25 次联合更新 | 通过，所有记录指标有限 |
| 独立 resume | 新进程加载第 25 轮，保留 replay，训练到第 26 轮 | 通过 |
| 独立 play | 加载策略文件，真实仿真运行 100 步 | 通过 |
| 独立 evaluation | 可用数据集逐动作从首帧运行，记录失败及误差 | 流程通过；短训策略在第 61 步失败 |
| 部分 evaluation | `--evaluate --steps 10` | 运行 10 步，`failed=false`、`partial=true`，没有误标为完整评估 |

## 聚焦检查与完整测试

```bash
# 第一阶段：后端、边界和随机化
SUPERMIMIC_TEST_MOTION_DIR=../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_tracking_env.py -k 'not differentiable and not world_fit' -x

# 第二阶段：replay、公式和可微任务模型
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_learning.py
SUPERMIMIC_TEST_MOTION_DIR=../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_tracking_env.py -k differentiable -x

# 第三阶段：真实窗口拟合、BPTT、冻结、恢复和实际训练更新
SUPERMIMIC_TEST_MOTION_DIR=../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_tracking_env.py -k world_fit -x

# 完整测试：包含全部 5 项真实集成测试
SUPERMIMIC_TEST_MOTION_DIR=../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

最终完整测试结果为 **11 passed，0 skipped**：6 项独立逻辑/注册检查和 5 项 Newton 集成检查。未设置动作目录时，5 项集成测试会 skip，不能将这种运行当作完整验收。

测试修复保留了已有物理与学习行为断言，更新了独立 runner 的接口；replay 测试增加完整状态及元数据同源、覆盖年龄、短轨迹和恢复边界检查。新增的回报检查区分自然截断补尾与真实失败不补尾，避免短片尾被错误视为困难。

## 默认 smoke 与新进程恢复

```bash
./supermimic.sh -t --profile smoke \
  --motion_dir ../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
  --log_dir logs/supermimic_rl/staged_final --headless

./supermimic.sh -t \
  --checkpoint logs/supermimic_rl/staged_final/checkpoint_final.pt \
  --iterations 26 \
  --motion_dir ../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
  --log_dir logs/supermimic_rl/staged_resume --headless
```

- 网络参数 **18032516**；encoder/decoder 和 dynamics 保持论文宽度。
- 16 环境，128 个向量步预采集，k=8、H=32；每次联合更新 10×8 个模型窗口、8 条想象轨迹、5×4 次 critic 更新。
- 第 25 轮共采集 3040 个环境 transition，即 `16*(128+62)`；UTD 余额为 0.5。第 26 轮再采集 3 个向量步，共 3088 个 transition，余额为 0，未重复 prefill。
- 恢复后 optimizer step：world=260、actor=26、critic=520；三个 scheduler 的 epoch 均为 26。
- 对比第 25/26 轮 checkpoint，world/actor/critic 的权重均有实际变化；本体归一化统计样本数为 3088。
- 第 25 轮模型损失约 1.574，第 26 轮约 1.381。它们来自变化中的 replay，不作为固定验证集的收敛指标；固定窗口测试另行要求拟合损失下降。
- 第 26 轮 actor 原始梯度范数约 2.606，随后按配置裁剪；所有日志指标有限。
- PyTorch 峰值显存约 **297.53 MiB**，不包含 Warp/模拟器；CPU replay 容量 **21472000 bytes**（约 20.48 MiB）。
- 恢复轮的采集/模型拟合/策略与 critic 阶段耗时约 2.56/0.54/4.16 秒，仅作为本机单轮诊断。

产物位于 git-ignored 的 `logs/supermimic_rl/staged_final/` 和 `staged_resume/`。训练 checkpoint 与独立 `checkpoint_final.policy.pt` 分开；推理读取后者，不加载 replay。配置、物理/数据 contract、内存记录和训练指标均随运行保留。

## 策略回放与评估

```bash
./supermimic.sh -p \
  --checkpoint logs/supermimic_rl/staged_resume/checkpoint_final.pt \
  --motion_dir ../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
  --steps 100 --headless

./supermimic.sh -p \
  --checkpoint logs/supermimic_rl/staged_resume/checkpoint_final.policy.pt \
  --motion_dir ../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
  --evaluate --output logs/supermimic_rl/staged_resume/evaluation.json --headless
```

完整评估结果：`dance.npz` 在第 61 步触发物理失败；`partial=false`，归一化回报约 **0.03003**，执行部分的平均 root 位置误差约 **0.1964 m**，对齐 body 位置误差约 **0.1072 m**。失败后的剩余 clip 长度计零回报。

这些结果证明流程能够训练、恢复和评估，不代表论文性能达标。当前评估保留 BeyondMimic 帧时序，归一化采用 `5*dt*L`；与使用 L−1 步的另一参考实现比较时需显式说明该差异。

## 尚未验收的范围

- 用户完整 LAFAN1/BONES 混合数据集及论文完整 BONES benchmark。
- paper 规模的 4096 环境、正式 batch 和 320000 次联合更新的长训结果。
- PhysX/OVPhysX、多 GPU 和实机部署。

这些范围不通过缩小正式配置或复用短训指标来替代。后续应先将实际 NPZ 接入并完成逐动作验证，再开始正式训练。

## 静态检查与构建

`ruff check`、`ruff format --check`（算法包、任务、入口和 tests）、Python 编译检查、`git diff --check` 均通过。`uv build` 成功生成 wheel 和源码包，wheel 包含全部 8 个 `supermimic_rl` Python 模块。
