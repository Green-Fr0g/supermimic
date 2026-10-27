# SuperMimic：motion tracking 复现计划

当前任务端实现和运行接口见 [motion_tracking_task.md](motion_tracking_task.md)。任务端和算法端已接入，世界模型、BPTT actor、λ-return critic 及运行方式见 [motion_tracking_algorithm.md](motion_tracking_algorithm.md)。本文保留复现目标与验收边界；正式数据集训练和论文性能验收仍待执行。

## 1. 目标与复现边界

实现论文 §3、附录 A/C/D 中的完整训练流程：

**仿真采集 → 拟合可微世界模型 → 想象 rollout → BPTT 更新 actor → 拟合 critic。**

第一版使用一个策略混合跟踪已有全部动作：

| 数据来源 | 数量 | 原始帧率 | 总时长 |
|---|---:|---:|---:|
| LAFAN1（目录名 `lanfan1`） | 40 | 30 FPS | 约 2.45 小时 |
| BONES | 17 | 120 FPS | 约 2.47 分钟 |

- 使用 Isaac Lab 3.0.0、G1 29 自由度和 Newton/MJWarp；训练不启用可视化。
- SuperMimic 在 BeyondMimic 的动作跟踪任务基础上修改观测、采样和训练方法。本项目复用 `z_lab` 已迁移到 3.0 的数据加载与任务实现，对照 BeyondMimic 原始代码，按 SuperMimic 论文逐项调整。
- 本机只验证完整流程；另一台电脑承担正式训练。
- 本阶段不包含 velocity 接口、蒸馏、实机部署或论文的大规模数据扩展实验。PPO 不作为训练前置条件。

## 2. 复用 BeyondMimic 数据链路与跟踪任务

### 已有数据准备与运行边界

现有 CSV 转换程序作为离线数据准备工具直接使用：两个 YAML 分别指定 LAFAN1 的 30 FPS 和 BONES 的 120 FPS 输入，现有插值、速度计算和机器人状态导出流程生成 **50 FPS NPZ**，统一保存到仓库根目录的 **`assets/motions/`**。保留现有配置结构、输出字段和命名方式。

**训练、策略回放和评估本身就只加载 NPZ。** CSV 和转换 YAML 属于离线准备阶段，不进入任务运行时的数据加载流程。

沿用现有 NPZ 格式，不增加数据来源、四元数顺序或格式版本字段，也不以重写转换器或重新生成一种数据格式作为任务开发前置条件。保存的四元数已经是 Isaac Lab 3.0.0 标准的 **`xyzw`**，任务直接使用这一约定。

### 源码中的数据路线

已对照以下实现确认数据在各层的职责：

- BeyondMimic：`../reference/whole_body_tracking/` 下的转换、回放、训练脚本，以及 `tasks/tracking/mdp/commands.py`、观测、奖励、终止和 G1 配置。原版训练脚本通过 WandB 下载 NPZ，然后把文件路径交给 `MotionCommandCfg.motion_file`；进入任务的仍然是 NPZ。
- `z_lab`：`../z_lab/scripts/mimic/` 与 `../z_lab/src/z_lab/tasks/mimic/`。该版本已经采用本地 NPZ、现有字段名、名称映射和 3.0 的 `xyzw` 约定，作为本项目移植的直接基础。
- 当前项目的 `csv_to_npz.py` 和 `replay_npz.py` 与 `z_lab` 对应脚本相比，仅减少了机器人注册项；沿用其 NPZ 数据接口。BeyondMimic 2.1 原版的字段名和 `wxyz` 约定只用于理解原始算法，不反向引入当前链路。

任务运行时的数据流为：

```text
assets/motions/*.npz
    → MotionLoader：加载数组并按名称映射关节与 body
    → MotionCommand：选择动作和帧，提供参考状态与对齐后的跟踪目标
    → 观测 / 奖励 / 终止 / reset
    → 策略动作 → 仿真中的实际机器人状态
```

沿用 `z_lab` 中现有字段到任务内部张量的映射：

| 现有 NPZ 字段 | 任务内部用途 |
|---|---|
| `fps` | 参考动作时间步；当前数据为 50 Hz |
| `dof_positions`、`dof_velocities` | `joint_pos`、`joint_vel` 参考状态与重置状态 |
| `body_positions`、`body_rotations` | body/root 参考位姿、目标对齐与跟踪误差 |
| `body_linear_velocities`、`body_angular_velocities` | body/root 参考速度与重置速度 |
| `dof_names`、`body_names`、`root_link_name`、`robot_name` | 名称映射和机器人、root 识别 |

参考位置在运行时加上对应的 `env_origins`。保留 BeyondMimic 的 anchor 与 body 目标对齐语义、名称映射及状态读取方式；涉及论文明确修改的 root 奖励与终止时，区分 pelvis root 和原任务的 torso anchor。

### 在现有任务目录中扩展

任务实现放在现有的 **`src/supermimic/tasks/super_motion_tracking/`**，沿用 `tracking_env_cfg.py`、`mdp/`、`config/g1/` 的组织方式，在该目录内完成 manager-based 任务 **`SuperMimic-MotionTracking-G1`** 的配置与注册。

- 复用 `z_lab` 的 `MotionLoader`、`MotionCommand` 和 G1 跟踪配置，将任务导入与机器人资源路径适配到本项目。
- 将单个 `motion_file` 扩展为加载 `assets/motions/` 中全部现有 NPZ 的动作库，保留 `--motion_dir` 作为目录覆盖入口。新增内容是多动作管理，不改变单文件格式。
- 每个并行环境独立维护动作编号与帧位置；从对应 NPZ 取得 root、关节位姿和速度完成 reset。动作结束时显式处理边界，清理该环境的观测历史，并通知训练采集端，避免跨动作或跨 reset 的轨迹。
- 在 `MotionCommand` 运行时查询未来 10 帧参考，提供 motion encoder 所需输入；历史观测、未来参考和采样统计均属于运行时数据，不写回 NPZ。
- 按论文调整观测、奖励、终止和 reset 采样；参考动作持续由 NPZ 提供，世界模型的监督数据则来自策略在仿真中实际产生的状态转移，写入第 3 节的 replay buffer。

### 在既有跟踪任务上的论文修改

| 项目 | 第一版要求 |
|---|---|
| 仿真频率 | 200 Hz；每 4 步输出一次控制，即 50 Hz |
| Actor 本体观测 | 每帧 93 维，使用最近 10 帧；噪声按表 A.1 |
| 参考输入 | 连续 10 帧未来参考，使用单独的 motion encoder |
| 动作 | 29 维归一化关节目标，经过论文对应的 PD 和 action scale |
| 跟踪奖励 | 表 A.2 的六项跟踪奖励，权重合计 5 |
| 正则项 | 动作平滑 −0.1；关节限位 −2；移除自碰撞惩罚 |
| 失败条件 | pelvis 高度误差超过 0.25 m，或重力方向 z 分量误差超过 0.8 |
| 其他调整 | 不以末端偏差终止，不施加随机速度推搡 |

G1 执行器以 `z_lab` 已迁移的 mimic 配置为基础，逐关节对照 BeyondMimic 的默认姿态、PD、armature 和动作缩放，记录已有差异及论文要求的调整。移除自碰撞**惩罚**不等于关闭物理自碰撞。

## 3. 世界模型与联合训练

### 模块和数据流

在 `src/supermimic_rl/` 内实现独立学习包和 runner，不使用 RSL-RL。

- **Replay buffer**：保存真实状态、实际执行的含噪动作、奖励、物理终止、动作帧和上下文 `ξ`。必须捕获自动 reset 前的终止状态，采样窗口不得跨 episode 或环形缓冲区覆盖边界。
- **世界模型**：学习物理状态的残差转移；姿态采用旋转增量组合。利用可微运动学解析计算观测与奖励，使用 sigmoid 终止预测头和 BCE。
- **Actor**：motion encoder → 32 维 tanh latent → decoder；decoder 同时接收 10 帧本体历史。想象过程中逐步更新历史，并重新加入传感器噪声。
- **Critic**：输入状态、参考和 `ξ`，拟合想象轨迹的 λ-return。Actor 更新时冻结 critic 和世界模型参数，但保留它们对输入状态、动作的梯度。
- **优化循环**：从随机初始化策略采集数据，不依赖预训练专家；先填充 replay，再交替执行模型拟合、actor 更新和 critic 更新。

正式配置以论文表 A.4/A.5 为基准：

| 参数 | 配置 |
|---|---|
| Dynamics 网络 | `[2048, 2048, 2048]` |
| Motion encoder、decoder | 各 `[2048, 1024, 512, 256]` |
| Critic | `[256, 128]` |
| 模型监督窗口 / 想象长度 | `k=8` / `H=32` |
| 模型拟合 | 每轮 10 批，每批 2048 个窗口 |
| 想象批量 / 仿真环境数 | 各 4096 |
| Replay 容量 | 每环境 1000 步 |
| 折扣 / λ | `0.99` / `0.95` |
| 采集动作噪声 | 每维 `N(0, 0.1²)` |
| 学习率 | `2e-4`；余弦衰减设置按表 A.5 |
| Critic 更新 | 5 epochs，每轮 4 个 minibatch |

Actor 目标严格实现公式 2 的软生存权重和末端价值 bootstrap；critic 目标按公式 3 计算并 detach，不增加 EMA target critic。

### 采样与随机化

- 按论文将动作时间线划成 1 秒 bins，采用 **50% 均匀、40% 低回报 softmax、10% 历史难度**的 reset 分布，替换 BeyondMimic 的失败次数采样。
- 两组数据作为同一动作库共享这些 bins，不要求在 NPZ 中标注来源；记录逐动作、逐 bin 的实际采样分布。
- 每个 episode 重采样摩擦、29 维标定偏移和躯干质心偏移，范围遵循表 A.3，并将实际参数写入 `ξ`。
- **质心逐 episode 更新作为正式训练前置验收项**：验证参数更新、派生惯性量刷新及并行环境隔离；与相同参数重新构建的模型比较动力学结果。若失败，先提交可版本化的兼容修复，不退化为构建时固定，也不只修改本机 `site-packages`。

## 4. 运行接口与明确的工程约定

统一入口调用独立 `scripts/supermimic/train.py` 和 `play.py`；评估作为 play 的 `--evaluate` 模式。提供 `smoke`、`paper` 两套配置，禁止显存不足时静默修改正式参数。

- **本机 smoke**：16 个仿真环境、8 条想象轨迹、模型批量 8；保留正式网络结构、`k=8`、`H=32` 和全部训练阶段。预采集 128 个向量步后运行 25 轮联合更新。
- **正式 paper**：使用上表规模和 320,000 轮更新；运行前输出显存预算，允许通过显式配置适配另一台电脑。
- 日志记录三阶段耗时、采样分布、模型误差、梯度、终止预测及显存；训练不依赖 WandB。
- Checkpoint 保存网络、优化器、归一化统计、调度器、采样器和配置；恢复时校验数据集及模型约定。

论文未公开的细节采用以下固定约定，并写入复现说明：

- 状态包含 root 位姿与速度、29 维关节位置与速度；上一动作和观测历史单独维护。
- 未来参考采用 `t+1…t+10`；包含参考关节位置、速度，以及当前参考 pelvis heading 坐标系下的未来 root 位姿和速度。
- 网络隐藏层使用 ELU；输入使用运行均值、方差归一化，想象更新期间冻结统计量。
- 历史不足时重复初始观测；未来参考越界时重复末帧，同时屏蔽动作结束后的损失。
- 训练 episode 上限先沿用 10 秒；完整动作评估取消该上限。
- update-to-data ratio `0.4` 定义为每个向量仿真步触发 0.4 轮联合更新，用累计器实现并记录该解释。

## 5. 分阶段验收与交付

按以下顺序实现，前一阶段通过后再进入下一阶段：

1. **既有数据链路接入**：任务直接加载现有格式的 57 个 NPZ，名称映射正确；复用现有回放工具抽查动作，核对 `MotionLoader → MotionCommand → reset/跟踪目标` 的状态一致性。
2. **环境闭环**：注册、reset、step、多动作切换正常；验证观测维度、奖励量纲、终止条件，以及逐 episode 随机化的实际效果。
3. **模型闭环**：验证可微运动学与仿真一致，监督窗口不串 episode，模型可拟合固定小批数据。
4. **训练闭环**：完成全部 smoke 更新；损失和梯度有限、actor 确实获得跨世界模型的梯度、冻结模块参数不变；checkpoint 可恢复并继续更新。
5. **回放与评估闭环**：加载 checkpoint 在真实仿真中运行；本机遍历所有动作做短程检查，正式训练后执行逐动作完整评估。

正式评估按附录 F：每个动作从首帧开始，摩擦为 1，无随机化和观测噪声。输出逐动作及总体的归一化回报、失败率和跟踪误差；提前失败后的剩余时长计零回报。

最终交付可迁移的代码、两套配置、数据转换命令、训练/恢复/回放/评估命令及复现差异说明。本机验收只证明流程与实现契约通过，不将未训练策略的表现视为复现结果；当前 57 动作混合集也不用于宣称论文完整 BONES benchmark 的成绩。
