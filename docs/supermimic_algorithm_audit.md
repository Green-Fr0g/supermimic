# SuperMimic 算法与训练流程审查

审查日期：2026-10-27。以下保留修复前工作区的审查快照，包含当时尚未提交的代码。后续修复和阶段性测试见 [验证记录](motion_tracking_validation.md)，当前实现说明见 [算法文档](motion_tracking_algorithm.md)。

## 结论与依据

**当前实现是任务端加数据采集骨架，尚未实现可学习的 SuperMimic 训练循环；默认 Newton 后端还存在首次 reset 就失败的问题。不能将当前 checkpoint 视为已训练策略。**

最终依据为 [论文](2610.28480v1.pdf) §3.1–3.3、Appendix A/C/D/F。代码参照为 `../reference/supermimic_vibed/`，审查时 HEAD 为 `a1508d4`；同时核对 BeyondMimic、z_lab 和本地 Isaac Lab 3.0 源码。

该参考项目的 README 明确说明它是**非官方复现**，并列出解析终止头、额外 reference-relative orientation、500 轮模型预热、随机化时机等实现选择。参考代码有价值，但这些选择不能自动当作论文要求。当前项目只加载现有 50 Hz、xyzw NPZ，这与复现目标兼容，无需改为参考项目运行时读取 CSV。

## 主要问题（按处理优先级）

### 1. P0：train 没有任何学习更新

- 位置：[runner.py](../src/supermimic_rl/runner.py#L57)、[learner.py](../src/supermimic_rl/learner.py#L1)。
- `train()` 只调用 `collect_step()`，增加 iteration 并保存。虽然创建了 actor optimizer，但没有 `zero_grad()`、`backward()` 或 `step()`。WorldModel 和 Critic 根本没有实例化到 runner 中。
- 缺失论文公式 1 的 k=8 自回归拟合、公式 2 的 H=32 想象和 actor BPTT，以及公式 3 的 critic 更新。单独定义 return 函数不等于接入训练。
- 独立假环境验证：训练 2 轮后 actor 所有参数逐位不变，optimizer state 数量为 0。
- 对照参考 `supermimic/train.py:147` 的 `wm_update()`、`:175` 的 `policy_update()` 和 `:302` 的主循环，实现三阶段闭环。

### 2. P0：默认后端首次 reset 报错

- 位置：[events.py](../src/supermimic/tasks/super_motion_tracking/mdp/events.py#L44)。
- `randomize_physics_material()` 调用 `root_view.get_material_properties()` / `set_material_properties()`；默认 Newton 使用的 `newton.selection.ArticulationView` 没有这两个方法。
- 用真实 NPZ 和两环境集成测试复现：`env.reset(seed=42)` 抛出 `AttributeError: 'ArticulationView' object has no attribute 'get_material_properties'`。
- `enabled=False` 仍执行同样的后端写入，所以 play 也无法靠关闭随机化绕过。
- 应依照 Isaac Lab 当前后端事件实现，确保写入后的实际摩擦与 `context[:, 0]` 一致。不能把 PhysX view 接口直接视为多后端通用接口。

### 3. P1：想象起点被改成不存在的状态

- 位置：[replay.py](../src/supermimic_rl/replay.py#L149)。
- `sample_starts()` 先抽取窗口，再将所有起点的 `state[..., 0]` 改为整个存储区第一个状态分量的最大值。在当前状态布局中，这个分量是 pelvis 的 x 位置，不是时间戳。
- 其余状态分量、context 和历史仍属于原样本，因而破坏真实 transition 的对应关系。未填满的存储来自 `torch.empty()`，最大值还可能读到未写入数据。
- 所有样本的末步 boundary 又被强制置为 True，与实际终止无关。当前采样结果也没有返回已存储的 motion ID / frame，无法据此可靠生成想象中的参考序列。
- 已用完整填满的小缓冲区复现：真实状态均满足 `[t, 2t]`，采样却返回 `[3, 4]`、`[3, 0]` 等不存在的组合。
- 起点必须按有效 `(环境, 时间槽)` 整体采样，同步取得完整状态、motion/frame、context、上一步动作和历史。

### 4. P1：覆盖与恢复后会构造错误的时间序列

- 位置：[replay.py](../src/supermimic_rl/replay.py#L113)、[runner.py](../src/supermimic_rl/runner.py#L87)。
- `_history()` 只检查 episode ID，不限制样本距缓冲区最旧有效元素的距离。同一长 episode 跨过环形覆盖后，向后取槽位会绕到样本未来。
- 已复现：容量 4、连续写入状态 0…5 后，状态 2 的三帧历史返回 `[4, 5, 2]`。
- 恢复时环境已经由构造函数重新 reset，但 `load()` 直接载入旧 replay，未标记新物理 episode。继续采集会把旧轨迹末尾与新 reset 状态拼接。
- 已复现恢复后 `[5, 100]` 被当作合法窗口。
- 需要同时校验有效年龄与 episode 边界，并在恢复后开启新的采集 episode。现有 `new_collection()` 会清空有效长度，不能直接用它实现“保留旧 replay”的恢复语义。

### 5. P1：当前网络不符合论文的策略结构，世界模型姿态更新也不合法

- 位置：[networks.py](../src/supermimic_rl/networks.py#L23)、[task_model.py](../src/supermimic_rl/task_model.py#L4)。
- Actor 将 history、goal 和 mask 拼接后送入单个 MLP，没有 Table A.4 的 motion encoder → 32 维 tanh latent → history-conditioned decoder。
- 现在 tanh 施加在最终 action，含噪采集动作还被裁到 `[-1, 1]`；参考实现的 tanh 在 latent，decoder 输出未限制到这个范围。当前限制会改变 `q_sp = q0 + calibration + scale * action` 的可达设定点。论文没有要求最终 action 必须 tanh，不能把 normalized action 直接理解为 `[-1, 1]`。
- WorldModel 返回 `state + delta`，包括直接相加四元数分量，未保持单位四元数；多步滚动时不是合法旋转更新。
- 学习式 termination head 没有 reference/goal 输入，但跟踪失败依赖参考高度和倾斜。同一机器人状态相对于不同参考可以有不同终止标签。应提供对应参考信息，或选择论文允许的解析软终止头；解析头不需要 BCE。
- `TrackingModel` 的 FK、reward、observation、goal 等均未实现。应先建立与真实环境一致且可微的任务模型，再做 BPTT。

### 6. P1：task 下的算法配置未被训练脚本使用

- 位置：[train.py](../scripts/supermimic/train.py#L33)、[config.py](../src/supermimic_rl/config.py#L7)、[G1 配置](../src/supermimic/tasks/super_motion_tracking/config/g1/agents/supermimic_cfg.py#L14)。
- 注册的 `supermimic_rl_cfg_entry_point` 和 smoke 配置没有被 train/play 解析；运行路径另用 `LearningCfg`，actor 网络还在 runner 中硬编码。
- 因而修改 task 下的网络、critic epochs、归一化、microbatch 等参数不会产生相应运行效果。`window/horizon/update_to_data` 等字段目前也没有进入优化循环。
- `paper().iterations=800128` 沿用旧采集步计数，但当前 runner 每 iteration 只采集一步且不执行 UTD。不能将它解释成论文的 320000 次联合更新。
- 应保留一套实际生效的配置，明确“向量采集步数”和“联合更新次数”。UTD=0.4 可按参考实现累计，每次联合更新平均采集 2.5 个向量仿真步。

### 7. P1：replay 的存储和抽样方式难以支持 paper 规模

- 位置：[replay.py](../src/supermimic_rl/replay.py#L25)、[runner.py](../src/supermimic_rl/runner.py#L39)。
- 每步保存完整的 930 维历史、710 维 goal 等全部 observation，并再次保存 next observation；不是文档所述的每步一个本体帧。
- 仅 obs 与 next obs 在 4096 环境 × 1000 步时就需要约 **53.53 GiB CPU 内存**，尚未计算动作、元数据、临时复制和 checkpoint。`state_dict()` 还克隆全部存储。
- `sample()` 每次遍历所有环境、所有候选槽位、窗口内所有 episode 标记，paper 设置约有四百万个候选点；尚未计入历史重建开销。
- 应保存紧凑状态、单帧观测或其重建信息以及必要元数据，使用张量化抽样。缓冲区可在 CPU 或 GPU，但选择和数据传输成本应明确。

### 8. P2：短失败轨迹被排除，采样回报又混淆失败与截断

- 位置：[replay.py](../src/supermimic_rl/replay.py#L94)、[commands.py](../src/supermimic/tasks/super_motion_tracking/mdp/commands.py#L334)。
- `_valid_starts()` 要求完整 k 步都属于同一 episode。短于 k 的失败 episode 完全不能用于世界模型拟合，临近终止处也不能作为独立 imagination 起点。
- 参考 `buffer.py:41` 使用逐步有效 mask，保留终止 transition，屏蔽其后的跨 reset 部分；起点采样不依赖后面还有 k 步。
- 目前 sampler 对失败、10 秒超时和自然片尾都直接使用有限累计回报，不区分截断。即使短片尾每步拿满奖励，其回报也会比长轨迹低，导致其被误判为更难。
- 参考 `env.py:221` 对非失败截断补尾值。论文没有指定具体补尾估计或 EMA 系数，但复现时应明确回报比较的时间范围，避免将片长当成难度。

### 9. P2：checkpoint 恢复与 play 混用了训练存储

- 位置：[runner.py](../src/supermimic_rl/runner.py#L75)、[play.py](../scripts/supermimic/play.py#L30)。
- checkpoint 没有模型/critic/normalizer/scheduler/sampler/RNG 的完整训练状态；前三者当前根本未接入训练。
- `load()` 没有使用或校验保存的配置、环境数、容量、机器人/数据集约定；也没有执行第 4 项所需的采集边界处理。
- play 用新 smoke 配置创建 runner，却加载整个训练 replay；`play()` 又通过 `collect_step()` 继续向其中写入。训练与回放环境数不一致时，存储布局与当前配置不一致。
- `torch.load(map_location=self.device)` 还可能先把巨大的 CPU replay 加载到 GPU。
- 应分开“只加载策略推理”和“恢复训练”，推理不创建、加载或写 replay；恢复训练则校验配置并恢复必要状态。

### 10. P2：play 尚不能用于论文评估

- 位置：[play.py](../scripts/supermimic/play.py#L30)、[runner.py](../src/supermimic_rl/runner.py#L71)。
- `play_mode()` 已关闭观测噪声与物理随机化，并从首帧开始，这是正确方向。
- 但所有环境固定使用默认 `evaluation_motion_id=0`，没有逐片遍历、失败计数或归一化回报。固定运行 1000 步并自动 reset 可能是在重复同一条动作。
- 这可作为单片可视化入口，但不满足 Appendix F.1“每条动作从首帧执行一次”的评估。可以继续只保留 train/play 两个脚本，将全数据集评估作为 play 的一个模式。

### 11. P2：文档和集成测试描述超过当前实现能力

- 位置：[算法文档](motion_tracking_algorithm.md#L40)、[测试](../tests/test_tracking_env.py#L217)。
- 文档写有完整 FK/BPTT、4.67 GiB replay、原子 checkpoint 和 11 项测试通过等记录，无法用于证明当前工作区具备这些能力。
- 当前集成测试调用 `TrackingModel.fk()`、`runner.collect()`、`runner.learner.fit_world()` 等未实现接口。即使修好首次 reset，它们也不能按现有代码通过。
- 当前通过的 replay 测试只核对起点第一个分量的最大值，没有核对完整状态、历史、context 和 frame 同源，因而未发现第 3 项。
- 后续验收应以实际权重更新、模型拟合下降、actor 梯度通过冻结模型以及真实/解析任务头一致性为准。既有历史运行记录需标明对应版本，不能替代当前验证。

## 命令推进与坐标系：需要统一契约，不能凭符号直接改顺序

当前 `MotionCommand._update_command()` 的推进位置和 anchor yaw 对齐缓存遵循 BeyondMimic/z_lab：

```text
参考帧 f、状态 s、历史 o → action
→ physics → 以当前 f 和已有对齐缓存计算 reward/termination
→ done 环境捕获 final_obs、reset
→ command 更新 f → f+1，并刷新对齐缓存
→ 返回新 observation
```

新参考项目 `supermimic_vibed/env.py:187` 则保存 `f`，在 physics 后用 `f1=f+1` 计算 reward/termination，解析 reward 用新状态重新计算 anchor 对齐。两者确有区别，但论文的时间符号本身没有规定 NPZ 数组应以哪一帧编号。因此本次不将“沿用 BeyondMimic 推进位置”单独判作错误，也不建议恢复 step 开头的额外 advance。

必须解决的是：未来可微任务模型不能直接照抄新参考项目的 `reward(S1, f+1)`，却让真实环境继续采用当前 `reward(S1, f)` 和旧 anchor 缓存。否则世界模型优化的奖励与真实奖励不同，actor 可能利用该差异。保留现有顺序时，需要提供真实使用的参考帧和对齐信息，并做数值一致性验证。

当前 goal 使用当前**参考 pelvis heading** 表示未来参考位姿，是参考动作内的相对坐标，不是把目标不断转换到机器人坐标系。论文只要求 K=10 未来参考，未规定这里必须用 4D quaternion 或参考项目的 6D orientation。这是可记录的表示选择；姿态表示连续性和多步模型的旋转合法性仍需保证。

## 已有的正确基础

- task 的 10 帧 × 93 维本体历史，以及 Table A.1 的噪声幅值与论文一致。
- 未来目标 K=10、状态 71 维、context 33 维的接口划分清楚；actor 没有直接拿到 privileged context。
- 奖励权重、joint-limit 的 -2、去掉 self-collision reward、pelvis 高度/重力 z 分量的失败阈值与 Appendix A 一致。
- `gamma=0.99`、`lambda=0.95` 和公式 2/3 的独立数值测试正确；`frozen()` 的设计允许冻结参数而保留输入梯度。
- 已保存实际执行的含噪 action；终止时捕获真实 final state 的设计正确，但其余运行契约仍需通过仿真验证。
- 随机化范围符合 Table A.3；当前实际后端写入是否全部正确，不应在 reset 修复前宣称已验证。

## 本次验证与限制

默认测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
# 6 passed, 5 skipped
```

真实 Newton 检查，使用已存在的 z_lab 50 Hz G1 NPZ；未转换或改写用户的数据集：

```bash
SUPERMIMIC_TEST_MOTION_DIR=../z_lab/src/z_lab/tasks/mimic/config/g1/motions \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_tracking_env.py -x
# 首次 reset 的摩擦接口报错；1 failed，-x 后停止，耗时约 72 秒
```

另执行了不修改源码的独立探针，验证了 actor 不更新、虚假起点、历史读取未来、恢复跨轨迹窗口和 replay 内存下界。未运行长期训练，也未声称参考项目自报的收敛成绩已在本机重现。当前仓库根目录未发现 `assets/motions/`，默认数据入口还需要实际 NPZ 目录或 `--motion_dir`。

## 建议修复顺序

1. 修复后端随机化接口，跑通 reset/step/partial reset；保留实际 context 与物理属性一致性检查。
2. 统一真实和解析任务模型的帧、anchor、link/CoM 速度契约，实现可微 FK/observation/reward/termination/goal，并验证数值一致性。
3. 修复 replay 的整体抽样、短轨迹 mask、历史覆盖和恢复边界，减少冗余存储。
4. 接通唯一生效的 task 配置，实现 encoder/decoder、合法旋转世界模型和 critic。
5. 完成 k=8 拟合 → H=32 actor BPTT → detached λ-return critic 更新，接上 UTD、归一化、裁剪和学习率调度。
6. 分离推理与恢复，完成全动作评估；更新文档和当前版本验收记录后再做长训。
