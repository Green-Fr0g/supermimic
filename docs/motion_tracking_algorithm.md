# SuperMimic motion tracking：算法实现

独立包 `src/supermimic_rl/` 直接连接 `SuperMimic-MotionTracking-G1`，不使用 RSL-RL、PPO 或专家预训练。最终方法依据为论文 §3、Appendix A/C/D/F；`../reference/supermimic_vibed/` 是非官方实现参考，BeyondMimic/z_lab 决定本项目保留的任务时序。

## 配置与文件职责

训练入口实际读取 task 注册的 `supermimic_rl_cfg_entry_point`（paper）或 `supermimic_rl_smoke_cfg_entry_point`：

- `tasks/super_motion_tracking/rl_cfg/supermimic_cfg.py`：公共配置 schema，使用 Isaac Lab `configclass`。
- `tasks/super_motion_tracking/config/g1/agents/supermimic_cfg.py`：G1 网络、paper/smoke 参数。
- `supermimic_rl/config.py`：将 task 配置转换为经过验证的运行配置；恢复训练以保存的配置为准。
- `networks.py`：运行均值/方差、motion encoder/decoder、几何世界模型、critic。
- `task_model.py`：从当前 USD 提取关节树，提供可微 FK、解析奖励/观测/终止和 NPZ goal。
- `replay.py`：紧凑 CPU 环形 replay，重建本体历史，处理 reset、覆盖和恢复边界。
- `learner.py`：公式 1–3、三类优化器、梯度裁剪、学习率调度。
- `runner.py`：真实采集、UTD、日志、checkpoint、推理和逐动作评估。
- `scripts/supermimic/train.py` / `play.py`：仅两个运行入口，使用 Isaac Lab 3.0 `launch_simulation()`。

`max_iterations`、`--iterations` 和学习率调度均按**联合更新次数**计数。paper 为 320000，smoke 为 25。UTD=0.4 表示平均每 2.5 个向量仿真步进行一次联合更新；128 步预采集不产生更新债务。恢复时 `--iterations 26` 表示训练到第 26 轮，不是再训练 26 轮。

## 三阶段训练

1. **真实采集**：确定性策略输出动作，叠加 `N(0,0.1²)` 噪声。记录实际执行动作、真实状态、参考 motion/frame 和本 episode 的物理 context；更新归一化统计。
2. **世界模型拟合**：每次联合更新拟合 10 批 replay 窗口。paper 每批 2048、k=8；自回归输入上一步预测状态和实际日志动作，不做 teacher forcing。终止 transition 保留，跨 reset、覆盖或最新存储边界后的部分用 mask 排除。
3. **想象与策略改进**：从 replay 独立抽取起点，H=32。逐步生成未来参考、执行策略、预测状态、计算解析奖励/软终止、更新本体历史并加入表 A.1 的传感器噪声。actor 沿全部 H 步 BPTT，优化公式 2 的软生存回报与末端 critic bootstrap。世界模型和 critic 仅冻结参数，保留输入 Jacobian。

Critic 的 λ-return 由同一条想象轨迹计算一次并 detach，λ=0.95；之后使用 5 epochs × 4 shuffled minibatches 拟合。公式 3 的回归不额外乘 actor 的软生存权重；片尾之外的时间步被屏蔽。无 EMA/target critic。

三类优化器均为 Adam、初始学习率 `2e-4`；actor β=(0.9,0.95)。学习率用 250000 次联合更新余弦衰减至 0.03 倍，然后保持；梯度全局范数裁剪 1.0，actor 再逐元素裁剪 0.1。非有限梯度会明确失败，不静默写入模型。

## 状态、网络和时间契约

G1 state 为 71 维：pelvis link 位置、xyzw、世界 CoM 线速度/角速度、29 维 q、29 维 qd。FK 先将根 CoM 速度换算为 link 原点速度，再递推各关节，最后返回各 tracked body 的 CoM 速度；torso CoM 使用当前 context 偏移。

世界模型使用去全局 XY/yaw 的状态特征、action 和 33 维 context，MLP `[2048,2048,2048]` 预测 70 维局部增量。姿态通过旋转向量生成单位四元数并组合，不能直接相加四元数分量。输入/增量尺度由真实数据统计；状态损失按分量尺度归一化，包含旋转误差。

策略为 motion encoder `[2048,1024,512,256]` → **32 维 tanh latent** → decoder `[2048,1024,512,256]`；decoder 同时读取 10 帧 ×93 维本体历史。最终动作不做 tanh/`[-1,1]` 裁剪，使用现有低增益 PD 映射。goal 为未来 10 帧参考加有效 mask，不把 privileged context 输入 actor。Critic `[256,128]` 接收状态特征、相对参考误差、goal、context 和上一动作；上一动作用于动作平滑奖励的状态信息。

真实任务保持 BeyondMimic 顺序：physics → 当前参考帧/已有 anchor 缓存的 reward/termination → final observation/reset → command 推进 → 新 observation。可微奖励据此使用**新状态、推进前参考帧、前一步机器人 anchor**，不会直接照搬另一参考项目的 `reward(S1,f+1)`。goal 则是当前参考 pelvis heading 下的未来参考，保持现有 NPZ 坐标约定。

奖励、观测和终止头均选用论文 §3.2 允许的**解析头**，因此没有这些头的监督损失或 termination BCE。软终止用高度/倾斜阈值附近的 sigmoid（宽度 0.02 m/0.04）代替硬阈值；真实环境仍输出硬失败。解析 soft termination 是与非官方参考一致的工程选择，不是论文公开的唯一实现。

## Replay、内存与恢复

每步保存两个 71 维状态、一个 93 维本体帧、动作/上一动作、context、奖励、motion/frame 和 episode 信息，不重复保存 930 维完整历史或未来 goal。采样是张量化索引；history 同时限制 episode 年龄和缓冲区年龄，不会绕回未来。历史不足时重复最早仍可用帧。

CPU replay 的容量内存约为 smoke 20.5 MiB / paper 5.12 GiB，另需动作库、网络、优化器与仿真内存。`memory_budget.json` 记录实际容量和参数数量；`metrics.jsonl` 的 `torch_peak_bytes` 只统计 PyTorch，不包含 Warp。microbatch 累积完整 batch 的梯度，MLP activation checkpoint 降低 BPTT 激活占用；不会自动缩小网络、总 batch 或截断时间反传。

训练 checkpoint 原子写入，保存模型及归一化、三个优化器/调度器、replay、sampler、UTD 余额、更新/仿真步数、配置和 RNG。校验动作名/顺序、NPZ 内容 SHA256、USD layers、机器人配置、物理配置等约定。恢复后开始新的物理 episode，保留旧 replay，标记新的采集边界；不宣称恢复接触求解器状态后的轨迹逐位一致。

每个训练 checkpoint 同时生成一个小型 `*.policy.pt`，只包含推理所需策略、配置和 contract。play 接受任一文件路径：给定训练 `*.pt` 时自动读取同名 `*.policy.pt`，不创建、加载或写入 replay。旧 collection-only checkpoint 无法恢复为已训练算法。

## 运行与评估

运行时仅加载已有 50 Hz、xyzw NPZ，默认目录 `assets/motions/`；可用 `--motion_dir` 覆盖。CSV/YAML 仍属于离线转换，不改变用户的数据格式。

```bash
./supermimic.sh -t --profile smoke --headless
./supermimic.sh -t --profile paper --headless

# 总更新数增加到 26；其余参数从 checkpoint 恢复
./supermimic.sh -t --checkpoint logs/supermimic_rl/checkpoint_final.pt \
  --iterations 26 --log_dir logs/supermimic_rl/resumed --headless

./supermimic.sh -p --checkpoint logs/supermimic_rl/checkpoint_final.policy.pt \
  --steps 100 --headless
./supermimic.sh -p --checkpoint logs/supermimic_rl/checkpoint_final.policy.pt \
  --evaluate --output logs/supermimic_rl/evaluation.json --headless
```

可显式覆盖 `--num_envs`、`--model_batch`、`--imagination_batch`、`--microbatch`、`--prefill`、`--iterations`、`--save_interval`、`--seed`。恢复训练只允许修改总更新数、保存间隔、设备和 microbatch；训练环境数及其他学习参数必须相同。可视化用 `--visualizer newton_gl` 等当前 Isaac Lab 选项；本项目 `--headless` 明确关闭 visualizers。

评估按排序后的 NPZ 逐片从首帧开始，单环境、摩擦 1、无随机化/观测或动作噪声，无 10 秒上限。报告逐片失败、归一化回报、root/对齐 body 位置误差。提前失败后剩余长度按零回报计；`--evaluate --steps N` 是每片上限，未执行完的片段标 `partial`。

保留现有 BeyondMimic 帧编号后，一条长度 L 的 clip 有 L 个 reward-bearing 控制步（含参考帧 0）；本项目的归一化分母为 `5*control_dt*L`。非官方参考从 0 推进后比较帧 1，使用 L−1 步。比较论文数字时必须同时报告这一帧的时序约定，不能把两种分母混用。

## 验证范围

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
SUPERMIMIC_TEST_MOTION_DIR=/path/to/npz \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
```

阶段性命令、结果和限制见 [分阶段验证记录](motion_tracking_validation.md)。没有数据变量时集成测试会 skip；不能据此宣称 FK、BPTT 或真实训练已验证。当前真实验证后端为 Newton/MJWarp；PhysX/OVPhysX 的摩擦写入有独立分支，但尚无这两个后端的训练验收。
