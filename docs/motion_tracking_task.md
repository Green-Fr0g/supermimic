# Motion Tracking 任务端

实现任务 `SuperMimic-MotionTracking-G1`，算法接入见 [motion_tracking_algorithm.md](motion_tracking_algorithm.md)。方法依据论文 Appendix A/D.1，任务项参考 BeyondMimic 和已迁移的 `z_lab`；目录结构采用 `z_lab/locomotion` 的公共配置加机器人配置方式。

## 结构与职责

```text
src/supermimic/tasks/super_motion_tracking/
├── tracking_env_cfg.py       # 公共 scene、观测、动作、奖励、终止和事件
├── tracking_env.py           # ManagerBasedRLEnv 的参考时间线与终止状态处理
├── mdp/
│   ├── commands.py           # NPZ 动作加载、reset 采样、每环境动作/帧与未来参考
│   ├── actions.py            # 带逐 episode 标定偏移的 PD 目标
│   ├── observations.py
│   ├── rewards.py
│   ├── terminations.py
│   ├── events.py             # 摩擦、标定、torso CoM 和上下文 ξ
│   └── physics.py            # Newton/MJWarp preset
└── config/g1/
    ├── __init__.py           # Gym 注册，不绑定学习算法
    └── tracking_env_cfg.py   # G1 资源、关节顺序、跟踪 body、执行器参数
```

新增机器人时继承 `TrackingEnvCfg`，提供资产、关节/body 名称、root、anchor 和动作缩放。公共任务不引用 G1 常量。

## 数据与时间线

默认递归加载仓库根目录 `assets/motions/**/*.npz`，按相对文件名排序。LAFAN1/BONES 混用，不按来源平衡采样。运行时不读取 CSV/YAML，不改变现有 NPZ 字段；要求离线转换结果为 **50 Hz、xyzw**。关节按 `G1_DOF_ORDER` 映射，物理 articulation 的原生顺序可以不同。

NPZ 格式以 `scripts/mimic/csv_to_npz.py` 的输出为准：元数据为 `fps`、`robot_name`、`root_link_name`、`dof_names`、`body_names`；数据为 `dof_positions`、`dof_velocities`、`body_positions`、`body_rotations`、`body_linear_velocities`、`body_angular_velocities`。`MotionLoader` 直接使用这些字段名，不提供旧字段别名、缺省值或四元数顺序转换。

每环境独立保存 `motion_ids`、`time_steps`；reset 从 NPZ 写入 pelvis 位姿、速度和关节状态。动作作用后，先用推进前的参考帧及已有 anchor 对齐缓存计算奖励/终止，再由 command manager 推进参考帧；解析奖励采用相同契约。末帧触发 `truncated`，不在同一 episode 内直接换动作。新 episode 从所采样帧开始，10 帧历史重复初始观测。

## 输出接口

| 观测组 | G1 形状 | 内容 |
|---|---|---|
| `policy` | `[N,930]` | 10 × 93 维本体观测，最旧到最新 |
| `goal` | `[N,710]` | t+1…t+10，每帧 71 维参考 |
| `goal_mask` | `[N,10]` | 有效未来帧；越界帧重复末帧并标 0 |
| `state` | `[N,71]` | 实际 pelvis 位姿/速度、q、qd，无噪声 |
| `context` | `[N,33]` | μ、29 维标定偏移、torso 三维 CoM 偏移 |

单帧本体顺序为 `q-q0, qd, omega_b, gravity_b, previous_action`，噪声遵循表 A.1。`q0` 包含该 episode 的关节标定偏移。未来参考顺序为 `q, qd, root_position, root_xyzw, root_linear_velocity, root_angular_velocity`，root 数据转换到**当前参考 pelvis heading** 坐标系，位置以当前参考 pelvis 为原点。`state` 顺序为 `root_position, root_xyzw, root_linear_velocity, root_angular_velocity, q, qd`；位置扣除 env origin，速度在世界坐标系。沿用现有 NPZ：body 位姿为 link 位姿，body 速度为 CoM 速度。

`step()` 的 `extras['transition']` 保存本次 reset 前的 `motion_id/frame/next_frame/context/motion_end`。发生 reset 时，`extras['final_env_ids']` 对应 `extras['final_obs']` 的各行，后者包含 reset 前真实终止状态、未来参考和更新后的终止历史；返回的常规观测则属于新 episode。未 reset 的 step 不保留上次 `final_obs`。未来采集器须使用这个边界接口。

## 物理、奖励与随机化

- 物理步长 5 ms、decimation=4、Newton `num_substeps=1`，控制 50 Hz。接触管线沿用 `z_lab` 的 Newton collision → MJWarp 组合。
- 六项跟踪奖励权重 `1/1/1/1/0.5/0.5`；平滑 −0.1，限位 −2。RewardManager 统一乘一次控制 dt。
- body 位置/朝向保留 BeyondMimic 的 torso XY/yaw 对齐；root 两项奖励与失败条件使用 pelvis。速度保持世界坐标系。
- pelvis 高度误差 >0.25 m 或重力 z 分量误差 >0.8 时 `terminated`；10 秒上限和动作结束为 `truncated`。不加末端偏差终止、碰撞惩罚或随机推搡；物理自碰撞开启。
- 每 episode 采样 μ∈[0.3,1.2]、标定±0.01 rad、torso CoM 偏移 x±0.025/y,z±0.05 m；由 nominal CoM 重新加偏移，避免累积漂移。MJWarp 接触参数使用 robot collider priority=1，确保有效接触摩擦取 robot μ，而不被地面的 μ=1 覆盖。
- 1 秒 bins 的 reset 比例为 50% uniform、40% 当前 softmax 难度、10% 累计历史 softmax。折扣 0.99，真实 dt-scaled episode 回报更新起始 bin；并行完成的同 bin 回报先取均值。论文未指定估计器细节，这里采用 rate=0.1 的 EMA，未访问 bin 初始化为 Gmax=5dt/(1−γ)。自然片尾和时间上限按本 episode 的平均奖励补估尾部回报，物理失败不补尾，避免将短片尾误判为困难。以上估计器细节是工程约定，优化器实现位于 `supermimic_rl`。

## G1 与参考实现的差异

资产沿用项目本地 USD；原 BeyondMimic 使用其 URDF/capsule 版本，碰撞几何仍存在差异。PD 与缩放沿用 `z_lab` 的 mimic 低增益配置：stiffness 为 40.2/99.1/28.5/14.3/16.8，damping 为 2.6/6.3/1.8/0.9/1.1，缩放逐关节为 `0.25*effort_limit/stiffness`。这些是原版 10 Hz/阻尼比 2 公式的舍入值。

G1 任务专门恢复原 BeyondMimic 的 nominal pose、soft-limit factor=0.9 和分电机 armature：髋 pitch/yaw、waist yaw 为 0.010177520，髋 roll/膝为 0.025101925，踝/waist roll,pitch 为 0.007219450，肩/肘/wrist roll 为 0.003609725，wrist pitch/yaw 为 0.00425。通用机器人配置及离线转换格式不受这些任务设置影响。

默认任务 preset 为 Newton/MJWarp。摩擦事件区分 Newton、PhysX 和 OVPhysX 接口；目前只完成 Newton/MJWarp 的真实仿真验收，不宣称其他后端的训练已验证。

## 运行与验证

```bash
uv run isaaclab list_envs --show_presets
uv run isaaclab zero_agent --task SuperMimic-MotionTracking-G1 --num_envs 2 \
  commands.motion.motion_dir=/path/to/npz
./supermimic.sh -t --profile smoke --motion_dir /path/to/npz --headless
./supermimic.sh -p --checkpoint /path/to/checkpoint.policy.pt --evaluate \
  --motion_dir /path/to/npz --headless
SUPERMIMIC_TEST_MOTION_DIR=/path/to/npz \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
```

play 从首帧开始，关闭随机化和观测噪声，取消 10 秒限制；`--evaluate` 遍历动作库。真实验证结果见 [分阶段验证记录](motion_tracking_validation.md)，算法与可微任务头见 [算法说明](motion_tracking_algorithm.md)。

未设置 `SUPERMIMIC_TEST_MOTION_DIR` 时，真实集成测试显式 skip；ROS pytest 插件自动加载干扰通过 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 关闭。本机使用 z_lab 的既有 NPZ 和临时裁出的边界片段，不改写用户数据，不等于已遍历用户全部 LAFAN1/BONES 动作。
