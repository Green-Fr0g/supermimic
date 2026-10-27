"""Register the G1 task and its standalone SuperMimic configurations."""

import gymnasium as gym

gym.register(
    id="SuperMimic-MotionTracking-G1",
    entry_point="supermimic.tasks.super_motion_tracking.tracking_env:MotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1TrackingEnvCfg",
        "supermimic_rl_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:G1SuperMimicRunnerCfg",
        "supermimic_rl_smoke_cfg_entry_point": f"{__name__}.agents.supermimic_cfg:G1SuperMimicSmokeRunnerCfg",
    },
)
