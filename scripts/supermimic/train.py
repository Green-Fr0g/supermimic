"""Train SuperMimic using the task's registered paper/smoke configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="SuperMimic-MotionTracking-G1")
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--iterations", type=int, help="Absolute total number of joint updates")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--motion_dir", type=Path)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/supermimic_rl"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", action="store_true", help="Disable visualizers")
    for name in ("model_batch", "imagination_batch", "microbatch", "prefill", "save_interval"):
        parser.add_argument(f"--{name}", type=int)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        args.visualizer = []

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import resolve_task_config

    import supermimic.tasks  # noqa: F401
    from supermimic_rl.config import LearningCfg
    from supermimic_rl.runner import SuperMimicRunner

    key = "supermimic_rl_cfg_entry_point" if args.profile == "paper" else "supermimic_rl_smoke_cfg_entry_point"
    env_cfg, agent_cfg = resolve_task_config(args.task, key, overrides=[])
    count = args.num_envs if args.num_envs is not None else (4096 if args.profile == "paper" else 16)
    cfg = LearningCfg.from_task(agent_cfg, count, args.profile)
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("version") != 1:
            raise ValueError("Cannot resume an old collection-only checkpoint")
        cfg = LearningCfg(**saved["config"])
        del saved
    for name in (
        "num_envs",
        "iterations",
        "seed",
        "model_batch",
        "imagination_batch",
        "microbatch",
        "prefill",
        "save_interval",
    ):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    cfg.device = args.device
    cfg.validate()
    env_cfg.scene.num_envs, env_cfg.sim.device, env_cfg.seed = cfg.num_envs, cfg.device, cfg.seed
    if args.motion_dir is not None:
        env_cfg.commands.motion.motion_dir = str(args.motion_dir)
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        try:
            runner = SuperMimicRunner(env, cfg, args.log_dir)
            if args.checkpoint is not None:
                runner.load(args.checkpoint)
            path = runner.train(cfg.iterations)
            print(f"Saved checkpoint: {path}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
