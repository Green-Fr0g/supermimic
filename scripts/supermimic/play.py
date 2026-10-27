"""Play or evaluate a SuperMimic policy without allocating training replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="SuperMimic-MotionTracking-G1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Playback length or per-clip evaluation cap")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--motion_id", type=int, default=0)
    parser.add_argument("--motion_dir", type=Path)
    parser.add_argument("--headless", action="store_true", help="Disable visualizers")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate every NPZ once from its first frame")
    parser.add_argument("--output", type=Path, default=Path("logs/supermimic_rl/evaluation.json"))
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        args.visualizer = []
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.evaluate and args.num_envs != 1:
        parser.error("--evaluate requires --num_envs 1")

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import resolve_task_config

    import supermimic.tasks  # noqa: F401
    from supermimic_rl.config import LearningCfg
    from supermimic_rl.runner import SuperMimicRunner

    policy_path = args.checkpoint
    if not policy_path.stem.endswith(".policy"):
        policy_path = policy_path.with_suffix(".policy.pt")
    saved = torch.load(policy_path, map_location="cpu", weights_only=False)
    if saved.get("version") != 1:
        raise ValueError("Unsupported checkpoint; expected a trained SuperMimic checkpoint")
    cfg = LearningCfg(**saved["config"])
    cfg.num_envs, cfg.device = args.num_envs, args.device
    env_cfg, _ = resolve_task_config(args.task, None, play_mode=True, overrides=[])
    env_cfg.scene.num_envs, env_cfg.sim.device, env_cfg.seed = cfg.num_envs, cfg.device, cfg.seed
    env_cfg.commands.motion.evaluation_motion_id = args.motion_id
    if args.motion_dir is not None:
        env_cfg.commands.motion.motion_dir = str(args.motion_dir)
    del saved
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        try:
            runner = SuperMimicRunner(env, cfg, args.checkpoint.parent, inference=True)
            runner.load(args.checkpoint)
            if args.evaluate:
                result = runner.evaluate(args.steps)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps(result, indent=2))
            else:
                runner.play(args.steps or 1000)
        finally:
            env.close()


if __name__ == "__main__":
    main()
