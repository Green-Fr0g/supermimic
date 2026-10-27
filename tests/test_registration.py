# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import supermimic.tasks  # noqa: F401


def test_task_registration():
    """Resolve the real task without requiring a motion dataset or trainer."""
    from isaaclab_tasks.utils import parse_env_cfg

    spec = gym.spec("SuperMimic-MotionTracking-G1")
    cfg = parse_env_cfg(spec.id, device="cpu", num_envs=2)
    assert callable(gym.envs.registration.load_env_creator(spec.entry_point))
    assert cfg.scene.robot.spawn.usd_path.endswith("g1_29dof_rev_1_0.usd")
    assert len(cfg.actions.joint_pos.joint_names) == 29
    assert cfg.sim.dt * cfg.decimation == 0.02
    assert cfg.sim.physics.num_substeps == 1
    cfg.play_mode()
    assert cfg.commands.motion.evaluation
    assert cfg.terminations.time_out is None
    assert not cfg.observations.policy.enable_corruption
