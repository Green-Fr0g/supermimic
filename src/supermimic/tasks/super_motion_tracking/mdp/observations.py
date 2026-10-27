"""Separate actor history, future reference, physical state, and sim context."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.noise import NoiseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def proprioception(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Unscaled [q-q0, qd, root omega_b, gravity_b, previous action], [N,3J+6]."""
    cmd = env.command_manager.get_term("motion")
    robot = cmd.robot
    root_quat = cmd.robot_body_quat_w[:, cmd.root_index]
    root_omega = robot.data.body_ang_vel_w.torch[:, cmd.body_ids[cmd.root_index]]
    q0 = robot.data.default_joint_pos.torch[:, cmd.joint_ids] + cmd.context[:, 1 : 1 + len(cmd.joint_ids)]
    return torch.cat(
        (
            robot.data.joint_pos.torch[:, cmd.joint_ids] - q0,
            robot.data.joint_vel.torch[:, cmd.joint_ids],
            quat_apply_inverse(root_quat, root_omega),
            quat_apply_inverse(root_quat, torch.nn.functional.normalize(robot.data.GRAVITY_VEC_W.torch, dim=-1)),
            env.action_manager.action,
        ),
        dim=-1,
    )


def proprioception_noise(data: torch.Tensor, cfg: NoiseCfg) -> torch.Tensor:
    """Table A.1 noise, generalized to the robot's number of actuated joints."""
    joints = (data.shape[-1] - 6) // 3
    scales = torch.cat(
        (
            data.new_full((joints,), 0.01),
            data.new_full((joints,), 0.5),
            data.new_full((3,), 0.2),
            data.new_full((3,), 0.05),
            data.new_zeros(joints),
        )
    )
    return data + (2 * torch.rand_like(data) - 1) * scales


def future_reference(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.command_manager.get_term("motion").future_goal.flatten(1)


def future_reference_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.command_manager.get_term("motion").future_mask.float()


def physical_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Root p, xyzw, world v/omega, q, qd, in action joint order [N,2J+13]."""
    cmd = env.command_manager.get_term("motion")
    robot = cmd.robot
    root = cmd.body_ids[cmd.root_index]
    return torch.cat(
        (
            robot.data.body_pos_w.torch[:, root] - env.scene.env_origins,
            robot.data.body_quat_w.torch[:, root],
            robot.data.body_lin_vel_w.torch[:, root],
            robot.data.body_ang_vel_w.torch[:, root],
            robot.data.joint_pos.torch[:, cmd.joint_ids],
            robot.data.joint_vel.torch[:, cmd.joint_ids],
        ),
        dim=-1,
    )


def sim_context(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Actual episode parameters: friction, joint calibration, torso CoM offset."""
    return env.command_manager.get_term("motion").context
