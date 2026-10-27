"""Paper Table A.2; RewardManager applies control dt exactly once.

Body target alignment follows BeyondMimic/z_lab. Pelvis replaces torso only
for the global root terms; body velocities retain their original world frame.
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_error_magnitude


def root_position(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    error = cmd.body_pos_w[:, cmd.root_index] - cmd.robot_body_pos_w[:, cmd.root_index]
    return torch.exp(-error.square().sum(-1) / std**2)


def root_orientation(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    error = quat_error_magnitude(cmd.body_quat_w[:, cmd.root_index], cmd.robot_body_quat_w[:, cmd.root_index])
    return torch.exp(-error.square() / std**2)


def body_position(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    return torch.exp(-(cmd.body_pos_relative_w - cmd.robot_body_pos_w).square().sum(-1).mean(-1) / std**2)


def body_orientation(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    return torch.exp(-quat_error_magnitude(cmd.body_quat_relative_w, cmd.robot_body_quat_w).square().mean(-1) / std**2)


def body_linear_velocity(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    actual = cmd.robot.data.body_lin_vel_w.torch[:, cmd.body_ids]
    return torch.exp(-(cmd.body_lin_vel_w - actual).square().sum(-1).mean(-1) / std**2)


def body_angular_velocity(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    actual = cmd.robot.data.body_ang_vel_w.torch[:, cmd.body_ids]
    return torch.exp(-(cmd.body_ang_vel_w - actual).square().sum(-1).mean(-1) / std**2)
