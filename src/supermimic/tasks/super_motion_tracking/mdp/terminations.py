"""Pelvis failure criteria and explicit motion-end truncation."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse


def root_height_error(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    return (cmd.body_pos_w[:, cmd.root_index, 2] - cmd.robot_body_pos_w[:, cmd.root_index, 2]).abs() > threshold


def root_tilt_error(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    gravity = torch.nn.functional.normalize(cmd.robot.data.GRAVITY_VEC_W.torch, dim=-1)
    target = quat_apply_inverse(cmd.body_quat_w[:, cmd.root_index], gravity)
    actual = quat_apply_inverse(cmd.robot_body_quat_w[:, cmd.root_index], gravity)
    return (target[:, 2] - actual[:, 2]).abs() > threshold


def motion_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.command_manager.get_term("motion").motion_ended
