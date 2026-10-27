"""NPZ motion loading, reset sampling, and BeyondMimic-style tracking commands.

Reference: HybridRobotics/whole_body_tracking and z_lab's Isaac Lab 3.0 port.
Root rewards use pelvis; relative body alignment retains the torso anchor.
Motion fields follow scripts/mimic/csv_to_npz.py and Isaac Lab 3.0 xyzw order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _indices(available: list[str], requested: Sequence[str], path: Path) -> list[int]:
    if len(set(available)) != len(available):
        raise ValueError(f"{path}: duplicate motion names")
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"{path}: missing names {sorted(missing)}")
    return [available.index(name) for name in requested]


class MotionLoader:
    """Load a mixed library of named NPZ clips, mapped to one robot's order.

    Only the NPZ schema written by scripts/mimic/csv_to_npz.py is supported.
    Tensor attributes retain its field names; rotations remain xyzw.
    """

    def __init__(
        self,
        motion_dir: str | Path,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        root_name: str,
        robot_name: str,
        device: str = "cpu",
        fps: float = 50.0,
    ) -> None:
        directory = Path(motion_dir).expanduser().resolve()
        self.files = sorted(directory.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No NPZ motions in {motion_dir}. Use scripts/mimic/config_csv_to_npz.py offline, "
                "or override commands.motion.motion_dir. Runtime loads NPZ only."
            )
        self.names = [str(path.relative_to(directory)) for path in self.files]
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.fps = fps
        chunks: dict[str, list[torch.Tensor]] = {
            "dof_positions": [],
            "dof_velocities": [],
            "body_positions": [],
            "body_rotations": [],
            "body_linear_velocities": [],
            "body_angular_velocities": [],
        }
        lengths = []
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                if not np.isclose(float(data["fps"][0]), fps):
                    raise ValueError(f"{path}: expected {fps} Hz NPZ; convert offline to the control frequency")
                if str(data["robot_name"][0]) != robot_name:
                    raise ValueError(f"{path}: robot_name does not match {robot_name!r}")
                if str(data["root_link_name"][0]) != root_name:
                    raise ValueError(f"{path}: root_link_name does not match {root_name!r}")
                dof_names = data["dof_names"].tolist()
                motion_bodies = data["body_names"].tolist()
                joints = _indices(dof_names, joint_names, path)
                bodies = _indices(motion_bodies, body_names, path)
                length = data["dof_positions"].shape[0]
                if length < 2:
                    raise ValueError(f"{path}: at least two reference frames are required")
                lengths.append(length)
                frame_shapes = {
                    "dof_positions": (len(dof_names),),
                    "dof_velocities": (len(dof_names),),
                    "body_positions": (len(motion_bodies), 3),
                    "body_rotations": (len(motion_bodies), 4),
                    "body_linear_velocities": (len(motion_bodies), 3),
                    "body_angular_velocities": (len(motion_bodies), 3),
                }
                for field, frame_shape in frame_shapes.items():
                    expected = (length, *frame_shape)
                    values = data[field]
                    if values.shape != expected or not np.isfinite(values).all():
                        raise ValueError(f"{path}: {field} must be finite with shape {expected}, got {values.shape}")
                    indices = joints if field in ("dof_positions", "dof_velocities") else bodies
                    selected = values[:, indices]
                    if field == "body_rotations" and not np.allclose(np.linalg.norm(selected, axis=-1), 1, atol=1e-3):
                        raise ValueError(f"{path}: body_rotations must contain unit xyzw quaternions")
                    chunks[field].append(torch.as_tensor(selected, dtype=torch.float32))
        self.dof_positions = torch.cat(chunks["dof_positions"]).to(device)
        self.dof_velocities = torch.cat(chunks["dof_velocities"]).to(device)
        self.body_positions = torch.cat(chunks["body_positions"]).to(device)
        self.body_rotations = torch.cat(chunks["body_rotations"]).to(device)
        self.body_linear_velocities = torch.cat(chunks["body_linear_velocities"]).to(device)
        self.body_angular_velocities = torch.cat(chunks["body_angular_velocities"]).to(device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.starts = torch.cat((self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1]))
        # Bins contain valid reset frames only: the last frame is a terminal target.
        frames_per_bin = int(round(fps))
        self.bin_motion = torch.repeat_interleave(
            torch.arange(len(lengths), device=device), (self.lengths - 2) // frames_per_bin + 1
        )
        bin_starts = [torch.arange(0, length - 1, frames_per_bin, device=device) for length in lengths]
        self.bin_start = torch.cat(bin_starts)
        self.bin_length = torch.minimum(
            self.bin_start.new_full(self.bin_start.shape, frames_per_bin),
            self.lengths[self.bin_motion] - 1 - self.bin_start,
        )

    def indices(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Resolve clip-local frames, repeating the final frame beyond a clip boundary."""
        return self.starts[motion_ids] + frames.clamp(min=0).minimum(self.lengths[motion_ids] - 1)


class ReturnBasedSampler:
    """Mix uniform, current difficulty, and historical difficulty over 1 s bins.

    The paper leaves the running-estimate update unspecified. We use an EMA
    (rate 0.1), initialize unseen bins to Gmax, and accumulate past softmax
    distributions. Episode returns update their reset bin; failures have zero
    reward for the unobserved remainder.
    """

    def __init__(self, num_bins: int, device: str, max_return: float, update_rate: float = 0.1) -> None:
        self.max_return = max_return
        self.update_rate = update_rate
        self.returns = torch.full((num_bins,), max_return, dtype=torch.float32, device=device)
        self.history = torch.full((num_bins,), 1.0 / num_bins, device=device)
        self.visits = torch.zeros(num_bins, dtype=torch.long, device=device)

    @property
    def probabilities(self) -> torch.Tensor:
        difficulty = torch.softmax(-25.0 * self.returns / self.max_return, dim=0)
        return 0.5 / len(self.returns) + 0.4 * difficulty + 0.1 * self.history / self.history.sum()

    def sample(self, count: int) -> torch.Tensor:
        """Draw global timeline bins, with no source-dataset balancing."""
        bins = torch.multinomial(self.probabilities, count, replacement=True)
        self.visits.add_(torch.bincount(bins, minlength=len(self.returns)))
        return bins

    def update(self, bins: torch.Tensor, returns: torch.Tensor) -> None:
        """Aggregate simultaneous episode returns before updating their start bins."""
        counts = torch.bincount(bins, minlength=len(self.returns))
        sums = torch.zeros_like(self.returns).scatter_add_(0, bins, returns)
        seen = counts > 0
        self.returns[seen] = torch.lerp(self.returns[seen], sums[seen] / counts[seen], self.update_rate)
        self.history.add_(torch.softmax(-25.0 * self.returns / self.max_return, dim=0))


class MotionCommand(CommandTerm):
    """Independent clip/frame per environment, with explicit episode boundaries."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids = self.robot.find_joints(cfg.joint_names, preserve_order=True)[0]
        self.body_ids = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self.root_index = cfg.body_names.index(cfg.root_body_name)
        self.anchor_index = cfg.body_names.index(cfg.anchor_body_name)
        self.motion = MotionLoader(
            cfg.motion_dir,
            cfg.joint_names,
            cfg.body_names,
            cfg.root_body_name,
            cfg.robot_name,
            device=self.device,
            fps=1.0 / env.step_dt,
        )
        if not 0 <= cfg.evaluation_motion_id < len(self.motion.files):
            raise ValueError("evaluation_motion_id is outside the motion library")
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros_like(self.motion_ids)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        # Isaac Lab 3.0 stores quaternions in xyzw order.
        self.body_quat_relative_w[:, :, 3] = 1.0
        self.start_bins = torch.zeros_like(self.motion_ids)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward_sum = torch.zeros_like(self.episode_returns)
        self.episode_steps = torch.zeros_like(self.episode_returns)
        self.discount = torch.ones_like(self.episode_returns)
        self.sampler = ReturnBasedSampler(
            len(self.motion.bin_motion), self.device, max_return=5.0 * env.step_dt / (1 - cfg.discount)
        )
        # One owner for the actual sampled physical parameters: mu, dq0, dCoM.
        self.context = torch.zeros(self.num_envs, 1 + len(cfg.joint_names) + 3, device=self.device)
        self.context[:, 0] = 1.0
        self.future_offsets = torch.arange(1, cfg.future_steps + 1, device=self.device)
        self.metrics = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("root_position_error", "body_position_error", "sampling_entropy")
        }

    @property
    def frame_indices(self) -> torch.Tensor:
        return self.motion.indices(self.motion_ids, self.time_steps)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.joint_pos, self.joint_vel), dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.dof_positions[self.frame_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.dof_velocities[self.frame_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_positions[self.frame_indices] + self._env.scene.env_origins[:, None]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_rotations[self.frame_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_linear_velocities[self.frame_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_angular_velocities[self.frame_indices]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.anchor_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.anchor_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.anchor_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.anchor_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot_body_pos_w[:, self.anchor_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot_body_quat_w[:, self.anchor_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot_body_lin_vel_w[:, self.anchor_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot_body_ang_vel_w[:, self.anchor_index]

    @property
    def motion_ended(self) -> torch.Tensor:
        return self.time_steps >= self.motion.lengths[self.motion_ids] - 1

    @property
    def future_mask(self) -> torch.Tensor:
        frames = self.time_steps[:, None] + self.future_offsets
        return frames < self.motion.lengths[self.motion_ids, None]

    @property
    def future_goal(self) -> torch.Tensor:
        """Future q, qd, root pose/velocity in current reference pelvis heading.

        Shape [N, K, 2J+13]. Positions are relative to the current reference
        pelvis; orientations are xyzw. Final-frame padding is masked separately.
        """
        frames = self.time_steps[:, None] + self.future_offsets
        ids = self.motion.indices(self.motion_ids[:, None], frames)
        root = self.root_index
        current = self.frame_indices
        inv_heading = quat_inv(yaw_quat(self.motion.body_rotations[current, root]))
        inv_heading = inv_heading[:, None].expand(-1, self.cfg.future_steps, -1)
        pos = self.motion.body_positions[ids, root] - self.motion.body_positions[current, root][:, None]
        return torch.cat(
            (
                self.motion.dof_positions[ids],
                self.motion.dof_velocities[ids],
                quat_apply(inv_heading, pos),
                quat_mul(inv_heading, self.motion.body_rotations[ids, root]),
                quat_apply(inv_heading, self.motion.body_linear_velocities[ids, root]),
                quat_apply(inv_heading, self.motion.body_angular_velocities[ids, root]),
            ),
            dim=-1,
        )

    def record_rewards(self, rewards: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Accumulate real, dt-scaled rewards from each sampled reset point."""
        if self.cfg.evaluation:
            return
        self.episode_returns[env_ids] += self.discount[env_ids] * rewards[env_ids]
        self.episode_reward_sum[env_ids] += rewards[env_ids]
        self.episode_steps[env_ids] += 1
        self.discount[env_ids] *= self.cfg.discount

    def finish_returns(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.evaluation:
            # Natural clip ends/time limits do not imply failed tracking. Estimate
            # their unobserved tail from the mean reward, as in the reference reproduction.
            mean_reward = self.episode_reward_sum[env_ids] / self.episode_steps[env_ids].clamp_min(1)
            tail = self.discount[env_ids] * mean_reward / (1 - self.cfg.discount)
            returns = self.episode_returns[env_ids] + torch.where(self._env.reset_terminated[env_ids], 0, tail)
            self.sampler.update(self.start_bins[env_ids], returns)

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        if self.cfg.evaluation:
            self.motion_ids[env_ids] = self.cfg.evaluation_motion_id
            self.time_steps[env_ids] = 0
        else:
            bins = self.sampler.sample(len(env_ids))
            self.start_bins[env_ids] = bins
            self.motion_ids[env_ids] = self.motion.bin_motion[bins]
            offset = (torch.rand(len(env_ids), device=self.device) * self.motion.bin_length[bins]).long()
            self.time_steps[env_ids] = self.motion.bin_start[bins] + offset
        self.episode_returns[env_ids] = 0
        self.episode_reward_sum[env_ids] = 0
        self.episode_steps[env_ids] = 0
        self.discount[env_ids] = 1
        root = self.root_index
        pose = torch.cat((self.body_pos_w[env_ids, root], self.body_quat_w[env_ids, root]), dim=-1)
        velocity = torch.cat((self.body_lin_vel_w[env_ids, root], self.body_ang_vel_w[env_ids, root]), dim=-1)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(
            position=self.joint_pos[env_ids], joint_ids=self.joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.joint_vel[env_ids], joint_ids=self.joint_ids, env_ids=env_ids
        )
        # Reset places the robot on the sampled reference frame. Initialize the
        # cached relative target here so the first post-reset reward is valid;
        # subsequent frames are updated by _update_command like BeyondMimic.
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]

    def _update_metrics(self) -> None:
        self.metrics["root_position_error"][:] = (
            self.body_pos_w[:, self.root_index] - self.robot_body_pos_w[:, self.root_index]
        ).norm(dim=-1)
        self.metrics["body_position_error"][:] = (
            (self.body_pos_relative_w - self.robot_body_pos_w).norm(dim=-1).mean(dim=-1)
        )
        p = self.sampler.probabilities
        self.metrics["sampling_entropy"][:] = -(p * p.log()).sum()

    def _update_command(self) -> None:
        """Advance the reference after the current physics/reward transition."""
        self.time_steps.add_(1)
        resample_env_ids = (self.time_steps >= self.motion.lengths[self.motion_ids]).nonzero(as_tuple=False).squeeze(-1)
        if len(resample_env_ids) > 0:
            self._resample_command(resample_env_ids)
        self.time_steps.copy_(self.time_steps.minimum(self.motion.lengths[self.motion_ids] - 1))

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Common motion-library configuration; robots supply names and assets."""

    class_type: type = MotionCommand
    asset_name: str = "robot"
    motion_dir: str = MISSING
    robot_name: str = MISSING
    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING
    root_body_name: str = MISSING
    anchor_body_name: str = MISSING
    future_steps: int = 10
    discount: float = 0.99
    evaluation: bool = False
    evaluation_motion_id: int = 0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
