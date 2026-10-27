"""Differentiable task heads using the robot's USD joints and the existing NPZ library.

Timing follows BeyondMimic: reward compares the post-action state with the current
reference and the anchor alignment computed from the pre-action state.
"""

from __future__ import annotations

import torch
from pxr import Usd, UsdPhysics

from .math import heading_quat, quat_apply, quat_inverse, quat_mul_xyzw, quat_rotvec, rotvec_quat


class TrackingModel:
    """Extract backend-independent kinematics and reproduce the manager task's heads."""

    def __init__(self, env) -> None:
        cmd = env.command_manager.get_term("motion")
        robot = cmd.robot
        self.device = env.device
        self.motion = cmd.motion
        self.joints = len(cmd.joint_ids)
        self.root = cmd.root_index
        self.anchor = cmd.anchor_index
        self.future_steps = cmd.cfg.future_steps
        self.dt = env.step_dt
        self.default_q = robot.data.default_joint_pos.torch[0, cmd.joint_ids].clone()
        self.limits = robot.data.soft_joint_pos_limits.torch[0, cmd.joint_ids].clone()
        self.reward_cfg = env.cfg.rewards
        self.height_threshold = env.cfg.terminations.root_height.params["threshold"]
        self.tilt_threshold = env.cfg.terminations.root_tilt.params["threshold"]
        self.history_length = env.cfg.observations.policy.history_length
        self.body_names = list(cmd.cfg.body_names)
        stage = Usd.Stage.Open(robot.cfg.spawn.usd_path)
        self.asset_layers = [layer.realPath for layer in stage.GetUsedLayers() if layer.realPath]
        pending = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parents or not children:
                continue
            name = prim.GetName()
            if prim.IsA(UsdPhysics.RevoluteJoint):
                index = cmd.cfg.joint_names.index(name)
                axis = "XYZ".index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
            elif prim.IsA(UsdPhysics.FixedJoint):
                index, axis = -1, 0
            else:
                raise ValueError(f"Unsupported joint for differentiable FK: {name}")
            poses = []
            for pos, quat in (
                (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            ):
                poses.append(
                    (
                        torch.tensor(tuple(pos), device=self.device),
                        torch.tensor((*quat.GetImaginary(), quat.GetReal()), device=self.device),
                    )
                )
            pending.append((str(parents[0]).split("/")[-1], str(children[0]).split("/")[-1], index, axis, poses))
        self.names = [cmd.cfg.root_body_name]
        self.chain = []
        while pending:
            ready = [joint for joint in pending if joint[0] in self.names]
            if not ready:
                raise ValueError("USD joint graph is disconnected from the configured root")
            for parent, child, index, axis, poses in ready:
                direction = torch.eye(3, device=self.device)[axis]
                self.chain.append((self.names.index(parent), index, direction, *poses[0], *poses[1]))
                self.names.append(child)
                pending.remove((parent, child, index, axis, poses))
        self.tracked = [self.names.index(name) for name in self.body_names]
        self.all_coms = robot.data.body_com_pos_b.torch[0, [robot.body_names.index(n) for n in self.names]].clone()
        self.torso = self.names.index(env.cfg.events.torso_com.params["asset_cfg"].body_names[0])
        self.all_coms[self.torso] -= cmd.context[0, -3:]
        self.noise = torch.cat(
            (
                torch.full((self.joints,), 0.01),
                torch.full((self.joints,), 0.5),
                torch.full((3,), 0.2),
                torch.full((3,), 0.05),
                torch.zeros(self.joints),
            )
        ).to(self.device)

    def fk(self, state: torch.Tensor, context: torch.Tensor):
        """Return tracked link poses and CoM velocities; state root velocity is also at its CoM."""
        p, q, v, w = state[..., :3], state[..., 3:7], state[..., 7:10], state[..., 10:13]
        joints, rates = state[..., 13 : 13 + self.joints], state[..., 13 + self.joints :]
        root_velocity = v - torch.cross(w, quat_apply(q, self.all_coms[0]), dim=-1)
        positions, rotations, velocities, angular = [p], [q], [root_velocity], [w]
        for parent, index, axis, p0, q0, p1, q1 in self.chain:
            rp = quat_apply(rotations[parent], p0)
            anchor_q = quat_mul_xyzw(rotations[parent], q0)
            joint_rotation = (
                rotvec_quat(axis * joints[..., index : index + 1]) if index >= 0 else q0.new_tensor([0, 0, 0, 1])
            )
            qi = quat_mul_xyzw(quat_mul_xyzw(anchor_q, joint_rotation), quat_inverse(q1))
            wi = angular[parent]
            if index >= 0:
                wi = wi + quat_apply(anchor_q, axis) * rates[..., index : index + 1]
            child_offset = quat_apply(qi, p1)
            positions.append(positions[parent] + rp - child_offset)
            rotations.append(qi)
            angular.append(wi)
            velocities.append(
                velocities[parent] + torch.cross(angular[parent], rp, dim=-1) - torch.cross(wi, child_offset, dim=-1)
            )
        com_velocity = []
        for index in range(len(self.names)):
            offset = self.all_coms[index]
            if index == self.torso:
                offset = offset + context[..., -3:]
            com_velocity.append(
                velocities[index] + torch.cross(angular[index], quat_apply(rotations[index], offset), dim=-1)
            )
        return tuple(
            torch.stack([values[i] for i in self.tracked], -2)
            for values in (positions, rotations, com_velocity, angular)
        )

    def proprioception(self, state, previous_action, context, noisy=False):
        q = state[..., 3:7]
        gravity = state.new_tensor([0, 0, -1])
        obs = torch.cat(
            (
                state[..., 13 : 13 + self.joints] - self.default_q - context[..., 1 : 1 + self.joints],
                state[..., 13 + self.joints :],
                quat_apply(quat_inverse(q), state[..., 10:13]),
                quat_apply(quat_inverse(q), gravity),
                previous_action,
            ),
            -1,
        )
        return obs + (2 * torch.rand_like(obs) - 1) * self.noise if noisy else obs

    def goal(self, motion_ids, frames):
        offsets = torch.arange(1, self.future_steps + 1, device=self.device)
        future = frames[..., None] + offsets
        ids = self.motion.indices(motion_ids[..., None], future)
        now = self.motion.indices(motion_ids, frames)
        inverse = quat_inverse(heading_quat(self.motion.body_rotations[now, self.root]))[..., None, :]
        pos = self.motion.body_positions[ids, self.root] - self.motion.body_positions[now, self.root][..., None, :]
        goal = torch.cat(
            (
                self.motion.dof_positions[ids],
                self.motion.dof_velocities[ids],
                quat_apply(inverse, pos),
                quat_mul_xyzw(inverse, self.motion.body_rotations[ids, self.root]),
                quat_apply(inverse, self.motion.body_linear_velocities[ids, self.root]),
                quat_apply(inverse, self.motion.body_angular_velocities[ids, self.root]),
            ),
            -1,
        )
        mask = future < self.motion.lengths[motion_ids, None]
        return goal.flatten(-2), mask

    def aligned_targets(self, previous_state, context, motion_ids, frames):
        """Reference targets aligned using the pre-action robot anchor, as in commands.py."""
        old_p, old_q, _, _ = self.fk(previous_state, context)
        ids = self.motion.indices(motion_ids, frames)
        ref_p, ref_q = self.motion.body_positions[ids], self.motion.body_rotations[ids]
        delta = heading_quat(quat_mul_xyzw(old_q[..., self.anchor, :], quat_inverse(ref_q[..., self.anchor, :])))
        origin = torch.cat((old_p[..., self.anchor, :2], ref_p[..., self.anchor, 2:3]), -1)
        target_p = origin[..., None, :] + quat_apply(
            delta[..., None, :], ref_p - ref_p[..., self.anchor : self.anchor + 1, :]
        )
        target_q = quat_mul_xyzw(delta[..., None, :], ref_q)
        return target_p, target_q

    def reward(self, state, action, previous_action, context, motion_ids, frames, previous_state):
        """Analytic post-action reward with the same cached-anchor semantics as the real task."""
        p, q, v, w = self.fk(state, context)
        ids = self.motion.indices(motion_ids, frames)
        ref_p, ref_q = self.motion.body_positions[ids], self.motion.body_rotations[ids]
        target_p, target_q = self.aligned_targets(previous_state, context, motion_ids, frames)
        errors = {
            "body_position": (target_p - p).square().sum(-1).mean(-1),
            "body_orientation": quat_rotvec(quat_mul_xyzw(quat_inverse(target_q), q)).square().sum(-1).mean(-1),
            "body_linear_velocity": (self.motion.body_linear_velocities[ids] - v).square().sum(-1).mean(-1),
            "body_angular_velocity": (self.motion.body_angular_velocities[ids] - w).square().sum(-1).mean(-1),
            "root_position": (ref_p[..., self.root, :] - p[..., self.root, :]).square().sum(-1),
            "root_orientation": quat_rotvec(quat_mul_xyzw(quat_inverse(ref_q[..., self.root, :]), q[..., self.root, :]))
            .square()
            .sum(-1),
        }
        result = torch.zeros_like(frames, dtype=state.dtype)
        for name, error in errors.items():
            cfg = getattr(self.reward_cfg, name)
            result = result + cfg.weight * torch.exp(-error / cfg.params["std"] ** 2)
        result = result + self.reward_cfg.action_rate.weight * (action - previous_action).square().sum(-1)
        joint_pos = state[..., 13 : 13 + self.joints]
        violation = torch.relu(self.limits[:, 0] - joint_pos) + torch.relu(joint_pos - self.limits[:, 1])
        return (result + self.reward_cfg.joint_limit.weight * violation.sum(-1)) * self.dt

    def termination(self, state, motion_ids, frames, soft=True):
        """Reference-dependent failure; clip/time truncation is handled separately."""
        ids = self.motion.indices(motion_ids, frames)
        height = (state[..., 2] - self.motion.body_positions[ids, self.root, 2]).abs()
        reference = self.motion.body_rotations[ids, self.root]
        actual = state[..., 3:7]
        tilt = (2 * (reference[..., :2].square().sum(-1) - actual[..., :2].square().sum(-1))).abs()
        if not soft:
            return (height > self.height_threshold) | (tilt > self.tilt_threshold)
        return 1 - (1 - torch.sigmoid((height - self.height_threshold) / 0.02)) * (
            1 - torch.sigmoid((tilt - self.tilt_threshold) / 0.04)
        )
