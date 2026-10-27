"""Episode randomization terms for the SuperMimic tracking task.

The terms follow BeyondMimic's event style: each reset operation is a plain
manager function. The sampled values are also written to ``motion.context`` so
the policy and world-model inputs see the exact parameters used by simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_physics_material(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float] = (0.3, 1.2),
    restitution: float = 0.0,
    enabled: bool = True,
) -> None:
    """Sample one friction coefficient per episode through the active backend view."""
    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = env_ids
    cmd = env.command_manager.get_term("motion")
    count = len(ids)
    if enabled:
        friction = torch.empty(count, device=env.device).uniform_(*friction_range)
    else:
        friction = torch.ones(count, device=env.device)
    cmd.context[ids, 0] = friction

    robot = env.scene[asset_cfg.name]
    root_view = robot.root_view
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_ov.physics import OvPhysxCfg

    if isinstance(env.cfg.sim.physics, NewtonCfg):
        from newton import ModelFlags

        manager = env.sim.physics_manager
        model = manager.get_model()
        materials = wp.to_torch(root_view.get_attribute("shape_material_mu", model)[:, 0])
        bounce = wp.to_torch(root_view.get_attribute("shape_material_restitution", model)[:, 0])
        materials[ids] = friction[:, None]
        bounce[ids] = restitution
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        return

    if isinstance(env.cfg.sim.physics, OvPhysxCfg):
        from isaaclab_ov.tensor_types import SHAPE_FRICTION_AND_RESTITUTION

        materials = wp.to_torch(root_view.get_attribute(SHAPE_FRICTION_AND_RESTITUTION))
        material_ids = ids.to(materials.device)
        materials[material_ids, :, :2] = friction.to(materials.device)[:, None, None]
        materials[material_ids, :, 2] = restitution
        root_view.set_attribute(
            SHAPE_FRICTION_AND_RESTITUTION,
            wp.from_torch(materials.contiguous(), dtype=wp.float32),
            indices=wp.from_torch(material_ids.to(torch.int32).contiguous(), dtype=wp.int32),
        )
        return

    materials = wp.to_torch(root_view.get_material_properties())
    material_ids = ids.to(device=materials.device, dtype=torch.long)
    selected = materials[material_ids].clone()
    selected[..., 0] = friction.to(materials.device)[:, None]
    selected[..., 1] = friction.to(materials.device)[:, None]
    selected[..., 2] = restitution
    materials[material_ids] = selected
    backend_env_ids = ids.to(device=materials.device, dtype=torch.int32).contiguous()
    root_view.set_material_properties(
        wp.from_torch(materials.contiguous(), dtype=wp.float32),
        wp.from_torch(backend_env_ids, dtype=wp.int32),
    )


def randomize_joint_calibration(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    joint_range: tuple[float, float] = (-0.01, 0.01),
    enabled: bool = True,
) -> None:
    """Sample episode-local joint calibration offsets into the motion context."""
    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = env_ids
    cmd = env.command_manager.get_term("motion")
    count = len(ids)
    joints = len(cmd.cfg.joint_names)
    offsets = torch.zeros(count, joints, device=env.device)
    if enabled:
        offsets.uniform_(*joint_range)
    cmd.context[ids, 1 : joints + 1] = offsets


def randomize_torso_com(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    com_range: tuple[float, float, float] = (0.025, 0.05, 0.05),
    enabled: bool = True,
) -> None:
    """Sample a torso CoM offset through the active backend articulation API."""
    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = env_ids
    cmd = env.command_manager.get_term("motion")
    count = len(ids)
    offsets = torch.zeros(count, 3, device=env.device)
    if enabled:
        offsets = (2.0 * torch.rand(count, 3, device=env.device) - 1.0) * offsets.new_tensor(com_range)

    # Remove the previous episode offset before applying the new one. This keeps
    # repeated partial resets anchored to the nominal CoM instead of accumulating.
    previous = cmd.context[ids, -3:].clone()
    cmd.context[ids, -3:] = offsets
    robot = env.scene[asset_cfg.name]
    coms = robot.data.body_com_pose_b.torch[ids][:, asset_cfg.body_ids].clone()
    coms[..., :3] -= previous[:, None, :]
    coms[..., :3] += offsets[:, None, :]
    robot.set_coms_index(coms=coms, body_ids=asset_cfg.body_ids, env_ids=ids)
