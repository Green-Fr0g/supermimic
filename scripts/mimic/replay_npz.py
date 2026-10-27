"""Replay a canonical Isaac Lab 3.0 Mimic NPZ motion."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion_file", "-f", required=True)
parser.add_argument(
    "--visual",
    choices=("newton_gl", "newton_rtx", "kit", "rerun", "viser", "none"),
    default="newton_gl",
    help="Visualizer backend. Defaults to the Newton GL window; use 'none' for headless replay.",
)
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier. 1.0 follows the NPZ frame rate; 0 disables real-time pacing.",
)
parser.add_argument(
    "--physics",
    choices=("physx", "isaacsim_physx", "newton_mjwarp", "newton_vbd", "ovphysx"),
    default="newton_mjwarp",
    help="Isaac Lab physics backend.",
)


add_launcher_args(parser)
args_cli = parser.parse_args()
# ``add_launcher_args`` exposes the Isaac Lab 3.0 ``--visualizer``/``--viz``
# options.  Replay keeps a smaller, explicit ``--visual`` interface and forwards
# the selected backend to the launcher after parsing.
args_cli.visualizer = [] if args_cli.visual == "none" else [args_cli.visual]

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from supermimic.assets.robots.g1 import G1_CFG  # noqa: E402

ROBOT_ASSET_REGISTRY: dict[str, ArticulationCfg] = {
    "g1": G1_CFG,
}


def _get_robot_asset(robot_name: str) -> ArticulationCfg:
    """Return the USD-backed articulation configuration for a robot name."""
    try:
        return ROBOT_ASSET_REGISTRY[robot_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOT_ASSET_REGISTRY))
        raise KeyError(f"Unknown robot_name '{robot_name}'. Supported: {supported}") from exc


class MotionLoader:
    """Load the canonical Mimic NPZ fields needed for replay."""

    def __init__(self, motion_file: str, device: torch.device | str = "cpu") -> None:
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
        with np.load(motion_file, allow_pickle=True) as data:
            self.robot_name = str(np.asarray(data["robot_name"]).reshape(-1)[0])
            self.root_link_name = str(np.asarray(data["root_link_name"]).reshape(-1)[0])
            self._dof_names = [str(name) for name in np.asarray(data["dof_names"]).tolist()]
            self._body_names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
            self.device = torch.device(device)
            self.dof_positions = self._tensor(data["dof_positions"])
            self.dof_velocities = self._tensor(data["dof_velocities"])
            self.body_positions = self._tensor(data["body_positions"])
            self.body_rotations = self._tensor(data["body_rotations"])
            self.body_linear_velocities = self._tensor(data["body_linear_velocities"])
            self.body_angular_velocities = self._tensor(data["body_angular_velocities"])
            self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])

        if self.fps <= 0.0:
            raise ValueError(f"Motion file '{motion_file}' has a non-positive fps: {self.fps}")
        if self.dof_positions.ndim != 2 or self.dof_positions.shape[1] != len(self._dof_names):
            raise ValueError(
                f"Motion file '{motion_file}' has {len(self._dof_names)} dof_names but "
                f"dof_positions has shape {tuple(self.dof_positions.shape)}."
            )
        if self.body_positions.ndim != 3 or self.body_positions.shape[1] != len(self._body_names):
            raise ValueError(
                f"Motion file '{motion_file}' has {len(self._body_names)} body_names but "
                f"body_positions has shape {tuple(self.body_positions.shape)}."
            )
        self.dt = 1.0 / self.fps
        self.num_frames = self.dof_positions.shape[0]

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    def get_body_index(self, names: list[str]) -> list[int]:
        return [self._body_names.index(name) for name in names]


def _value(data):
    return data.torch if hasattr(data, "torch") else data


def _scene_cfg(robot_cfg: ArticulationCfg):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return SceneCfg


def main() -> None:
    if args_cli.playback_speed < 0.0:
        raise ValueError("--playback_speed must be non-negative")
    motion = MotionLoader(args_cli.motion_file, args_cli.device)
    robot_cfg = _get_robot_asset(motion.robot_name)
    if motion.root_link_name not in motion.body_names:
        raise ValueError(
            f"root_link_name '{motion.root_link_name}' is not present in the motion body_names: {motion.body_names}"
        )
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.physics = PhysicsCfg()
    sim_cfg.dt = motion.dt
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(_scene_cfg(robot_cfg)(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot: Articulation = scene["robot"]
        if motion.root_link_name not in robot.body_names:
            raise ValueError(
                f"root_link_name '{motion.root_link_name}' is not present in the robot bodies: {robot.body_names}"
            )
        root_index = motion.get_body_index([motion.root_link_name])[0]
        joint_ids, matched_joint_names = robot.find_joints(motion.dof_names, preserve_order=True)
        if matched_joint_names != motion.dof_names:
            raise ValueError(
                f"Motion DOF names do not match the robot joints. Requested: {motion.dof_names}; "
                f"matched: {matched_joint_names}"
            )
        start_time = time.perf_counter()
        for index in range(motion.num_frames):
            if args_cli.playback_speed > 0.0:
                target_time = start_time + index * motion.dt / args_cli.playback_speed
                time.sleep(max(0.0, target_time - time.perf_counter()))
            root_pose = _value(robot.data.default_root_pose).clone()
            root_pose[:, :3] = motion.body_positions[index, root_index]
            root_pose[:, 3:7] = motion.body_rotations[index, root_index]
            root_vel = _value(robot.data.default_root_vel).clone()
            root_vel[:, :3] = motion.body_linear_velocities[index, root_index]
            root_vel[:, 3:] = motion.body_angular_velocities[index, root_index]
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            joint_pos = _value(robot.data.default_joint_pos).clone()
            joint_vel = _value(robot.data.default_joint_vel).clone()
            joint_pos[:, joint_ids] = motion.dof_positions[index]
            joint_vel[:, joint_ids] = motion.dof_velocities[index]
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim.get_physics_dt())
        print(f"[INFO] Replayed {motion.num_frames} frames from {args_cli.motion_file}")


if __name__ == "__main__":
    main()
