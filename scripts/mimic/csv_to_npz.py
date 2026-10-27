"""Replay a CSV motion into Isaac Sim and export it as an NPZ.

This script is config-free: every input comes from the command line. The
config-driven front end ``config_csv_to_npz.py`` reads a motion yaml and
invokes this script with the matching arguments.

Example:
    python csv_to_npz.py \
        --input_file motion_data/motion_tracking/g1/G1_Take_102.bvh_60hz.csv \
        --input_fps 120 \
        --robot_name g1 \
        --root_link_name pelvis \
        --csv_joint_names <joint names in csv column order> \
        --output_name /tmp/g1_motion.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=120, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument(
    "--physics",
    choices=("physx", "isaacsim_physx", "newton_mjwarp", "newton_vbd", "ovphysx"),
    default="physx",
    help="Physics backend selected by Isaac Lab 3.0's simulation launcher.",
)
parser.add_argument(
    "--visual",
    choices=("newton_gl", "newton_rtx", "kit", "rerun", "viser", "none"),
    default="none",
    help="Visualizer backend. Defaults to the Newton GL window; use 'none' for headless replay.",
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# Robot metadata, normally supplied by the config-driven launcher.
parser.add_argument("--robot_name", type=str, required=True, help="Robot registry name used to select the asset.")
parser.add_argument("--root_link_name", type=str, required=True, help="Root link used as the motion anchor.")
parser.add_argument(
    "--csv_joint_names",
    nargs="+",
    required=True,
    help="Joint names matching the csv column order. Falls back to the robot joint order when omitted.",
)
parser.add_argument(
    "--root_quat_order",
    type=str,
    default="xyzw",
    choices=("wxyz", "xyzw"),
    help="Quaternion order of the root rotation inside the csv.",
)
# Kept as a compatibility flag for older command lines. Isaac Lab 3.0 is headless when no
# visualizer is selected, so the flag intentionally has no additional runtime effect.
parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)

add_launcher_args(parser)
args_cli = parser.parse_args()

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from supermimic.assets.robots.g1 import G1_CFG

ROBOT_ASSET_REGISTRY: dict[str, ArticulationCfg] = {
    "g1": G1_CFG,
}


def _get_robot_asset(robot_name: str) -> ArticulationCfg:
    """Return the registered asset, joint order, and root link for a robot."""
    try:
        return ROBOT_ASSET_REGISTRY[robot_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOT_ASSET_REGISTRY))
        raise KeyError(f"Unknown robot_name '{robot_name}'. Supported: {supported}") from exc


def _value(data):
    """Return a tensor from an Isaac Lab 3.0 proxy array or a plain tensor."""
    return data.torch if hasattr(data, "torch") else data


def _root_quat_to_xyzw(quat: torch.Tensor, order: str) -> torch.Tensor:
    if order == "wxyz":
        return quat[:, [1, 2, 3, 0]]
    if order == "xyzw":
        return quat
    raise ValueError(f"Unsupported root quaternion order: {order}")


def _build_scene_cfg(robot_cfg: ArticulationCfg) -> type[InteractiveSceneCfg]:
    @configclass
    class ReplayMotionSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionSceneCfg


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        root_quat_order: str,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.root_quat_order = root_quat_order
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion_array = np.loadtxt(self.motion_file, delimiter=",")
        else:
            motion_array = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
        if np.asarray(motion_array).size == 0:
            raise ValueError(f"Motion frame range contains no data: {self.motion_file}")
        motion = torch.from_numpy(np.atleast_2d(motion_array))
        if motion.shape[1] < 7:
            raise ValueError(f"Motion CSV must contain at least 7 root-state columns: {self.motion_file}")
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = _root_quat_to_xyzw(motion[:, 3:7], self.root_quat_order)
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        if self.input_frames == 0:
            raise ValueError(f"Motion file contains no frames: {self.motion_file}")
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        if self.duration == 0.0:
            times = torch.zeros(1, device=self.device, dtype=torch.float32)
        else:
            times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_frames == 1:
            zeros = torch.zeros(times.shape[0], dtype=torch.long, device=self.device)
            return zeros, zeros, torch.zeros_like(times)
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.clamp(index_0 + 1, max=self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        if self.output_frames < 2:
            self.motion_base_lin_vels = torch.zeros_like(self.motion_base_poss)
            self.motion_dof_vels = torch.zeros_like(self.motion_dof_poss)
        else:
            self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
            self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        if rotations.shape[0] < 3:
            return torch.zeros((rotations.shape[0], 3), device=rotations.device, dtype=rotations.dtype)
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    motion: MotionLoader,
    csv_joint_names: list[str],
    output_name: str,
    root_link_name: str,
    robot_name: str,
):
    robot = scene["robot"]
    if root_link_name not in robot.body_names:
        raise ValueError(f"root_link_name '{root_link_name}' not found in robot bodies: {robot.body_names}")

    robot_joint_indexes, matched_joint_names = robot.find_joints(csv_joint_names, preserve_order=True)
    if matched_joint_names != csv_joint_names:
        raise ValueError(
            f"CSV joint names do not match the robot joints. Requested: {csv_joint_names}; "
            f"matched: {matched_joint_names}"
        )
    if motion.motion_dof_poss_input.shape[1] != len(csv_joint_names):
        raise ValueError(
            f"CSV contains {motion.motion_dof_poss_input.shape[1]} joint columns, "
            f"but {len(csv_joint_names)} joint names were provided."
        )

    log = {
        "fps": [args_cli.output_fps],
        "robot_name": [robot_name],
        "dof_names": np.array(robot.joint_names),
        "body_names": np.array(robot.body_names),
        "root_link_name": [root_link_name],
        "dof_positions": [],
        "dof_velocities": [],
        "body_positions": [],
        "body_rotations": [],
        "body_linear_velocities": [],
        "body_angular_velocities": [],
    }
    for _ in range(motion.output_frames):
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_pose = _value(robot.data.default_root_pose).clone()
        root_pose[:, :3] = motion_base_pos + scene.env_origins
        root_pose[:, 3:7] = motion_base_rot
        root_velocity = _value(robot.data.default_root_vel).clone()
        root_velocity[:, :3] = motion_base_lin_vel
        root_velocity[:, 3:] = motion_base_ang_vel
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)

        joint_pos = _value(robot.data.default_joint_pos).clone()
        joint_vel = _value(robot.data.default_joint_vel).clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_pose[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["dof_positions"].append(_value(robot.data.joint_pos)[0].cpu().numpy().copy())
        log["dof_velocities"].append(_value(robot.data.joint_vel)[0].cpu().numpy().copy())
        log["body_positions"].append(_value(robot.data.body_pos_w)[0].cpu().numpy().copy())
        log["body_rotations"].append(_value(robot.data.body_quat_w)[0].cpu().numpy().copy())
        log["body_linear_velocities"].append(_value(robot.data.body_lin_vel_w)[0].cpu().numpy().copy())
        log["body_angular_velocities"].append(_value(robot.data.body_ang_vel_w)[0].cpu().numpy().copy())

    for key in (
        "dof_positions",
        "dof_velocities",
        "body_positions",
        "body_rotations",
        "body_linear_velocities",
        "body_angular_velocities",
    ):
        log[key] = np.stack(log[key], axis=0)
    output_path = Path(output_name).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **log)
    print(f"[INFO]: Motion saved to: {output_path}")


def main():
    input_path = Path(args_cli.input_file).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input motion csv file does not exist: {input_path}")

    if args_cli.robot_name is None:
        raise ValueError("--robot_name is required for conversion")
    robot_name = str(args_cli.robot_name)
    robot_cfg = _get_robot_asset(robot_name)
    root_link_name = args_cli.root_link_name
    root_quat_order = str(args_cli.root_quat_order)
    robot_dof_names = list(args_cli.csv_joint_names)

    if args_cli.input_fps <= 0 or args_cli.output_fps <= 0:
        raise ValueError("--input_fps and --output_fps must be positive")
    if args_cli.frame_range is not None:
        start, end = args_cli.frame_range
        if start < 1 or end < start:
            raise ValueError("--frame_range must be a 1-based inclusive range with START <= END")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / args_cli.output_fps)
    # Isaac Lab 3.0 resolves this placeholder from args_cli.physics in the launcher.
    sim_cfg.physics = PhysicsCfg()
    with launch_simulation(sim_cfg, args_cli) as physics_cfg:
        sim_cfg.physics = physics_cfg
        sim = SimulationContext(sim_cfg)
        scene_cfg = _build_scene_cfg(robot_cfg)
        scene = InteractiveScene(scene_cfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        print(f"[INFO]: Setup complete with physics backend: {type(physics_cfg).__name__}")

        motion = MotionLoader(
            motion_file=str(input_path),
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
            device=sim.device,
            frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
            root_quat_order=root_quat_order,
        )
        run_simulator(sim, scene, motion, robot_dof_names, args_cli.output_name, root_link_name, robot_name)


if __name__ == "__main__":
    main()
